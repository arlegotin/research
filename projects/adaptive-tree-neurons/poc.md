# Research and proof-of-concept plan: selectively growing tree-neurons

**Research cut-off:** 16 October 2026.  
**Target execution platform:** one MacBook Pro with an M3 Max; its actual memory, macOS version, free storage, and installed tools must be discovered.  
**Status:** researched proposal with executed CPU algebra/autograd checks; no predictive benchmark or M3 Max experiment has been executed.  
**Primary deliverable of the proposed project:** reproducible evidence about the value of allocating conditional computation to selected individual neurons, including a defensible negative result.

## 1. Executive assessment and decision

### 1.1 Verdict

The basic architecture is established, not a new type of computational intelligence. A soft tree with affine experts belongs to the hierarchical mixture-of-experts tradition [R01]. More decisively, a two-leaf affine tree is **exactly an affine bypass plus a gated linear unit**, as derived in Section 3; GLUs and related gated architectures are established [R02, R03]. Growing neural trees, selecting growth using validation loss, and splitting neurons also have substantial prior art [R04–R08].

There remains a worthwhile, locally testable question:

> At a fixed capacity budget, can an inexpensive, training-data-derived score identify which ordinary neurons should acquire an input-dependent pair of affine specialists, producing better generalization than random placement, static placement, or spending the same resources on ordinary capacity?

**Select this question as the primary direction.** Start with one growth event in the first hidden layer of a small regression network. Use a function-preserving insertion, an explicitly specified residual-weighted moment score, and matched controls. Only investigate repeated growth, deeper trees, pruning, or hard execution after the first event earns those complications.

The prospective contribution is a **tested capacity-allocation procedure and empirical characterization**, not the invention of tree-neurons. The score and initialization are candidate refinements, not established methodological novelties. A closely related 2026 paper, *Arboreal Neural Network*, was located in the official ICML listing, but its full text was blocked by OpenReview browser verification [R12]. Its unresolved overlap blocks a broad novelty claim, not implementation of the diagnostic experiments.

### 1.2 What would count as useful progress?

There are three different outcomes worth distinguishing.

1. **Component result:** selective placement improves held-out prediction over a precisely matched random-placement control on independently generated problems. This can be meaningful without beating every tabular predictor.
2. **Practical result:** the resulting pipeline improves a measured prediction–training-cost–inference-cost tradeoff against strong, locally executable alternatives on the prespecified transfer suite. This requires counting architecture search and failed candidates, not just the final network.
3. **Negative or replication result:** the proposed effect is absent, explained by initialization or ordinary capacity, or already implemented elsewhere. A reliable boundary-condition map and reproducible equivalence demonstration still resolve the original question.

Do not announce a breakthrough from fitting a piecewise function, reducing the number of named “neurons,” or winning one dataset. There is no application-specific utility threshold in the brief, so this plan does not invent one.

### 1.3 Immediate ordering

First verify algebra, growth initialization, local dependency support, and the evaluation harness. Next run the selective-versus-random experiment and capacity controls. Then, contingent on a real mechanism signal or an informative unresolved transfer question, run strong tabular baselines and the fixed transfer suite. Hard routing is an optional efficiency hypothesis, not an assumed benefit.

The original intent is preserved: an individual scalar-output neuron acquires different functions for different input regions. Two explicit refinements are made: an ordinary ReLU is retained **after** the conditional affine preactivation so growth starts from an ordinary nonlinear network; and only one growth event is initially studied so the placement mechanism is identifiable.

## 2. Problem, scope, assumptions, and hypotheses

### 2.1 Task and intended beneficiary

The initial task is supervised scalar regression, with inputs `x ∈ R^d`, labels `y ∈ R`, and squared-error training loss. Intended beneficiaries are researchers and developers of compact predictors for heterogeneous, potentially regime-dependent relationships. No actual production bottleneck or customer requirement has been established. Small tabular data provide a useful transfer test, not evidence that tabular prediction is the only relevant use of the primitive.

The conventional workflow chooses a fixed-width MLP, tunes its optimizer and regularization, and trains all units from the start. The proposed change lets selected units acquire conditional affine behavior after a useful base representation has formed. The possible bottleneck is **uniform allocation of representational capacity**, not an inability of ordinary neural networks to approximate the target function.

Use the following terminology consistently:

- A **stump** has one binary gate and two leaves; depth counts routing decisions on a root-to-leaf path.
- A **tree-neuron** produces one scalar preactivation or activation. State which one.
- **Soft routing** evaluates a weighted mixture; it does not imply conditional execution.
- **Growth** increases the number of active trainable parameters during training.
- **Allocation** specifies which units grow, holding the amount of added capacity fixed.
- **Budget** means a stated resource: parameters, updates, elapsed local time, or memory. These are not interchangeable.
- **F, G, V, T** denote fitting data, growth/checkpoint-selection data, hyperparameter-selection data, and protected test data.

### 2.2 Established, assumed, proposed, and speculative

| Status | Statement | Consequence |
|---|---|---|
| Established / derived | Affine-leaf soft stumps are equivalent to affine-plus-gated-linear expressions. | Include an exact equivalent control, not merely a conventional MLP. |
| Established | Neural trees, local affine models, adaptive growth, and neuron splitting predate this proposal [R01, R04–R10]. | Claims must identify a narrower difference. |
| Assumption | Some tasks benefit from conditional relationships concentrated in only part of a learned representation. | Test heterogeneous and homogeneous problems, rather than selecting favorable examples after results. |
| Assumption | Local training-loss information predicts where added conditional capacity generalizes. | This is the central uncertainty; a large score is not evidence of out-of-sample benefit. |
| Proposed | Insert a centered gate–contrast product without changing initial predictions; rank units using the moment construction in Section 4. | Isolate insertion, direction choice, and placement in separate comparisons. |
| Speculative | Repeated growth, deep trees, sparse routing, interpretability, or larger-scale networks add further value. | None is needed for the first claim; each requires new tests. |

### 2.3 Hypotheses and falsifiers

| ID | Hypothesis and prediction | Support or motivation | Strong alternative explanation | Evidence weakening it |
|---|---|---|---|---|
| H0-E | The two formulations of an affine stump are computationally equivalent under mapped parameters. | Algebra in Section 3; GLU relationship [R02]. | Any observed advantage is optimizer parameterization or implementation, not representational novelty. | A discrepancy beyond numerical tolerance indicates a bug or unequal parameterization; it does not establish novelty. |
| H1-A, primary | Training-derived placement beats random placement with the same base checkpoint, growth size, contrast construction, and training allowance on the specified heterogeneous synthetic population. | Gradient-derived local opportunity; analogous motivation in neuron-splitting research [R07]. | Scores identify large gradients or optimization transients without useful specialization. | The paired effect is nonpositive with enough precision, or random placement matches it after proper tuning. |
| H2-C | Conditional growth improves a capacity frontier beyond ordinary widening and static gated networks. | A gate can express a change of slope directly. | More parameters, favorable initialization, longer training, or a generic multiplication explains the gain. | Ordinary capacity or static GLU controls recover the entire gain. |
| H3-T | The procedure yields a useful local tradeoff on the fixed transfer suite against a validation-selected strong baseline. | H1/H2 plus diverse non-synthetic tasks, not historical benchmark claims. | Dataset choice, weak baseline recipes, preprocessing, or selection leakage explains the result. | Tuning strong alternatives erases the benefit; improvement is confined to one favorable slice. |
| H4-H, extension | Hard execution preserves acceptable prediction quality and improves measured latency. | Exact soft–hard discrepancy formula in Section 3. | Only FLOPs decline; routing/dispatch costs erase actual savings. | Measured latency does not improve or prediction degradation exceeds the declared deployment tolerance. |

H1 is the confirmatory mechanism question. H2 is a required attribution assessment, with additional claims handled as specified in Section 7. H3 is a separate finite-suite practical assessment. H4 must not be inferred from H1–H3.

### 2.4 Strongest case for and against

**For:** a small gate can switch the sign or magnitude of an affine response; a network need not devote equal complexity to every unit; a function-preserving insertion avoids a prediction discontinuity at growth; the core calculations use modest dense linear algebra. The original example, `f(x1,x2)=x2` for `x1<0` and `1-x2` otherwise, is expressible by one hard affine-leaf stump.

**Against:** the architecture is already a gated family; its effective function may be more redundant than its neuron count suggests; selection can fit noisy gradients; extra experts increase parameters and soft evaluation cost; gate saturation suppresses learning; learned hidden units are not identifiable semantic objects; and a well-tuned modern MLP, boosted tree, or pretrained tabular model may already use capacity more effectively. A score that is useful for immediate training descent need not predict final generalization.

For the discontinuous example, a finite soft gate only approximates the jump. A ReLU network is continuous too. Comparison requires a distribution and error tolerance; counting units for exact representation of a discontinuity would be misleading. On low-noise smooth targets, conditional boundaries can be harmful rather than helpful.

## 3. Mathematical analysis and checks already performed

### 3.1 Exact relation to GLUs

Let augmented input `x̃=(x,1)`, affine leaves `L(x)=lᵀx̃`, `R(x)=rᵀx̃`, and `g(x)=sigmoid(aᵀx̃/τ)` for `τ>0`. Then

\[
T(x)=(1-g)L+gR=L+g(R-L).
\]

Thus `base=L`, `delta=R-L` gives an affine bypass plus a multiplicatively gated affine response. A “tree” implementation and a “gated unit” implementation with this mapping must predict the same values. The GLU literature supplies the gate-times-linear primitive [R02]; the exact affine-leaf identity here is a direct algebraic deduction.

Training independent `(L,R)` and training `(base,delta)` with ordinary AdamW are **not** necessarily equivalent optimization procedures: coordinate changes alter optimizer states and regularization. The equivalence harness must evaluate both formulas using the same underlying tensors or map the entire optimizer and penalty. A performance difference between independently optimized coordinates is an optimization result, not new expressive capacity.

For a one-hidden-layer network **without an intervening nonlinear activation** and with scalar linear head,

\[
\hat y=c+\sum_j\alpha_j T_j(x)
=c+\left(\sum_j\alpha_j l_j\right)^T\tilde x
+\sum_j g_j(x)\left[\alpha_j(r_j-l_j)\right]^T\tilde x.
\]

All base affine terms collapse into one global affine term. This exposes redundancy and explains why “fewer neurons” is not a sufficient compression claim. This collapse does not apply unchanged after the ReLU in the selected mechanism.

**Permutation symmetry matters.** In a single dense hidden layer, choosing a different subset of unit IDs but retaining the same counts of ordinary and gated units does not create a different architecture up to permutation. A benefit from score-based IDs must come from the inherited learned functions and training trajectory, not a uniquely discovered static topology. The static comparator can be built directly from those counts without paying for a scoring pass. A learned distribution of growth across layers or different depths would be a richer topology question, deferred here.

### 3.2 Selected unit and exact insertion

The ordinary preactivation is `b_j(x)=w_jᵀx̃`. For a selected unit, replace it by

\[
z_j(x)=b_j(x)+s_j(x)d_j(x),\qquad
s_j(x)=2\sigma(a_j^T\tilde x/\tau)-1
=\tanh(a_j^T\tilde x/(2\tau)),
\]
\[
d_j(x)=v_j^T\tilde x,\qquad h_j(x)=\operatorname{ReLU}(z_j(x)).
\]

Its two affine leaves are `b_j-d_j` and `b_j+d_j`. An unselected unit remains `ReLU(b_j)`. Initialize `a_j=0` and a nonzero `v_j`; then `s_j=0` and the complete network's predictions are unchanged, apart from implementation roundoff. The contrast is initially hidden, but the gate can receive a nonzero gradient.

If both `a_j=0` and `v_j=0`, both new factors have zero first-order gradient. This is a failed insertion, not evidence against specialization. Duplicating leaves with a nonconstant gate is different: unequal mixing weights can break expert symmetry. Do not generalize the zero–zero failure to every duplicated-expert initialization.

**Activation convention:** `ReLU((1-g)L+gR)` is not equal to `(1-g)ReLU(L)+gReLU(R)`. The first is the primary design. The second and identity activation are explicit controls, never silent substitutions.

### 3.3 Parameter and execution costs

For a layer with input dimension `d`, width `h`, and `m` grown stumps:

\[
P_{\rm ordinary}=h(d+1),\qquad
P_{\rm partial}=(h+2m)(d+1).
\]

A complete depth-`D` affine-leaf binary tree has `2^D-1` affine gates and `2^D` affine leaves, hence `(2^{D+1}-1)(d+1)` parameters per scalar output. With constant leaves the count is different: `(2^D-1)(d+1)+2^D`. Do not conflate these versions.

An ordinary unit evaluates one affine projection; a soft stump evaluates three. A hard affine stump still needs a gate and its selected leaf: approximately two projections, plus routing overhead. Sparse or axis-aligned gates change these counts and must be named as different variants. Multiplying a fully computed branch by a mask does not avoid its computation.

Example, derived rather than measured: `d=32`, two hidden layers of width 64, and a scalar head gives 6,337 ordinary parameters. Growing 16 first-layer units adds 1,056 parameters, giving 7,393; growing all 64 gives 10,561. The final head and all subsequent layers are included. A parameter is not a uniform unit of complexity across trees, pretrained models, and MLPs; report model bytes, effective operations, and actual timing too.

For total inference time `t=t_other+t_target`, even removing all target-layer cost yields speedup at most `t/(t_other)`. A local operation reduction has little end-to-end headroom when that layer is a small runtime fraction. Measure this fraction before developing custom kernels.

### 3.4 Hardening discrepancy

For `g>=1/2`, select `R`; otherwise select `L`. Pointwise,

\[
|T_{soft}-T_{hard}|=\min(g,1-g)|R-L|.
\]

ReLU is nonexpansive, so applying it to both sides cannot increase this difference. A downstream Lipschitz bound can propagate it, but an estimated bound is not a guarantee of acceptable task loss. Low gate entropy alone is inadequate when leaf disagreement is large. Measure both factors, output discrepancy, and final risk. Gate thresholds must be fixed or selected on G/V, never T.

### 3.5 Executed validation, and its limits

The research environment ran the self-contained script reproduced in Appendix A. It used Python 3.13.5, PyTorch 2.10.0+cpu, Linux x86-64, one CPU thread, float64, seed 1602026, and tensors with `N=37, d=7, h=5`.

| Check | Observed result |
|---|---:|
| Tree versus affine-plus-gate forward maximum absolute error | `8.882e-16` |
| Gradient equivalence maximum absolute error | `1.776e-15` |
| Scalar-head affine collapse maximum absolute error | `2.665e-15` |
| Prediction change immediately after centered insertion | `0.0` |
| Analytic versus autograd gate gradient maximum absolute error | `5.551e-17` |
| Initial contrast-gradient norm | `0.0` |
| Initial gate-gradient norm | `0.4451935795` |
| Illustrative training loss before / after one tiny gate step | `18.7354818901 / 18.7354681759` |
| Soft–hard discrepancy identity maximum absolute error | `8.882e-16` |
| ReLU discrepancy-bound violation | `6.661e-16`, rounding scale |

These checks establish consistency of the tested algebra and gradients. They do not establish selection quality, convergence, generalization, superiority, M3 compatibility, or typical runtime. The RMS-normalized score received the additional single-batch checks in Appendix B: gate-gradient agreement within `5.551e-17`, positive-ReLU-rescaling score agreement within `1.388e-17`, and permutation agreement within `2.776e-17`. A gate-only first Adam step had predicted/observed loss decrease `8.94131658e-7 / 8.94131585e-7`. The complete allocation procedure below has **not** been benchmarked. No public dataset was downloaded or fitted during this research. The browser inspected documentation and one NODE source module; repositories were not cloned and executed.

## 4. Implementable primary mechanism

### 4.1 Minimal architecture and numerical conventions

The smallest experimental network is `x → partial-specialist layer → ReLU → scalar affine head`. Begin with `d=16, h=64` for synthetic calibration. Its ordinary parameter count is `P0=h(d+2)+1=1,153`; growing `m=16` adds 544. These are starting experimental coordinates, not permanent scale limits. Width, sample density, and depth expansion follow the tests below.

The causal experiment has no dropout, BatchNorm, trainable input normalization, ensembling, or hidden data-dependent preprocessing. Standardize each numerical feature using F only; use population standard deviation, replace a zero scale by 1, and keep a constant-column manifest. Standardize `y` using F only, invert predictions for physical-unit metrics. Set `τ=1` initially and hold it fixed in the main comparison. Learned temperature and numerical embeddings are separate ablations.

Use float32 for training, float64 on CPU for score construction and mathematical tests. Use eager PyTorch initially. No reduced precision, straight-through estimator, quantization, or implicit CPU fallback in the primary neural path. The contrast uses `tanh(a/(2τ))` for the centered gate. Store the bias as an explicit last coordinate in score calculations; normal `nn.Linear` biases are acceptable in training.

### 4.2 Score construction

At the chosen base checkpoint, evaluate F in deterministic evaluation mode. Let `e_ij=∂ell_i/∂z_ij` for **per-example loss**, at the candidate layer's preactivation. For squared error this includes the complete downstream derivative and the ReLU derivative. PyTorch's zero derivative at exactly zero is the convention. Avoid extracting gradients of a batch mean without correcting the scaling.

For each eligible unit, compute

\[
M_j=\frac{1}{n_F}\sum_{i\in F}e_{ij}\tilde x_i\tilde x_i^T.
\]

This is a symmetric, residual-weighted second-moment matrix, not an ordinary covariance matrix and not a positive-semidefinite Hessian. With `a_j=0`,

\[
u_j:=\nabla_{a_j}\mathcal R=\frac{M_jv_j}{2\tau}.
\]

The formula follows by differentiating the centered gate. It was checked numerically in Section 3.5.

Choose an eigenvector `q_j` of `M_j` corresponding to the largest absolute eigenvalue. Use a deterministic sign convention: its largest-magnitude coordinate is positive. Record eigenvalues and spectral gaps; near ties make the direction unstable, not invalid. Set

\[
v_j=\epsilon\,\operatorname{RMS}_F(b_j)\,
\frac{q_j}{\max(\operatorname{RMS}_F(\tilde x^Tq_j),10^{-12})}.
\]

Start calibration at `ε=0.05`, with `0.01` and `0.2` as sensitivity coordinates. A unit with zero preactivation RMS, zero score, or a nonfinite statistic is not silently selected; record its status. Positive rescaling of a ReLU unit with reciprocal rescaling of its outgoing weights should leave the proposed gate-gradient opportunity essentially unchanged. This is an explicit test, not a claimed invariance of every optimizer trajectory.

The eigenvector maximizes a Euclidean-gradient criterion under a Euclidean contrast constraint. The subsequent RMS normalization and Adam preconditioning change that optimization problem. Therefore the direction is a computationally convenient heuristic, not a proof of an optimal split.

**Primary optimizer-aware ranking:** new gate parameters start with zero Adam moments. With Adam epsilon `ε_A` and a common gate learning rate, rank by the predicted first gate-step decrease per added parameter,

\[
S_j=\frac{1}{2(d+1)}\sum_k
\frac{u_{jk}^2}{|u_{jk}|+\epsilon_A}.
\]

The omitted common learning-rate factor does not change the order. This is the first-step Adam expression after bias correction, ignoring higher-order changes and common updates to the old model. For an SGD gate-step diagnostic use `||u_j||²/[2(d+1)]` instead. Do not claim that the SGD score precisely forecasts Adam's step. Both are train-loss proxies; the primary question is whether their ranking predicts held-out improvement.

Set `m_eff=min(m, number_of_eligible_units)` before forking, identically for top-ranked and random placement. Include zero-growth cases in the results; never discard an easy/fully fitted teacher to create headroom. A zero-score tie does not justify arbitrary extra branches. Record both requested and actual growth counts.

Dense scoring costs roughly `O(n_F h(d+1)^2)` work and `O(h(d+1)^2)` storage if all moments are retained. Accumulate one unit or a small block at a time to bound memory. Run only the first-layer version initially. A matrix-free or randomized eigensolver would be a new implementation variant requiring score/ranking parity checks before use.

### 4.3 One-event training procedure

```text
INPUT: F, G; fixed architecture/training recipe; growth size m;
       predeclared base stopping rule; paired seeds and post-growth allowance.

1. Fit the ordinary base on F; choose its checkpoint using G only.
   Save weights, optimizer, schedule, RNGs, sampler position, and preprocessing.
2. Using that checkpoint, accumulate M_j on F and construct v_j and S_j.
3. Create separate forks, loaded and trained one at a time:
   A: top-m eligible units by S_j;
   B: m uniformly sampled eligible units; same per-unit v_j construction;
   C: no growth, continued base training;
   D: selected units but random normalized contrast directions (ablation).
4. Add A=0 and stored V to grown units. Verify prediction preservation.
   Preserve all old optimizer states; new parameters receive zero moments.
5. Train each fork on F using the paired post-growth minibatch stream.
   Select its checkpoint on G with the same rule and allowance.
6. For the causal comparison, report A and B even when either loses to C.
   For the practical algorithm, G may choose keep-versus-grow; record both.
7. V selects hyperparameters or the practical candidate, not individual
   reported test outcomes. T remains inaccessible until the protocol is frozen.
```

One event deliberately does not use repeated significance tests on G. Repeatedly selecting a minimum validation loss is model selection, not proof that every accepted branch generalizes [R32]. The separate outer test protects evaluation of the entire selection procedure.

### 4.4 Tensor implementation contract

`SelectiveAffineLayer(d,h)` owns `base: Linear(d,h)`, a sorted `grown_ids: int64[m]` buffer, and compact `gate: Linear(d,m)` / `contrast: Linear(d,m)` modules when `m>0`. Do not allocate a full dormant tree for every unit and then omit dormant tensors from memory accounting.

```python
# Reference forward logic; package/module construction is a project task.
def forward(self, x):
    z = self.base(x)                            # [B, h]
    if self.grown_ids.numel() == 0:
        return z
    s = torch.tanh(self.gate(x) / (2.0 * self.tau))
    delta = s * self.contrast(x)                # [B, m]
    selected = z.index_select(1, self.grown_ids) + delta
    return z.index_copy(1, self.grown_ids, selected)
```

The caller applies ReLU. Test `index_copy` forward and backward on the actual selected backend. A dense selection-matrix implementation is a faithful algebraic fallback for unsupported indexing but adds arithmetic and must be timed as such. Full CPU execution of the same module is the preferred fallback over an undocumented mixed CPU/GPU path.

`grow(ids, contrasts)` must reject duplicate/out-of-range IDs, wrong dtype/shape, nonfinite values, and unsupported second growth in the core version. It returns a parameter-name migration manifest for the optimizer. Exported model configuration must fully reconstruct topology before loading the state dictionary.

### 4.5 Later recursive version, only after an evidence gate

A full soft tree has output `sum_leaf pi_leaf(x) f_leaf(x)`, where path probability is the product of left/right gate probabilities. Replacing one affine leaf by a gate and two affine children adds `2(d+1)` parameters. A centered replacement of that leaf preserves the whole tree because its local value is unchanged, including its upstream path weight.

A later score must include the derivative with respect to that leaf output, which incorporates path responsibility. Log-space path accumulation or a stable recurrence is needed as depth grows. Specify maximum depth from observed saturation, sample-per-leaf information, and memory, not a universal hard-coded depth. Depth 2 is the first extension because it tests compositional routing that a stump cannot express as one path. Do not increase depth merely because the stump result is unfavorable.

## 5. Research findings and novelty boundary

### 5.1 Prior-art comparison

The table distinguishes mechanism overlap from evidence of superiority. A related method's published success is not controlled evidence for this proposal.

| Prior art | Problem and mechanism | Evidence / limitations relevant here | Cost, implementation, and exact overlap |
|---|---|---|---|
| Hierarchical mixtures of experts, Jordan–Jacobs [R01] | Hierarchical probabilistic gates combine expert responses, including affine/generalized-linear experts. | Historical supervised modeling and EM; not a contemporary small-tabular benchmark. | Direct overlap with a soft tree of affine specialists. Reuse the mathematics; no need to reinvent a new model family. |
| GLU / Highway networks [R02, R03] | Multiplicatively gated responses and gated carry/transform paths. | Original applications and training regimes differ from this project. | The affine-stump identity is especially close to GLU plus a learned affine carry. Mandatory equivalent-family control. |
| Distributed / budding decision trees [R04] | Differentiable trees with learnable structure/leafness; distributed routing variants. | Historical tree experiments do not resolve modern matched-compute comparisons. | Growth and soft routing are known. Budding-tree material was inspected through the distributed-tree paper and its references, not a fresh reproduction. |
| Adaptive Neural Trees, ANT [R05, R06] | Trainable routers, transformations, and leaves; structural growth selected using validation performance. | Published adaptation is broader than a per-unit affine stump. | Closest growth precedent. Official code is MIT but documents Python 2.7-era dependencies. Port a specified restricted control; do not claim original results reproduced. |
| Splitting Steepest Descent [R07] | Select neuron splits using a loss-derived splitting matrix. | Local optimization analysis, not a guarantee of test improvement. | Strong overlap with derivative-informed allocation. Its activation-second-derivative splitting matrix is not simply the proposed gate–contrast moment. Compare score motivation without claiming identical formulas. |
| Flat Channels to Infinity [R08] | Analyzes neuron splitting, parameterization, and gated-linear limits. | Theory of loss landscapes, not evidence for this allocation recipe. | Invalidates casual claims that neuron splitting or its gated relationships are new; highlights conditioning and parameterization as confounders. |
| LOLIMOT / local model networks [R09] | Incremental local affine models with region-dependent validity functions. | Adjacent system-identification tradition; different region parameterizations. | The goal of allocating local linear expertise is old. Official MATLAB toolbox has noncommercial terms; it is not a mandatory dependency. |
| Dendritic-tree neurons / Backpropagation Neural Tree [R10, R11] | More complex scalar computational units with internal tree structure. | Different wiring and nonlinear computations, not necessarily routing decisions; small or task-specific demonstrations. | Relevant to the level of abstraction, but not proof that this mechanism is novel or biologically faithful. |
| Arboreal Neural Network, 2026 [R12] | Official conference listing and search-indexed descriptions indicate a close tree/cell relationship. | Full paper was blocked; exact equations, growth behavior, and implementation remain unverified. | Mandatory novelty hold. Do not invent an implementation or use a secondary AI summary as evidence. |
| Deep Neural Decision Forests [R13] | Soft stochastic classification trees coupled to learned representations. | Different output/learning setup from scalar affine hidden units. | Establishes differentiable tree–network integration; not the strongest regression baseline by itself. |
| NODE [R14, R15] | Differentiable oblivious-tree ensembles; depth-shared feature choices. | End-to-end tabular learning; different sharing and sparsity than independent stumps. | Directly relevant executable family. Official `odst.py` inspected: train-data initialization, dense leaf responses, feature-selection and binning functions. Preserve the actual configured functions. |
| GRANDE [R16] | Gradient-trained hard decision-tree ensembles. | A hard-routing alternative, not equivalent to soft affine specialists. | Optional hard-tree comparison when pursuing H4; current dependency/backend compatibility must be tested. |
| LightGBM linear-leaf trees [R39] | Piecewise-linear boosted trees without a neural representation. | Official implementation establishes an immediately reusable simpler alternative; not evidence of its performance here. | Mandatory practical control for whether local affine experts need neural growth at all. |
| KAN [R17] | Learnable univariate edge functions. | Adjacent function-learning approach, not a decision-tree neuron. | A learned univariate/spline control addresses proposal A more directly than claiming KAN and trees are identical. |
| RealMLP / TabM [R19, R20] | Strong modern MLP recipes and parameter-efficient ensembling. | Contemporary empirical evidence that simplistic MLP baselines are insufficient. | Mandatory practical competitors before broad accuracy claims. Reuse official implementations and recommended preprocessing. |
| TabPFN-3.5 / TabICLv2 [R22–R24] | Pretrained tabular inference, with reusable weights and context-dependent computation. | Strong but different resource regime; published accelerator timings are not laptop measurements. | Conditional local baselines after license/memory checks; separately disclose pretraining and retained context. |

### 5.2 Findings that change the proposed research

**What already exists?** All four broad ideas in the motivating proposal have precedents: learned scalar functions, tree-like scalar units, trees of experts, and adaptive growth. HME and GLU provide the closest mechanism equivalence; ANT and neuron-splitting methods constrain the growth claim [R01–R08].

**What can be reused?** PyTorch autograd and dense layers; official TabM/RealMLP/NODE code; existing boosting packages; public dataset metadata and files; existing statistical and benchmark design lessons. No custom accelerator kernel is needed to test H1.

**What may be different?** The exact combination of per-unit centered insertion, training-moment screening, optimizer-aware ranking, and a fixed-budget allocation experiment may differ from inspected implementations. This is a plausible differentiation, not verified novelty. The underlying algebra is simple enough that independent prior discovery is credible.

**Why could it matter?** It might reduce how much conditional capacity must be trained, or reduce search cost compared with trying every split. Its value must appear in held-out loss and resource curves; a convenient description alone has no scientific value.

**Which experiment isolates it?** Top-ranked versus uniformly random unit selection with identical growth count, base checkpoint, contrast-direction construction, optimizer treatment, and post-growth training. A second contrast-direction ablation separates scoring from initialization. Static gated and ordinary-growth controls separate adaptation from capacity.

**What invalidates novelty?** A previous publication or implementation of the same insertion/score/allocation rule; an algebraic reduction to an existing growth procedure without a meaningful new result; or the inaccessible 2026 paper covering the claimed distinction. Such a finding changes the project to replication and comparison. It does not justify relabeling the method.

### 5.3 Reconciling apparently conflicting empirical literature

A 2022 study documents strong tree-based performance and tabular inductive-bias issues [R18]. More recent RealMLP, TabM, TabArena, and foundation-model work changes the relevant comparison set [R19–R24]. These observations are not reconciled by declaring a universal winner: dataset populations, tuning, preprocessing, ensembling, pretraining, and compute differ. Therefore historical numbers are contextual evidence, not values to place in this project's controlled result table.

TabArena is useful for understanding how tuning and ensembling affect rankings [R21]. The present small, grouped/temporal suite is intentionally not a recreation of its full benchmark. An improvement here cannot establish state of the art on TabArena or general tabular learning.

### 5.4 Chosen direction versus alternatives

| Direction | Decision and reason |
|---|---|
| Replace every activation with a learned scalar tree/spline | Keep as a cheap control only. It tests univariate flexibility but cannot directly isolate multivariable conditional allocation. |
| Put a deep tree in every unit immediately | Reject as the first experiment. It multiplies optimization and execution confounders before demonstrating a useful allocation effect. |
| Fixed affine-leaf stumps everywhere | Required baseline. It tests the original C mechanism and may be the better engineering solution. |
| One-event selective stump growth | Primary. Preserves the intended low-level change while exposing a falsifiable allocation question. |
| Recursive growth and pruning | Conditional extension after H1/H2, with fresh confirmatory data. |
| Hard-routing acceleration | Separate branch after measured soft–hard discrepancy and runtime profiling demonstrate available headroom. |

Patent search located Microsoft's *Neural trees*, including adaptive neural tree construction [R38]. This technical prior-art screen is neither a patentability determination nor a freedom-to-operate opinion. Legal status and claim coverage are outside this research plan.

## 6. Baselines and data

### 6.1 Three comparison tracks

**Track A — causal component comparison.** Same data transformation, ordinary base checkpoint, final parameter count where exact matching is possible, paired minibatches, and training allowance. This tests why the mechanism helps.

**Track B — local practical comparison.** Method-native preprocessing and official strong recipes, same permissible data/labels, and measured end-to-end cost. Use both common local wall-time budgets and unconstrained-within-feasibility quality–cost curves. An exact parameter match is inappropriate between an MLP and a pretrained model, but its parameter/storage advantage must not be overstated.

**Track C — nonlocal reference context.** Published large benchmark ensembles, foundation-model pretraining, and API-only variants. They are not controlled rivals unless actually run locally under this protocol. Never place published benchmark numbers alongside locally measured values without a clear separation.

### 6.2 Required baseline inventory

| ID | Implementation and required configuration | Role and reproduction check |
|---|---|---|
| B0 | Mean predictor and ridge regression. Tune ridge regularization on V using F-fitted scaling. | Harness sanity and linear/no-regime controls. Compare to closed-form ridge on a small test. |
| B1 | Ordinary ReLU MLP; same depth/width as parent; continued training fork. | Determines whether simply training longer explains improvement. |
| B2 | Wider ordinary MLP under a parameter budget; plus ordinary function-preserving widening from the same checkpoint. | Controls capacity and growth warm-start. For one hidden layer each new unit costs `d+2`; use the nearest feasible width below and above the target and report both counts, never pad with useless parameters. |
| B3 | Static full and partial affine-bypass GLU networks; both direct-leaf and centered coordinates explicitly labeled. | Closest representational family. Because unit indices are permutation-symmetric, a fresh static network needs only the counts of ordinary/gated units; do not charge it unnecessary topology-discovery cost. |
| B4 | Random placement with the proposed per-unit contrast construction; same `m`. | Primary causal control for H1. Random subset variability is averaged within each independent synthetic problem. |
| B5 | Same selected IDs with random RMS-normalized contrasts; SGD-score and Adam-score variants. | Separates direction initialization and optimizer-aware ranking from placement. |
| B6 | Restricted greedy ANT-style split/keep control, defined below. | Closest executable structural-selection control. It is not a reproduction of the unrestricted ANT image experiments [R05, R06]. |
| B7 | Official NODE configuration for regression, pinned repository revision [R14, R15]. | Close differentiable-tree baseline; verify initialization uses F alone and custom activations preserve forward/backward behavior. CPU execution is acceptable. |
| B8 | RealMLP-TD regression and its documented tuning recipe via `pytabkit` [R19]. | Strong MLP competitor; do not silently remove its preprocessing or numerical embeddings. |
| B9 | TabM default and piecewise-linear embedding recipe; `k=32` as in the documented search [R20]. | Strong contemporary ensemble baseline. Train on mean member loss, not loss of the mean prediction; average predictions at inference. |
| B10 | CatBoost CPU, research snapshot `1.2.10`; official macOS universal2 wheels are listed [R25]. | Strong boosted-tree baseline with native categoricals. Confirm the resolved wheel is native arm64, fit succeeds, and logged model parameters match the requested recipe. |
| B11 | XGBoost CPU histogram method, stable release selected by verified compatible wheel and then pinned [R26]. | Independent strong boosting implementation. No CUDA dependency; record OpenMP/native runtime and thread count. |
| B12 | TabICLv2 local regressor, checkpoint `tabicl-regressor-v2-20260212.ckpt`, official defaults including the documented ensemble size [R24]. | Current pretrained baseline, contingent on weights/license/backend gate. Include context-building and prediction cost. |
| B13 | Local open-weight TabPFN-3.5 base model, explicit checkpoint/hash [R22, R23]. | Current strong reference conditional on permitted use and faithful local execution. Do not substitute an API-only Thinking variant or call older TabPFN results current. |
| B14 | LightGBM CPU serial learner with `linear_tree=true`, plus its constant-leaf counterpart; pin the resolved stable release [R39]. | Simpler non-neural local-affine-expert baseline. Use F-fitted feature scaling, inspect linear coefficients after the first tree, and count its higher memory use. |

A spline-feature ridge/additive model is a cheap control for learned univariate functions: fit numerical spline knots on F, concatenate separate per-feature spline bases, and tune ridge on V. It has no learned multivariable gate. An oracle model supplied with the synthetic teacher's gates is an information-advantaged diagnostic, never a fair competing baseline.

**B2 ordinary growth:** the primary widening control adds randomly initialized hidden units with zero outgoing weights. This preserves predictions, leaves old optimizer states untouched, and allows nonzero gradients into the new outgoing weights. A secondary duplication control copies a hidden unit and halves the two outgoing weights; it requires explicit optimizer-state migration and a documented symmetry-breaking treatment. Do not silently perturb a supposedly function-preserving insertion. Do not grant ordinary growth fewer optimizer steps or a worse restart policy than tree growth.

**B6 restricted greedy algorithm:** start from the same ordinary parent. At the growth event, separately try each eligible single-unit stump insertion, train each candidate for the same probe allowance on F, and rank by G loss improvement. Select `m` IDs from these scores, insert them jointly into a fresh parent copy, and apply the normal continuation. Include the continued-parent candidate. This is a validation-probed allocation control inspired by ANT's split/keep principle, not the full published ANT method. Charge every probe, including losers. If this exhaustive control is much better than the moment score but too expensive, the screening cost–quality tradeoff is the remaining research question.

### 6.3 Baseline strength, tuning, and versions

Use two locks: a minimal core environment and separate environments for incompatible baselines. Every environment writes a package lock, Python/macOS/architecture record, package metadata, source revision when applicable, and exact model checkpoint identifier. A floating `latest` reference is not an experimental version.

The plan deliberately does not fabricate an untested universal dependency combination. At WP00, choose the newest mutually compatible stable native-arm64 Python/PyTorch/package combination supported by the actual laptop; first try Python 3.12 where package support permits. Pin the resolved versions. Preserve official configs as files under `configs/vendor/`, with source URL and retrieval date. If a later release is used, document why and re-run adapter tests.

For the custom MLP/gated family, initialize the practical search with learning rate log-uniform `[1e-4,5e-3]`, weight decay including zero and log-uniform `[1e-6,1e-1]`, depth 1–4, and widths spanning 32–512 on a multiplicative grid. These are initial search supports, not exclusion of larger or smaller useful models: if a winner lies at a boundary in exploratory validation, expand that boundary before protocol freeze. Fix causal Track A architecture and share its optimizer choices across forks; do not tune only the proposed fork.

TabM's inspected documentation specifies `k=32`, width 64–1024 in steps of 16, learning rate `[1e-4,5e-3]`, and both plain and numerical-embedding searches. Its default AdamW recipe uses learning rate 0.002 and weight decay 0.0003 [R20]. Preserve those anchor configurations even when adding a local-budget search. RealMLP's documented defaults are a required anchor, not a generic MLP labeled RealMLP.

Use a seeded TPE search for practical HPO, with identical sampler-policy accounting across methods; the first trials include required official defaults. Archive the sampler state. For CPU boosting, seed the search with official defaults and explore learning-rate/depth/regularization tradeoffs. Positive rate/penalty ranges below are log-uniform, row/column fractions uniform, and listed depths categorical. A concrete initial support is CatBoost depth `{4,6,8,10}`, learning rate log-uniform `[0.01,0.2]`, L2 leaf penalty `[1,30]`; and XGBoost max depth `{2,4,6,8}`, learning rate `[0.01,0.2]`, min-child-weight `[0.1,100]`, row/column fractions `[0.6,1]`, and L2 penalty `[1e-3,100]`. For B14, include `num_leaves` in `{4,8,16,32,64}`, learning rate `[0.01,0.2]`, `min_data_in_leaf` in `{5,10,20,50}`, and `linear_lambda` including zero and log-uniform `[1e-6,10]`, expanding boundaries during exploration when warranted. Set `linear_tree` when constructing the native dataset as well as the training configuration. Its first tree has constant leaves, and later linear leaves use numerical branch features, so it is not algebraically identical to the unrestricted affine-neuron family [R39]. Tree count is set by G-based early stopping or the common budget, with a raised ceiling whenever training is still improving at it. These are project search choices, not claims about universal best settings. Use squared-error objectives and no categorical target encoding fitted outside F.

Exploratory budgets are calibrated by doubling elapsed search allowances, inspecting validation learning curves, and comparing marginal gains with validation noise. This is a way to identify under-tuned baselines, not a significance test. Before confirmation, freeze a common wall-time budget and a stronger quality-oriented budget informed by the slowest credible baseline and hardware measurements. Include the official default result regardless of search quality. If a strong baseline cannot finish even a faithful default, report it as unavailable at that budget and restrict the resulting claim; do not silently replace it with a weaker model.

Pretrained models may be evaluated first with official inference defaults, then documented cache/ensemble choices. Fine-tuning becomes an additional candidate only when a measured local budget supports the documented training path. Pretraining cost is not charged as project training time, but pretrained parameters, download size, training-data dependence, cache bytes, and unavailable pretraining compute are disclosed. Small newly generated synthetic test problems reduce direct benchmark contamination risk; they do not erase the value of a learned prior.

### 6.4 Fixed transfer suite and provenance

The suite is deliberately small enough for local repeated experiments and spans different data-generation conditions. It is **five observational datasets plus one simulated dataset**, not six independent demonstrations of deployment value. Select it now, before seeing model results. Each UCI record currently provides a CC BY 4.0 license; archive the exact license/metadata with raw-file checksums and verify it again at acquisition [R27–R31, R34]. Use raw filenames and hashes as the dataset version, because a repository ID alone does not freeze bytes.

| Dataset / verified UCI ID | Data and primary target | Required split and leakage precautions |
|---|---|---|
| Airfoil Self-Noise / 291 [R27] | 1,503 rows, 5 numerical inputs; sound-pressure target in dB. | Group exact test configurations by angle, chord, speed, and displacement thickness, omitting frequency from the group key. This is a proposed conservative grouping proxy; verify actual group counts. |
| Concrete Compressive Strength / 165 [R28] | 1,030 rows, 8 inputs; compressive strength in MPa. | Group identical seven-ingredient mixtures across ages; keep age as an input. No duplicate-mixture leakage across partitions. |
| Energy Efficiency / 242 [R29] | 768 simulated building cases, 8 inputs; heating load only in the primary report. | Exclude cooling load from inputs. Group building geometry across orientations/glazing configurations. Report separately from observational transfer; limited geometry count limits inference. |
| Combined Cycle Power Plant / 294 [R30] | 9,568 rows, 4 inputs; net output in MW. | Workbook contains five shuffled versions for the original evaluation. Use one, not five stacked copies; verify row multisets. No temporal claim without timestamps. Group exact duplicate input records. |
| Parkinsons Telemonitoring / 189 [R31] | 5,875 measurements from 42 people; motor UPDRS target. | Use age, sex, test time, and the 16 voice measures; exclude subject identifier and total UPDRS from inputs. Split by subject. Targets include interpolation; this is not a diagnostic or clinical validation study. |
| Bike Sharing / 275 [R34] | Hourly table, 17,379 rows; total rental count `cnt`. | Drop `casual` and `registered` because they sum to the target; drop `instant`. Use forward calendar splits by day, never random-hour splits. Observed weather makes this retrospective conditional-demand prediction, not a verified weather-forecast deployment. |

All dimensions are source metadata and must be asserted against acquired bytes, with documented differences. For Airfoil, inspect numeric values rather than relying on a possibly misleading categorical type label. For Bike, use the documented calendar/weather categories, numerical weather variables, year/month/hour/day-of-week indicators, and a parsed date only to establish chronology; do not add future-derived aggregates. Use raw categories for CatBoost and explicit train-fitted categorical encoders for neural baselines.

**Missingness and duplicates:** assert target finiteness; do not impute labels. For custom networks, numerical missing values use F medians plus missingness indicators, and categorical unknowns map to a reserved unknown category. No whole-table quantile transforms, feature screening, target scaling, or supervised preprocessing. Exact duplicated inputs remain in the same split; retain their multiplicity unless a source establishes accidental duplication. Never remove hard examples because they make the proposal lose.

**Unavailable data:** retry the official mirror/download route and verify identity. If license or bytes cannot be verified, block that dataset rather than silently substituting a convenient winner. Before any tests are revealed, a recorded amendment may change the suite and its claims. After results are revealed, replacements belong to a new exploratory study and need fresh confirmation.

### 6.5 Synthetic population and controls

Use independent teacher seeds, data seeds, and optimizer seeds. A teacher is a randomly generated target function, not a single fixed example repeatedly resampled and called many tasks.

The main population uses `x ~ N(0,I_16)` and four independently drawn regime components. Draw unit-norm Gaussian vectors `u_r,v_r,w_r`; sample the gate prevalence from `{0.1,0.5}` and use the corresponding standard-normal threshold. Independently choose a hard threshold or a smooth sigmoid gate with transition scale `{0.2,1.0}`. Define

\[
f(x)=0.3\sin(u_0^Tx)+\frac{1}{\sqrt4}\sum_{r=1}^4
\left[0.3u_r^Tx+\bigl(2g_r(x)-1\bigr)w_r^Tx\right].
\]

Normalize this teacher's output using 65,536 separately seeded reference observations generated only for defining the distribution, not any model's fitting/validation/test observations. This is a fixed generator convention, not a cap on experimental sample sizes. Store that normalization with the teacher. Set `y=f(x)+σξ`, `ξ~N(0,1)`, with `σ=0.1` at the primary coordinate and zero/high-noise conditions as sensitivity tests. Neither teacher gates nor teacher normalization parameters are supplied as predictor features. The sine term and hard-gate cases prevent the entire population from being exactly the proposed smooth affine-stump model class.

Initial primary coordinates are `h=64`, `m=16`, and `n_F=2P0=2,306`, where `P0=1,153`. Set `n_G=n_V=ceil(n_F/4)`. The primary growth event is the base checkpoint selected by the common stopping rule, not a point hand-picked for a favorable gain. Sample densities `n_F/P0 ∈ {0.5,2,8}` probe estimation versus approximation limits. These coordinates can be amended after **exploratory-only** calibration, before independent confirmation; any amendment must justify sensitivity or feasibility, not conceal unfavorable results.

Required controls are a globally affine target, a smooth additive sine/quadratic target, a purely noisy target, an independent random-label permutation, the original two-dimensional conditional-slope example, axis-aligned versus obliquely rotated regimes, added irrelevant features, and rare-region prevalence. Add a nested-gate teacher only when evaluating depth 2. Oracle-gate least squares checks that the data generator and task metric expose the intended signal. A smooth control where every component has similar complexity tests whether selective allocation loses its supposed advantage when heterogeneity is absent.

Choose test sample count by Monte Carlo precision: on exploratory teachers, estimate the variance of paired per-example loss differences and choose a fixed test size making per-teacher test-sampling standard error no more than one tenth of the pilot between-teacher standard deviation. Compute the required count as `max(1, ceil(100 * variance_of_paired_row_loss / between_teacher_variance))` using conservative pilot variance estimates; if the denominator is unresolved or zero, use the equivalence/estimation branch in Section 7.5 rather than an infinite or arbitrary claim of precision. Freeze that size for all confirmatory teachers. Do not enlarge only test sets whose effect is close to significance.

## 7. Evaluation and confirmatory protocol

### 7.1 Freeze before test access

Maintain separate exploratory and confirmatory namespaces. Exploratory synthetic teacher/data seeds are never reused for confirmation. Build split manifests using inputs, grouping keys, and dates, without fitting models on T. All transfer datasets are chosen before observing comparative results.

Before confirmation, write and hash `protocol/frozen.yaml`. It specifies the exact hypotheses, primary coordinate, independent units, metrics, data hashes, exclusions, preprocessing, baseline revisions, tuning algorithms and budgets, seeds, training stopping rules, sample/repetition counts, timing procedure, and analysis code revision. A test command refuses to run without this file and refuses configuration drift. Test access is an auditable procedural protection, not a claim that local files are cryptographically inaccessible to their owner.

Methodological changes after test access create `protocol/amendments/*.md`, mark old results as exploratory for the changed claim, and require fresh independent confirmation. Do not keep modifying the method on the same protected test set.

### 7.2 Data partitions

For synthetic problems, generate independent F, G, V, and T observations under each teacher. All paired methods use identical observations. Hyperparameters for the **causal H1 experiment** are chosen on exploratory teachers and frozen across confirmatory teachers; there is no per-teacher HPO advantage for the proposed fork.

For the five non-temporal transfer datasets, use five outer folds at the group level. Assign groups deterministically by a seeded hash and greedy row-count balancing, without looking at labels. Each outer T contains whole groups; its development complement is divided approximately 70/15/15 into F/G/V by the same group-respecting procedure. Require at least two groups in each inner partition; when small group counts prevent the nominal fractions, use the closest feasible allocation and record actual counts. If fewer than six development groups exist, nested confirmation is unsupported: report that dataset descriptively or create a pre-test amendment, not row-level pseudo-replication.

For Bike, sort unique dates. Use five forward test windows bounded by the 50%, 60%, 70%, 80%, 90%, and 100% date quantiles. Each fold trains only on dates before its window. Within that prefix, use chronological F/G/V with the same nominal fractions. Exclude the 24 hours immediately before each G/V/T boundary as a one-day embargo. Never train on a future fold to predict an earlier fold. This tests later-period performance from earlier data, not arbitrary future operational conditions.

All fitting, checkpoint selection, and HPO happens inside an outer fold. F supplies training labels; G selects checkpoints and growth/keep decisions; V selects hyperparameters and the baseline champion. In the primary experiment, **do not refit the winning pipeline on F+G+V**, because preserving the growth trajectory and matching label access are easier to audit without it. A later conventional full-development refit is a separately labeled practical analysis, with frozen topology, schedule, and transformations refitted only on the permitted development data.

Foundation-model context contains F labels only in this protocol. Methods that internally split F for their documented recipe may do so; log the effective fitting rows. Disable any undocumented use of T labels. T features may be processed in a documented transductive batch by a foundation model, but that transductive use must be disclosed and tested against batch composition; it is not identical to per-row inductive prediction.

### 7.3 Primary and secondary metrics

Primary predictive risk is row-averaged normalized mean squared error:

\[
R=\frac{1}{n_T}\sum_{i\in T}(\hat y_i-y_i)^2/s_F^2,
\]

where `s_F²` is the target population variance on F. If it is zero, normalized risk is undefined; report unnormalized squared error and classify the task as a degenerate target diagnostic. Do not silently divide by a tiny arbitrary number and include an enormous normalized score in an aggregate.

For a pair, define `D=R_control-R_proposed`; positive values favor the proposal. Units are squared-error divided by fitting-target variance. Report raw MSE, RMSE and MAE in the source target units as well. On synthetic data also compute noise-free approximation risk against the known `f(x)`; this is a diagnostic metric, not an undisclosed replacement for noisy-label test risk.

Report group-macro error, worst prespecified regime/group slices, and tail absolute error as guardrails. Row averaging and group averaging answer different questions, especially for repeated patient measurements. Record both rather than allowing prolific groups to hide failures.

Primary cost outcomes are total method-specific local search-and-fit seconds, post-selection fit seconds, first prediction seconds, warm prediction latency at batch 1 and representative batched workloads, throughput in rows/second, peak observed memory metrics in bytes, serialized model bytes, retained training/context bytes, and trainable/total parameter counts. Record operation estimates separately from timings.

An aggregate transfer table averages normalized risks equally over the **five observational datasets**. Energy is shown separately. Outer folds contribute predictions according to their covered rows; fold estimates are not independent experiments. Dataset-level ranks and geometric means of positive risk ratios may be secondary summaries, with absolute differences always present. No global “percent improvement” is reported without naming whether it refers to MSE, RMSE, latency, or model bytes.

### 7.4 Independent units and uncertainty

For H1, the independent unit is a newly sampled teacher plus its independently sampled dataset. Average optimizer/random-placement repetitions within that unit; do not count them as new problems. Estimate the paired mean effect across independent units and its 95% confidence interval. Use a paired bootstrap resampling whole units, with a Student-t interval as a sensitivity check. Inspect skew/outliers; report median and the full effect distribution. Bootstrap repetition count is chosen so Monte Carlo error of interval endpoints is negligible relative to the reported precision, using a fixed bootstrap seed.

The transfer suite is a fixed collection, not a random sample of all possible datasets. Report each task's paired effects and the finite-suite aggregate. Conditional uncertainty from held-out sampling can be estimated by resampling whole test groups within datasets, retaining method pairing and predictions. For Bike use moving date blocks, with block length selected using development residual dependence before T access and sensitivity to half/double that length. Report training/HPO variability separately using independent pipeline repetitions. These intervals do not fully capture uncertainty from retraining on new populations; do not label them universal algorithm-generalization intervals.

Five overlapping outer training folds are not five independent datasets. Repeated timings are not independent prediction samples. A seed-level t-test over hundreds of runs on one fixed dataset is not the primary significance test [R33].

### 7.5 Pilot-calibrated sample and repetition counts

The pilot has two purposes: estimate where variance comes from, and measure local cost. Begin with at least two optimizer/random-placement repetitions per independent teacher so within-teacher variance can be separated from between-teacher variance. Begin with eight independent pilot teachers, an initial batch rather than a stopping quota, and add batches by doubling their count. Stop variance calibration when a whole-teacher bootstrap 95% interval for the paired-effect standard deviation has upper/lower ratio at most 2 and the projected per-teacher runtime is stable across the last two batches within 20%; archive both diagnostics. These are uncertainty/cost calibration tolerances, not required performance improvements. If variability is effectively zero, verify a same-model/equivalence explanation instead of dividing by zero. Calibration does not stop merely because an effect is significant. More optimizer seeds are useful only when they meaningfully reduce within-teacher uncertainty at lower cost than adding independent teachers.

Use a conservative pilot estimate `s_D` of the standard deviation of teacher-level paired effects. For a declared planning effect `δ_plan`, a normal-approximation design is

\[
K\approx\left\lceil\frac{(z_{0.975}+z_{0.8})^2s_D^2}{\delta_{plan}^2}\right\rceil.
\]

This is a planning approximation, not a guarantee. Account for the chosen multiplicity level if more than H1 is confirmatory. Compare the calculation to resampling-based power under the observed effect distribution. Also publish the precision–cost curve and the effect detectable at the selected `K`.

There is no stated application tolerance from which to obtain `δ_plan`. Use this explicit design rule: if the pilot paired 95% interval for H1 excludes zero, set `δ_plan` to half the absolute pilot mean difference, regardless of its sign. Otherwise, if the prespecified static-partial-GLU versus upper-parameter-bracket ordinary-MLP control (the smallest ordinary width with at least the gated model's parameter count) has a pilot interval excluding zero, set `δ_plan` to half that absolute control difference. Use the upper 95% bootstrap bound on the pilot standard deviation as `s_D`. These half-effect choices guard against an optimistic pilot and are planning effects, not minimum improvements required for success. If neither difference is resolved, run an independent estimation cohort with the same number of teachers and repetitions as the completed pilot, report its attainable interval width, and accept a potentially inconclusive answer rather than pretending it was powered for an arbitrarily tiny effect. When a calculated confirmation workload is physically impractical, write the measured constraint and attainable precision before any confirmation access; either retain an explicitly underpowered estimation study or block the larger claim. The numeric calculation and local forecast go into `pilot/design_decision.md`, a blocking WP05 output.

Once `K`, inner repetition counts, and test sample sizes are frozen, run the full independent cohort regardless of intermediate signs. No repeated testing until a positive result appears. Additional confirmation after that is a new registered cohort with its own analysis; combine only under a predeclared rule.

### 7.6 Claims, multiplicity, and practical significance

H1 is the sole initial confirmatory superiority hypothesis at the primary coordinate. Its two-sided paired 95% interval answers whether the effect differs from zero. A positive interval supports only the stated population and procedure. An interval entirely below zero is evidence against beneficial allocation there. An interval spanning both meaningful gains and regressions is inconclusive, not “no effect.”

Additional superiority claims across H2 contrasts, alternative primary coordinates, or individual transfer tasks require a declared family and Holm adjustment. Otherwise label them descriptive/exploratory. Do not convert whichever secondary metric wins into the primary claim.

A statistically resolved tiny effect is not automatically useful. Show its absolute target-unit size, training/search cost, inference cost, and regressions. Because no stakeholder tolerance is specified, describe practical magnitude rather than declaring an invented minimum viable improvement. A Pareto claim requires that all named resource/quality dimensions support it; one accuracy win with higher latency is a tradeoff, not dominance.

For H3, select the competitor champion using V among the eligible locally executed baselines, then evaluate that selection on T. Do not select a different “best rival” using T and test only against convenient losers. Report the complete baseline table regardless of who becomes champion. Published larger-scale results stay in Track C.

### 7.7 Training and search budgets

For Track A, pair the base checkpoint and the number of post-growth optimizer updates, use the same data order, and retain the same checkpoint rule. Report total examples processed as well as updates. The continuation allowance and early-stopping patience are calibrated on exploratory learning curves so neither grown nor ordinary models are still systematically improving when cut off. Checkpointing may stop a particular run early, but the unused allowance is not covertly reassigned only to the favored method.

For Track B, charge preprocessing, base training, score computation, rejected probes, hyperparameter selection, final fitting, and compilation when used. Shared raw-data download is separately reported; algorithm-specific model downloads are separately reported as cold acquisition cost. Give static baselines the same opportunity to spend the saved growth/search cost on tuning or training. For a standalone random-placement method, compute contrasts only for its sampled IDs when possible; do not charge it the full top-m ranking cost simply because the paired research harness happened to share all moments.

Choose common local budget levels from measured search learning curves: a low-budget level that completes each fast official default and a quality-oriented level near the strongest baseline's observed validation plateau. Extend exploratory budgets when boundary configurations or improving curves indicate under-tuning. Freeze the resulting numeric levels before confirmation. An infeasible baseline can remain a nonlocal reference, but missing it limits the competitiveness claim.

## 8. M3 Max feasibility and execution measurements

### 8.1 Hardware and operating-system discovery

Run these existing system commands and preserve their output:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.logicalcpu
df -h .
xcode-select -p
python3 -c 'import sys, platform; print(sys.version); print(platform.machine())'
```

Check AC power state and competing workloads; do not require privileged system changes. Confirm the process is native `arm64`, not accidentally under Rosetta. Discover actual available disk and unified memory instead of assuming a maximum M3 Max configuration.

Apple and PyTorch documentation differ in their displayed minimum macOS requirements across release tracks: the inspected Apple page refers to macOS 14+, while the inspected current PyTorch MPS guidance includes macOS 15+ checks [R35, R36]. Treat support as **version-specific**. Select a stable wheel appropriate to the actual OS; record the exact version. Do not automatically update macOS or erase environments to satisfy a guessed minimum.

```python
import torch
print(torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
```

Availability is only a first check. Verify the actual operations: float32 GEMM, sigmoid/tanh, ReLU, index selection/copy, autograd, AdamW with a conservative `foreach=False` path, serialization, and resumed training. Use CPU float64 for eigenvalue decomposition and gradcheck. Do not attempt to store float64 tensors on MPS or treat nominal Apple Silicon support as kernel-level validation.

### 8.2 Software strategy and licensing

Core packages are PyTorch, NumPy, SciPy, pandas, scikit-learn, a YAML parser, and pytest; Optuna is the optional practical-search dependency after the causal harness works. Use a native virtual environment and a resolved, pinned lock. Reading the small CCPP workbook can use `openpyxl`; no spreadsheet application is required. Use a small CLI package for project orchestration rather than introducing a distributed training framework.

Install optional baselines in isolated environments when their dependency constraints differ. LightGBM documents native macOS build/install routes; a Python wheel may require a compatible OpenMP runtime, which must be checked rather than assumed [R39]. Do not change system toolchains or install privileged components without authorization. CatBoost's inspected release provides macOS universal2 wheels [R25]; this is evidence of a distribution path, not proof of correct execution on this laptop. NODE's custom sparse functions and data-aware initialization may require a faithful CPU path. ANT's old requirements should not force the core environment back to Python 2 [R06, R15].

TabM code is Apache-2.0, ANT is MIT, and the inspected TabICL code license is BSD-3-Clause; record licenses for all copied modules [R06, R20, R24]. TabPFN code and weights have distinct licenses, and the current 3.5 open weights have noncommercial restrictions [R23]. Do not equate downloadable weights with unrestricted open-source use. Verify the exact intended use and weight terms before acquiring or redistributing an asset; preserve notices, and do not bundle baseline weights with project results by default. If an access gate requires an agreement the agent is not authorized to accept, mark it blocked rather than bypass it.

### 8.3 Peak-memory model

For a float32 model with `P` trainable parameters and AdamW, a first estimate is

\[
M_{model+optimizer}\simeq16P\text{ bytes}
\]

for weights, gradients, and two moment tensors. This excludes optimizer temporaries, allocator overhead, saved activations, checkpoint copies, libraries, and the operating system. A rough layer-activation estimate is `4B Σ_l c_l h_l`, with empirically calibrated `c_l` reflecting saved projections/gates/nonlinearities. For a conditional layer it is larger than for one affine layer.

Add host/device data copies of roughly `4N(d+1)` bytes each, preprocessing copies, CPU moment matrices of `8h(d+1)^2` bytes if retained simultaneously, model export/checkpoint staging, and the Python/framework runtime. Serialize forks and load one at a time rather than keeping every candidate on device. At `d=32,h=64`, all moment matrices occupy only about 0.53 MiB; at thousands of features this approach becomes inappropriate. The primary first-layer suite is low dimensional, so dense score construction is plausible, subject to timing.

A 17,379-by-64 float32 feature matrix is about 4.24 MiB. Even several copies are small relative to model/runtime allocations. A `P=10^6` AdamW parameter state alone is about 15.3 MiB; total runtime can be much larger. These are byte calculations, not measurements of process or GPU memory.

Foundation-model memory must be measured separately: weights, retained context, preprocessing ensembles, attention/workspace, KV caches, and offload buffers can dominate. Do not extrapolate from the small-network parameter formula. Cache construction and prediction batch size can move cost between `fit` and `predict` [R23, R24].

MPS allocated/driver memory and process RSS refer to overlapping portions of unified physical memory; **do not sum them as disjoint quantities**. Record them separately alongside system memory pressure and swap changes [R37].

### 8.4 Calibration and adaptive configuration

Begin with one small synthetic batch and one training step on CPU and MPS. Compare outputs, loss, and parameter gradients with explicit tolerances appropriate to float32; use CPU float64 as the numerical reference. Then benchmark a complete epoch and a scoring pass at several batch sizes, doubling batch size until throughput stops improving or the safety margin is reached. Select the smallest tested batch achieving at least 95% of the maximum median throughput among memory-safe tested batches; this is an engineering calibration tolerance, not a required model-performance improvement. Freeze the logical batch size before the controlled comparison so calibration does not introduce unequal optimization changes.

Reserve at least `max(8 GiB, 0.25 × physical RAM)` for OS/framework fluctuations as an initial safety rule; tighten it if actual pressure demands. Use the lesser of current usable headroom and a conservative fraction of the MPS recommended working-set limit as the accelerator budget. These are safety reservations, not model-size or runtime ceilings. Inspect `torch.mps.recommended_max_memory()` where available and preserve normal allocator safeguards [R37]. Never set a watermark to zero to force a run through memory pressure.

Run with implicit MPS fallback disabled while validating kernels. When an operation is unsupported, either move the **whole method** to CPU or implement and verify an algebraically equivalent path, then disclose it. CPU execution is faithful for H1; it changes the hardware-performance claim. Do not replace NODE's sparse choice function with sigmoid and still call it NODE.

Reducing prediction batch size or serializing ensemble members is often a faithful memory fallback, but verify output parity. Reducing context rows, ensemble size, precision, or model size changes the tested baseline; name it separately and preserve the original baseline's unavailable status. For pretrained models, first follow documented CPU/offload support and measure cost; do not assume CUDA-only fused attention has a faithful fast MPS equivalent. The inspected TabPFN README recommends PyTorch 2.13 or newer for Apple Silicon and documents a 5,000-row CPU allowance for its current 3/3.5 families [R23]. Some full-F transfer folds can exceed that allowance. Do not disable guards or silently reduce context rows; use a verified supported MPS path, or mark that baseline/fold unavailable. A separate environment may be needed even when the smaller core works on an older compatible PyTorch release.

### 8.5 Stage resource forecasts

| Stage | Resource model / calibration | Escalation condition |
|---|---|---|
| Algebra and adapter checks | Tiny tensors; CPU float64 and a small float32 MPS batch. Framework installation dominates disk usage. | All indispensable operations and invariants pass. |
| One base fit and scoring pass | Measure seconds/update, updates to G plateau, scoring seconds, and memory. Estimate `t_fit = steps × t_step + validation + I/O`. | Fit and continuation curves are interpretable, with no resource pressure. |
| H1 pilot | Sum base, all fork, and scoring costs over independent teachers and repetitions. Shared base work is counted once computationally but disclosed for each standalone method. | Variance/precision and learning-curve measurements support a fixed design. |
| Confirmatory H1 | Frozen teacher count × measured per-teacher cost, including failures and full allowed continuations. | Expected precision justifies the local workload; run serially and resume safely. |
| Transfer suite | Sum per-fold, per-method search costs plus final fits and timing. Do not multiply a single fastest model's time by the number of baselines. | Real baseline adapters and license checks pass; useful question remains after H1/H2. |
| Foundation models | Measure default checkpoint load, one complete context build and prediction, then ensemble/cache memory. | Faithful full-F execution fits or documented CPU/offload is usable. Otherwise restrict claims. |
| Recursive/hard extension | Measured path probabilities, discrepancy, per-layer runtime, and added state. | Evidence of compositional need or actual routing headroom, not merely enthusiasm. |

No numerical laptop wall time is claimed here. A complete run can be substantially longer than the small checks; elapsed-time forecasts must come from this laptop, and long runs are acceptable when the scientific precision and information gain justify them. Store the forecast before execution and explain deviations afterward.

### 8.6 Timing protocol

Use `time.perf_counter_ns()` around declared measurement boundaries. Call `torch.mps.synchronize()` immediately before and after timed MPS regions; otherwise queued work yields misleading timings [R37]. Use inference mode and the same evaluation semantics. Report CPU thread counts and do not oversubscribe parallel trials.

Measure separately: fresh-process import/load and first prediction; warm batch-1 prediction including host-to-device input transfer and device-to-host output transfer; device-resident kernel-only prediction; and batched full-pipeline throughput. For context models include both uncached and documented cached modes, retained training data, and cache-building time. A tiny `fit()` call does not mean learning is free if computation occurs inside `predict()`.

Warm up until a rolling timing window is stable within the predeclared measurement precision; log the discarded warm-up work. Interleave method order in ABBA or randomized balanced blocks, pause on demonstrable thermal/system interference, and retain exclusions with reasons. Calibrate repetition count using the pilot timing variance to estimate median and tail latency at useful precision. Report block-level intervals and distribution summaries; repeated microbenchmarks are not new datasets. Optional energy measurements require a supported, understood local instrument and explicit boundaries; do not infer joules from a model's parameter count.

### 8.7 Checkpoint, interruption, and recovery

Save model/topology, optimizer and scheduler states, Python/NumPy/Torch RNG states, available MPS RNG state, data order/sampler position, epoch/update count, preprocessing, growth manifest, selected IDs/contrasts, config hash, split hashes, and code revision. Use write-to-temporary plus atomic rename and a checksum. Never overwrite the last valid checkpoint before the new one is complete.

At growth, migrate optimizer state by stable parameter name and explicit slice mapping. Old base/head parameters and moments are unchanged; new gates/contrasts have zero moments and their own step state. A unit test compares uninterrupted and interrupted/resumed runs through the growth event. Bitwise identity across CPU and MPS is not assumed; same-backend deviations beyond the declared tolerance block resumption claims.

On out-of-memory or severe pressure: retain the failed-run record, reload the last consistent checkpoint, release caches only outside timing, and retry a documented memory-safe path. A changed batch size alters the optimization path; either use verified gradient accumulation to preserve the logical batch or restart the affected paired comparison. Do not resume halfway through an optimizer update or silently mix old and new split manifests.

## 9. Risk-ordered work packages and decision gates

All `python -m treeneuron.cli ...` commands below are **proposed project commands that the coding agent must implement**. They are not existing installed package interfaces. System discovery commands and the Appendix A Python script are independently runnable. Implement only the infrastructure needed for the next gate; do not build a general neural-architecture-search platform first.

### WP00 — Reproducible prerequisites and source/asset gate

**Objective / risk:** avoid building on an unsupported backend, unavailable asset, or unsupported novelty claim.  
**Dependencies / inputs:** this plan, a fresh target laptop, public internet for permitted downloads; no prior experiment results.  
**Actions:** create the repository and core environment; run Section 8 discovery; pin dependencies; archive package and dataset license metadata; save the official baseline configuration snapshots. Attempt full-text acquisition of R12 through the official conference/OpenReview route and any author-provided public copy, without bypassing access controls. Compare any newly accessible equations and algorithm against Section 4.  
**Files / command:** `src/treeneuron/doctor.py`, `environment/`, `provenance/`, `protocol/prior_art_delta.md`; `python -m treeneuron.cli doctor --output environment/report.json`.  
**Outputs / resources:** machine-readable hardware/backend report, resolved locks, source/asset manifest, novelty status. No large model checkpoint download until license and estimated memory gates pass.  
**Acceptance:** core CPU float32 and float64 tests run; intended MPS operations pass or a faithful CPU path is selected. No unknown licenses on copied code.  
**Branches:** unsupported MPS → CPU core, retain separate performance scope; incompatible baseline → isolate its environment; blocked R12 → continue empirical work but keep methodological novelty provisional; exact prior implementation found → adopt it as a baseline and reclassify the project as replication/refinement before confirmation.

### WP01 — Mathematical and implementation falsification

**Objective / risk:** establish that the proposed unit and insertion implement the stated mechanism.  
**Dependencies:** WP00 core environment.  
**Actions:** reproduce Appendix A; implement vectorized `SelectiveAffineLayer`; test exact forward mapping, shared-coordinate gradient mapping, parameter counts, post-mixture activation convention, zero-growth behavior, insertion, dead zero–zero initialization, analytic score gradients, and finite values. Symmetrize accumulated moment matrices as `(M+M.T)/2` before eigendecomposition. Add sign/permutation/rescaling tests and finite-difference checks away from ReLU kinks.  
**Files / command:** `models/units.py`, `growth/score.py`, `tests/test_algebra.py`, `tests/test_growth.py`; `python -m treeneuron.cli verify-math`.  
**Outputs / resources:** JSON with max-absolute/relative errors, norms, parameter counts, backend/dtype, seeds; tiny CPU tests plus one MPS parity batch.  
**Acceptance:** float64 identity/gradient discrepancies below `1e-10` on the specified random tests; function-preserving insertion equal within float32 backend tolerance; nonzero gate gradient in a constructed nonstationary case. Any less stringent MPS tolerance must be justified by CPU reference and no systematic error.  
**Branches:** algebra failure → fix implementation, do not run predictive comparisons; near-zero score on a fitted sample → inspect whether residual/gradient information is genuinely absent; unsupported operation → faithful fallback, with speed claim restricted.

### WP02 — Data and evaluation harness

**Objective / risk:** prevent leakage, pseudo-replication, and a broken metric from generating apparent gains.  
**Dependencies:** WP00; does not depend on a successful proposed model.  
**Actions:** implement teacher generators and raw-data adapters; create immutable row IDs, group IDs, dates, source hashes, and split manifests. Test prohibited columns and disjoint groups. Build nested evaluation and a test-access freeze guard. Fit all transforms using F only. Implement B0 and the row/group-aware metric pipeline.  
**Files / command:** `data/`, `evaluation/splits.py`, `evaluation/metrics.py`, `tests/test_leakage.py`; `python -m treeneuron.cli prepare-data --manifest configs/data.yaml`.  
**Outputs / resources:** validated feature/target schema, per-partition row/group counts, raw-file/transform hashes, label-availability table; small local data only.  
**Acceptance:** target-derived Bike columns are rejected, Parkinson subjects cannot cross partitions, CCPP shuffled copies are not stacked, synthetic gates remain hidden from predictors, and transforms do not change when only T labels change.  
**Branches:** same-method duplicate produces a difference → stop and fix harness; oracle or mean/ridge controls behave incorrectly → fix generator/metric; insufficient groups → descriptive task or pre-test amendment; source/license unavailable → block that task, no silent substitution.

### WP03 — Baseline correctness before competitiveness

**Objective / risk:** avoid concluding that the proposal is good because competitors are broken or weak.  
**Dependencies:** WP01 and WP02 for shared interfaces.  
**Actions:** implement B1–B7 and official adapters for B8–B11 and B14. Use vendor unit tests/examples where available, small overfit checks, output-shape/finite-loss tests, official default anchors, and independent CPU comparison of custom sparse functions. Test TabM's member-loss semantics explicitly. Pin NODE's actual experiment choice/bin functions and initialize only on F; constructor defaults alone do not establish the paper configuration. Gate B12/B13 by license, actual regression checkpoint, memory, and output parity.  
**Files / command:** `baselines/`, `configs/vendor/`, `tests/test_adapters.py`; `python -m treeneuron.cli verify-baselines --track causal`.  
**Outputs / resources:** adapter capability matrix, reproduction notes and discrepancy log, measured minimal/default run costs. No broad tuning yet.  
**Acceptance:** mandatory causal controls run; strong practical baselines have a tested faithful path or an explicit documented exclusion. A published score is not “reproduced” unless the same split/preprocessing/configuration was actually used.  
**Branches:** failed baseline → debug before comparing; legacy code only → port minimal specified behavior with parity tests and label the restriction; current strong model infeasible → retain as Track C and prohibit claims of beating that unavailable model.

### WP04 — Minimum complete growth experiment

**Objective / risk:** test whether the local score does anything useful before substantial experimentation.  
**Dependencies:** WP01–WP03.  
**Actions:** train one ordinary parent on exploratory teachers; build the score; fork top-m/random-m/keep/random-direction models sequentially; preserve optimizer state and predictions at insertion. Log predicted first-step decrease versus observed tiny gate-step decrease, post-growth loss curves, gate saturation, leaf disagreement, and G outcomes. Include static GLU and ordinary-capacity controls.  
**Files / command:** `growth/insert.py`, `train.py`, `configs/explore.yaml`; `python -m treeneuron.cli run --config configs/explore.yaml`.  
**Outputs / resources:** complete run records, per-unit scores/eigenvalues/contrasts, prediction-preservation checks, training curves, first local throughput and memory measurements.  
**Acceptance:** the indispensable mechanism learns in at least a positive-control case; keep/random/proposed comparisons are correctly paired. Acceptance does not require the proposed method to win.  
**Branches:** large scores but wrong gradient prediction → implementation/optimizer mismatch; correct local descent but worse validation → plausible overfitting, retained as evidence; all models fail oracle-easy task → optimizer/proxy/harness diagnosis; saturated curves and no gain → informative negative, not automatic escalation to deep trees.

### WP05 — Sensitivity pilot and locked design

**Objective / risk:** avoid underpowered confirmation, favorable-coordinate selection, and arbitrary resource allocation.  
**Dependencies:** WP04 functioning mechanism and WP03 baseline anchors.  
**Actions:** run exploratory variance decomposition, sample-density/capacity curves, rare/smooth/no-regime controls, optimizer/contrast-scale sensitivity, and timing calibration. Compare the moment score with B6's actual validation-probed ranking. Calculate fixed confirmatory sample/repetition counts using Section 7.5, plus measured total workload. Choose and document numeric stopping rules, tuning budgets, primary coordinate, and hardware-safe batching; freeze these with the protocol hash.  
**Files / command:** `pilot/design_decision.md`, `pilot/resource_forecast.json`, `protocol/frozen.yaml`; `python -m treeneuron.cli calibrate --config configs/pilot.yaml`; then `python -m treeneuron.cli freeze --design pilot/design_decision.md`.  
**Outputs / resources:** effect/variance estimates clearly labeled exploratory; numeric design and attainable resolution; fixed candidate/default lists; complete forecast and baseline-exclusion rationale.  
**Acceptance:** another agent can execute confirmation without choosing the hypothesis, metric, exclusions, counts, or budget after viewing T.  
**Branches:** informative no-advantage pilot → an independent negative/estimation confirmation is still allowed; unstable pilot → increase independent problems or simplify the claim; infeasible precision → report achievable resolution and an inconclusive design instead of claiming sensitivity; method change → amend before generating fresh confirmation teachers.

### WP06 — Independent component confirmation

**Objective / hypothesis:** H1, with capacity/initialization attribution.  
**Dependencies:** frozen WP05 protocol; new teacher/data seeds; all relevant correctness tests.  
**Actions:** execute the full fixed paired cohort, including unfavorable and failed runs. Record the cause of each failure and apply the prespecified retry policy symmetrically. Compute paired teacher-level effects, intervals, guardrails, and capacity curves; keep any additional analysis marked exploratory.  
**Files / command:** `evaluation/confirm.py`, `analysis/primary.py`; `python -m treeneuron.cli confirm --protocol protocol/frozen.yaml --study synthetic`.  
**Outputs / resources:** immutable predictions, complete teacher-level table, uncertainty and effective sample counts, measured-versus-forecast cost. Serial checkpointed workload sized by WP05.  
**Acceptance:** completed cohort and audited statistics, not a positive p-value.  
**Branches:** H1 positive but B2/B3 equally good → placement result only, no practical architectural superiority; H1 negative with narrow interval → reject beneficial placement in this regime; wide interval → inconclusive at achieved precision; oracle/compositional evidence suggests unsuitable scale → register a targeted extension, not a post-hoc reinterpretation of H1.

### WP07 — Fixed-suite transfer and practical tradeoffs

**Objective / hypothesis:** H3 and boundary conditions; no universal state-of-the-art claim.  
**Dependencies:** functioning baseline adapters and frozen transfer protocol. Normally follows WP06; may still proceed after a negative H1 if the fixed/static family or a distinct prespecified transfer question remains informative.  
**Actions:** run nested grouped/temporal comparisons, official defaults and calibrated tuning, local foundation-model variants that pass their gates, and V-selected competitor champions. Evaluate protected T once per locked pipeline. Run timing on selected models with cold/warm and cache boundaries.  
**Files / command:** `configs/transfer.yaml`, `evaluation/timing.py`; `python -m treeneuron.cli confirm --protocol protocol/frozen.yaml --study transfer`.  
**Outputs / resources:** per-dataset/per-group error tables, observational-suite summary, separate simulation results, complete cost/Pareto tables, exclusions and license/version manifests. Use WP05 local forecasts, not paper accelerator throughput.  
**Acceptance:** comparisons include strong credible local alternatives and no leakage; apparent improvements remain interpretable after full costs.  
**Branches:** synthetic-only gain → narrow to a component result; fair baseline tuning erases it → no practical advantage established; tiny gain plus regression → report tradeoff; strongest relevant baseline unavailable → qualify scope, not a win by default.

### WP08 — Evidence-triggered extensions

**Objective / risks:** assess remaining causes and scale-dependent predictions without adding complexity blindly.  
**Dependencies:** a recorded WP06/WP07 result and a concrete unresolved hypothesis.  
**Actions:** choose only the relevant branch from Section 10: recursive depth-2 routing for compositional targets; repeated growth with independent validation protection; pruning; learned temperature; sparse/axis gates; hard execution; or insertion into a stronger MLP recipe. Specify new equations, interfaces, baselines, and fresh confirmation before testing the extension.  
**Files / command:** `protocol/extensions/<id>.yaml`, extension-specific modules/tests; `python -m treeneuron.cli run-extension --protocol protocol/extensions/<id>.yaml`.  
**Outputs / resources:** discriminating measurements and an extension-specific resource forecast. No automatic custom Metal kernel, large dataset download, or deeper tree search.  
**Acceptance:** extension resolves the stated uncertainty and passes the same fairness/leakage/accounting checks.  
**Branches:** absent prediction/headroom → do not pursue that branch; resource limitation → retain locally verified component claim and identify unverified scaling; failed explanation → revise or stop rather than invent another favorable metric.

### WP09 — Reproduction, audit, and decision report

**Objective / risk:** ensure the result survives a clean rerun and can be understood without chat memory.  
**Dependencies:** any completed confirmatory or diagnostic study, including negative outcomes.  
**Actions:** rerun selected checks and at least one representative paired pipeline from a clean environment; test interrupted growth/resume; rebuild reports from raw records; independently recompute primary metrics from saved predictions. Revisit closest prior art at the final claim boundary, not merely the project start.  
**Files / command:** `reports/final.md`, `audit/`, `DECISIONS.md`; `python -m treeneuron.cli audit`; `python -m treeneuron.cli report`.  
**Outputs / resources:** a reproducible result package, licenses/provenance, complete favorable/unfavorable results, exact claim wording, resource measurements, unresolved limitations, and continue/revise/narrow/stop decision.  
**Acceptance:** reports regenerate without manual score editing; raw records account for all runs; resumption and source revisions agree.  
**Branches:** discrepancy → block scientific conclusions until explained; valid negative result → release the diagnostic finding; useful but nonnovel result → report as replication/engineering; useful differentiated result → state the tested scope and unresolved novelty limits without claiming independent review that did not occur.

## 10. Attribution, robustness, and failure analysis

### 10.1 Minimum discriminating ablations

| Test | Confound or boundary isolated | Required interpretation |
|---|---|---|
| Shared-tensor tree versus GLU expression | Naming / computational implementation | Equal values and gradients are required; speed differences are engineering only. |
| Top-m versus random-m, same contrasts | Placement of acquired conditional behavior | Primary H1 contrast. |
| Same IDs, eigenvector versus random contrast | Direction initialization | A gain here does not establish a better placement rule. |
| Adam-aware versus SGD-gradient ranking | Optimizer/score mismatch | Differences are allocation-policy evidence, not model-class evidence. |
| Continued parent | Additional updates / base warm-start | Growth must not claim a gain explained by simply training longer. |
| Static partial/full gated model | Existing gated family and symmetry | The same final unit counts are not a novel topology. |
| Ordinary parameter-bracketing and ordinary growth | Additional capacity / function-preserving optimization | Count every head/hidden parameter; compare cost curves when exact matching is impossible. |
| Validation-probed B6 | Cheap scoring versus actual local trial selection | A cheaper but weaker score may offer a tradeoff, not dominance. |
| No-regime, smooth, noise, shuffled-label controls | Favorable teacher design / fitting noise | No universal expected win. Excessive growth on noise is a failure mode. |
| Hard and smooth teachers, rare regimes, nuisance dimensions, rotations | Gate family, sample information, and input alignment | Characterize where effects disappear; do not drop losing conditions. |
| Contrast magnitude, fixed/learned temperature, regularization | Scaling and saturation | Report gate derivatives, score stability, and training sensitivity. |
| Post-mixture versus leaf-wise ReLU; identity activation | Silent architectural substitution | Distinct models, with distinct claims. |
| Float64 CPU reference, float32 CPU/MPS, repeated resume | Numerical/backend artifact | Correctness first; benchmark compatibility is not inferred from math. |
| Same-model sham and known underfit control | False-positive/insensitive harness | Sham produces zero paired differences when predictions are identical; known difference must be detectable. |

The full cross-product is not mandatory. Run the primary paired comparison and mandatory controls first; use targeted follow-up experiments to resolve an identified cause. Any post-confirmation discovery must be marked exploratory until independently confirmed.

### 10.2 Gate and optimization diagnostics

Log gate mean, variance, quantiles, fraction near 0/1, `E[g(1-g)]`, expert disagreement, input norms, gradient norms, per-unit score/spectral gap, and occupancy of known synthetic regions. Measure score–observed-improvement correlation using independent probe data. Occupancy is a property of a soft gate, not a hard count of executed experts.

Check whether a single extreme observation dominates `M_j`; compare leave-one-group-out or robustly clipped-score diagnostics as separate variants. Do not silently clip primary gradients or change the loss after observing outliers. Check whether selected units are merely those with largest outgoing weights; compare a cheap outgoing-norm/gradient-norm ranking as an alternative explanation. Learned hidden-unit identities are permutation- and scale-dependent, so interpretability requires stability and faithfulness tests beyond plotting a tree.

### 10.3 Evidence-triggered continuation

- **H1 survives, but training-moment computation dominates cost:** test a streamed or low-rank approximation, with score/ranking parity and a same-information comparison. This is an engineering/estimation refinement, not a new tree family.
- **Oracle gating helps but learned gating does not:** test optimization and initialization, not more datasets. Distinguish missing routing information from lack of representational capacity.
- **Stumps plateau specifically on a nested-regime teacher while ordinary and oracle controls expose available signal:** test depth 2 with matched capacity and new teacher draws. This preserves a scale/composition hypothesis that the smallest test cannot express.
- **Most first-event branches are useful and occupancy remains healthy:** repeated growth may be justified. Predeclare a total capacity path, charge each search round, and use nested validation; do not repeatedly test significance on one G set.
- **Branches become redundant:** prune by replacing a subtree with its best F-fitted affine surrogate, selecting the decision on G and measuring prediction discrepancy. Fresh post-pruning fitting and all selection cost are counted.
- **Soft gates are decisive and disagreement-weighted hardening error is small:** prototype hard CPU execution and compare actual batch-1/batched latency. Only then consider vectorized dispatch or custom kernels.
- **A small compact-network advantage does not survive RealMLP/TabM:** report the boundary. A subsequent insertion into the strong recipe must share generic preprocessing/ensembling with its baseline and receive new confirmation.

### 10.4 Failure classification

Use one of these explicit statuses, with evidence:

`implementation_invalid`: algebra, gradients, data partitioning, or baseline behavior is wrong. No scientific conclusion.

`measurement_invalid`: asynchronous timing, resource interference, leakage, or inference code changes invalidate the measurement. Rerun only after correction, with an audit trail.

`proxy_unsuitable`: the test cannot express the claimed mechanism, established by oracle/compositional controls. This warrants a targeted new proxy, not a universal rejection or unconditional escalation.

`resource_limited`: a faithful operation/model cannot be run locally at the required scale. Report what component was tested and what remains unknown.

`inconclusive`: uncertainty spans materially different outcomes. Report achieved resolution; do not describe it as equivalence.

`mechanism_without_advantage`: the unit learns the behavior, but existing methods match or exceed it under fair comparison.

`evidence_against_hypothesis`: the appropriate independent experiment has sufficient sensitivity and consistently contradicts the predicted benefit in its declared regime.

`qualified_improvement`: the claimed metric and attribution tests support a positive effect, with costs, regressions, novelty limits, and population stated.

## 11. Repository, interfaces, and experiment records

### 11.1 Minimal layout

```text
tree-neuron-poc/
  README.md
  pyproject.toml
  locks/                         # core and baseline-specific resolved environments
  environment/                   # hardware, OS, wheels, backend tests
  protocol/
    draft.yaml
    frozen.yaml
    prior_art_delta.md
    amendments/
    extensions/
  configs/
    data.yaml
    explore.yaml
    pilot.yaml
    transfer.yaml
    vendor/                      # original configs + local adapter mappings
  src/treeneuron/
    cli.py
    doctor.py
    data/                        # raw parsers, teachers, transforms, manifests
    models/units.py
    growth/score.py
    growth/insert.py
    baselines/
    train.py
    checkpoint.py
    evaluation/                  # splits, metrics, confirmation, timing
    analysis/
  tests/
  provenance/                    # URLs, licenses, source/code/checkpoint hashes
  data/raw/                      # immutable, never hand-edited
  data/derived/                  # regenerable, transformation hashes
  pilot/
  runs/exploratory/
  runs/confirmatory/
  reports/
  audit/
  DECISIONS.md
```

Avoid storing downloadable third-party weights in Git. Model artifacts and predictions use content hashes and manifests; raw data redistribution follows the archived licenses.

### 11.2 Shared adapter contract

This is a **project interface**, not a claim that every vendor already implements these signatures:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import numpy as np

@dataclass(frozen=True)
class FitRequest:
    X_fit: Any                  # numeric array or documented mixed-type frame
    y_fit: np.ndarray           # shape [N], finite
    X_select: Any               # G only; never T
    y_select: np.ndarray
    seed: int
    config: dict[str, Any]
    budget: dict[str, Any]      # frozen seconds/updates; includes timing boundary
    artifact_dir: Path

class Predictor(Protocol):
    def fit(self, request: FitRequest) -> "Predictor": ...
    def predict(self, X: Any, *, batch_size: int | None = None) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    def resource_summary(self) -> dict[str, Any]: ...
```

`predict` must return finite `[N]` predictions in original target units, not standardized labels or a hidden ensemble axis. The HPO runner alone evaluates V. The protected evaluator alone reads T labels after the freeze check. Adapters expose whether they use sample weights, native categories, internal validation, preprocessing, retained training context, transductive test-feature processing, caching, and backend fallback. Unsupported features cause a recorded error, not silent coercion.

For output ensembles, specify whether aggregation occurs before or after a nonlinear transformation. Regression TabM averages member predictions at inference while training individual member losses [R20]. Save enough metadata to rebuild every fitted preprocessing step.

### 11.3 Configuration and records

Use human-readable YAML for experiment configs and a canonical JSON representation for hashing. Unknown config keys are errors. Every value influencing results is explicit or captured as a resolved vendor default. Do not rely on mutable library defaults across reruns.

Each run writes `config.resolved.json`, `manifest.json`, `events.jsonl`, `metrics.json`, checkpoints, and predictions. The manifest schema includes:

```text
run_id, parent_run_id, study_id, phase, status, failure_class
protocol_hash, code_commit, dependency_lock_hash, adapter_revision
method_id, baseline_role, resolved_hyperparameters, growth_manifest_hash
hardware_report_hash, backend, dtype, cpu_threads, fallback_mode
dataset_id, raw_sha256, split_manifest_hash, transform_hash
teacher_seed, data_seed, optimizer_seed, allocation_seed, hpo_seed
fit_rows, selection_rows, hpo_rows, test_rows, group_counts
parameter_count_trainable, parameter_count_total, serialized_model_bytes
pretrained_checkpoint_id, checkpoint_sha256, weights_license_id
retained_context_bytes, cache_bytes, acquisition_cost_separate
search_ns, preprocessing_ns, base_fit_ns, scoring_ns, probe_ns,
continuation_ns, validation_ns, first_predict_ns, warm_predict_ns
process_rss_bytes, mps_allocated_bytes, mps_driver_bytes,
system_memory_pressure, swap_delta_bytes
started_at_utc, ended_at_utc, interruption_history, timing_exclusions
```

All times carry measured/estimated status and boundary definitions. Counts or memory metrics that a vendor cannot expose are `null` with an explanation, not zero. Failed runs keep their configuration, partial logs, resource state, exception, and retry linkage.

Predictions include `run_id`, outer fold, immutable row ID, group/date block ID, actual target, predicted target, and split. Synthetic records additionally retain the noise-free target for diagnostics. Store prediction files only after authorized test access; their presence is logged. Source metadata and target-derived columns must not accidentally enter the feature arrays through these record schemas.

### 11.4 Agent authority and reporting

The coding agent may choose routine file organization details, safe batch scheduling, dependency-isolation mechanics, and equivalent numerical implementations after parity testing. It may not silently change the primary hypothesis, population, target, test split, metric, baseline family, information access, precision, or fairness rule. Such changes require an amendment and possibly fresh confirmation.

Update `DECISIONS.md` after every work package with inputs, evidence, decision, unresolved blockers, and next command. Resume from these files, not chat memory. Report all prespecified outcomes, per-task regressions, failures, excluded baselines, and unsuccessful extensions. A clean final table must never be produced by deleting unfavorable runs.

## 12. Audit record, remaining risks, and final interpretation rules

### 12.1 Scientific / prior-art audit

This was an author-side challenge of the plan, not independent peer review.

| Material weakness found | Revision made | Residual risk / acceptance criterion |
|---|---|---|
| A tree-neuron could be presented as a new primitive despite exact gated equivalence. | Added the affine-bypass derivation, scalar-head collapse, shared-coordinate test, and static GLU baseline. | No representational novelty claim for the stump; optimization differences must be attributed. |
| “Selected topology” overstates what choosing unit IDs in one symmetric layer changes. | Reframed H1 around inherited learned units/trajectory; removed unnecessary discovery cost from static baselines. | A genuinely different topology requires layer/depth allocation evidence. |
| Adaptive validation-based splitting and derivative-informed growth already exist. | Included ANT, budding/distributed trees, splitting steepest descent, flat-channel theory, local model trees, and related patent text. | Exact score novelty remains provisional, especially while R12 is inaccessible. |
| A neural-versus-constant-tree comparison misses a simpler affine-expert alternative. | Added LightGBM linear-leaf and constant-leaf controls with current official compatibility/configuration details. | Their configured leaves differ from unrestricted affine experts; report the distinction. |
| A local descent score could be confused with an optimal split or a test-risk theorem. | Derived the gate gradient, made Adam ranking explicit, and required random/direction/validation-probe controls. | Generalization and final optimization behavior remain empirical questions. |
| Biological analogies or visible tree structure could imply interpretability. | Kept dendritic work as adjacent prior art and required stability/faithfulness before interpretability claims. | No biological or semantic interpretation follows from a scalar tree output alone. |

### 12.2 Experimental-validity audit

| Material weakness found | Revision made | Residual risk / acceptance criterion |
|---|---|---|
| Repeated G-based growth selection could masquerade as repeated validated discoveries. | One-event core, nested F/G/V/T roles, freeze protection, and no per-split significance language. | Adaptive overfitting remains part of the algorithm's outer-test performance. |
| More parameters, warm-start, or training time could explain a positive result. | Added ordinary widening/growth, static partial/full gates, continued parent, matched forks, and complete resource accounting. | No broad advantage unless the relevant controls are surpassed. |
| Current baselines could be omitted or weakened. | Added RealMLP, TabM with correct member loss, current TabICLv2/TabPFN-3.5, and vendor default anchors. | Locally unavailable strong baselines limit practical claims. |
| Dataset leakage and pseudo-replication could manufacture improvement. | Subject/configuration groups, forward Bike windows, target-derived column bans, CCPP shuffled-copy check, simulation separation. | Grouping proxies and few Energy geometries limit population claims. |
| Seeds, folds, or timings might be treated as independent data. | Independent synthetic teachers; nested seed averaging; conditional grouped/blocked transfer intervals; separate training and timing variability. | The fixed five-dataset observational aggregate is not universal inference. |
| Pilot-driven sample sizing could become optional stopping. | Fixed independent cohort after a written numeric design; retain negative signs and full cohorts. | Bootstrap/power approximations and small pilot effects can still yield an inconclusive study. |
| An impressive aggregate could hide a rare-region or latency regression. | Group/regime error, physical-unit effects, and explicit Pareto/guardrail reporting. | Practical value cannot be certified without application tolerances. |

### 12.3 Implementation / feasibility audit

| Material weakness found | Revision made | Residual risk / acceptance criterion |
|---|---|---|
| Nominal MPS support could hide unsupported float64/custom kernels. | CPU float64 math/scoring, explicit kernel tests, no implicit fallback, whole-method CPU path. | Actual M3 operation and training paths are untested until WP00–WP03. |
| Legacy ANT requirements and changing current packages could conflict. | Separate pinned environments and restricted, explicitly named ports. | Adapter correctness must be demonstrated; no fictional universal lock is provided. |
| TabPFN CPU row limits could invalidate a fallback on larger folds. | Added the inspected 5,000-row allowance and Apple PyTorch recommendation; prohibited silent context reduction. | Full-F MPS support or a qualified missing baseline is required. |
| Dormant trees, forks, and unified-memory double counting could distort efficiency. | Compact active tensors, serial forks, explicit state/activation/context models, separate overlapping memory counters. | Resource forecasts need local measurements. |
| A zero–zero insertion would never learn; a batch-mean gradient could shrink scores by an extra `n`. | Nonzero hidden contrasts; per-example/summed-loss gradient convention; executed gradient checks. | Score usefulness is not established by these checks. |
| Asynchronous timing and free cached context could create false speedups. | Synchronization, cold/warm/full-pipeline boundaries, cache construction and retained-data accounting. | Thermal and operating-system variability require measured intervals. |
| Growth could corrupt optimizer or resume state. | Stable-name migration, atomic checkpoints, paired interruption/resume tests, explicit restart on changed logical batches. | This project's checkpoint implementation has not yet been built or validated. |

A subsequent dependency check incorporated the linear-leaf baseline, clarified the primary ordinary-growth initialization, made the pilot-to-confirmation rule numeric, and checked the normalized score's invariances in a small CPU experiment. No further material correction was identified in the final document consistency pass; this is not a claim of perfection or independent review.

### 12.4 Allowed conclusion templates

Use wording supported by the completed experiment, substituting actual numbers and scope only after execution.

**Component improvement:** “On the registered distribution of independently generated regime-dependent regression problems, score-guided one-event specialization reduced normalized test MSE relative to matched random placement by [effect and interval], with [measured cost]. This is a capacity-allocation/training result within an established gated model family.”

**Practical improvement:** add the actual observational datasets, full-F locally run baseline set, V-selected competitor, grouped/temporal protocol, and measured resource tradeoff. Do not claim to beat excluded or API-only models.

**Negative result:** “The tested growth procedure implemented and learned the intended conditional behavior, but did not improve over [matched controls] at the resolution [interval/MDE].” Distinguish a narrow negative effect from a wide inconclusive interval.

**Novelty unresolved:** retain the R12 access limitation and any further overlap. A new benchmark result, an engineering optimization, and a new learning algorithm are different contributions.

**Scaling uncertainty:** identify which locally observed trend or oracle/compositional result motivates further scale. Do not assert that larger scale must rescue a failed mechanism, and do not reject an untested scale-dependent mechanism merely because a toy cannot express it.

The correct final decision can be continue, revise, narrow, replicate, or stop. The decision must refer to the recorded hypothesis and evidence, not the desired story.

## 13. Search coverage and references

### 13.1 Scope and access record

Research was conducted on 16 October 2026 using current public primary papers, official proceedings, author repositories, package documentation, UCI dataset records, and published patent text. Search concepts included: affine-leaf soft trees; hierarchical mixtures of experts; GLU/highway equivalence; budding and distributed trees; adaptive neural trees; neuron splitting and splitting matrices; loss-landscape gated limits; local linear model trees; dendritic computational units; differentiable/hard tree ensembles; current tabular baselines; Apple/MPS support; and validation-selection bias.

Refinement searches included combinations of “function-preserving,” “neuron splitting,” “gated linear,” “residual,” “growth,” “tree neuron,” and the exact 2026 Arboreal title/ArborCell term. Backward tracing connected adaptive neural trees to earlier soft/budding trees and mixtures of experts; forward/adjacent tracing found splitting/flat-channel work and the 2026 close hit. A final audit search added a directly executable non-neural linear-leaf tree baseline rather than relying only on the historical LOLIMOT analogy.

The material mechanism families stabilized over the later targeted searches. This is practical coverage, **not proof that no unindexed, differently named, non-English, unpublished, or proprietary prior art exists**. Patent searching was a technical overlap screen, not a comprehensive legal search. Large-scale sequence/vision experiments, all possible tabular predictors, and production deployments were not exhaustively surveyed because they are not necessary to identify the initial component mechanism.

Important access limits: the 2026 *Arboreal Neural Network* was confirmed in an official listing, but several official OpenReview forum/PDF paths, including a static hash PDF URL, returned browser verification rather than paper text. A separate downward-growing-neural-network lead had limited text access and is not used to support a detailed technical claim. Secondary search summaries were used to locate material, not to substitute for inspecting a consequential algorithm. Public documentation/code inspection does not mean the packages were executed.

### 13.2 Reference and evidence ledger

All live documentation/repository links below were inspected on the research date; they should be snapshotted and version-pinned when executing the plan. “Inspected” specifies the scope, not a claim that every supplement and experiment was reproduced. The `[Rxx]` labels in the document refer to this ledger.

**[R01] Michael I. Jordan and Robert A. Jacobs — “Hierarchical Mixtures of Experts and the EM Algorithm.”** Historical conference version, 1993; journal treatment in *Neural Computation* 6(2), 1994. [Public full-text version](https://www.cs.toronto.edu/~hinton/absps/hme.pdf); [MIT record](https://dspace.mit.edu/entities/publication/61ef010b-392d-4e7d-b6ad-9797a4d3b2f8). Inspected the public PDF's model description/equations, including rendered pages. Establishes soft hierarchical gating and affine/generalized-linear expertise; historical experiment settings do not establish current competitiveness.

**[R02] Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier — “Language Modeling with Gated Convolutional Networks.”** ICML 2017. [Paper](https://proceedings.mlr.press/v70/dauphin17a/dauphin17a.pdf). Inspected full-text formulation and rendered gate equation. Supplies the gated-linear primitive; the affine-stump reduction in this plan is an explicit deduction, not a claim that this paper studied these tabular growth experiments.

**[R03] Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber — “Highway Networks.”** 2015, arXiv:1505.00387. [Paper](https://arxiv.org/abs/1505.00387); [HTML version inspected](https://arxiv.org/html/1505.00387v2). Inspected gated carry/transform formulation. Related gating precedent; not an identical empirical setup.

**[R04] Ozan İrsoy and Ethem Alpaydın — “Distributed Decision Trees.”** 2014, arXiv:1412.6388. [Full text](https://arxiv.org/html/1412.6388v1). Inspected soft/distributed recurrences and the discussion/references for budding trees. Supports structural-growth and routing prior art; original budding-tree experiments were not separately reproduced.

**[R05] Riki Tanno, Kai Arulkumaran, Daniel C. Alexander, Antonio Criminisi, and Aditya V. Nori — “Adaptive Neural Trees.”** ICML 2019. [Proceedings record](https://proceedings.mlr.press/v97/tanno19a.html); [Paper](https://proceedings.mlr.press/v97/tanno19a/tanno19a.pdf). Inspected architecture and validation-driven structural optimization, including rendered figures. Closest growth precedent; broader transformations and tasks differ from the restricted baseline specified here.

**[R06] Tanno and contributors — AdaptiveNeuralTrees official implementation.** [Repository](https://github.com/rtanno21609/AdaptiveNeuralTrees). Inspected README requirements and license. Its Python 2.7-era stack is a porting constraint, not current MPS validation. Used for reuse/port planning, not a claimed executed reproduction.

**[R07] Lemeng Wu, Dilin Wang, and Qiang Liu — “Splitting Steepest Descent for Growing Neural Architectures.”** NeurIPS 2019, arXiv:1910.02366. [Paper](https://arxiv.org/abs/1910.02366); [Full-text route inspected](https://arxiv.org/html/1910.02366). Inspected loss-derived splitting criterion and matrix interpretation. Constrains derivative-based growth novelty; does not prove the proposed score generalizes.

**[R08] Flavio Martinelli, Alexander van Meegen, Berfin Şimşek, Wulfram Gerstner, and Johanni Brea — “Flat Channels to Infinity in Neural Loss Landscapes.”** 2025, arXiv:2506.14951v2. [Full text](https://arxiv.org/html/2506.14951v2). Inspected neuron-duplication and gated-limit analysis. Important parameterization/splitting overlap; no direct validation of the present allocation algorithm.

**[R09] University of Siegen, Measurement and Control Engineering — Local Model Network Toolbox / LOLIMOT documentation.** Live official documentation. [Toolbox page](https://www.mb.uni-siegen.de/mrt/lmn-tool/?lang=de). Inspected local-model/growth description and toolbox conditions. Provides a system-identification precedent; the MATLAB/noncommercial toolbox is not a required local dependency.

**[R10] Han and coauthors — “Parameter efficient dendritic-tree neurons outperform perceptrons.”** 2022, arXiv:2207.00708; ICML 2022 workshop contribution. [Full text](https://arxiv.org/html/2207.00708v1); [Workshop paper](https://dynn-icml2022.github.io/papers/paper_13.pdf). Inspected architecture and reported study scope. Adjacent scalar-unit complexity, not identical soft decision routing; do not cite it as a main-conference universal superiority result.

**[R11] Varun Ojha and Giuseppe Nicosia — “Backpropagation Neural Tree.”** *Neural Networks* 149, 66–83, 2022; arXiv:2202.02248. [Full text](https://arxiv.org/html/2202.02248); [DOI](https://doi.org/10.1016/j.neunet.2022.02.003). Inspected computational-tree formulation. Distinguishes tree-shaped neural computation from decision-tree routing.

**[R12] Wubin Yan, Wei Ma, Shixiang Wan, Dongchen Li, Shaoshun Kang, Qing Yang, and Dongliang Xu — “Arboreal Neural Network.”** ICML 2026 listing. [Official conference listing](https://icml.cc/Downloads/2026); [OpenReview record](https://openreview.net/forum?id=NDM1ltvBOf); [Blocked PDF route](https://openreview.net/pdf/5800594ecbf7f51db4b3b98499665c0100a30ba4.pdf). Inspected listing/metadata; full text was inaccessible due browser verification. Exact mechanism/evidence is unresolved. This is a novelty blocker, not a source for invented technical details.

**[R13] Peter Kontschieder, Madalina Fiterau, Antonio Criminisi, and Samuel Rota Bulò — “Deep Neural Decision Forests.”** ICCV 2015. [Official paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf). Inspected stochastic differentiable-tree formulation and rendered first page. Establishes neural/tree integration; classification architecture and published large-scale results are not this plan's regression evaluation.

**[R14] Sergei Popov, Stanislav Morozov, and Artem Babenko — “Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data.”** 2019 preprint / ICLR 2020, arXiv:1909.06312. [Paper](https://arxiv.org/abs/1909.06312). Inspected paper record and model context. Establishes the relevant NODE baseline family; implementation-specific conclusions below come from R15.

**[R15] NODE authors — official implementation and `lib/odst.py`.** [Repository](https://github.com/Qwicen/node); [Source module inspected](https://raw.githubusercontent.com/Qwicen/node/master/lib/odst.py). Inspected feature-selection/binning defaults, response/path tensors, and train-data-aware threshold/temperature initialization. Preserve configured functions and initialization semantics; source inspection is not an executed MPS test.

**[R16] Sascha Marton and coauthors — “GRANDE: Gradient-Based Decision Tree Ensembles.”** ICLR 2024. [Official implementation](https://github.com/s-marton/GRANDE); [Paper](https://arxiv.org/abs/2309.17130). Inspected repository/model description. Relevant hard-tree ensemble alternative; no local package execution or hard-routing speed claim was verified.

**[R17] Ziming Liu and coauthors — “KAN: Kolmogorov–Arnold Networks.”** 2024, arXiv:2404.19756. [Paper record](https://arxiv.org/abs/2404.19756). Inspected abstract/model positioning. Supports the distinction between learnable univariate edge functions and multivariable tree routing, not a mandatory competitor in the core allocation experiment.

**[R18] Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux — “Why do tree-based models still outperform deep learning on tabular data?”** NeurIPS 2022, arXiv:2207.08815. [Paper record](https://arxiv.org/abs/2207.08815). Inspected benchmark/inductive-bias summary. Historical evidence for strong tree baselines, not a timeless ranking that overrides later studies.

**[R19] David Holzmüller, Léo Grinsztajn, and Ingo Steinwart — “Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data.”** NeurIPS 2024, arXiv:2407.04491. [Full text inspected](https://arxiv.org/html/2407.04491v3); [Official pytabkit implementation](https://github.com/dholzmueller/pytabkit). Inspected strong default recipes/preprocessing and repository guidance. Motivates a faithful RealMLP baseline; no target-laptop fit was run.

**[R20] Yury Gorishniy and coauthors — “TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling.”** ICLR 2025, arXiv:2410.24210. [Paper](https://arxiv.org/abs/2410.24210); [Official implementation/documentation](https://github.com/yandex-research/tabm). Inspected model/training description and current README's loss semantics, numerical embeddings, defaults/search supports, and license. Its reported accelerator results are not imported into this study.

**[R21] Nick Erickson and coauthors — “TabArena: A Living Benchmark for Machine Learning on Tabular Data.”** 2025, arXiv:2506.16791. [Full text inspected](https://arxiv.org/html/2506.16791v4). Inspected benchmark/tuning/ensembling context. Supports contemporary comparison scope and cautions about rankings; the full benchmark is not reproduced by six small datasets.

**[R22] Prior Labs — TabPFN-3.5 model report.** September 2026, arXiv:2609.17895. [Report](https://arxiv.org/html/2609.17895v1). Inspected model-family and evaluation context. Distinguishes local base weights from broader/API/enterprise variants; published scores and GPU throughput are reference-only.

**[R23] Prior Labs — TabPFN implementation and TabPFN-3.5 model card.** Live documentation. [Repository](https://github.com/PriorLabs/tabpfn); [Model card and files](https://huggingface.co/Prior-Labs/tabpfn_3_5). Inspected default model, separate code/weight licenses, Apple guidance, CPU allowance, and memory guidance. Local execution and authorization for a specific use remain WP00/WP07 gates.

**[R24] Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan — “TabICLv2: A better, faster, scalable, and open tabular foundation model.”** ICML 2026, arXiv:2602.11139, revised September 2026. [Paper](https://arxiv.org/abs/2602.11139); [Official repository](https://github.com/soda-inria/tabicl); [Code license inspected](https://raw.githubusercontent.com/soda-inria/tabicl/main/LICENSE). Inspected current paper/record, regression checkpoint, default ensemble/device/caching interfaces and code license. Verify the downloaded weight terms separately; repository support is not a measured M3 performance guarantee.

**[R25] CatBoost developers — official project and release distribution.** Release 1.2.10, February 2026, as inspected. [Project](https://github.com/catboost/catboost); [PyPI release files](https://pypi.org/project/catboost/). Inspected CPU support and macOS universal2 wheel listings. These establish an installation path, not an executed native-arm64 regression benchmark.

**[R26] XGBoost developers — Installation Guide.** Live stable documentation. [Official guide](https://xgboost.readthedocs.io/en/stable/install.html). Inspected platform/package support. Use a resolved native-compatible stable version and CPU histogram implementation; local binary/runtime verification remains required.

**[R27] UCI Machine Learning Repository — Airfoil Self-Noise, record 291.** [Dataset and metadata](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise); [DOI](https://doi.org/10.24432/C5VW2C). Inspected dimensions, variables, provenance and license. The proposed configuration grouping is this plan's conservative evaluation choice, not an official benchmark split.

**[R28] UCI Machine Learning Repository — Concrete Compressive Strength, record 165.** [Dataset and metadata](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength). Inspected dimensions, composition/age variables, target and license. Grouping by mixture protects the stated unseen-mixture evaluation; raw group counts remain to be checked.

**[R29] UCI Machine Learning Repository — Energy Efficiency, record 242.** [Dataset and metadata](https://archive.ics.uci.edu/dataset/242/energy+efficiency). Inspected simulated-building provenance, two targets, dimensions and license. Heating-only prediction and geometry grouping are specified project choices; this is not field evidence.

**[R30] Pınar Tüfekci, Heysem Kaya / UCI — Combined Cycle Power Plant, record 294.** 2014 dataset. [Dataset and metadata](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant); [DOI](https://doi.org/10.24432/C5002N). Inspected variables, full-load collection context, five shufflings, files, dimensions and license. Crucial duplicate-copy and non-temporal-evaluation guardrails.

**[R31] UCI Machine Learning Repository — Parkinsons Telemonitoring, record 189.** [Dataset and metadata](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring); [DOI](https://doi.org/10.24432/C5ZS3N). Inspected repeated-subject structure, voice features, target/interpolation details, dimensions and license. Supports patient-group splitting; no clinical effectiveness claim follows.

**[R32] Gavin C. Cawley and Nicola L. C. Talbot — “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.”** *JMLR* 11, 2079–2107, 2010. [Official record and paper link](https://www.jmlr.org/papers/v11/cawley10a.html). Inspected official abstract and bibliographic record. Used for the selection-bias rationale, not an imported numerical estimate of this project's bias.

**[R33] Xavier Bouthillier and coauthors — “Accounting for Variance in Machine Learning Benchmarks.”** *MLSys* 3, 2021. [Official proceedings record](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html). Inspected official abstract/record. Motivates data/initialization/HPO variance separation; its published numerical efficiency claims are not used as predictions here.

**[R34] UCI Machine Learning Repository — Bike Sharing, record 275.** [Dataset and metadata](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). Inspected hourly/daily tables, columns, calendar context, counts and license. Supports target-component removal and chronological evaluation; retrospective observed-weather information is distinguished from forecasting.

**[R35] Apple Developer — Accelerated PyTorch training on Mac.** Live documentation. [Official page](https://developer.apple.com/metal/pytorch/). Inspected platform and installation requirements. Release-specific guidance differs from other current documentation; not proof of a particular kernel path.

**[R36] PyTorch contributors — MPS backend notes.** Live documentation inspected October 2026. [Official notes](https://docs.pytorch.org/docs/main/notes/mps.html). Inspected backend checks and OS guidance. Drives actual-machine discovery and version pinning, not an assumption that every operation or baseline is supported.

**[R37] PyTorch contributors — MPS memory, synchronization, and environment APIs.** [Environment variables](https://docs.pytorch.org/docs/main/mps_environment_variables.html); [synchronize](https://docs.pytorch.org/docs/main/generated/torch.mps.synchronize.html); [recommended working-set memory](https://docs.pytorch.org/docs/main/generated/torch.mps.recommended_max_memory.html); [allocated memory](https://docs.pytorch.org/docs/main/generated/torch.mps.current_allocated_memory.html). Inspected official API documentation. Supports explicit timing/memory boundaries; all local resource numbers remain to be measured.

**[R38] Microsoft / Aditya Nori, Antonio Criminisi, and Riki Tanno — “Neural trees,” US20200005148A1.** Published 2 January 2020; priority record 29 June 2018. [Publication text](https://patents.google.com/patent/US20200005148A1/en). Inspected description and technical overlap with differentiable/growing neural trees. No legal status, enforceability, patentability, or freedom-to-operate conclusion is drawn.

**[R39] LightGBM developers — Parameters and Installation Guide.** Stable documentation displayed version 4.7.0 on the research date. [Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html); [Installation](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html). Inspected `linear_tree`, `linear_lambda`, leaf-feature restrictions, memory warning, and native macOS routes. Establishes a reusable simpler local-affine competitor; local installation and performance are untested.


## Appendix A. Executed algebra/autograd check script

Save the following block as `tests/reference_math_checks.py` and run it in a fresh Python process with PyTorch installed. It is the script executed during this research, not the future project CLI. It prints a JSON record and raises an assertion on failure. The measured environment and values are in Section 3.5. CPU float64 execution on a Mac is a new local check; the earlier Linux result does not certify it.

```python
import json, platform, sys
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(1602026)
N,d,h=37,7,5
x=torch.randn(N,d,requires_grad=True)
L=torch.randn(h,d+1,requires_grad=True)
R=torch.randn(h,d+1,requires_grad=True)
G=torch.randn(h,d+1,requires_grad=True)
X=torch.cat([x,torch.ones(N,1)],1)
g=torch.sigmoid(X@G.T)
l=X@L.T;r=X@R.T
tree=(1-g)*l+g*r
skip=l+g*(r-l)
probe=torch.randn_like(tree)
grad1=torch.autograd.grad((tree*probe).sum(),(x,L,R,G),retain_graph=True)
grad2=torch.autograd.grad((skip*probe).sum(),(x,L,R,G),retain_graph=True)
v=torch.randn(h)
c=torch.randn(())
collapsed=X@(v@L)+((g*(X@(v[:,None]*(R-L)).T)).sum(1))+c
raw=tree@v+c

# Exact function-preserving growth of one preactivation, and analytic gate gradient.
B=torch.randn(h,d+1)
head=torch.randn(h)
y=torch.randn(N)
base=X.detach()@B.T
pre=base.clone().requires_grad_()
old_pred=pre.relu()@head
old_loss=((old_pred-y)**2).mean()
# autograd of a SUM yields per-example (not 1/N scaled) delta.
delta=torch.autograd.grad(((old_pred-y)**2).sum(),pre)[0].detach()
j=2; X0=X.detach(); M=X0.T@(delta[:,j:j+1]*X0)/N
lam,Q=torch.linalg.eigh(M)
q=Q[:,lam.abs().argmax()]
eps=0.05
D=(eps*q).detach().requires_grad_(); A=torch.zeros(d+1,requires_grad=True);tau=1.0
correction=torch.tanh((X0@A)/(2*tau))*(X0@D)
z=base.clone();z[:,j]=z[:,j]+correction
new_pred=z.relu()@head
new_loss=((new_pred-y)**2).mean()
gA,gD=torch.autograd.grad(new_loss,(A,D),retain_graph=True)
analytic=M@D.detach()/(2*tau)
step=1e-4/(1+gA.norm().item())
with torch.no_grad():
    A1=A-step*gA
    z1=base.clone(); z1[:,j]+=torch.tanh(X0@A1/2)*(X0@D)
    after=((z1.relu()@head-y)**2).mean()

# Exact affine-leaf hard/soft gap, then contraction after ReLU.
hard=torch.where(g>=.5,r,l)
bound=torch.minimum(g,1-g)*(r-l).abs()
errors={
 'forward_equivalence_max_abs':(tree-skip).abs().max().item(),
 'gradient_equivalence_max_abs':max((a-b).abs().max().item() for a,b in zip(grad1,grad2)),
 'scalar_head_collapse_max_abs':(raw-collapsed).abs().max().item(),
 'growth_preservation_max_abs':(new_pred-old_pred).abs().max().item(),
 'gate_gradient_formula_max_abs':(gA-analytic).abs().max().item(),
 'initial_delta_gradient_norm':gD.norm().item(),
 'initial_gate_gradient_norm':gA.norm().item(),
 'local_loss_before':old_loss.item(),
 'local_loss_after_one_small_gate_step':after.item(),
 'soft_hard_gap_formula_max_abs':((tree-hard).abs()-bound).abs().max().item(),
 'relu_soft_hard_bound_max_violation':((tree.relu()-hard.relu()).abs()-bound).max().item(),
}
assert errors['forward_equivalence_max_abs']<1e-12
assert errors['gradient_equivalence_max_abs']<1e-12
assert errors['scalar_head_collapse_max_abs']<1e-12
assert errors['growth_preservation_max_abs']==0
assert errors['gate_gradient_formula_max_abs']<1e-12
assert errors['initial_delta_gradient_norm']==0
assert after<old_loss
assert errors['soft_hard_gap_formula_max_abs']<1e-12
assert errors['relu_soft_hard_bound_max_violation']<1e-12
out={'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'device':'cpu','dtype':'float64','threads':torch.get_num_threads(),'seed':1602026},'shape':{'N':N,'d':d,'h':h},'results':errors}
print(json.dumps(out,indent=2))
```

## Appendix B. Executed normalized-score and first-Adam-step checks

This second small check used the same Linux/PyTorch CPU environment as Appendix A, seed 1602027, and `N=41,d=5,h=4`. It verifies the RMS-normalized formula, positive ReLU-rescaling and hidden-unit permutation behavior, and the infinitesimal first-step interpretation of the Adam score. It does not fit a useful prediction model or validate allocation across tasks. The first-step check holds the parent and contrast fixed so it measures the claimed gate contribution rather than the complete joint update.

Observed maximum absolute errors were `1.388e-16` for rescaled gate gradients, `1.388e-17` for rescaled scores, `2.776e-17` for permuted scores, and `5.551e-17` for the normalized analytic gradient. Predicted/observed gate-step loss decrease was `8.941316578e-7 / 8.941315848e-7`; relative approximation error was `8.168e-8` in this case. These are targeted numerical observations, not universal numerical-error bounds.

Save and execute this block in a fresh process:

```python
import json
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(1602027)
N, d, h = 41, 5, 4
X = torch.cat([torch.randn(N, d), torch.ones(N, 1)], 1)
W, head, y = torch.randn(h, d+1), torch.randn(h), torch.randn(N)

def score(W, head):
    z = (X @ W.T).detach().requires_grad_()
    loss_i = (z.relu() @ head - y).square()
    e = torch.autograd.grad(loss_i.sum(), z)[0]
    contrasts, gradients, scores = [], [], []
    for j in range(h):
        M = X.T @ (e[:, j, None] * X) / N
        M = (M + M.T) / 2
        eigval, eigvec = torch.linalg.eigh(M)
        q = eigvec[:, eigval.abs().argmax()]
        q = q * torch.sign(q[q.abs().argmax()])
        v = 0.05 * z[:, j].square().mean().sqrt() * q / (X @ q).square().mean().sqrt()
        u = M @ v / 2
        S = (u.square() / (u.abs()+1e-8)).sum() / (2*(d+1))
        contrasts.append(v.detach()); gradients.append(u.detach()); scores.append(S.detach())
    return torch.stack(contrasts), torch.stack(gradients), torch.stack(scores)

V, U, S = score(W, head)
scales = torch.tensor([0.3, 2.0, 7.0, 1.1])
Vs, Us, Ss = score(W * scales[:, None], head / scales)
perm = torch.tensor([2, 0, 3, 1])
Vp, Up, Sp = score(W[perm], head[perm])
j = S.argmax().item()
A = torch.zeros(d+1, requires_grad=True)
optimizer = torch.optim.AdamW([A], lr=1e-6, eps=1e-8, weight_decay=0.0)

def loss(A):
    z = X @ W.T
    correction = torch.tanh(X @ A / 2) * (X @ V[j])
    z = z.clone(); z[:, j] += correction
    return (z.relu() @ head - y).square().mean()

before = loss(A)
before.backward()
err = (A.grad-U[j]).abs().max().item()
predicted = 1e-6 * (U[j].square()/(U[j].abs()+1e-8)).sum().item()
optimizer.step()
actual = (before.detach()-loss(A).detach()).item()
out = {
    'rescaled_gate_gradient_max_abs': (Us-U).abs().max().item(),
    'rescaled_score_max_abs': (Ss-S).abs().max().item(),
    'permuted_score_max_abs': (Sp-S[perm]).abs().max().item(),
    'normalized_gate_gradient_max_abs': err,
    'predicted_first_adam_loss_decrease': predicted,
    'observed_first_adam_loss_decrease': actual,
    'relative_first_step_prediction_error': abs(actual-predicted)/predicted,
}
assert out['rescaled_gate_gradient_max_abs'] < 1e-12
assert out['rescaled_score_max_abs'] < 1e-12
assert out['permuted_score_max_abs'] < 1e-12
assert err < 1e-12
assert actual > 0
assert out['relative_first_step_prediction_error'] < 1e-4
print(json.dumps(out, indent=2))
```

---

**Execution starting point:** implement WP00, reproduce Appendices A/B, and preserve the first decision record before attempting any comparative claim. The research question is whether selective growth provides a reproducible advantage—not whether the experiment can be arranged to make a tree-shaped unit win.
