# Research and proof-of-concept plan: bounded interactions and selective nonlinear computation

**Research date:** 16 October 2026  
**Target:** one MacBook Pro with an M3 Max; actual memory, storage, macOS, and software availability must be discovered.  
**Deliverable type:** research assessment and executable experimental handoff, not a completed model benchmark.  
**Primary research object:** the proposed Adaptive Interaction Neuron, evaluated as a bounded quadratic residual and as a physically compressible network component.  
**Evidence convention:** references `[Rxx]` identify inspected external evidence; **Deduction** identifies analysis developed here; **Measured check** identifies the small CPU checks actually executed; everything in the work packages is **proposed work** unless explicitly stated otherwise.

**Reading path:** start with §§1–4 for the verdict and mathematical limits, §§5–8 for implementation/evaluation, §§9–12 for laptop execution, and §§13–14 for audits and evidence. Appendix A contains the self-contained numerical-check script.

## 1. Executive assessment and decision

### 1.1 Recommendation

Proceed with a **bounded-interaction versus polynomial-baseline study**, followed, only when justified, by **cost-validated structured allocation of interaction rank**. Begin with the original one-branch scalar neuron. Establish its exact relationship to ordinary quadratic models before constructing a deep network.

The practical candidate is a linear path plus a small bank of energy-normalized bilinear features, initially contributing zero, trained with a scale-controlled group penalty, and exported by physically deleting unnecessary feature branches. Shared projections across outputs are an explicit implementation refinement. They are not an equivalence to all independently parameterized multi-output neurons.

The key question is:

> At an explicitly measured local inference budget, does bounding and selectively retaining quadratic computation improve prediction, reliability, or the accuracy–cost frontier beyond well-tuned polynomial, rational, multiplicative, and conventional alternatives?

Do **not** frame the project as demonstrating a previously unknown source of neural nonlinearity. Low-rank interactions, shared polynomial bases, sparse basis selection, rational neurons, and selective removal of nonlinear computation already exist. Especially consequential precedents are multi-output polynomial networks (2017), PAON (2024, extended in 2026), and Rational ANOVA Networks (February 2026). [R02], [R03], [R04], [R06], [R10]

No methodological novelty or empirical superiority is established by this investigation. A defensible outcome could instead be a reproducible empirical finding about when global energy normalization helps, a useful compression implementation, a precise failure characterization, or a negative result showing that a simpler normalized polynomial model suffices.

### 1.2 The most important findings

**The proposed scalar function is exactly an affine function plus a normalized quadratic form.** Its interaction residual has an exact spectral-norm amplitude bound and a useful Lipschitz bound. These are deductions, not claims of new theorems. Signed-square polynomial features can represent the same function; this baseline is indispensable.

**There are important limitations beyond the original warning about quadratic extrapolation.** The one-layer nonlinear residual is even in the input; irrelevant input energy suppresses useful interactions; and any finite composition of these affine-plus-bounded blocks remains globally affine plus a bounded remainder. The denominator is therefore a real inductive restriction, not free stability.

**Naive coefficient sparsity is not identifiable.** Penalizing only the original interaction coefficients can be defeated by increasing projection norms. Control factor scales and validate removal in function space. A zero interaction coefficient also blocks projection gradients initially, although a nonzero random projection bank allows coefficient learning to start.

**Current close baselines matter.** The RAN paper and its inspected released tabular implementation differ materially. They must not be silently treated as one reproduced method. PAON supplies another directly relevant, reusable rational-neuron comparison. [R04], [R05], [R06]

**Local feasibility is plausible but not measured on an M3 Max.** The indispensable operations are small dense matrix products, elementwise products, reductions, division, and their gradients. Official PyTorch documentation supports an MPS path, but an operation-level local test is a blocking prerequisite for acceleration claims. [R18], [R19], [R20]

### 1.3 Selected scope and claim ladder

The initial beneficiaries are researchers and developers of small supervised predictors or learned feature-processing components, especially where a limited number of feature interactions might replace wider nonlinear layers. This is a chosen test domain because no application was specified, not evidence that tabular prediction is the best eventual application.

Distinguish four claim levels:

| Level | Evidence required | What it does not establish |
|---|---|---|
| Mathematical component property | Proof, exact limiting cases, independently checked implementation | Better training or prediction |
| Mechanistic learning advantage | Controlled teachers, closest functional controls, matched optimization and resource analysis | Real-task usefulness |
| Local practical advantage | Protected real data, strong executable baselines, uncertainty, actual complete inference costs | Universal superiority or larger-system gains |
| Broader architecture or hardware claim | New representative tasks/scales and suitable measurements | Not available merely from this laptop PoC |

A useful component result need not beat a pretrained foundation model in absolute accuracy. Conversely, beating a weak MLP on a multiplication teacher is not a practical advance.

## 2. Problem statement, original proposals, and assumptions

### 2.1 Inputs, outputs, objectives

A static layer receives a finite vector \(x\in\mathbb R^d\) and produces a scalar or vector prediction. Regression uses squared error; classification produces logits and uses cross-entropy. Softmax in the loss or prediction interface is not counted as a hidden activation, but must be disclosed when describing an “activation-free” architecture.

The objectives are predictive loss, successful and reliable optimization, inference latency in milliseconds, throughput in examples/second, peak memory in bytes, stored parameters/assets in bytes, and training/model-selection expenditure in seconds. Report a frontier; do not silently collapse these into an arbitrary weighted score.

The alleged bottleneck is that affine-plus-elementwise-activation layers may require more intermediate features than a low-rank interaction representation. This bottleneck is a **hypothesis**. Whether arithmetic, launch overhead, memory traffic, or predictive capacity dominates must be measured separately.

### 2.2 Original formulas, preserved

**Adaptive Interaction Neuron (AIN):**

\[
f(x)=w^Tx+b+\sum_{r=1}^{R}\gamma_r\frac{(a_r^Tx)(c_r^Tx)}{1+\|x\|_2^2}.
\tag{1}
\]

All projections in (1) are bias-free. The affine bypass has a bias. The denominator uses the entire input to that neuron, not the projections, not a batch statistic, and not the mean squared energy. Altering any of these is an ablation or amendment.

**Dendritic Compression Neuron:** for prescribed input groups \(G_j\),

\[
s_j=a_j^Tx_{G_j},\qquad
 d_j=s_j+\beta_j\frac{s_j(c_j^Tx_{G_j})}{1+\|x_{G_j}\|^2},\qquad
 y=\sum_jv_jd_j.
\tag{2}
\]

This changes connectivity and where reductions occur. It does not automatically reduce physical data movement on a unified-memory GPU.

**Temporal Memory Neuron:**

\[
m_t=\lambda m_{t-1}+(1-\lambda)p^Tx_t,\quad
 y_t=w^Tx_t+\beta m_{t-1}(q^Tx_t),\quad0\le\lambda<1.
\tag{3}
\]

The output uses the **previous** state. Updating the state before computing the output changes the model.

### 2.3 Explicit refinements, not silent replacements

The main implementation refines (1) in three ways: a shared projection bank across outputs, normalized effective projection directions to remove a coefficient-rescaling loophole, and static structured pruning after training. It tests allocation of deployed nonlinear computation, **not** an online system that invents computational operations during training.

A later greedy growth experiment is permitted only after the basic model is useful; established sparse polynomial basis-selection methods remain its baseline. Learned denominators, input-dependent routing, new optimizers, custom Metal kernels, and hybrid raw/bounded branches are not prerequisites for the core PoC.

Equation (3) does not literally implement the stated temporal XOR example. When the current scalar bit is zero, its output is zero regardless of the previous bit. An explicit correction is \(y_t=w^Tx_t+d_m m_{t-1}+\beta m_{t-1}q^Tx_t\). This permits \(x_t+m_{t-1}-2x_tm_{t-1}\) with \(\lambda=0\). Alternatively append a constant input and specify all corresponding weights. Keep the literal and corrected versions separately named.

### 2.4 Assumptions and hypotheses

| ID | Hypothesis and predicted observation | Support or motivation | Strong alternative explanation | Weakening or falsifying observation |
|---|---|---|---|---|
| H1 | Bounded interactions improve learning/reliability in regimes where target interactions saturate or nuisance amplitudes vary. | Exact bounded residual; established divisive and rational mechanisms. [R04], [R06], [R07] | Ordinary input normalization, smaller initial outputs, or better regularization explains the gain. | Benefit disappears under equal preprocessing, constant-denominator, initialization-scale, and signed-square controls. |
| H2 | A small learned interaction budget yields a better loss–cost frontier than a wider conventional layer on some real tasks. | Low-rank interaction structure is a plausible bias. [R01], [R02], [R03] | More parameters, extra tuning, or favorable teacher design. | Strong local baselines dominate after resource accounting; only self-generated teachers show gains. |
| H3 | Removing selected whole branches saves actual deployed cost without an uncompensated predictive loss. | Structured deletion reduces dense matrix dimensions. | Masks do not save operations; simply training a smaller model is sufficient. | Export has no measured benefit, or a small-from-scratch baseline matches it more cheaply. |
| H4 | Nonlinear capacity is useful unevenly across blocks or branches. | Sparse basis selection/nonlinearity allocation precedents. [R03], [R09], [R10] | Factor rescaling creates apparent sparsity; noisy validation drives selection. | Function-space removal tests show no reproducible heterogeneity or uniform allocation matches it. |
| HD | Grouped local computation provides a useful locality/latency advantage beyond matched conventional grouping. | An architectural possibility; communication-oriented dendritic work exists. [R11] | Connectivity reduction alone explains everything. | Grouped linear/MLP controls have the same or better cost–quality tradeoff. |
| HT | A tiny exponential state plus bilinear readout is useful for narrow streaming tasks. | Equation (3) is a compact filter–interaction model. | An equivalent linear state-space filter bank with the same readout does equally well. | Required state size, horizon errors, or loop overhead removes the advantage. |

Unknown hardware capacity, measurement noise, effective number of independent data groups, and baseline compatibility are consequential unknowns. Resolve them before dependent experiments. Do not assume maximum M3 Max memory, a particular GPU core count, or a permissible model-weight license.

## 3. Landscape, closest prior art, and novelty assessment

### 3.1 Comparison matrix

“Evidence” below means what the cited work reports, not an independent reproduction. Published scores are not imported into the controlled comparison.

| Prior art | Mechanism/problem and assumptions | Evidence and computational implications | Reuse and exact overlap | Discriminating test here |
|---|---|---|---|---|
| Factorization machines, 2010 [R01] | Low-rank pairwise feature interactions, particularly sparse features; classic form excludes self-squares. | Efficient interaction evaluation without enumerating all pairs. | Reimplement the stated equation; interaction learning itself is established. | Include both classic FM and a signed full-quadratic model so omission of diagonal terms cannot weaken the comparison. |
| Polynomial networks, 2016 [R02] | Low-rank polynomial/tensor representations and efficient fitting. | Relevant optimization alternatives to generic neural training. | Reuse formulation; apply the same energy feature map. | Convex explicit quadratic fitting at small dimension versus factorized learning. |
| Multi-output polynomial networks, 2017 [R03] | Shared quadratic basis across outputs, group sparsity, iterative useful-basis selection. | Common basis reduces duplicated projections; a structured optimization approach is provided. | Direct precedent for shared factors and selective capacity. | Normalized shared signed-square implementation; distinguish optimizer effects from function-class effects. |
| Rational ANOVA Networks, February 2026 [R04] | Sparse coordinate-pair rational units and near-linear initialization. | Closely overlaps adaptive pairwise rational computation; paper evidence is not a local reproduction. | Paper formulation and released implementation require separate identities. | Paper-style model and released-equations model; rotation and denominator controls. |
| PAON, 2024/2026 [R06] | Learned rational neurons rather than merely a scalar rational activation. | Image-model experiments; dense linear implementation is available. | Direct rational-neuron predecessor; code has an inspected MIT license. | Paon-S [1/1] and [2/1] layers at measured local budgets. |
| Quadratic divisive neural circuits, 2011 [R07] | Multiplicative responses with divisive normalization for probabilistic computations, under specific coding assumptions. | Analytical/simulated neuroscience evidence, not an M3 benchmark. | Even the combination of quadratic computation and division is old. | Test the specific global energy denominator rather than asserting the combination is new. |
| Π-nets and regularized polynomial networks [R08] | Deep multiplicative polynomial expansions; normalization/initialization matter. | Existing activation-free image architectures; strong regularization changes conclusions. | Do not compare only to an unstable, untuned product network. | Well-initialized and regularized raw polynomial control; full image models are context only here. |
| NAFNet [R12] | Multiplicative SimpleGate inside a restoration architecture. | Task-specific engineering, including other architectural elements. | Inspect/reuse gate algebra, not claim that a gate-only MLP reproduces NAFNet. | SimpleGate MLP control; image-restoration superiority would require a separate task study. |
| DCN V2 [R13] | Low-rank cross layers for feature interactions and ranking. | Efficient feature-cross alternative with application-specific evidence. | Locally implementable low-rank cross network. | Cross network plus its parallel MLP, not just an impoverished cross-only variant. |
| L0 regularization / DeepReDuce [R09], [R10] | Structured capacity selection / removing expensive nonlinearities. | Different sparsity and deployment cost settings. | “Only use nonlinearity where useful” is not independently novel. | Generic pruning and small-from-scratch controls for all relevant families. |
| Dendritic communication work, 2026 [R11] | Communication-oriented local dendritic processing. | Full quantitative paper was not successfully accessed during this investigation. | Official repository exists; no quantitative replication claim is made. | Real contiguous-group execution against equally grouped conventional models. |
| LRU / S4 [R14] | Stable linear recurrence and structured state-space sequence models. | Much broader sequence capabilities than a single exponential trace. | Stronger temporal context; same-filter/same-readout controls isolate the tiny proposal. | Equal state and identical bilinear readout; tiny GRU as a nonlinear recurrent alternative. |

Additional adjacent work includes quadratic adaptation around a linear path (QuadraNet V2), reinforcing that linear initialization/bypasses are not a unique contribution. Its broader architecture is contextual rather than a required local baseline. [R15]

### 3.2 Exact novelty verdict

**Already established:** multiplicative feature interactions; low-rank quadratic neurons; rational nonlinear neurons; divisive normalization; linear bypasses; sparse shared polynomial bases; near-linear initialization; pruning nonlinear operations; linear temporal memory with nonlinear readout.

**Potentially distinctive but unproven:** the empirical tradeoff of this exact fixed \(1+\|x\|^2\) denominator together with cost-validated branch deletion on Apple Silicon and carefully identified nuisance-energy regimes. This is initially a proposed **empirical/engineering contribution**, not a new mathematical family.

**What would invalidate a stronger novelty claim:** an existing implementation or paper that performs the same transformed-feature polynomial fitting and allocation; mathematical reduction to that method; or a normalized polynomial baseline matching the candidate. The last two are already sufficiently plausible that no “new neuron” claim should be registered now.

**What can be reused:** standard tensor operations, convex small-dimensional solvers, mature tabular packages, PAON code with attribution, existing benchmark metrics, and conventional structured pruning. Do not rebuild dataset ingestion, optimizers, or a training framework merely to give the project a new name.

The refinement search covered shared polynomial bases, sparse rational interactions, Padé neurons, divisive quadratic circuits, nonlinearity pruning, and quadratic adaptation. It uncovered relevant equivalents rather than establishing an unoccupied methodological niche. A narrow, fair diagnostic program remains scientifically worthwhile; a speculative new method is not forced into the plan.

### 3.3 RAN reproduction caveat

The inspected RAN paper uses a positive softplus-based denominator and discusses near-identity gating. Its released `TAB_model.py` instead evaluates \(P/(|Q|+10^{-5})\); its tabular assembly does not implement the discussed residual gates. Do not combine the paper's guarantees with the released implementation's results. The inspected repository did not expose a verified reuse license. Independently implement published equations or obtain permission before incorporating source; keep provenance explicit. [R04], [R05]

### 3.4 Search scope and limitations

Searches used original names and equivalents: factorization machines, sigma–pi and bilinear neurons, polynomial networks, normalized quadratic forms, rational/Padé neurons, divisive normalization, structured and nonlinear-operation sparsity, dendritic communication, exponential/Volterra memory, Apple MPS operation support, and strong 2025–2026 tabular baselines. Backward tracing found the 2010/2011 and 2016/2017 precedents; forward/current searches found PAON extensions, RAN, and updated application baselines.

The search was directed by questions that could change design or interpretation, not by a paper quota. The last refinement searches reinforced already-covered function classes. This is not exhaustive worldwide coverage. Important blind spots are inaccessible full dendritic results, some repository licensing, uninspected implementation variants, non-English literature, and unpublished work. A limited public patent screen found older quadratic dendritic/synaptic mechanisms; it is not a patentability or freedom-to-operate opinion. [R28]

## 4. Mathematical specification and implications

### 4.1 Exact quadratic reduction

**Deduction.** Define the symmetric matrix

\[
Q=\frac12\sum_r\gamma_r(a_rc_r^T+c_ra_r^T).
\]

Then \(\operatorname{rank}(Q)\le2R\) and

\[
f(x)=w^Tx+b+q_Q(x),\qquad q_Q(x)=\frac{x^TQx}{1+\|x\|^2}.
\tag{4}
\]

Let \(h(x)=x/\sqrt{1+\|x\|^2}\). Then \(q_Q(x)=h(x)^TQh(x)\). Thus explicit quadratic features of \(h\), plus the original linear features of \(x\), span the shallow model. Their homogeneous polynomial kernel is

\[
k(x,z)=\frac{(x^Tz)^2}{(1+\|x\|^2)(1+\|z\|^2)}.
\]

No kernel implementation is needed at low dimension: enumerate the features directly. The same transformation can be supplied to established polynomial models. The mathematical connection to polynomial fitting is more important than the choice of name. [R02], [R03]

### 4.2 Exact amplitude and safe sensitivity bounds

For symmetric \(Q\),

\[
\sup_x|q_Q(x)|=\|Q\|_2,\qquad
\max_{\|x\|\le M}|q_Q(x)|=\frac{M^2}{1+M^2}\|Q\|_2.
\tag{5}
\]

The first is generally a supremum approached at infinite radius along an eigenvector of largest absolute eigenvalue. It need not be achieved at a finite input. This is tighter and less factorization-dependent than \(\sum_r|\gamma_r|\|a_r\|\|c_r\|\).

Differentiation gives

\[
\nabla q_Q(x)=\frac{2Qx}{1+\|x\|^2}
-\frac{2(x^TQx)x}{(1+\|x\|^2)^2}.
\tag{6}
\]

The Jacobian of \(h\) has norm at most \((1+\|x\|^2)^{-1/2}\). Consequently

\[
\|\nabla q_Q(x)\|\le\frac{2\|Q\|_2\|x\|}{1+\|x\|^2}\le\|Q\|_2.
\tag{7}
\]

This upper bound is tight over the family: take \(Q=\operatorname{diag}(1,-1)\), \(x=(1,1)/\sqrt2\). The entire scalar neuron has Lipschitz constant at most \(\|w\|_2+\|Q\|_2\). For vector output with matrices \(Q_j\), a safe bound is \(\|W\|_2+\sqrt{\sum_j\|Q_j\|_2^2}\). Composed layer bounds multiply and can be enormous. Bounded interactions do not constrain learned linear amplification or guarantee adversarial robustness.

### 4.3 Signed squares and compression certificates

\[
(a^Tx)(c^Tx)=\left(\frac{(a+c)^Tx}{2}\right)^2
-\left(\frac{(a-c)^Tx}{2}\right)^2.
\tag{8}
\]

A single bilinear branch has at most one positive and one negative eigenvalue in its symmetric quadratic matrix. Thus rank at most two is not a sufficient one-branch representability condition: a positive-definite rank-two quadratic needs two branches. More generally, for a scalar output the minimum unconstrained bilinear branch count is the larger of the positive and negative inertia counts, by pairing opposite-sign eigencomponents and leaving unmatched signed squares. This deduction is a capacity-accounting rule, not a claimed new theorem; it does not imply the same bound for a bank shared across outputs.

An exactly converted signed-square control must match predictions before training or timing. An unconstrained learned signed-square network may use a different parameter budget; count it honestly rather than calling equal branch counts equal capacity.

For scalar \(Q=U\operatorname{diag}(\lambda)U^T\), sort eigenvalues by absolute value and retain the first \(k\). Then

\[
\sup_x|q_Q(x)-q_{Q_k}(x)|=|\lambda_{k+1}|.
\tag{9}
\]

This is the optimal worst-case rank-\(k\) symmetric approximation under this normalized function error, by the spectral low-rank approximation result. It is a diagnostic and export option, not a new spectral theorem. On a radius-\(M\) domain multiply the certificate by \(M^2/(1+M^2)\).

For independently approximated outputs use the safe vector bound \(\sqrt{\sum_j\|Q_j-\widetilde Q_j\|_2^2}\). Independent eigendecompositions may destroy shared-factor economy; compare actual stored matrices and measured latency. For a deep block, propagate its output error through bounds on all following blocks; do not report the local certificate as a global prediction certificate.

Exact eigendecomposition is appropriate for the small core dimensions. A power-method estimate alone is not a certified upper bound. At larger dimensions use a mathematically safe Frobenius upper bound or a verified residual-bound method, explicitly labeling looseness.

### 4.4 Structural failure regimes

**Even residual.** \(q_Q(-x)=q_Q(x)\), so a single original neuron obeys \(f(x)-f(-x)=2w^Tx\). It cannot fit arbitrary odd nonlinear functions on a symmetric domain. Biased transformations in a deeper model change this shallow symmetry; report that distinction.

**Asymptotically affine composition.** Write each finite block as \(F_l(z)=W_lz+b_l+r_l(z)\), where \(r_l\) is globally bounded for fixed finite weights. Induction shows any finite composition has the form \(Ax+c+R(x)\) with bounded \(R\). Therefore it cannot represent globally unbounded superlinear targets such as \(x_1x_2\) along rays. This is a limitation of the exact family, not proof of poor approximation on compact domains.

**Nuisance energy.** Keeping \(x_1=x_2=1\), appending \(m\) irrelevant unit coordinates changes \(x_1x_2/(1+\|x\|^2)\) from \(1/3\) to \(1/(3+m)\). The normalization couples useful and irrelevant inputs. A learnable preceding layer may filter nuisance energy, but introduces additional parameters and changes the explanation of any gain.

**Almost-constant denominator.** A LayerNorm with fixed unit gains makes squared energy approximately the feature dimension. The denominator can then be little more than a constant scale. Do not place such normalization immediately before the candidate in the attribution experiment without a corresponding control.

**Scale degeneracy.** Replacing \(a,c,\gamma\) by \(ta,tc,\gamma/t^2\) leaves (1) unchanged while shrinking a gamma-only L1 penalty. Unit effective directions or an invariant path penalty are necessary. Even then, cancellation between branches and downstream rescaling mean coefficient magnitude is not a complete importance measure.

**Initialization.** With \(\gamma=0\), first-step projection gradients vanish; coefficient gradients need not. Initialize projection directions randomly and nonzero. A bank of zero projections plus zero coefficients cannot learn through the intended path.

### 4.5 Numerical checks actually performed

These checks ran on **Linux x86-64, CPU, PyTorch 2.10.0+cpu, NumPy 2.3.5**, seed 20261016. No MPS device was available. Float64 was used except for the explicitly identified overflow check. Dimensions were \(d=8,R=3\), with 512 random inputs spanning scales \(10^{-5}\) through \(10^5\). They are correctness checks, not evidence of learning superiority.

| Check | Measured outcome |
|---|---|
| Bilinear expression versus symmetric quadratic form | Maximum absolute difference \(1.78\times10^{-15}\) |
| Autograd versus equation (6) | Maximum absolute difference \(8.88\times10^{-16}\) |
| Shared bilinear output versus converted signed squares | Maximum absolute difference \(3.55\times10^{-15}\) |
| Largest sampled gradient norm divided by \(\|Q\|_2\) | 0.60619; the constructed tight example attained 1.0 |
| Rescale both projections by 100 and coefficient by 1/10,000 | Output difference \(1.78\times10^{-15}\); coefficient L1 penalty became 0.0001 of its original value |
| Rank-2 discarded spectral bound | 2.813283087831697; error at radius \(10^6\) along omitted eigenvector was 2.8132830878288844 |
| Zero-gate optimization check | First projection gradient norm 0; after one coefficient update the next projection gradient norm was 0.00049515 |
| Literal temporal formula on all four bit transitions | Best least-squares MSE 0.25; adding direct state readout reduced it to \(6.16\times10^{-32}\) |
| FP32 branch at \(x=(10^{20},10^{20})\) | Naive product/energy evaluation nonfinite; scaled equivalent gave 0.4999999702 |
| LayerNorm energy, dimension 8, epsilon \(10^{-5}\) | Range 7.9993475–7.9999694 over 128 random examples |
| Additional zero-direction row-norm audit | Explicit square-root formulation produced NaN gradients; `vector_norm` formulation produced finite gradients of \(10^8\) with epsilon \(10^{-8}\) |

A separate CPU float64 smoke test of the §5.2 class used seven 3-dimensional inputs, two outputs, and two branches. The stable map passed numerical gradient checking; the class differed from the literal calculation by at most \(2.22\times10^{-16}\), unpruned export differed by zero, rank-zero output was exactly affine, and tested gradients were finite. This is a small implementation check, not complete training-path validation.

The recipe in Appendix A is sufficient to reproduce the central checks independently. Raw task data, model benchmarks, M3 timing, MPS backward compatibility, and complete baseline reproductions have **not** been executed.

## 5. Candidate implementation and deployment specification

### 5.1 Required model variants and names

Implement a single registry with explicit, immutable method identifiers:

| Identifier | Meaning |
|---|---|
| `ain_literal` | Independent original scalar equation, or independent factors for every output; the fidelity reference |
| `bir_shared` | Shared bounded interaction residual in equation (10), effective direction normalization enabled |
| `bir_raw` | Same shared architecture without the input-energy denominator |
| `bir_constant` | Same architecture with a scalar denominator calibrated once from training inputs and then frozen |
| `pn_square_raw`, `pn_square_bounded` | Learned signed-square polynomial features, without/with the same energy map |
| `bir_export` | Physically compact, frozen-direction implementation of a selected `bir_shared` checkpoint |
| `bir_projected_energy` | Conditional amendment only; not part of the initial selected method |

The primary primitive experiment has one branch, scalar output, and no hidden network. For learning curves expand rank only when one branch cannot express a specified teacher or development evidence supports additional capacity.

The shared block is

\[
F(x)=Wx+b+V\big[(\bar A h(x))\odot(\bar C h(x))\big],
\quad W\in\mathbb R^{o\times d},\ A,C\in\mathbb R^{K\times d},\ V\in\mathbb R^{o\times K},
\tag{10}
\]

where each effective row \(\bar A_k=A_k/\max(\|A_k\|,\epsilon_w)\), and similarly for \(C\). Set \(\epsilon_w=10^{-8}\) in the FP32 core and record it. Effective row norms are at most one. The coefficients are contained in \(V\); there is no redundant separate gamma parameter.

For nonzero raw directions, this normalization does not remove possible scale from an individual branch because its coefficient can absorb that scale. Sharing directions across outputs **does** constrain the model. Do not claim the shared model equals arbitrary independent-output (1).

### 5.2 Stable numerical implementation

For finite inputs, compute the energy map without squaring huge input values:

\[
s=\max(1,\max_i|x_i|),\quad z=x/s,\quad
h=z/\sqrt{s^{-2}+\sum_i z_i^2}.
\tag{11}
\]

This is algebraically the original map, not clipping. The bounded path remains evaluable at the tested extreme FP32 inputs. It does not prevent the bypass from overflowing for arbitrary weights or values. Reject NaN/Inf inputs explicitly in data validation; do not silently map them to zero. Overflow events during training are logged failures, not missing runs.

The following is **new reference code for the agent to place in `src/neuron_poc/models/bir.py`**. A small CPU float64 smoke test of this class passed; it has not undergone an MPS test or task-training evaluation.

```python
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def energy_map(x: Tensor) -> Tensor:
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    z = x / scale
    denom = torch.sqrt(scale.reciprocal().square()
                       + z.square().sum(dim=-1, keepdim=True))
    return z / denom


def unit_rows(weight: Tensor, eps: float = 1e-8) -> Tensor:
    norm = torch.linalg.vector_norm(weight, dim=-1, keepdim=True)
    return weight / norm.clamp_min(eps)


class BoundedInteraction(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        if d_in < 1 or d_out < 1 or rank < 0:
            raise ValueError("Invalid dimensions or negative rank")
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        self.linear = nn.Linear(d_in, d_out)
        self.a = nn.Parameter(torch.randn(rank, d_in))
        self.c = nn.Parameter(torch.randn(rank, d_in))
        self.v = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.d_in:
            raise ValueError("Wrong input feature dimension")
        y = self.linear(x)
        if self.rank == 0:
            return y
        h = energy_map(x)
        feature = F.linear(h, unit_rows(self.a))
        feature = feature * F.linear(h, unit_rows(self.c))
        return y + F.linear(feature, self.v)

    def branch_penalty(self) -> Tensor:
        # Group penalty; do not additionally penalize a redundant gate.
        return torch.linalg.vector_norm(self.v, dim=0).sum()

    @torch.no_grad()
    def export_tensors(self, keep: Tensor) -> dict[str, Tensor]:
        # The caller must provide validated unique integer indices in range.
        return {
            "w": self.linear.weight.detach().clone(),
            "b": self.linear.bias.detach().clone(),
            "a": unit_rows(self.a)[keep].detach().clone(),
            "c": unit_rows(self.c)[keep].detach().clone(),
            "v": self.v[:, keep].detach().clone(),
        }
```

Use fresh independent initialization generators for model weights, data, and sampling. Do not let the number of earlier trials alter subsequent seeds. Use the stable map by default for attribution; an algebraically equivalent faster map may be an engineering variant after parity testing. Do not compare stable candidate execution with an unnecessarily slow baseline implementation.

The exporter must construct a new module with compact contiguous tensors, no training-time row normalization, and no evaluation of deleted branches. For \(K=0\), emit exactly a linear layer. In a stack, fold consecutive affine layers where no nonlinear block remains; apply the same folding to affine baseline networks.

### 5.3 Integration architectures

Two architectures answer different questions and must be reported separately.

**Direct model:** one block from standardized raw inputs to output logits/scalars. This is the cleanest test of the original mechanism. Dense quadratic fitting is an exact function-space reference here.

**Block stack:** affine input lift \(d\to h\); \(L\) blocks \(h\to h\); affine readout \(h\to o\). Every nonlinear block uses its own input energy. Default exploratory anchors are \(h\in\{32,64,128,256\}\), \(L\in\{1,2,4\}\), and \(K\in\{1,4,16,64\}\), restricted by calibration, not permanent search boundaries. The affine lift changes the normalization geometry; its cost and effect must be disclosed. Conventional and multiplicative stacks receive the same integration opportunity and resource accounting.

No hidden LayerNorm, batch-dependent normalization, dropout, activation, attention, or input clipping belongs in the clean direct comparison. In the integration/native-baseline track such mechanisms may be tuned, but they invalidate claims that *all* hidden nonlinearity comes from the proposed interaction unless explicitly accounted for.

### 5.4 Training, scale control, and pruning

Minimize

\[
\mathcal L_{\mathrm{train}}+\lambda_g\sum_{l,k}\|V_{l,:,k}\|_2
+\lambda_W\sum_l\|W_l\|_F^2.
\tag{12}
\]

For the clean candidate implementation, use AdamW with optimizer `weight_decay=0` and add exactly the displayed penalties to the loss; include every affine matrix, including lift and readout, in the stated L2 term, excluding biases. This avoids accidentally applying both an explicit L2 penalty and decoupled weight decay. Native-strength baselines keep their documented optimizer/decay recipe, while causal controls use the same specified objective.

The group penalty promotes small branch contributions; it does not by itself guarantee exact zeros under AdamW. Do not interpret the number of small entries as deployment sparsity. Keep direction parameters out of ordinary weight decay in the main scale-controlled variant; their effective norms are already constrained. Test invariant path-norm regularization of unnormalized directions as an attribution alternative.

Initialize \(V=0\), nonzero random directions, and the affine path using the same linear initialization as the matched control. Allow the coefficient path to move before ramping the group penalty. The ramp duration is selected on development data and recorded; include zero penalty and no-ramp controls. Measure coefficient and direction gradient norms for the first updates.

For raw versus bounded controls, tune coefficient learning-rate scales separately. Also include a training-energy-based preconditioner control: scale the raw or constant-denominator feature outputs so their initial empirical variance matches the bounded features. Otherwise a claimed stability advantage could simply reflect smaller effective steps.

**Static allocation algorithm:**

1. Fit a modest branch bank; save the unpruned checkpoint, not just its winning export.
2. Rank branches provisionally by \(\|V_{:,k}\|_2\). Form nested retained-rank candidates by repeated halving down to one and zero, including the full bank. For several blocks, perform coordinate-wise rank reduction, evaluating every affected export; retain the complete search record. This is a heuristic, not a globally optimal allocator.
3. Compute validation loss, prediction changes, the removed-branch bound, and **actual exported latency** for each candidate. Count selection and optional fine-tuning cost.
4. Fine-tune selected compact candidates on training data only. Choose using the frozen development selection rule. Compare compact-from-scratch training at exactly the same retained dimensions.
5. Freeze the final architecture and export before protected evaluation. No test-driven branch recovery or additional fine-tuning is permitted.

For unit directions, deleting set \(D\) gives a safe local vector-error bound \(\sum_{k\in D}\|V_{:,k}\|_2\). Equation (9), or an exact \(\Delta Q\) bound, is often tighter and handles cancellations. Report both where practical. All certificates assume fixed finite weights and the stated denominator.

## 6. Baselines and reproduction requirements

### 6.1 Baseline tiers

**Tier A, required before a mechanism claim:** affine least squares/logistic regression; tuned ReLU/GELU MLP; raw shared multiplication; learned raw and bounded signed-square polynomial models; explicit raw and bounded quadratic fitting; constant-denominator and equal-input-normalization controls.

**Tier B, required before a local competitiveness claim:** classic FM, low-rank DCN V2 plus MLP, SimpleGate MLP, PAON, the RAN comparison defined below, and strong task-level CatBoost, RealMLP, and TabM. All are small enough in principle for this problem size, but no package receives a compatibility exemption: its actual local path must pass the same checks.

**Tier C, strong contemporary context and local execution when permitted/feasible:** pretrained TabICLv2, and the locally available TabPFN version permitted by its license and actual memory. These are not neuron-level causal controls. Their reusable pretraining expenditure and retained training context differ from scratch-trained models.

Models excluded from a budget-constrained comparison can still appear on the unrestricted frontier. Missing a strong baseline blocks a “best contemporary method” claim, not necessarily a narrow component result. Published scores on different splits or hardware are never substituted for direct runs.

### 6.2 Exact local definitions

| Baseline | Implementation/configuration and reproduction checks |
|---|---|
| Affine | Fit ridge regression or regularized multinomial logistic regression. Independently verify that an activation-free affine stack collapses to one affine map. No LayerNorm or nonlinear preprocessing beyond the common, frozen standardization. |
| Explicit quadratic | Features \([1,x_i,h_i^2,\sqrt2h_ih_j\ (i<j)]\) for bounded fitting; substitute \(x\) for \(h\) for raw fitting. Enumerate all diagonal and off-diagonal terms. Ridge or convex logistic optimization; no penalty on intercept. A uniform quadratic coefficient penalty then corresponds to a Frobenius penalty on symmetric \(Q\). Verify feature ordering and a constructed \(Q\). |
| Signed-square PN | \(Wx+b+U[(Hx)^{\odot2}]\), and the identical model on \(h(x)\) for its interaction path. Signed output coefficients are mandatory. Include the exact conversion (8) and an independently trained model. Match parameters or latency, not the name “rank.” |
| Classic FM | \(b+w^Tx+\frac12\sum_k[(v_k^Tx)^2-\sum_i v_{ki}^2x_i^2]\). Start ranks 1, 4, 16, 64; expand only if development boundary evidence warrants. Use squared/cross-entropy loss as appropriate. |
| MLP | ReLU and GELU; tune depth, width, learning rate, weight decay, dropout, and residual option. Direct-comparison mode uses common preprocessing. Native-strength mode permits justified normalization/embeddings and records their cost. |
| SimpleGate MLP | \(Wx+b+U[(Ax+a_0)\odot(Cx+c_0)]\), followed by optional residual blocks. Native-strength variant may use LayerNorm; causal raw-product control does not. This is not the full NAFNet architecture. [R12] |
| DCN V2 low-rank | \(z_{l+1}=z_0\odot[U_l(V_l^Tz_l)+b_l]+z_l\). Test 1–4 cross layers and geometric ranks, with a tuned parallel ReLU/GELU MLP and combined head. No claim to reproduce an industrial ranking system. [R13] |
| PAON | Use licensed `PaLaLinear(in_ch=d, out_ch=o, degrees=(1,1), paon_type="s")`; include (2,1). For [1/1], \(P_1=a_0+a^Tx,Q_1=1+c^Tx\), output \((Q_1P_1+a_0)/(Q_1^2+1)\). Preserve initialization and compare a stacked form. Isolate the linear module if unused deformable-convolution imports block installation; verify algebraic parity. Higher-denominator-degree smoothing is not automatically bounded away from zero. [R06] |
| RAN released-equations | Independently implement the inspected form: 1D degree-3 numerator/degree-2 denominator and all total-degree bivariate monomials through 3/2; denominator \(\lvert Q\rvert+10^{-5}\). Sum coordinate/pair/head outputs. Native assembly: widths 32,16, LayerNorm between layers, two heads; Adam 0.01, 30 epochs, batch 256 as the released starting configuration, plus fair tuning. Core raw dimensions are below its first-50-coordinate pair restriction. [R05] |

For RAN also run a **paper-form rational-feature control**, clearly labeled an independent instantiation, not a reproduced author score: univariate numerator monomials through degree 3; denominator \(1+\operatorname{softplus}(\beta_1x+\beta_2x^2)+10^{-6}\); bivariate numerator monomials of total degree ≤3 and denominator monomials of total degree 1–2. Concatenate main and selected-pair outputs and use a linear head. Use all pairs when affordable and a seeded fixed subset otherwise. The paper does not unambiguously specify a scalar “identity” for a two-input gate, so this control omits that ambiguous gate rather than inventing a faithful reproduction. An optional affine-bypass, zero-head version is an explicitly labeled adaptation. [R04]

This baseline distinction is essential: an inaccessible license, a paper–code discrepancy, or an unstable implementation is not evidence that the competing idea loses. Report an unresolved reproduction gate when appropriate.

### 6.3 Strong task-level implementations and assets

**CatBoost:** official documentation lists macOS arm64 wheels and CPU execution, not CUDA GPU execution on macOS. Use its native CPU implementation, classifier/regressor, native default as an included configuration, and validation-selected depth, learning rate, iterations, and L2 regularization. Record thread count; do not try to force it onto MPS. [R21]

**RealMLP:** use the official `pytabkit` implementation of the pre-tuned RealMLP model, including its prescribed preprocessing, as a separate native-strength track. Preserve the resolved estimator configuration and source revision. Use its documented classifier/regressor interface in the installed revision; test serialization and prediction ordering before runs. This avoids freezing an unverified package API name in this handoff. [R16]

**TabM:** use the official package/repository and native parameter-efficient ensemble behavior, including its averaging and embedding choices where applicable. Do not set ensemble size to one merely to simplify a neuron comparison. Also show a budget-matched configuration. Record all constructor defaults after resolution and count the complete ensemble computation. The inspected project license is Apache-2.0. [R17]

**TabICLv2:** the inspected official implementation exposes `TabICLClassifier` and `TabICLRegressor`, supports an MPS selection path, and identifies checkpoints `tabicl-classifier-v2-20260212.ckpt` and `tabicl-regressor-v2-20260212.ckpt`. Its code/model-card licensing is BSD-3-Clause. Verify checkpoint revisions and download hashes. Disable CUDA-specific acceleration on the laptop and test a faithful FP32 path. A model with context stored during `fit` can perform substantial work at `predict`; measure both cold and cached use, including context storage. MPS selection in a README is not proof that every operation works. [R22]

**TabPFN:** official documentation inspected on the research date lists newer local versions through TabPFN-3.5, rather than stopping at 2.5. Model-weight terms differ from code terms and include use restrictions. Treat the eligible local base model as an additional reference after license and memory gates; never use a paid/hosted variant. If terms or hardware exclude it, disclose the exclusion and narrow claims. Do not accept contractual terms on the user's behalf. [R23]

Do not require pretraining these assets locally. Disclose pretraining as reusable external computation, distinct from dataset-specific fitting and inference. Investigate documented training/evaluation overlap where available; lack of a disclosed overlap is not proof of contamination-free benchmarking.

### 6.4 Tuning and fairness policy

Maintain two tracks: **mechanism-matched** experiments with shared preprocessing and controlled architecture changes, and **native-strength** models with their own justified recipes. Neither track substitutes for the other.

For custom neural models, initial optimizer searches cover learning rate \(10^{-5}\)–\(3\times10^{-2}\), zero or log-uniform weight decay \(10^{-7}\)–\(10^{-2}\), coefficient/denominator learning-rate multiplier 0.1, 1, or 10, and batch sizes selected from the hardware calibration. Group penalty includes zero and \(10^{-7}\)–\(10^{-1}\) after target/loss normalization. Try ReLU/GELU, dropout 0–0.3 only in the appropriate track, and the dimensions in §5.3. These are exploratory starting ranges, not restrictions that may conceal a boundary optimum.

Convex quadratic/linear models use a logarithmic regularization grid including zero where numerically appropriate. Solve regression with QR/SVD or a stable regularized system, not an explicit matrix inverse. For logistic models check gradient norm, objective decrease, and convergence status; a solver that hit a limit is not a successfully fitted baseline.

CatBoost starts from its native default and depth 4, 6, 8, 10; learning rates 0.01–0.3; L2 values 1–100. Retain native RealMLP/TabM/PAON starting configurations and expose their important regularization/capacity dimensions. Record failed trials and their resource cost. Do not punish a model for a bug in the common harness.

Calibrate training horizons on development data, beginning with a modest power-of-two epoch horizon. Double it when competitive configurations are still improving or choosing their best validation checkpoint near its end. Freeze the horizon/early-stopping rule only after representative learning curves show that truncation is not deciding the comparison. Native published/default schedules remain included even when they are longer.

Allocate equal **measured selection-compute budgets** within the causal neural comparison, and separately report validation loss versus total selection time for the native-strength families. Equal trial counts alone are not fairness. Use successive budget doublings until competitive families' validation improvements over the last two doublings are below paired validation sampling uncertainty, or a documented resource limitation prevents resolution. Fix the resulting budget and search procedure before confirmation. This is development-stage convergence assessment, not repeated significance testing on protected data.

Every model receives the same labels, allowed feature set, and outer split. Preprocessing is fitted on training data only. Budget matching must count extra input maps, learned embeddings, projection parameters, export operations, and retained training examples. Generic optimizations and structured pruning are offered to competing models when applicable.

## 7. Data and experimental populations

### 7.1 Synthetic mechanism suite

Synthetic tasks identify the mechanism; they do not establish real-world usefulness. Define generators in `data/synthetic.py`, with separate teacher, training-sample, validation-sample, and test-sample seeds. Draw independent teacher instances for replication. Save coefficients and rotations, not only a seed.

Start with \(d=8\), 2,048 training, 1,024 validation, and 8,192 test points per teacher instance. These sizes are exploratory anchors for cheap function fitting, not the final power calculation. Use independent Gaussian and uniform-on-box inputs, and an orthogonal rotation generated by QR with a fixed sign convention. Increase sample size when train/test error indicates estimation, not approximation, is limiting.

| Generator | Exact target before training-only standardization | Purpose and expected boundary |
|---|---|---|
| Linear | \(w^Tx\) | No interactions are needed; pruning should not systematically invent a gain. |
| Raw quadratic | \(w^Tx+x^TQx\), rank and eigenvalue signs controlled | Raw polynomial baseline has the correct unrestricted growth; bounded model is expected to fail at sufficiently large radius. |
| Bounded quadratic | \(w^Tx+x^TQx/(1+\|x\|^2)\) | Exact representability check, explicitly favorable to the proposal. |
| Additive nonlinear | \(\sum_{i=1}^4\sin x_i\) | Odd, non-pairwise negative control; use a compact domain to separate shallow symmetry from deep approximation. |
| Local cross plus nuisance | \(x_1x_2/(1+x_1^2+x_2^2)\), with independent remaining coordinates | Exposes harmful global denominator coupling; compare coordinate and rotated versions. |
| Higher order | \(x_1x_2x_3\) on a compact box | Tests when second-order shallow capacity is the wrong mechanism. |

For the quadratic teachers choose signed spectra, e.g. \((1,-0.7,0.3)\), plus random orthogonal bases; vary rank relative to fitted capacity. These values define a generator, not an expected experimental result. Include zero-noise correctness runs and independent Gaussian label noise with standard deviation 0.1 times the noise-free training-target standard deviation. Classification diagnostics can threshold a teacher, but do not replace regression error with accuracy to hide amplitude failure.

Use amplitude factors \(1/4,1/2,1,2,4,8,16\) with **recomputed ground-truth targets**. For a separately specified scale-invariant directional classification generator, unchanged labels are legitimate. Scaling a real dataset while retaining its labels is merely a stress probe, not a valid distribution-shift accuracy benchmark.

Nuisance dimensions progress geometrically from zero through values where the energy term dominates. Compare fixed useful-coordinate scale, fixed total energy, and train/test nuisance-scale shift; otherwise dimension and amplitude are confounded. Rotate useful subspaces to prevent coordinate-pair methods from receiving an artificial disadvantage or advantage.

### 7.2 Real core suite

The core contains two physical regression problems and one multiclass engineered-feature problem, covering different target types and different dependence structures. These small datasets are a **fixed evaluation panel**, not a random sample of all machine-learning tasks.

| Dataset and provenance | Role, expected size, license | Split and leakage controls |
|---|---|---|
| UCI Concrete Compressive Strength, ID 165, I-Cheng Yeh [R24] | 1,030 rows, 8 features, strength in MPa; CC BY 4.0 | Group by the seven ingredient amounts, excluding age. All ages of the same recorded recipe stay together. Target: unseen recorded recipes, not random rows from already-seen recipes. |
| UCI Airfoil Self-Noise, ID 291, Brooks/Pope/Marcolini [R25] | 1,503 rows, 5 features, scaled sound pressure in dB; CC BY 4.0 | Group by the four non-frequency inputs so frequency sweeps for the same recorded condition do not cross splits. Audit actual unique groups and coverage. Treat angle as numeric despite inconsistent metadata typing. |
| UCI Dry Bean, ID 602, Koklu/Ozkan [R26] | 13,611 rows, 16 numerical image-derived descriptors, 7 classes; CC BY 4.0 | Group exact duplicate feature tuples; stratify class proportions at group level where possible. Do not claim independence across unreported acquisition batches. This is not a raw-image recognition experiment. |

Download via the official UCI pages or their documented `ucimlrepo.fetch_ucirepo(id=...)` interface. Preserve raw files, metadata, license text/reference, retrieval date, SHA-256, parser version, feature names, target names, row IDs, and grouping keys. Exact counts are verified from bytes during execution; expectations are not file-integrity checks.

For each core dataset, assign approximately 60% of groups to training, 20% to development validation, and 20% to protected test, using a deterministic seeded group allocation. Counts may differ because groups have different sizes. Optimize only split balance and coverage, never predictive scores. Group allocation for Dry Bean must retain every class in training and evaluation; save any unavoidable imbalance. Keep exact conflicting-label duplicate groups together rather than selectively discarding hard examples.

If grouping yields too few independent groups or inadequate input support, report the limitation before running models. A secondary row-random split may diagnose interpolation versus group transfer, but does not silently replace the primary population. A grouped-data result with a wide interval is inconclusive, not rescued by pretending its rows are independent.

### 7.3 Independent transfer confirmation

Reserve **UCI Bike Sharing, ID 275**, for an evidence-triggered, fresh application confirmation after the candidate and search policy are frozen. Its public description covers hourly/daily rental counts in 2011–2012 and gives a CC BY 4.0 license. The page's reported count must be reconciled with the actual `hour.csv`; do not force a remembered row count. [R27]

Use `cnt` as target. Exclude `casual`, `registered` (components of the target), `instant`, and raw date as predictors. Use date for chronological splitting: training through 30 June 2012; validation July–September 2012; protected test October–December 2012. Calendar variables and available contemporaneous weather are allowed. This is **conditional count prediction/nowcasting**, not a demonstrated weather-free future forecast.

Use one-hot encoding of categorical calendar/weather variables for the common neural track, fitted on training categories with an explicit unknown-category policy. Native tree models may use their own supported treatment. Bootstrap contiguous day blocks for uncertainty; set block length from training/development dependence diagnostics, not test results. A seasonal transfer result does not establish broad distributional robustness.

### 7.4 Shared preprocessing conventions

Fit numeric means and standard deviations on training only. Replace a training-zero standard deviation by one, retain the constant-feature flag, and log it. No target-dependent feature selection. Regression targets are standardized by training mean and standard deviation for optimization; all reported physical-unit metrics are inverse-transformed. Classification class maps are fitted once on training labels and frozen; unexpected test classes block ordinary closed-set evaluation.

For the no-clipping mechanism track, do not apply quantile transforms or soft clipping. In native-strength baselines such transforms may be part of a documented recipe; this creates a separate comparison, not evidence specifically about bounded interactions. Missing values, if unexpectedly present, trigger an ingest audit; do not silently fabricate a missingness-free dataset.

The current downloads were not inspected byte-for-byte during this research. Data access and checksum creation are WP02 obligations. Foundation-model pretraining/evaluation overlap remains a separate disclosure item; synthetic-pretraining descriptions do not exclude all benchmark-selection contamination.

## 8. Confirmatory evaluation and statistical decision rules

### 8.1 Primary claim, endpoint, and comparison

Register one primary practical question: **on the fixed three-dataset panel, does the frozen candidate obtain lower predictive loss than the validation-selected strongest eligible baseline at a common measured single-example inference ceiling?** This is a local budget-constrained claim, not a claim about all tasks or a new function class.

For each regression dataset, define normalized example loss as squared error in original units divided by the **training-only** target variance. For classification, use negative log probability of the true class divided by training-label entropy. Use natural logarithms. A zero variance/entropy blocks this endpoint and requires a recorded, pre-test amendment. Compute classification loss from stable log-softmax or validated native probabilities; record any numerical probability floor rather than silently clipping away failures.

Average example losses within each protected group, then average groups equally. Average losses across independently trained seeds; **do not average predictions and inadvertently create an ensemble**. The primary endpoint is the unweighted mean of the three dataset-normalized losses. Also report every dataset's physical-unit RMSE/MAE or classification NLL/accuracy, both row-weighted and group-weighted where they differ. The fixed panel's macro score is a transparent normalization convention, not an estimate over a population of all possible datasets.

For each dataset, establish the latency ceiling during development as the calibrated upper 95% confidence bound on complete batch-one latency of a conventional two-hidden-layer, width-128 GELU MLP using common preprocessing. Freeze its software, thread policy, precision, timing boundaries, and ceiling **before choosing the candidate operating point**. This reference is a reproducible deployment-budget anchor, not a claim that width 128 is optimal. The well-tuned MLP baseline remains separately eligible.

A candidate or baseline is eligible at the anchor only when its own calibrated latency upper bound is at or below that fixed ceiling. Choose one candidate configuration/export and one strongest baseline per dataset by minimum mean development group-normalized loss across the registered finalist pilot replicates among eligible configurations. Never select a best-performing seed as though it were a hyperparameter. Freeze the baseline identities; do not select the comparator from protected results. Include all successfully reproduced Tier A/B competitors in selection. If a required competitor cannot be reproduced, explicitly downgrade the claim to the named verified set before confirmation. Report the full development frontier and the predeclared protected operating-point frontier, not only the anchor. Protected evaluation includes only points and replicate counts frozen in the run manifest, not every exploratory search trial. Models outside the anchor remain visible.

A lower loss at a common ceiling is **not necessarily Pareto dominance** over the selected comparator: the candidate could use more of the permitted time. Claim pairwise dominance only when the appropriately adjusted uncertainty bounds support both lower loss and no increase in measured latency; otherwise describe the actual tradeoff. If no nonlinear candidate fits the ceiling, the practical anchor experiment fails feasibility, while the mathematical/mechanistic experiments can still proceed. An exported rank-zero winner supports affine sufficiency, not a benefit from interactions. Recheck the frozen ceiling during confirmatory timing; a final deployment that no longer satisfies it has not established the budget-constrained claim. Report the failed or uncertain cost guardrail without choosing a replacement after seeing protected results.

### 8.2 Development versus confirmation

The validation set may be used for debugging, tuning, pruning, budget calibration, and exploratory hypotheses. The protected test set must not be used for any of them. Splitting code can inspect group keys and labels only to establish the registered coverage/stratification rules; training, visualization, and model-selection code receive no test predictions or metrics before freeze.

Before any protected score is produced, write `artifacts/protocols/confirm_v1.json` and its SHA-256. It must resolve all conditional choices: exact dataset/split hashes; model and baseline revisions; selected architectures and pruning masks; tuning procedure and expenditure; final training/early-stopping rules; number of seeds; latency ceilings and timing protocol; primary endpoint; every inferential secondary contrast; confidence level; bootstrap procedure; failure handling; and the permitted interpretation. Store the source commit and dependency locks. A missing required field blocks confirmation.

Final training seeds must be independent of pilot seeds. Each final fit uses the original training set, with the frozen validation-based checkpoint rule, so training/validation roles do not change at the last minute. Do not refit on training plus validation unless that alternative, including epoch selection, was explicitly frozen for every applicable model. Complete the fixed run manifest, including failed runs, before revealing aggregate test results. After unblinding, changed hypotheses/configurations become exploratory and require fresh confirmation; rerunning seeds on the same test set is not a fresh holdout.

### 8.3 Independent units and uncertainty

For the core panel, the data unit is the protected **recipe group, airfoil-condition group, or duplicate-feature group**, not an individual timing repetition or a training seed. These recorded groups are a practical dependence control, not proof that all acquisition dependence has been observed. Interpret intervals conditionally on these datasets, training samples, selection procedure, and assumed exchangeability of groups within each task.

Retain per-example predictions and losses for every seed. For model pair A/B, first form within-group, seed-specific loss differences. For the primary interval use a **crossed paired bootstrap**:

1. Resample the final training-seed replicate IDs with replacement. Treat a replicate as the vector of fits for all three datasets and both compared procedures; never break its pairing. Independent seed streams for each dataset are generated from that replicate ID.
2. Independently within each dataset, resample protected groups with replacement, retaining all their examples and pairwise model results.
3. Average group differences for each sampled seed, average seeds, then average the three dataset values.
4. Use percentile endpoints for a two-sided 95% interval of candidate minus baseline loss. Save all bootstrap draws and the original estimate.

Start with 10,000 bootstrap draws and a fixed analysis seed. Check the Monte Carlo precision of the endpoints with independent analysis-seed batches; increase draw count if endpoint variation is more than 5% of the interval width. This controls numerical uncertainty of the analysis, not the required effect size. In degenerate zero-width cases, report exact equality or deterministic error rather than dividing by zero. Report a data-only group bootstrap and the separate across-seed standard deviation as sensitivity analyses.

This bootstrap is an approximate uncertainty procedure, especially with few groups or highly skewed losses. Before confirmation, validate interval behavior in the harness using simulated clustered data with known zero and nonzero mean differences and similar group-size patterns. If group counts are too small for stable interval behavior, use a declared descriptive result or acquire a genuinely more informative evaluation population; do not manufacture precision by increasing only optimizer seeds. No distribution-free finite-sample coverage is claimed.

A comparison across a large independent collection of tasks would call for a different task-level analysis; the literature explicitly distinguishes that setting. Do not perform a 3-dataset critical-difference ranking or count seeds as extra datasets. [R29] Claims about the entire learning algorithm under new training samples need independent training-sample replication or additional independent tasks; the core fixed-split intervals do not supply it.

### 8.4 Pilot-calibrated replication and sensitivity

Use eight independent **development-only** seed replicates for the shortlisted candidate/comparator procedures. This initial pilot estimates noise; eight is not the final evidential sample size. Let \(s^2_{seed}\) be the variance of the macro paired development-loss difference, and obtain a conservative upper variance estimate from the chi-square bound under an explicitly checked approximate-normal seed-effect model. If that model is implausible, expand the development pilot and use a nonparametric variance uncertainty estimate; record the choice before freeze.

For a normal seed-effect model with \(n_p\) pilot replicates, the one-sided 95% upper variance estimate is \((n_p-1)s^2_{seed}/\chi^2_{0.05,n_p-1}\).

Let \(SE_{data}\) be the development group-bootstrap standard error after averaging the pilot seeds. Choose the final count

\[
S=\max\left(8,\left\lceil\frac{s^2_{seed,upper}}{(0.25SE_{data})^2}\right\rceil\right).
\tag{13}
\]

The eight-replicate starting minimum avoids reducing the seed distribution to a two-point bootstrap; pilot diagnostics may require more, and eight alone does not guarantee coverage. The 0.25 criterion makes optimizer Monte Carlo error a small contribution relative to data uncertainty; it is **not** a minimum acceptable improvement. Also require enough seed replicates for the crossed-bootstrap diagnostics to be stable. If the expression is numerically degenerate, classify the task as deterministic or use a declared absolute numerical-resolution target instead. If the required fit count cannot be completed within measured local capacity, freeze a feasible count, report the resulting detectable-effect curve, and label unresolved differences inconclusive; do not lower the precision standard after seeing test effects.

For planning only, a two-sided normal approximation gives an 80%-power detectable effect near \(2.8SE\). Report the corresponding absolute losses and physical-unit changes where meaningful, along with its assumptions. There is no required percentage gain. If data uncertainty dominates, additional independent groups/tasks are more informative than more seeds. Teacher experiments use independent teacher instances as their replication unit; repeated samples from one teacher answer a narrower estimation question. Exact algebraic identities require numerical tests, not a significance test.

### 8.5 Multiple comparisons, failures, and interpretation

There is one primary candidate-versus-frozen-best-baseline contrast. For a predeclared secondary family of \(m\) contrasts, report simultaneous Bonferroni intervals at confidence \(1-0.05/m\), using the same paired resampling structure and enough draws to resolve the tails. A separate exploratory family is labeled exploratory without confirmatory significance language. Do not silently grow the family after inspecting results. Joint loss/latency dominance claims use simultaneous bounds across both quantities and any declared operating points, not two unrelated unadjusted positive findings.

Report every unsuccessful fit, including divergence, nonfinite predictions, unsupported operations, interruption, and resource exhaustion. Infrastructure interruptions can be resumed under §12.4. Genuine numerical divergence is a reliability outcome. An inferential comparison requiring finite loss is not completed by dropping failed seeds; if failures prevent a well-defined primary mean, report the primary predictive comparison as unresolved and compare reliability separately. A fallback deployment predictor may be evaluated only if that fallback rule was frozen and charged consistently for all methods. Zero observed failures is not proof of zero failure probability.

A confidence interval below zero supports an improvement on the stated fixed panel. An interval crossing zero is inconclusive unless it tightly excludes the scientifically relevant effect under consideration. A nonsignificant difference is not equivalence. An application-defined equivalence margin can be introduced only with a justified use case, before confirmation. Small genuine effects are reported with their costs; absent an application-specific utility threshold, do not call them practically transformative.

## 9. M3 Max feasibility and measurement specification

### 9.1 Discover the actual machine and resolve dependencies

**No M3 Max measurement has been executed for this report.** The core matrix/reduction workload is modest by construction, but actual acceleration, peak memory, and completion time remain local measurements.

The following are existing macOS/shell commands, not project commands:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
system_profiler SPHardwareDataType SPDisplaysDataType
vm_stat
df -h .
xcode-select -p
python3 --version
```

Save a redacted hardware report: do not include machine serial numbers or personal paths in public artifacts. Confirm `arm64` rather than accidentally running through Rosetta. Record actual unified memory, free storage, current memory pressure, CPU/GPU identification, power mode, attached-display state, and available developer tools. Do not assume the largest M3 Max configuration.

The inspected Apple setup page specifies a stable PyTorch 2.11.0 installation path with macOS 14 or later, Python 3.10 or later, and command-line tools. These are source-dated compatibility facts, not a guarantee for the user's installed system. Select a stable, mutually compatible arm64 Python/PyTorch wheel combination actually available at execution, preferably CPython 3.11 or 3.12 if all required packages support it; record the resolved versions and hashes rather than inventing a universal version lock now. [R18]

Use PyTorch as the primary stack because the candidate and most neural baselines can share tensor operations and optimizers. Use NumPy/SciPy for float64 references and small convex problems; pandas/scikit-learn for ingestion/preprocessing and validated estimators; `ucimlrepo` where its official metadata interface suffices; pytest, PyYAML, and psutil for testing/configuration/resource records. An example bootstrap command, to be executed only after compatibility resolution, is:

```bash
python3 -m venv .venv
source .venv/bin/activate
mkdir -p artifacts/environment locks
python -m pip install --upgrade pip
python -m pip install torch numpy scipy pandas scikit-learn \
    ucimlrepo pyyaml pytest psutil
python -m pip freeze > artifacts/environment/core-freeze.txt
```

The command does not pin unknown future releases. After smoke tests, produce an exact environment lock and a wheel/source manifest with package hashes and platform tags. Keep optional CatBoost/RealMLP/TabM/foundation dependencies in separate environments when their requirements conflict. Their adapters exchange schema-validated prediction artifacts; do not force incompatible packages into one environment. Install optional Excel readers only if the verified UCI files require them. No baseline may silently disappear because it conflicts with a convenience dependency.

Verify code licenses separately from model-weight/data terms. Preserve third-party notices. Do not redistribute raw datasets, checkpoints, or vendored code without satisfying their licenses. A public download link does not imply an open-source license. PAON and TabM license checks were performed; RAN and some other package details remain explicit local gates. [R05], [R06], [R17], [R22], [R23]

### 9.2 Blocking operation-level checks

Before training, exercise the actual chosen PyTorch build on MPS: dense linear forward/backward, broadcasting, multiplication, row reductions, `amax`, square root, reciprocal, clamp, vector norms, indexing during export, AdamW updates, serialization, and prediction transfer. Compare the candidate, raw/signed-square controls, and the required baseline paths against CPU float64 on fixed small tensors.

Use CPU double-precision gradient checks at nondegenerate inputs, including away from the clamp's transition. Then compare FP32 CPU/MPS values and gradients with relative/absolute tolerances calibrated to input/gradient scale; initial normal-scale tolerances of about \(10^{-5}\) for outputs and \(10^{-4}\) for gradients are diagnostic starting points, not permission to ignore systematic discrepancies. Test multiple steps and loss trajectories, not only a forward pass. Check the chosen norm implementation at zero directions; avoid an undefined square-root derivative that can generate NaN when a row becomes exactly zero.

Keep MPS fallback disabled/unset during support tests and performance claims. The documented fallback switch can redirect unsupported operations to CPU; such fallback must never be hidden inside a purported GPU timing. [R20] A faithful all-CPU implementation on the same laptop is an allowed fallback, with CPU results labeled as such. If an optional pretrained model lacks a faithful local path, exclude it explicitly rather than replacing its core operation with an unvalidated approximation.

Core comparisons use FP32 training/inference and CPU float64 references. Do not assume MPS supports float64. Mixed precision, quantization, compilation, and alternative kernels belong to separate optimized tracks after numerical checks; offer generally applicable optimizations to competitors. No custom Metal kernel is required initially. MLX or another backend is a conditional engineering alternative only after a documented MPS limitation and a parity plan; it is not a way to avoid baseline fidelity.

### 9.3 Memory model and numerical estimates

For one shared block, excluding any optional normalization parameters,

\[
P=od+o+2Kd+oK.
\tag{14}
\]

FP32 parameters, gradients, and two Adam moments account for approximately \(16P\) bytes. This is a **lower component estimate**, not peak process memory: optimizer temporaries, activations retained for backward, inputs, CPU copies, framework caches, and workspaces can dominate. A rough saved-activation model is

\[
M_{act}\approx4B\sum_l c_l(d_l+o_l+4K_l),
\tag{15}
\]

where \(c_l\) is a measured tape/implementation factor, not an assumed universal constant. Add resident dataset and preprocessing buffers, model-selection concurrency, loaded baseline contexts, and checkpoint-writing copies.

For illustration, a 16→256 lift, four width-256 blocks with \(K=64\), and a 7-output head contain 465,927 parameters. The parameter/gradient/two-moment component is 7,454,832 bytes (about 7.11 MiB). This is an arithmetic estimate, not a measured memory footprint. A Dry Bean-sized full quadratic design with 153 columns occupies 8,329,932 bytes in FP32 before solver workspaces. CPU float64 doubles that matrix storage. These estimates support a small local starting point, not a claim that the whole process uses only a few megabytes.

Measure process RSS, MPS `current_allocated_memory`, `driver_allocated_memory`, and recommended working-set size separately. These API quantities have different meanings; do not sum overlapping unified-memory accounts into a fictitious total. Driver allocation is not DRAM traffic. PyTorch exposes synchronization, allocator information, RNG-state functions, and profiling signposts, but their presence alone does not supply physical transfer counts. [R19]

Set a conservative execution budget from the minimum of installed-memory headroom, currently available memory, and the reported recommended GPU working set. An initial guard such as 60% of installed memory and 80% of currently available/recommended memory is a **safety policy**, not an assertion about usable hardware capacity. Revise from observed pressure without disabling allocator safeguards. Run one training job at a time initially. Optional large checkpoints are loaded only after file size and context/workspace probes demonstrate headroom; available disk must also cover raw assets, two atomic checkpoint copies, logs, and the frozen dataset/prediction artifacts.

### 9.4 Calibration and stage resource forecasts

For each representative family/shape, measure cold startup, first forward/backward, steady training-step time, validation time, and export/prediction costs. Explore batches 32, 64, 128, 256 and larger only while memory is safe and throughput improves. Keep batch-one latency measurement separate. Do not force all algorithms to use the same physical batch if it makes native execution unfair; matched-learning comparisons must nevertheless preserve effective batch, update count, and loss weighting.

Gradient accumulation is allowed for a batch-independent model when it reproduces the same effective batch and properly weights the final partial microbatch. It is not assumed equivalent with batch-dependent normalization or a different optimizer-step schedule. Record the exact adaptation. Lowering model width/rank is a changed configuration, not a faithful substitute for a larger frozen model.

Forecast stage cost from measured quantities:

\[
T_{stage}=\sum_{runs}\left[E\,\lceil N_{train}/B\rceil t_{step}
+T_{validation}+T_{startup}+T_{export}+T_{analysis}\right].
\tag{16}
\]

Use actual horizons, seeds, search trials, early-stopping distributions, failed-run costs, and uncertainty ranges. Do not advertise an invented number of laptop hours. Core assets are small; broad model selection and replication, rather than parameter storage, may dominate expenditure. If a stage's calibrated cost is excessive, choose the cheapest uncertainty-resolving branch in the work packages, or mark the desired claim locally unresolved. Longer runs are acceptable when their information value and measured capacity justify them.

### 9.5 Latency, throughput, and communication measurements

Define two boundaries and never mix them:

* **Primary complete prediction:** already-loaded deployed model receives a CPU-resident raw feature batch; includes fitted preprocessing, encoding, transfer when needed, inference, output transfer, and probability/target transformation. Training and model loading are excluded but reported separately. Batch-one milliseconds is the primary cost.
* **Secondary device-resident kernel:** prepared tensors and model are resident; include all candidate/baseline inference operations but exclude ingress/egress. Report this as a diagnostic, alongside batch-256 examples/second and the calibration batch range.

For pretrained in-context models, report cold fit/context construction, cold prediction, warm cached prediction, retained context bytes, and cache invalidation conditions. A warm cache is not a free preprocessing operation; show amortization over query counts. For every family distinguish cold process/model startup from steady use.

Use `time.perf_counter_ns`, inference/evaluation mode, and explicit `torch.mps.synchronize()` immediately before and after the measured device work where applicable. Force outputs to be materialized. Warm up until consecutive rolling timing summaries stabilize relative to the chosen measurement precision; log the warm-up history rather than assuming a fixed number is sufficient. Batch multiple repeated calls inside a timed block when timer overhead is material, but preserve true independent-query semantics and charge preprocessing each time in the primary boundary.

Interleave candidate/baseline order randomly within a session. Repeat across separate sessions/restarts, logging power mode, temperature/throttling indicators when available, competing load, and process thread count. Use session-level paired summaries for latency uncertainty; thousands of calls in one session are not thousands of independent machines. Select the final number of sessions from development variability and freeze it. Report mean, median, p95, sample counts, and intervals; confidence intervals for mean latency do not automatically characterize the p95 tail. Keep raw durations. Exclude only predeclared extraneous interruptions, retaining their logs; do not discard slow but valid thermal behavior.

Estimate affine/product/normalization contributions by profiling and corresponding controlled variants. If a fraction \(p\) of total inference time is improved by factor \(s\), the end-to-end upper model is \(1/(1-p+p/s)\); use measured \(p\), not a parameter-count proxy. This Amdahl-style deduction explains why kernel-level improvements may have little complete-prediction value.

For the dendritic alternative, count theoretical reads, writes, intermediate tensor sizes, and branch messages as **proxies**. Claim actual memory-traffic reduction only if a supported hardware counter/profiler provides attributable byte counts with validated semantics. No available counter means the hardware-traffic claim remains untested; measured latency can still be reported. Unified memory, contiguous grouped operations, and GPU scheduling are part of the experiment, not proof of physical decentralization.

## 10. Risk-ordered work packages and decision gates

All commands beginning `python -m neuron_poc` below are **proposed project commands to implement**, not claims that a package or executable already exists. The repository/interface contract is in §12. Work packages may share infrastructure, but each gate must leave a machine-readable decision record.

### WP00 — Discover and freeze an executable local substrate

**Objective/risk:** eliminate unsupported hardware, dependency, and licensing assumptions before expensive assets or training. **Entry:** this plan and the target laptop; no earlier project artifacts required.

**Actions/files:** create the repository skeleton and CLI; collect §9.1 discovery into `artifacts/environment/hardware.json`; resolve core dependencies into `locks/core.txt` plus a hash manifest; implement `discover` and `check --suite backend`. Perform the MPS operation checks, CPU fallback checks, and a ten-update representative training/serialization test. Create `artifacts/licenses/registry.json` with code/data/weight licenses, notices, unresolved terms, and permitted local use. Do not download foundation weights yet.

**Outputs/tests:** actual memory/storage in bytes, software and architecture versions, device flags, per-operation pass/fail and numerical differences, resolved package imports, a saved/reloaded prediction parity result. Record actual peak-memory/time measurements for the smoke test. Require a functioning CPU path even when MPS passes, because it is the correctness reference.

**Resources:** only the core environment and small tensors; record installation footprint and measured smoke-test peaks. **Decision:** pass→WP01/WP02. MPS failure with faithful CPU success→continue explicitly on CPU and calibrate scope. Critical CPU failure→repair the minimal environment; do not develop higher layers. Ambiguous third-party terms→isolate that baseline and flag its claim restriction, not automatic permission to reuse.

### WP01 — Establish mathematical and implementation fidelity

**Objective/risk:** test the indispensable mechanism, exact equivalences, initialization, and negative limits. **Dependencies/entry:** WP00 CPU path; equations (1), (4)–(11), Appendix A.

**Actions/files:** implement `models/bir.py`, a literal scalar reference, signed-square conversion, explicit quadratic feature construction, and `tests/test_algebra.py`, `test_gradients.py`, `test_limits.py`. Run `check --suite mathematics`. Reproduce Appendix A; add zero input, zero rank, zero/tiny direction norms, unequal dimensions, negative coefficients, cancellations, extreme finite inputs, and affine-stack folding tests. Test both normal and intentionally bad initialization. Verify spectral deletion certificates against adversarial eigenvectors, not only random samples.

**Outputs/tests:** one JSON result per invariant with tolerance, dtype, input scale, seed, measured error, and environment; exact formula conversions; first/second-update gradient traces. Pass means algebraic/gradient tests agree within justified numerical tolerance and known failure controls fail as predicted. The original temporal XOR formula must fail the four-case truth table and the corrected one must pass.

**Resources:** small matrices and CPU float64; no dataset/model-selection expense. **Decision:** algebra/gradient failure→fix or amend before learning experiments. Correctness plus structural limitations→retain the limitations as boundary tests, not reject all finite-domain learning. Exact polynomial equivalence→continue as a normalization/allocation study with no expressivity novelty claim.

### WP02 — Acquire, license, and freeze the evaluation population

**Objective/risk:** prevent data leakage, unstable grouping, unavailable assets, and invalid population claims. **Dependencies/entry:** WP00 environment/terms check. Can run alongside WP01; no predictive results are needed.

**Actions/files:** implement `prepare-data --suite core --config configs/data/core.yaml`; retrieve the three UCI datasets; save raw bytes and metadata; verify feature/target semantics and CC BY obligations; generate immutable row and group IDs. Group exact parsed numerical tuples without unregistered rounding. Audit near-duplicates and alternative physical grouping on training metadata before freeze. Use a seeded greedy group assignment targeting the stated 60/20/20 proportions and Dry Bean class balance; break ties by seeded ordering, never loss. Save complete allocations rather than relying on future library behavior.

**Outputs/tests:** `artifacts/provenance/data.json`, license notices, raw checksums, `artifacts/splits/core_v1.*`, class/feature schema, rows and groups per split, group-size distribution, missingness/conflicting-label report, leakage tests with zero shared groups. Store protected targets separately; loaders used by tuning cannot return them. The guard is an auditable workflow restriction, not a claim of cryptographic secrecy for public datasets.

**Resources:** measure downloaded byte counts and parser peaks; these core tables do not justify a large-model download. **Decision:** valid group coverage→WP03. Download/parser/metadata mismatch→resolve from source bytes and log it. Unusable license→do not use the asset; select a pre-test replacement with the same role and record an amendment. Too few independent groups→retain descriptive/mechanistic scope or register a new population before model evaluation; do not silently substitute an easier random split. Leave Bike test data unopened for the later transfer stage.

### WP03 — Build an adversarially checked harness and faithful baselines

**Objective/risk:** establish that the experiment can detect a real difference and cannot invent one. **Dependencies/entry:** WP01 mathematical pass, WP02 fixed core data, WP00 baseline terms resolved for implementations being used.

**Actions/files:** implement adapters, training, metrics, checkpointing, and `reproduce --suite core-baselines`. Fit affine and explicit quadratic baselines first; check solver convergence and constructed teachers. Implement/resolve every Tier A/B family in §6, preserving native and common-preprocessing modes. For third-party code, record the source revision, entry point, resolved constructor/defaults, and local changes. Independently reimplement published RAN equations when source reuse is not licensed; do not assert reproduction of author benchmark results.

Add harness controls: identical predictions through two adapter names must produce exactly zero paired loss; a deliberately better known predictor must improve the endpoint; shuffled training labels must not retain teacher predictability; fitted transformations must be unchanged by modifications to protected data; all models must preserve row/class order; CPU/MPS and serialization parity must pass. Test interval behavior on simulated clustered losses. Test native ensemble aggregation for TabM and native context/cache semantics for any admitted foundation model.

**Outputs/tests:** baseline reproduction cards, per-family loss curves, convergence statuses, preprocessor schemas, failed-case fixtures, interval-simulation report. Reproduction means equation/implementation/configuration fidelity and sensible diagnostics, not matching incomparable published scores. Native default recipes must remain included even when a custom tuned alternative is faster.

**Resources:** start with synthetic and training/development subsets solely for correctness; promote to complete training data after passing. Measure each representative family's memory and training cost. **Decision:** harness bug→block all rankings. Missing strongest runnable baseline→repair or narrow claims before freeze. A mathematically identical method reporting a gain→stop and audit measurement/selection, not celebrate.

### WP04 — Integrate the minimum candidate and physical export

**Objective/risk:** implement the original causal change without hiding architectural or deployment costs. **Dependencies/entry:** WP03 working interfaces and WP01 invariants.

**Actions/files:** implement `ain_literal`, `bir_shared`, `bir_raw`, `bir_constant`, direct and stack architectures, penalty schedules, and compact exporter. Add `tests/test_export.py`, parameter/MAC accounting, spectral/branch bounds, and per-layer gradient/energy logging. Implement compact-from-scratch and generic MLP/PN structured-pruning controls. `export --run-id ... --selection validation` must create independent contiguous deployment tensors and record their parent checkpoint.

**Outputs/tests:** prediction parity before pruning; correct nonzero error/bounds after pruning; exact rank-zero affine export and affine folding; no dead-branch evaluation in a traced/profiled forward; correct stored parameter counts and reproducible initialization. A group penalty must not be treated as exact zero sparsity. Confirm removal speed by measurement later, not by an operation-count assertion.

**Resources:** cheap synthetic/direct fits first; no broad architecture sweep. **Decision:** implementation correct→WP05. Shared model loses a literal teacher that independent-output factors represent→classify as the explicit sharing restriction and include the literal control; do not conflate it with failure of normalization. No ability to physically shrink tensors→block compression claims until repaired.

### WP05 — Calibrate resources, explore mechanisms, and finalize search

**Objective/risk:** determine whether the hypothesis is testable and which uncertainty deserves further computation. **Dependencies/entry:** WP03/04, no protected scores.

**Actions/files:** implement `pilot --config configs/explore/core.yaml`, `benchmark --stage development`, and the fair selection-compute accounting. Run the synthetic suite with independent teachers, controlled ranks, rotations, input scales, and nuisance energy. Establish the native baseline recipes, optimization ranges, training horizons, batch/thread choices, and inference anchor. Inspect raw/bounded initialization-scale controls. Run the eight-seed development variance pilot on shortlisted procedures; use §8.4 for final replication. Use measured budget-doubling curves to finalize HPO effort rather than an arbitrary trial count.

If promising, evaluate the nested physically pruned exports and compact-from-scratch controls; count every selection/fine-tuning run. Probe optional foundation models only after checkpoint license/size and CPU/MPS support gates. Write `artifacts/protocols/pilot_report.md`, calibration tables, learning curves, and a complete tuning ledger.

**Outputs/tests:** per-family measured peak memory/time; identifiable causal contrast or a specific confound; a proposed finite confirmatory manifest with resource forecast and detectable effects. At least one known representable teacher must be learned by a functioning method; if none is, the task/optimizer/harness is invalid as a discriminator.

**Resources:** equation (16), actual model-selection history, sequential jobs, checkpointing. Allocate extra runs to unresolved discriminating questions, not repeated favorable teachers. **Decision:** plausible local headroom and adequate precision→WP06. Synthetic-only success with domination on development real data→retain a narrow mechanistic claim or stop practical escalation. Underpowered data→revise the evaluation before freeze, not add seed pseudoreplicates. Resource limit→document which claim is untested; a larger-scale hypothesis survives only with a concrete scale-dependent prediction and supportive local scaling evidence.

### WP06 — Freeze the confirmatory experiment

**Objective/risk:** remove adaptive-selection ambiguity. **Dependencies/entry:** all mandatory correctness/reproduction gates passed, development search complete, feasible calibrated run manifest, pre-test audits completed.

**Actions/files:** implement `freeze --config configs/confirm/core.yaml`. Validate every field in §8.2; resolve baseline identities, latency-eligible exports, final seeds, statistical contrasts, resource forecasts, failure and amendment rules. Create signed-by-hash protocol/config/split/environment manifests and a run list with expected artifacts. Run the three audits in §13 against the actual implementation; log defects and changes.

**Outputs/tests:** immutable protocol hash, source commit, no unresolved primary-critical dependencies, and a dry run of the entire report pipeline on mock predictions. **Resources:** metadata and mock analysis, not expensive training. **Decision:** pass→WP07. Any undefined comparator/endpoint/seed count or unresolved fidelity gate→return to the appropriate development package. A knowingly omitted baseline must have its consequence for the claim written into the protocol, not a hidden post-hoc footnote.

### WP07 — Execute the controlled comparison once

**Objective/risk:** obtain unbiased protected results under the frozen procedure. **Dependencies/entry:** WP06 protocol hash and locked inputs.

**Actions/files:** implement `run --protocol artifacts/protocols/confirm_v1.json`; train the fixed replicate manifest; export the fixed selected structures; save per-example raw predictions and losses. Benchmark frozen deployments with randomized paired sessions. The runner verifies hashes and prevents configuration drift. Complete failed-run records and permitted resumes before `analyze --protocol ... --unblind` releases aggregate protected results.

**Outputs/tests:** all scheduled run IDs accounted for; predictions in original units or ordered logits/probabilities; per-group/per-seed loss differences; bootstrap draws/intervals; full inference distributions and cost summaries; no unexplained missing cells. Verify result reproduction from stored predictions without retraining.

**Resources:** measured forecast from WP05, dynamically monitored memory safety, no concurrent jobs beyond the calibrated concurrency. **Decision:** positive primary result with passed guardrails→WP08 and potentially WP09. Negative finite result→report it and investigate already-frozen attribution contrasts. Missing precision→inconclusive with achieved sensitivity. Divergence/harness failure→use the predeclared failure classification; do not replace failed configurations with winners after test access.

### WP08 — Attribute, independently rerun, and challenge the gain

**Objective/risk:** distinguish normalization/allocation from extra capacity, search, preprocessing, and measurement artifacts. **Dependencies/entry:** WP07 complete; all inferential ablations already registered, new probes explicitly exploratory.

**Actions/files:** run the frozen ablation matrix in §11; inspect spectral changes and nuisance-energy sensitivity; compare pruned versus compact-from-scratch models and generic baseline pruning; verify complete-cost versus device-only results. Reproduce selected results from a clean environment and saved manifests, using independent final seed streams where registered. Run interruption/resume parity on CPU and check MPS numerical consistency. Store `artifacts/reports/attribution.md` and `replication.md`.

**Outputs/tests:** which causal contrast remains after controls; measured rather than masked compression; matching hash/provenance chains; separate engineering reproduction and new-data confirmation labels. A clean rerun on the same data is not independent evidence of population generalization.

**Resources:** prioritize decisive controls based on the registered hypotheses; all new adaptive checks retain exploratory status and cost records. **Decision:** surviving attributed gain→WP09 or a bounded component conclusion. Loss of gain under a stronger control→revise the explanation/claim, possibly stop. Equality with the normalized PN baseline→no distinctive neuron contribution; report any useful implementation or negative finding without relabeling it.

### WP09 — Test fresh application transfer when justified

**Objective/risk:** determine whether a frozen useful mechanism extends beyond the development panel. **Dependencies/entry:** WP08 identifies a meaningful mechanism or tradeoff; candidate/search policy frozen, Bike protected outcomes untouched.

**Actions/files:** acquire and audit UCI Bike data under §7.3; register `confirm_transfer_v1.json` before protected prediction. Carry the candidate recipe and equal selection-compute policy forward; tune only on the chronological validation period. Reproduce eligible native baselines, charge preprocessing/caches, and freeze day-block bootstrap length from training/validation autocorrelation. Evaluate the final quarter of 2012 once.

**Outputs/tests:** target-leakage exclusion tests, chronological split hashes, available-at-prediction feature declaration, block-level uncertainty, performance/cost and count-regression diagnostics. Seasonal test shift must remain visible.

**Resources:** small raw data but a fresh set of calibrated fits; no assumption that core-panel winners need less baseline tuning here. **Decision:** consistent gain→claim evidence on this additional named task, not all streaming/time-series reasoning. No transfer→narrow scope and report the failed confirmation. Changed post-test mechanism→requires another genuinely fresh protocol/data source, not further tuning on this quarter.

### WP10 — Synthesize and choose the next direction

**Objective/risk:** preserve negative evidence and prevent open-ended search for a favorable outcome. **Dependencies/entry:** completed or explicitly blocked preceding stages.

**Actions/files:** implement `report --protocol ...`; generate the fixed results tables, claim/evidence ledger, cost frontiers, failure register, and final `DECISIONS.md`. Each statement must identify whether it is a proof, measured result, estimate, exploratory association, or confirmed comparison. Reconcile unfavorable tasks and regressions; list excluded baselines and untested scales.

**Outputs/tests:** reproducible report built from immutable artifacts; every primary/secondary planned cell present or explained; no superiority claim without its required baseline/metric/gate. **Resources:** analysis from stored results, not additional unregistered tuning. **Decision:** continue, narrow, redesign, or stop according to §11.3. No automatic transition to a bigger architecture because the original formula has an appealing name.

## 11. Attribution, adversarial cases, and evidence-triggered alternatives

### 11.1 Required ablation and sensitivity matrix

The matrix specifies contrasts, not an instruction to perform an uncontrolled full Cartesian product. Use exact/cheap tests first; select and freeze inferential contrasts at WP06.

| Question | Discriminating comparison | Interpretation if the apparent gain disappears |
|---|---|---|
| Is bounding itself useful? | Bounded versus raw versus constant denominator; equal initial feature variance and tuned coefficient step scales | Scale conditioning, not a special energy-dependent mechanism |
| Is the contribution a new function family? | Exact quadratic features, trained bounded signed-square PN, exact converted checkpoints | Polynomial parameterization/optimization explains it; no new expressivity |
| Is normalization merely preprocessing? | Same transformed interaction features supplied to established models, with original bypass retained | Generic input map explains the gain |
| Is the denominator effectively constant? | Energy histograms, no-LayerNorm versus LayerNorm, frozen denominator fitted on training only | Do not attribute the result to adaptive energy response |
| Does sharing help or just restrict capacity? | Independent-output literal model versus shared bank, actual parameter/time frontier | State the multi-output sharing tradeoff rather than a neuron-wide property |
| Does selective allocation pay? | Unpruned, physically pruned, equal-shape trained-from-scratch; generic MLP/PN pruning | Generic architecture selection or compression, not special learned allocation |
| Is coefficient sparsity meaningful? | Unit directions versus invariant path penalty versus intentionally flawed gamma-only penalty | Factor scaling or cancellation explains “sparsity” |
| Is optimization accidentally disadvantaged? | Zero versus small nonzero interaction head; coefficient warm-up; separately tuned raw/bounded scales | Initialization/training policy, not representational superiority |
| Does irrelevant energy harm learning? | Useful-coordinate-fixed nuisance growth versus total-energy-fixed inputs; rotations | Identify the denominator's domain limitation |
| Does global boundedness improve reliability? | Large finite inputs, raw target-rescaled ground truth, bypass norm monitoring, finite-gradient rates | Bounded residual alone does not imply safe total-network behavior |
| Does the architecture rely on odd or higher-order capacity? | Additive odd and cubic teachers, direct versus biased stacks | Keep direct-function limitations distinct from deep approximation |
| Does speed survive deployment? | Physical compact export, optimized CPU baselines, complete versus device-only timing, batch-size curves | Kernel/launch/preprocessing overhead erases the practical gain |

Also test near-collinear factors, large positive/negative branch cancellation, constant/zero-variance features, branch permutations, reparameterized checkpoints with equal functions, input dimension changes at fixed useful signal, and very small datasets where a convex quadratic baseline can dominate a nonconvex factorization. A high-confidence mathematical certificate can be loose for real inputs; do not replace empirical loss with the certificate.

### 11.2 Conditional alternative D — grouped dendritic computation

**Trigger:** global energy demonstrably suppresses useful local interactions, or profiling shows a plausible locality/communication bottleneck. A merely negative AIN result is not sufficient evidence for the dendritic direction.

**Entry and implementation:** first freeze a local-interaction hypothesis and budget. Implement equation (2) with disjoint contiguous groups using reshape and batched dense matrix multiplication. Save the exact group partition; include equal-size random groups and a permutation that preserves mathematical function but changes access pattern. Test synthetic targets with known within-group versus cross-group interactions. Biases, overlaps, routing, or learned grouping are separate variants.

**Mandatory controls:** identical group connectivity with affine branches; grouped ReLU/GELU branches; grouped raw multiplication; dense versions with matched prediction budgets; equal theoretical connectivity and matched *actual* latency frontiers. Simply comparing sparse dendrites with a much larger dense layer cannot attribute a nonlinear/locality benefit.

**Outputs/resources:** `models/dendritic.py`, shape/group invariants, memory-access proxy counts, actual exported latency, measured activation peaks, and hardware-counter byte counts only if available and validated. Calibrate group sizes geometrically and stop expansion when it no longer distinguishes locality or crosses measured capacity. If no valid traffic counter exists, say “modeled communication plus observed latency,” not measured communication reduction.

**Gate:** pursue architecture work only if an accuracy–cost gain survives equally grouped conventional controls, or a precisely validated local communication mechanism remains promising at a scale the laptop can expose. Otherwise report no local advantage. Physical inter-device communication or a custom distributed architecture remains unverified on one unified-memory laptop. The cited dendritic paper's detailed evidence was not fully accessible during this investigation, so its claimed speedups are not inherited by this design. [R11]

### 11.3 Conditional alternative T — minimal temporal state

**Trigger:** a concrete streaming use case requires persistent low-state processing, or a properly scoped component study seeks to test the corrected temporal mechanism. Do not advertise sequential reasoning from the static tabular experiment.

**Implementation:** retain the literal equation (3) and a separately named corrected version with direct state readout. Generalize only explicitly to a small diagonal bank: \(m_t=\Lambda m_{t-1}+(I-\Lambda)Px_t\), with affine/bilinear readout from \((x_t,m_{t-1})\). Parameterize each \(\lambda\) inside \([0,1)\); guard against FP32 rounding to one and record the guard. Compute output before state update, reset state at every independent episode, and prohibit future inputs in preprocessing. Fixed \(\lambda=e^{-1/\tau}\) on a geometric timescale grid is an interpretable control; learned persistence is a separate comparison.

For zero initial state, the scalar memory is exactly an exponentially weighted sum of past projections. With bounded projected inputs it stays bounded for \(0\le\lambda<1\), but the input-state product and output weights can still grow. Expanding the sum makes the readout a restricted second-order temporal interaction/Volterra form. Thus an ordinary exponential filter bank with the identical bilinear readout is an **exact functional baseline**, not just a weak linear recurrent competitor. Linear recurrent and state-space literature supplies broader context. [R14]

**Tests/controls:** four-case temporal XOR; delayed bit comparison; geometric delay lengths and sequence lengths; several simultaneous independent bits; streaming anomaly generators with known event rules. Compare affine readout on the same memory, the identical filter-bank bilinear readout, a tiny GRU, and a diagonal linear recurrence with sufficient matched state. Give every model the same current/past observations; report state bytes, total parameters, training cost, episode accuracy/loss, and single-stream per-step latency, not only highly batched throughput.

**Gate:** exact filter-bank equivalence defeats a new-memory-method claim. Small-state success restricted to short delays supports only that component claim. If required state growth eliminates the cost advantage, stop the efficiency claim. If synthetic success warrants a real streaming task, acquire and register that task's causal feature availability, reset policy, splits, and baselines **before** making a real-world streaming claim; no such task is silently supplied by this plan's static panel.

### 11.4 Outcome decision table

| Observed outcome | Required next action |
|---|---|
| Broken algebra, gradient, split, or timing harness | Repair; invalidate affected runs. This is not evidence against the idea. |
| Teacher outside the function family or insensitive proxy | State why the proxy cannot identify the proposed mechanism; use the registered faithful teacher before concluding. |
| Local memory/kernel limitation | Report the blocked claim; use a faithful CPU path or explicitly narrower scale, not an undocumented approximation. |
| Bounded models beat raw models, normalized PN matches candidate | Normalization may be useful; no distinct neuron advance. Reuse the simpler model. |
| Candidate wins only at larger prediction cost | Report the frontier/tradeoff, not overall superiority. |
| Pruning reduces parameter count but not complete latency | Storage/compression result only; no speed claim. |
| Small stable real-task effect with uncertainty excluding zero | Report the effect and costs without an arbitrary minimum percentage or “breakthrough” label. |
| Interval wide or method unreliable | Inconclusive predictive comparison; report achieved sensitivity and failure modes. |
| Fair strong baselines dominate across tested regimes | Stop practical escalation of this exact design; retain the negative finding and mathematical characterization. |
| Attributed component gain with locally observed favorable scaling | Escalate only the dimension/rank/task range needed to test that scaling prediction; retain baseline and full-cost controls. |
| Claimed benefit requires unavailable physical distribution | Record a locally testable proxy and a separate unverified larger-system hypothesis; do not claim hardware validation. |

## 12. Repository, interfaces, reproducibility, and reporting

### 12.1 Minimal repository layout

```text
neuron-poc/
  README.md
  research_poc_plan.md
  pyproject.toml
  DECISIONS.md
  locks/                         # exact core/optional environment locks
  configs/
    data/core.yaml
    explore/core.yaml
    confirm/core.yaml
  scripts/mathematical_checks.py # copy/adapt Appendix A
  src/neuron_poc/
    __main__.py                  # command dispatcher
    cli.py, types.py, registry.py
    train.py, export.py, checkpoint.py
    metrics.py, statistics.py, benchmark.py, report.py
    models/
      bir.py, polynomial.py, mlp.py, rational.py, crosses.py
      temporal.py, dendritic.py # only when their gates activate
    adapters/                   # native package/environment wrappers
    data/synthetic.py, uci.py, splits.py
  tests/
    test_algebra.py, test_gradients.py, test_limits.py
    test_leakage.py, test_export.py, test_resume.py
    test_harness.py, test_backend.py
  data/raw/                     # immutable local source bytes, not committed by default
  artifacts/
    environment/, licenses/, provenance/, splits/
    protocols/, runs/, timings/, predictions/, decisions/, reports/
```

Commas in the tree mean separate files, not literal comma-containing filenames. Core numerical code should remain small; do not build a distributed experiment service or a general-purpose training platform. Preserve enough organization to resume from artifacts without chat memory.

### 12.2 Shared interfaces and configuration contract

Use one task specification with fields: `task_id`, `task_kind` (`regression`/`multiclass`), ordered feature names/dtypes, ordered class names, original target units, row IDs, group IDs, split IDs, and fitted preprocessing provenance. Inputs are two-dimensional arrays; scalar regression predictions are consistently `(N,1)`, not a mixture of `(N,)` and `(N,1)`. Classification raw predictions are `(N,C)` in frozen class order.

Every adapter implements this conceptual contract; these are **project interfaces**, not assumed third-party package APIs:

```python
class EstimatorAdapter:
    def fit(self, train, validation, *, seed: int, config: dict,
            budget: dict, artifact_dir: str) -> dict:
        """Fit only allowed inputs. Return status, provenance, costs, checkpoint."""

    def predict_raw(self, X, *, context: dict | None = None):
        """Return ordered raw scores or original-unit regression values."""

    def export(self, destination: str, *, selection: dict) -> dict:
        """Write a deployable artifact, preprocessing, shape, and cache contract."""

    def load(self, artifact: str) -> None:
        """Restore the exact estimator and preprocessing state."""
```

Specify `raw_score_kind` explicitly: logits, log probabilities, probabilities, or regression value. The common metric adapter must not softmax native probabilities a second time. Validate finite outputs, row counts/order, class ordering, and inverse target transformation. Methods with native ensembles average according to their own frozen recipe; ordinary seed repetitions are separate fitted models, not a free ensemble. Do not deserialize untrusted pickle/checkpoint files without an appropriate safe format and provenance check.

Configuration files must include: stage; parent protocol; method ID/version; native/common track; exact architecture; denominator and scale convention; precision/backend; initialization; optimizer parameter groups; regularization and warm-up; training batch and accumulation; epoch/early-stopping policy; HPO sampler/search range/budget; pruning/export rule; dataset/split hashes; seed namespace; timing/cache boundary; analysis family; and failure policy. Unknown keys are errors. Canonicalize config serialization and compute its hash. Mathematical changes cannot be hidden behind an unchanged method ID.

Use deterministic seed namespaces derived from a cryptographic hash of `(master_seed, stage, task, method, trial, replicate, purpose)`, not Python's process-randomized `hash()`. Separate data, model, minibatch order, teacher generation, bootstrap, and timing-order seeds. Preserve full generator states for resumption, because a seed alone does not specify how far a generator has advanced.

### 12.3 Command behavior and artifact schemas

Implement only the commands needed by the packages: `discover`, `check`, `prepare-data`, `reproduce`, `pilot`, `freeze`, `run`, `export`, `benchmark`, `analyze`, and `report`. Each accepts `--help`, an explicit output directory, and a configuration or protocol where applicable. A stage may invoke the common trainer rather than duplicate code. `run` requires a frozen protocol for confirmatory mode; `analyze --unblind` verifies run completeness and hashes. Exploratory analysis must never write to the confirmatory result namespace.

Append one JSON object per event to `artifacts/runs/events.jsonl`, plus immutable summaries. Required run fields are:

```text
run_id, parent_run_id, stage, protocol_hash, config_hash,
method_id, source_revision, environment_lock_hash, hardware_id,
dataset_hash, split_hash, preprocessing_hash, task_id,
trial_id, replicate_id, seed_namespace, rng_state_artifact,
status, failure_category, failure_detail, start_utc, end_utc,
training_seconds, selection_seconds, validation_seconds,
parameter_count, stored_bytes, retained_context_bytes,
process_peak_rss_bytes, mps_live_bytes, mps_driver_bytes,
device, dtype, thread_count, batch_size, accumulation_steps,
checkpoint_path, checkpoint_hash, export_path, export_hash,
prediction_path, metric_schema_version, resume_parent
```

Store absent/inapplicable measurements as null with a reason, not zero. Units belong in field names or the metric schema. For each prediction retain `task_id, row_id, group_id, split_id, replicate_id, target, prediction/logits, loss`. Use Parquet or NPZ plus JSON schema; do not discard raw predictions after aggregation. For timings retain session/call IDs, order, boundary, cache state, warm-up flag, durations in nanoseconds, environment conditions, and any exclusion reason. Save analysis seeds and resampled statistics.

For dependency/data/model provenance record version or source commit, retrieval URL/date, license, SHA-256, and local modifications. A source URL plus “latest” is not a reproducible identifier. Container downloads of source snapshots were not available in the research environment, so no upstream hashes are supplied here; the agent must resolve and save them before confirmatory use. References below identify what was inspected, not executable locks.

### 12.4 Checkpointing, recovery, and agent authority

Write checkpoints atomically: temporary file on the same filesystem, flush/fsync when appropriate, then rename; retain the previous complete checkpoint until replacement succeeds. Save model and optimizer state, schedules, current epoch/update, sampler position, RNG states (Python/NumPy/CPU/device where supported), preprocessing, architecture and retained masks, configuration/protocol/data hashes, and completed timing/selection costs. A mid-epoch resume without the correct sampler/RNG state is not an exact continuation.

Test interrupted versus uninterrupted CPU training for identical continuation under a deterministic fixture. Test MPS continuation for justified numerical agreement; bitwise deterministic behavior is not assumed. On interruption, resume only if all hashes, software, and state are consistent. A changed environment requires a labeled restart or engineering-replication run, not an invisible continuation. Preserve failed prefixes and charge their project-specific expenditure. Never erase an unfavorable run by assigning its run ID to a replacement model.

The agent may make routine non-destructive choices—code organization, safe parser fixes, equivalent vectorization, logging formats—while documenting them. Changes to hypothesis, denominator, task population, feature availability, primary metric, comparator eligibility, precision, search budget, statistical procedure, or interpretive scope require a methodological amendment. After test access, amendments cannot produce a new confirmatory claim on that same holdout. Store decisions with date, evidence, affected run IDs, authority, and next action in `DECISIONS.md` and `artifacts/decisions/`.

### 12.5 Required result report

The generated report must contain: executed versus blocked work-package status; dataset and baseline provenance; environment and fidelity results; all planned predictive results with uncertainty and failures; development search expenditure; protected loss–latency frontiers; model/context/storage and training costs; mechanism attribution; unfavorable tasks and boundary cases; reproducibility/recovery results; and an evidence-to-claim table. Report absolute effects and meaningful relative effects; omit relative ratios when the denominator is near zero.

Separate three panels: exact correctness/mechanism evidence; scratch-trained local models; externally pretrained/native methods. A combined deployment frontier is useful but does not erase differences in pretraining and retained data. State plainly whether the original equation, an explicitly modified version, or a conventional baseline performed best. Include a minimal reproduction command sequence with the actual frozen paths. Do not suggest that public code alone establishes independent replication.

## 13. Audit record, revisions, and remaining uncertainty

These were **self-audits during research and drafting**, not independent peer review. Small CPU numerical checks were executed; the full experiment and target-laptop checks were not. The following material issues were found and incorporated into this plan.

| Audit | Defect or adversarial explanation | Revision and acceptance criterion |
|---|---|---|
| Scientific/prior art | The original neuron is a known quadratic function class after a deterministic input map. | Add exact reduction, bounded signed-square/convex quadratic baselines; prohibit new-expressivity claims. |
| Scientific/prior art | Shared bases, sparsity, near-linear rational units, and nonlinearity removal already have close precedents. | Include 2017 multi-output PN, RAN, PAON, and pruning precedents; restrict novelty to unproven empirical/engineering differentiation. |
| Scientific/prior art | The RAN paper and release are not the same denominator/gating implementation. | Separate released-equation and labeled paper-form controls; no inherited guarantee or author-score reproduction claim. |
| Scientific | Gamma-only regularization can be made arbitrarily small by rescaling factors. | Normalize effective directions or use invariant penalties; require function-space removal checks. |
| Scientific | Stability of one residual can conceal a large linear path, odd-function limitations, and nuisance suppression. | Add explicit bounds, asymptotic proof, odd/nuisance teachers, bypass-norm monitoring, and recomputed-target amplitude tests. |
| Scientific | The temporal XOR example is unsupported by the literal temporal formula. | Preserve literal model and explicitly add direct state readout in the corrected model; test all four cases. |
| Experimental | Normalization or smaller initial feature variance could explain a “bounded neuron” gain. | Add equal-preprocessing, constant-denominator, LayerNorm-energy, and learning-rate/variance controls. |
| Experimental | More tuning, capacity, favorable teachers, or weak MLPs could manufacture a win. | Strong polynomial/rational/native baselines, measured search budgets, independent teachers, native recipe preservation, protected fixed-panel endpoint. |
| Experimental | Training seeds and repeated timings could be mistaken for independent data. | Group-based and session-based inference, separate seed effects, fixed-panel scope, pilot-calibrated precision. |
| Experimental | Recipe ages, airfoil sweeps, duplicate beans, or Bike target components could leak information. | Group/time splits and explicit feature exclusions; split overlap and feature-availability tests block training. |
| Experimental | Pruning masks and kernel-only timing could exaggerate deployment savings. | Physical compact exports, affine folding, equal-shape training controls, complete-prediction timings and warm/cold disclosures. |
| Implementation | Direct FP32 squaring overflows on large finite inputs. | Stable algebraically equivalent energy map; overflow check and total-path failure logging. |
| Implementation | A square-root-of-sum-squares row norm has NaN gradients at an exactly zero row in the original snippet. | Replace with `torch.linalg.vector_norm`; CPU float64 check found finite gradient \(10^8\) with epsilon \(10^{-8}\), while the original produced NaNs. Add a zero-row regression test. Large finite gradients remain a monitored risk. |
| Implementation | Unified-memory counters can be double-counted or mistaken for traffic. | Report allocator/RSS measures separately; physical-byte claims require a validated counter. |
| Implementation | MPS availability or a model README does not guarantee the training/prediction path. | Operation/optimizer/cache smoke tests, explicit faithful CPU fallback, no hidden fallback timings. |
| Implementation | Optional dependencies, data metadata, and current model terms can differ from assumptions. | Separate environments, raw-byte data audit, source/weight license gates, version/hash freeze, and explicit baseline exclusions. |
| Implementation | Resuming only weights would change optimizer, sampling, and experiment expenditure. | Atomic complete-state checkpoints, exact CPU resume test, numerical MPS check, preserved failed prefixes and cost ledger. |

A final cross-check must verify that each primary claim has its baseline, metric, independent unit, and decision gate; each expensive run depends on relevant checks; every command is either standard software or explicitly a proposed project interface; all references and equations resolve; and every unknown is reflected in a blocking gate or limitation. If a challenge reveals a new material defect, amend the affected protocol and recheck downstream packages before unblinding.

**Residual uncertainties:** undiscovered or inaccessible prior art; detailed evidence from the dendritic communication paper; native implementation/license drift; hidden data-acquisition dependence and foundation-model benchmark exposure; whether sparse low-dimensional operations are fast on this exact M3 Max; sensitivity with few physical groups; and whether any bounded-interaction benefit transfers beyond the chosen tasks. None is resolved by the mathematical checks. No end-to-end model accuracy, energy efficiency, physical communication reduction, or hardware performance has been demonstrated here.

## 14. References and search-coverage record

The references identify stable primary sources or official implementations/documentation. Dates are publication/version dates where known; online documentation was inspected on **16 October 2026**. Access labels distinguish inspected content from blocked or partial access. No assertion of exhaustive prior-art clearance or legal patentability is made.

**[R01] Steffen Rendle. “Factorization Machines.” ICDM, 2010.**  
[Author-hosted paper](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf). Inspected the formulation and linear-in-rank evaluation; PDF equation page also inspected visually. Establishes a directly relevant efficient pairwise baseline, not the proposed energy denominator.

**[R02] Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, Naonori Ueda. “Polynomial Networks and Factorization Machines: New Insights and Efficient Training Algorithms.” ICML, 2016.**  
[Proceedings](https://proceedings.mlr.press/v48/blondel16.html). Inspected the primary proceedings abstract and formulation summary, not a complete experimental replication. Informs the polynomial/FM relationship and the need for signed quadratic controls.

**[R03] Mathieu Blondel, Vlad Niculae, Takuma Otsuka, Naonori Ueda. “Multi-output Polynomial Networks and Factorization Machines.” NeurIPS, 2017.**  
[Proceedings](https://proceedings.neurips.cc/paper/2017/hash/dea9ddb25cbf2352cf4dec30222a02a5-Abstract.html); [paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/dea9ddb25cbf2352cf4dec30222a02a5-Paper.pdf). Inspected the shared-basis representation, norm constraints, and sparse basis-selection treatment; relevant equation page inspected visually. A particularly close precedent for shared nonlinear features and allocation, without the exact fixed energy map.

**[R04] Jusheng Zhang, Ningyuan Liu, Qinhan Lyu, Jing Yang, Keze Wang. “Rational ANOVA Networks.” arXiv:2602.04006v1, 3 February 2026.**  
[Record](https://arxiv.org/abs/2602.04006); [versioned text](https://arxiv.org/html/2602.04006v1). Inspected equations, architecture, and appendices. Close rational/pairwise/sparse-unit prior art; no local reproduction was executed. The paper–implementation distinction is documented in §3.3.

**[R05] Zhang and collaborators. Rational ANOVA Networks official implementation, inspected 2026-10-16.**  
[Repository](https://github.com/jushengzhang/Rational-ANOVA-Networks); [inspected tabular source](https://raw.githubusercontent.com/jushengzhang/Rational-ANOVA-Networks/main/TAB_model.py). Inspected the tabular module directly. Source reuse license and a frozen commit were not verified; source incorporation is therefore gated. A changing `main` URL is evidence provenance, not a reproducible dependency lock.

**[R06] Onur Keleş, A. Murat Tekalp. “PAON: A New Neuron Model using Padé Approximants.” ICIP, 2024; and “Padé Neurons for Efficient Neural Models,” arXiv:2601.04005v1, 7 January 2026.**  
[2024 paper](https://arxiv.org/abs/2403.11791), DOI [10.1109/ICIP51287.2024.10648214](https://doi.org/10.1109/ICIP51287.2024.10648214); [extended text](https://arxiv.org/html/2601.04005v1); [repository](https://github.com/onur-keles/Paon); [core implementation](https://raw.githubusercontent.com/onur-keles/Paon/main/paon.py); [license](https://raw.githubusercontent.com/onur-keles/Paon/main/LICENSE). Inspected rational smoothing, linear-module implementation/initialization, and MIT license. Provides a close reusable rational-neuron baseline; higher-order denominator safety must be checked rather than assumed.

**[R07] Jeffrey M. Beck, Peter E. Latham, Alexandre Pouget. “Marginalization in Neural Circuits with Divisive Normalization.” Journal of Neuroscience 31(43), 15310–15319, 26 October 2011.**  
DOI [10.1523/JNEUROSCI.1706-11.2011](https://doi.org/10.1523/JNEUROSCI.1706-11.2011); [publisher](https://www.jneurosci.org/content/31/43/15310); [institutional full text](https://access.archive-ouverte.unige.ch/access/metadata/ea3c1500-56e2-473a-945a-b36d5d7b3119/download). Inspected the quadratic/divisive formulation. It establishes related mechanisms in neural computation, not an identical energy-normalized machine-learning layer or an M3 speed result.

**[R08] Grigorios G. Chrysos and collaborators. “Π-nets: Deep Polynomial Neural Networks.” CVPR, 2020; and “Regularization of Polynomial Networks for Image Recognition.” CVPR, 2023.**  
[Π-nets](https://arxiv.org/abs/2003.03828); [regularization paper](https://arxiv.org/abs/2303.13896); [full text](https://arxiv.org/pdf/2303.13896); [regularization implementation](https://github.com/grigorisg9gr/regularized_polynomials). Inspected the first paper's abstract and the second paper's method/initialization discussion. Informs strong deep-polynomial optimization controls; image-recognition results are contextual and not imported into the tabular comparison.

**[R09] Christos Louizos, Max Welling, Diederik P. Kingma. “Learning Sparse Neural Networks through L0 Regularization.” ICLR, 2018; arXiv:1712.01312.**  
[Paper record](https://arxiv.org/abs/1712.01312); [official code](https://github.com/AMLab-Amsterdam/L0_regularization). Inspected the primary abstract and repository context. Establishes learned sparsity precedent, not an assertion that this older implementation is maintained or required by the core PoC.

**[R10] Nandan Kumar Jha, Zahra Ghodsi, Siddharth Garg, Brandon Reagen. “DeepReDuce: ReLU Reduction for Fast Private Inference.” ICML, 2021.**  
[Proceedings](https://proceedings.mlr.press/v139/jha21a.html). Inspected the primary abstract. Relevant to selective nonlinear computation; private-inference operation costs differ materially from ordinary Apple Silicon execution, so reported speed benefits are not transferable without measurement.

**[R11] Xundong Wu and collaborators. “Dendritic nonlinearities mitigate communication costs.” Patterns, 2026.**  
DOI [10.1016/j.patter.2026.101520](https://doi.org/10.1016/j.patter.2026.101520); [author repository](https://github.com/motifMachine/Dendrite_communication). Publication identity and repository were located; full publisher/PMC access and detailed code traversal were limited. **No quantitative communication or hardware-performance finding from this paper is treated as verified here.** This limitation blocks inherited claims, not a faithful new local experiment.

**[R12] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, Jian Sun. “Simple Baselines for Image Restoration.” ECCV, 2022; arXiv:2204.04676.**  
[Paper](https://arxiv.org/abs/2204.04676); [NAFNet repository](https://github.com/megvii-research/NAFNet); [inspected architecture](https://raw.githubusercontent.com/megvii-research/NAFNet/main/basicsr/models/archs/NAFNet_arch.py). Inspected the abstract and actual SimpleGate/residual code. Establishes multiplication without conventional hidden activations; the proposed SimpleGate MLP is not a full image-restoration reproduction.

**[R13] Ruoxi Wang and collaborators. “DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems.” WWW, 2021; arXiv:2008.13535.**  
[Paper record](https://arxiv.org/abs/2008.13535). Inspected the primary formulation and low-rank-cross discussion. Supports a strong cross-feature comparison; a small supervised local adaptation cannot reproduce industrial system conditions.

**[R14] Antonio Orvieto and collaborators. “Resurrecting Recurrent Neural Networks for Long Sequences.” ICML, 2023; arXiv:2303.06349. Albert Gu, Karan Goel, Christopher Ré. “Efficiently Modeling Long Sequences with Structured State Spaces.” ICLR, 2022; arXiv:2111.00396.**  
[Linear recurrent work](https://arxiv.org/abs/2303.06349); [structured state-space work](https://arxiv.org/abs/2111.00396). Inspected primary abstracts. These provide recurrence context, not a claim to reproduce their benchmark configurations. The exact exponential-filter equivalence in §11.3 is a deduction here.

**[R15] Chenhui Xu, Xinyao Wang, Fuxun Yu, Jinjun Xiong, Xiang Chen. “QuadraNet V2: Efficient and Sustainable Training of High-Order Neural Networks with Quadratic Adaptation.” arXiv:2405.03192, 2024.**  
[Versioned text](https://arxiv.org/html/2405.03192v2). Inspected abstract and method context. Adjacent precedent for quadratic adaptation around a linear model; no local reproduction or direct tabular superiority is asserted.

**[R16] David Holzmüller, Léo Grinsztajn, Ingo Steinwart. “Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data.” NeurIPS, 2024; arXiv:2407.04491.**  
[Paper](https://arxiv.org/abs/2407.04491); [official pytabkit implementation](https://github.com/dholzmueller/pytabkit). Inspected the primary record and official usage material. Informs RealMLP/native preprocessing baselines. Current package APIs, source revision, and reuse terms must be frozen locally; the attempted license retrieval did not resolve them here.

**[R17] Yury Gorishniy, Akim Kotelnikov, Artem Babenko. “TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.” ICLR, 2025; arXiv:2410.24210.**  
[Paper](https://arxiv.org/abs/2410.24210); [official repository](https://github.com/yandex-research/tabm); [usage](https://raw.githubusercontent.com/yandex-research/tabm/main/README.md); [license](https://raw.githubusercontent.com/yandex-research/tabm/main/LICENSE). Inspected native usage and Apache-2.0 license. Supports preservation of native ensemble behavior; no M3 performance claim is inferred from repository availability.

**[R18] Apple Developer. “Accelerated PyTorch training on Mac.” Inspected 2026-10-16.**  
[Official setup documentation](https://developer.apple.com/metal/pytorch/). Inspected installation prerequisites and MPS setup. Used for dependency discovery rules; local installed-system compatibility remains a blocking test.

**[R19] PyTorch. `torch.mps` API, version 2.11 documentation. Inspected 2026-10-16.**  
[API](https://docs.pytorch.org/docs/2.11/mps.html). Inspected synchronization, allocator/working-set metrics, RNG and profiler interfaces. Supports measurement implementation, not a promise of physical DRAM-traffic counters or support for every downstream model operation.

**[R20] PyTorch. “MPS Environment Variables,” version 2.11 documentation. Inspected 2026-10-16.**  
[Environment-variable documentation](https://docs.pytorch.org/docs/2.11/mps_environment_variables.html). Inspected fallback and allocator-related settings. Establishes why hidden CPU fallback must be excluded from device-performance claims; this plan does not recommend disabling memory safeguards.

**[R21] CatBoost. Python pip-installation documentation. Inspected 2026-10-16.**  
[Official installation documentation](https://catboost.ai/docs/en/installation/python-installation-method-pip-install). Inspected platform/wheel and GPU conditions. Supports native macOS arm64 CPU execution as the initial path; exact package support is still validated at installation.

**[R22] Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan. “TabICLv2: A better, faster, scalable, and open tabular foundation model.” arXiv:2602.11139, February 2026.**  
[Paper](https://arxiv.org/abs/2602.11139); [versioned text](https://arxiv.org/html/2602.11139v1); [official code](https://github.com/soda-inria/tabicl); [model card/checkpoints](https://huggingface.co/jingang/TabICL); [code license](https://raw.githubusercontent.com/soda-inria/tabicl/main/LICENSE). Inspected paper, model card, device-selection/usage documentation, and BSD-3-Clause terms. Actual checkpoint memory, complete MPS path, and context costs require local tests.

**[R23] Prior Labs. Model-version and availability documentation. Inspected 2026-10-16.**  
[Official model documentation](https://docs.priorlabs.ai/models). Inspected current local-model listings through TabPFN-3.5 and distinctions between code and weight-use terms. Used to avoid an outdated reference set, not as legal advice or proof of superiority on the selected panel.

**[R24] I-Cheng Yeh / UCI Machine Learning Repository. Concrete Compressive Strength, dataset 165.**  
[Official dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength); DOI [10.24432/C5PK67](https://doi.org/10.24432/C5PK67). Inspected feature/target description, reported shape, provenance, and CC BY 4.0 terms. Raw data and group counts were not downloaded/audited in this investigation.

**[R25] Thomas Brooks, D. Pope, Michael Marcolini / UCI. Airfoil Self-Noise, dataset 291.**  
[Official dataset](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise); DOI [10.24432/C5VW2C](https://doi.org/10.24432/C5VW2C). Inspected variable semantics, reported shape, source attribution, and CC BY 4.0 license. Metadata typing is not used to turn a numerical angle into a binary feature; raw-byte verification remains WP02.

**[R26] Murat Koklu, Ilker Ali Ozkan / UCI. Dry Bean, dataset 602, 2020.**  
[Official dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset); DOI [10.24432/C50S4B](https://doi.org/10.24432/C50S4B). Inspected numerical descriptor/class metadata, reported shape, and CC BY 4.0 terms. Exact duplicate groups and acquisition dependence remain empirical ingest questions.

**[R27] Hadi Fanaee-T / UCI. Bike Sharing, dataset 275, 2013.**  
[Official dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset); DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894). Inspected chronology, hourly/daily variable meanings, target-component relationship, and CC BY 4.0 terms. The proposed nowcasting split is new experimental design, not the dataset's standard benchmark split.

**[R28] Hendricus G. Loos. “Method and apparatus for associative memory.” US5995954A, published 30 November 1999.**  
[Public patent record and document](https://patents.google.com/patent/US5995954A/en). Inspected older quadratic/dendritic/synaptic mechanisms during a limited terminology screen. It is not established as an exact match to equation (1). No legal validity, patentability, infringement, or freedom-to-operate conclusion is made.

**[R29] Janez Demšar. “Statistical Comparisons of Classifiers over Multiple Data Sets.” Journal of Machine Learning Research 7, 1–30, 2006.**  
[Primary journal record](https://www.jmlr.org/papers/v7/demsar06a.html). Inspected the abstract's distinction and recommendations for multi-dataset comparison. This plan deliberately does not apply a broad-dataset ranking test to three fixed tasks; the registered group/seed bootstrap is its own specified analysis, not a claim to reproduce this paper's procedure.

### 14.1 Search coverage and stopping rationale

| Search branch | Representative query concepts / citation tracing | Material consequence |
|---|---|---|
| Function class and historical equivalents | Factorization machines; sigma–pi/bilinear/quadratic neurons; polynomial networks; shared multi-output bases | Exact polynomial baselines and a narrowed novelty claim |
| Rational and normalized computation | Rational ANOVA; Padé/PAON; divisive quadratic circuits; fixed-energy normalization | Close current competitors, denominator controls, paper–code fidelity checks |
| Selective nonlinear computation | Sparse polynomial basis selection; L0 networks; nonlinear-operation pruning; quadratic adaptation | Allocation treated as a tested deployment strategy, not an invented principle |
| Alternative proposals | Dendritic communication; local grouped computation; exponential memory; Volterra/readout equivalence; linear recurrence/state spaces | Conditional, mechanism-preserving alternatives with explicit unverified hardware/scaling claims |
| Current executable baselines | RealMLP/pytabkit; TabM; CatBoost; TabICLv2; current local TabPFN | Strong native comparisons, pretraining/context accounting, license/resource gates |
| Laptop and data prerequisites | Apple/PyTorch MPS docs, fallback/memory APIs, official package installation and UCI metadata/licenses | Operation-level checks, faithful fallback, hardware-adaptive resource plan, grouped/time splits |
| Falsification and public patent screen | Existing quadratic dendrites/products; equivalent transformed-feature fitting | No assertion that an unfamiliar formula is a patentable or scientifically new invention |

Backward tracing reached foundational 2010/2011 and 2016/2017 mechanisms; forward/current searches reached relevant 2026 formulations and software/assets. Additional targeted refinement searches ceased to change the selected competing function classes or the essential experiment, rather than meeting an arbitrary paper count. This is a bounded search outcome, not proof that no undiscovered alternative exists. Source snapshots could not be downloaded in this environment; inspected web text, code views, and relevant PDF pages informed the plan. Full papers/code were not uniformly accessible, and the access labels above are part of the evidence record.

## Appendix A. Reproduce the small mathematical checks

The following script reproduces the checks reported in §4.5 and adds the zero-row norm regression check found during the implementation audit. Place it in `scripts/mathematical_checks.py` and run it in the CPU environment. It requires only NumPy and PyTorch. It does not train task models, measure M3 performance, or test the complete candidate class. Exact last bits can differ across numerical libraries; assertions use tolerances except where the result is exactly structural.

```python
"""Small CPU mathematical checks, not a task benchmark or an M3 compatibility test."""
from __future__ import annotations
import json, platform
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
torch.manual_seed(20261016)
torch.set_default_dtype(torch.float64)
d, rank, n = 8, 3, 512
A, C, gamma = torch.randn(rank,d),torch.randn(rank,d),torch.randn(rank)
Q = sum(gamma[r]*(torch.outer(A[r],C[r])+torch.outer(C[r],A[r]))/2 for r in range(rank))
x = torch.randn(n,d)
x = x*torch.logspace(-5,5,n)[:,None]
x.requires_grad_(True)
q = ((x@A.T)*(x@C.T)*gamma).sum(1)/(1+x.square().sum(1))
q_matrix = (x*(x@Q)).sum(1)/(1+x.square().sum(1))
g = torch.autograd.grad(q.sum(),x)[0]
e = x.square().sum(1,keepdim=True)
g_formula = 2*(x@Q)/(1+e)-2*(x*(x@Q)).sum(1,keepdim=True)*x/(1+e).square()
spec = torch.linalg.matrix_norm(Q,ord=2)
uval,U=torch.linalg.eigh(Q)
order=torch.argsort(uval.abs(),descending=True)
uval,U=uval[order],U[:,order]
k=2
Qk=(U[:,:k]*uval[:k])@U[:,:k].T
z=1e6*U[:,k]
prune_error=(z@(Q-Qk)@z)/(1+z@z)
V=torch.randn(5,rank)
shared=((x@A.T)*(x@C.T)/(1+x.square().sum(1,keepdim=True)))@V.T
plus,minus=(A+C)/2,(A-C)/2
signed=(((x@plus.T).square()-(x@minus.T).square())/(1+x.square().sum(1,keepdim=True)))@V.T
scaled=((x@(100*A).T)*(x@(100*C).T)*(gamma/10000)).sum(1)/(1+x.square().sum(1))

def stable_h(t:torch.Tensor)->torch.Tensor:
    s=t.abs().amax(dim=-1,keepdim=True).clamp_min(1)
    z=t/s
    return z/torch.sqrt(s.reciprocal().square()+z.square().sum(dim=-1,keepdim=True))

xb=torch.tensor([[1e20,1e20]],dtype=torch.float32)
naive=xb[:,0]*xb[:,1]/(1+xb.square().sum(1))
robust=stable_h(xb).prod(1)
# Verify the analytical Lipschitz upper bound is attainable for an indefinite Q.
tight_x=torch.tensor([2**-0.5,2**-0.5],requires_grad=True)
tight_Q=torch.diag(torch.tensor([1.,-1.]))
tight_q=tight_x@tight_Q@tight_x/(1+tight_x@tight_x)
tight_g=torch.autograd.grad(tight_q,tight_x)[0]
# Starting exactly affine does not give direction gradients until the gate changes.
a=torch.nn.Parameter(torch.randn(d)); c=torch.nn.Parameter(torch.randn(d)); v=torch.nn.Parameter(torch.zeros(()))
opt=torch.optim.SGD([a,c,v],lr=.1)
xx=torch.randn(32,d); yy=torch.randn(32)
grads=[]
for _ in range(2):
    opt.zero_grad(); p=v*(xx@a)*(xx@c)/(1+xx.square().sum(1)); loss=(p-yy).square().mean(); loss.backward()
    grads.append({'a_grad_norm':a.grad.norm().item(),'c_grad_norm':c.grad.norm().item(),'gate_grad_abs':v.grad.abs().item()}); opt.step()
# Four patterns, previous bit first, current bit second.
bits=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
m,t=bits[:,0],bits[:,1]; target=m+t-2*m*t
literal_features=torch.stack([t,m*t],1)
corrected_features=torch.stack([t,m,m*t],1)
lf=literal_features@torch.linalg.lstsq(literal_features,target).solution
cf=corrected_features@torch.linalg.lstsq(corrected_features,target).solution
# LayerNorm without learned affine scale makes energy approximately d.
y=torch.randn(128,d)
ln=torch.nn.functional.layer_norm(y,(d,),eps=1e-5)
out={
 'environment':{'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'mps_available':torch.backends.mps.is_available(),'device':'cpu','default_dtype':'float64','seed':20261016},
 'dimensions':{'d':d,'R':rank,'samples':n},
 'max_bilinear_matrix_abs_error':(q-q_matrix).abs().max().item(),
 'max_gradient_formula_abs_error':(g-g_formula).abs().max().item(),
 'max_observed_gradient_over_Q_spectral':(g.norm(dim=1)/spec).max().item(),
 'tight_example_gradient_norm':tight_g.norm().item(),
 'max_shared_signed_square_abs_error':(shared-signed).abs().max().item(),
 'gauge_scaling_max_abs_error':(scaled-q).abs().max().item(),
 'gauge_L1_ratio':(gamma/10000).abs().sum().item()/gamma.abs().sum().item(),
 'rank2_certificate':uval[k].abs().item(),
 'rank2_error_on_omitted_eigenvector':prune_error.abs().item(),
 'stable_h_max_abs_error_moderate':(stable_h(x.detach())-x.detach()/torch.sqrt(1+x.detach().square().sum(1,keepdim=True))).abs().max().item(),
 'extreme_fp32_naive_isfinite':bool(torch.isfinite(naive).all()),
 'extreme_fp32_stable_value':robust.item(),
 'zero_gate_gradients':grads,
 'nuisance_m0_value':1/3,
 'nuisance_m100_value':1/103,
 'literal_temporal_best_four_pattern_mse':(lf-target).square().mean().item(),
 'corrected_temporal_best_four_pattern_mse':(cf-target).square().mean().item(),
 'layernorm_input_energy_range':[ln.square().sum(1).min().item(),ln.square().sum(1).max().item()]
}
assert out['max_bilinear_matrix_abs_error']<1e-12
assert out['max_gradient_formula_abs_error']<1e-12
assert out['max_observed_gradient_over_Q_spectral']<=1+1e-12
assert out['max_shared_signed_square_abs_error']<1e-11
assert out['zero_gate_gradients'][0]['a_grad_norm']==0
assert out['zero_gate_gradients'][1]['a_grad_norm']>0
assert abs(out['literal_temporal_best_four_pattern_mse']-.25)<1e-12
assert out['corrected_temporal_best_four_pattern_mse']<1e-28
assert not out['extreme_fp32_naive_isfinite']
# Regression test for finite gradients at an exactly zero direction row.
row=torch.zeros(1,3,requires_grad=True)
norm=torch.linalg.vector_norm(row,dim=-1,keepdim=True)
(row/norm.clamp_min(1e-8)).sum().backward()
out['zero_row_normalization_gradient_finite']=bool(torch.isfinite(row.grad).all())
assert out['zero_row_normalization_gradient_finite']
path=Path('artifacts/reports/mathematical_checks.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
```

**Completion status of this handoff:** literature/code/documentation inspection and the labeled CPU checks were performed. All training, protected evaluation, deployment benchmarking, and M3 Max compatibility work remains explicitly assigned to the work packages. The recommendation is to test bounded interactions and cost-validated allocation against their closest established equivalents—not to presume a novel or superior neuron.

<!-- Ordinary Markdown citation targets; fuller provenance and access notes are in §14. -->
[R01]: https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf
[R02]: https://proceedings.mlr.press/v48/blondel16.html
[R03]: https://proceedings.neurips.cc/paper/2017/hash/dea9ddb25cbf2352cf4dec30222a02a5-Abstract.html
[R04]: https://arxiv.org/abs/2602.04006
[R05]: https://github.com/jushengzhang/Rational-ANOVA-Networks
[R06]: https://arxiv.org/html/2601.04005v1
[R07]: https://doi.org/10.1523/JNEUROSCI.1706-11.2011
[R08]: https://arxiv.org/abs/2303.13896
[R09]: https://arxiv.org/abs/1712.01312
[R10]: https://proceedings.mlr.press/v139/jha21a.html
[R11]: https://doi.org/10.1016/j.patter.2026.101520
[R12]: https://arxiv.org/abs/2204.04676
[R13]: https://arxiv.org/abs/2008.13535
[R14]: https://arxiv.org/abs/2303.06349
[R15]: https://arxiv.org/html/2405.03192v2
[R16]: https://arxiv.org/abs/2407.04491
[R17]: https://arxiv.org/abs/2410.24210
[R18]: https://developer.apple.com/metal/pytorch/
[R19]: https://docs.pytorch.org/docs/2.11/mps.html
[R20]: https://docs.pytorch.org/docs/2.11/mps_environment_variables.html
[R21]: https://catboost.ai/docs/en/installation/python-installation-method-pip-install
[R22]: https://arxiv.org/abs/2602.11139
[R23]: https://docs.priorlabs.ai/models
[R24]: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
[R25]: https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
[R26]: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
[R27]: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
[R28]: https://patents.google.com/patent/US5995954A/en
[R29]: https://www.jmlr.org/papers/v7/demsar06a.html
