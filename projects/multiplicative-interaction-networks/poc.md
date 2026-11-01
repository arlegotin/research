# Multiplicative neurons: research assessment and executable proof-of-concept plan

**Research cutoff:** 29 October 2026.  
**Target machine:** one MacBook Pro with an M3 Max; actual memory, free storage, macOS version and acceleration support must be discovered.  
**Deliverable status:** researched experimental plan, with small CPU numerical/data checks; no neural-network training or M3 Max benchmark has been performed.  
**Purpose:** determine whether compact learned products provide a reproducible statistical, optimization or resource advantage beyond strong existing alternatives. Positive, negative and inconclusive outcomes are all valid.

This document is a complete handoff. Commands under the proposed `multneur` package describe software the implementing agent must create; they are not represented as existing executables. External package/system commands are identified separately. References use ordinary links and the bibliography in §12. No access to the originating conversation is needed.

## 1. Executive assessment and selected direction

### 1.1 Verdict

**Investigate low-order learned-factor polynomial residuals, with a separate normalization diagnostic. Do not begin by replacing every affine layer with an unrestricted product or by combining products, roots and complex numbers.**

The central idea is scientifically legitimate but substantially anticipated. Learned all-subsets products and degree-weighted ANOVA components appear in higher-order factorization machines; products of affine projections appear in Pi-Sigma/ridge polynomial networks; modern polynomial networks, cross networks and gated layers already exploit multiplicative interactions. The simple two-input product is a bilinear model. These are direct competitors, not merely historical context. [R01–R07]

The selected model is therefore an **experimental realization of an established family**, not an asserted new neuron. Its operational name in the repository is `product_residual`. The opportunity is to establish a useful, well-characterized operating region: whether low-degree products with learned factor directions and deliberate numerical conditioning improve prediction, training reliability or the accuracy/cost frontier after fair tuning of the nearest existing methods. No methodological novelty is established by this investigation.

Three changes from the original proposal are explicit:

1. Preserve a separate affine path, so ordinary main effects do not require cancellation among products.
2. Begin with two to four learned linear factors per component, and expose their polynomial degrees separately. This relaxes coordinate alignment and coefficient tying. It also moves further into known polynomial-network territory.
3. Keep roots and complex representations in isolated experiments. They answer different questions and can obscure the source of a gain if combined immediately.

The literal shifted-input neuron remains a required diagnostic. High fan-in and compositional depth remain conditional extensions, because a small low-degree test cannot settle their possible benefits.

### 1.2 What a successful local result could establish

The strongest attainable core claim is:

> Under a frozen training and selection policy, on a specified distribution of independently generated regression problems, the selected product model changes held-out predictive risk relative to a validation-selected portfolio of strong local alternatives, with quantified uncertainty and measured costs. Its attribution is supported by matched function-class, degree, preprocessing and resource controls.

A second, narrower practical claim can concern **two specified measured engineering datasets with held-out experimental-condition groups**. It cannot establish superiority on tabular learning in general, new laboratory populations, images, language, production systems, or large-scale training.

Evidence levels must remain separate:

| Level | Required evidence | Permitted interpretation |
|---|---|---|
| Algebra/correctness | Exact identities, gradient checks, equivalent implementations | The mechanism computes the intended function |
| Representation | Oracle parameters or solved feature readouts | A function has a compact representation; optimization remains unproved |
| Learning | Independent target/data draws, strong trained controls | An inductive-bias or optimization effect in the stated population |
| Practical local utility | Real-data group holdouts plus full cost accounting | A useful tradeoff on the named workloads |
| Broader advance | Independent domains, reproductions and scale tests | Requires subsequent evidence; outside this PoC |

### 1.3 Findings established during planning

**Inspected-source findings:** the main product constructions have close prior art; contemporary polynomial work already studies normalization and regularization; strong 2026 tabular comparators include models beyond ordinary MLPs; current MPS compatibility depends on exact PyTorch/macOS versions. See §§3, 5, 6 and the evidence ledger.

**Measured here:** on Linux x86_64, Python 3.12.14 and NumPy 2.3.5, a four-factor float32 residual calculation with each increment approximately `1e-6` gave approximately `-1.8530e-7` using direct subtraction, whereas the reference interaction was approximately `6.0000e-12`. The recurrence in §4 agreed with the float64 reference to approximately `1.51e-8` relative error. This is a numerical example, not a training result or a claim that the recurrence is new. All comparable methods must receive the same correction.

**Measured here:** official UCI files contain repeated experimental conditions. Concrete has 428 exact mix keys before deduplication; Airfoil has 106 condition keys; Yacht has 22 hull keys. These observations change the split design. They do not prove that the keys identify statistically independent experiments. Details and file hashes are in §5.

**Deductions:** fixed degree rescaling changes parameterization rather than expressivity; magnitude roots change the function being learned; complex multiplication is exactly implementable with tied real arithmetic. These conclusions are derived below.

**Unperformed checks:** installation, training, numerical parity, speed and memory on the target Mac; reproduction of baseline training results; confirmatory evaluation; broad novelty clearance. They are explicit early gates, not assumed successes.

## 2. Problem, hypotheses and mathematical limits

### 2.1 Problem statement and assumptions

Let `x ∈ R^d` be a numerical feature vector and `y ∈ R` a scalar target. An ordinary affine neuron is `wᵀx+b`. The originating proposal is

\[
h(x)=a\prod_{i=1}^{d}(x_i+w_i)+b,
\]

possibly followed by an nth root, and possibly using complex intermediate values. The scientific question is whether making products explicit reduces the samples, parameters or computation required to learn useful relationships, rather than requiring an additive network to approximate those relationships indirectly.

The initial beneficiaries are researchers and developers of small numerical predictors with plausible feature interactions. No application-specific need has been supplied. It is an **assumption**, not established evidence, that inadequate interaction representation is their bottleneck. The experiments must distinguish that bottleneck from insufficient tuning, noisy labels, poor preprocessing, excessive variance or a weak baseline.

The core uses scalar regression because it exposes approximation error and amplitude behavior directly. Binary parity is a diagnostic, not the primary benchmark. No pretrained asset is needed for the mechanism study. A public pretrained tabular model is included in the separate practical comparison, with its prior training accounted for explicitly.

All initial neural comparisons use float32, the same training observations, identical validation information, and train-only transforms. CPU float64 is the numerical oracle. The primary model is shallow; adding depth is contingent on evidence that a compositional target or representation bottleneck requires it.

### 2.2 Hypotheses and alternatives

| ID | Testable hypothesis and predicted observation | Strong alternative explanation | Weakening/falsifying observation within scope |
|---|---|---|---|
| H1, primary | Low-order learned products improve predictive-risk/cost behavior on the frozen interaction-task population | Any low-degree polynomial, kernel, bilinear or gated model offers the same benefit | Benefit vanishes against polynomial ridge/kernel, HOFM, CP/Pi-Sigma, CCP or DCN controls under adequate tuning |
| H2, attribution | Degree separation or fixed scale calibration improves optimization reliability | Different initialization, effective learning rate or regularization causes the result | A function-preserving rescaling control or fair per-block learning-rate tuning removes it |
| H3, literal proposal | Shifted products learn coordinate-aligned interactions efficiently | The task directly encodes the answer or provides the relevant support | Oracle succeeds but learned factors fail; benefit is limited to answer-matched teachers or disappears under irrelevant features/rotation |
| H4, normalization | A magnitude-only power map improves conditioning without unacceptable degradation of amplitude-sensitive prediction under the specified readout | It merely changes the readout's approximation burden | Gains occur only on geometric-mean/phase targets; multiplication targets degrade; factor gradients remain unstable |
| H5, conditional phase study | A phase-respecting inductive bias helps phase composition/sample efficiency | Parameter tying or explicit sine/cosine features explain it | An exactly equivalent tied-real model disagrees, indicating an implementation error; an untied real bilinear model performs as well at comparable cost |
| H6, transfer | A selected product model improves the finite engineering cohort's accuracy/cost tradeoff | Leakage, duplicate rows, univariate nonlinearities, more tuning or extra capacity | Group splits, additive splines, full low-degree polynomial baselines or current tabular models erase the gain |

H2–H6 are secondary or exploratory unless separately frozen and allocated a confirmatory error budget. A positive result for one does not imply the others. A root can improve geometric-mean learning while making multiplication less accurate; both findings can be correct.

### 2.3 Expressivity of the literal shifted product

For two inputs,

\[
a(x_1+w_1)(x_2+w_2)+b
=ax_1x_2+aw_2x_1+aw_1x_2+aw_1w_2+b.
\]

This directly represents the signed interaction on `{-1,+1}²`, but a comparison only with one affine threshold is not a competitive architecture test. For `d` inputs the expansion contains up to `2^d` monomials, whose coefficients obey strong multiplicative constraints. Exponentially many terms do not provide exponentially many independently learned coefficients.

Further deductions that affect the experiment:

- One nonconstant full-input product with `a ≠ 0` cannot exactly represent a purely affine function on an open domain when `d > 1`: its highest-degree coefficient remains `a`.
- A sum of coordinate products is multilinear in each raw coordinate. In a single such layer it cannot represent `x_i²`; a full quadratic or learned-projection product can. Include this negative control.
- Products of projections can contain squared coordinates and other repeated powers. Their nominal degree is not the number of distinct original features interacting.
- A fixed finite-degree shallow polynomial family is not a universal approximator of arbitrary continuous functions. Increasing width alone does not remove its degree restriction.
- If every shift is nonzero,
  \[
  a\prod_i(x_i+w_i)=a\Bigl(\prod_iw_i\Bigr)\prod_i(1+x_i/w_i).
  \]
  The zero-shift cases remain well-defined in the original formula and in affine-factor networks; never implement this equivalence by dividing by a zero shift.

Representational savings are most plausible when the coefficient structure has low effective rank, relevant feature combinations can be identified from available samples, and the product degree is controlled. Dense high-rank polynomials, additive smooth functions and mismatched symmetries are credible counterexamples.

### 2.4 Products, moments and gradients

For factors `f_i`,

\[
P=\prod_i f_i,\qquad \frac{\partial P}{\partial f_j}=\prod_{i\ne j}f_i.
\]

The derivative is a product of the *other* factors, including when `f_j=0`. Writing it as `P/f_j` is invalid at zero. One zero factor can have a nonzero derivative; two zero factors make all first derivatives with respect to factors zero. This is a local property, not a claim that every nearby point is flat.

For independent factors with common second moment `q`, `E[P²]=q^k` and `E[(∂P/∂f_j)²]=q^(k−1)`. This demonstrates exponential sensitivity to moment scale under the stated independence assumption. Real learned factors are often correlated; use the formula as a diagnostic model, not a bound for an arbitrary trained network. Products also admit scale symmetries: multiplying one factor by `c` and another by `1/c` leaves the product unchanged but changes conditioning and parameter penalties.

For independent centered unit-variance `t_j`, the elementary symmetric polynomial

\[
e_m(t)=\sum_{1\le j_1<\cdots<j_m\le k}\prod_{q=1}^m t_{j_q}
\]

has variance `C(k,m)`. Hence `sqrt(C(k,m))` is a useful combinatorial scale. It is **not** a general variance guarantee for correlated projections. Even with independent raw Gaussian inputs, learned factors share those inputs and need not be independent.

The expansion of `∏(1+τt_j)` weights degree `m` by `τ^m`. Small `τ` stabilizes the branch while suppressing high degrees. A method that works by almost eliminating degree-three/four terms has not demonstrated their usefulness. Record per-degree contributions and use retrained order ablations.

### 2.5 What roots fix and what they change

For positive nonzero factors, `G=P^(1/k)` has derivative `G/(k f_j)`. Its output scale is controlled, but its factor sensitivities can diverge near zero. Computing `P` first can overflow before the root is evaluated.

For signed real products define, explicitly, `G_s(P)=sign(P)|P|^(1/k)`, with `G_s(0)=0`. It is not the real even root of a negative number; it is a signed power map. Its derivative with respect to `P` is singular at zero for `k>1`. Clamping `|P|` or its logarithm changes the function and must be an identified ablation.

The proposal's smooth magnitude-only map is

\[
N_\epsilon(P)=P\bigl(|P|^2+\epsilon^2\bigr)^{-\alpha},
\qquad \alpha=\frac{k-1}{2k},\quad\epsilon>0.
\]

For a real or complex product it preserves sign/phase and has magnitude approximately `|P|^(1/k)` when `|P|≫ε`. It is smooth as a map of real and imaginary coordinates, but generally not holomorphic. For `r=|P|`, its real two-dimensional Jacobian has tangential and radial eigenvalues

\[
\lambda_t=(r^2+\epsilon^2)^{-\alpha},\qquad
\lambda_r=(r^2+\epsilon^2)^{-\alpha-1}(\epsilon^2+r^2/k).
\]

Thus its own operator norm is bounded by `ε^(-(k−1)/k)`, which can be large. The chain rule still multiplies this by products of the other factors. At `k=4`, `ε=.01` and factors `[0,10,10,10]`, the derivative with respect to the zero factor is approximately `31,622.8`. Two zero factors still kill all first factor derivatives. Smooth normalization is not a uniform conditioning guarantee.

Changing all factors by a positive scale `c` changes a raw degree-`k` product by `c^k`, while the approximate root-magnitude output changes by `c`. Therefore the normalized neuron no longer implements the same amplitude law with a linear readout. A comparison on product regression must distinguish changed capacity from optimization. Signed power normalization of bilinear features and phase-preserving complex nonlinearities also have substantial precedent. [R11–R13]

The signed power and `N_ε` preserve amplitude information in exact arithmetic: their radial maps are strictly increasing and unbounded, hence invertible. The limitation concerns the normalized unit with the specified restricted readout. A sufficiently expressive downstream model could learn the inverse at additional capacity/optimization cost. Clamping, underflow or finite-precision rounding can cause actual information loss.

### 2.6 Complex numbers: useful structure, no magical extra arithmetic

Real multiplication cannot spontaneously create an imaginary component. Complex inputs/parameters or an explicitly chosen complex operation are required. A nonzero complex value has `k` distinct kth roots; a principal root chooses a branch and can jump across its cut. Rooting the product and multiplying principal roots of factors need not agree.

For `z_j=r_j exp(iθ_j)`, multiplication composes phases by addition. But

\[
(a+ib)(c+id)=(ac-bd)+i(ad+bc)
\]

is an exactly equivalent real computation. A complex parameter counts as two real scalars. Compare the phase model with this tied-real realization first, then with an untied real bilinear model. A performance difference between mathematically identical implementations is an implementation/backend question.

Preserving phase is not sufficient for a physical symmetry. If every input gains a common phase `φ`, a product of `k` unconjugated factors gains `kφ`; a relative-phase invariant may instead require `z_1 conjugate(z_2)`. Complex affine biases can break rotation equivariance. Specify the desired transformation law before building a phase architecture. On unit-modulus inputs, `N_ε(∏z_j)` is only a constant rescaling of the original product; normalization contributes no new phase information there.

## 3. Prior art, competing explanations and novelty assessment

### 3.1 Closest alternatives

The matrix is about mechanisms and evidence, not a numerical ranking. `d` denotes input width, `r` rank and `m` degree. Complexity omits batch/output factors where indicated.

| Family / source | Problem and mechanism | Assumptions / empirical scope | Cost and implementation | Exact relationship and consequence |
|---|---|---|---|---|
| Product units, 1989 [R02] | Weighted powers multiplied instead of affine aggregation | Historical approximation/learning proposal; fractional powers need domain care | Small units; historical reference rather than core software | General idea of product neurons predates this project; shifts differ from exponents |
| Pi-Sigma / ridge polynomial, 1995 [R03] | Product of learned affine sums; sums/growth of polynomial units | Fixed order restricts capacity; older function/classification experiments | `O(rmd)` projections; equations directly implementable | Learned-factor product is in this family; mandatory CP/Pi-Sigma control |
| HOFM / all-subsets, 2016 [R01] | ANOVA products, shared/independent degree factors and learned degree weights | Distinct-coordinate interactions; link-prediction evaluation | DP approximately `O(rdm)` for shared degrees; archived BSD code and MIT alternative | Directly anticipates shifted-feature residual and degree control; unlimited degree was not uniformly best |
| Sparse-interaction FMs, 2021 [R04] | Regularization on feature interactions, including all-subsets extensions | Structured sparsity; synthetic and real evaluations | Proximal/CD/SGD variants; optional if sparsity becomes a claim | Sparse product selection is also an established research direction |
| Π-Nets, 2020/2022 [R05] | Tensor-factorized polynomial recurrences and skip paths | Vision/audio/graph evidence; several distinct architectures | Projection/Hadamard chains; author code has noncommercial terms | Modern closest architecture family; specify CCP baseline exactly |
| Regularized polynomial nets, 2023 [R06] | Degree-sensitive initialization and normalization/regularization | Image/audio comparisons under particular training recipes | Extra normalization and training cost; source available, reuse license needs checking | Generic stabilization cannot be claimed as a new principle; share applicable controls |
| DCN-V2, 2021 [R07] | `x_next=x0*(W x+b)+x` | Explicit crosses in ranking; local numeric adaptation is a component test | Full matrix `O(Ld²)` or stated low-rank variant; Apache reference | Strong residual interaction competitor; industrial published numbers are contextual |
| Multiplicative integration / gated layers [R08] | Affine plus bilinear terms; gate/value products | RNN evidence; standard gated feedforward controls | Approximately two projections plus product | A two-factor improvement may be ordinary gating/bilinearity |
| NMU / NPU / arithmetic benchmark [R09,R10] | Learn selected arithmetic products/powers, sometimes through complex arithmetic | Synthetic arithmetic/extrapolation; optimization success varies | Small local models; official implementations, domains must match | Required for arithmetic claims; not compulsory for ordinary tabular risk |
| QuasiNet, 2023/2024 [R14] | Product of learned interpolations between one and bounded hidden activations | XOR/parity/spirals against basic MLPs | Small model; repository access limited in this investigation | Close gated-factor idea; its evidence does not establish broad practical superiority |
| Bilinear power normalization / complex networks [R11–R13] | Signed power transforms, complex weight tying, modulus/phase processing | Vision/audio or sequence settings | Additional real arithmetic; paired-real implementation possible | Root-like compression and phase-preserving computation are established concepts |
| Stable polynomial training, 2025; ULD-Net, 2026 [R15,R16] | Boundary penalties/selective clipping; polynomial normalization | Largely encrypted-inference-oriented image/audio training | Large published runs; local recipe controls possible | Recent normalization literature is material; full ImageNet comparisons are not laptop reproductions |

The apparent tension between positive polynomial-network papers and negative/unimpressive product results is not a contradiction. Different studies use different factor sharing, degrees, data, regularization and computational budgets. HOFM's all-subsets result concerns sparse link prediction; modern deep polynomial results concern substantially different backbones and training regimes. Neither establishes what happens on the present finite numerical cohort.

### 3.2 Six explicit prior-art answers

1. **What exists?** Products of coordinates, affine projections and learned powers; degree-weighted ANOVA components; low-rank polynomial tensors; residual feature crosses; real and complex multiplicative gates; several stabilization techniques.
2. **What can be reused?** Author equations and permissively licensed reference implementations; PyTorch autodiff; scikit-learn solvers/transforms; official TabM, RealMLP and TabICL packages; CatBoost; public UCI data. Do not rebuild these packages or import an old research environment wholesale.
3. **What differs here?** The exact experimental combination, the selected parameter-sharing pattern, numerical implementation and evaluation population. Fixed initial scale calibration is an invertible reparameterization. None currently supports a substantive invention claim.
4. **Why could it matter?** Low-rank products may use limited samples and parameters effectively; degree control can avoid irrelevant high-order variance; a favorable optimizer coordinate system might increase reliable training. These are testable conditional benefits.
5. **What isolates it?** Equivalent-function controls; matched bilinear and polynomial competitors; degree-two-only and independent-degree ablations; shared normalization/tuning; raw-coordinate versus learned-projection factors; negative tasks and genuine held-condition transfer.
6. **What would invalidate novelty?** An algebraic mapping to the cited constructions already invalidates broad new-neuron claims. An earlier implementation of the same conditioning/degree recipe would further narrow any engineering differentiation. A new empirical finding remains possible, but must describe its precise population and protocol.

### 3.3 Selected direction versus alternatives

| Direction | Value and falsifiability | Local difficulty | Decision |
|---|---|---|---|
| Literal all-input shifted product | Directly preserves original proposal; easy to disprove overbroad capacity claims | Easy code; increasing fan-in can be ill-conditioned | Required diagnostic, not primary practical architecture |
| Low-order projected degree residual | Preserves explicit multiplication and isolates degree/sharing/conditioning | Small, ordinary operations; strong competitors available | **Primary research direction** |
| Raw-coordinate HOFM or polynomial ridge | Strong simpler explanations; sometimes likely to win | Very feasible at chosen scales | Mandatory baselines and legitimate final recommendation if best |
| Deep unrestricted products | Could expose composition-dependent benefits | Degree grows rapidly; confounded with depth/optimization | Escalate only on a predeclared compositional target and successful small-block checks |
| Magnitude-normalized products | Testable stability/amplitude tradeoff | Easy small-scale probe; singular/log/overflow traps | Separate diagnostic, no default normalization |
| Complex phase products | Appropriate for known phase transformation laws | Paired-real path feasible; meaningful domain unspecified | Conditional synthetic mechanism study; real-world claim requires later domain selection |

No credible differentiated *architecture invention* survives the inspected closest prior art. The plan therefore explicitly permits the final answer to be “a known polynomial mechanism, with no advantage over a simpler alternative.” It does not manufacture novelty by renaming it.

## 4. Selected mechanism and exact implementation contract

### 4.1 Input and parameter conventions

Fit feature means and population standard deviations on the training partition only. Set a zero-variance column's divisor to one and record it; do not drop columns inconsistently across methods. Use `s=(x−μ_x)/σ_x`. Fit target mean/std on training labels only, using divisor one for an exactly constant target; train in standardized target units and invert predictions for physical-unit metrics. Record constant-target cases explicitly. Core preprocessing has no winsorization, quantile transform or target logarithm. Method-native transforms are permitted in the practical comparison and logged.

For `R` components and `k ∈ {2,3,4}` factors,

\[
t_{rj}=w_{rj}^{\mathsf T}s,\quad
q_{rm}=\frac{e_m(t_{r1},\ldots,t_{rk})}{\sqrt{\binom{k}{m}}},\quad
\hat y=b+v^{\mathsf T}s+\sum_{r=1}^{R}\sum_{m=2}^{k} c_{rm}q_{rm}.
\]

Parameter shapes are `W[R,k,d]`, `C[R,k−1]`, `v[d]`, `b[1]`. The scalar trainable parameter count is

\[
P=Rkd+R(k-1)+d+1.
\]

There are **no factor biases** in this primary form, to make degree relative to centered `s` unambiguous. The model is a shallow degree-`k` polynomial. An independent-degree CP baseline includes affine factor offsets so the comparison does not unfairly remove its lower-order flexibility.

Call `q_rm` a degree component, not a functional-ANOVA interaction. Projected terms can express univariate nonlinearities; different degrees need not be orthogonal under the data distribution. In particular, degree three can correlate with an affine component. Degree ablation is architectural attribution, not causal feature attribution.

Initialize factor rows as `W = factor_scale * torch.randn(R,k,d)/sqrt(d)`, with independent draws and default `factor_scale=1`. Thus the **standard deviation** is `factor_scale/sqrt(d)` and the variance is `factor_scale²/d`; do not pass `1/d` as a Normal-distribution standard deviation. Initialize `C` independently with standard deviation `0.01/sqrt(R(k−1))`; initialize `b=v=0`. This is a transparent starting configuration, not a proven optimum. Include factor scale `{0.25,0.5,1}` and output scale `{0.01,0.1}` in development tuning. Do not initialize all factor weights to zero or copy identical rows: degree-two-and-higher gradients can vanish or preserve restrictive symmetries. Do not set every output coefficient to zero if immediate factor learning is intended.

### 4.2 Stable degree recurrence

The following is reference code for a proposed module, using real PyTorch tensors:

```python
import math
import torch

def degree_terms(t: torch.Tensor) -> torch.Tensor:
    """t: [B,R,k]; return e_0,...,e_k as [B,R,k+1]."""
    k = t.shape[-1]
    e = [torch.ones_like(t[..., 0])]
    e += [torch.zeros_like(t[..., 0]) for _ in range(k)]
    for j in range(k):
        old = e
        e = [old[0]] + [old[m] + t[..., j] * old[m-1]
                        for m in range(1, k+1)]
    return torch.stack(e, dim=-1)

def product_residual_forward(s, W, C, v, b, scale=None):
    # Reshape + matmul is preferred initially; no custom einsum kernel required.
    R, k, d = W.shape
    t = (s @ W.reshape(R*k, d).T).reshape(-1, R, k)
    e = degree_terms(t)[..., 2:]
    denom = s.new_tensor([math.sqrt(math.comb(k,m))
                         for m in range(2,k+1)])
    q = e / denom
    if scale is not None:
        q = q / scale  # Fixed [R,k-1] buffer, not a moving statistic.
    return b + s @ v + (q * C).sum(dim=(-2,-1))
```

The recurrence is ordinary polynomial arithmetic with no division by a factor and no nondifferentiable sign operation. It works at zero and for negative inputs. Out-of-place updates are deliberate: using already-updated degrees in the same iteration changes the polynomial and can also break autograd.

For the original equal-degree-weight residual, an `O(k)` recurrence avoids subtracting nearly equal terms:

```python
def higher_degree_residual(t):
    """Exactly prod(1+t)-1-sum(t) in real arithmetic."""
    linear = torch.zeros_like(t[..., 0])
    higher = torch.zeros_like(t[..., 0])
    for j in range(t.shape[-1]):
        z = t[..., j]
        old_linear = linear
        higher = higher * (1 + z) + old_linear * z
        linear = old_linear + z
    return higher
```

Neither recurrence guarantees good conditioning for huge factors or arbitrary cancellation among monomials. They remove one avoidable subtraction failure. Use them in all equivalent product/all-subsets baselines; report the naive subtraction implementation only as a numerical control.

### 4.3 Optional fixed scale calibration

The default scientific model uses the combinatorial divisor above. A **prespecified secondary ablation** computes `scale[r,m] = RMS_train(q_rm)` at initialization, in streaming CPU float64, before any label-based fitting. It uses training features only, includes the full training partition rather than the validation/test set, and freezes the resulting scale permanently. Transfer it as a float32 buffer for training; store the exact initial weights and scale in checkpoints.

Use an uncentered RMS to avoid changing the intercept convention. In dimensionless standardized units set `scale=max(RMS,1e-6)` for positive RMS; set scale to one for exactly zero RMS. Log every floor/zero event and inspect degenerate data/initialization. The floor is a declared numerical convention, not a stability guarantee; sensitivity values `1e-8` and `1e-4` are development-only ablations unless frozen separately. Do not redraw a confirmation initialization, clip features, or discard a failed seed. A zero component remains zero; its nonzero scale does not manufacture a signal.

Fixed calibration is **not continuing normalization**. As `W` changes, feature scales can drift. Log their held-training-probe RMS and the contribution norms `||c_rm q_rm||`; do not update scales in response to validation/test behavior. Compare with common input standardization and with the same fixed-calibration option applied to CP, HOFM and CCP features.

For an exact expressivity control, map calibrated coefficients to raw coordinates by `C_raw=C_cal/scale`. Check identical predictions and appropriately transformed gradients. Weight decay on `C_cal` and on `C_raw` is not the same functional penalty. In a dedicated attribution experiment use zero coefficient decay or an explicit penalty on `C_cal/scale`, and allow per-degree learning rates in the raw model. A transformed plain-SGD update can be made exactly equivalent using raw coefficient learning rate `η/scale²`; keep the rest of the parameters and regularizer consistently mapped. If these equivalent trajectories disagree on CPU float64, fix the harness before drawing conclusions.

### 4.4 Literal and normalized controls

- `shift_product`: `b + Σ_r a_r ∏_i(s_i+w_ri)`, with exact dimension/fan-in logged. For designated relevant coordinates, distinguish an oracle-support version from a learned/all-input version.
- `affine_factor_product`: `b+vᵀs+Σ_r a_r∏_{j=1}^k(u_rjᵀs+c_rj)`; this is the CP/Pi-Sigma competitor, not a novel alternative.
- `signed_root`: apply the explicitly defined signed kth power to the product. Report undefined/singular derivative behavior honestly.
- `radial_power`: apply `N_ε` to the same product, with `ε` in the product's normalized units. Tune `ε` on development over log-spaced values around the training product magnitude; do not compare different unstated units.
- `principal_complex_root`: diagnostic only, CPU complex128 oracle; use the documented principal branch and test crossings explicitly.

Compute small products by successive multiplication, never by `exp(sum(log(abs(f))))` plus a detached sign in the core. The latter changes behavior at zero and can give misleading gradients. Large-fan-in log-domain methods require their own zero/sign/backward proof. Do not call an epsilon-clamped approximation the exact original neuron.

### 4.5 Mandatory correctness invariants

1. Degree recurrence equals enumerated monomials for `k≤8`; higher residual equals the sum of degrees ≥2.
2. `k=2` agrees with a tied bilinear implementation in predictions, loss, gradients and a mapped optimizer step.
3. Scalar product gradients at `[0,2,3]` are `[6,0,0]`; at `[0,0,3]` all first factor derivatives are zero.
4. CPU float64 finite differences/`gradcheck` agree away from singular root points. Test factor weights and readout weights, not only input derivatives.
5. Batch size one, partial batches, noncontiguous views and prediction chunks yield the same samplewise function.
6. No preprocessing/calibration accesses validation or protected test observations.
7. Equivalent raw/calibrated parameter mappings preserve predictions; changes in penalty and optimizer coordinates are explicit.
8. MPS and CPU float32 agree with CPU float64 on well-conditioned probes within frozen, condition-aware tolerances. Begin with absolute/relative tolerances around `1e-5/1e-4` for outputs and `1e-4/1e-3` for gradients, then validate against measured oracle roundoff; these are debugging tolerances, not scientific success thresholds.
9. Nonfinite values become recorded failures. No automatic `nan_to_num`, unreported clipping or selective seed deletion.

## 5. Evaluation designed before implementation

### 5.1 Three comparison tracks

**Track A — mechanism and learning, primary.** Fresh, independently generated regression worlds; no pretrained weights. Compare the proposed training recipe with the closest factorized models and strong simple/additive methods. Report both capacity-controlled learning curves and a common local fitting/tuning budget. An exact algebraic clone is a correctness control, not a rival to be beaten.

**Track B — practical finite-cohort transfer.** Concrete and Airfoil grouped holdouts. Preserve strong practical settings of official packages, including ensembling and native preprocessing. Compare actual risk, project-specific adaptation cost, prediction latency, storage and memory. Here forcing every method to have identical trainable parameter counts would obscure the relevant tradeoff.

**Track C — boundary and attribution diagnostics.** Literal high-fan-in products, roots, phase composition, numerical limits, representation oracles and adversarial cases. These identify causes and limits; they do not substitute for Track A or establish broad practical superiority.

The primary hypothesis concerns Track A's declared family portfolio. It is not a claim to beat every contemporary ML method. Track B includes stronger pretrained context comparisons; neither track imports incomparable published benchmark scores as local results.

### 5.2 Precisely defined local baselines

Every adapter must implement the interfaces in §9. Freeze its package version or project commit, resolved constructor parameters, initialization, optimizer, training policy and transform state. Include the author's documented default as a mandatory candidate where applicable.

| ID | Exact mechanism / configuration family | Role and reproduction requirement |
|---|---|---|
| `ridge` | Affine least squares with L2 penalty; unpenalized intercept; CPU float64 stable solver | Additive lower bound. Match a direct normal-equation/SVD solution on a small problem |
| `additive_spline` | Sum of per-feature cubic spline features plus ridge, no cross-products; degree-matched univariate polynomial variant | Detect a win explained by univariate curvature. Knots fit on train only |
| `poly_ridge` | Full monomials through degree `m=1..4`, including squares; intercept separate; feature scaling learned on train only | Mandatory on `d≤8`, especially both real datasets. Explicit reference enumerates exponent vectors |
| `poly_kernel` | Kernel ridge with `K(x,z)=(γ xᵀz+c0)^m`, `m=2..4`, `c0∈{0,1}`, `γ>0`; CPU float64 Cholesky/SVD | Strong non-enumerating polynomial control. Its monomial weighting differs from ordinary polynomial-feature ridge; report both where affordable |
| `rbf_kernel` | Kernel ridge, `K(x,z)=exp(−γ||x−z||²)` | Strong smooth small-data competitor. Median-distance scale estimated from train only |
| `mlp` | Affine input/output; 1–3 hidden layers with GELU; trainable skip from inputs to output; dropout/weight decay selectable | Conventional trained baseline. Match width from measured parameter count and verify positive-control learning |
| `gated_mlp` | Hidden transform `A_out[SiLU(A_gate s+b_gate) ⊙ (A_value s+b_value)]` plus affine skip; optional 1–3 stacked blocks | Tests whether ordinary gated computation explains the effect. Count both projections; plain bilinear no-SiLU case is a separate exact control |
| `hofm_shared` | `b+vᵀs+Σ_rm θ_rm e_m(p_r⊙s)`, `m=2..k`, `k∈{2,3,4}`; shared factors across degrees | Raw-coordinate ANOVA, excluding repeated coordinate powers. DP and explicit subsets must agree |
| `hofm_independent` | Separate `p_(m,r)` per degree with the same degree range and explicit readout coefficients | Tests cost of sharing. Report both equal rank and equal parameter budgets; they answer different questions |
| `all_subsets` | `b+vᵀs+Σ_r λ_r[∏_i(1+p_ri s_i)−1−Σ_i p_ri s_i]` using stable recurrence | Closest raw-coordinate counterpart to the proposed residual. Full input degree is a deliberate baseline, not assumed beneficial |
| `cp_affine` | Both a single degree-`k` affine-factor bank and independent banks at degrees `2..k`, `k∈{2,3,4}`, with affine skip | Pi-Sigma/CP competitor, arbitrary signed readout coefficients. Tune rank allocation, including zero rank in unused banks; share one family budget |
| `ccp` | `h1=U1ᵀs`; `hj=(Ujᵀs)⊙h_(j−1)+h_(j−1)` for `j=2..k`; linear readout plus affine skip | Precisely specified Π-Net CCP component. This is an equation reproduction, not reproduction of all published Π-Net systems |
| `dcn_matrix` | `h0=s`; `h_(l+1)=h_l+s⊙(W_l h_l+b_l)`; affine readout, `L=1..3` | Full-matrix DCN-V2 cross component, no unmentioned preactivation/diagonal scale/mixture experts. Check forward/gradient parity against an Apache reference |

Group closely related variants into predeclared baseline families for tuning rather than giving each variant a separate unrestricted budget. At minimum use these eight families: linear/additive; explicit/kernel polynomial; RBF kernel; MLP/gated; HOFM/all-subsets; CP/projected-independent; CCP; DCN. The projected-independent control is defined in §8.1 and is included in the CP family search. Publish every evaluated variant's results, including adverse ones. Family selection is part of the pipeline and is charged to its cost.

For CP independent banks, tune allocation over equal rank, equal parameter expenditure per order, and single-order allocations, allowing zero-rank banks. For a total parameter tier, derive integer ranks by flooring the allocated budget divided by `m(d+1)+1` per component and reserve `d+1` for the affine skip. Also include the single maximum-degree affine bank: its offsets/constant factors permit lower-degree functions without forcing equal ranks at all orders. Record exact ranks and unused budget; do not pad. The same-rank sharing ablation is separate from this practical allocation search.

**Ridge convention and intercept:** use mean training squared error plus `λ` times squared coefficient/RKHS norm, with an unpenalized intercept. Thus a library solver using summed residuals receives `alpha=n_train*λ`. For kernel ridge, center the training kernel `Kc=H K H`, `H=I−11ᵀ/n`, and labels `yc=y−mean(y)`, then solve `(Kc+nλI)α=yc`. Center a test cross-kernel using its mean over training columns, the saved training-column means and the saved grand mean; predict `mean(y)+Ktest_centered α`. Target centering alone with an uncentered kernel is not the same unpenalized-intercept solution. Never estimate centering from protected target values. [R21]

For HOFM, a modern transparent equation implementation is safer than forcing archived build dependencies into the main environment. Validate degrees two/three against the permissive reference where it installs, and all implemented degrees against explicit monomials. A failure to build old software is not evidence against the mathematics. Use the same stable evaluator, scaling options and first-order optimizer in matched training comparisons; an additional official coordinate-descent solver is a useful optimization control if HOFM loses because of its optimizer rather than representation. [R01,R17]

For `k=2`, `product_residual` is already a sum of factorized quadratics. The exact counterpart must use the same free parameters, initialization and optimizer. A single term satisfies `(uᵀs)(vᵀs)=sᵀ(uvᵀ+vuᵀ)s/2`; it is not an arbitrary full quadratic. The stronger full-quadratic ridge control answers the latter comparison.

### 5.3 Contemporary practical comparators

Add the following in Track B; do not describe a stripped or reduced configuration as the full author's baseline.

| Method | Required practical configuration | Local feasibility / limits |
|---|---|---|
| TabM, official `tabm` | Include author's numerical embeddings and 32-member setting; train mean per-member loss, average predictions at inference; include embedding-free variant only as ablation | Version 0.0.3/source dependencies inspected; Apache-2.0. MPS/CPU path still locally probed [R18] |
| RealMLP, `pytabkit` | Official `RealMLP_TD_Regressor`, documented preprocessing/tuned defaults, explicit project validation membership | Source supports CPU/MPS; use package instead of reproducing selected tricks. Isolate dependencies if needed [R19] |
| CatBoost | Official CPU regressor with RMSE loss, explicit `eval_set`, fixed threads, documented early stopping | macOS arm64 CPU wheels documented; its GPU training is not a Metal path [R20] |
| Histogram gradient boosting | Official scikit-learn regressor; explicit group-respecting validation data, not its default random row split | Current documented `fit` accepts `X_val/y_val`; select a version with that interface or implement external staged selection [R21] |
| TabICLv2 | Official `TabICLRegressor`; checkpoint `tabicl-regressor-v2-20260212.ckpt`; eight inference views; squared-error evaluation uses predicted mean | Current code documents CPU/MPS; checkpoint listed at about 114 MB; BSD-3-Clause code/weights. Local smoke test precedes comparison [R22] |
| TabPFN 3.5 | Optional current local model with exact checkpoint/ensemble/preprocessing settings recorded | Account, license acceptance and eligibility/use terms apply. Include only with existing authorized eligible access; otherwise record exclusion. No paid API substitution [R23] |

TabICLv2 makes a contemporary public-weight comparison possible without making the core depend on a gated model. Inspect the selected package's regression API rather than copying classifier-only parameters. Use its own documented inference path, disabling accidental mixed precision for the initial parity check. If a supported mixed-precision path materially improves its normal performance, include it as a documented practical configuration after parity, not as an undisclosed asymmetry.

The pretrained models have reusable prior training and may have benchmark-selection exposure even when their pretraining data are synthetic. Log those assets separately from local fitting/tuning. Never use their predictions to construct candidate training targets. These comparisons assess local deployment tradeoffs, not equal total lifetime compute or equal prior information. TabArena's current scope is much broader; this project makes no general tabular state-of-the-art claim. [R22–R24]

**Nonlocal context:** full Π-Net/regularized-polynomial image systems, ULD-Net's reported multi-A100 training, web-scale DCN deployments, and foundation-model pretraining are context only. Their published scores must not share a head-to-head table with the present new splits as though they were comparable.

### 5.4 Synthetic workloads and seeds

Separate four sets of synthetic worlds by explicit seed namespace: `unit`, `development`, `pilot`, `confirm_v1`. After a methodological amendment, use `confirm_v2` with new worlds and retain the previous result as exploratory. Derive random generators using NumPy `SeedSequence` from integers encoding `(protocol_version, stage, stratum, world, purpose)`. Do not use Python's process-randomized `hash()` or a shared mutable generator whose draw order changes when methods are added.

**Small algebra/representation tasks:** `x1+x2`, `x1*x2`, `x1+x2+x1*x2`, `(x1²−1)/sqrt(2)`, and Boolean parity with known factors. Include oracle parameters and a fixed-feature linear solve. These are positive/negative correctness controls, never proof of general superiority.

**Primary generated population:** twelve equally weighted strata:

- Degree `m ∈ {2,3,4}`.
- Active **nonlinear** coordinate count `q ∈ {8,32}`, within `d=32` observed features; the separate linear signal may use all coordinates.
- Alignment `axis` or `rotated`.

For each independently drawn world:

1. Draw all inputs independently from `Normal(0,I_d)`. For `axis`, use the identity rotation; for `rotated`, draw an independent Haar orthogonal matrix by QR of a Gaussian matrix and correct the signs of `R`'s diagonal. The marginal observed-input distribution is then identical across alignment regimes.
2. Select `q` active coordinates uniformly without replacement. Draw `M=min(2q,C(q,m))` distinct subsets of size `m`, uniformly without replacement. Draw one independent normal coefficient per subset and normalize the coefficient vector to Euclidean norm one.
3. Let `z=Qx` and `g(x)=Σ_S a_S∏_(j∈S) z_j`. Each monomial uses distinct coordinates; Gaussian independence implies `E[g]=0` and `Var(g)=1`. This generator is coefficient-sampled, not produced by the candidate's factorization.
4. Draw an independent unit vector `v0`; define `f(x)=0.5 v0ᵀx+sqrt(0.75) g(x)`. The linear and distinct-coordinate degree-`m` components are orthogonal in this Gaussian population, so `Var(f)=1` without test-derived scaling.
5. Draw training/validation labels `y=f(x)+η`, `η~Normal(0,0.1²)`. Keep the latent `f` unavailable to training and tuning adapters. It is available only to the final synthetic evaluator.
6. The primary sample-size anchor is `n_train=1024`, `n_val=512`. Learning-curve extensions use nested training sizes 256 and 4096 within the same world and must preserve the resulting dependence. Primary inference uses only the anchor; these sizes are design points, not limits on later justified scaling.
7. Select a fixed Monte Carlo test size from the pilot, starting at 16,384 and evaluating in chunks. Increase it on pilot worlds until its contribution to the variance of paired log-risk is at most one tenth of the between-world contribution. Freeze it before generating confirmation predictions. Adding test points alone never adds independent worlds.

The active-dimension and rotation mixture intentionally includes favorable and adverse conditions. The low-dimensional cell tests exploitable structure; the full-dimensional cell challenges restrictive factorization. No selection on which cells “work” is permitted after confirmation.

Here `q` measures the nonlinear subspace only. Because `v0` generally spans all 32 observed coordinates, the other coordinates are not wholly irrelevant. A separate distractor diagnostic confines both linear and nonlinear targets to the same relevant subset before adding unused coordinates.

**Required secondary controls:** additive linear; additive nonlinear `Σ_i sin(x_i)/sqrt(d)` with variance derived analytically or from a separate fixed reference draw; univariate polynomial/spline targets; mixed independent degrees; candidate-aligned shared-factor low-rank targets; independently factored degrees; noise-only targets; irrelevant-feature growth; correlated Gaussian inputs; heavy tails with finite moments appropriate to the degree; covariate rescaling; and radial extrapolation `x_test=c x`, `c∈{0.5,2}`. State distributional changes explicitly. For heavy tails, choose degrees of freedom giving finite target squared error; e.g. Student-t with only three degrees of freedom is invalid for a degree-four MSE experiment. Use `ν=20` initially, and scale each coordinate to unit variance.

High-order parity and full-input products use separate fan-in sweeps `k=2,4,8,16,32,64,128` with carefully specified factor distributions. Known-support/oracle parity is a representation diagnostic; learnable support selection requires an NMU comparator and distractor features. If all methods are at the noise/optimization floor, the test is insensitive rather than evidence that multiplication has no value.

### 5.5 Real datasets, provenance and protected groups

Use the official UCI archives, preserve raw bytes, record SHA-256 and license text, and attribute their creators. All selected UCI pages state CC BY 4.0. No account is required. Do not silently replace an unavailable file with a transformed Kaggle/OpenML copy. [R25–R28]

| Dataset and role | Inputs/target | Protected grouping | Scope |
|---|---|---|---|
| Concrete, UCI 165; mandatory | 1,030 rows, eight numerical inputs; strength in MPa | Exact tuple of the first seven mixture inputs, omitting age; measured 428 keys | Generalization to held mixture compositions within this source, not new laboratories |
| Airfoil, UCI 291; mandatory | 1,503 rows, five inputs; scaled sound pressure in dB | Exact tuple of angle, chord, free-stream velocity, displacement thickness; omit frequency; measured 106 keys | Held operating-condition frequency sweeps, not unseen airfoil families |
| Yacht, UCI 243; robustness only | 308 rows, six inputs; dimensionless resistance | First five hull inputs, omitting Froude number; measured 22 keys ×14 rows | Few hull groups; likely low-power transfer test |
| Energy, UCI 242; optional simulation check | 768 rows, eight inputs, heating/cooling targets | Base geometry rather than random rows | Ecotect simulations; two outputs are not two independent real-world domains |

Airfoil's metadata labels one angle field “binary,” but the original numeric column is an angle in degrees. Preserve numeric values; do not infer categorical encoding from that label. Concrete's original workbook is XLS, requiring a compatible reader such as `xlrd`; preserve its full stored precision. Group on parsed exact source values before scaling, not rounded standardized values.

Measured archive/member hashes, useful for identifying the exact snapshot:

| Dataset | Archive SHA-256 | Data member SHA-256 |
|---|---|---|
| Concrete | `dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7` | `710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4` |
| Airfoil | `5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c` | `74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3` |
| Yacht | `aa52b68f88c4bb552187a53ef4c5753fa178f6a36035a3771c5bc04e078487ac` | `00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1` |

Archive URLs are linked in §12. A changed hash triggers inspection and a recorded dataset-version decision; it does not automatically mean corruption. Assert row/column counts, finite values, column order, duplicate counts and group counts. The measured Concrete snapshot contains 996 distinct input vectors and 1,005 distinct complete rows. Keep repeated outcomes/identical rows in the same group. Retain all records in the primary analysis; drop only exactly identical complete rows in a prespecified sensitivity analysis. Do not average different outcomes at identical inputs silently.

**Protected split:** sort unique group keys by a stable serialization, shuffle with a saved generator seed, and reserve `ceil(0.2 G)` entire groups for one final test. Thus Concrete reserves 86 keys and Airfoil 22. Preserve the remaining groups for development; use three-fold group CV within them for tuning. Fix all membership before performance comparisons. The repository must separate sealed labels from fit adapters and record test access. This is procedural protection, not a claim that public benchmark labels are secret or uncontaminated.

For final fitting, use the full 80% development partition with a frozen training-duration rule learned from internal CV: neural/tree early-stopping epochs/iterations are the median selected duration across folds, rounded up, then refit without consulting the protected test. **Default: median-duration full-development refit.** Methods without iterative early stopping refit directly. Preprocessing is refit only on those development groups. Candidate settings are tuned separately for each real dataset; this is transfer of the architecture/recipe, not zero-shot transfer of the synthetic hyperparameters.

Freeze resolved learning rates and schedule semantics from the selected CV configuration. For CatBoost explicitly pass its resolved rate, fixed iteration count and `use_best_model=False`; its automatic rate can depend on iteration count. For histogram boosting set `early_stopping=False` in the fixed-duration refit. For neural schedules preserve the original schedule horizon and stop at the frozen duration, or use a documented native refit schedule fixed before test. Verify that every intended development row participates: an official estimator may otherwise introduce a hidden `val_fraction` split. If its supported API cannot execute the stated refit faithfully, freeze a native-refit exception or a retained, group-respecting inner-validation policy before protected evaluation; record actual rows/duration. No silent package-specific random-row split is allowed.

For each dataset, designate the real-data comparator before test by the lowest pooled inner-CV group-MSE across the complete mandatory local comparison set, averaged over prescribed fit seeds. Pool individual group squared errors across folds rather than averaging fold RMSE. Freeze that baseline identity/configuration, the candidate's chosen configuration and any documented access exclusions. Publish the separate test results of every compared pipeline as well.

Exact feature tuples are proxies for dependence. Concrete mixes may share undocumented sources; Airfoil conditions share airfoil/laboratory structure. Grouping prevents obvious frequency/age leakage but cannot establish physical independence. Report group-bootstrap uncertainty as conditional on this proxy-group exchangeability assumption. Add leave-chord/nearest-condition sensitivity for Airfoil and age-range sensitivity for Concrete; these are robustness analyses, not automatic independent confirmations.

### 5.6 Tuning, training and resource fairness

**Development training contract for project neural models:** MSE loss in training-standardized units; AdamW starting at `lr=1e-3`, `betas=(0.9,0.999)`, `eps=1e-8`, `foreach=False`; effective batch initially 128; initial probe horizon 256 epochs, patience 64, strict validation improvement (`min_delta=0`), validation once per epoch, and no learning-rate scheduler by default. Restore the best validation checkpoint. These are probe settings, expanded by the duration rule below, not a universal training limit. Tune learning rate, weight decay, factor/output scale, rank/width, degree/depth and dropout where applicable. Native practical packages keep documented schedules. No test-based learning-rate or failure recovery.

Initial search spaces are explicit starting ranges, to be expanded during development if the optimum hits a boundary:

| Family | Initial search choices |
|---|---|
| Project neural families | `lr` log-uniform `[1e-4,1e-2]`; weight decay `{0,1e-6,1e-4,1e-2}`; effective batch `{64,128,256}`; gradient clip `{none,1,10}`; degree/depth as specified; nonlinearity-specific dropout `{0,.1,.2}` |
| Product/CP/HOFM/CCP | Factor and output initialization scales from §4; fixed RMS calibration `{off,on}`; degree-specific learning-rate multiplier `{.3,1,3}` as a fairness option where calibration is under test |
| Ridge/polynomial/kernel | Penalty grid `10^j`, `j=-8..4`, then refine neighboring log intervals; polynomial `γ∈{.1,1,10}/d`; `c0∈{0,1}`; RBF `γ` equal to `{.1,1,10}` times reciprocal train median squared distance, then refine |
| Additive splines | Cubic splines, knots `{4,8,16}`, train-quantile knots, no feature crosses; same ridge-penalty search |
| CatBoost | Include default; depth `{4,6,8}`; learning rate log-uniform `[.01,.2]`; L2 leaf penalty log-uniform `[1,100]`; sufficient iterations with explicit validation/early stopping |
| Histogram boosting | Include default; leaf count `{15,31,63}`; minimum leaf size `{5,10,20}`; learning rate `[.01,.2]`; L2 `{0,.01,1,10}`; explicit validation |
| TabM/RealMLP | Official defaults plus the inspected author's search space, archived into config; include embeddings/ensemble behavior, preprocessing and author duration schedules |
| TabICLv2 | Frozen default eight-view model as mandatory point; an optional separate development-selected view/cache/fine-tuning configuration if its additional adaptation cost is charged |

These are development proposals, not claims that the best hyperparameters are known. Use deterministic Sobol/random-search proposals with a saved search seed. Include defaults before random proposals. Do not use a different search algorithm for the candidate unless all applicable baselines receive it or the difference is the explicit treatment.

**Capacity comparisons:** construct approximately logarithmic parameter tiers beginning around `2^10`, `2^12`, `2^14` and `2^16` scalars, restricted by the memory calibration. For each architecture choose the largest valid width/rank at or below the tier and record its exact count; do not pad unused parameters. Treat these as response-curve points. The primary anchor tier is selected on development from the smallest tier at which both the candidate and a serious interaction baseline learn the nontrivial control above the null/noise floor; if no tier does, diagnose before confirmation. Do not select the tier maximizing candidate-minus-baseline advantage. Full-kernel and explicit-basis solvers need not fit a neural parameter tier; they remain strong separate resource-accounted controls.

**Duration calibration:** start with the 256-epoch project-model probe, inspect validation/training curves, and double the horizon when a competitive method is still improving at its end. Set each family's confirmation horizon to `ceil(2 × pilot 90th-percentile best-validation epoch)`, never below a demonstrated necessary horizon, with patience `ceil(horizon/4)`. Use the same data-access/validation cadence for comparable models. Freeze these integer values before confirmation. A method that repeatedly reaches the horizon while still improving is under-trained; do not use its score for an unqualified superiority claim. Native defaults with fixed schedules are evaluated as intact recipes and any duration extension is explicit.

**Tuning-budget rule:** obtain search-learning curves in development at 8, 16, 32, ... completed proposals, counting failed trials and wall time. Continue common budget expansion until (a) no serious comparator's optimum lies on a search boundary and (b) two successive budget doublings change its best independently reassessed development risk by less than the reassessment's bootstrap uncertainty. This stopping criterion uses development only. Set the shared time allowance per family to the largest allowance required by the serious competitors; also report the equal-proposal-count results. Include at least two search seeds in this reassessment. If convergence is unaffordable, retain the finite-budget claim and label the optimal-performance question unresolved; do not weaken the baseline quietly.

This rule avoids declaring a fixed small number of trials universally adequate. It does not require running obviously failed configurations forever. Early termination is allowed for nonfinite loss, prerequisite violations or a frozen validation-based pruning rule shared across comparable methods. Rung budgets must be long enough for the slowest viable polynomial method; retain a full-budget default control to detect pruning bias.

“Serious comparator” includes every one of the eight required core families and every mandatory practical package; poor initial settings cannot remove the closest prior art from budget reassessment. For the first tuning pass use four independent development worlds per stratum, shared across methods; use a disjoint reassessment set of four worlds per stratum. If rankings/budget curves remain too noisy to assess, expand development/reassessment before freezing, not after test. Development is not confirmatory evidence, and budget saturation is not proof of a globally optimal model. [R36,R37]

**Primary baseline selection policy:** on every confirmation world, train the frozen family configurations and select the baseline family using noisy validation MSE, never test or latent targets. Candidate degree/rank/scaling policy is frozen from development; any within-world candidate selection uses the same validation information and is charged. The primary comparator is this explicit validation-selected portfolio. Its total selection cost exceeds one family's cost and must be reported. Publish individual baseline test outcomes too. Beating the portfolio is not automatically beating every individual baseline; any stronger all-baseline claim requires simultaneous inference.

Share applicable generic improvements, including stable product evaluation, target standardization, optimizer choices and useful initialization. In Track B keep method-native preprocessing; add a common-preprocessing ablation if a practical gain is attributed specifically to the neuron. Parameter matching, FLOP matching and wall-clock matching are different comparisons. Report parameter/error curves and error-versus-total-fit-time curves rather than asserting that one stands in for the other.

### 5.7 Metrics and confirmatory inference

**Synthetic primary risk:** after inverting training target scaling, for method `a`, stratum `s`, independent world `w`, optimizer seed `j`,

\[
R_{swaj}=\frac1T\sum_{i=1}^{T}\bigl(\hat f_{swaj}(x_i)-f_{sw}(x_i)\bigr)^2+0.01.
\]

The added `.01` is the known irreducible noise variance in the canonical unit-variance target population. It is the expected fresh-noise predictive risk, not an arbitrary epsilon hiding exact zero error. Do not add `.01` in scaled units without transforming it consistently.

Average **risks**, not predictions, over the frozen number `J` of optimizer seeds: `Rbar_swa=mean_j R_swaj`. This is an arithmetic estimate of seed-averaged risk under the stated selection policy. A prediction ensemble is a different named method with all members' fit, storage and inference charged.

For candidate `C` and the validation-selected baseline policy `B`, define

\[
D_{sw}=\log\bar R_{swC}-\log\bar R_{swB},\qquad
\Delta=\frac1{12}\sum_s\frac1{N_s}\sum_wD_{sw}.
\]

This frozen endpoint is a **finite-`J` replicated-policy log-risk effect**, equally weighting strata. In general, `E[log(mean_J R)]` is not `log(E_seed[R])`; different optimization variances can create different finite-seed bias. Report `J`, pilot sensitivity to doubling `J`, and arithmetic risks as well. `exp(Δ)−1` gives the relative geometric-mean effect for this defined endpoint. Geometric averaging can conceal severe regressions. The independent analysis unit is the **world**, not a seed, minibatch, test row or timing repeat.

Use a stratified paired **percentile** bootstrap of worlds, retaining all method/seed outcomes of each world; use the same resampled worlds for every paired effect. Strata are generated independently, so cross-stratum blocks are unnecessary in the primary design. For learning curves and other reused-world sensitivities, retain whole-world blocks. Report a two-sided 95% interval for `Δ` and all prespecified guardrails. In the pilot, start with 2,000 bootstrap resamples and double using independent bootstrap random seeds until endpoint Monte Carlo variation is below one hundredth of the planned interval half-width; freeze count and seeds before confirmation. If pilot resampling/coverage diagnostics show unacceptable skew or rare-failure sensitivity, amend the estimator/sample design before the freeze rather than changing the interval after seeing the result.

**Pilot-based fixed sample:** estimate paired world-level variances `s_s²` from independent pilot worlds, after averaging training-seed risks. Begin with eight pilot worlds per stratum; increase only the pilot if its variance estimate is dominated by a few worlds or remains too uncertain for sample planning. Select `J` by pilot variance decomposition: start at three optimizer seeds and increase until the seed-averaging contribution to paired-risk variance is at most one quarter of between-world variation, or report why resolving optimizer variability consumes the local budget.

Implement that decomposition by resampling seed indices within each pilot world jointly across all methods, recomputing validation-family selection for each resample. Estimate the conditional variance of `D` from these seed resamples. Estimate test Monte Carlo variance by independent equal-sized test chunks with fitted models and validation choices held fixed, scaled to the intended full test size. Estimate residual between-world variance as the stratum variance minus the mean estimated seed/Monte Carlo components, truncated at zero; bootstrap worlds to expose uncertainty in this approximation. If subtraction is unstable, collect more pilot worlds/seeds rather than trusting a negative variance. If paired variance is numerically zero, run exact-equivalence/A-A checks and resolve a deterministic or insensitive comparison; do not divide by zero, create an infinite sampling loop or claim population certainty. A changed nondegenerate population requires a pre-confirmation amendment.

For balanced `N` worlds per stratum, estimated `SE(Δ)=sqrt(Σ_s s_s²/(12²N))`. The default precision objective is a 95% half-width no larger than one tenth of the pilot's typical between-world log-risk standard deviation, `h=.1 sqrt(mean_s s_s²)`. This resolves effects small relative to natural task variability; it is **not a required improvement for success or a practical-value threshold**. Compute `N` from that formula, inflate using the upper bootstrap uncertainty bound on variance, and verify interval coverage with pilot resampling/simulation. Also publish the implied detectable log-risk effect at 80% power, approximately `(1.96+.84)SE(Δ)`, translated into relative risk units. If a concrete application later supplies a smaller decision-relevant effect, plan the additional sample before looking at its confirmation results.

Freeze `N_s`, `J`, test size, selected configurations, budgets, failure policy and analysis code before confirmation. Do not add worlds until a p-value becomes favorable. A larger follow-up is a new registered study with fresh worlds. If the required fixed sample cannot be run locally within measured storage/operational constraints, execute a clearly labeled feasibility study and report its detectable-effect limit; do not pretend its wide interval falsifies a small effect.

**Decision:** an interval entirely below zero supports lower average log risk for the declared portfolio comparison. An interval spanning zero is inconclusive about the sign at that precision. An interval above zero is evidence against this implementation in this population. Failure to reject zero does not prove equivalence. Practical value additionally depends on absolute effect, cost, stability and application tolerance; absent an application, provide the measured tradeoff without declaring a tiny effect transformative.

**Real-data metrics:** primary per-dataset metric is equal-group RMSE in original units,

\[
\operatorname{RMSE}_{group}=\sqrt{\frac1G\sum_g\frac1{n_g}\sum_{i\in g}(\hat y_i-y_i)^2}.
\]

Also report ordinary row-weighted RMSE, group/row MAE, worst-group error, optimizer-seed dispersion and test-group counts. Use paired cluster bootstrap of held-out groups for differences/ratios, keeping all rows of each group together. This uncertainty is conditional on fitted pipelines and the proxy-group assumption; it does not cover full training/HPO variability. Display leave-one-test-group influence, especially for Airfoil's 22 held-out groups. No naive t-test across overlapping CV folds or training seeds. No MAPE near zero. A macro summary across two datasets is descriptive finite-suite evidence, not an estimate of the universe of engineering tasks.

For real-data stochastic pipelines, compute group MSE per seed, average squared errors over seeds and groups, and take **one** square root. Do not average seed RMSEs or average predictions unless the latter is a separately charged ensemble. Keep all seed outcomes together when a test group is bootstrapped; deterministic methods do not gain extra observations by copying predictions. Use two 97.5% paired percentile group-bootstrap intervals for the two frozen candidate-versus-designated-comparator differences, a Bonferroni simultaneous-interval approach with approximate coverage under the stated group assumptions. No unspecified Holm p-values or CI-overlap test is used. Other methodwise or ablation hypotheses either receive a separately frozen multiplicity plan or remain exploratory. An omnibus synthetic primary test does not license selecting a winning degree, seed or dataset after inspection.

### 5.8 Operational failures and guardrails

Use the same frozen operational fallback for comparable iterative methods. With a validation partition, retain the best finite validation checkpoint; if none exists, predict the training-target mean. For a fixed-duration full-development refit with no validation, retain the **last finite completed checkpoint** before failure, explicitly flag premature termination, and otherwise use the development-target mean. Noniterative solver failures use a documented stable-solver retry (such as SVD instead of a failed Cholesky factorization) with identical objective; if none succeeds, use the same mean predictor and record failure. Score those actual predictions in pipeline risk. Count and publish the original failure, fallback use, nonfinite steps, total attempted time and the unrescued method's failure rate. An operational fallback prevents undefined averages; it must not make a fragile method look reliably trained.

A positive mean must be accompanied by per-stratum effects, high-error tails, additive/nonpolynomial regressions, failure rates, fit/tuning time, batch-one latency, batch throughput, serialized model/context size and memory. A material regression cannot be erased by averaging. Without an application-specific tolerance, use explicit tradeoff wording rather than an invented universal noninferiority margin.

**Harness sensitivity checks:** duplicate a prediction vector under two method IDs and require exactly zero paired effect; inject a known offset and match a hand calculation; deliberately cross a duplicate group between splits and require rejection; permute row order and verify ID-aligned metrics; verify fitted scalers never touch excluded indices; compare exact algebraic clones before training; and use random independent labels to check that the system does not manufacture generalization. These checks are required because a persuasive but broken harness is more dangerous than a weak model.

## 6. M3 Max feasibility, software and calibrated resources

### 6.1 Hardware and environment discovery

Before dependency resolution or substantial downloads, collect machine architecture, chip, macOS version, physical/logical cores, total unified memory, free disk, current memory pressure/swap and available acceleration. Do not assume the maximum M3 Max memory configuration or infer memory from the chip name.

These are **existing macOS commands**, not proposed project modules:

```bash
uname -m
sw_vers
sysctl -n machdep.cpu.brand_string
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu
df -k .
vm_stat
sysctl vm.swapusage
```

Store selected hardware fields, not serial numbers or unrelated machine identifiers. Check that Python is native `arm64`, not an x86 interpreter under Rosetta. Record AC/battery state, power mode, ambient/thermal conditions where observable, CPU thread limits and significant competing applications for timing sessions.

**Verified version decision as of the research cutoff:** use native Python 3.12. For macOS ≥15, start with `torch==2.15.0`; for macOS 14, use `torch==2.14.1` for MPS. The 2.15 release notes explicitly changed the MPS OS requirement. Its wheel's macOS 14 installation tag does not establish MPS support on that OS. On older/unavailable MPS configurations use CPU; this plan does not require an operating-system upgrade. [R29,R30]

The newest release is not assumed bug-free. Run the exact operation probe. If a released version fails, first reproduce on CPU float64, inspect the failure, and test the documented earlier compatible release in a separate environment. Freeze the successful version before confirmation. Do not switch versions mid-study without a new run lineage.

### 6.2 Stack and dependency strategy

Use PyTorch eager float32 for project neural models and CPU NumPy/SciPy float64 for reference arithmetic and kernel/linear solvers. Add scikit-learn, pandas plus an XLS reader, pytest, a YAML parser, and plotting packages. Install official baseline packages only for selected comparisons. `torchvision`, `torchaudio`, CUDA, a remote GPU and paid APIs are unnecessary for the core.

An **existing package-installation pattern**, after selecting the compatible Torch version, is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.15.0 numpy scipy scikit-learn pandas xlrd pytest pyyaml matplotlib
```

Replace the Torch pin with `2.14.1` only under the rule above. Other versions are resolved from actual compatible published wheels and then locked; this document does not invent their future dependency combinations. Record distribution versions, wheel filenames/hashes, `pip freeze`, Python build, `torch.version.git_version`, `torch.__config__.show()` and source revisions. Prefer hashed requirements/lockfiles and archive the resolver result. `pip freeze` alone does not identify wheel bytes.

Keep `requirements-core.lock` and separate baseline locks when package requirements conflict. An external baseline may run in an isolated environment and communicate through immutable NPZ/JSON files. Do not weaken it merely to fit the candidate's environment, and do not inherit an author's old development lock as if it were a runtime requirement. Model initialization/fit/predict scripts must use actual installed API signatures verified in WP02.

Licensing: public code is not automatically permissive. Official Π-Net code inspected here uses CC BY-NC 4.0; independently implement the cited equations unless those reuse terms are appropriate and recorded. Polylearn is archived BSD-2-Clause; nimfm is MIT but adds a toolchain. Official DCN references, TabM and pytabkit provide permissive implementations; record the precise selected revision's notices. TabICL code/weights and TabPFN's separately restricted weights must not be conflated. Dataset redistribution requires attribution. No gated account/license acceptance is essential to the core.

MLX is a plausible optional engineering port, not a necessary research dependency. Inspected documentation supports complex64 and CPU-only float64 and describes lazy evaluation; timings must force materialization of updated state. A port must cover the candidate and matched baselines and pass parity before comparing frameworks. A faster MLX candidate against an unoptimized PyTorch control does not isolate the neuron. [R34]

### 6.3 Exact-operation compatibility gate

`torch.backends.mps.is_available()` is necessary but insufficient. In a fresh process set `PYTORCH_ENABLE_MPS_FALLBACK=0` and `PYTORCH_MPS_FAST_MATH=0` **before importing Torch**. Record these values, other backend/determinism flags and CPU thread counts. With fallback disabled, exercise the actual pipeline: reshape/matmul, elementwise products and additions, degree recurrence, output reductions, loss, backward, clipping if selected, AdamW update, serialization and reload. Check required external-baseline embeddings, indexing and prediction operations too. A wrapper must set the environment before launching a worker; changing it after a library has initialized MPS is not an adequate probe. [R31]

PyTorch's current MPS documentation explicitly excludes float64 and complex128 allocation/copy; fallback cannot repair that dtype limitation. Current complex64 support is operation-dependent, so neither assume it works nor assert that no complex operation is supported. Keep the default phase implementation in paired real float32, with CPU complex128 as the oracle. [R31,R32]

At every core accelerated adapter boundary assert float32 for **floating-point** inputs, regression labels, parameters and buffers. Preserve legitimate integer/bool indexing tensors, counters and RNG states; do not cast every buffer indiscriminately. Scikit-learn/NumPy frequently produce float64; convert deliberately before moving floating-point tensors to MPS. Practical external methods retain their verified native precision, recorded explicitly. Metrics are accumulated in CPU float64 after prediction.

Compare bounded random and adversarial zero/sign cases against CPU float64, then compare one optimizer step. Test several complete training steps; an inference-only probe misses optimizer state allocation. For nonsmooth/singular root cases report their mathematical pathology separately from implementation mismatch. No implicit CPU fallback in primary speed measurements. If a required kernel fails but CPU mathematics succeeds, use whole-method CPU execution for the affected controlled comparison and state the narrower acceleration claim.

### 6.4 Memory model

For the selected scalar-output model,

\[
P=Rkd+R(k-1)+d+1,\qquad
M_{\mathrm{Adam32,base}}\simeq16P\text{ bytes},
\]

covering weights, gradients and two Adam moments. Add optimizer temporaries, graph metadata, saved activations, framework/Metal workspaces, preprocessing arrays, model buffers and checkpoints. Core factor storage is `4BRk` bytes. The all-degree recurrence costs `O(BRk²)` operations and can retain multiple `B×R` intermediates; estimate its actual memory multiplier by calibration, not by counting just the final tensor.

For example, `d=64,R=256,k=4,B=512` gives **66,369** real trainable scalars, approximately **1.06 MB decimal** of base Adam state and **2.10 MB** for one factor tensor. Several recurrence copies remain small; framework overhead and other methods can dominate. These are arithmetic estimates, not measured peaks.

For a full polynomial basis through degree `m`, the number of columns including the constant is `C(d+m,m)`. At `d=64,m=4`, 814,385 columns over 10,000 float32 rows occupy about **32.6 GB**, before any solver copies. Do not materialize that baseline blindly. At the real-data dimensions, degree four needs only 495/126 columns. At larger dimensions use dual polynomial kernels or factorized models; a dual `n×n` float64 matrix uses `8n²` bytes, and factorization/work copies can multiply that. Never explicitly form a matrix inverse for ridge.

Use an initial conservative process budget `min(0.5 × physical RAM, 0.7 × MPS recommended working set)` for MPS calibration, then reduce it if idle system memory pressure requires more reserve. This is an operational starting safeguard, not a claimed device limit. Before model allocation, enforce the corresponding MPS allocator limit with `torch.mps.set_per_process_memory_fraction(budget_bytes / recommended_max_memory_bytes)`, checking that the denominator is positive and the fraction lies in `(0,1]`. Preflight host arrays and work copies separately: the MPS limit does not constrain all CPU allocations or total unified-memory pressure. For CPU estimate framework plus array peaks and retain comparable system reserve. Query actual values; never set an unlimited allocator/high-water budget to make a run appear feasible. Reject an oversized explicit polynomial matrix from its dimensions before attempting allocation. [R33]

Record process RSS/high-water, `torch.mps.current_allocated_memory()`, `driver_allocated_memory()`, `recommended_max_memory()`, swap change and memory pressure separately. They can refer to overlapping unified pages, so do not simply add RSS and driver bytes. Current allocation counters are not automatic peak counters; sample at stated boundaries and label the maximum as a **sampled peak**. [R33]

### 6.5 Calibration and hardware adaptation

Run one job at a time initially. In a disposable process, allocate the actual largest planned model, warm it, perform full forward/backward/optimizer steps, validation and checkpoint creation, then measure. Sweep physical batches geometrically; choose the fastest size below the measured memory/pressure budget, not merely the largest size that once avoided OOM. CPU may be faster than MPS for tiny models; select the device from validated pilot measurements.

Adapt in this order:

1. Reduce prediction chunks and data-loader duplication; use immutable memory-mapped data when useful.
2. Reduce physical microbatch while preserving effective batch for the sample-independent core.
3. Use a whole-method CPU path if acceleration is incorrect or slower.
4. Only before confirmation, revise rank/tier or experiment scope if necessary and record the changed claim. Do not silently shrink a baseline's model, dataset or indispensable mechanism.

For accumulation, multiply each microbatch mean loss by `microbatch_n/effective_batch_n`; use the actual tail effective-batch size. Accumulate all gradients, then clip once and perform one optimizer/weight-decay/scheduler update. Equal averaging of unequal microbatch means changes the objective. BatchNorm and dropout in external methods can prevent identical trajectories; preserve their native semantics or report a separate configuration.

No reduced precision or quantization is required. Optional lower precision, compilation, fused operations or MLX are separate engineering experiments after correctness. Apply applicable optimizations to both sides, measure compilation/cold costs, and recheck predictions, gradients and results. Never substitute positive-only factors, clipped logs or a different root to satisfy a kernel limitation without recording a changed mathematical model.

### 6.6 Resource estimates and scheduling

The following envelopes are planning estimates. They deliberately avoid a claim that M3 throughput is known.

| Stage | Material storage/memory | Time-estimation rule / likely character |
|---|---|---|
| System/dependency probe | One Python environment; allow a few GB disk for wheels/packages | Install plus tens of short oracle/kernel checks; no large assets |
| Raw selected datasets | Well under 1 MB of raw data across the mandatory pair; transforms negligible | Download/parse/hash; network latency dominates |
| Algebra/numerical checks | Small arrays; normally below 1 GB including runtime | Seconds to minutes on ordinary CPU, verify locally |
| Single small neural fit | Model examples above; runtime/framework often larger than tensor state | `fit_s ≈ steps × measured_step_s + validation_s + calibration_s + checkpoint_s` |
| Kernel/full-basis controls | Explicit real-data matrices small; synthetic dual matrices `8n²` bytes plus work copies | Measure factorization separately; cubic solver time can dominate at large `n` |
| HPO/pilot | Serial trials avoid multiplying RAM; retain configs/metrics and selected checkpoints | Sum actual trial costs, including failures; choose budgets from convergence evidence |
| Confirmation | `12 × N × J × (candidate + trained baseline variants)` fits, plus selection overhead | Use pilot distribution of fit times, not just its fastest case; report median and pessimistic total |
| TabICLv2 practical point | About 114 MB checkpoint plus framework, context and optional KV cache | Charge first `fit+predict` and warm predictions separately; measure local peak |
| Robustness/phase extension | Reuses core infrastructure; independent new experiment records | Scale only after its scientific entry gate and a fresh cost projection |

The initial precision target in §5.7 implies about **33 worlds per stratum before pilot uncertainty inflation** when the observed variance matches its planning estimate. With 12 strata, `J=3` fit seeds and one candidate plus eight baseline representatives, this is `12 × 33 × 3 × 9 = 10,692` confirmation fits. At an illustrative 5 seconds per fit this is about **14.85 hours**; at 60 seconds it is about **178.2 hours**, excluding development, HPO, extra variants, test prediction, practical comparisons and reporting. These are arithmetic scenarios, not laptop measurements; deterministic solvers need not be redundantly refitted for identical seeds. The actual queue and pilot timing distribution determine the total. Long runs are acceptable when justified; unattended execution must checkpoint and respect thermal/memory constraints. If this precision is too costly, record a scientifically justified wider-interval or narrower-population design **before** confirmation; do not silently reduce replication or force the experiment into an arbitrary overnight budget.

Before a run reserve disk for environments, permitted assets, raw data, all scalar logs, predictions and at least two recoverable checkpoints per active trial. Estimate prediction storage as `N_predictions × bytes_per_prediction`, and checkpoint storage as actual serialized bytes. Store only best/last checkpoints for unselected trials once their raw metrics/failure evidence is safely retained; do not delete unfavorable trial records. Generate synthetic test chunks deterministically rather than keeping all feature tensors in RAM.

### 6.7 Timing and recovery

Time with a monotonic clock; synchronize MPS immediately before and after each measured boundary. Measure:

- Warm resident-tensor model inference at batch one and at the calibrated throughput batch.
- End-to-end request inference including transforms, transfers and output/ensemble aggregation.
- Full fit/adaptation including preprocessing, calibration, validation and checkpoints.
- Total HPO and failed-attempt expenditure.
- Cold import/model load/checkpoint download separately from reusable warm operation.

Select warmup and repeat counts from pilot stabilization/variation. An initial 20 warmup iterations is a starting probe; extend until timing no longer drifts beyond session variability. Freeze the actual counts and interval estimator before comparative timing. Alternate/randomize method order in several separate sessions. Repeated synchronized loops estimate machine noise, not independent data-world evidence. Disable unrelated background workloads where practicable and disclose remaining variability. [R33]

Checkpoint only at completed optimizer steps unless accumulated gradients are explicitly saved. Save model, optimizer, scheduler, best-validation state, fixed calibration, all RNG states, data permutation and offset, transform state, trial hash, source/environment identifiers, completed work and elapsed-time counters. Snapshot mutable state before asynchronous serialization. Write a versioned checkpoint and manifest, verify their checksums, then atomically update one pointer to the new complete pair. A separate overwrite of two “latest” files is not atomic as a pair.

On interruption, resume the last complete compatible checkpoint. On OOM, record the failure and restart in a fresh process; follow the frozen microbatch policy. A device/precision/model/optimizer change creates a new run lineage rather than silently continuing the original. Prove resume behavior with an uninterrupted-versus-restarted test in the same environment. PyTorch does not promise cross-device/version bitwise identity; report statistical reproducibility separately from exact restart checks. [R35]

## 7. Ordered work packages and decision gates

All work packages produce artifacts under the repository in §9. They include the next action for substantive outcomes. The implementing agent may make routine non-destructive software decisions; changes to hypotheses, populations, comparisons, metrics or interpretation require a recorded methodological amendment.

### WP00 — Freeze the question and evidence boundaries

**Objective/risk:** prevent rediscovery from becoming a false novelty claim; define intended claims before coding.

**Inputs/entry:** this plan; no prior implementation required.

**Actions:** create the repository, `PROTOCOL.md`, `DECISIONS.md`, source ledger and a claim-to-baseline matrix. Record the three tracks, primary twelve-stratum population, exact candidate equation, and exclusions. Archive cited equations/metadata and selected repository commit/license information. Attempt to inspect the inaccessible recent normalization leads only if advancing a normalization novelty claim; their unresolved status does not block a non-novel empirical study.

**Outputs:** `protocol/draft.yaml`, `evidence/sources.jsonl`, `evidence/licenses/`, initial decision log, project README with explicit unimplemented commands.

**Validation/resources:** source URLs and equations resolve to the stated sources; all contribution language distinguishes replication, engineering, empirical result and unresolved novelty. Small text storage.

**Gate:** proceed with the stated empirical claim. If the exact recipe is also found in prior art, cite it and treat the experiment as replication/extension. If a new claim is proposed, map its closest predecessor and discriminating test before coding it. No “new neuron” claim based only on naming.

### WP01 — Verify machine, assets and dependency route

**Objective/risk:** expose hard compatibility/access limits before large downloads or tuning.

**Dependencies:** WP00.

**Actions:** implement `multneur doctor`; collect §6 hardware facts; select compatible Python/Torch; create core and necessary baseline environments. Resolve actual package APIs and locks. Download only the small UCI archives and license metadata first. Check TabICL checkpoint availability/license; leave TabPFN gated unless authorized eligible access exists.

**Outputs:** `artifacts/hardware.json`, environment locks, package/wheel manifests, raw-data hashes, `artifacts/asset_access.json`.

**Tests/resources:** native architecture; free storage versus projected needs; raw hashes/counts; import tests. Installation footprint measured, not assumed.

**Gate:** CPU route suffices to proceed. Unavailable MPS narrows acceleration claims. Missing optional/gated assets do not block core. Missing mandatory dataset pauses only its transfer branch: record the unfilled scope rather than silently choose a favorable substitute. Any replacement dataset requires a pre-result amendment and equivalent provenance/grouping design.

### WP02 — Prove algebra, numerical behavior and backend parity

**Objective/risk:** invalidate incorrect gradients, unstable subtractive formulas or unsupported kernels early.

**Dependencies:** WP01; tiny generated arrays only.

**Actions:** implement `math/elementary.py`, `math/products.py`, `math/complex_pair.py`, CPU reference enumeration and backend probes. Implement §§4.2–4.5 and root/Jacobian tests. Reproduce the planning-stage cancellation example. Test `e2` on repeated projections to expose square terms and correlated-factor scaling. Compare the tied residual using `C_rm=λ_r sqrt(C(k,m)) scale_rm`, with `λ` remaining the free parameter, against the direct stable residual.

**Outputs:** `artifacts/math_checks.json`, `artifacts/backend_matrix.json`, reproducible test fixtures and failures with dtype/device/shape.

**Tests/resources:** CPU float64 output/gradient parity, zero cases, tiny increments, branch cuts, paired-real/native-complex equality; then actual MPS forward/backward/update and checkpoint. Small CPU memory and short disposable training probes.

**Gate:** fix any CPU oracle discrepancy before ML experiments. If CPU succeeds but MPS fails, use CPU for affected comparisons and investigate separately. A singular root is a mathematical diagnostic, not a failed smooth-product implementation. Do not reject the scientific mechanism because of an invalid backend measurement.

### WP03 — Build immutable data and a tested evaluation harness

**Objective/risk:** prevent leakage, ID misalignment, biased aggregation and fake statistical certainty.

**Dependencies:** WP01–WP02.

**Actions:** implement synthetic generator with stage-separated seeds, canonical target variance and latent-target access control; parse UCI files; construct group keys/protected split manifests; implement train-only transforms; implement metric aggregation and paired stratified/cluster bootstrap. Implement the null, injected-error and intentional-leakage checks in §5.8. Verify every adapter receives only its authorized train/validation arrays.

**Outputs:** `data/manifests/*.json`, `splits/*.npz`, `tests/test_leakage.py`, `tests/test_metrics.py`, `artifacts/harness_checks.json`.

**Tests/resources:** raw counts/hashes, no group overlap, stable row IDs, no test-fit calls, duplicated-array zero effect, canonical synthetic variance check on a separate diagnostic draw. Small dataset memory; generated data chunked.

**Gate:** all invariants pass before performance comparisons. If grouping is too weak/ambiguous for the desired physical claim, narrow that claim; do not use a random-row split as a silent fix. If the harness cannot recover injected differences at the intended resolution, revise measurement before models.

### WP04 — Establish the competitive baseline suite

**Objective/risk:** prevent a win caused by incomplete or weak baselines.

**Dependencies:** WP02–WP03.

**Actions:** implement the precise adapters in §5.2 and official package adapters in §5.3. Start with ridge/full quadratic, a factorized bilinear model, HOFM, CP, CCP and MLP; complete the remaining mandatory portfolio before freezing a claim. Verify direct basis/kernel algebra where applicable; compare DCN forward/gradients to its source; retain full quadratic/spline controls. Record actual author defaults and search spaces. Test official TabM loss/inference aggregation and group-validation routing in all packages.

**Outputs:** `baselines/`, resolved baseline configs/locks, `artifacts/baseline_reproduction.json`, per-baseline test predictions on known small fixtures.

**Tests/resources:** ridge recovers a noiseless affine target to solver tolerance; polynomial ridge recovers a sufficiently sampled known quadratic; exact factor products with oracle parameters reproduce targets; standard MLP learns the nontrivial two-input diagnostic with a validated schedule. Code-level parity is described as component reproduction, not published benchmark replication. Calibrate each baseline's full fit cost and memory.

**Gate:** if a strong baseline fails its control, fix tuning/implementation before comparing. If an official package cannot run faithfully, try its supported CPU/isolated environment. If still unavailable, state which superiority claim is blocked; an easier substitute does not count as that baseline.

### WP05 — Implement the minimum complete product model

**Objective/risk:** determine whether degree sharing and learned factors can train at all without conflating roots/complexity.

**Dependencies:** WP02–WP04.

**Actions:** implement `models/product_residual.py`, literal controls and fixed-calibration buffers; initialize as specified; connect to the common trainer. Include `k=2` exact equivalence and tied-degree clone modes. Add fixed-random-factor plus ridge-readout control before learning factor weights. Log gradients, row norms, factor correlations, per-degree contribution RMS and nonfinite counts on fixed training-only probe rows.

**Outputs:** candidate config, model state schema, short diagnostic learning curves, `artifacts/model_checks.json`.

**Tests/resources:** declared parameter count matches code; initialization gives nonzero higher-branch gradients; exact clone agrees; learned and frozen-factor variants share data/loss. Full small-model memory probe after optimizer steps.

**Gate:** oracle success with learning failure triggers optimizer/conditioning diagnosis. Both oracle and learner failure first trigger capacity/task checks. A `k=2` “win” over its exact clone is a defect. No depth/complex-number expansion is allowed to conceal an uncorrected basic issue.

### WP06 — Pilot, tune, calibrate precision and freeze confirmation

**Objective/risk:** resolve practical cost/noise and avoid underpowered or adaptively selected confirmation.

**Dependencies:** WP03–WP05; protected results remain unopened.

**Actions:** run independent development tasks to tune all families and identify common learning/parameter regimes; perform search-budget reassessment, horizon/batch/device calibration, full baseline fairness checks and operational failure-policy tests. Use independent pilot worlds for variance, test-Monte-Carlo and training-seed planning. Apply the explicit fixed-sample calculation. For real data use only development groups. Project total fits, wall time, storage and expected uncertainty.

**Outputs:** `artifacts/pilot/`, HPO curves, `artifacts/resource_projection.json`, `protocol/confirm_v1.yaml`, analysis-code hash, selected settings and an immutable freeze record.

**Tests/resources:** all critical baselines sufficiently trained; no unresolved search-boundary optimum; frozen-data membership; CPU/MPS parity accepted; restart test passed. Resource totals calculated from actual measurements; do not use the illustrative times as measurements.

**Gate:** proceed only when the target claim is sensitive and locally executable. If baseline and candidate both saturate the known floor, increase challenge under a documented development rule. If no model learns, distinguish optimization from proxy misfit. If effect resolution is inadequate, declare the scope/precision limit or plan a larger fixed study before opening tests. An adverse development result is retained; it may justify a bounded negative diagnostic report without expensive confirmation.

### WP07 — Execute protected comparison

**Objective/risk:** obtain uncontaminated, reproducible evidence for the frozen primary claim.

**Dependencies:** a complete WP06 freeze and sufficient calibrated resources.

**Actions:** run all predetermined worlds/seeds/configurations with one queued process at a time initially; retain failures and resumptions. Select baseline family from validation only. Generate predictions with row IDs, unlock the evaluator once the prescribed run manifest is complete, and compute the frozen primary result, guardrails and all individual baseline tables. Run the separate real-data test once with already frozen pipelines.

**Outputs:** `runs/confirm_v1/`, prediction files, selection records, status manifest, `reports/confirm_v1.json` and plots.

**Tests/resources:** every scheduled run has success/failure/not-run status; no missing adverse seeds; stored test inputs/labels never fed to fit; aggregate counts match the freeze. Measured resource projection from WP06, with resumability mandatory.

**Gate:** interval below zero supports the exact primary claim only; above zero is evidence against the recipe; overlap is inconclusive at stated resolution. A cheaper but less accurate result is a tradeoff. Missing mandatory baseline or invalid run lineage blocks the associated competitive conclusion; do not fill it with a published score.

### WP08 — Attribution, robustness and independent confirmation

**Objective/risk:** explain a positive result without presuming the proposed mechanism caused it.

**Dependencies:** WP07 or a prespecified diagnostic reason; use fresh confirmation when changing the claim.

**Actions:** run the ablations in §8, strengthen baseline tuning on development, inspect parameterization/regularization equivalence, assess negative-control regressions and sensitivity to coordinate transformations. Repeat the selected claim with new independent synthetic worlds and frozen settings; for real confirmation use untouched reserved data/new permissible dataset, not a reshuffle of exposed rows. Perform a clean-environment replay of selected good, bad and failed runs.

**Outputs:** `reports/attribution.md`, `protocol/amendments/`, `runs/confirm_v2/` when required, clean-replay manifest.

**Tests/resources:** attribute benefits to degree, learned representation, regularization or implementation only when the corresponding matched control distinguishes it. Estimate each extension cost before launching.

**Gate:** if stronger baselines erase the effect, revise to the simpler explanation. If only favorable synthetic teachers work, retain a representation result and narrow applicability. If robust gains persist with tolerable costs, justify one next domain/scale experiment. Do not call multiple seeds on the same old holdout independent confirmation.

### WP09 — Conditional high-order/depth or phase branch

**Objective/risk:** preserve ambitious possibilities without making the core depend on speculative combinations.

**Dependencies:** WP02 correctness plus an explicit scientific reason from WP05–WP08. The entry evidence need not be a positive real-data result, but must identify a target property absent from the shallow test.

**Actions:** choose exactly one branch initially. For composition, use a known tree/compositional polynomial teacher and depth-matched additive/bilinear networks; track true degree growth and active signal. For phase, implement §8.3's charge/symmetry-controlled tasks, native-complex CPU oracle and tied/untied real controls. For high fan-in, study scale/zero/gradient response surfaces before training. Do not compute all 128 elementary-symmetric orders merely because a pure 128-factor product is cheap.

**Outputs:** branch-specific hypothesis, amended protocol, cost projection, complete negative as well as positive diagnostic results.

**Tests/resources:** branch topology or symmetry is expressed by the task; the closest arithmetic/phase competitor is included; exact operation support proven. Local shapes are calibrated without losing the indispensable mechanism.

**Gate:** a shallow failure does not reject a truly compositional claim, but escalation requires a compact oracle and identifiable missing structure. If all asserted benefit requires inaccessible scale, document the locally testable prediction and the unresolved broader claim. If complex and tied-real models disagree, fix implementation before training conclusions.

### WP10 — Synthesize, audit and package the research result

**Objective/risk:** prevent selective reporting, overclaiming or an irreproducible finish.

**Dependencies:** completed applicable stages, including all failures and amendments.

**Actions:** generate the result schema and tables in §9; classify evidence with §10; run scientific, experimental and execution audits again against the final repository; reconcile every scheduled run and source/data/config hash. State what was measured, inferred, estimated and left untested. Produce a rerunnable command manifest and a short next-decision record.

**Outputs:** `reports/final_report.md`, `reports/results.csv`, `reports/claims.json`, complete run manifest, current `DECISIONS.md`, final audit record and environment/data provenance.

**Acceptance/resources:** a fresh coding agent can reproduce one small end-to-end run without chat memory; all quantitative claims trace to raw outputs; no fabricated download/model/API/version; no arbitrary missing negative cells. This final synthesis uses existing artifacts and modest storage.

**Gate:** finish as a confirmed finite-scope benefit, a known-method replication, a tradeoff, a negative finding, an implementation limitation or an explicitly underpowered result. Unresolved prerequisites remain named blocking gates for their dependent claims.

## 8. Attribution, robustness and failure-analysis program

### 8.1 Required discriminating ablations

Do not run every possible factorial combination. Use the following causal questions to select the minimum comparisons needed for the actual claim, prespecifying the mandatory rows before confirmation. Cost estimates come from WP06. Report complete results for every attempted cell.

| Question | Matched intervention/control | Interpretation if gain disappears |
|---|---|---|
| Is multiplication necessary? | Affine, additive splines/polynomials, GELU MLP, gated MLP, full quadratic and polynomial kernel | A simpler nonlinear/additive/polynomial explanation suffices |
| Are degrees above two useful? | Retrain the same candidate with only degree two, then `2+3`, then `2+3+4`; matched capacity curves and equal-rank curves | A bilinear mechanism explains the result; do not advertise high-order value |
| Is projection learning useful? | Raw-coordinate HOFM; fixed random projections plus ridge readout; jointly learned projections | Feature expansion or coordinate alignment explains most of the effect |
| Does sharing factors across degrees help? | Exact `projected_independent` control defined below, at equal rank and equal cost | Shared factors are only parameter compression or impose harmful bias |
| Does a fixed scale help beyond optimizer coordinates? | Function-matched raw/calibrated pair, mapped penalties, per-degree learning-rate tuning and common initialization | Report optimization/regularization recipe, not expressivity or new arithmetic |
| Is numerical evaluation responsible? | Stable recurrence shared by all equivalents versus naive subtraction as diagnostic | Engineering correction only; no new hypothesis class |
| Is the advantage simply more capacity/compute? | Error-versus-parameters, fit-time and HPO-time curves; matched depth/batch; count all ensemble members | State the measured tradeoff or withdraw efficiency claim |
| Is the additive path responsible? | Remove/retrain affine skip; give equivalent skip to CP, CCP and neural controls | Residual/main-effect routing explains it |
| Is high-degree signal actually active? | Held-probe contribution RMS and retrained order ablations, with factor norms/covariance | Stable model may have effectively collapsed to lower degree |
| Does generic training explain it? | Shared initialization/scaling/clipping where appropriate; enlarged baseline HPO on development | Initial comparison was under-tuned or asymmetrically optimized |
| Is preprocessing the treatment? | Common train-only transforms versus each method's native practical pipeline | Claim a pipeline gain rather than a neuron gain |
| Are favorable supports/teachers essential? | Axis/rotation, active dimension, coefficient rank, independent versus shared-factor teachers | Narrow the inductive-bias claim to the demonstrated structure |

The **exact projected independent-degree control** uses separate `W[m,R,k,d]` for every retained degree, with

\[
q_{rm}^{ind}=e_m(W_{m,r}s)/\sqrt{\binom{k}{m}},
\]

and the same bias-free factors, readout, initialization, optional calibration and affine skip as the candidate. At equal `R` it contains the shared model but uses more parameters. At equal scalar/compute budget it tests efficiency but does not necessarily contain the candidate family. Raw-coordinate independent HOFM and affine CP alone do not isolate this sharing decision because they also change the basis/offsets.

Readout coefficients are not identifiable measures of importance. For example, jointly scaling all rows in `W_r` by `a` and `C_rm` by `a^(−m)` preserves the represented polynomial. Record prediction contributions, correlations and ablation effects, not just `|C|`. Factor-row permutations and component permutations are additional equivalences. Do not renormalize rows after every optimizer step as a supposedly harmless optimization: individual row scaling changes the internal elementary-symmetric coefficient relationships.

### 8.2 Boundary cases and escalation rules

Run zeros, two zeros, mixed signs, factors near the neutral element, near-canceling monomials, large/small magnitudes, correlated factors, almost-identical projection rows, train-only scale drift and changes of physical units. Use absolute-plus-relative numerical error and condition estimates; relative error alone near a genuine zero is misleading.

For statistical robustness vary label noise, sample size, irrelevant feature count, target coefficient density and rotation. Include a nonpolynomial smooth target and additive nonlinear targets to detect costly regressions. Investigate heteroscedastic noise only as a separately specified generator, with the correct expected-noise term in risk. Do not reuse the `.01` floor after changing its noise law.

A high-fan-in product test must distinguish `f_j` centered near zero from `1+τt_j` centered near one. Study `τ` as an explicit factor scale and record per-degree signal attenuation. Scaling `τ` with fan-in can stabilize moments while suppressing high-order content; numerical survival alone is not preservation of the original computational advantage.

Depth escalation requires a compositional target that a shallow fixed-degree model cannot express at the same budget. Give every comparator the same topology information and comparable depth opportunities. Record total polynomial degree as well as layer count: composing degree-`k` blocks can reach degree `k^L`, whereas input-injected cross recurrences grow degree differently. A successful small compositional model motivates a larger test; it does not prove an eventual large-scale advantage.

A numerical product remedy that maps every factor positive removes sign interactions. A clipped log remedy changes tail behavior. An amplitude normalizer changes the readout's functional burden. Treat these as explicitly different models, not invisible implementation repairs.

### 8.3 Conditional phase experiment

This branch tests a known useful algebra with honest real-valued controls. It is not a prerequisite for completing the core and cannot establish domain utility without a subsequent suitable dataset.

**Inputs:** paired real features `(r_j cos θ_j, r_j sin θ_j)` for `j=1..d_phase`, initially `d_phase=4`, with independent `θ_j~Uniform(−π,π)`. Begin with `r_j=1`; then use `log r_j~Normal(0,.25²)` and explicit zero-amplitude probes. All models receive the same real/imaginary information; do not give raw wrapped angles only to the conventional model.

**Phase preprocessing overrides §4.1:** do not subtract channel means or scale real/imaginary coordinates separately in the equivariance experiment. Use either identity preprocessing or one fixed positive training-only RMS per complex channel, `sqrt(mean_train |z_j|²)`, shared by its real and imaginary parts; an all-zero channel uses divisor one. Apply the same rule to every compared method. For the two-output target use no centering and a single positive RMS divisor shared by both components, then invert it for metrics. Such radial scales commute with common phase rotation; ordinary componentwise standardization generally does not. Store these choices and test the entire preprocessing/model/readout path for the claimed symmetry.

**Targets:** separately evaluate two-output real/imaginary representations of `z1*z2` (phase charge two), `z1*conj(z2)` (relative-phase invariant under common rotation), `z1*z2*z3*z4` (charge four), and `z1*z2*conj(z3)*conj(z4)` (charge zero). Include distractors and a separately labeled known-support oracle. Use restricted training-angle ranges versus full-angle testing only as an explicit extrapolation experiment. Include an amplitude-sensitive raw-product task and a phase-only task to distinguish the normalizer's effect.

**Models:** direct phasor oracle; learned complex-affine factor product; exact tied-real implementation; untied real bilinear/quadratic model; real MLP with sine/cosine inputs; explicit Fourier/product features plus ridge readout. In the exact-equivariance branch use bias-free complex-linear factors and a complex-linear readout with no output bias; sum only terms with the target's phase charge and omit an affine skip of a different charge. If biases or differently charged paths are enabled, test and report the resulting symmetry rather than assuming it is preserved. Ordinary real baselines need not be equivariant; their symmetry error is measured. Include conjugation only when the target's transformation law requires it.

Use this paired-real multiplication, retaining both old components:

```python
def cmul(ar, ai, br, bi):
    return ar*br - ai*bi, ar*bi + ai*br

def radial_power_pair(pr, pi, k, eps):
    # Exact stated small-scale formula; large-range safety is separately tested.
    alpha = (k - 1.0) / (2.0*k)
    gain = (pr.square() + pi.square() + eps*eps).pow(-alpha)
    return pr*gain, pi*gain
```

The formula can overflow when squaring very large values or while forming the product. A scaled/hypot implementation may extend range, but requires equivalence and gradient tests. Do not switch representations silently. Avoid angle extraction in the training path where paired products suffice.

**Metrics:** two-component MSE; circular phase error only when true amplitude exceeds a frozen meaningful threshold; amplitude error separately; branch-cut discontinuity; equivariance residual under held rotations; sample efficiency and all real arithmetic/storage costs. Phase is undefined at zero amplitude. For the **direct product of unit-magnitude factors**, radial normalization is only a constant scale. Learned complex-linear mixtures can have variable magnitudes even when the original inputs have unit magnitude; normalization can then change their amplitude response and optimization, while remaining the invertible map described in §2.5.

**Decision:** exact complex/tied-real disagreement blocks conclusions. A tied-complex advantage over an untied model can support inductive bias/parameter tying, not inherently greater arithmetic expressivity. Only an improvement beyond simple Fourier/bilinear alternatives justifies selecting a phase-sensitive real workload and conducting a new provenance/license/grouping search. MRI/audio/radio claims are not established by this synthetic demonstration. Arithmetic extrapolation claims additionally require the NMU/NPU benchmark conventions and relevant sign/zero ranges. [R09,R10,R12,R13,R41]

## 9. Repository, interfaces and reproducibility contract

### 9.1 Minimal project layout

The implementing agent creates this repository; the present deliverable contains its specification, not a prebuilt training framework.

```text
multneur/
  README.md
  PROTOCOL.md
  DECISIONS.md
  pyproject.toml
  requirements-core.lock
  requirements-baselines/
  protocol/
    draft.yaml
    confirm_v1.yaml
    freezes.jsonl
    amendments/
  configs/
    models/
    baselines/
    workloads/
    searches/
  src/multneur/
    __main__.py
    doctor.py
    math/{elementary,products,complex_pair}.py
    models/{product_residual,controls}.py
    baselines/{linear_kernels,polynomial,neural,packages}.py
    data/{synthetic,uci,transforms,splits}.py
    training/{trainer,search,checkpoint}.py
    evaluation/{metrics,statistics,timing,guard}.py
    reporting/{tables,plots,claims}.py
  tests/
    test_algebra.py
    test_gradients.py
    test_backend.py
    test_leakage.py
    test_metrics.py
    test_resume.py
  data/{raw,manifests,processed,sealed}/
  splits/
  evidence/{sources.jsonl,licenses}/
  artifacts/{hardware.json,backend_matrix.json,pilot}/
  runs/{development,pilot,confirm_v1}/
  reports/
```

Braced file groups in this tree are shorthand for separate files, not literal brace-containing filenames. Avoid an elaborate distributed platform: serial process isolation, clear files and a small job manifest suffice on one laptop.

### 9.2 Shared interfaces

Use typed structures with a schema version. This is a proposed project interface, not an external package API:

```python
@dataclass(frozen=True)
class DataPartition:
    X: np.ndarray          # host float64, shape [n,d], immutable
    y: np.ndarray          # host float64, shape [n]
    row_id: np.ndarray     # unique stable strings, shape [n]
    group_id: np.ndarray   # stable strings, shape [n]
    data_hash: str
    split_hash: str

class MethodAdapter(Protocol):
    def fit(self, train: DataPartition, validation: DataPartition | None,
            config: dict, seed: int, budget: dict,
            artifact_dir: Path) -> "FitResult": ...
    def predict(self, X: np.ndarray, row_id: np.ndarray,
                chunk_size: int) -> "PredictionBundle": ...
    def save(self, destination: Path) -> dict: ...
    def load(self, source: Path, expected_hashes: dict) -> None: ...
```

The implementation must add the imports and concrete dataclasses. `frozen=True` prevents attribute reassignment, not mutation of a NumPy array: mark shared arrays `writeable=False`, or hand adapters private copies and verify shared-array hashes before and after fitting. Host float64 does not authorize MPS float64: each neural adapter explicitly casts train-fitted transforms to float32. Avoid in-place modification of shared input arrays. Official packages may own their native preprocessing, but must return its resolved parameters/provenance and prove that fitting used only the supplied training partition.

`FitResult` records status, best-validation value and selection metric, total/time components, steps/epochs, stop reason, model and transform hashes, parameter count, serialized bytes, resource observations, failures and fallback use. `PredictionBundle` contains method/run ID, row IDs, original-unit prediction float64, prediction hash and timing boundaries. Tests/latent `f` are provided only to the evaluator, never to `fit` or HPO.

For Track A use one development-selected configuration per method family across all twelve strata. Adapters do not receive true degree, active support, rotation matrix, target coefficients or latent outputs. Oracle support/degree is allowed only in named diagnostic controls. The baseline family on a confirmation world is selected by mean noisy-validation MSE across its prescribed training seeds; the selected family's **mean test risk** then enters `Rbar`. Do not choose a best seed. Candidate within-world search is off by default; enabling it requires a frozen, charged selection policy.

For Track B, HPO ranks pipelines using common group-MSE over internal validation groups. A package's native internal early-stopping criterion may remain its documented row-based loss if it cannot accept group weights; log that criterion and still use common group-MSE for final hyperparameter selection. Internal validation membership must remain explicitly group-respecting. Do not fabricate unsupported package keywords; inspect and test the installed signature or use a transparent external train/validation adapter.

### 9.3 Configuration conventions

Example **proposed** YAML; unresolved pilot-derived values must be filled before a freeze is valid:

```yaml
schema_version: 1
stage: development
protocol_id: multneur_v1
model:
  family: product_residual
  degree: 4
  rank: 64
  factor_bias: false
  factor_init_std_rule: scale_over_sqrt_input_dim
  factor_scale: 1.0
  output_scale: 0.01
  degree_scaling: combinatorial
  fixed_rms_calibration: false
training:
  dtype: float32
  device: selected_by_verified_pilot
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  effective_batch: 128
  physical_microbatch: selected_by_memory_probe
  max_epochs: selected_by_duration_pilot
  validation_interval_epochs: 1
  restore_best: true
data:
  workload: coefficient_polynomial_v1
  d: 32
  n_train: 1024
  n_validation: 512
  noise_std_canonical: 0.1
evaluation:
  primary: stratified_mean_log_seed_mean_risk_ratio
  test_size: selected_by_mc_pilot
  worlds_per_stratum: selected_by_precision_pilot
  optimizer_seeds: selected_by_seed_variance_pilot
  failure_policy: validation_best_else_refit_last_finite_else_mean
```

An unresolved string in a numeric field is valid only in development templates. `freeze` must reject it, missing baseline versions, missing data hashes or an unimplemented required method. Canonicalize resolved JSON and hash it; never edit a config under an existing run ID. Explicitly record training and evaluation units, parameter count conventions and whether predictions are an ensemble.

### 9.4 Experiment records and provenance

Separate the **scientific run ID** (hash of the frozen run specification) from a unique **execution attempt ID**. A retry/resume appends a new attempt with a parent attempt ID, start/end timestamps, checkpoint origin and its own immutable directory; it must not overwrite the failed attempt or erase its cost. Store lifecycle events in append-only `attempts.jsonl` and reconcile all attempts to the scientific run's final disposition. Every attempt writes a record, including an interrupted, invalid or failed run. Recommended minimum JSON schema:

```json
{
  "schema_version": 1,
  "run_id": "sha256-of-canonical-run-spec",
  "attempt_id": "unique-execution-id",
  "parent_attempt_id": null,
  "started_at_utc": null,
  "ended_at_utc": null,
  "stage": "pilot",
  "protocol_hash": "filled-at-execution",
  "source_commit": "filled-at-execution",
  "source_tree_sha256": "filled-at-execution",
  "environment_lock_hash": "filled-at-execution",
  "dataset_hash": "filled-at-execution",
  "split_hash": "filled-at-execution",
  "transform_hash": "filled-at-execution",
  "workload": "coefficient_polynomial_v1",
  "stratum_id": "degree3_active8_rotated",
  "world_id": 0,
  "training_seed": 0,
  "search_seed": 0,
  "method_id": "product_residual",
  "status": "planned",
  "failure_category": null,
  "fallback_used": false,
  "trainable_real_scalars": null,
  "model_bytes": null,
  "fit_wall_seconds": null,
  "validation_metric": null,
  "test_metric": null,
  "test_accessed": false,
  "prediction_sha256": null,
  "checkpoint_lineage": [],
  "resource_record": "relative/path/to/resource.json"
}
```

All example strings are schema placeholders, not claimed hashes or commits. A confirmation record must contain actual values. Freeze a clean committed source tree, or archive and hash the exact dirty working tree plus its diff and untracked implementation files; a commit ID alone does not identify uncommitted code. Use enumerated statuses `planned/running/completed/failed/interrupted/invalid/not_run` and failure categories `implementation/measurement/backend/resource/optimization/data_access/protocol`. A scientific null/negative result is not a software “failure.” Use standard JSON `null` for missing/nonfinite metrics plus an explicit reason; do not serialize `NaN` as if it were a valid measurement.

Retain seed namespaces, raw output predictions, selection decisions, train/validation learning curves, test-access logs, environment variables relevant to numerics, package/source revisions, original raw data/model asset hashes, preprocessing state, time/memory samples and all amendments. Preserve failed runs and logs even when their checkpoints can be reclaimed. Distinguish downloaded reusable asset size/pretraining provenance from project-specific HPO/fit expenditure.

Progress records in `DECISIONS.md` contain date, completed WP, evidence, decision, unresolved blocker and next executable action. Recovery reads these artifacts, not chat memory. A methodological amendment records the old/new hypothesis or procedure, reason, data/results already inspected, affected runs and the fresh confirmation required.

### 9.5 Proposed command sequence

The agent must implement these entry points and their `--help` before using them. They are a desired CLI contract, not commands already present on a fresh laptop. After creating `pyproject.toml` and installing locked dependencies, install the project into its active environment with `python -m pip install --no-deps -e .` so the `src/` layout resolves independently of the shell's working directory:

```bash
python -m multneur doctor --output artifacts/hardware.json
python -m multneur prepare-data --manifest configs/workloads/uci.yaml
python -m multneur check-math --device cpu --dtype float64
python -m multneur check-backend --device mps --fallback disabled
python -m multneur validate-harness
python -m multneur reproduce-baselines --suite configs/baselines/core.yaml
python -m multneur calibrate --config protocol/draft.yaml
python -m multneur run --stage development --config protocol/draft.yaml
python -m multneur run --stage pilot --config protocol/draft.yaml
python -m multneur freeze --config protocol/confirm_v1.yaml
python -m multneur run --stage confirm_v1 --frozen protocol/confirm_v1.yaml
python -m multneur evaluate --frozen protocol/confirm_v1.yaml
python -m multneur report --frozen protocol/confirm_v1.yaml
```

`check-backend` reports unsupported MPS and permits the documented CPU branch rather than causing the whole project to fail. `freeze` records the actual resolved run queue, precision/power calculation, budget, analysis hash and all prerequisites. `evaluate` refuses unexpected/missing run IDs and requires explicit disposition of failures; it never silently aggregates whatever files happen to exist. `run --resume RUN_ID` verifies checkpoint lineage, hashes and backend before resuming. Implement a dry-run mode that prints cost and scope without training or exposing test labels.

### 9.6 Required reporting tables and figures

The report must stand alone and contain:

1. Claim ledger: claim, evidence status, exact population, comparator, endpoint, interval and limitations.
2. Full method table: original-unit or canonical risk, absolute/relative effect, uncertainty, failure/fallback rate, parameters, serialized model/context bytes, fit/HPO and inference costs, memory observations.
3. Every primary stratum plus negative controls; no winner-only subset.
4. Accuracy versus samples, parameters and measured time, with repeated-world dependence retained.
5. Numerical/gradient response surfaces over factor scale, fan-in, zeros and normalization; these are diagnostics, not ML benchmarks.
6. Tuning-learning curves, baseline reproduction status, exposed versus protected evaluation history and amendments.
7. Real-data group counts, leakage checks, row/group metrics and influential-group sensitivity.
8. Reproduction manifest and outcome-based next step.

Plots are generated from raw machine-readable results; include units and uncertainty definitions. Do not create a simulated “result plot” and present it as observed performance. If no positive result occurs, report that plainly and retain the same tables.

## 10. Outcome interpretation and evidence-triggered next steps

| Observation | Conclusion supported | Next action |
|---|---|---|
| Algebra or gradient parity fails | Broken implementation or ambiguous specification | Repair and rerun prerequisite checks; discard affected performance claims |
| CPU oracle works, MPS does not | Local backend/precision problem | Faithful CPU comparison; isolate acceleration issue |
| Oracle representation works, learned model does not | Optimization/signal-identification limitation at tested scale | Diagnose initialization, scales, data regime and stronger optimizer before altering theory |
| All methods fail a proxy or all hit its floor | Insensitive/unsuitable workload | Change development difficulty under an amendment; do not call it scientific falsification |
| Product beats affine/weak MLP only | Known representational bias demonstrated | Prefer nearest strong polynomial/bilinear comparison; no novelty/practical claim |
| Calibrated recipe wins only with altered penalty or effective learning rate | Optimization/regularization effect | Name the recipe; publish mapped-coordinate controls; no extra-expressivity claim |
| Higher-degree candidate loses to bilinear/full quadratic | Higher orders unhelpful in tested regime | Recommend simpler degree-two model for that regime; high-order-specific targets remain separately testable |
| Stable candidate wins only over subtractive implementation | Numerical evaluation improvement | Share recurrence; withdraw architectural advantage claim |
| Synthetic benefit survives controls, real benefit inconclusive/absent | Finite synthetic operating region established | Narrow claim; use real error analysis to justify any next domain, not a favorable-domain search disguised as confirmation |
| Real gain disappears under grouping or strong tuning | Leakage/weak-baseline explanation | Correct report, stop practical superiority claim |
| Risk improves but latency, tails or failure rate regress | A measured tradeoff | Present both; select only with an application-specific tolerance |
| Interval remains wide | Insufficient precision for the claimed sign/size | Report detectable effect; plan a fresh fixed-size confirmation only if decision value justifies it |
| Robust finite-scope benefit survives tuning, attribution and fresh confirmation | A useful empirical advance within known model families | Publish reproducible result; choose one next scale/domain based on the supported cause |
| Exact prior recipe is located | Replication or extension | Cite it, compare directly and revise novelty wording |
| No effect survives strong alternatives | No demonstrated advantage for this recipe | Stop replacing neurons; recommend the winning existing method and preserve the negative evidence |

An eventual claim should name `[workload population]`, `[baseline policy/methods]`, `[metric and units]`, `[effect and interval]`, `[costs/regressions]` and `[attribution supported by controls]`. It should also state whether the contribution is a replication, engineering improvement, new empirical finding, new application or a separately substantiated method. Avoid “multiplication is better than addition” as a scientific conclusion.

Local scale is a limit on evidence, not proof that larger-scale benefits are impossible. Escalation needs a specific prediction: for example, an observed advantage grows with target compositional depth while matched alternatives' approximation error remains high, or a parameter/error frontier consistently improves with rank without worsening conditioning. A hypothetical future advantage without such evidence is not a reason to continue indefinitely.

## 11. Audit record, unresolved risks and limitations

### 11.1 Audits performed during preparation

Three distinct assistant design reviews challenged the scientific/prior-art basis, experimental validity, and fresh-laptop implementation path. Outline reviews were followed by reviews of the concrete draft and targeted correction checks. These were not independent human peer reviews, baseline reproductions or M3 Max validation.

| Audit defect found | Revision incorporated | Residual risk / acceptance condition |
|---|---|---|
| Shifted products and degree weights already exist | Removed new-neuron assertion; exact HOFM/Pi-Sigma/Π-Net relationships and direct competitors | Search is not proof of absence; no method novelty claimed |
| Projected degree terms misread as distinct-feature effects | Degree terminology, square/additive controls, exact projected independent-degree ablation | Learned factors remain nonidentifiable; causal interpretation excluded |
| Combinatorial scaling treated as universal stability | Independence assumptions stated; correlated-factor probes and scale-drift logging | Real factor covariance can invalidate idealized moment intuition |
| Root normalization described as losing information | Separate invertible amplitude map from restricted-readout burden and finite-precision loss | Approximate inverse learning may be costly; test explicitly |
| Generic preprocessing or skip paths could break phase equivariance | Phase-specific radial scaling, no centering, explicit target charges and matching readout paths | Test the complete composed pipeline, not only multiplication |
| Naive residual subtraction unstable near identity | Cancellation-resistant recurrence; numerical example; same implementation given to all equivalent baselines | Genuine signed-term cancellation/overflow still possible |
| Calibration alters initialization and regularization | Mapped-function/penalty/SGD controls and shared calibration options | Adam/clipping dynamics still need their own attribution |
| Raw-coordinate HOFM does not isolate projected sharing | Added same-basis independent-degree control at equal rank and equal cost | Equal-cost function classes need not contain one another |
| Synthetic teachers could predetermine a win | Coefficient-sampled Gaussian population plus low-rank favorable and adverse diagnostics | The population is still designer-chosen and must be reported exactly |
| Rotation of a cube would change support | Isotropic Gaussian primary inputs; independent stratum draws | Non-Gaussian robustness has additional distribution changes |
| Seeds/folds/timings could masquerade as independent data | World-level inference; seed-risk aggregation; cluster-level real analysis | Finite-seed nonlinear estimands and proxy-group dependence remain explicit |
| Real datasets contain repeated conditions | Measured groups, immutable grouped holdout, duplicate sensitivity and influence checks | Hidden laboratory/source dependence remains unknown |
| Weak/outdated practical baselines | Polynomial/RBF controls, official TabM/RealMLP/trees and current TabICLv2; gated 3.5 context | Two datasets do not characterize contemporary tabular performance broadly |
| Baseline portfolio win could be overgeneralized | Name the selection policy; publish all members; multiplicity for broader claims | Validation selection can choose a suboptimal member |
| Refitting without validation made failure rules ambiguous | Separate fixed-duration refit fallback and explicit native-refit checks | Native schedules and automatic parameters must be frozen correctly |
| Package installation mistaken for kernel compatibility | Exact forward/backward/update checks; version/OS branch; CPU path | No actual Mac operation has been tested here |
| Memory accounting omitted optimizer/overlapped unified pages | Full-step calibration, parameter formulas, separate sampled counters | Short-lived peaks may exceed sampled maxima |
| Resume changed model/selection state | Fixed buffers, RNG/data offsets, optimizer/schedule/best-state hashes, atomic checkpoint pair | Nondeterministic backends may limit bitwise restart parity |
| Unfavorable or failed trials could disappear | Complete run manifest, operational fallback scored, failure-rate guardrail | A fallback-supported metric must not be sold as reliable native training |

The audits also changed concrete implementation details: explicit Gaussian **standard deviation** rather than ambiguous variance notation; an unpenalized intercept for kernel ridge; sample-size-aware solver regularization; a numerical RMS floor; correct weighting of unequal microbatches; disabling hidden validation in final refits; freezing CatBoost's resolved learning rate and schedule semantics; and distinguishing floating-point dtype requirements from legitimate integer/bool buffers.

### 11.2 Stress-test outcomes and blocking gates

- **Idea exists under another name:** already true for the main mechanisms. Continue as empirical/optimization research with citations; no novelty gate blocks an honest replication.
- **Strong published system cannot run locally:** its published result stays contextual. Include the strongest faithful local component/package; narrow the claim. A reduced substitute is not that system's result.
- **Memory lower than expected:** small core is designed to remain feasible; chunk/microbatch or use CPU. Explicit high-dimensional basis materialization is blocked by its preflight estimate. No assumed maximum-memory laptop.
- **Critical operation unsupported:** CPU float64 decides correctness; ordinary paired-real arithmetic removes a native-complex dependency. Any mathematical replacement requires a new method label.
- **Data/license unavailable:** stop only the affected branch, keep core progress, and record a replacement decision before protected evaluation. No silent unauthorized model download or paid service dependency.
- **Synthetic success does not transfer:** report the mechanism's finite operating region; do not call it a universal replacement. Additional domains require a reason based on failure analysis.
- **Mechanism needs absent scale:** identify a compact compositional/fan-in prediction and test it; otherwise mark scale benefit unresolved. Toy failure is not an automatic global rejection.
- **Tiny/unstable gain or stronger baseline erases it:** report precision/tuning sensitivity and revise the conclusion. No arbitrary percentage threshold defines a breakthrough.
- **Primary gain hides regression:** show tails, costs, fallback rates and negative controls. An uncompensated tradeoff is not overall dominance.
- **Interrupted run resumes inconsistently:** invalidate that lineage, fix state capture and repeat the affected comparison. An unrecorded backend/config change cannot be “resume.”
- **Results conflict across strata:** investigate predeclared moderators; report the complete surface. A newly selected subgroup needs fresh confirmation.
- **Physical claim exceeds computation:** predictive fit on old engineering data does not identify governing laws, prove causality or establish safe deployment.
- **Additional product-specific traps:** zero-initialized branches, equal-row symmetry, compensated factor/readout norm growth, branch-cut jumps, incorrect phase charge, finite-precision false sign, scale calibration from test data, singular percentage metrics, and polynomial feature explosion all have explicit controls above.

### 11.3 What remains unknown

There is no observed training advantage, effect size or M3 throughput result yet. Exact baseline compatibility must be measured. The finite real cohorts may be too small for a stable transfer conclusion. Researcher/benchmark-selection contamination cannot be ruled out for widely used public data or externally tuned assets. Recent blocked normalization papers remain concrete literature blind spots. A patent disclosure search provides technical context only; it is neither an exhaustive search nor a patentability/freedom-to-operate opinion.

## 12. References, evidence ledger and search coverage

### 12.1 Reading this ledger

All URLs were researched as of **29 October 2026**. “Inspected” describes the actual paper sections, documentation or code consulted during preparation; “discovery only” means it must not support a substantive empirical claim. Dynamic repository/docs URLs must be converted to exact revisions in WP00/WP01. The plan's equations, generators, statistical design and resource calculations are project deductions/proposals unless explicitly marked as source findings or measurements.

**R01 — Mathieu Blondel, Akinori Fujino, Naonori Ueda, Masakazu Ishihata (2016). _Higher-Order Factorization Machines_. NeurIPS.** [Primary paper](https://proceedings.neurips.cc/paper/2016/file/158fc2ddd52ec2cf54d3c161f2dd6517-Paper.pdf). Inspected algorithms, §§5.1–5.2 and comparison table. Decisive overlap: shared degree factors/weights and all-subsets products. Link-prediction experiments include adverse results for unrestricted order; they do not rank the present tabular models. Supports the closest baseline and contradicts broad novelty.

**R02 — Richard Durbin, David E. Rumelhart (1989). _Product Units: A Computationally Powerful and Biologically Plausible Extension to Backpropagation Networks_. Neural Computation 1(1).** [DOI](https://doi.org/10.1162/neco.1989.1.1.133). Bibliographic identity and its role in later primary papers verified; publisher full text was not accessible in this investigation. Historical predecessor only; exact novelty conclusions rely on accessible R01/R03, not an uninspected implementation.

**R03 — Yoan Shin, Joydeep Ghosh (1995). _Ridge Polynomial Networks_. IEEE Transactions on Neural Networks 6(3), 610–622.** [Author-hosted primary PDF](https://www.ideal.ece.utexas.edu/pubs/pdf/1995/shgh95.pdf). Inspected Pi-Sigma equation and ridge-polynomial construction. Products of learned affine sums directly overlap the learned-factor proposal. Older surface-fitting/classification experiments support feasibility, not present-day competitive rankings.

**R04 — Kyohei Atarashi, Satoshi Oyama, Masahito Kurihara (2021). _Factorization Machines with Regularization for Sparse Feature Interactions_. JMLR 22.** [Primary PDF](https://jmlr.org/papers/volume22/20-1170/20-1170.pdf). Inspected formulation and all-subsets extension. Relevant to sparse interaction claims and competing regularizers; synthetic/real conditions differ from this plan. No automatic reuse of its empirical results.

**R05 — Grigorios G. Chrysos, Stylianos Moschoglou, Giorgos Bouritsas, Yannis Panagakis, Jiankang Deng, Stefanos Zafeiriou (2020). _Π-Nets: Deep Polynomial Neural Networks_. CVPR.** [Paper](https://arxiv.org/pdf/2003.03828), [official proceedings](https://openaccess.thecvf.com/content_CVPR_2020/html/Chrysos_P-nets_Deep_Polynomial_Neural_Networks_CVPR_2020_paper.html), [code](https://github.com/grigorisg9gr/polynomial_nets), [license](https://github.com/grigorisg9gr/polynomial_nets/blob/master/LICENSE.txt). Inspected recurrences, repository and CC BY-NC 4.0 notice. Supports the CCP control; published vision/generation results are broader context. Journal extension: [DOI](https://doi.org/10.1109/TPAMI.2021.3058891).

**R06 — Grigorios G. Chrysos, Bohan Wang, Jiankang Deng, Volkan Cevher (2023). _Regularization of Polynomial Networks for Image Recognition_. CVPR.** [Paper](https://arxiv.org/pdf/2303.13896), [author code](https://github.com/grigorisg9gr/regularized_polynomials). Inspected initialization/normalization/regularization sections. Supports sharing generic training improvements. Its image/audio scope differs; an explicit unrestricted repository license was not established from inspected material.

**R07 — Ruoxi Wang et al. (2021). _DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems_. WWW.** [Paper](https://arxiv.org/pdf/2008.13535), [official TensorFlow layer](https://github.com/tensorflow/recommenders/blob/main/tensorflow_recommenders/layers/feature_interaction/dcn.py), [API](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/layers/dcn/Cross), [PyTorch reference](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/layers/interaction.py). Inspected equations and matrix/low-rank forward paths. Apache reference reusable. Industrial evidence supports relevance; this plan tests a local component, not the original deployed system.

**R08 — Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua Bengio, Ruslan Salakhutdinov (2016). _On Multiplicative Integration with Recurrent Neural Networks_. NeurIPS.** [Paper](https://arxiv.org/pdf/1606.06630). Inspected affine-plus-bilinear integration. Supports additive residual/control rationale; recurrent-task gains merely resemble the proposal and do not settle the present hypothesis.

**R09 — Andreas Madsen, Alexander Rosenberg Johansen (2020). _Neural Arithmetic Units_. ICLR.** [Primary record](https://arxiv.org/abs/2001.05016), [official code](https://github.com/AndreasMadsen/stable-nalu). Inspected primary description and benchmark lineage; relevant to selected products, constrained weights and arithmetic extrapolation. Use its ranges/initialization when that claim is made; it is not a generic tabular baseline obligation.

**R10 — Niklas Heim, Tomáš Pevný, Václav Šmídl (2020). _Neural Power Units_. NeurIPS.** [Primary PDF](https://proceedings.neurips.cc/paper/2020/file/48e59000d7dfcf6c1d96ce4a603ed738-Paper.pdf), [Julia repository](https://github.com/nmheim/NeuralPowerUnits). Inspected complex-arithmetic formulation and repository workflow. Demonstrates that complex internal arithmetic for learnable powers is established. Arithmetic/dynamical-system results do not establish a universal complex advantage.

**R11 — Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji (2015), _Bilinear CNN Models for Fine-Grained Visual Recognition_; Tsung-Yu Lin, Subhransu Maji (2017), _Improved Bilinear Pooling with CNNs_.** [2015 proceedings](https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html), [2017 author PDF](https://people.cs.umass.edu/~smaji/papers/improved_bcnn.pdf), [2017 record](https://arxiv.org/abs/1707.06772). Inspected bilinear/power-normalization descriptions and normalization comparisons; the 2015 CVF PDF fetch was blocked. Relevant antecedent to signed compression, not evidence for this exact epsilon/k-dependent normalizer.

**R12 — Chiheb Trabelsi et al. (2018). _Deep Complex Networks_. ICLR.** [Primary PDF](https://www.microsoft.com/en-us/research/uploads/prod/2018/04/Deep-Complex-Networks.pdf). Inspected real-pair representation, complex building blocks, normalization and experimental scope. Supports a faithful paired-real implementation and contextual complex-network feasibility. Vision/audio comparisons do not establish an advantage over this plan's exactly tied real control.

**R13 — Martin Arjovsky, Amar Shah, Yoshua Bengio (2016). _Unitary Evolution Recurrent Neural Networks_. ICML.** [Proceedings](https://proceedings.mlr.press/v48/arjovsky16.html), [paper](https://proceedings.mlr.press/v48/arjovsky16.pdf). Inspected the phase-preserving, modulus-changing activation `modReLU`, equal to `ReLU(|z|+b) z/|z|` away from zero. The unitary recurrent transition preserves the norm; this activation generally does not. Relevant antecedent for phase-preserving activations, with a different architecture from a product neuron.

**R14 — Kristína Malinovská, Slavomir Holenda, Ľudovít Malinovský (2023; arXiv revision 2024). _QuasiNet: A Neural Network with Trainable Product Layers_. ICANN.** [Full text](https://arxiv.org/html/2401.06137v2). Inspected learned factor interpolation and XOR/parity/spiral experiments. Close product-layer idea with limited baseline scope. Linked repository access failed; its current license/operability is unresolved.

**R15 — Forsad Al Hossain, Tauhidur Rahman (2025). _A Training Framework for Optimal and Stable Training of Polynomial Neural Networks_.** [Full text](https://arxiv.org/html/2505.11589v1), [code](https://github.com/Forsad/Stable-Polynomial-Train). Inspected boundary penalties, selective gradient clipping and limitations. Supports a current stabilization alternative; experiments concern polynomial activations and encrypted-inference-oriented architectures, not exact all-subsets residuals.

**R16 — Xi Xie, Ran Ran, Jiahui Zhao, Bin Lei, Zhijie Shi, Wujie Wen, Caiwen Ding (2026). _ULD-Net: Enabling Ultra-Low-Degree Fully Polynomial Networks for Homomorphically Encrypted Inference_. ICLR.** [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0feb70f08090d165e6bbfa9c85d05e87-Abstract-Conference.html), [paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/0feb70f08090d165e6bbfa9c85d05e87-Paper-Conference.pdf). Inspected PolyNorm, conditions and experimental setup, including eight-A100 training. Recent competing normalization principle; large reported systems are nonlocal reference points.

**R17 — Polylearn and nimfm maintainers, implementation references.** [Polylearn](https://github.com/scikit-learn-contrib/polylearn), [nimfm](https://github.com/neonnnnn/nimfm). Inspected READMEs/licensing: archived BSD-2-Clause polylearn documents degree-two/three FM support; MIT nimfm documents arbitrary HOFM. No current ARM build was tested. Useful equation/solver oracles; neither is a mandatory extra toolchain.

**R18 — Yury Gorishniy, Akim Kotelnikov, Artem Babenko (2025). _TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling_. ICLR.** [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html), [official code](https://github.com/yandex-research/tabm), [package definition](https://github.com/yandex-research/tabm/blob/main/pyproject.toml). Inspected defaults, numerical embeddings, per-member losses, inference aggregation, version 0.0.3 and Apache-2.0 licensing. Strong from-scratch practical control; actual grouped-data performance unmeasured.

**R19 — David Holzmüller, Léo Grinsztajn, Ingo Steinwart (2024). _Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data_. NeurIPS.** [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf), [pytabkit](https://github.com/dholzmueller/pytabkit). Inspected meta-tuned-default rationale and official estimator/device interfaces. Supports retaining the official RealMLP pipeline. Broad medium/large tabular evidence does not prove performance on tiny condition-group holdouts.

**R20 — CatBoost maintainers, official documentation, accessed 2026-10-29.** [Installation](https://catboost.ai/docs/en/installation/python-installation-method-pip-install), [parameters](https://catboost.ai/docs/en/references/training-parameters/common), [documentation](https://catboost.ai/docs/en/). Inspected macOS arm64 CPU installation and training/validation/default-rate behavior. Supports CPU comparator and freezing resolved learning rate on refit. No CatBoost model was trained here.

**R21 — scikit-learn maintainers, current estimator documentation, accessed 2026-10-29.** [Histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [KernelRidge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html). Inspected explicit validation support (documented since 1.7; stable 1.9.1 observed) and solver interfaces. Final resolved version is locked after compatibility. API availability does not establish data-split correctness.

**R22 — Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan (2026). _TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model_. ICML.** [Proceedings](https://proceedings.mlr.press/v306/qu26a.html), [official code](https://github.com/soda-inria/tabicl), [API](https://tabicl.readthedocs.io/en/latest/api.html), [changes](https://github.com/soda-inria/tabicl/blob/main/CHANGES.md), [weights/card](https://huggingface.co/jingang/TabICL), [license](https://github.com/soda-inria/tabicl/blob/main/LICENSE). Inspected regression checkpoint, eight-view API, MPS/CPU path, version 2.2.0 fixes and BSD notices. Supports a current local pretrained comparator, subject to actual probe. Pretraining is outside local cost/equal-information claims.

**R23 — Prior Labs, TabPFN current model/access documentation and 3.5 license (September 2026).** [Versions](https://docs.priorlabs.ai/models), [local code](https://github.com/PriorLabs/TabPFN), [accessing weights](https://docs.priorlabs.ai/models/accessing-model-weights), [3.5 license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE). Inspected current family, Apple support and access/use restrictions, including noncommercial/eligibility conditions. This is a gated contextual/local comparator, not freely assumed open-source weights. No account or weights were accessed for this plan.

**R24 — TabArena maintainers, current benchmark documentation, accessed 2026-10-29.** [Official repository](https://github.com/autogluon/tabarena). Inspected current multi-method/foundation-model and BeyondArena scope. Supports the limitation that a two-dataset study does not establish broad tabular leadership. Full benchmark artifacts are unnecessary and potentially very large for this question.

**R25 — I-Cheng Yeh (1998; UCI donation 2007). _Concrete Compressive Strength_.** [Dataset/variables/license](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), [DOI](https://doi.org/10.24432/C5PK67), [official archive](https://archive.ics.uci.edu/static/public/165/concrete%2Bcompressive%2Bstrength.zip). Inspected metadata/license and downloaded original workbook; measured counts/hashes in §5.5. Supports the finite new-mixture prediction task. Group keys were derived from the actual data, not claimed official subject IDs.

**R26 — Thomas Brooks, D. Pope, Michael Marcolini (1989; UCI donation 2014). _Airfoil Self-Noise_.** [Dataset/variables/license](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise), [DOI](https://doi.org/10.24432/C5VW2C), [official archive](https://archive.ics.uci.edu/static/public/291/airfoil%2Bself%2Bnoise.zip). Inspected metadata/license and original table. Supports held-condition wind-tunnel prediction; computed sweep groups do not imply new airfoil geometries or independent laboratories.

**R27 — J. Gerritsma, R. Onnink, A. Versluis (1981; UCI donation 2013). _Yacht Hydrodynamics_. UCI.** [Dataset](https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics), [DOI](https://doi.org/10.24432/C5XG7R), [official archive](https://archive.ics.uci.edu/static/public/243/yacht%2Bhydrodynamics.zip). Inspected provenance/license, the UCI-listed creators and original table. Twenty-two hull-like groups constrain inference; use only as declared robustness evidence.

**R28 — Athanasios Tsanas, Angeliki Xifara (2012). _Energy Efficiency_. UCI.** [Dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency), [DOI](https://doi.org/10.24432/C51307). Inspected description/license: Ecotect simulations based on twelve building shapes. Metadata only; not part of the measured-data structural download checks. Simulation status prevents mislabeling it as physical measurement evidence.

**R29 — PyTorch maintainers (28 October 2026). _PyTorch 2.15.0 Release Notes_.** [Pinned release](https://github.com/pytorch/pytorch/releases/tag/v2.15.0), [announcement](https://pytorch.org/blog/pytorch-2-15-release-blog/). Inspected MPS OS requirement and numerical/operator changes. Supports macOS ≥15 for 2.15 MPS and prior-release fallback. Does not prove these models' kernels work.

**R30 — PyTorch publisher package metadata; Apple setup documentation, accessed 2026-10-29.** [Torch 2.15.0 wheels](https://pypi.org/project/torch/2.15.0/), [Apple setup](https://developer.apple.com/metal/pytorch/). Inspected Python requirement, arm64 wheel availability and 2.14.1 release history. Apple's setup page lagged the newest PyTorch release; the pinned release notes control the OS branch.

**R31 — PyTorch MPS backend documentation, current and 2.14.** [Current](https://docs.pytorch.org/docs/main/notes/mps.html), [2.14](https://docs.pytorch.org/docs/2.14/notes/mps.html), [environment flags](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html). Inspected device detection, float64/complex128 limitation and fallback behavior. Exact float32/complex64 operation coverage remains a local check.

**R32 — PyTorch _Complex Numbers_ documentation, 2.14.** [Documentation](https://docs.pytorch.org/docs/2.14/complex_numbers.html). Inspected conjugate-Wirtinger gradient and optimizer/real-view semantics. Supports CPU native-complex versus paired-real validation; a global-reduction optimizer need not have identical details to independently packed tensors.

**R33 — PyTorch MPS API documentation.** [API overview](https://docs.pytorch.org/docs/2.14/mps.html), [synchronization](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html), [current allocation](https://docs.pytorch.org/docs/2.14/generated/torch.mps.current_allocated_memory.html), [driver allocation](https://docs.pytorch.org/docs/2.14/generated/torch.mps.driver_allocated_memory.html), [memory fraction](https://docs.pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction.html). Inspected synchronization and counter semantics. Supports timing boundaries and separate memory observables, not a measured device capacity.

**R34 — Apple MLX maintainers, documentation/license, accessed 2026-10-29.** [Dtypes](https://ml-explore.github.io/mlx/build/html/python/data_types.html), [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html), [peak memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html), [license](https://github.com/ml-explore/mlx/blob/main/LICENSE). Inspected current 0.32.4 documentation, complex64/CPU float64, evaluation and memory semantics, MIT license. Optional port only; no MLX measurements performed.

**R35 — PyTorch _Reproducibility_ documentation, 2.14.** [Documentation](https://docs.pytorch.org/docs/2.14/notes/randomness.html). Inspected cross-platform/version reproducibility limits and deterministic controls. Supports separating same-environment restart checks from statistical replication.

**R36 — Gavin C. Cawley, Nicola L. C. Talbot (2010). _On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation_. JMLR 11, 2079–2107.** [Primary PDF](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf). Inspected model-selection bias argument and conclusions. Supports development/confirmation separation and caution about adaptive validation; the present fixed-sample protocol is a project design.

**R37 — Xavier Bouthillier et al. (2021). _Accounting for Variance in Machine Learning Benchmarks_. MLSys.** [Primary PDF](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf). Inspected variance sources and benchmark framing. Supports accounting for data/HPO variation, not treating initialization seeds as independent tasks.

**R38 — Ang-Kun Wu, Fangdi Wen, Jingtao Zhang (September 2026). _Geometric Mean Pooling for Equal-Weight Multiplicative Coarse-Graining_.** [Primary record](https://arxiv.org/abs/2609.21876). **Discovery-only close lead:** the indexed primary abstract describes signed geometric-mean pooling; attempts to open primary full text and linked code failed. No empirical conclusion in this plan relies on it. It is a specific remaining normalization novelty question.

**R39 — _PU-UNet: Stable Multiplicative Interactions for Medical Image Segmentation_ (June 2026).** [Primary record](https://arxiv.org/abs/2606.20035). **Discovery-only lead:** indexed description suggests stable residual products using positive/log-domain handling; primary full text could not be inspected. Author and implementation details are intentionally not invented. Not a competitive result used in the plan.

**R40 — CN121039674A (published 28 November 2025), _Methods and apparatus for identifying anomalies in the behavior of dynamic systems_.** [Technical disclosure](https://patents.google.com/patent/CN121039674A/en). Inspected description of complex/logarithmic product-related computation. Relevant technical vocabulary/prior-art lead; no equivalence to the exact normalizer or legal conclusion established.

**R41 — Bhumika Mistry, Katayoun Farrahi, Jonathon Hare (2022). _A Primer for Neural Arithmetic Logic Modules_. JMLR 23(185).** [Primary PDF](https://jmlr.org/papers/volume23/21-0211/21-0211.pdf), [benchmark code](https://github.com/bmistry4/nalm-benchmark). Inspected its original harmonized arithmetic benchmark and training-failure analysis, not just its review summaries. Supports the need for consistent ranges/success definitions in any arithmetic fork.

**R42 — Noam Shazeer (2020). _GLU Variants Improve Transformer_.** [Primary paper](https://arxiv.org/pdf/2002.05202). Inspected gated-linear variant formulation. Supports the SwiGLU-style competing primitive; published sequence-model results do not predict this tabular comparison.

### 12.2 Search-coverage record and stopping rationale

Searches used two web search systems, original proceedings/arXiv/author PDFs, official GitHub/source documentation, package metadata, dataset pages and a scoped patent-disclosure search. Consequential sources were opened rather than relying solely on snippets. Backward tracing connected modern polynomial architectures to Pi-Sigma/product units and kernels; forward tracing included regularization, sparse interactions, arithmetic benchmarks and 2025–2026 normalization work.

| Search cluster | Representative queries / tracing | Material consequence |
|---|---|---|
| Literal neuron equivalents | “shifted product neuron”, “product units”, “Pi-Sigma”, “ridge polynomial network”, “sigma pi” | Historical and exact affine-factor overlap |
| Product residual / degree weights | “all-subsets kernel”, “higher-order factorization machines”, “ANOVA shared parameters”, “elementary symmetric polynomial neural network” | Closest HOFM baseline and degree-control novelty correction |
| Learned projections / practical crosses | “polynomial neural networks”, Π-Net references/citations, “DCN V2”, “multiplicative integration”, gated variants | CP/CCP/DCN/bilinear controls |
| Stability / refined candidate | “polynomial normalization”, “degree calibration”, “product initialization”, “stable polynomial training”, recent PolyNorm | Known stabilization approaches and parameterization caution |
| Roots / complex phase | “signed power normalization”, “geometric mean pooling”, “phase preserving complex activation”, “neural power units” | Separate amplitude/phase hypotheses and recent blocked leads |
| Current practical baselines | TabM/RealMLP official implementations, current TabArena, TabICLv2, TabPFN model/version/license pages | Current local comparison and access gates; avoided stale 2.5-only view |
| Hardware and packages | Pinned PyTorch release, MPS dtype/flags/memory/sync, Apple setup, MLX, baseline APIs/licenses | macOS-version branch, paired-real fallback, measurable resource plan |
| Data and inference validity | Official UCI files/provenance; grouping/duplicates; model-selection/variance papers | Actual structural measurements, group splits and appropriate independent units |
| Technical disclosures | Product-neuron/complex-arithmetic patent terms | Specific technical lead; no legal opinion |

Additional targeted refinement searches and citation tracing converged on these same competing families rather than revealing a clearly distinct surviving mechanism. Research stopped when new hits no longer changed the selected direction, indispensable baselines or execution constraints. This is a coverage-based stopping decision, not a paper-count quota or proof that no prior art exists elsewhere.

Remaining blind spots include inaccessible recent full texts, non-English/unindexed technical disclosures, unpublished systems, unverified current ARM builds of old research packages, exact training-data/benchmark-selection exposure of external assets, and all target-Mac execution. If a later novelty claim depends on one of those areas, resolve that specific gap before making the claim.

## Appendix A. Reproducible planning-stage checks

### A.1 Numerical checks actually executed

Environment: Linux x86_64, Python 3.12.14, NumPy 2.3.5; random generator seed `20261029`. No Torch installation or model training was used. For 1,000 random factor-increment vectors per `k`, sampled `Normal(0,.25²)`, the largest float64 difference between the stable higher-degree recurrence and direct subtraction was `2.66e-15` at `k=8`. The degree-sum recurrence and the higher-residual recurrence differed by at most `8.88e-16` in those samples.

Four identical float32 increments give this cancellation diagnostic (reference uses the exactly represented float32 inputs promoted to float64):

| Increment | Float64 reference residual | Float32 subtractive result | Float32 recurrence relative error |
|---:|---:|---:|---:|
| `1e-2` | `6.040099729e-4` | `6.039962173e-4` | `3.09e-8` |
| `1e-4` | `6.000399707e-8` | `6.638583727e-8` | `1.91e-8` |
| `1e-6` | `6.000003970e-12` | `-1.853027243e-7` | `1.51e-8` |
| `1e-8` | `5.999999967e-16` | `-3.999999976e-8` | `4.44e-8` |

Minimal reproduction code, using NumPy only:

```python
import numpy as np

def residual(t):
    lin = np.zeros(t.shape[:-1], dtype=t.dtype)
    high = np.zeros(t.shape[:-1], dtype=t.dtype)
    for z in np.moveaxis(t, -1, 0):
        old_lin = lin
        high = high * (1 + z) + old_lin * z
        lin = old_lin + z
    return high

for increment in (1e-2, 1e-4, 1e-6, 1e-8):
    t = np.full((1, 4), increment, dtype=np.float32)
    reference = residual(t.astype(np.float64))[0]
    naive = (np.prod(1+t, axis=-1) - 1 - t.sum(-1))[0]
    stable = residual(t)[0]
    print(increment, reference, naive, stable,
          abs(stable-reference)/abs(reference))
```

The radial-map analytic Jacobian at `P=.7−.9i`, `k=4`, `ε=.01` agreed with central real-coordinate finite differences using step `1e-6` to a maximum absolute discrepancy of approximately `7.26e-11`. Crossing the negative-real phase cut at angles `π−1e-8` and `−π+1e-8` gave a principal-fourth-root jump of approximately `1.41421356`; the radial map changed by approximately `2.00e-8`. These demonstrate the specified local behavior; they are not global error bounds or measured learning advantages.

### A.2 Data checks actually executed and how to repeat them

Download the exact official archives in R25–R27; hash archive bytes and extracted data members. Parse Concrete's first sheet as a numeric nine-column matrix with its header removed, and the Airfoil/Yacht whitespace data as numeric matrices. Do not round. Then perform:

```python
# A: [rows, features+target], with target last.
X, y = A[:, :-1], A[:, -1]
unique_X = len(np.unique(X, axis=0))
unique_Xy = len(np.unique(A, axis=0))
# concrete columns 0:7; airfoil columns 1:5; yacht columns 0:5
keys, inverse, counts = np.unique(X[:, group_columns], axis=0,
                                  return_inverse=True, return_counts=True)
assert np.isfinite(A).all()
print(A.shape, unique_X, unique_Xy, len(keys), counts.min(), counts.max())
```

Expected outputs for the inspected snapshots are Concrete `(1030,9), 996,1005,428,1,20`; Airfoil `(1503,6),1503,1503,106,8,19`; Yacht `(308,7),308,308,22,14,14`. These derived checks require no predictive training. They are reproducible structural facts about those byte snapshots, not proof of subject independence.

## Appendix B. Completion checklist for the implementing agent

- The original shifted neuron, root proposal and complex hypothesis remain identifiable, with every refinement labeled.
- Every claimed contribution has a closest-prior-art mapping and an isolating comparison.
- Every main performance claim names a population, comparator policy, metric, uncertainty method and cost scope.
- All expensive stages depend on mathematical, backend, data and baseline gates.
- The confirmation freeze contains actual versions, hashes, numeric budgets and sample sizes, not template strings.
- Every scheduled run, including failures and interruptions, has an auditable disposition.
- Strong baselines receive faithful preprocessing/training and suitable tuning; nonlocal references are separate.
- Hardware adaptations preserve the indispensable mechanism, or the claim is amended.
- Protected evaluation is not reused as fresh confirmation after a methodological change.
- The final report labels source findings, actual measurements, estimates, deductions and unperformed work separately.
- The next action follows from observed evidence, including the possibility of stopping with no demonstrated advantage.
