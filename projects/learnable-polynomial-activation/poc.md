# Research and proof-of-concept plan: learned polynomial activations

**Research date:** 29 October 2026  
**Target execution environment:** one MacBook Pro with an M3 Max; actual memory, operating system and free storage must be discovered.  
**Deliverable status:** researched experimental handoff, not a claim of an implemented or successful neural-network method.  
**Primary decision:** investigate a learned cubic center with explicitly bounded slopes and continuously differentiable linear tails, as a controlled refinement of existing methods. Methodological novelty is **not established**.

## 1. Executive assessment and recommended direction

The original proposal—learning polynomial activation coefficients jointly with network weights—is substantially known. Adaptive polynomial neurons appeared in 1993, neuron-specific Hermite activations in 2005, and modern work includes scalable orthogonal activations, Bernstein activations, smooth adaptive piecewise polynomials, constrained splines and rational activations. Matching forward and backward second moments at initialization is already part of the closest ICLR 2026 polynomial work. A new name or an implementation on Apple Silicon would not change that assessment. [R01] [R02] [R03] [R04] [R05] [R06] [R07]

The recommended experiment preserves the original intent: **allow the network to learn its scalar nonlinear transformations**. It changes the global-polynomial requirement explicitly. The candidate is cubic on a fixed central interval and linear outside, with a hard bound on its scalar derivative. This sacrifices globally polynomial inference, but permits a precise stability intervention that can be isolated experimentally. It is a small, reproducible refinement, not a proposed new category of neural network.

The central question is:

> On two specified physical-regression datasets, does learning this bounded cubic produce a reproducible improvement in prediction error or a useful error–cost–reliability tradeoff over well-tuned, locally executable activation alternatives, and can the improvement be attributed to learned shape, controlled slopes, or linear tails?

The indispensable competitors are current Hermite activations, a learned quadratic with the same free tails, BerLU, constrained learned splines, safe rational activations, PReLU, and strong fixed activations. A separate practical comparison includes CatBoost, TabM, wider ordinary MLPs, and an eligible local TabPFN checkpoint when its access terms and actual laptop execution permit. Winning against an intentionally unstable monomial cubic is insufficient.

**Recommended outcome hierarchy:**

1. A reproducible empirical advantage over the strongest eligible comparisons, surviving attribution and cost checks, is a meaningful local result.
2. An improvement only over unrestricted polynomials is a narrower stability result.
3. An advantage only on polynomial synthetic targets establishes a matching inductive bias.
4. A stable implementation with no competitive advantage is a useful negative finding.
5. Wide uncertainty, a defective implementation, or insufficient local resources is an inconclusive result with an identified cause.

No evidence currently establishes a breakthrough, general-purpose superiority, encrypted-inference advantage, or large-model scaling benefit. The program is designed to discover a real advantage if one exists, while making these other outcomes possible.

### 1.1 What has actually been verified

This handoff distinguishes five evidence classes:

| Label | Meaning | Present status |
|---|---|---|
| **Source finding** | Inspected primary publication, author repository, official documentation or dataset record | Prior-art mechanisms, current baseline families, dataset provenance and documented platform constraints |
| **Deduction** | Mathematics or engineering reasoning given stated assumptions | Scalar slope proof; polynomial degree limitation; Gaussian variance/derivative identity; memory equations |
| **Measured here** | A calculation or data inspection actually executed during preparation | CPU float64 scalar checks and public-dataset schema/group/split checks |
| **Proposed** | Work the coding agent must implement or execute | All neural-network training, tuning, comparisons, timing, MPS parity and resumption checks |
| **Unknown** | Not established by available evidence | Target-laptop compatibility/performance, a beneficial effect, exact code–paper parity for several dependencies, broad novelty |

The calculations ran on Linux x86_64, Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0. Dataset XLS parsing used xlrd 2.0.1. They are **not M3 Max measurements**. No neural-network baseline was trained, and no protected predictive test score was computed.

## 2. Problem statement, scope and hypotheses

### 2.1 Intended beneficiary and bottleneck

The immediate beneficiaries are researchers and developers of compact neural regressors who might benefit from learning nonlinear response shapes with few extra parameters. The existing workflow selects a conventional activation and tunes network architecture and optimization. The proposal adds trainable scalar function parameters.

There are two different possible bottlenecks:

- A fixed activation may use parameters or samples inefficiently for a particular smooth interaction structure.
- An expressive polynomial may generate large activations and derivatives when its input distribution changes.

The first is an application-dependent assumption. The second is a mathematical risk for non-affine global polynomials, but its practical severity depends on normalization, residual connections, initialization, weights and workload. Neither is assumed to dominate a well-tuned contemporary baseline.

Inputs are numeric features \(x\in\mathbb R^d\); outputs are scalar responses \(y\). The training objective is squared prediction error in train-standardized target units, with ordinary weight regularization as specified below. Primary reporting uses held-out group-balanced RMSE in original units and a dimensionless cross-task log-RMSE contrast.

### 2.2 Scope decisions

- **Primary mechanism:** a per-layer shared, learnable cubic with slope constraints and linear tails.
- **Real confirmation panel:** UCI Concrete Compressive Strength and Superconductivity, with grouped splits defined in Section 5.
- **Development:** Airfoil and balanced synthetic controls.
- **Core model scale:** compact residual and plain MLPs. Deep stress experiments are diagnostic, not substitutes for practical models.
- **Training:** from scratch, full float32 on MPS if validated; otherwise a faithful all-CPU path.
- **Numerical oracle:** CPU float64.
- **Excluded from the core claim:** LLM training, ImageNet replication, physical-law discovery, full robustness certification, homomorphic encryption, neural architecture search and large-scale foundation-model pretraining.

These choices preserve a locally testable component-level question. They do not imply that two tabular tasks represent all neural-network applications.

### 2.3 Hypothesis ledger

| ID | Hypothesis and predicted observation | Supporting rationale | Strong alternative explanation | Weakening or falsifying observation |
|---|---|---|---|---|
| H1 | Learned bounded cubic improves the frozen finite-panel prediction procedure over the validation-selected activation champion | Flexible shape with controlled tails may fit useful nonlinear structure | Better initialization, generic affine calibration, extra tuning or parameters | Effect disappears against nested quadratic, spline/rational competitors, frozen-shape control or fair tuning |
| H2 | Tails and slope projection improve behavior when their interventions are actually exercised | Bounded derivative and linear tails remove polynomial tail growth | Gradient clipping, normalization or residual scaling already solves the issue | No benefit after those generic protections; weights compensate for the cap; cap/tails are inactive |
| H3 | The cubic degree itself is useful beyond a learned quadratic | Derivative curvature adds one scalar shape degree | Free tail slopes or offset explain the gain | Learned quadratic with identical tails and offset matches or improves the result |
| H4 | Any improvement survives end-to-end cost | Four parameters per shared activation are small | Extra kernels and tuning dominate despite low parameter count | Wider fixed activation, small ensemble, CatBoost or TabM reaches a better error–cost frontier |
| H5 | Polynomial structure helps interaction learning | Quadratic activations represent products directly | Ground truth was constructed to match the model | Success only on aligned synthetic functions, or explicit quadratic regression performs as well |
| H0 | No useful differentiated advantage exists at this scale | Strong existing families already span useful shapes | Experiment lacks exposure, sensitivity or scale | A sufficiently sensitive, correctly implemented comparison gives consistent negative results across the tested settings |

H1 is the main confirmatory claim. H2–H5 guide predeclared controls and narrower interpretations. Supporting evidence for one is not automatically evidence for the others.

### 2.4 Strongest case for and against the idea

**For:** local nonlinear shape adaptation could reduce the width needed for a response function; low-degree polynomial evaluation can be simple; explicit tail behavior can protect against rare preactivation excursions; and a small activation parameter budget makes mechanism studies inexpensive.

**Against:** existing rational functions and splines already provide this flexibility; modern normalization/residual schemes may leave little stability headroom; scalar slope limits can be compensated by adjacent weights; the bounded family may underfit sharp gating or genuine polynomial extrapolation; and many small tensor operations can be slower than optimized conventional activations.

For a residual model of width \(h\) and \(L\) blocks, four shared activation parameters per block add \(4L\) parameters to roughly \(2Lh^2\) dense weights. At \(h=128,L=4\), that is 16 activation parameters against roughly 131,072 block weight entries, before stem/head/biases. This is a parameter-count calculation, **not a speed prediction**.

If activation work occupies fraction \(p\) of measured baseline runtime and a replacement changes that cost by factor \(r\), the simplified total-runtime ratio is \(1-p+pr\), before launch, memory and fusion effects. Measure \(p\) and end-to-end runtime; do not infer a win from polynomial FLOPs or coefficient count.

## 3. Prior art and precise novelty assessment

### 3.1 Comparison matrix

Evidence descriptions below are deliberately qualitative. Published scores from different architectures, splits and budgets are not controlled comparisons for this project.

| Prior work | Problem and mechanism | Assumptions/evidence and cost | Reuse/availability | Exact overlap and implication |
|---|---|---|---|---|
| Adaptive polynomial neurons, Piazza et al., 1993 [R01] | Jointly adapt polynomial neuron responses in multilayer networks | Historical work; publisher metadata/abstract inspected, not full chapter | No execution-ready code verified | Directly anticipates the original idea |
| Hermite constructive networks, Ma & Khorasani, 2005 [R02] | Neuron-specific Hermite activations and structural adaptation | One-hidden-layer constructive setting; abstract inspected | No current implementation verified | Orthogonal learned activations are longstanding |
| SAAF, Hou et al., 2017 [R03] | Learned smooth piecewise-polynomial activations with complexity/Lipschitz regularization | Regression CNNs; representational flexibility grows with segments | Historical implementation need not be ported for the core study | Smooth polynomial learning plus slope control is not new |
| Hermite SSL, Lokhande et al., 2020 [R08] | Hermite expansions for pseudo-labeling | Reports favorable semi-supervised results but no clear supervised verdict; task and stabilization choices matter | Author code linked by paper | Important mixed evidence against universal benefit |
| Chebyshev/Range Norm, Wang et al., 2022 [R09] | Polynomial basis plus range normalization | Synthetic and precipitation settings; normalization is part of method | Author-hosted paper available; code parity unverified | Range control and scientific-data application already exist |
| Orthogonal activations, Khalfaoui-Hassani & Kesselheim, ICLR 2026 [R04] | Hermite/Fourier/tropical functions; matching forward/backward second moments at initialization | ConvNeXt, GPT-2 and smaller experiments; distributional initialization is not a global training-time bound | torchortho, GPL-3.0; pure-tensor paths and kernels must be distinguished [R20] | Closest modern polynomial baseline; “manage both moments” is insufficient novelty |
| SLAF, Pulido-Gaytan & Tchernykh, 2024 [R10] | Learn polynomial coefficients for encrypted CNN inference | MNIST/HE context; arithmetic costs differ from ordinary inference | Linked CNN-HE-SLAF repository; code license not verified | Direct original mechanism; tails/LayerNorm do not preserve its polynomial-only premise |
| DeepBern-Nets, Khedr & Shoukry, AAAI 2024 [R05] | Bernstein activations with interval-bound propagation | Certification/reachability; uses activation domains, not a scalar global guarantee from basis choice alone | Official BSD-3-Clause repository [R22] | Bernstein coefficients and interval-derived bounds are established |
| DeepBern 2026 revisions [R11] | v1 cumulative positive coefficient increments, normalization/clamping; v2 approximation and parameter-efficiency study | v2 explicitly assumes relevant polynomial approximator is representable; HIGGS/SUSY claims require their conditions | Exact follow-on implementation not verified | v1 already uses slope-related parameterization; both revisions count as prior art |
| BerLU, Zhang et al., 2026 [R06] | Quadratic center, learnable negative slope, \(C^1\) linear tails | Vision experiments; true scalar bound is \(\max(1,\lvert\alpha\rvert)\), not automatically one during learning | Equation implementation feasible; official code not verified | Direct simpler local competitor to the candidate |
| Controlled deep splines, 2020–2025 [R07] [R12] [R13] | Learn linear splines with slope constraints and TV regularization | Whole-network guarantees additionally constrain linear operators; inverse problems/classification examples | Current standalone DeepSplines module is MIT [R21] | Learned shape under derivative constraints is mature prior art |
| PAU/rational activations, 2020 onward [R14] | Learn numerator and safe-denominator polynomial coefficients | Flexible shapes; positive denominator prevents poles, not all amplification | MIT rational_activations; CUDA-oriented paths require a faithful tensor alternative [R23] | Essential challenge to polynomial-only design preference |
| KAN and KAT [R15] [R16] | Edge-wise learned functions; rational grouped variants in transformers | Changes architecture and parameter allocation beyond a scalar activation swap | Author repositories; some optimized kernels require CUDA/Triton | Relevant adjacent architecture, not interchangeable evidence for this candidate |
| Polynomial networks/factorization machines [R17] | Low-rank interaction/tensor models | Direct structured interaction learning; cost depends on rank/order | Existing mathematical algorithms and author implementations | Product-feature reasoning is established; use direct quadratic controls |
| Moment-SOS polynomial activation training, 2026 [R18] | Recasts joint fitting as polynomial optimization | Different optimizer and scaling regime; publisher abstract inspected | No laptop-scale SGD baseline reproduction established | Real supplied reference, but not evidence for scalable gradient training |
| Constrained rational / sinusoidal degree work, 2026 [R19] [R24] | Structural constraints for rational stability; learned spectral components | RL/continual or parity/frequency settings differ from this PoC | Author code links exist; compatibility/license details remain gates | Preempts broad “constraint + learned shape” or “allocate degree” originality claims |

### 3.2 What is known, reusable and potentially different

1. **Already exists:** learned coefficients, orthogonal bases, Bernstein parameterizations, moment-aware initialization, piecewise polynomial activation learning, slope-constrained splines, linear tails, rational functions and interaction-learning networks.
2. **Reuse:** official equations and small activation modules; PyTorch autograd/optimizers; established data/statistical packages; UCI data; author implementations for parity checks; maintained TabM/CatBoost. Do not rebuild GPU kernels or a benchmark orchestration service.
3. **Potentially different:** this exact signed-slope cubic, its transparent factorial intervention and its finite-panel, failure-aware, cost-accounted comparison may produce an empirical result. Exact first publication of this combination has **not** been established.
4. **Possible value:** identify a compact shape family with a favorable accuracy/reliability/cost tradeoff, or establish that cheaper existing choices remove the apparent headroom.
5. **Discriminating experiment:** the same cubic coordinates with tails and projection independently toggled; learned versus frozen shape; nested quadratic with identical offset and both tails free; strong spline/rational/Hermite competitors.
6. **Novelty invalidation:** an earlier equivalent parameterization, constraint, tail extension or comparison protocol invalidates the corresponding method claim. A positive result can still be a new empirical observation if its workload/protocol and limits are reported honestly.

**Classification:** replication plus controlled refinement, with a possible new empirical finding and later engineering optimization. No claim of new mathematical principles or established methodological novelty is made.

### 3.3 Reconcile seemingly conflicting evidence

The ICLR polynomial work shows that global polynomials can train at substantial scale under suitable initialization; it therefore weakens any premise that tails are always necessary. Its distributional calculations do not establish uniform bounds throughout learned training. The SSL paper’s mixed supervised evidence warns that favorable activation behavior may be regime-specific. Bernstein and spline results concern different constraints, domains and architectures. Published “parameter efficiency” or “stability” statements cannot be imported into this handoff without their assumptions. [R04] [R08] [R11] [R12]

The correct experiment gives unrestricted Hermite and generic stabilizers a fair chance. If those methods are already stable and accurate, added constraints must earn their overhead.

### 3.4 Direction selection

| Direction | Value and distinctiveness | Local falsifiability and dependencies | Decision |
|---|---|---|---|
| Unrestricted normalized Hermite | Strong established baseline; original intent preserved | Very easy to implement, but already close to published work | Mandatory comparator and possible final recommendation |
| Bounded cubic with linear tails | Explicit tail intervention, small scalar budget; novelty uncertain | Exact scalar properties and matched causal controls are cheap | **Primary controlled refinement** |
| Interaction-specific polynomial network | Clear inductive bias; extensive prior art | Synthetic demonstration easy; real application evidence required | Diagnostic and conditional application pivot |
| Learned degree for HE | Potential arithmetic value, already explored | Must measure actual encrypted circuit/security/accuracy costs | Outside core; only pursue after a concrete HE requirement |
| Rational or spline family | Strong flexible alternatives, potentially better | Locally implementable without custom GPU kernels | Mandatory competitors; adopt if they dominate |
| Large model or architecture redesign | Potentially larger scale effects | Confounded and costly before component evidence | Escalate only after specified evidence |

## 4. Mathematical analysis and implementation specification

### 4.1 Correct limitations of the original global polynomial

For \(L\) affine layers with scalar polynomial activations of degrees \(d_1,\ldots,d_L\), and no nonpolynomial operations, total degree is **at most** \(\prod_l d_l\). Cancellations can lower it. The network does not thereby span every polynomial of that degree; width, rank and parameter sharing constrain the representable set. LayerNorm, division and piecewise tails invalidate the claim that the complete model is a single polynomial.

For a finite polynomial of degree at least two, the derivative is unbounded on the real line. Changing from monomials to Hermite or Bernstein coordinates changes conditioning and optimization geometry, but not the unrestricted polynomial function class.

For normalized probabilists’ Hermites \(h_k=\mathrm{He}_k/\sqrt{k!}\), \(X\sim N(0,1)\), and \(f=\sum_{k=0}^K c_kh_k\),

\[
\operatorname{Var}f(X)=\sum_{k=1}^Kc_k^2,\qquad
E[f'(X)^2]=\sum_{k=1}^Kk c_k^2.
\]

Consequently,

\[
E[f'(X)^2]-\operatorname{Var}f(X)
=\sum_{k=2}^K(k-1)c_k^2\geq0.
\]

This directly derived Gaussian spectral identity means a genuinely nonlinear centered polynomial cannot simultaneously have unit output variance and unit derivative second moment under the stated scalar input law. It is not a new theorem. Nonzero \(c_0\) changes the **uncentered second moment** \(E[f^2]\), not the variance, so equality of forward/backward initialization gains in [R04] does not contradict it. Random affine preactivation variance calculations can depend on \(E[f^2]\).

LayerNorm does not make each coordinate Gaussian across examples. Even an exactly correct \(E[f'^2]\) calculation is not a theorem about the complete trained Jacobian: weight correlations, normalization derivatives, residual paths and singular values matter.

The interaction identity

\[
\sum_j v_j(w_j^\top x)^2=x^\top
\left(\sum_jv_jw_jw_j^\top\right)x
\]

shows rank at most the number of terms. It motivates explicit quadratic/rank-factorized controls; it does not by itself establish new interaction learning. [R17]

### 4.2 Selected scalar family

Let radius \(R>0\), cap \(S>0\), offset \(\beta\in\mathbb R\), and three learned slopes \(s_0,s_1,s_2\in[-S,S]\). For \(t\in[0,1]\), define

\[
q(t)=s_0(1-t)^2+2s_1t(1-t)+s_2t^2.
\]

This is a quadratic derivative in the Bernstein basis. Set

\[
A=\frac{s_0+2s_1+s_2}{4},\quad
B=\frac{s_2-s_0}{4R},\quad
C=\frac{s_0-2s_1+s_2}{12R^2},
\quad P(x)=\beta+x(A+x(B+xC)).
\]

For \(u=\operatorname{clip}(x,-R,R)\), use

\[
\phi(x)=P(u)+s_{\mathrm{tail}}(x-u),\qquad
s_{\mathrm{tail}}=
\begin{cases}
s_0,&x<-R,\\
s_2,&x\geq-R.
\end{cases}
\]

Inside the interval, \(x-u=0\). Outside, this is the tangent line at the appropriate endpoint.

**Invariants and proof:**

- \(P'(x)=q((x+R)/(2R))\) on the interval.
- Bernstein weights are nonnegative and sum to one there, so \(|P'|\le S\).
- The tail slopes equal the central endpoint derivatives, so \(\phi\in C^1(\mathbb R)\).
- Hence \(\phi\) is globally \(S\)-Lipschitz over real arithmetic.
- \(|\phi(x)|\le|\beta|+S|x|\).
- This is generally not \(C^2\) at the joins.
- This does **not** certify a Lipschitz bound, robustness radius or nonvanishing gradient for the full network.
- The box constraint is sufficient, not necessary: some derivative quadratics stay within the cap while an interior Bernstein control lies outside it.

Use the direct slope coordinates with a projection after each optimizer step. Do not silently replace them with tanh coordinates: that changes optimization geometry and saturation behavior.

**Primary defaults:** \(R=1.5,S=1.5\), cubic center, one shared function per activation layer. These are starting configurations, not asserted optima. Primary architecture comparison may tune radius/cap under the explicit validation procedure. The matched factorial fixes them.

### 4.3 Reference module

The following is a **proposed project implementation**, not code executed on MPS during preparation. It specifies semantics; the coding agent must implement and validate it before use.

~~~python
import torch
from torch import nn

class BoundedCubic(nn.Module):
    # Scalar function shared across all entries of this layer.
    def __init__(self, beta, slopes, radius=1.5, cap=1.5,
                 tails=True, project=True):
        super().__init__()
        if radius <= 0 or cap <= 0 or len(slopes) != 3:
            raise ValueError("Invalid scalar activation configuration")
        self.radius, self.cap = float(radius), float(cap)
        self.tails, self.use_projection = bool(tails), bool(project)
        self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))
        self.slopes = nn.Parameter(torch.tensor(slopes, dtype=torch.float32))
        self.project_()

    def forward(self, x):
        r = self.radius
        s0, s1, s2 = self.slopes.unbind()
        u = x.clamp(-r, r) if self.tails else x
        a = (s0 + 2*s1 + s2) / 4
        b = (s2 - s0) / (4*r)
        c = (s0 - 2*s1 + s2) / (12*r*r)
        y = self.beta + u * (a + u * (b + u*c))
        if self.tails:
            side = torch.where(x < -r, s0, s2)
            y = y + side * (x - u)
        return y

    @torch.no_grad()
    def project_(self):
        if self.use_projection:
            self.slopes.clamp_(-self.cap, self.cap)
~~~ 

Preserve radius/cap/toggle values in the frozen constructor configuration and checkpoint manifest. Per-channel variants use shape \((h,3)\) and last-dimension broadcasting; they are a separate exploratory treatment, not an unnoticed replacement.

**Numerical convention:** evaluate the polynomial only on the clamped input when tails are enabled. Selecting tails after evaluating \(P(x)\) at enormous \(x\) can overflow even in an inactive eager branch. The \(x-u\) expression also avoids double-counting input gradients at the exact joins. Floating-point overflow remains possible for sufficiently extreme finite inputs even for a linear function; the real-arithmetic proof is not a universal floating-point finiteness guarantee.

### 4.4 Optimizer and projection semantics

- Slopes and \(\beta\) get no AdamW weight decay by default; ordinary dense weights have their tuned decay. Biases follow the common no-decay convention.
- Shape-parameter learning-rate multiplier is selected from \(0.1,1,10\) during fair tuning; matched controls use the same value.
- Check loss and gradients for finiteness before update. Apply any **shared, predeclared** gradient clipping after logging the unclipped norm.
- Step AdamW; check weights and optimizer state for finiteness before projection, so clipping cannot conceal an infinite slope. Then project slopes, retain optimizer moments unchanged, and check again.
- Record pre/post-projection values, fraction of coordinates projected, and cap occupancy. Do not reset momentum undocumented.
- Compute the actual maximum absolute derivative from q(0), q(1), and q(t*) when \(t^*=(s_0-s_1)/(s_0-2s_1+s_2)\) lies strictly between zero and one. Omit t* when its denominator is zero; then q is affine or constant. The maximum of these absolute values also covers the linear tails. This distinguishes a conservative coefficient cap from the attained slope.

The cap is an activation intervention, not an attempt to certify the unconstrained affine weights. Log weight norms and whole-model sensitivity as specified later.

### 4.5 Initialization

Use a deterministic, label-free, CPU float64 bounded least-squares fit of the candidate to exact GELU \(g(x)=x\Phi(x)\):

1. Grid: 1,025 evenly spaced points on \([-6,6]\).
2. Function evaluation uses the candidate’s tails and selected radius.
3. Build four design columns: constant one and the function produced by each unit slope vector with zero offset.
4. Multiply residual rows by \(\exp(-x^2/4)\), so squared residuals receive Gaussian-proportional weights.
5. Solve for \((\beta,s_0,s_1,s_2)\), with free offset and slopes bounded by \([-S,S]\), using CPU bounded linear least squares.
6. Save solver version/tolerance, design hash, fit residual and coefficients. Copy the resulting coefficients into float32.
7. The unconstrained factorial cells start from the **same coefficients**, not a separate advantageous fit. Their central functions match initially; tail behavior intentionally differs outside the interval.

For \(R=S=1.5\), the calculation performed during this research gave approximately

\[
\beta=0.0457000,\quad
(s_0,s_1,s_2)=(-0.243142,0.500000,1.243142),
\]

with Gaussian-weighted grid RMS fit error approximately 0.043737. These are initialization calculations, not prediction results. Recompute rather than treating rounded values as canonical.

An earlier zero-offset, wider-interval fit saturated the cap and fit poorly; allowing the redundant affine offset and using the stated radius removed that particular initialization defect. Identity initialization remains a diagnostic because an all-identity network can obscure the value of nonlinear learning.

Native published initializers remain available to the corresponding baselines. Record their fit/gain differences rather than forcing every method into a weak common approximation.

### 4.6 Mandatory nested controls

The following controls distinguish shape, degree, constraints and parameterization:

| Control | Exact change | What it can identify |
|---|---|---|
| Global/free cubic | Same \(A,B,C,\beta\), no tails and no projection | Base family and optimizer geometry |
| Global/boxed cubic | Same cubic everywhere, slope box applied | Interval coefficient control without global tails |
| Tailed/free cubic | Linear tails, unconstrained slopes | Tail effect without a slope cap |
| Tailed/boxed cubic | Full candidate | Combined intervention |
| Frozen candidate | Same initialized shape, no shape updates | Whether learning the activation earns value |
| Learned quadratic | Enforce \(s_1=(s_0+s_2)/2\); learn \(\beta,s_0,s_2\), same tails/cap | Cubic degree beyond free endpoint slopes and offset |
| Affine member | \(s_0=s_1=s_2=c\) | Limiting case and nonlinear headroom |
| Affine-calibrated GELU | \(a\,\mathrm{GELU}(x)+b\), learned \(a,b\) | Generic gain/offset explanation |

For the quadratic, optimize only the two endpoint parameters and derive the middle value each forward pass; project endpoints. Do not train three independent parameters and overwrite one after each step.

A monomial cubic with free offset spans the same global cubic family. Adam is not invariant to this linear coordinate change, so a difference between monomial and slope coordinates may be optimization conditioning rather than representational power.

### 4.7 Smallest meaningful mechanism experiment

Before real-data tuning, fit a small two-layer network on four equally weighted synthetic families: affine, low-rank quadratic interaction, smooth nonpolynomial and sharp piecewise response. Use the same base functions and noise scales for all methods, with separately generated development and confirmation samples.

A concrete starting suite in \(d=8\):

\[
\begin{aligned}
f_1(x)&=x_1-0.5x_2,\\
f_2(x)&=2x_1x_2+0.6(x_3^2-1),\\
f_3(x)&=\sin(x_1)+0.3\tanh(x_2+x_3),\\
f_4(x)&=\max(0,x_1)-0.5\max(0,-x_2).
\end{aligned}
\]

Normalize each noiseless target using training-distribution moments estimated from a separate generator stream; add independent Gaussian noise with standard deviation 0.1 in those standardized units. Also run a noiseless correctness condition. Begin with 256 and 2,048 training examples, 1,024 validation examples and 8,192 fresh diagnostic examples; enlarge only to resolve a documented sample-size or approximation question.

Use \(N(0,I)\) inputs first. For shift diagnostics generate new inputs from \(N(0,1.5^2I)\) and variance-normalized Student-\(t_5\), and **recompute correct labels**. No task is allowed to dominate the suite because it is naturally polynomial. Include degree-two ridge and a directly factorized quadratic model for \(f_2\).

Define that factorized diagnostic as \(b+u^\top x+\sum_{j=1}^r v_j(w_j^\top x)^2\), with unrestricted signed v, ranks r in {1,3,8}, and a scalar intercept. Initialize w entries with standard deviation \(1/\sqrt d\), v with \(1/\sqrt r\), and u/b at zero using separate stored random streams. Train with the same squared loss and common optimizer/horizon choices. Rank three can represent the noiseless f2; compare its optimizer result with degree-two ridge so a failure of nonconvex factor training is not confused with absent interaction capacity.

This experiment can establish implementation correctness, exposure to tails and an inductive-bias pattern. It cannot establish practical superiority; continue to the protected real tasks even if the most favorable synthetic target wins.

### 4.8 Small checks executed during plan preparation

CPU float64 checks used seed 20261029, 200 randomly drawn slope triples in \([-1.5,1.5]^3\), and 4,001 input points on \([-30,30]\), with \(R=3\) for the property scan:

| Check | Measured result | Interpretation |
|---|---:|---|
| Maximum scanned absolute derivative | 1.4943151871 | Consistent with bound 1.5; proof, not a grid, establishes global validity |
| Identity case maximum absolute error | 0 | Correct affine limiting case on this grid |
| Finite-difference derivative maximum error | \(1.71\times10^{-10}\) | Small CPU numerical check away from joins |
| Centered BerLU special-case maximum error | \(8.88\times10^{-16}\) | Confirms direct overlap with simpler quadratic family |
| Normalized Hermite mean, \(\alpha=.3,\beta=-.2\) | approximately \(-8.33\times10^{-17}\) | Consistent with zero |
| Hermite second moment | 0.9999999999999997 | Consistent with unit variance under the Gaussian assumption |
| Hermite derivative second moment, quadrature/formula | 1.1504424778761062 / same | Confirms forward variance does not imply unit derivative energy |

The Hermite check used 64-point probabilists’ Gauss–Hermite quadrature. PyTorch autograd, coefficient gradients, exact clamp-boundary gradients and MPS behavior remain required implementation tests.

## 5. Data, provenance and protected evaluation population

### 5.1 Selected datasets and measured access check

All three official UCI records identify the datasets as publicly available under CC BY 4.0. Archive the attribution and license. The raw assets were downloaded and schemas/groups inspected during preparation; no predictive scores were computed. [R25] [R26] [R27]

| Dataset | Role | Rows × input features | Group definition | Train / validation / test groups | Train / validation / test rows |
|---|---|---:|---|---:|---:|
| Airfoil Self-Noise, UCI 291 | Development only | 1,503 × 5 | Same angle, chord, velocity and displacement thickness; frequency varies within group | 63 / 21 / 22 | 895 / 285 / 323 |
| Concrete Compressive Strength, UCI 165 | Confirmatory task 1 | 1,030 × 8 | Same seven ingredients; age varies within mixture | 256 / 85 / 87 | 599 / 215 / 216 |
| Superconductivity, UCI 464 | Confirmatory task 2 | 21,263 × 81 | Same normalized elemental composition | 9,098 / 3,032 / 3,034 | 12,930 / 4,282 / 4,051 |

Targets are sound pressure in dB, compressive strength in MPa and critical temperature in kelvin. The superconductivity data concern known superconductors; they do not test whether arbitrary materials superconduct. [R25] [R26] [R28]

Airfoil has only 22 held-out groups under this split. It is therefore used for development and illustrative diagnostics, without a confirmatory generalization claim. The two real confirmation tasks form a **fixed panel**, not a random sample of scientific or tabular problems.

Measured checks: all numeric values were finite, targets were nonconstant, superconductivity files aligned row-for-row on their target columns, and no identical complete input vector crossed the proposed groups/splits. Unknown laboratory/provenance correlations may remain.

### 5.2 Immutable download specification

| ID | Public archive URL | Archive SHA256 |
|---|---|---|
| 291 | https://archive.ics.uci.edu/static/public/291/airfoil%2Bself%2Bnoise.zip | 5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c |
| 165 | https://archive.ics.uci.edu/static/public/165/concrete%2Bcompressive%2Bstrength.zip | dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7 |
| 464 | https://archive.ics.uci.edu/static/public/464/superconductivty%2Bdata.zip | 87f4490d73390ff94ee01dbf0d7d32abc80b22f2c803d471765cfc46a9f6371e |

Expected materialized data hashes:

- airfoil_self_noise.dat: 74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3
- Concrete_Data.xls: 710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4
- train.csv: 4dfb6e3a1f6ffd969e5a5e42f093c4800d1e2a6c8b1e309f8fcd9f23d86952f3
- unique_m.csv: b68ae6b55ea8581eff8b1ffba073a899db7e2d2f7f3b781bb0802f643f51e5f7

The three archives total about 8.4 MB; materialized files about 28.5 MB. Retain original archives and hashes. A changed hash is a provenance change to inspect, not a reason to silently accept different data or search for a favorable mirror.

### 5.3 Exact group and split construction

Preserve original row order and use zero-based row IDs after excluding headers. No score-dependent exclusions are permitted.

1. Read Airfoil decimal text as strings; convert each key token with Fraction(Decimal(token)). The key uses columns 1–4, excluding column 0 (frequency).
2. Read Concrete XLS using the recorded xlrd version. The key is the first seven feature values, excluding age. Convert each XLS float through its shortest Python string, then Decimal, then Fraction. XLS has binary numeric cells; do not claim original decimal strings were recovered.
3. For Superconductivity, read the 86 elemental columns H through Rn from unique_m.csv as decimal strings. Convert to exact Fractions and divide each by the row’s positive total. Exclude critical_temp and material. Assert target alignment with train.csv solely for integrity, then remove targets from group construction.
4. Serialize each ordered vector as JSON of \([numerator,denominator]\) pairs with json.dumps(pairs, separators=(",", ":")); SHA256 of its UTF-8 bytes is the group ID.
5. Sort the distinct group IDs lexicographically.
6. Seed NumPy PCG64 with the unsigned big-endian integer represented by the first eight SHA256 bytes of ASCII “split|20261029|DATASET_ID”.
7. Permute the sorted group IDs. Assign the first \(\lfloor0.6G\rfloor\) to training, the next \(\lfloor0.2G\rfloor\) to validation, the remainder to test.
8. Persist all group and row assignments plus hashes. Assert counts match the table for the recorded files. The derived integer seeds are 7273367300170510378 for Concrete and 10056838054366971847 for Superconductivity.

This is the exact procedure used for the measured counts. Do not redraw splits because model performance looks inconvenient. Exact composition grouping is not a guarantee that chemically related materials or all laboratory dependencies are separated.

Retain repeated observations inside a group. The primary metric gives each group equal weight; dropping duplicates would change the evaluation population and requires an amendment. Source data may contain repeated compositions and measurements; random-row scores published elsewhere are not comparable.

### 5.4 Preprocessing and information access

- On the training rows only, compute feature means and population standard deviations. Replace zero scale with one and record the affected columns. Convert neural inputs to float32 after those statistics are computed in float64.
- Compute training target mean and population standard deviation, standardize for training and invert for reporting.
- The primary neural track uses this affine transformation only. No clipping, quantile transform or outlier removal is hidden in preprocessing.
- Tree baselines may use their ordinary raw numeric inputs. Practical neural baselines may use their documented transforms, fitted only on the same training rows.
- Train minibatches sample rows uniformly. Loss is ordinary mean squared error; group-balanced evaluation is explicit and shared across methods. A group-weighted training-loss variant is a separate sensitivity treatment offered to all methods.
- Group metadata and test features may be inspected for integrity. Test target values may be read only by integrity assertions and, after freeze, the protected evaluator. No target distribution plots, loss calculations, early stopping or tuning on protected test data.
- Use all training rows for the core comparison. A small common subset is allowed for operator/harness tests only, not silently as the final benchmark.
- Keep training, tuning, diagnostic, bootstrap and confirmation RNG streams separate.

There is no pretrained asset in the core activation comparison. An optional foundation-model baseline has reusable pretraining, different information history and separate license conditions; report those facts rather than treating all project expenditures as equivalent.

## 6. Fair evaluation protocol

### 6.1 Three distinct comparison tracks

**Track A — activation competitiveness.** Use the same backbone, data, loss, input normalization, training information and search-allocation rule. Tune each activation fairly. Select the baseline champion on validation only. This answers whether the candidate earns its place as a replacement component.

**Track B — attribution.** Hold the entire training recipe fixed across the factorial and nested controls in Section 4.6. Choose that common recipe using the fixed-activation baselines on development/validation data, before inspecting candidate comparative results. Pair initial dense weights, data order and RNG namespaces where dimensions agree. This isolates interventions rather than comparing independently optimized pipelines.

**Track C — practical prediction.** Allow each competing pipeline its standard architecture, preprocessing and implementation, under a measured equal selection-time budget. Compare the full error–cost–reliability frontier, including wider fixed-activation MLPs, CatBoost, TabM and eligible foundation models. This answers an application question; it does not identify activation causality.

Never substitute Track B’s deliberately stressed architecture for Track C’s strongest practical baseline. A fair independent tuning result is not the same causal estimand as a fixed-recipe intervention.

### 6.2 Direct local baseline set and reproduction contract

The following are required unless a documented compatibility/access gate prevents faithful local execution. Exclusion narrows the claim; it does not silently promote the remaining methods to “state of the art.”

| Family | Exact starting definition/configuration | Reproduction and fairness requirement |
|---|---|---|
| Fixed activations | ReLU; exact GELU; SiLU; tanh as bounded smooth diagnostic | Same architecture; own LR/weight-decay tuning; no weak single untuned ReLU comparator |
| PReLU | Per-layer scalar negative slope, initialized 0.25 | No shape-parameter decay; allow the same shape-LR multiplier choices |
| Affine-calibrated GELU | \(a\,g(x)+b\), initialized \(a=1,b=0\) | Tests whether generic scale/offset adaptation explains the result |
| Naive monomial | \(a_0+a_1x+a_2x^2+a_3x^3\), Horner evaluation | Use a label-free fitted initialization as well as near-identity diagnostic; do not rely on an intentionally explosive initializer |
| User’s normalized Hermite | \((x+\alpha h_2+\beta h_3)/\sqrt{1+\alpha^2+\beta^2}\) | Diagnostic of the supplied idea; distinct from the published torchortho convention |
| Published/current Hermite | Degrees 2, 3 and 5; exact conventions below | Mandatory closest polynomial comparison; evaluate native and corrected mathematical paths transparently |
| BerLU | Formula below; learn \(\alpha\), native radius 0.01 and a validation-tuned radius | Its native unconstrained alpha and a matched constrained variant must be labeled separately |
| Learned quadratic | Candidate with \(s_1=(s_0+s_2)/2\) | Same free offset and two tail slopes; indispensable cubic-degree control |
| Learned spline | Signed slope-constrained linear spline with linear extrapolation; coefficient/range/TV choices below | Use current DeepSplines source and explicit tensor-reference parity; transplant is an adaptation, not full JMLR network reproduction |
| Safe rational | Both source versions A and B, degree (5,4), per-layer sharing | Version A is the package default; include each version's own strong initializer; 6 numerator + 4 denominator parameters |
| Same cubic factorial | All four tails/projection cells | Same coordinates, starting center, offset and common optimization recipe |
| Ridge/quadratic ridge | Standardized numeric features; explicit degree-two features and regularized linear solve | Important simple interaction baseline; feature expansion memory calibrated |
| RBF kernel ridge | Tune kernel bandwidth and ridge strength | Exact kernel for Concrete/development; larger task only if memory/time feasible, otherwise explicit Nyström variant |
| CatBoost | Official CPU regressor, RMSE objective, validation stopping | Standard features and own tuning preserved; Apple arm64 wheel is documented [R30] |
| TabM | Official TabM, \(k=32\), per-member training loss and mean prediction at evaluation | Preserve author semantics, defaults and useful preprocessing; \(k=16\) is a labeled smaller variant, not silent replacement [R29] |
| Wider/ensembled conventional MLP | Widths 64, 128, 256; independent small ensemble if comparable budget permits | Establish whether ordinary extra capacity/compute gives the same benefit |
| Eligible TabPFN | Exact available local checkpoint, official package and license | Optional contemporary direct competitor after access, correctness and resource checks [R31] [R32] |

The direct comparison is not limited to matching the candidate’s four activation coefficients. A stronger spline/rational can use more if it wins on actual cost. Report both equal-backbone and realistic cost tradeoffs.

### 6.3 Exact close-baseline semantics

#### Hermite

The current paper uses \(F(x)=\sum_{k=0}^n a_k\mathrm{He}_k(x)/k!\), not orthonormal coefficients with the same numeric values. Its second moments are \(Q=\sum a_k^2/k!\) and \(G=\sum_{k=1}^n a_k^2/(k-1)!\). Convert coefficients explicitly when changing bases. [R04]

During preparation, the official source was retrieved at commit:

~~~text
torchortho: aa0822da6b8aa5a4294e89485fde75dfaab8478d
torchortho/hermite_activation.py SHA256:
00ab8cde719909dc57dec8604917e4f7933b4177d14cb0dc02afebdf38f65217
~~~

The native default sets unscaled \(a_0=\sqrt{1-1/n!}\), \(a_{k\ge1}=1\), then divides all by \(\sqrt e\). Use this initializer as a baseline. An exact finite-degree normalization divides instead by \(\sqrt{\sum_{j=0}^{n-1}1/j!}\); label it as a separate mathematical modification.

Implement a low-degree recurrence reference:

\[
b_0=1,\quad b_1=x,\quad
b_{k+1}=\frac{x b_k-b_{k-1}}{k+1},\qquad b_k=\mathrm{He}_k/k!.
\]

Accumulate \(\sum a_k b_k\) without constructing an unnecessary high-rank tensor. Preserve native initialization, sharing and coefficients.

**Source-inspection findings, not executed PyTorch results:** the inspected native basis uses absolute-value/sign powers, which require a specific gradient check at zero; its optional function-fitting initializer multiplies lower-order bases by \(\sqrt{k}\), whereas the derivative of \(b_k\) is \(b_{k-1}\). Therefore, default native initialization with that optional fitter disabled is the primary source reproduction. Do not present a gain over a discovered source defect as an activation advance. Archive a discrepancy report; test native and mathematical paths on CPU before choosing the correct reference for scientific comparison. [R20] [R46]

The separate deterministic GELU fit uses the same 1,025-point grid and Gaussian row weights as Section 4.5. Stack value columns \(B_{ik}=b_k(x_i)\) and derivative columns \(D_{i0}=0,D_{ik}=b_{k-1}(x_i)\); stack targets \(g(x_i)\) and \(g'(x_i)=\Phi(x_i)+x_i\varphi(x_i)\). Multiply both blocks and both targets by the same row weights. Give value and derivative residuals equal weight, solve unconstrained CPU float64 least squares with rcond \(10^{-12}\), apply no subsequent gain rescaling, and save rank, singular values and residuals. This is an explicitly defined proposed initializer, not the upstream cached fitter.

#### BerLU

For radius \(\epsilon>0\),

\[
f(x)=
\begin{cases}
\alpha x,&x<-\epsilon,\\
(1-\alpha)x^2/(4\epsilon)+(1+\alpha)x/2+(1-\alpha)\epsilon/4,&|x|\le\epsilon,\\
x,&x>\epsilon .
\end{cases}
\]

The bound is \(\max(1,|\alpha|)\); learning alpha without a constraint does not preserve a unit cap. Native initialization is \(\alpha=0.01,\epsilon=0.01\). Include native radius and the matched candidate radius in validation. Clamp before central polynomial evaluation. [R06]

#### Spline

For an explicitly specified reference, use knots \(t_j=-R+2Rj/(K-1)\), values \(v_j\), linear interpolation between knots and extension of the first/last segment outside. Segment slopes are \(d_j=(v_{j+1}-v_j)/(t_{j+1}-t_j)\). An equivalent direct-slope representation integrates those slopes with a free offset. Box projection \(|d_j|\le S\) gives the same scalar slope cap; total variation of the derivative is \(\sum_j|d_{j+1}-d_j|\).

Use the official module's slope transform and TV regularization as the source-based baseline, and the direct-slope reference for mathematical checks. They need not have identical optimizer trajectories: a different projection metric or parameterization must be labeled. The pinned repository's demo anchor is K=51, R=1, identity initialization, no slope constraints, TV coefficient zero, and apply_scaling=False. Include it as a source-behavior anchor, not a claim that its demo settings are optimal. The constructor requires an explicit initializer. The tuned constrained baseline uses K in {5,9,17,51}, R in {1.5,3,6}, S in {1,1.5,3}, and TV coefficient in \(0,10^{-5},10^{-3}\) in standardized-loss units. Include identity, source-native ReLU and GELU-at-knots initial values; label GELU values as an adapter initializer and pass all of them through the source slope transform. Extend ranges prospectively if best validation settings sit on a boundary. [R12] [R21] [R47]

The inspected standalone source is pinned at 654a8bd24b77433618169fc1e8607df123946302. It implements an in-forward transform: clamp raw segment slopes, integrate them into \(q_0=0,q_i=h\sum_{j<i}d_j\), then set effective coefficients \(\tilde v_i=q_i+\operatorname{mean}(v-q)\). It is neither a Euclidean coefficient projection nor the candidate's post-update operation. Its TV2 is \(\sum|\Delta^2\tilde v|/h\), and interpolation fractions use the original input, preserving linear extrapolation. [R47]

Use source-supported apply_scaling=False for the core. The optional upstream scale computes \(f(sx)/s\) without a zero safeguard; a guarded scale would be a separate variant. For layer sharing, flatten activation inputs to (-1,1), use one scalar function, then restore shape. Implement efficient device-aware gather interpolation with the source's right-segment derivative at knots. Compare effective coefficients, outputs, input/raw-coefficient gradients and TV2 against the pinned CPU source. Cache immutable numeric endpoints after parity to avoid per-forward host synchronization. The repository has no installable-package manifest; load its small reference module directly rather than inventing a pip command.

#### Rational

Keep both inspected source variants eligible:

\[
r_A(x)=\frac{\sum_{i=0}^{5}a_i x^i}
{1+\sum_{j=1}^{4}|b_jx^j|},\qquad
r_B(x)=\frac{\sum_{i=0}^{5}a_i x^i}
{1+\left|\sum_{j=1}^{4}b_jx^j\right|}.
\]

The inspected package at commit 2496d918d924b117ef06ec35bdca7f207720c62c defaults to A. Its rational/rationals_config.json contains separate A/B degree-(5,4) GELU, Leaky-ReLU and “swish” initializers. Use the exact selected vectors, record their hashes and select only on validation. The fitting interval in that file is not an inference clamp. Denominator at least one prevents poles, not arbitrary slopes or floating-point overflow. [R23] [R48]

Use an initial successive-powers reference matching the CPU source and then a parity-tested Horner implementation. A's denominator is \(1+|x|(|b_1|+|x|(|b_2|+|x|(|b_3|+|x||b_4|)))\); B uses the absolute value of the summed denominator polynomial. Keep the source absolute-value subgradient at zero. Some upstream constants are created on CPU; the MPS adapter must inherit the input device/dtype. Compare output, input gradients and every coefficient gradient separately for A and B. Do not silently insert an extra denominator epsilon.

### 6.4 Contemporary practical baselines and exclusions

TabM is a strong modern MLP-based ensemble competitor; training each member’s error is different from training only the ensemble mean. Include official defaults and its author-supported architecture choices. Its larger tuning study is context, not a reproduced result from this smaller grouped experiment. [R29]

At the research date, the official TabPFN repository identifies generation 3.5; the inspected changelog includes package 9.2.0 dated 12 October 2026. Code and weights have different terms: the current weight license is noncommercial/nonproduction. Treat the checkpoint as a separately gated local comparator, record its exact generation/file/hash, and disclose reused pretraining. The changelog also documents MPS/cached-inference correctness fixes, so CPU–MPS agreement is indispensable. Do not call 2.5 the latest version or silently change generation during confirmation. [R31] [R32] [R44]

If permitted weights cannot be obtained, a required kernel cannot run faithfully, or realistic local memory/time prevents a full-task run, keep that reference outside the controlled table with the reason. Do not use a paid hosted API or subset only that model’s training set and call it an equal-data comparison. A common smaller-data secondary experiment is allowed if labeled and offered to all methods.

Nonlocal contextual results include full ConvNeXt/ImageNet, GPT-2/OpenWebText, full DeepBern certification and broad TabArena runs. Their role is to bound ambition and identify competitors. Their published scores must not appear as if generated on this split. [R04] [R05] [R43]

SAAF, older constructive networks and Moment-SOS need not all be ported: the local comparison covers their closest relevant function/constraint principles through modern spline, polynomial and rational methods. Do not claim direct superiority to an omitted implementation.

### 6.5 Architecture and common training definitions

The following defaults are deliberately simple enough to isolate an activation, while including standard stabilizers.

**Practical common residual backbone:**

\[
h_0=W_{\rm in}x+b_{\rm in},\quad
z_l=\operatorname{LN}(W_{l,1}h_l+b_{l,1}),\quad
h_{l+1}=h_l+L^{-1/2}(W_{l,2}\phi_l(z_l)+b_{l,2}),
\quad \hat y=W_{\rm out}h_L+b_{\rm out}.
\]

Use width 128, four residual blocks, LayerNorm epsilon \(10^{-5}\), no learnable LayerNorm affine parameters, no dropout in the matched track, and biases in affine maps. The activation is directly followed by an affine map; its offset can therefore be absorbed into that map’s bias. This reduces an offset-placement confound.

Initialize each dense matrix from a zero-mean Gaussian with standard deviation inverse square root of fan-in, biases zero, using CPU-generated stored tensors. Do not zero the second block matrix in only one method. Published activation-specific gains are eligible tuning alternatives, always recorded.

The matched **plain diagnostic backbone** has repeated Linear → optional LayerNorm → activation, then a linear head. Study activation depths 2, 8 and 16 at width 128. Begin with depth 2/8; add 16 if the diagnostic traces or claimed mechanism require it. A no-normalization condition probes exposure but does not define practical superiority.

For the primary residual backbone, affine-free LayerNorm itself bounds normalized coordinates at finite width (up to approximately \(\sqrt{h-1}\), apart from epsilon effects). Thus there is no premise of literally unbounded normalized preactivation in this configuration. Tails may still matter inside that finite range. Report actual occupancy rather than manufacturing extreme-input claims.

**Training defaults to calibrate before freeze:**

- AdamW, beta parameters (0.9, 0.999), epsilon \(10^{-8}\), with an unfused/non-foreach path initially for compatibility.
- Loss: mean squared standardized-target error.
- Candidate LR range: \(10^{-4}\) to \(3\times10^{-3}\), log-spaced exploration, plus exact author defaults where different.
- Candidate radius/cap search starts with \(R\in\{1.5,3,6\}\), \(S\in\{1,1.5,3\}\); include \(R=S=1.5\) explicitly. Offer the same cap choices to the matched spline/quadratic variants. An unresolved boundary optimum triggers prospective range extension before freeze.
- Weight decay range: \(0,10^{-5},10^{-3},10^{-2}\) on dense weights only.
- Common clip choices: no clipping or global norm 1.0; both available to every neural competitor. Fix one setting across Track B.
- Batch begins at 128 for Concrete and 256 for Superconductivity. Select a common effective batch before confirmation using Section 7 calibration.
- Sample a fresh stored row permutation each epoch, without dropping the final incomplete batch. No batch-dependent normalization in the matched models.
- Validate once per epoch using group-balanced validation RMSE; retain the best finite checkpoint. Start pilot maximum at 200 epochs and patience 30 validation checks. These are pilot settings, **not an arbitrary final runtime cap**.
- If validation improvement continues near the cap, double the horizon for all matched methods and rerun affected tuning candidates. Stop increasing after validation curves have flattened relative to paired validation uncertainty across two horizon extensions, or document a resource-limited comparison. Freeze horizon, patience and scheduler before confirmation.
- Begin with constant LR to avoid a hidden schedule advantage. A shared cosine schedule is an explicit validation candidate and is also available to strong baselines.

All target-dependent choices use training/validation only. The highest-performing test method never determines the baseline champion.

**Random-stream contract:** use master seed 20261029 and derive named stream seeds from the first eight big-endian SHA256 bytes of a canonical UTF-8 key containing phase, task, seed-block number and stream name. Freeze the exact keys and resulting integers. Tuning, pilot and confirmation have different phase namespaces. Within a Track A/B block, reuse identical CPU-generated dense tensors for matching parameter names/shapes and identical row permutations; activation initialization must not advance those streams. Give method-specific stochastic operations their own named streams. Distinct architectures can only share the applicable data/order streams, not nonexistent matching tensors. Record any package-required 32-bit seed conversion explicitly, for example seed64 modulo (2^31-1), and store all concrete seeds in the run manifest. Analysis/bootstrap/timing streams are separate from training.

### 6.6 Tuning allocation and stopping rules

A baseline is not “well tuned” merely because a search function ran.

1. Run exact source/default configurations first and verify ordinary learning on development data.
2. Use a pre-generated, seeded Sobol/random candidate list covering the shared ranges and method-specific choices. Include defaults explicitly; do not rely on random sampling to hit them.
3. Begin with a batch of eight new candidates per method/task after defaults. This is a sequential allocation unit, not a fixed total limit.
4. Refit the two best validation candidates on two additional tuning seeds. Select using mean group-balanced validation RMSE.
5. Extend searches in equally allocated batches while any plausible leading method has an unresolved boundary optimum or gains exceeding validation uncertainty on the additional tuning seeds. Share generic improvements with all relevant methods.
6. After two successive equal-allocation batches give no resolvable validation improvement and no unresolved range boundary, record local search saturation. This is an engineering adequacy criterion, not a statistical theorem.
7. A physical time/resource limitation may stop search earlier; report the achieved candidate counts, boundary warnings and search curves, then narrow claims accordingly.

For Track A, use equal attempted candidate/refit opportunities and report unequal actual compute. Numerical failures consume their opportunity; compatible interruption/restart is the same opportunity with all consumed time retained.

For Track C, derive a common cumulative selection budget per task from measured defaults and the first complete tuning batch. The budget must be at least the largest already-consumed selection cost across eligible pipelines plus the forecasted final-refit reserve. Charge every selection-informing default, preprocessing/calibration run, initial batch, failed attempt and finalist refit against this budget. Report shared engineering/environment setup separately. Prefer an independent, pre-generated Track C search order. If Track A configurations, scores or caches inform Track C selection, charge their original selection cost too; there is no free reuse of a more expensive search. A pipeline that cannot meet the common budget is a recorded resource-limited exclusion, not a zero-cost default.

Reserve the p90 forecast of the required finalist refits before launching additional candidates. A trial may start only if its pilot p90 full-trial forecast fits the remaining unreserved allocation. A running trial finishes, its actual overrun is retained and reported, and no new trial starts after the budget. If a realized overrun prevents the planned refits, retain the best fully evaluated candidate under the frozen rule and report that incomplete allocation; do not silently compare it as equally tuned. The same rules apply to all pipelines. Report both tracks; a slower model completing fewer candidates is part of the practical selection-cost question, not a causal activation comparison.

Never spend extra manual tuning on the proposal after seeing test scores. Code fixes, schedule changes, feature transformations and implementation optimizations that help all methods must be offered fairly and logged.

### 6.7 Primary metric, evaluation population and failures

For task \(d\), held-out group \(g\), method \(m\), and training replicate \(s\), define

\[
E_{d,m,s}=
\sqrt{\frac1{G_d}\sum_{g=1}^{G_d}
\left(\frac1{n_g}\sum_{i\in g}
(y_i-\hat y_{m,s,i})^2\right)} .
\]

This is **group-balanced RMSE** in MPa or kelvin. It targets a uniformly weighted held-out mixture/composition group, not a uniformly weighted row.

Let \(B_d\) be the activation champion selected using validation only, excluding the proposal but including the mandatory learned quadratic and all eligible strong activation families. The primary seed-block contrast is

\[
\Delta_s=\frac12\sum_{d\in\{\mathrm{Concrete,Superconductivity}\}}
\log(E_{d,C,s}/E_{d,B_d,s}).
\]

Negative is favorable. Report \(\bar\Delta\), \(100(\exp(\bar\Delta)-1)\) as a geometric-mean relative RMSE change, each original-unit RMSE, each absolute difference, and each task’s sign. The two tasks receive equal weight; they are not treated as an independent sample of all tasks.

Use a numerical floor of \(10^{-12}\) times training target standard deviation only if an error is exactly zero; report every floor event. Synthetic noiseless tasks do not enter this log aggregate.

**Failure-aware prediction procedure for every method:**

- Before training, retain the training-mean predictor.
- Retain each finite validation-best model checkpoint.
- A numerical failure stops the run and records its cause and consumed resources.
- Its evaluation uses the last finite validation-best checkpoint, or the training-mean predictor if none exists.
- Do not retry with a new learning rate, delete the seed or average only successes.
- Report divergence, projection saturation, fallback rate, successful-run scores, checkpoint age and failure time separately. The fallback does not make a failed training run a success.
- Hardware/interruption failures are classified separately. A compatible interruption resumes; a changed backend/configuration requires a new recorded run under the prespecified recovery rule.

This evaluates a concrete deployable training procedure rather than imposing an arbitrary infinite-error penalty. A positive score primarily due to rescue/fallback behavior must be described as a reliability effect.

**Inference failures:** a finite checkpoint can still produce nonfinite predictions on new inputs. Before reading target labels, an identical wrapper for every method replaces each nonfinite prediction with the training target mean. Record affected rows/groups and the raw failure separately. Preserve all finite predictions, however large; do not winsorize favorable scores. Compute errors in CPU float64 with a scaled sum-of-squares calculation if needed, and compute the primary log ratio as log(error_candidate) minus log(error_baseline). This wrapper is part of the evaluated procedure, not evidence of stable underlying inference.

**Secondary measures:** row-weighted RMSE and MAE; per-group error distributions and worst-decile group error; training/selection seconds; convergence curves by optimizer steps and elapsed seconds; batch-1 latency and common-batch throughput; peak memory counters; parameter/checkpoint bytes; failure rates; activation and Jacobian diagnostics.

### 6.8 Confirmation, power calibration and uncertainty

Freeze methods, exact implementations, source revisions, split hashes, preprocessing, tuning results, champion identity per task, architectures, metrics, horizons, seeds, failure handling and analysis code before protected evaluation. A plain local protocol JSON and a hash are sufficient; cryptographic signing or an external service is unnecessary.

**Pilot and replication procedure:**

1. On development/validation data, run five new paired pilot seed blocks after configurations are chosen. These are distinct from tuning and future confirmation seeds.
2. Estimate the paired standard deviation of the prospective log-RMSE contrast and its uncertainty. Estimate held-out-group uncertainty with a validation-group bootstrap.
3. Construct a precision–cost table for prospective replication counts, including at least 8, 12, 20 and 32 and additional counts if indicated.
4. As a default information-balance rule, choose the smallest \(N\ge8\) for which the conservative estimated optimizer standard error \(s_U/\sqrt N\) is no larger than the validation-group standard error. Under approximately normal pilot contrasts, use the one-sided 90% upper SD estimate \(s_U=s\sqrt{(n_p-1)/\chi^2_{0.10,n_p-1}}\). Record that planning assumption. With obvious skew/failures, add five prospectively specified pilot blocks and inspect the full distribution before freezing the precision-cost plan.
5. If that number is locally affordable according to measured throughput, freeze it. If not, remove optional exploratory breadth first; then choose and record a physically feasible fixed \(N\), its detectable effect and the resulting limitation. Do not infer “no effect” from inadequate precision.
6. If group uncertainty is near zero or the pilot variance estimate is unstable, do not chase an infinite replication target: expand the validation pilot prospectively or use a justified application-relevant precision target recorded before test access. The coding agent must report the unresolved precision decision rather than invent a meaningful percentage.

If substantial skew/failure mixtures persist after ten pilot blocks, freeze a feasible N from the measured budget, retain empirical pilot distributions and label the normal-theory power forecast provisional. Do not keep adding pilot blocks until a favorable shape or effect appears.

A rough two-sided 5%/80%-power detectable log effect is

\[
(1.96+0.842)s_{\rm pilot}/\sqrt N.
\]

Use finite-\(N\) t/noncentral-t calculations for the actual table and acknowledge uncertainty in the pilot variance. This is sensitivity planning, **not a minimum improvement required for scientific success**.

**Main interval:** paired t interval for the mean \(\Delta_s\), with every seed-block value shown. This describes optimizer randomness conditional on these data, splits, tuned configurations and implementation. Seeds are independent training runs, not independent datasets.

**Separate test-group interval:** resample whole test groups within each task with replacement, retaining multiplicities and using identical draws for all methods and seeds; recompute each seed’s group-balanced RMSE, its log ratio, then the task/seed mean. Use the 2.5th and 97.5th percentiles of at least 10,000 bootstrap draws for a 95% percentile interval; assess Monte Carlo stability of its endpoints without rerunning training. Do not average predictions unless explicitly evaluating an ensemble. Do not change the estimand to log of seed-averaged squared error.

Report seed-only and group-only intervals separately. Neither covers training-set sampling, hyperparameter-search randomness, unobserved laboratory dependence or a population of tasks. A positive finite-test claim may rest on the first; a claim about new groups also needs compatible group uncertainty and its assumptions. Do not combine the two intervals mechanically or count seed × row pairs as independent.

With heavily skewed failure outcomes, show a seed-block bootstrap and sign-based descriptive sensitivity; a small sample may remain inconclusive. A permutation/sign-flip test requires appropriate exchangeability/symmetry assumptions and is not automatic merely because seeds are paired.

**Multiplicity and attribution:** H1 versus the frozen activation champion is the primary test at two-sided alpha 0.05. Only after it passes in the favorable direction may four prespecified attribution claims be promoted through a Holm-adjusted family at alpha 0.05: full versus tailed/free (cap effect conditional on tails), full versus global/boxed (tail effect conditional on cap), learned versus frozen, and cubic versus nested quadratic. Freeze this gatekeeping order. If H1 does not pass, report these contrasts as diagnostic evidence; an affirmative narrower claim needs a prospectively separate confirmation.

For factorial log-error cells \(Y_{tp}\), with t=tails and p=projection, also report interaction \(Y_{11}-Y_{10}-Y_{01}+Y_{00}\) and averaged main effects. These are descriptive unless included in the frozen family. Full versus free/global changes two things and cannot alone identify either.

Track C is descriptive by default. If its practical-pipeline comparison is prospectively promoted to a second primary claim, allocate alpha 0.025 to the activation branch and alpha 0.025 to the practical primary. Within the activation branch, test H1 at 0.025 and, only after favorable rejection, test its four attribution claims with Holm at 0.025. This fixed Bonferroni split controls error across the two branches; do not run a second unrestricted 0.05 family after testing two primaries. Freeze this choice before test access and use the corresponding confidence levels and power calculations.

**Repeated inspection:** do not change \(N\) after seeing test significance. Operational monitoring can detect crashes without displaying interim test metrics. A post-test methodological change requires an amendment and fresh protected confirmation data/tasks for a new affirmative claim; merely adding seeds does not erase test-driven selection. Selection bias is a material risk in small method comparisons. [R40] [R41]

### 6.9 Success, practical value and headroom

- A small effect with an interval excluding zero can be scientifically real.
- A nonsignificant difference is not equivalence.
- An equivalence/noninferiority margin is valid only if justified by an application decision before test access. None is supplied here.
- A narrow interval near zero plus material extra cost suggests limited practical value, without inventing a universal percentage cutoff.
- A gain on one task and loss on another is a heterogeneous result even when the aggregate improves.
- A primary-error improvement with worse instability, latency or memory is a tradeoff. Report it as such.
- Practical advantage requires an observed error–cost frontier competitive with the strongest eligible local pipelines. Parameter savings alone do not suffice.
- If strong baselines leave no detectable error headroom, test whether a cheaper candidate reaches comparable accuracy using a prospectively justified margin, or report that there is no demonstrated practical advantage.

## 7. M3 Max feasibility and hardware-adaptive execution

### 7.1 What is documented and what must be measured

Apple lists multiple M3 Max configurations, including 36 GB rather than only the maximum memory option. Discover the actual machine. [R33]

Official PyTorch documentation describes MPS acceleration but does not establish support or correctness of every required graph. Metal/MPS cannot allocate float64 tensors; a CPU fallback flag does not remove that limitation. Use CPU double for numerical oracles and float32 for MPS training. [R34]

Versioned documentation is a moving target. At inspection, Apple’s landing page named PyTorch 2.11.0, while “stable” PyTorch documentation redirected differently and development documentation had different OS requirements. **Do not infer the installed/released wheel or minimum OS from one mutable page.** Select an actual official released arm64 wheel compatible with the discovered OS/Python, validate the required graph, then freeze the environment. [R34] [R45]

### 7.2 Discovery commands: existing commands, not project scripts

Run these real macOS commands separately; retain stdout/stderr/exit status:

~~~bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu
system_profiler SPHardwareDataType SPDisplaysDataType
df -h .
vm_stat
pmset -g batt
xcode-select -p
python3 -V
~~~

Record CPU/GPU/core configuration, unified-memory bytes, OS build, native arm64 execution, available disk, power status and interpreter. Redact serial numbers/hardware UUIDs from public reports. If an interpreter runs under Rosetta, obtain a compatible native environment before drawing hardware conclusions.

Real PyTorch probe APIs:

~~~python
import platform, sys, torch
print(platform.machine(), sys.version, torch.__version__)
print(torch.backends.mps.is_built(), torch.backends.mps.is_available())
print(torch.__config__.show())
if torch.backends.mps.is_available():
    print(torch.mps.recommended_max_memory())
    print(torch.mps.current_allocated_memory())
    print(torch.mps.driver_allocated_memory())
~~~

### 7.3 Software and dependency strategy

Use a fresh native Python environment with PyTorch, NumPy, SciPy, scikit-learn, xlrd, a YAML/JSON parser, a test runner and plotting tools. Add CatBoost and the pinned TabM package. No torchvision, torchaudio, MLX, CUDA, custom Metal kernel or remote training service is required for the core.

Prefer PyTorch because the mechanisms and closest research code already use its tensor/autograd model. A full MLX port would introduce a second implementation variable; consider it only as a later, separately validated engineering experiment.

Before installing a package, confirm released wheel tags, Python requirements, macOS deployment target and architecture. Use official distributions. Preserve exact package versions, wheel hashes, source SHAs, Python/OS and resolver output. Do not automatically upgrade after freeze. An isolated environment for a legacy source-reproduction check is allowed; it is not the timed main environment.

Code licenses verified from repositories include GPL-3.0 for torchortho, MIT for the current standalone spline/rational modules, BSD-3-Clause for DeepBern and Apache 2.0 for TabM. Archive the pinned licenses and notices; do not assume a paper license also covers its code. Formula-based independent adapters should be identified as such. Weights may have separate conditions from source code. [R20] [R21] [R22] [R23] [R29] [R32]

### 7.4 Critical graph eligibility gate

Use a fresh subprocess with implicit MPS CPU fallback disabled, fast math disabled, no compilation and no mixed precision. Test exactly:

- dense matrix products, chosen LayerNorm and epsilon;
- polynomial/Horner operations, clamp/where and parameter broadcasting;
- spline indexing/gather and projection;
- rational division/absolute-value path;
- loss, gradient clipping if enabled, AdamW step and shape projection;
- serialization, reload and continued updates.

The initial optimizer path uses unfused and non-foreach execution where the installed API supports those options. Verify that constructor, rather than assuming an optimized path exists. [R38]

Generate a fixed state and input on CPU, run the identical graph in CPU float32 and MPS float32, and compare outputs, losses, every parameter-gradient family and one optimizer update. CPU float64 finite differences/gradcheck provide the oracle.

Starting tolerances are forward absolute \(10^{-6}\), relative \(10^{-5}\), and gradient normalized L2 discrepancy \(10^{-4}\) for non-negligible gradients. These are diagnosis thresholds, not permission to ignore a systematic error. Explain conditioning and use the double oracle before relaxing them.

Test zero, ordinary random inputs, \(\pm R\), one-sided neighbors, \(\pm10R\), large finite inputs, and multiple coefficient settings. Verify both coefficient and input gradients. Account explicitly for intentional nondifferentiable spline/rational points. Check float32 behavior near the slope cap and for very small LayerNorm variance.

**Failure routes:**

- CPU correct/MPS incorrect: first repair an implementation issue if identified; otherwise use an entirely CPU comparison.
- No faithful MPS implementation for one essential method: compare all relevant methods on CPU for that claim, or give separate device-specific fronts. Hidden mixed-device execution is disallowed for timing.
- No faithful local implementation at all: block the claim requiring that comparator and record the limitation. Do not silently omit it.
- A changed kernel, precision or approximation is a changed treatment and needs parity checks plus an amendment.

### 7.5 Memory model

For a plain MLP with input dimension \(d\), output \(o\), hidden width \(h\), and \(L\) hidden layers,

\[
P=(d+1)h+(L-1)(h+1)h+(h+1)o.
\]

Add affine-normalization parameters if used and the actual activation coefficients. For the selected residual backbone,

\[
P_{\rm residual}=(d+1)h+L[2(h+1)h+p_\phi]+(h+1)o,
\]

where \(p_\phi=4\) for a shared cubic and LayerNorm here has no learned affine parameters.

Float32 weights, gradients and two Adam moments cost approximately \(16P\) bytes before additional buffers. A model-plus-optimizer checkpoint is roughly \(12P\) bytes plus metadata and extra buffers. Saved activation tensors begin at approximately \(4cBLh\) bytes, with \(c\) dependent on the actual graph; use \(c=6\)–24 only as an initial planning range and measure it.

At \(d=81,h=128,L=4,o=1\), the residual formula gives 142,737 parameters, approximately 2.28 MB for these optimizer/model tensors. At \(d=81,h=256,L=16\), it gives 2,126,657 parameters: 34,026,512 bytes for \(16P\) and 25,519,884 bytes for \(12P\). These are arithmetic estimates, not allocator peaks.

For \(B=512,h=256,L=16\), \(4BLh\) is 8 MiB; the illustrative \(c\) range gives 48–192 MiB, before any additional residual/normalization graph factors, driver allocation, workspace and Python/runtime state. The complete larger residual graph may have a greater effective \(c\); calibrate rather than treating 192 MiB as a cap.

A float32 copy of Superconductivity inputs/target is about 7 MB; raw CSV, exact group keys, float64 preprocessing arrays and duplicate copies add memory. Small-task exact kernel ridge uses \(O(N^2)\) memory; the larger task’s kernel can become a material cost and must be gated or explicitly approximated.

Report GPU tensor allocations, driver allocations, process RSS, memory pressure and swap **separately**. Unified-memory counters overlap; summing them does not yield physical peak use. The documented MPS memory/synchronization APIs support observation but require local version checks. [R35]

### 7.6 Adaptive configuration rules

Select a process budget before calibration: initially the smaller of half discovered physical memory and 80% of the documented recommended MPS working set, reduced if other applications create pressure. This is an operational headroom policy, not a scientific acceptance threshold. Keep allocator protections enabled; never set a zero high-watermark merely to force a run through. [R36]

If MPS or that telemetry API is unavailable, use half discovered physical RAM as the initial CPU-process ceiling, reduced for observed application/OS pressure. Do not call an unavailable MPS API to decide the CPU fallback budget.

Calibrate batches 64, 128, 256 and 512 with the largest scheduled backbone and most memory-demanding eligible activation. Include a full update, validation and checkpoint in the check. Choose a shared effective batch that fits. Run one accelerator training process at a time.

Preferred faithful memory fallbacks: keep data on CPU; stream batches; lower evaluation batch; detach and summarize diagnostics; avoid retaining intermediate graphs; use gradient accumulation only with the same effective batch, correctly weighted final microbatch, identical optimizer-update schedule and no batch-dependent semantics. Recheck parity for the accumulated path.

Reducing width, depth or training data changes the scientific treatment. Do it symmetrically, record the amendment and re-evaluate whether the mechanism is still expressed. Do not shrink only the strongest baseline.

### 7.7 Runtime, disk and calibration

No end-to-end M3 runtime is measured yet. Forecast it from actual local measurements:

\[
T_{\rm campaign}=
\sum_{r\in\mathrm{all\ attempts}}(\mathrm{steps}_r\,t_{\rm step,r}
+\mathrm{validation}_r+\mathrm{checkpoint}_r+\mathrm{setup}_r)
+T_{\rm nonrun\ preparation}+T_{\rm analysis}.
\]

The sum includes tuning, pilots, confirmation and failed attempts exactly once. Include model-specific initialization/preprocessing in setup or nonrun preparation without double-counting. Measure default and worst-case eligible methods before authorizing a campaign in the project decision log.

| Stage | Initial scale | Resource forecast or measurement rule |
|---|---|---|
| Environment/data | Three public archives, small scalar source modules | Data bytes verified above; wheel/environment size measured after resolution |
| Scalar/gradient tests | Small arrays; CPU double | Measure one run; no GPU dependency |
| Graph parity/resume | Two minibatches, then short interrupted trajectory per family | Record wall time, peak counters and mismatch |
| Synthetic diagnostic | Width 32/128, depth 2 then 8, stated sample sizes | Pilot complete runs, extrapolate from actual steps |
| Real defaults/pilot | Concrete + Superconductivity, shared backbone | Measure 200 steady-state update steps plus validation/checkpoint |
| Tuning | Defaults + equal candidate batches | Continue using validation adequacy rule; log exact cost and saturation |
| Confirmation | Frozen methods/tasks × fixed calibrated seed count | Sum actual forecast with p90 step time; uncertainty in forecast reported |
| Timing/analysis | Frozen checkpoints and raw prediction arrays | Separate model-only and full-pipeline boundaries; count sessions |

As an illustration only, one million steps at a **measured** 10 ms/step would take 2.78 hours before overhead; at 100 ms/step it would take 27.8 hours. Neither number predicts this laptop.

Keep enough free disk for environment caches, immutable data, two checkpoints per active run, predictions, failures and raw logs. The larger example’s model-plus-Adam checkpoint is about 25.5 MB before overhead; 200 runs retaining two such checkpoints would approach 10.2 GB. Most core models are smaller. Compute actual serialized sizes, avoid saving every epoch indefinitely, and reserve space for atomic temporary writes.

### 7.8 Timing protocol

Keep the laptop on power with the same power mode and record background load. After 20 warm-up steps, inspect consecutive 20-step windows. Warm-up is stable when the last three window medians each differ from their joint median by at most 5%; otherwise continue up to 400 steps. If still drifting, stop timing, record thermal/background instability and repeat in a fresh session. The 5% rule is operational, not an improvement threshold.

Synchronize MPS immediately before and after timed blocks, using a monotonic high-resolution clock. Do not time asynchronous dispatch alone or synchronize every tiny operation. Randomize method order across at least three separately started timing sessions; blocks within a session estimate measurement variability, not independent model-quality observations. [R35]

Report:

- preprocessing/initialization, search, training, validation/checkpoint and loading separately;
- full training time including routine logging;
- diagnostic overhead via a matched diagnostic-on/off measurement;
- inference with evaluation mode and inference-mode autograd disabled;
- resident-input model-only timing and host-input-to-host-output timing;
- batch 1 and a shared throughput batch;
- median and tail block times, actual repetitions and session variation.

For the timing pilot, collect ten 20-step blocks per method/session. For each such cell, use 5,000 seeded within-session block-bootstrap draws to estimate a 95% percentile interval for its median; let h10 be half the interval width and t10 its median. Use the planning approximation m=ceil(10*(h10/(0.02*t10))^2), with a minimum of ten. Choose the largest m across pilot cells as the common fixed final block count per method/session, subject to a recorded measured timing budget. The inverse-square-root extrapolation is a planning approximation, not a guarantee under drift or autocorrelation. Freeze the count and budget before final timing.

Report final within-session median/percentile intervals, the median and range across separately started sessions, and paired session-level time ratios. Do not pool all blocks as independent cross-session replicates. If the precision target exceeds the budget or final intervals remain wide, report imprecision; do not selectively extend favorable methods. These targets concern measurement quality, not a minimum worthwhile speedup. Do not treat warm caches or preprocessing amortization as free for one method only.

### 7.9 Checkpointing and recovery

At completed optimizer-step boundaries, preserve model/buffers, optimizer, scheduler, epoch/global step, best-checkpoint score, early-stop state, all Python/NumPy/CPU/MPS RNGs, current row permutation and cursor, calibration state, elapsed resource counters, and source/config/environment/data/split hashes. [R39]

Start with data-loader workers zero and explicit stored permutations. Write a temporary checkpoint, flush/fsync, checksum, atomically rename on the same filesystem, then update the latest pointer. Keep the preceding verified checkpoint.

Before confirmation, compare uninterrupted \(k+m\) steps with \(k\) steps followed by restart and \(m\) steps on the same backend. Use deterministic algorithms where supported and record any remaining envelope. Seeds do not guarantee equality across platforms or releases. [R37]

A compatible interruption resumes. An OOM is logged, the process is stopped, a faithful predeclared fallback is calibrated, and the full affected run restarts with a new attempt ID linked to the failure. A numerical divergence is a method outcome, not a reason to secretly lower LR. Changing device midway through a confirmatory trajectory is a new treatment, not exact continuation.

## 8. Risk-ordered work packages and decision gates

These are concrete implementation tasks for the coding agent. “Accept” means the specified gate is satisfied, not that the research hypothesis is true. Each package writes its decision and evidence to the project artifacts before proceeding.

### WP00 — Establish the executable and provenance boundary

- **Objective/risk:** prevent an unsupported backend, inaccessible asset or incompatible license from invalidating later work.
- **Dependencies/entry:** this plan; access to the target laptop and permitted public downloads.
- **Actions:** run Section 7 discovery; create an isolated environment; resolve actual arm64 packages; archive licenses, wheel hashes and source pins; download and hash the three exact datasets; record optional TabPFN eligibility separately. Do not download large model assets before their access/resource gate.
- **Files/outputs:** environment.json, dependencies.lock, sources.json, licenses/, data/raw/manifest.json, decisions.jsonl.
- **Validation:** actual package imports; installed version/architecture; data hashes; available memory/disk. Check that torchortho's unconditional helper imports do not introduce a hidden CUDA requirement.
- **Resources:** small public assets plus measured package environment; no training.
- **Decision:** accept if the CPU path and essential data are available; MPS may remain provisional. If one optional asset fails, narrow that comparison. If required data cannot be lawfully accessed, stop its task and record a protocol amendment before substituting data.

### WP01 — Prove and check scalar definitions

- **Objective/risk:** mathematical or numerical defects masquerading as training behavior.
- **Dependencies:** WP00 CPU environment; formulas in Section 4.
- **Actions:** implement candidate, nested quadratic, affine/frozen controls and all four factorial cells. Implement Hermite recurrence and scalar rational/spline references. Reproduce the Gaussian identities and special cases.
- **Files:** src/lpa/activations/, tests/test_scalar_properties.py, tests/test_gradients.py.
- **Outputs:** properties.json containing tested dtype, coefficient settings, max errors and explicit exceptions; initialization JSON files.
- **Tests:** affine identity; BerLU special case; endpoint values/slopes; exact attained derivative bound; coefficient/input gradients; small/large finite inputs; degree conversion; no hidden output clipping. Check parameter finiteness before as well as after projection.
- **Resources:** CPU float64 arrays; measured seconds/minutes, not model training.
- **Decision:** any failed indispensable invariant blocks training. Repair the implementation; do not classify it as evidence against learned activations.

### WP02 — Reproduce close baseline semantics and establish backend eligibility

- **Objective/risk:** weak or mislabeled implementations, source bugs and silent mixed-device execution.
- **Dependencies:** WP01 mathematical references, pinned source snapshots.
- **Actions:** compare native torchortho with recurrence; report zero-input/initializer discrepancies; compare DeepSplines effective coefficients, TV2 and derivatives; compare rational A and B separately with their own initializer assets. Implement minimal device-aware adapters without changing formulas. Run complete CPU32/MPS32 forward/backward/update tests.
- **Files:** src/lpa/baselines/adapters.py, tests/test_reference_parity.py, tests/test_backend.py, artifacts/parity/.
- **Outputs:** per-method eligibility table with exact source SHA, formula variant, discrepancies, adapter changes and tolerances.
- **Resources:** tiny tensors and two-batch models; no full public benchmark.
- **Decision:** all-CPU is a faithful local fallback. If a baseline cannot be faithfully reproduced, its comparison is blocked. A native defect triggers a corrected mathematical baseline plus source-reproduction record, never a celebratory win against the defect.

### WP03 — Materialize data splits and validate the evaluation harness

- **Objective/risk:** leakage, wrong units, dependent-row pseudoreplication or a metric that manufactures a win.
- **Dependencies:** WP00 data hashes; WP01 basic models.
- **Actions:** implement exact group canonicalization/split; reproduce Section 5 counts; fit train-only preprocessing; implement group RMSE, prediction fallback, raw prediction export and paired analyses. Protect test scoring behind an explicit protocol-freeze file.
- **Files:** src/lpa/data.py, src/lpa/metrics.py, src/lpa/evaluate.py, tests/test_data_metrics.py; data/splits/.
- **Tests:** zero group overlap, target alignment, inverse scaling, bootstrap multiplicity, toy groups with unequal sizes, equivalent alias models, known degraded predictor, label-shuffle negative control, finite extreme predictions and injected NaN prediction.
- **Expected outputs:** split/preprocessing hashes, exact hand-calculated metric matches, alias difference zero under deterministic CPU execution, failures visibly retained.
- **Resources:** public datasets fit comfortably in ordinary CPU memory; inspect actual peak.
- **Decision:** fix any harness defect before comparing methods. If the expected easy-task contrast is invisible, distinguish lack of sensitivity from a method failure.

### WP04 — Validate training, logging and exact resumption

- **Objective/risk:** frozen coefficients, wrong gradients, zero nonlinear capacity or inconsistent restarted trajectories.
- **Dependencies:** WP02 eligible device and WP03 correct harness.
- **Actions:** implement the backbone and optimizer groups; fit tiny affine/quadratic tasks; use targeted nondegenerate gradient/update probes to verify trainable parameters are connected and frozen controls remain fixed. Legitimate zero gradients or projection saturation on ordinary data are not implementation failures. Save/restart a short run with explicit sampler state.
- **Files:** src/lpa/models.py, train.py, checkpoint.py, tests/test_resume.py.
- **Outputs:** trajectory comparison, optimizer-state/parameter-group dump, finite-state logs and measured restart envelope.
- **Acceptance:** intended task loss decreases; direct quadratic control fits its noiseless case; uninterrupted and resumed runs agree within the documented same-backend envelope; changing a provenance hash is rejected.
- **Resources:** tiny models and short runs.
- **Decision:** implementation defect → repair; backend nondeterminism → quantify or CPU fallback. A replica silently continuing with different data order is invalid, not an additional seed.

### WP05 — Diagnose mechanism exposure with cheap experiments

- **Objective/risk:** an unsuitable proxy or a stability intervention that never acts.
- **Dependencies:** WP04.
- **Actions:** execute Section 4.7 synthetic suite and Airfoil development; begin shallow and normalized, then add depth/no-normalization diagnostics only if necessary to expose the stated mechanism. Run the four-cell factorial, nested quadratic and frozen control.
- **Files:** configs/explore/synthetic.yaml, airfoil.yaml, diagnostics.py; artifacts/explore/.
- **Outputs:** learning curves, sample-size/depth effects, tail occupancy, actual slope bounds, projection rates, derivative energy, layer gradients and branch/model sensitivity.
- **Acceptance:** correct easy-case controls, documented exposure status (including inactive tails/cap), and a plausible diagnostic interpretation. Inactivity blocks an exposed-regime H2 conclusion but does not block Track A or require a favorable error difference.
- **Resources:** forecast from completed cheap runs; do not start a large architecture.
- **Decision:** inactive tails/cap → report lack of exposure and use a labeled stress condition; synthetic-only success → limited inductive-bias finding; broad failure with correct simple controls → investigate optimization/parameterization; no plausible advantage → permit a bounded real comparison, not an open-ended search for a winner.

### WP06 — Calibrate resources and reproduce practical baselines

- **Objective/risk:** local feasibility, undertrained baselines or unrealistic cost assumptions.
- **Dependencies:** WP02–WP05.
- **Actions:** run actual update/validation/checkpoint timing; select common effective batches; measure memory. Execute native/default configurations for all required activation families plus CatBoost, TabM and simple methods. Validate ordinary convergence before tuning.
- **Files:** configs/calibration.yaml, calibrate.py, benchmark.py; artifacts/calibration.json, baseline_reproduction.md.
- **Outputs:** per-method step/epoch/selection cost, memory, convergence horizon, eligible device, forecasted tuning/confirmation cost.
- **Acceptance:** comparable implementations, affordable defaults/pilots/tuning, and a provisional campaign cost envelope. Final replication precision and whole-campaign affordability are resolved in WP07 after its pilot determines N. “Affordable” must be documented from available disk/memory and measured elapsed-work plan, not assumed from chip branding.
- **Decision:** resource issue → faithful batch/evaluation fallback; undertraining → common horizon extension; missing strong baseline → narrower claim or repair before confirmation; healthy defaults → WP07.

### WP07 — Tune fairly and freeze the confirmatory protocol

- **Objective/risk:** proposal-only effort, adaptive selection and undefined statistical sensitivity.
- **Dependencies:** calibrated costs and correct baselines.
- **Actions:** run Section 6 tuning allocations; preserve defaults and failures; select champions on validation; complete paired pilot and precision-cost plan. Freeze Track A, Track B and any Track C affirmative hypothesis before test scoring.
- **Files:** configs/search/, configs/frozen/, preregistration.json, analysis_plan.json, selection.csv.
- **Outputs:** exact candidate lists, completed/refit counts, search curves, selected configs, \(N\), seed blocks, uncertainty assumptions, multiplicity family, cost forecast and signed-off decision record. A human signature is not required; an immutable hash and logged transition suffice.
- **Acceptance:** no unresolved implementation issue; explicit local precision and budget; essential baseline search boundaries resolved or transparently limited.
- **Decision:** weak sensitivity → expand prospective pilot/replication if feasible or label limited precision; promising result → proceed without changing the held-out population; no special claim emerges → retain the planned neutral comparison.

### WP08 — Execute protected confirmation

- **Objective/risk:** genuine replication versus selection of favorable runs.
- **Dependencies:** immutable WP07 protocol; all earlier gates.
- **Actions:** run every frozen seed/method/task combination with equal input access; preserve numerical failures, apply the fixed fallback procedure; evaluate once under the frozen analysis. Randomize method run order in blocks to reduce temporal hardware bias.
- **Files:** run.py, evaluate.py, analyze.py; artifacts/confirm/RUN_ID/.
- **Outputs:** complete manifest, checkpoints, per-row predictions/flags, original-unit scores, paired contrasts, seed/group intervals and predefined tests.
- **Acceptance:** complete scheduled run roster or an explicit, reason-coded missing-attempt record. No optional stopping for favorable significance.
- **Resources:** measured campaign forecast and automatic checkpoints.
- **Decision:** positive/negative/inconclusive classification follows Section 11; operational failure may resume only under Section 7.9; any methodological amendment after exposure requires fresh confirmation for a revised affirmative claim.

### WP09 — Complete attribution, robustness and performance accounting

- **Objective/risk:** a gain explained by initialization, generic regularization, capacity or an omitted cost.
- **Dependencies:** frozen attribution plan; usable checkpoints. It may execute before statistical unblinding if more efficient.
- **Actions:** perform the existing factorial/nested/frozen comparisons, source-native versus optimized evaluator parity, common clipping/norm sensitivity, per-task/group analyses and frozen timing protocol. Include every predeclared guardrail.
- **Files:** configs/robustness/, benchmark.py, diagnostics.py, artifacts/robustness/.
- **Outputs:** conditional cap/tail effects and interaction, degree/learning evidence, cost frontier, failure table, worst-task regressions.
- **Acceptance:** attribution claims follow gatekeeping/multiplicity, and every favorable claim has its strongest plausible alternative explanation examined.
- **Decision:** gain disappears → narrow or reject the proposed explanation; important regression → report tradeoff; unresolved confound → further exploratory diagnosis, not retroactive confirmation.

### WP10 — Reproduce the result from preserved artifacts

- **Objective/risk:** results depend on ephemeral state or undocumented choices.
- **Dependencies:** complete runs and source/environment locks.
- **Actions:** in a fresh local environment, rerun scalar/harness tests and one predetermined representative seed block. Recompute all tables from raw predictions/logs without retraining. Check checkpoint provenance and hashes.
- **Files:** reproduce.py, artifacts/reproduction.json, report/.
- **Outputs:** successful artifact rebuild, recorded differences/envelope, missing-file audit.
- **Resources:** one representative run block plus inexpensive analysis; no indiscriminate full rerun.
- **Decision:** deterministic data/metric mismatch → invalid report until fixed; a mismatch outside the preregistered same-backend reproducibility envelope or systematic drift that changes interpretation → investigate before an affirmative claim; reproducible analysis → WP11. Ordinary seed variation is assessed through the planned mean-effect interval; a single-run SD larger than the mean effect does not by itself invalidate a sufficiently precise replicated result.

### WP11 — Decide and report

- **Objective/risk:** overclaiming a local or negative result, or pivoting until something wins.
- **Dependencies:** complete evidence and all known limitations.
- **Actions:** classify each hypothesis, include unfavorable runs and source-adapter differences in the report, and identify the exact tested domain and practical tradeoffs. Record continue/revise/narrow/stop with a concrete reason. Public posting is not required by this handoff.
- **Files:** report/results.md, report/summary.csv, report/figures/, decisions.jsonl, progress.md.
- **Outputs:** complete research report and a next-experiment contract only if evidence warrants it.
- **Acceptance:** every headline has a baseline, metric, uncertainty, cost and scope; no claim extends beyond the executed program.
- **Decision:** use Section 11 outcome map. No automatic escalation to a large model or a different dataset merely because the original result was unfavorable.

## 9. Repository, interfaces and experiment records

### 9.1 Minimal repository layout

The following tree is a **proposed project structure**, not a claim that those files already exist:

~~~text
learned-activation-poc/
  README.md
  pyproject.toml
  dependencies.lock
  research_poc_plan.md
  progress.md
  decisions.jsonl
  amendments.jsonl
  preregistration.json
  analysis_plan.json
  src/lpa/
    __init__.py
    __main__.py
    cli.py
    activations/
      cubic.py
      hermite.py
      berlu.py
      spline.py
      rational.py
    baselines/
      adapters.py
      tabular.py
    data.py
    models.py
    train.py
    checkpoint.py
    diagnostics.py
    metrics.py
    evaluate.py
    analyze.py
    benchmark.py
    calibrate.py
    reproduce.py
  configs/
    explore/
    search/
    frozen/
    robustness/
  tests/
    test_scalar_properties.py
    test_gradients.py
    test_reference_parity.py
    test_backend.py
    test_data_metrics.py
    test_resume.py
  data/
    raw/
    splits/
    processed/
  third_party/
    source_manifest.json
    licenses/
  artifacts/
    environment.json
    calibration.json
    explore/
    confirm/
    timing/
  report/
    results.md
    summary.csv
    figures/
~~~

Avoid a database, web service, distributed scheduler or custom experiment platform. JSON/JSONL, CSV/NPZ, source control and ordinary checkpoint files are sufficient.

### 9.2 Shared interfaces

Define project-owned interfaces; they are not asserted third-party APIs.

~~~python
class ActivationProtocol:
    # nn.Module implementations obey these semantics.
    def forward(self, x): ...
    def project_(self): ...       # no-op for methods without post-step projection
    def diagnostics(self, x): ... # detached scalar summary, no training-state change
    def spec(self): ...           # JSON-safe immutable definition

class PredictorProtocol:
    def fit(self, train, validation, config, seed, run_dir): ...
    def predict(self, x, batch_size): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path, manifest): ...
~~~

The PredictorProtocol boundary uses raw numeric feature arrays of shape (N,d), in the fixed source feature order, and targets of shape (N,) in original physical units. The adapter owns train-fitted preprocessing, method-specific training and inverse target transformation. Its predict method always returns a CPU array of shape (N,) in original target units. Save/load includes every transform and the feature schema; no transform may be refitted during prediction.

The common evaluator validates shape/units, performs group aggregation and nonfinite-prediction handling, and writes raw outputs. It must not inverse-transform an already restored prediction. Separate raw predictions from the evaluated fallback predictions. A round-trip test with nonzero target mean and nonunit scale must cover neural, tree and simple-regression adapters.

Dataset objects contain features, target, row ID, group ID, split ID and preprocessing metadata. Do not carry protected labels into training loaders. Diagnostics receive only declared train/validation probe inputs.

**Configuration requirements:**

- Parse strictly; reject unknown keys and missing mandatory fields.
- Store formula variant, sharing, degree, radius/cap, initializer source/hash, normalization epsilon/affine flag, architecture, optimizer groups, clipping, schedule, horizon, effective batch, seeds, data/split hashes, backend, dtype and reference/source revisions.
- Resolve all defaults into a canonical sorted JSON before hashing.
- Confirmation requires either a clean recorded source commit or an archived effective source tree, including patches and untracked code, with a file-content manifest and hash. A commit SHA alone does not identify dirty code. Recompute that effective-code hash on resume and reproduction.
- Expand search candidates to explicit immutable configs before running them.
- Method changes that alter the hypothesis, population, metric, baseline fairness or interpretation require a methodological amendment; mundane file names, logging fixes and equivalent batching can be decided autonomously with a record.

Example **project configuration**, not an existing package interface:

~~~yaml
schema_version: 1
phase: explore
task: concrete_165
model:
  backbone: residual_mlp
  width: 128
  blocks: 4
  residual_scale: inverse_sqrt_blocks
  layernorm_eps: 0.00001
  layernorm_affine: false
activation:
  family: bounded_cubic
  sharing: layer
  radius: 1.5
  cap: 1.5
  tails: true
  projection: post_adam_step
  initializer: gaussian_grid_gelu_ls
optimizer:
  family: adamw
  lr: 0.001
  weight_decay_dense: 0.001
  weight_decay_shape: 0.0
  shape_lr_multiplier: 1.0
  clip_global_norm: 1.0
training:
  batch_size: 128
  epochs: 200
  patience: 30
  validation_metric: group_rmse
runtime:
  device: selected_by_eligibility_gate
  dtype: float32
  implicit_mps_fallback: false
~~~

This is an exploratory example. The CLI must refuse a confirmation run until runtime choices, hashes, seed roster and the calibrated fixed replication count are resolved in the frozen configuration.

### 9.3 Concrete practical search spaces

These define initial domains; Section 6.6 controls equal allocation and prospective extension.

| Method | Initial candidates and required behavior |
|---|---|
| Candidate/quadratic | R = 1.5, 3, 6; S = 1, 1.5, 3; shared LR/decay/shape-LR domains; no hidden learnable radius |
| Hermite | degree = 2, 3, 5; native asymptotic-gain, finite-gain, deterministic value/derivative-fit initializers; no clamp in the primary reproduction |
| Spline | Exact K=51/R=1/identity/unconstrained/TV=0 demo anchor; tuned K = 5, 9, 17, 51; R = 1.5, 3, 6; S = 1, 1.5, 3; TV coefficient = 0, 1e-5, 1e-3; identity/ReLU/GELU-knot initial values; apply_scaling=False |
| Rational | A and B; degree (5,4); own GELU, Leaky-ReLU and swish assets; common shape-LR domain |
| BerLU | epsilon = 0.01, 0.1, 1.5, 3; alpha starts 0.01; native free-alpha and explicitly matched bounded-alpha variants |
| Ridge | train-standardized features; alpha in 10 raised to integers -6 through 6; include intercept |
| Quadratic ridge | degree-two products including squares, no redundant constant column; train-only centering/scaling of expanded features; same ridge grid |
| RBF ridge | \(\exp(-\gamma\|x-x'\|^2)\); \(\gamma\) multipliers 0.1, 0.3, 1, 3, 10 times inverse median nonzero train-pair squared distance; same alpha grid |
| Nyström RBF variant | only when exact larger-task cost is excessive; components 256, 1024, 2048, train-only seeded landmarks, same kernel/ridge domains; explicit approximate-method label |
| CatBoost | default plus depth 4, 6, 8, 10; LR 0.01, 0.03, 0.1; L2 leaf regularization 1, 3, 10; initial 2,000-tree horizon, extended if best iteration is boundary-limited |
| TabM | preserve \(k=32\); width 128, 256, 512; blocks 2, 3, 4; dropout 0, 0.1, 0.3; LR 0.0005, 0.001, 0.002, 0.003; weight decay 0, 0.0003, 0.003; include pinned author default |
| TabM numeric embeddings | include the author-supported no-embedding and piecewise-linear-embedding route where available; bins 16, 32, 64 and embedding dimensions 8, 16; train-only bins; count extra preprocessing/parameters |
| Wider ordinary MLP | widths 64, 128, 256; blocks 2, 4; fixed activation selected on validation; same optimizer search; small ensemble only as a separately costed procedure |
| TabPFN | official default eligible checkpoint/preset first; any local ensemble/preprocessing options come from the pinned supported configuration, recorded explicitly before scoring; no paid API |

Use exact official package argument names only after inspecting the pinned API. The table defines experimental choices, not speculative constructors.

For CatBoost early stopping, provide validation-group weights proportional to inverse group size when the supported weighted-RMSE route is verified, matching the primary criterion. Also keep the native stopping recipe as a practical candidate. For any baseline whose native stopping metric differs, disclose it and offer primary-metric checkpoint selection where feasible.

Compute RBF median distance from all training pairs on Concrete; for the larger task use 100,000 reproducibly sampled train-only pairs, excluding exact zero distances. If no nonzero distance exists, the dataset/harness gate fails. For quadratic ridge on Superconductivity, there are 3,402 degree-one/two nonconstant features; an expanded float64 training matrix is about 352 MB. Select a converged standard solver under the memory budget, record residual/tolerance and include preprocessing/solve time.

An approximate kernel or smaller ensemble is not silently interchangeable with its exact/larger version. A method may be excluded from the equal-data direct comparison only with a specific access or resource reason.

### 9.4 Proposed CLI commands

All commands beginning with “python -m lpa” below are **interfaces the coding agent must implement**. They do not exist merely because they are listed here.

After creating pyproject.toml with the src-layout package configuration, install the project in its isolated environment using the existing command `python -m pip install -e .`. Implement src/lpa/__main__.py to import main from .cli and invoke it under the standard __name__ == "__main__" guard. Verify python -m lpa --help before using the proposed subcommands.

~~~bash
python -m lpa doctor --output artifacts/environment.json
python -m lpa data prepare --manifest configs/data.yaml
python -m lpa verify --suite scalar
python -m lpa verify --suite reference-parity
python -m lpa verify --suite backend
python -m lpa verify --suite harness
python -m lpa verify --suite resume
python -m lpa run --config configs/explore/synthetic.yaml
python -m lpa calibrate --config configs/calibration.yaml
python -m lpa tune --config configs/search/activation.yaml
python -m lpa freeze --analysis analysis_plan.json --output preregistration.json
python -m lpa run-confirmation --protocol preregistration.json
python -m lpa evaluate --protocol preregistration.json
python -m lpa analyze --protocol preregistration.json
python -m lpa benchmark --protocol preregistration.json
python -m lpa reproduce --protocol preregistration.json
~~~

The freeze command validates gate records, immutable configs, hashes and analysis choices. It does not invent missing measurements. The evaluator refuses a missing/stale protocol. Running a source-reproduction script in an isolated environment is a separate recorded command.

### 9.5 Run record and output schemas

Each attempt receives a unique run ID derived from canonical config hash, task, phase, seed block and attempt number. A rerun never overwrites a failure.

**manifest.json fields:**

~~~json
{
  "schema_version": 1,
  "run_id": "derived-at-execution",
  "phase": "explore-or-confirm",
  "method_id": "explicit-formula-and-variant",
  "task_id": "concrete_165",
  "seed_block": 0,
  "attempt": 1,
  "parent_attempt": null,
  "config_sha256": "resolved-at-execution",
  "source_revision": "resolved-at-execution",
  "environment_sha256": "resolved-at-execution",
  "data_sha256": "resolved-at-execution",
  "split_sha256": "resolved-at-execution",
  "backend": "cpu-or-mps",
  "dtype": "float32",
  "status": "created",
  "failure_kind": null
}
~~~

Placeholder strings above describe fields; production records must contain actual values and fail validation otherwise.

**metrics.jsonl**, one record per declared step/epoch: run ID; step; epoch; examples seen; training MSE; validation group RMSE; validation row RMSE; LR per group; unclipped/clipped gradient norm; update/weight norm ratio; activation tail fraction; empirical derivative second moment; actual scalar max slope; projection fraction; beta/weight/branch statistics; elapsed training/validation/checkpoint seconds; process RSS bytes; GPU tensor/driver bytes; finite-state flags.

**predictions.csv or NPZ:** run ID, task, row ID, group ID, split, true target in original units, raw prediction, evaluated prediction, fallback flag/reason. Store predictions by seed, not just means. Retain test outputs only in the confirmation artifacts.

**timings.csv:** method/checkpoint/config hash, session, run-order block, power mode, dtype/device, batch, boundary definition, warm-up count, repetitions, duration_ns, preprocessing/transfer inclusion, memory counters, drift notes.

**decisions.jsonl:** timestamp, package/gate, evidence paths/hashes, outcome, reason, next action and unresolved uncertainty. **amendments.jsonl:** what changed, why, whether test results were exposed, which claims are invalidated and what fresh confirmation is required.

Update progress.md after each gate with the last complete stage, active run, next command and known failures. Execution must resume from files, not conversational memory.

### 9.6 Checkpoint and report integrity

- Clone model/optimizer tensors or serialize immediately; a shallow state-dict reference can mutate as training continues.
- Save latest and best separately. Selection uses validation only.
- Preserve native initializer assets, fitted coefficients, solver metadata, source modifications and failed checks.
- Analysis operates from raw records and verifies roster completeness before aggregation.
- Report every planned method, including crashes, exclusions, null results and substantial regressions.
- A source change, package update, precision change or optimized evaluator invalidates prior backend/timing parity until rechecked.
- Do not claim bitwise cross-device reproducibility. Report the exact level reproduced.

## 10. Ablation, robustness and adversarial program

Run expensive extensions only after the cheap invariants and relevant exposure checks. Many tests use already trained checkpoints.

| Threat or alternative explanation | Required test | Interpretation/action |
|---|---|---|
| Polynomial target was designed to guarantee success | Balanced synthetic families; explicit quadratic regression | Restrict synthetic claims to inductive bias |
| Cubic simply adds a free tail/offset | Nested quadratic with both tails/offset free; affine-calibrated GELU | If matched, extra cubic curvature has not earned value |
| Good fixed approximation explains gain | Frozen initialized activation and logged initial value/derivative distributions | Distinguish learned shape from initialization |
| Better conditioning rather than expressiveness | Same slope-coordinate factorial; monomial/Hermite conversion checks | Name optimization/parameterization effect accurately |
| Tails or cap never activate | Tail occupancy and projection/saturation rates | Lack of exposure is not a test of exposed-regime efficacy |
| Tail comparison begins with a different function outside R | Record initial predictions/derivatives and tail occupancy | A positive tail effect includes changed initialization/tail inductive bias; require trajectory evidence before calling it explosion prevention |
| Adjacent weights compensate for cap | Layer weight norms, branch gain and full-model input-gradient norms; rescaling-equivalence check | Scalar certificate alone cannot support a network-stability claim |
| Spline constraint implemented incorrectly | Source transform/TV/gradient parity, far-tail and knot tests | Repair before any comparison |
| Rational denominator changed by adapter | Separate A/B parity and initializers | No mislabeled rational baseline |
| Rare preactivation excursions dominate | Train-distribution quantiles; correctly relabeled synthetic scale/tail shifts; feature corruption only under explicit measurement-noise assumptions | Do not relabel arbitrary physical-feature perturbations with unchanged targets as true OOD ground truth |
| Generic stabilization explains gain | Shared gradient clipping, normalization and residual scaling; no-normalization stress clearly separated | Candidate must beat properly stabilized competitors |
| Extra capacity or search effort explains gain | Wider fixed network, simple ensemble, same search allocation and reported actual cost | If ordinary compute matches, no special efficiency claim |
| Coefficient values appear “interpretable” due to basis choice | Compare actual functions on common grids/distributions, not raw coefficient magnitudes | Avoid basis-dependent interpretation and scale nonidentifiability |
| Box cap wastes feasible derivative space | Compare cap-coordinate saturation with exact derivative extrema | High saturation with unused actual slope range motivates an exact-constraint follow-up, not retrospective relabeling |
| Finite state becomes nonfinite only at inference | Fixed label-independent fallback, raw-failure flags, large-finite-error retention | Reliability effect remains visible |
| LayerNorm creates a numerical issue | Near-zero-variance vectors; exact epsilon; full input-gradient probe | Scalar activation alone may not be causal |
| MPS discrepancy changes a tiny gain | CPU64 oracle and CPU32/MPS32 parity; one full seed on each where feasible | Backend-sensitive result remains provisional |
| Thermal order creates speedup | Randomized order across sessions; explicit synchronization/drift policy | Timing artifact blocks efficiency claim |
| Real tasks contradict synthetic success | Preserve both results; inspect approximation and noise structure | No generic superiority; only an application-specific hypothesis remains |
| Restart changes experiment identity | Full state and provenance hash checks; interrupted/uninterrupted comparison | Invalid continuation must be repaired or rerun |
| Parameter contraction kills nonlinearity | Function plots, nonlinear energy and actual derivative distributions | A trained near-affine solution may show constraints remove useful capacity |
| Polynomial extrapolation is genuinely correct | Noiseless quadratic/cubic targets outside training range, with correct labels | Linear tails may rationally lose; report this boundary rather than calling it a failure of all polynomials |

For whole-model sensitivity, use a fixed 128-example training/validation probe. For a scalar output, the input-gradient L2 norm is a direct local sensitivity statistic. For intermediate vector branches, use fixed Rademacher vector–Jacobian probes with recorded vectors; label them estimates, not exact spectral norms. Compute diagnostic summaries at initialization and selected epochs, detached from the training objective. SVD/spectral estimates of weight matrices can run on CPU off the timed path.

A generic rescaling control should transform activation output by a nonzero constant and compensate the immediately following affine weight/bias to preserve predictions. The evaluator must recognize functional equivalence within numerical tolerance. This illustrates why a raw coefficient or scalar cap cannot be interpreted independently of adjacent weights.

## 11. Outcome interpretation and evidence-triggered continuation

| Observed outcome | Defensible conclusion | Required next decision |
|---|---|---|
| Scalar/gradient/reference test fails | Broken implementation or ambiguous baseline | Stop dependent comparisons; repair |
| CPU correct but required MPS graph fails | Local acceleration limitation | Run faithful CPU comparison or narrow device claim |
| Correct experiment lacks tail/cap exposure | Unsuitable proxy for that stability question | Add a labeled exposure diagnostic; retain practical null |
| Real-task uncertainty remains wide | Underpowered/inconclusive within local budget | Report sensitivity/cost; expand only under a prospective plan |
| Candidate improves over raw cubic but loses strong learned alternatives | Known stabilization idea works; differentiated advantage absent | Prefer stronger baseline; preserve negative result |
| No resolved advantage over learned quadratic at the achieved precision | Cubic degree has not shown added value | Prefer the simpler model when the measured cost/accuracy tradeoff supports that decision; do not infer equivalence from nonsignificance |
| Frozen shape matches learned shape | Learned coefficients have not shown value | Prefer the fixed function; investigate only a new justified regime |
| Candidate improves only synthetic interactions | Restricted inductive-bias evidence | Require a real interaction-dominated application before pivot |
| Small robust gain with no important regression | Genuine finite-panel empirical result | Replicate on a fresh independently selected task and one larger backbone |
| Error improves with substantial time/memory/reliability cost | Quantified tradeoff | Evaluate application value without claiming overall superiority |
| Candidate beats activation champion but loses CatBoost/TabM/foundation comparator | Component benefit, no best-pipeline claim | Report both; choose deployment baseline on the practical frontier |
| Gain disappears after stronger baseline tuning | Original apparent advance was a comparison artifact | Revise conclusion; retain all versions and search cost |
| Both tasks/guardrails show precise negative results | No useful advantage under this protocol | Stop this refinement or narrow its claim; do not keep changing tasks |
| Global polynomial wins on correct extrapolation | Tail constraint trades away useful structure | Consider the established global/Hermite method for that explicit application |
| High coefficient saturation but low attained slope with underfit | Conservative box parameterization may be restrictive | Prospectively compare exact interval derivative constraints; original result stands |
| Effect grows consistently with depth/scale while local proxies remain informative | A scale-dependent hypothesis has evidence | Authorize the next feasible scale with predicted observable and cost |
| Only hypothetical future scaling could rescue a null | No positive escalation evidence | Do not increase cost merely to preserve optimism |

A later exact-constraint experiment could enforce the cubic derivative's actual extrema instead of the Bernstein box. Its motivation must be measured wasted constraint capacity, and it must retain the original negative result. A later interaction-focused application needs real data whose structure is independently justified, together with direct polynomial regression/factorization baselines. A later HE study needs a genuine polynomial-only graph and measured encrypted arithmetic; the current model does not supply that.

A positive result at this scale does not establish a benefit in transformers, large image models, scientific deployment, certified robustness or physical discovery. Conversely, a shallow proxy cannot refute a clearly specified depth-dependent mechanism it never expresses. Escalation requires a measured trend or a theoretically necessary observable absent at the smaller scale, not an unfalsifiable promise.

Fresh confirmation after test-driven method changes must use data not already used to choose that change. A new seed or a new split of already inspected rows is not automatically fresh evidence. Select any new task by provenance, dependence structure and relevance criteria before obtaining candidate comparative scores, document access/license, and freeze its protocol.

## 12. Audit record, remaining risks and completion checklist

### 12.1 Audits conducted while drafting

Three distinct delegated challenge reviews were performed during preparation: scientific/prior-art, experimental-validity, and implementation/feasibility. A second pass examined the complete work packages and interfaces; the affected sections were revised and targeted final rechecks found no further substantive correction in their reviewed scope. These were agent reviews within this project, **not independent external peer review**, and they do not certify unknown experimental outcomes. Source inspection and small CPU/data checks were also performed; full-model validation was not.

| Material weakness found | Revision made | Remaining obligation |
|---|---|---|
| Original idea and derivative-management novelty overstated by initial framing | Historical and modern direct precedents documented; contribution classified as empirical refinement | No universal absence-of-prior-art claim |
| Bernstein tails closely overlap BerLU/DeepBern and splines | Added those competitors and explicit novelty threats | Faithful local implementations and tuning |
| Cubic-versus-BerLU confounds free tails/offset with degree | Added nested quadratic with identical free endpoint slopes/offset | Run degree attribution |
| Zero-offset, wide-range initialization fit poorly | Used an explicit offset and label-free fitted default; reported CPU calculation | Validate task behavior, not just fit residual |
| Full-versus-global contrast bundles cap and tails | Added separate conditional contrasts and interaction | Respect frozen multiplicity/gatekeeping |
| Native Hermite code differs from casually assumed conventions | Retrieved exact source pin; specified factorial basis, native gain and mathematical recurrence; flagged fitter/zero-gradient checks | Execute CPU/autograd parity and correct-source comparison |
| Rational “safe denominator” was ambiguous | Included both A and B with their own assets | Device-aware parity; no variant substitution |
| Spline source uses a forward transform, not the candidate's projection | Specified exact transform, TV scaling and tail interpolation | Preserve source gradients/knots in adapter |
| Training-failure handling did not cover nonfinite held-out predictions | Added label-independent inference fallback and raw failure reporting | Verify metric remains finite without trimming finite errors |
| Apparent sample size ignored repeated configurations | Downloaded data; grouped mixtures/compositions; made Airfoil development-only | Unknown higher-level dependence remains |
| Backend support was inferred too broadly | Added exact graph/parity/import gate and whole-CPU fallback | M3 execution still unperformed |
| Larger-model parameter count was arithmetically wrong | Corrected residual formula example to 2,126,657 parameters | Measure true allocator peak and runtime |
| Timing and recovery instructions were underspecified | Added window/repetition/budget rules, copied checkpoints and complete sampler/RNG state | Run timing calibration and restart equivalence |
| Tuning choices and pilot-noise handling were implicit | Added initial grids, budget-boundary rule and upper-SD planning rule | Complete validation-only adequacy/precision gates |
| Equal-time search could receive free initial/reused results or omit refits | Charged cumulative selection costs and reserved finalist-refit time | Enforce budgets from raw attempt logs |
| Optional second primary could spend alpha again on attribution | Added a fixed 0.025 allocation to each branch if promoted | Freeze the chosen branch structure |
| Adapter output units and module entry point were ambiguous | Required original-unit outputs and round-trip tests; added __main__ and project installation | Execute all adapter and CLI gates |
| Resource gate depended on precision not yet calibrated | Made WP06 provisional and WP07 responsible for final N/budget | Do not start confirmation before WP07 acceptance |
| Reproduction rule confused individual variation with mean precision | Separated same-backend drift from replicated statistical uncertainty | Apply the frozen interval and reproducibility envelope |
| Commit SHA could omit dirty code; spline defaults were vague | Required effective-source manifest; specified exact demo and tuned initializers | Preserve actual source tree and initializer assets |

### 12.2 Unresolved risks and blind spots

- The exact combined parameterization may have prior art not retrieved, including patents, non-English material or inaccessible technical reports.
- Some historical publications and newer adjacent work were inspected only through primary abstracts; their detailed empirical claims are not used as proof of superiority.
- The scalar family is deliberately small. A negative result cannot reject every learnable activation or every globally Lipschitz cubic parameterization.
- The selected benchmarks cover two physical-regression tasks, not broad deployment value. Grouping reduces obvious leakage but cannot reconstruct unobserved laboratory structure.
- Conditional seed/group uncertainty does not include arbitrary new training sets or new tasks.
- Local timing, package compatibility, MPS gradients and checkpoint resumption still require execution on the actual laptop.
- Stronger nonlocal models or unavailable checkpoints may limit competitive claims.
- Current sources and defaults may change; immutable snapshots and validated environment locks, not mutable “latest,” determine a run.
- A mathematically bounded activation does not prevent affine-weight, normalization, optimizer or floating-point failure.
- Positive scalar/unit tests do not establish a successful learned system.

### 12.3 Handoff completion checklist

The coding agent must keep this checklist in the repository and mark it with evidence paths:

- Original purpose preserved; global-polynomial relaxation explicit.
- Each possible contribution linked to prior art and a discriminating control.
- Each headline linked to a direct baseline, metric, population and cost boundary.
- Protected test data not used for selection.
- Scalar, source, backend, harness and restart gates completed before expensive confirmation.
- All work packages have recorded entry/exit decisions and next actions.
- Actual laptop memory/OS/storage/throughput discovered before final campaign sizing.
- Versions, source/data hashes, seeds, raw outputs, failures and amendments retained.
- Numerical and hardware limitations distinguish defect, unsuitable proxy, resource limit, negative result and inconclusive result.
- Final report does not confuse a component observation with a broader scientific or deployment claim.

## 13. References and search-coverage record

### 13.1 Search scope and stopping rationale

Searches were performed on 29 October 2026 across original conference/journal proceedings, arXiv revisions, author-hosted manuscripts, official source repositories, UCI records, Apple/PyTorch documentation and a limited published-patent screen.

Search families included:

- adaptive/trainable/learnable polynomial neurons and activation functions;
- Hermite, Chebyshev, orthogonal expansions and forward/backward moments;
- Bernstein/Bézier activations, cumulative coefficients, derivative constraints and linear tails;
- smooth adaptive piecewise polynomials, deep splines, Lipschitz/TV regularization;
- Padé/rational activations, KAN/KAT and polynomial/factorization networks;
- degree allocation, parity/frequency methods and homomorphic-inference motivation;
- contemporary local tabular baselines, license terms, MPS support/correctness and reproducibility;
- dataset provenance, repeated configurations/compositions and model-selection uncertainty.

Backward tracing recovered 1993/2005 predecessors and SAAF. Forward/revision tracing found BerLU and the materially different February/May 2026 DeepBern versions. Source inspection then changed the Hermite, rational and spline implementation specifications. Further targeted searches of the selected bounded-cubic mechanism returned the same principal competing families rather than a newly decisive exact competitor within this search.

This is coverage saturation for the **selected local research decision**, not a paper quota and not proof that no prior art exists. The patent screen found an example explicitly discussing polynomial/rational/piecewise learned activations and search-based customization; it did not assess claim scope, legal status or freedom to operate. [R42] No legal novelty or patentability opinion is offered.

### 13.2 Bibliography and evidence-use notes

Each entry provides a stable source and its role. Repository/source findings are inspections, not executed reproductions. All web materials were accessed on the research date unless an immutable revision is given.

**R01.** F. Piazza, A. Uncini and M. Zenobi. *Neural Network Complexity Reduction Using Adaptive Polynomial Activation Functions* (1993). [Publisher/DOI][R01]. Historical concept priority; publisher metadata/abstract inspected. The chapter's later online-publication date is not the date of the underlying work.

**R02.** L. Ma and K. Khorasani. *Constructive feedforward neural networks using Hermite polynomial activation functions* (2005), IEEE Transactions on Neural Networks. [PubMed record][R02]; [DOI](https://doi.org/10.1109/TNN.2005.851786). Abstract-level evidence for neuron-specific polynomial activation learning; no modern reproduction claimed.

**R03.** Le Hou, Dimitris Samaras, Tahsin M. Kurc, Yi Gao and Joel H. Saltz. *ConvNets with Smooth Adaptive Activation Functions for Regression* (AISTATS 2017). [Proceedings][R03]; [full author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC6521876/). Direct precedent for smooth learned piecewise polynomials and complexity/Lipschitz control.

**R04.** Ismail Khalfaoui-Hassani and Stefan Kesselheim. *Polynomial, trigonometric, and tropical activations* (ICLR 2026; earlier preprint in 2025). [Inspected arXiv v3][R04]; [official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/933b5d002cf251b3e854d586e55ac58c-Abstract-Conference.html). Principal modern global-polynomial comparator; initialization equations and experimental scope inspected. Preserve the revision because its title/content differ from the original supplied citation.

**R05.** Haitham Khedr and Yasser Shoukry. *DeepBern-Nets: Taming the Complexity of Certifying Neural Networks Using Bernstein Polynomial Activations and Precise Bound Propagation* (AAAI 2024). [Proceedings][R05]; [DOI](https://doi.org/10.1609/aaai.v38i19.30117). Bernstein activation and certification precedent; not a reproduced certification result here.

**R06.** Wentao Zhang, Yutong Zhang, Yifan Zhu and Wentao Mo. *Universal Smoothness via Bernstein Polynomials: A Constructive Approximation Approach for Activation Functions* (2026 arXiv version; metadata reports ICIC 2026 acceptance). [Inspected v1][R06]; [publication metadata](https://arxiv.org/abs/2605.02591). BerLU equations, initialization and linear-tail construction supply a close, inexpensive comparator. Reported findings are not treated as universally established empirical superiority.

**R07.** Shayan Aziznejad, Harshit Gupta, Joaquim Campos and Michael Unser. *Deep Neural Networks with Trainable Activations and Controlled Lipschitz Constant* (IEEE Transactions on Signal Processing, 2020). [Author publication record][R07]; [DOI](https://doi.org/10.1109/TSP.2020.3014611). Earlier trainable-spline and Lipschitz-control framework.

**R08.** Vishnu Suresh Lokhande and colleagues. *Generating Accurate Pseudo-Labels in Semi-Supervised Learning and Avoiding Overconfident Predictions via Hermite Polynomial Activations* (CVPR 2020). [Full author manuscript][R08]; [official conference PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lokhande_Generating_Accurate_Pseudo-Labels_in_Semi-Supervised_Learning_and_Avoiding_Overconfident_Predictions_CVPR_2020_paper.pdf). Relevant positive and mixed evidence in a distinct learning regime.

**R09.** Jiachuan Wang, Lei Chen and Charles Wang Wai Ng. *A New Class of Polynomial Activation Functions of Deep Learning for Precipitation Forecasting* (WSDM 2022). [Author-hosted paper][R09]; [DOI](https://doi.org/10.1145/3488560.3498448). Chebyshev activations and Range Norm; an application-specific precedent, not evidence for this grouped tabular protocol.

**R10.** Bernardo Pulido-Gaytan and Andrei Tchernykh. *Self-learning activation functions to increase accuracy of privacy-preserving Convolutional Neural Networks with homomorphic encryption* (PLOS ONE, 2024). [Open article][R10]; [DOI](https://doi.org/10.1371/journal.pone.0306420). Direct learned-polynomial mechanism in a different arithmetic and application setting.

**R11.** Ibrahim Albool, Malak Gamal El-Din, Salma Elmalaki and Yasser Shoukry. *Exponential Approximation Rates and Parameter Efficiency of Learnable Bernstein Activations* (2026 preprint). [Inspected v2][R11]; [also-inspected v1](https://arxiv.org/html/2602.04264v1). The February version's cumulative increments and normalization/clamping are prior art even though the May version changes emphasis. The approximation result has representability assumptions.

**R12.** Stanislas Ducotterd, Alexis Goujon, Pakshal Bohra, Dimitris Perdios, Sebastian Neumayer and Michael Unser. *Improving Lipschitz-Constrained Neural Networks by Learning Activation Functions* (JMLR, 2024). [Paper and author-code links][R12]. Strong constrained-spline baseline; whole-network constraints must not be confused with an activation-only transplant.

**R13.** Michael Unser, Alexis Goujon and Stanislas Ducotterd. *Controlled learning of pointwise nonlinearities in neural-network-like architectures* (Applied and Computational Harmonic Analysis, 2025). [Publisher/DOI][R13]. Publisher abstract inspected for slope constraints and representer-framework scope; detailed empirical claims are not imported.

**R14.** Alejandro Molina, Patrick Schramowski and Kristian Kersting. *Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks* (ICLR 2020; preprint began in 2019). [Paper][R14]. Rational-function comparator and broad learned-shape precedent; current implementation variants are distinguished below.

**R15.** Ziming Liu and colleagues. *KAN: Kolmogorov–Arnold Networks* (2024, subsequently revised). [Paper and revision history][R15]. Adjacent edge-function architecture; not a same-architecture activation-only baseline.

**R16.** Xingyi Yang and Xinchao Wang. *Kolmogorov-Arnold Transformer* (ICLR 2025). [Paper][R16]. Grouped rational/KAN-style transformer work; architecture and kernel differences limit direct comparability.

**R17.** Mathieu Blondel, Masakazu Ishihata, Akinori Fujino and Naonori Ueda. *Polynomial Networks and Factorization Machines: New Insights and Efficient Training Algorithms* (ICML 2016). [Proceedings][R17]. Established low-rank interaction modeling and motivation for explicit interaction controls.

**R18.** Zhang, Nie and Tang. *Learning Polynomial Activation Functions for Deep Neural Networks* (TaG-DS 2025 proceedings, published in PMLR volume 321 in 2026). [Proceedings][R18]. Publisher abstract inspected; polynomial optimization is not ordinary large-scale gradient-training evidence.

**R19.** Rafał Surdej, Michał Bortkiewicz, Alex Lewandowski, Mateusz Ostaszewski and Clare Lyle. *Balancing Expressivity and Robustness: Constrained Rational Activations for Reinforcement Learning* (CoLLAs 2026). [Proceedings][R19]. Abstract-level evidence of closely related expressivity/stability tradeoffs and structural constraints in different tasks.

**R20.** K-H-Ismail. *torchortho*, official implementation repository. [Repository][R20]. GPL-3.0 and source structure inspected. This plan's Hermite inspection uses commit aa0822da6b8aa5a4294e89485fde75dfaab8478d; the exact module is [R46].

**R21.** Biomedical Imaging Group. *DeepSplines*, official standalone spline modules. [Repository][R21]. Inspected commit 654a8bd24b77433618169fc1e8607df123946302 and MIT license. This modern module is distinguished from the full experimental code accompanying older papers; the exact module is [R47].

**R22.** RCPSL. *DeepBern-Nets*, official repository. [Repository][R22]. BSD-3-Clause code license and available implementation inspected; no local reproduction or equivalence to the 2026 follow-on implementation is claimed.

**R23.** ML Research. *rational_activations*, official rational-activation package. [Repository][R23]. Inspected commit 2496d918d924b117ef06ec35bdca7f207720c62c. Its LICENSE file is MIT; a conflicting package classifier must not replace the actual license text. Source and initializer assets are [R48].

**R24.** Daehwa Ko, JaeHyeon Kim, SeungHyun Ham and Jay Hoon Jung. *TESLA: Taylor Expansion of Sinusoidal Learnable Activations* (AISTATS 2026). [Proceedings][R24]. Abstract-level context for degree/frequency allocation and constrained learned nonlinearities, not direct evidence for polynomial-tail regression.

**R25.** UCI Machine Learning Repository. *Airfoil Self-Noise*, dataset 291. [Official record][R25]; [dataset DOI](https://doi.org/10.24432/C5VW2C). Source schema, units and CC BY 4.0 record; data downloaded and integrity/group checks executed.

**R26.** UCI Machine Learning Repository. *Concrete Compressive Strength*, dataset 165. [Official record][R26]; [dataset DOI](https://doi.org/10.24432/C5PK67). Source schema, units and CC BY 4.0 record; data downloaded and grouped by mixture before splitting.

**R27.** UCI Machine Learning Repository. *Superconductivty Data*, dataset 464 (spelling retained from the source). [Official record][R27]; [dataset DOI](https://doi.org/10.24432/C53P47). Data, companion composition table and CC BY 4.0 record; alignment and grouping checks executed.

**R28.** Kam Hamidieh. *A data-driven statistical model for predicting the critical temperature of a superconductor* (Computational Materials Science, 2018). [Publisher/DOI][R28]; [author preprint](https://arxiv.org/abs/1803.10260). Primary source for target meaning and the limitation to known superconductors. Published performance is not reused as a comparable score.

**R29.** Yandex Research. *TabM*, official implementation. [Repository][R29]; [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210). Current practical MLP-based competitor; member-loss semantics, supported preprocessing/embeddings and code license inspected. Freeze the actual version after local eligibility checks.

**R30.** CatBoost documentation. *Python package installation*. [Official instructions][R30]. Documents supported macOS arm64 distribution. Installation support does not substitute for executing this project's CPU training path.

**R31.** PriorLabs. *TabPFN*, official implementation repository. [Repository][R31]. Research-date generation and local-use options; weights, code and runtime eligibility are separate gates.

**R32.** Prior Labs. *TabPFN 3.5 model license*. [Exact generation's license][R32]. Noncommercial/nonproduction terms were inspected at the research date. Archive the exact accepted terms and checkpoint hash; do not assume that the code's license grants the same rights to weights.

**R33.** Apple. *MacBook Pro (16-inch, Nov 2023) technical specifications*. [Official specification][R33]. Evidence for multiple M3 Max memory configurations, not a statement about the user's actual RAM or throughput.

**R34.** PyTorch documentation. *MPS backend*. [Official documentation][R34]. Backend usage and precision/platform caveats. The linked main documentation is mutable; the installed release and tested graph control eligibility.

**R35.** PyTorch. *torch.mps API implementation*. [Official source][R35]. Synchronization and allocator/driver-memory observation APIs; check exact installed-version availability and units before using them.

**R36.** PyTorch documentation. *MPS Environment Variables*. [Official documentation][R36]. Allocator and fallback controls; no permission to conceal mixed-device execution or disable memory safeguards.

**R37.** PyTorch documentation. *Reproducibility*. [Official documentation][R37]. Limits of seeds/determinism across versions and devices; environment/RNG logging requirements.

**R38.** PyTorch documentation. *AdamW*. [Official API documentation][R38]. Optimizer parameter groups and execution options. Validate the actual installed constructor before freezing the unfused/non-foreach path.

**R39.** PyTorch tutorials. *Saving and Loading Models*. [Official tutorial][R39]. Checkpoint/state-dictionary background. This plan adds sampler, RNG, protocol and timing state needed for its own restart contract.

**R40.** Gavin C. Cawley and Nicola L. C. Talbot. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation* (JMLR, 2010). [Paper][R40]. Motivation for protected evaluation, complete tuning accounting and restrained interpretation after selection.

**R41.** Yoshua Bengio and Yves Grandvalet. *No Unbiased Estimator of the Variance of K-Fold Cross-Validation* (JMLR, 2004). [Paper][R41]. Warning against simplistic variance claims from dependent resampling; it is not the derivation of this plan's conditional seed interval.

**R42.** Garrett Bingham and Risto Miikkulainen. *Artificial neural networks with customized activation functions*, published application WO2022035948A1 (2022; priority recorded in 2020). [Published patent record][R42]. Limited prior-art screen only. No legal status, infringement, freedom-to-operate or patentability conclusion is drawn.

**R43.** Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas and Frank Hutter. *TabArena: A Living Benchmark for Machine Learning on Tabular Data* (NeurIPS Datasets and Benchmarks, 2025). [Paper and revision history][R43]. Context for strong current baselines, ensembling and selection sensitivity; broad leaderboard numbers are not a controlled local comparison.

**R44.** PriorLabs. *TabPFN changelog*. [Official changelog][R44]. Research-date package chronology and reported MPS/cached-inference correctness fixes. A changelog entry is not proof of correctness for this task.

**R45.** Apple Developer. *Accelerated PyTorch training on Mac*. [Official guidance][R45]. Platform/setup reference cross-checked against actual release compatibility. Changing documentation versions are why the environment is discovered and locked rather than guessed.

**R46.** K-H-Ismail. *torchortho/hermite_activation.py*, immutable source snapshot. [Pinned module][R46]. Commit aa0822da6b8aa5a4294e89485fde75dfaab8478d; inspected module SHA-256 00ab8cde719909dc57dec8604917e4f7933b4177d14cb0dc02afebdf38f65217. Evidence for native initialization, basis convention and source-level concerns that still require executed parity tests.

**R47.** Biomedical Imaging Group. *spline_module/linearspline.py*, immutable source snapshot. [Pinned module][R47]. Commit 654a8bd24b77433618169fc1e8607df123946302. Evidence for the forward slope transform, TV2 normalization and extrapolation semantics.

**R48.** ML Research. *rational/torch/rational_pytorch_functions.py*, immutable source snapshot. [Pinned functions][R48]; [pinned initializer assets](https://github.com/ml-research/rational_activations/blob/2496d918d924b117ef06ec35bdca7f207720c62c/rational/rationals_config.json). Commit 2496d918d924b117ef06ec35bdca7f207720c62c. Evidence for separate A/B equations, device-placement concerns and exact degree-(5,4) assets.

[R01]: https://doi.org/10.1007/978-1-4471-2063-6_108
[R02]: https://pubmed.ncbi.nlm.nih.gov/16121724/
[R03]: https://proceedings.mlr.press/v54/hou17a.html
[R04]: https://arxiv.org/html/2502.01247v3
[R05]: https://ojs.aaai.org/index.php/AAAI/article/view/30117
[R06]: https://arxiv.org/html/2605.02591v1
[R07]: https://bigwww.epfl.ch/publications/aziznejad2001.html
[R08]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7811889/
[R09]: https://dominatorx.github.io/files/22WSDM-p.pdf
[R10]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0306420
[R11]: https://arxiv.org/html/2602.04264v2
[R12]: https://www.jmlr.org/papers/v25/22-1347.html
[R13]: https://doi.org/10.1016/j.acha.2025.101764
[R14]: https://arxiv.org/abs/1907.06732v3
[R15]: https://arxiv.org/abs/2404.19756
[R16]: https://arxiv.org/abs/2409.10594
[R17]: https://proceedings.mlr.press/v48/blondel16.html
[R18]: https://proceedings.mlr.press/v321/zhang26a.html
[R19]: https://proceedings.mlr.press/v330/surdej26a.html
[R20]: https://github.com/K-H-Ismail/torchortho
[R21]: https://github.com/Biomedical-Imaging-Group/DeepSplines
[R22]: https://github.com/rcpsl/DeepBern-Nets
[R23]: https://github.com/ml-research/rational_activations
[R24]: https://proceedings.mlr.press/v300/ko26a.html
[R25]: https://archive.ics.uci.edu/dataset/291/airfoil%2Bself-noise
[R26]: https://archive.ics.uci.edu/dataset/165/concrete%2Bcompressive%2Bstrength
[R27]: https://archive.ics.uci.edu/dataset/464/superconductivty%2Bdata
[R28]: https://doi.org/10.1016/j.commatsci.2018.07.052
[R29]: https://github.com/yandex-research/tabm
[R30]: https://catboost.ai/docs/en/installation/python-installation-method-pip-install
[R31]: https://github.com/PriorLabs/TabPFN
[R32]: https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE
[R33]: https://support.apple.com/en-us/117736
[R34]: https://docs.pytorch.org/docs/main/notes/mps.html
[R35]: https://github.com/pytorch/pytorch/blob/main/torch/mps/__init__.py
[R36]: https://docs.pytorch.org/docs/stable/mps_environment_variables.html
[R37]: https://docs.pytorch.org/docs/main/notes/randomness.html
[R38]: https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html
[R39]: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
[R40]: https://www.jmlr.org/papers/v11/cawley10a.html
[R41]: https://www.jmlr.org/papers/v5/grandvalet04a.html
[R42]: https://patents.google.com/patent/WO2022035948A1/en
[R43]: https://arxiv.org/abs/2506.16791v4
[R44]: https://github.com/PriorLabs/TabPFN/blob/main/CHANGELOG.md
[R45]: https://developer.apple.com/metal/pytorch/
[R46]: https://github.com/K-H-Ismail/torchortho/blob/aa0822da6b8aa5a4294e89485fde75dfaab8478d/torchortho/hermite_activation.py
[R47]: https://github.com/Biomedical-Imaging-Group/DeepSplines/blob/654a8bd24b77433618169fc1e8607df123946302/spline_module/linearspline.py
[R48]: https://github.com/ml-research/rational_activations/blob/2496d918d924b117ef06ec35bdca7f207720c62c/rational/torch/rational_pytorch_functions.py
