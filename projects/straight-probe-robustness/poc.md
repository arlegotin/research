# Research and proof-of-concept plan: straight-probe robustness with finite probe sources and preserved semantics

**Research date:** 28 October 2026.  
**Target execution environment:** one MacBook Pro with an M3 Max; actual memory, storage, macOS, and Python configuration must be discovered.  
**Deliverable status:** a researched mathematical handoff, including two newly derived finite-source theorems and their proofs, targeted checks performed during this investigation, and an executable plan for independent local reproduction. Historical originality and practical quantization value are not established.

## 1. Executive assessment and selected direction

### 1.1 Recommendation

Pursue the **exact finite-probe-height extension of the PDF's straight-probe minimax problem while retaining its full semantic promise on the protected horizontal strip**. Include a second, weaker compact-semantic problem as a diagnostic control. The primary PoC is a mathematical characterization, an exact separation from all additively separable readouts, and an independent certificate/comparison package. It needs no training data, pretrained weights, GPU framework, or paid service.

The decision is **go for a bounded theorem, additive-class separation, and independent reproduction project; no-go for claiming superiority over the strongest simple expressive baseline, established unrestricted extension methods, or quantization efficacy**. Both an interval extension and the elementary varying-coefficient readout \(G(u,v)=\mu[k(u)v+b(u)]\) reach the primary optimum. A correct, potentially new geometric theorem or a well-established replication is a legitimate outcome.

The supplied report studies the probe

\[
T_t(x,y)=(x+t,y+xt)
\]

while requiring a readout to preserve vertical distinctions. Its exact strip theorem is a credible starting point: the proof's coordinate transformation, permitted-edge accounting, lower bounds, and continuous upper construction survived the present algebraic audit. Its numerical claims were not independently reproduced from its original code because that code was not attached. The original problem already has an exact optimum; no method can improve that objective without changing an assumption. [P0]

The explicit refinement is to bound the height of evaluated probe bases. In straightened coordinates

\[
u=x,\qquad v=y-x^2/2,
\]

use the probe-source domain \(D=[-L,L]\times[0,H]\) and retain **every** probe based in \(D\). Targets may leave \(D\). The evaluated source set is a parabolic band in the original coordinates, not an ordinary rectangular calibration box. Two contracts are deliberately kept separate:

| Contract | Semantic lower slope | Role |
|---|---|---|
| `protected_strip_v1` | At every height \(v\in\mathbb R\), whenever \(\lvert u\rvert\le L\); exactly the PDF's original semantic promise | **Primary** |
| `compact_parabolic_slab_v1` | Only between heights within \([0,H]\), whenever \(\lvert u\rvert\le L\) | Weaker companion/control |

For the primary contract, the new exact value is

\[
\boxed{R_{\rm protected}(L,r,H;\mu)=\mu\min\{S(L,r),\max(I(L,r),H/2)\}.}
\tag{P1}
\]

Every continuous **additively separable** readout \(G(u,v)=a(v)+b(u)\) satisfying that same full-height strip promise has optimum \(\mu S\). Thus a strict separation exists exactly when \(L>0,r>3L,H<2S\). Interactions can be necessary in these fixed coordinates, but the elementary form \(k(u)v+b(u)\) already exhausts their benefit. The complete proof, the additive-class lower bound, and a full-plane semantic negative control are in Sections 4.7–4.9. This is a statement about the explicitly defined class in straightened coordinates, not a coordinate-independent architecture ranking, a requirement for complex nonlinear models, or a neural-training result.

The weaker companion has a different exact value, proved in Sections 4.1–4.5:

\[
\boxed{R_{\rm compact}(L,r,H;\mu)
=\mu\min\{H,S(L,r),\max(I(L,r),H/2)\},}
\tag{1}
\]

where

\[
I(L,r)=\min(r^2/2,2L^2)
\]

and

\[
S(L,r)=
\begin{cases}
r^2/2,&0\le r\le2L,\\
2L^2,&2L\le r\le3L,\\
L(r-L),&r\ge3L.
\end{cases}
\tag{2}
\]

Set \(S(0,r)=0\), including at the degenerate boundaries. The minimum in (1) is attained by a continuous function on the whole plane. The proof applies to arbitrary continuous nonlinear readouts, not just a fitted polynomial or a sampled mesh.

### 1.2 What is and is not an improvement

For the primary problem at \(L=1,r=4,H=3,\mu=1\), every additive readout has worst defect at least 3, while the varying-coefficient and interval readouts attain 2. At \(L=1,r=5,H=4\), the corresponding comparison is 4 versus 2. These are **analytical values**, not measured application accuracy. In the weaker compact companion, plain clipping also has defects 3 and 4, respectively, while its optimum has the same values 2 and 2 in these examples. Clipping is inadmissible for the primary contract.

However, a stronger simple baseline resolves the apparent need for a new architecture. The complete saturation-plus-potential family

\[
G_h(u,v)=\mu[\operatorname{clip}(v,0,H)+h(u)]
\]

has the exact optimum

\[
e_{\rm sat}=\max\{\min(H,I),\tfrac12\min(H,r^2/2)\}
\tag{3}
\]

in units before multiplying by \(\mu\). Choosing the better of this family and the PDF's strip optimizer attains (1):

\[
R_{\rm compact}=\mu\min(S,e_{\rm sat}).
\tag{4}
\]

**For the compact-semantic companion, this strongest simple portfolio already achieves the unrestricted optimum.** Its saturation members violate the primary contract outside the evaluated height range, so they must not be reported as valid primary comparators. For the primary contract, mandatory strong baselines are the elementary varying-coefficient model and unrestricted minimax/Lipschitz extension. Both attain (P1). The potential contributions are the exact frontiers, the primary additive-class separation, and the universal lower certificates. Varying-coefficient models are established [R21, R22]; neither their form nor saturation, potential fitting, or minimax interval centering constitutes a new method here.

Both compact portfolio members are additively separable, so interaction is unnecessary for that weaker contract. The primary full-height promise changes the conclusion: coupling the coordinates outside the protected strip can be necessary to beat the entire additive class. The exact closed-form comparators in this handoff are **derived in this investigation**, not asserted to be published off-the-shelf implementations. The general extension mechanism is established prior art.

The primary interval optimizer equals \(\mu v\) on the entire protected strip, at all heights. The compact saturating optimizer equals \(\mu v\) on its source domain. Their gains come from admissible behavior at off-source targets; the primary interaction occurs outside the protected horizontal strip. Neither result demonstrates a learned source representation or an improvement to a real quantizer or neural network.

### 1.3 Evidence status

| Item | Status at handoff |
|---|---|
| Original strip and global proof arguments | Reconstructed and challenged; no material algebraic/domain error found. Not proof-assistant verified. |
| Protected-strip theorem, additive separation, compact theorem, and optimized saturation family | Derived here; complete arguments supplied below and checked separately during this investigation. |
| Numerical corroboration | Exact rational checks, independent floating implementations, 14 compact finite LPs and 22 primary/additive finite LPs executed on Linux x86_64. Details in Section 5. |
| M3 Max execution | Not performed. Native CPU packages are available; target-machine checks precede dependent work. |
| Originality | No exact match located in targeted searches. Closely related optimization and approximation machinery is established. Historical priority remains unresolved. |
| Competitive algorithmic gain | Primary improvement over the complete additive class; no improvement over a simple varying-coefficient readout or established unrestricted extension. Compact portfolio is already optimal. |
| Practical quantization gain | Unverified; no quantizer-to-probe mapping or application dataset has been supplied or established. |

### 1.4 Research direction selection

| Direction | Value and distinctiveness | Local feasibility | Decision |
|---|---|---|---|
| Reproduce the original unbounded strip | Necessary control; already solved in the supplied report | Very small | Required baseline, not a new efficacy claim |
| Extend the global polynomial/Elfving calculation | Useful control; central extremal polynomials are known [R1] | Small | Reuse selected identities; do not rebuild a design-theory project |
| Finite-height probe bases, original full-height strip semantics | Preserves the PDF's semantic promise; exact frontier and separation from every additive readout | Scalar arithmetic and small certificates | **Primary** |
| Finite-height semantics as well as probe bases | Reveals how weakening the semantic contract erases the interaction requirement | Same machinery | Required companion/control |
| Ordinary rectangle in original \((x,y)\) coordinates | Tests dependence on the convenient geometry; its transformed vertical intervals vary with \(u\) | Small diagnostic first, possibly a larger certified approximation problem | Conditional follow-on after the primary result is settled |
| Asymmetric probes or nonlinear disturbance fields | May change active certificates substantially | Depends on a separately specified model | Pursue only when the intended application supplies a reason |
| Neural-network quantization | Potentially consequential, but currently disconnected from the specified objective | Model-dependent and unnecessary for this theorem | No model downloads or training in the core PoC |
| The PDF's separate rounding appendix | A different problem with different baselines | Locally testable | Record as out of scope; do not silently substitute it |

The coding agent may make routine, non-destructive implementation decisions. Changes to the domain, probe set, semantic condition, primary metric, comparator information, or interpretation require a written methodological amendment and a fresh confirmation record.

## 2. Complete problem statement, assumptions, and hypotheses

### 2.1 Original mechanism and the actual limitation

An infinitesimal disturbance direction can be transverse to the meaningful direction without allowing finite invariance. Here the vector field is \((1,x)\). The function \(y-x^2/2\) is constant along its exact flow, but an Euler-like straight probe changes it by \(-t^2/2\). The exact flow would instead be \((x+t,y+xt+t^2/2)\), a different map.

Under the polynomial coordinate change, the actual probe is

\[
S_t(u,v)=(u+t,v-t^2/2).
\tag{5}
\]

The maps commute, but \(S_{-t}\) is not \(S_t^{-1}\). Their discrepancy is finite-step drift, not a claim about noncommuting controls. In fact,

\[
S_t^{-1}(u,v)=(u-t,v+t^2/2),\qquad
S_{-t}S_t(u,v)=(u,v-t^2).
\]

First-integral preservation and special straight-flow maps have established geometric/algebraic antecedents. The exact identities above are direct calculations for this model. [R9, R10]

The observed mathematical bottleneck is that different allowed probes and semantic comparisons can impose conflicting requirements on the **same readout value**. Some original obstruction configurations require probe bases outside the evaluated height range. Restricting those bases can reduce the floor while retaining the primary semantic promise. Whether this is a bottleneck in an actual application is unknown.

### 2.2 Definitions, two semantic contracts, and units

Inputs are \(L,r,H\ge0\) and \(\mu>0\). Let

\[
D_{L,H}=\{(u,v): -L\le u\le L,\ 0\le v\le H\}.
\]

The compact companion's admissible class consists of all continuous \(G:\mathbb R^2\to\mathbb R\) such that

\[
G(u,v_2)-G(u,v_1)\ge\mu(v_2-v_1)
\quad\text{for }|u|\le L,\ 0\le v_1\le v_2\le H.
\tag{6}
\]

Define

\[
E_{L,r,H}(G)=\sup_{(u,v)\in D_{L,H},\ |t|\le r}
|G(u+t,v-t^2/2)-G(u,v)|,
\tag{7}
\]

and minimize (7) over (6). A single globally defined function must supply both source and target values. Identical points must have identical values even when reached by different probes. There is no target deletion, target projection, or clipped disturbance map.

For the **primary protected-strip contract**, replace the height restriction in (6) by all real \(v_1\le v_2\), while leaving (7) unchanged. Call this condition (6P). Probe-source eligibility and semantic-pair eligibility are different predicates. A target with \(|u|\le L\) and negative height is semantically protected under (6P), but it is not an eligible next probe base unless its height lies in \([0,H]\). A horizontal target with \(|u|>L\) has no required lower slope in either contract.

This is **one-step robustness relative to the stated source set**. No uniform guarantee for repeated probes or arbitrary subsequently reached states follows. If an intermediate point is outside \(D\), its next outgoing probe is outside the evaluated population, even though its readout value is defined.

For \(\mu>0,H>0\), (6) makes each source vertical restriction strictly increasing. A deterministic readout that factors through a finite codebook cannot satisfy that condition on a whole vertical interval: distinct heights would eventually share a code/output. This directly excludes using a finite-level scalar quantizer as \(G\) itself. Quantized model weights with continuous outputs are a different setting; neither this elementary obstruction nor the present robustness result establishes their behavior.

The units of \(u,t,L,r\) agree; \(v,H,I,S\) have their squared units. \(\mu\) converts semantic-coordinate units to output units. The primary metric is worst absolute output defect. For \(L>0\), use

\[
\rho=r/L,\qquad \eta=H/L^2,\qquad \mathcal E=E/(\mu L^2).
\]

The original-coordinate source set is exactly

\[
|x|\le L,\qquad x^2/2\le y\le x^2/2+H,
\tag{8}
\]

and \(F(x,y)=G(x,y-x^2/2)\). Vertical differences are preserved by the coordinate change, so (6) still means preservation of vertical distinctions. It does not require preservation of every horizontal distinction or an application-specific label.

### 2.3 Consequential assumptions

| Assumption | Why it matters | Handling |
|---|---|---|
| Symmetric continuous amplitude interval \([-r,r]\) | Both reciprocal and reflected witnesses use it | Encode explicitly; an asymmetric set is a new problem |
| Every source height in \([0,H]\) is evaluated | Sparse calibration points do not establish the same supremum or slope condition | Use the continuum proof; finite samples are diagnostics |
| Targets can be outside the source set | Exterior collisions and saturation determine the compact regimes | Preserve all endpoints exactly |
| Primary lower slope holds at all heights on \(\lvert u\rvert\le L\); companion lower slope holds only in \(D\) | This difference changes the optimum and whether additive readouts suffice | Use separate problem IDs and semantic predicates; reject inadmissible comparators |
| No fixed horizontal smoothness or complexity constraint | Allows continuous piecewise constructions | Additional regularity is a separate constrained problem |
| Known \(L,r,H,\mu\) for every method | Enables the optimal domain-dependent choice | Give identical information to all baselines; include calibration cost if these are estimated later |
| No specified practical user/data distribution | Prevents an application-level success criterion | Primary beneficiaries are researchers studying finite invariance limits and verification tools |
| Actual laptop resources unknown | Prevents reliable wall-time and peak-memory promises | Discover and calibrate before installing optional diagnostics or increasing graph size |

### 2.4 Claims to test and ways to challenge them

The mathematical statements have proofs below; the implementation program tests their faithful realization and attempts to uncover defects. They are not statistical hypotheses about a hidden population of users.

| ID | Claim/prediction | Support | Alternative explanation or failure | Discriminating evidence |
|---|---|---|---|---|
| C1 | Original strip value is \(\mu S\) | Supplied proof; reconstructed finite witnesses and upper bound | Wrong inverse/source accounting, or transcription error | Reproduce lower certificates and continuous construction independently |
| C2 | Compact value is (1) for the complete class | Two finite lower constructions plus attaining portfolio | A witness uses an illegal source or the upper fails at a seam | Validate every incidence and inequality; exhibit any counterexample rather than averaging it away |
| C3 | The optimized saturation class has value (3) | Internal witness and two-height residual separation; explicit upper | Poor tuning creates a false gap | Analytic class optimum and independently solved finite potential LP must agree where witnesses are represented |
| C4 | Full nonlinear models cannot beat (4) in the companion, or (P1) in the primary | Universal lower bounds do not assume covariance or differentiability | A finite mesh omits decisive amplitudes | A valid continuum improvement would contradict a checked finite lower certificate; inspect the harness first |
| C5 | Gains over plain clipping/strip come from off-source behavior | Saturation optimizer is exactly \(\mu v\) throughout \(D\) | Source encoding capacity or extra information causes the gain | Source-identity check, horizontal-offset ablation, and equal parameter information |
| C6 | The core can execute locally | Constant-size formula and rational witnesses; native package distributions | Missing Python/optional solver or excessive mesh construction | Target-native smoke tests and measured RSS; retain the exact arithmetic fallback |
| C7 | The result may survive a more practical source shape | No positive evidence yet | The parabolic geometry is essential | A separately specified original-coordinate rectangle diagnostic; no transfer claim before a valid upper/lower bracket |
| C8 | Primary optimum is (P1), while every additive \(a(v)+b(u)\) has optimum \(\mu S\) | Full-height internal witness, exterior pairs, interval upper, and four-residual additive cancellation | A purported gain secretly saturates protected strip heights or uses the weaker companion | Test arbitrary protected heights outside \([0,H]\), verify all four residuals, and compare with unrestricted fixed-label extension |
| C9 | A varying-coefficient readout already attains the primary optimum | Explicit coefficients and residual-interval proof in Section 4.9 | A weak expressive comparator makes a more elaborate model appear necessary | Compare the exact class attainer with (P2), retaining every equality case; no learned fitting is required |

### 2.5 Headroom and strongest case for/against the project

Let \(B=\min(H,S)\), the exact value of the better of plain clipping and the strip construction. Equation (1) gives \(B/2\le R_{\rm compact}/\mu\le B\). The largest possible advantage against this **restricted two-baseline comparison** is a factor of two, attained for example at \(r=5L,H=4L^2\). Against the optimized saturation/strip portfolio the headroom is zero.

That factor-two statement applies only to the compact companion comparison. In the primary problem, \(S/\min(S,\max(I,H/2))\) is the exact ratio against the complete additive class. For fixed \(L>0,H\), it grows without bound as \(r\to\infty\): the additive floor grows linearly while the primary value eventually stays at \(\mu\max(2L^2,H/2)\). This is an analytical property of ever more distant exterior targets, not a practical scaling claim. Floating cancellation and application relevance become more problematic at large radius.

The strongest case for the project is an explicit solution of an infinite-dimensional optimization problem, including exact phase changes, an additive-class separation, and a simple attaining function that exhausts further robustness headroom. The strongest case against broader significance is equally concrete: the geometry is constructed, the useful freedom lies outside the region where semantics are protected, the algorithm uses elementary operations, and no application has been shown to satisfy the assumptions. Formula evaluation is already constant size; no end-to-end bottleneck has been demonstrated. Both conclusions should survive into the final report.

## 3. Research findings and prior-art assessment

### 3.1 Closest alternatives and exact overlap

Technical claims here are grounded in inspected primary sources. Inspection scope and access limits are recorded in Section 12. Secondary indexes were used to locate publications, not to certify their theorems.

| Prior work | Problem/mechanism and assumptions | Evidence and computational requirements | Reuse and exact overlap |
|---|---|---|---|
| Supplied straight-probe report [P0] | Same quadratic probe; unbounded vertical strip or fully global states | Constructive proofs; reported Python checks whose source was not attached | Reuse the strip optimizer and global controls. Compact source height is the explicit change. |
| Dette, Melas, Shpilev [R1] | No-intercept polynomial regression; Elfving geometry and special extremal polynomials | Explicit support/weight formulas; low-dimensional polynomial computations | The PDF's global approximation constants are established ingredients, not a fresh Chebyshev contribution. |
| Diliberto–Straus [R2] | Uniform approximation by sums of independent coordinate functions; closed paths | Theorem and levelling construction on compact domains | Ancestry of path lower bounds. Independent coordinate functions are not the same as one shared potential. |
| Karp [R3] | Minimum cycle mean on a finite weighted digraph | Exact characterization; classical dynamic programming | Finite cycle methods are established. Mixed semantic/probe constraints require a cycle ratio, not division by all edges. |
| Krivulin [R4, R5] | Common-potential/log-Chebyshev fitting of pairwise comparisons; tropical spectral radius and minimizers | Explicit finite algebraic solutions, with a 2026 follow-up | Direct prior art against calling common-potential minimax fitting a new algorithm. Does not state this compact continuum geometry. |
| Difference-constraint networks [R6] | Feasibility of inequalities between node values | Negative-cycle characterization and shortest-path potentials | Reuse for an independent finite verifier. The new work is not LP duality or graph feasibility. |
| Kyng et al. [R7] | Minimize maximal graph gradient with fixed labels on a subset; directed variants | Algorithms and public code link; graph-sized workloads | Direct finite baseline once the proved optimal source labels \(G=\mu v\) are fixed. Unrestricted lower-slope graphs are a separate relaxation. |
| Varying-coefficient models [R21, R22] | Linear dependence on predictors with coefficients that vary with an index variable | Established statistical model class; generic fitting is unnecessary for this deterministic problem | The two-term form \(k(u)v+b(u)\) attains the primary optimum with explicit coefficients. Mandatory strongest simple expressive comparator; the model family is not new. |
| Ettehad–Foucart [R8] | Worst-case optimal recovery from bounded uncertain observations | Chebyshev-center formulations and computational constructions | Interval centering is standard. Its elementary scalar specialization explains an exterior extension, not novelty by itself. |
| de Bondt / geometric integration [R9, R10] | Quasi-translations, first integrals, exact flows and discretizations | Algebraic conditions and geometric methods | Explains why straight steps need not preserve a flow invariant; not a compact minimax formula. |
| Ok and Gargiulo–Ok [R11, R12] | Order-preserving Lipschitz extension under geometric compatibility | Positive theorems and nonextension results; current 2026 follow-up | Warns against inferring a continuous monotone extension from arbitrary feasible mesh values. Our explicit formula avoids that assumption. |
| Discrete weak KAM [R13] | Potential inequalities and critical costs on compact spaces | Established closed-measure/variational framework | Related dual mechanism, with hypotheses different from a restricted relation and partial semantic order. |
| DiscQuant [R14] | Fixed-grid neural-network rounding and network-output approximation | A practical quantization paper with linked implementation | Different objective and assets; published model scores are not comparable to (7). No practical benchmark claim is imported. |

### 3.2 Precise novelty assessment

**Established:** coordinate changes and first-integral analysis; common-potential fitting; finite difference constraints; cycle-based lower certificates; scalar minimax centering; varying-coefficient model classes; the no-intercept extremal polynomials used in the global model.

**Plausibly differentiated:** the protected value (P1), its separation from the entire additive class without weakening the original strip semantics, the compact value (1), and the universal optimality proofs. The exact source geometry, semantic scope, and exterior target policy are part of these claims. Established unrestricted extension can match the constructions, so this is a geometric characterization and class separation, not a new general extension algorithm.

**Unresolved:** whether this formula or an immediately equivalent special case has appeared under another formulation, and whether the result is considered a substantive new theorem or a short worked consequence of existing approximation/extension machinery. No search establishes the absence of prior art everywhere. This is not a patentability or freedom-to-operate opinion.

**Not supported:** a new general optimization algorithm, a need for learned nonlinear source features, neural-network compression gains, or a mathematical breakthrough designation.

### 3.3 Explicit answers to the prior-art questions

1. **What exists?** Exact strip/global solutions in the supplied report and mature approximation, potential-fitting, graph, and optimal-recovery tools in the inspected literature.
2. **What can be reused?** The strip construction, native sparse LP solvers, standard rational arithmetic, established difference-constraint algorithms, and known polynomial controls.
3. **What is substantively different?** Finite evaluated probe height changes the available obstruction while the primary retains the original full-height strip semantics. Its exact frontier and additive-versus-varying-coefficient separation are the narrow result derived here. Weakening semantic height as well yields the separate compact frontier and removes the interaction requirement.
4. **Why might that difference have value?** It identifies when an unbounded probe-base population overstates the floor, when coupling coordinates can beat every additive readout while retaining the semantic promise, and when a simple established extension already exhausts the benefit.
5. **What isolates it?** Within each contract, hold probe, source population, semantic condition, and parameter information fixed. For the primary compare the all-additive optimum with varying-coefficient, interval, and fixed-label extension baselines; for the companion include clipping and optimized saturation. Use independent legal finite witnesses for the universal floor. Changing contracts is an ablation, not a same-task superiority claim.
6. **What invalidates originality?** A prior equivalent theorem or statement with the same domain/edge/semantic assumptions and value defeats historical originality. Finding that the result is a short specialization of known machinery lowers its mathematical significance but does not by itself prove it was previously stated. Record either reduction and distinguish replication from a possibly new, modest corollary.

Additional targeted searches after selecting the finite-source refinement found established interval-centering, ordered-extension, and varying-coefficient mechanisms but no exact formula match. This was the research stopping point: the next useful work was completing and challenging the proof, not accumulating more loosely related papers. Search coverage and blind spots remain explicit in Section 12.

## 4. Mathematical specification and complete proofs

### 4.1 Original strip construction, reproduced for a standalone implementation

For sources \(|u|\le L\) and all \(v\in\mathbb R\), the minimum defect is \(\mu S(L,r)\).

For \(r\le2L\), use \(G_{\rm strip}(u,v)=\mu v\). Its defect is \(\mu t^2/2\). For a matching lower bound, take \(a=\min(r,2L)\), start at horizontal coordinate \(-a/2\), apply amplitudes \(a,-a\), and obtain the same horizontal coordinate with vertical displacement \(-a^2\). Both forward bases lie in the source strip. Two probe defects and the semantic comparison imply \(E\ge\mu a^2/2\).

For \(r\ge2L\), put \(M=\max(2L^2,L(r-L))\) and define an even continuous potential \(g\). On \(|w|\le L\), set \(g(w)=-w^2/2\). For \(L\le w\le L+r\), let

\[
a(w)=\max(-L,w-r),
\]

\[
\ell(w)=w^2/2-wa(w)-M,\qquad
b(w)=w^2/2-wL+M,
\]

\[
g(w)=\operatorname{clip}(-w^2/2,\ell(w),b(w)).
\tag{9}
\]

Extend evenly to negative \(w\), and constantly beyond \(|w|=L+r\). Use

\[
G_{\rm strip}(u,v)=\mu[v+g(u)].
\tag{10}
\]

For source \(u\) and target \(w=u+t\), the required residual becomes

\[
\left|g(w)-\left(w^2/2-wu\right)\right|\le M.
\]

Inside the strip its value is \(|w(u-w)|\le2L^2\). At an exterior target, all permissible source values lie in the interval whose radius is \(w(L-a(w))/2\). Its maximum is \((r+L)^2/8\le2L^2\) when \(2L\le r\le3L\), and \(L(r-L)\) when \(r\ge3L\). Thus (9) has ordered endpoints, is continuous, and matches the interior value at \(L\).

The second lower configuration traverses a forward \(S_r\), the inverse edge of \(S_{r-2L}\), a forward \(S_{-r}\), and the inverse edge of \(S_{-(r-2L)}\). Horizontal coordinates are

\[
-L,\ r-L,\ L,\ L-r,\ -L,
\]

and total vertical displacement is \(-4L(r-L)\). Each underlying forward edge is based at \(-L\) or \(L\). The source height is unrestricted in this original problem. Consequently \(E\ge\mu L(r-L)\). Together with the first bound this proves (2), including attainment. For \(L=0\), use \(G=\mu(v+u^2/2)\), corresponding to \(F=\mu y\). [P0]

This proof must not be reused on a compact source slab without checking its source heights. The compact lower proof below uses different finite configurations.

### 4.2 First compact lower bound: reciprocal probes plus semantic links

Let

\[
a=\min(r,2L,\sqrt{2H}),\qquad d=a^2/2=\min(H,I).
\]

Use sources \((-a/2,H)\) and \((a/2,H)\), with amplitudes \(a\) and \(-a\). Their targets are \((a/2,H-d)\) and \((-a/2,H-d)\), respectively. All four points are in \(D\). The two absolute probe bounds imply

\[
\begin{aligned}
2E&\ge G(-a/2,H)-G(a/2,H-d)\\
&\quad+G(a/2,H)-G(-a/2,H-d)\\
&\ge 2\mu d.
\end{aligned}
\]

Hence

\[
E\ge\mu\min(H,I).
\tag{11}
\]

The two vertical semantic links are essential. Requiring these probes to form a single chronological trajectory would unnecessarily demand twice the vertical space and produce a weaker bound.

### 4.3 Second compact lower bound: two common-target pairs

Assume \(L>0,r\ge2L,H\ge2L^2\). Set

\[
d=\min(H,2L(r-L)),\qquad w=d/(2L).
\]

Then \(L\le w\le r-L\). Sources \((-L,H)\) and \((L,H-d)\), with amplitudes \(w+L\) and \(w-L\), reach the identical target

\[
(w,H-(w+L)^2/2).
\]

Reflecting horizontally produces a second identical-target pair from \((L,H)\) and \((-L,H-d)\). All four amplitudes are allowed; all source heights are in \([0,H]\). Targets need not be in the source slab.

Each pair bounds its oriented source-output difference above by \(2E\). Adding the two inequalities and regrouping by source column gives

\[
4E\ge [G(-L,H)-G(-L,H-d)]
+[G(L,H)-G(L,H-d)]\ge2\mu d.
\]

Therefore

\[
E\ge\mu\min(H/2,L(r-L)).
\tag{12}
\]

For \(r\le2L\), (11) is already \(\mu\min(H,S)\). For \(r\ge2L,H\le2L^2\), it is \(\mu H\). In the remaining case, (11) and (12) give

\[
E/\mu\ge\max(2L^2,\min(H/2,L(r-L))).
\]

Splitting at \(r=3L\) and the height thresholds produces exactly the lower side of (1). No covariance, differentiability, upper slope, or finite-dimensional parameterization was assumed. Continuity is not needed for these finite lower inequalities.

### 4.4 Exact optimized saturation baseline and upper construction

Consider all continuous \(h:\mathbb R\to\mathbb R\) in

\[
G_h(u,v)=\mu[\operatorname{clip}(v,0,H)+h(u)].
\]

For fixed source horizontal coordinate \(u\) and amplitude \(t\), write \(q=t^2/2\) and \(\Delta h=h(u+t)-h(u)\). The signed residual divided by \(\mu\) is

\[
\Delta h-\min(v,q).
\]

Its exact maximum absolute value over \(0\le v\le H\) is

\[
\max\{|\Delta h|,|\Delta h-\min(H,q)|\}.
\tag{13}
\]

The universal bound (11) applies to this family. Also, take \(t=r\) and compare source heights 0 and \(H\): the two residuals differ by \(\min(H,r^2/2)\). At least one has absolute value at least half that separation. Thus the family optimum is at least (3).

To attain it, define for every real \(w\)

\[
Q(w)=\tfrac12\min(r,|w|+L)^2,\quad
D(w)=\min(H,Q(w)),\quad
h_*(w)=\max(0,D(w)-e_{\rm sat}).
\tag{14}
\]

Use

\[
G_{\rm sat}(u,v)=\mu[\operatorname{clip}(v,0,H)+h_*(u)].
\tag{15}
\]

All operations in (14)–(15) are continuous. On \(|u|\le L\),

\[
D(u)\le\min(H,I)\le e_{\rm sat},
\]

so \(h_*(u)=0\) and \(G_{\rm sat}(u,v)=\mu v\) throughout \(D\). The semantic condition holds with equality.

For a permitted target \(w=u+t,z=v-q\), one has \(|t|\le\min(r,|w|+L)\), hence

\[
0\le c:=\min(v,q)\le D(w).
\]

Also \(D(w)\le\min(H,r^2/2)\le2e_{\rm sat}\). Therefore

\[
h_*(w)\le e_{\rm sat},\qquad
D(w)-h_*(w)\le e_{\rm sat}.
\]

The residual \(h_*(w)-c\) lies in \([-e_{\rm sat},e_{\rm sat}]\). This proves the full continuum upper bound and the exact family optimum, including targets far outside the source set.

Choosing (10) or (15) **once per parameter instance**, according to the smaller certified bound, gives (4). A pointwise choice between readout values or a per-edge choice after observing the error is not this algorithm. Elementary case analysis makes (4) equal to (1), matching the universal lower proof.

For completeness, write \(Q=r^2/2\). The needed inequalities are \(I\le S\le Q\) and \(S\le\max(I,Q/2)\). The latter follows from \(S=I\) for \(r\le2L\), and from \(L(r-L)\le r^2/4\) otherwise. If \(H\le I\), both expressions equal \(H\). If \(I<H\le Q\), (3) becomes \(\max(I,H/2)\le H\), giving (1). If \(H\ge Q\), (3) is \(\max(I,Q/2)\ge S\), while every entry other than \(S\) in (1) is at least \(S\). Both expressions then equal \(S\). This proves the portfolio identity without a numerical fit.

### 4.5 Compact-companion phase diagram and degenerate cases

For \(r\le3L\), the compact value is \(\mu\min(H,S)\). For \(r>3L\):

| Source height | Exact defect divided by \(\mu\) | Active explanation |
|---|---:|---|
| \(0\le H\le2L^2\) | \(H\) | Internal reciprocal probes span the meaningful height |
| \(2L^2\le H\le4L^2\) | \(2L^2\) | Longest internal horizontal separation |
| \(4L^2\le H\le2L(r-L)\) | \(H/2\) | Common-target pairs use the available semantic height |
| \(H\ge2L(r-L)\) | \(L(r-L)\) | Full exterior obstruction recovered |

At shared boundaries the formulas agree. \(H\) is the **full vertical width**, never a half-height.

If \(H=0\), (6) is vacuous except at equal points and a constant readout gives zero. If \(r=0\), all probes are identities. If \(L=0\), \(G(u,v)=\mu(v+u^2/2)\) gives exact invariance for probes based at \(u=0\). Handle these cases before divisions or normalized coordinates. Negative, infinite, or NaN parameters are invalid inputs; \(\mu=0\) is excluded from the stated problem, though its limiting value is zero.

Global exact vertical covariance forces \(G=\mu[v+g(u)]\); its residual is independent of \(v\), so the optimal value remains \(\mu S\) even when probe bases have a finite, nonempty height interval. Source-only covariance is different: (15) already has it and can achieve the lower compact value. Do not conflate these conditions.

### 4.6 Optional global controls from the supplied report

These validate the original mechanism; they are not additional contribution targets. In dimension \(d\), set \(x_0=1\), \((T_tx)_j=x_j+t x_{j-1}\), and

\[
\kappa_j(x)=[z^j]\log(1+x_1z+\cdots+x_dz^d).
\]

All operations are formal polynomials truncated modulo \(z^{d+1}\). Then

\[
\kappa_j(T_tx)-\kappa_j(x)=(-1)^{j+1}t^j/j.
\]

For global states, the optimum reduces to the minimum sup-norm of a polynomial with no constant term and leading coefficient \((-1)^{d+1}/d\). The known no-intercept approximation/design literature supplies the extremal family. [P0, R1]

For the smallest nontrivial higher-dimensional check, use

\[
G_3=\mu(\kappa_3-r^2\kappa_1/4),\quad
\kappa_3=x_3-x_1x_2+x_1^3/3.
\]

Its residual is \(\mu(t^3/3-r^2t/4)\), whose maximum on \([-r,r]\) is \(\mu r^3/12\). Check endpoints and derivative roots \(t=\pm r/2\). The signed amplitude counts \((1,-2,2,-1)\) at \((r,r/2,-r/2,-r)\) cancel lower coordinates and yield displacement \(r^3/2\) in the third coordinate over six traversed edges. This is a global control; it is not automatically an admissible compact-domain trajectory.

### 4.7 Primary theorem: finite probe bases, full-height protected-strip semantics

Keep (7), but impose

\[
G(u,v_2)-G(u,v_1)\ge\mu(v_2-v_1)
\quad\text{for every }|u|\le L,\ v_1\le v_2\text{ in }\mathbb R.
\tag{6P}
\]

Let \(T=\min(S,\max(I,H/2))\). Then the exact minimum is \(\mu T\), attained continuously. This contract preserves the supplied PDF's semantic requirement and changes only the evaluated probe bases.

**Universal lower bound.** Set \(a=\min(r,2L)\), \(q=a^2/2=I\). Use two reciprocal probes from \((-a/2,0)\) and \((a/2,0)\) to \((a/2,-q)\) and \((-a/2,-q)\). Their source heights are allowed for every \(H\ge0\), and their target-to-source semantic links are legal under (6P), even below zero. Summing gives \(E\ge\mu I\). When an exterior bound exceeds this, the common-target proof of Section 4.3 applies and gives \(E\ge\mu\min(H/2,L(r-L))\). If \(r\le3L\), \(S=I\). If \(r>3L\), combining these bounds gives

\[
E/\mu\ge\max(I,\min(H/2,S))
=\min(S,\max(I,H/2))=T.
\]

**Single attaining interval extension.** For all real \(w,z\), define

\[
q_-(w)=\tfrac12\max(|w|-L,0)^2,\qquad
q_+(w)=\tfrac12\min(r,|w|+L)^2,
\]

\[
a(w,z)=\max(0,z+q_-(w)),\qquad
b(w,z)=\min(H,z+q_+(w)),
\]

\[
\boxed{G_{\rm protected}(w,z)
=\mu\operatorname{clip}(z,b(w,z)-T,a(w,z)+T).}
\tag{P2}
\]

For reachable horizontal targets, the permitted squared drift values fill \([q_-,q_+]\). For a fully reachable target \((w,z)\), the admissible predecessor heights fill \([a,b]\). At all horizontal coordinates, including unreachable ones,

\[
q_+(w)-q_-(w)\le2S.
\tag{P3}
\]

To check (P3), inside the strip the difference is at most \(I\). For \(w\ge L\), if \(r\le2L\), the difference is at most \(r^2/2=S\). If \(r\ge2L\), it is \(2Lw\) on \([L,r-L]\), then decreases from \(2L(r-L)\) on \([r-L,L+r]\); beyond reach it is nonpositive. Reflection handles negative \(w\). Thus it is bounded by \(2S\).

Consequently \(b-a\le\min(H,2S)\le2T\), so the clamp interval in (P2) is ordered everywhere, even if \([a,b]\) is empty. The formula is globally continuous. On \(|w|\le L\), \(q_-=0,q_+\le I\le T\). For every real \(z\), \(b-T\le z\le a+T\); hence \(G_{\rm protected}(w,z)=\mu z\) at **all** protected heights. This verifies (6P), not merely a finite set of sampled slopes.

For an evaluated edge, the source output is \(\mu v\) and \(v\in[a(w,z),b(w,z)]\). Every point in \([b-T,a+T]\) is within \(T\) of every such \(v\). Thus the edge defect is at most \(\mu T\), matching the lower bound.

**Exact additive-class optimum.** Let \(G(u,v)=A(v)+B(u)\) be continuous and satisfy (6P). Then \(A(v_2)-A(v_1)\ge\mu(v_2-v_1)\) for all real heights. For \(r\ge2L\), set \(w=r-L\), \(q_1=r^2/2\), \(q_2=(r-2L)^2/2\), and use the four probes

\[
(-L,0,r),\ (L,0,r-2L),\ (L,0,-r),\ (-L,0,-(r-2L)).
\]

Here each triple lists source horizontal coordinate, source height, and amplitude. Let their signed output residuals be \(d_1,d_2,d_3,d_4\). Additivity cancels both exterior and source horizontal terms:

\[
d_1-d_2+d_3-d_4
=2[A(-q_1)-A(-q_2)]
\le-4\mu L(r-L).
\]

Since each \(|d_i|\le E\), this forces \(E\ge\mu L(r-L)\). The internal bound supplies \(\mu I\); the strip optimizer attains their maximum \(\mu S\). For \(r\le2L\), the internal bound already equals \(\mu S\). This proves the optimum for **all additive \(A(v)+B(u)\)**, including nonlinear \(A\); comparing only with linear or quadratic representatives would be weaker.

Strict improvement therefore occurs iff \(L>0,r>3L,H<2S\). This does not establish a restriction on multiplicative separability, finite-rank tensor representations, or any unspecified architecture class. It establishes that a single additive sum cannot attain the smaller value under the same full-height strip semantics.

**Semantic-scope negative control.** If instead the lower slope is required at every horizontal coordinate in the plane, the preceding four-residual argument works for arbitrary \(G\): compare the two target heights at \(w\) and at \(-w\), where the now-global vertical lower slope applies. Source differences cancel. The optimum returns to \(\mu S\). Thus the primary gain relies on freedom at horizontal target columns outside the protected strip; it does not preserve the lower slope throughout the whole plane.

**Degenerate cases.** \(H=0\) gives \(R_{\rm protected}=\mu I\), generally positive. This differs from the compact companion's zero. \(L=0\) or \(r=0\) gives zero. Formula (P2) handles these cases without dividing by \(L\), but dedicated fixtures must check them.

### 4.8 Simpler strip optimizer and the established extension baseline

The same range argument yields a simpler alternative to (9):

\[
\widetilde g(w)=\max(0,q_+(w)-S),\qquad
\widetilde G_{\rm strip}(u,v)=\mu[v+\widetilde g(u)].
\tag{P4}
\]

On the source strip \(q_+\le I\le S\), so this readout is exactly \(\mu v\). For every reachable target, (P3) gives
\(q_+-S\le q_-+S\). Therefore \(\widetilde g(w)\) lies within \(S\) of every possible \(q=t^2/2\), proving defect at most \(\mu S\). Formula (P4) is globally continuous and vertically covariant. It also handles \(L=0\) and \(r=0\). Reproduce the PDF's construction as a control, but prefer this simpler expression for the primary implementation.

Once optimal protected-strip values are fixed at \(G=\mu v\), the remaining task is a standard minimax extension: choose each exterior target value to stay close to all incident labeled source values. Its interval center/feasible-interval rule is established optimal-recovery machinery [R8]. On a finite probe graph with unit edge lengths and fixed strip labels, it is directly an infinity-norm Lipschitz extension problem of the type studied by Kyng et al. [R7]. These methods are a mandatory strong comparison, not merely background. They can match (P2); the narrow contribution candidate is evaluating this geometry exactly and proving that fixing these strip labels loses nothing compared with the full lower-slope class.

### 4.9 A simple varying-coefficient readout already attains the primary optimum

The primary separation does not require a complicated joint function. The established varying-coefficient form [R21, R22]

\[
G_{\rm affine}(w,z)=\mu[k(w)z+b(w)]
\tag{P5}
\]

already achieves \(\mu T\). Its exact coefficients are supplied here so that the strongest simple expressive comparator cannot be handicapped by weak fitting or tuning.

Set \(e=T\), with \(q_-,q_+\) as in Section 4.7. If \(e=S\), take \(k=1\) and \(b=\max(0,q_+-S)\), recovering (P4). Otherwise define

\[
D(w)=\max(0,q_+(w)-q_-(w)),\qquad
k(w)=\operatorname{clip}\!\left(\frac{2e-D(w)}{2e-I},0,1\right),
\]

\[
b(w)=\max\{0,(1-k(w))H+k(w)q_+(w)-e\}.
\tag{P6}
\]

The strict branch \(e<S\) implies \(L>0\), \(I>0\), \(2e-I>0\), and \(H\le2e\); test the equality branch first to avoid division in degenerate cases. All coefficients are continuous. On the protected strip, \(D=q_+\le I\), so \(k=1,b=0\) and (P5) equals \(\mu z\) at **every** height.

For a permitted edge ending at horizontal coordinate \(w\), write \(q=t^2/2\). The normalized residual is

\[
b(w)-[(1-k(w))v+k(w)q].
\]

The bracket lies between \(A=kq_-\) and \(B=(1-k)H+kq_+\). If \(k=0\), its width is \(H\le2e\). If \(k>0\), (P6) gives \(D<2e\), and the width \((1-k)H+kD\le2e\). Since \(A\ge0\), \(b=\max(0,B-e)\) belongs to \([B-e,A+e]\). Every residual therefore has magnitude at most \(e\). The universal lower bound makes this exact.

Thus the benefit of allowing interactions is completely realized by a function affine in the semantic coordinate, with two scalar functions of the horizontal coordinate. There is **no robustness headroom over this comparator**. A neural model, generic spline, or unrestricted LP could match it but cannot improve the same continuum objective. This is a model-class characterization; the coefficient formulas and the geometry-specific proof are the potential contribution, not the varying-coefficient form itself.

For numerical evaluation, the range width may be formed as \(D=\max(0,(p-m)(p+m)/2)\), where \(p=\min(r,|w|+L)\) and \(m=\max(|w|-L,0)\), to avoid subtracting two large squares. This rearrangement does not certify floating accuracy: forming \(w\), \(p-m\), or the target height can still lose significant information. Exact rational evaluation remains the reference.

## 5. Checks performed during this investigation

The supplied PDF is source material [P0]. Its previously reported experiments are not counted as new executions. The following checks were actually performed in the research workspace, on **Linux x86_64**, using Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 where needed. They establish neither M3 timing nor historical novelty.

| Check | Executed result | Interpretation |
|---|---|---|
| Root implementation using `fractions.Fraction` | 324 parameter configurations; 10,044 exact probe inequalities and 10,044 exact source identities; no violation. Both compact formulas agreed exactly. Seed 28102026. | Tested the sharper optimized saturation construction, including zero parameters. Finite validation of implementation. |
| Separate mathematical implementation | 484 configurations; 1,938,904 random/critical float64 edges; maximum positive excess over the analytical bound \(3.55\times10^{-15}\) | Diagnostic corroboration of an independently coded upper portfolio; not a rigorous floating-error enclosure. |
| Separate rational checks | 10,000 edges for the first, looser saturation construction and 18,000 for the sharper family optimum; no violation | Exact sampled inequalities, not a proof over all real inputs. |
| Separate feasibility implementation | 216 configurations; 437,832 sampled/corner edges; maximum positive excess \(3.12\times10^{-15}\) | Additional implementation cross-check, with the earlier looser upper construction. |
| Sparse LP interface fixture | Three states, objective \(1/2\), zero reported primal violation | Exercised the sparse HiGHS interface and free-value formulation. |
| Independent finite-grid LPs | Fourteen cases in the table below matched (1); zero reported primal violations | Consistent lower relaxations, supported by the independent analytical upper. Solver success alone is not an exact certificate. |
| Primary interval implementation | 225 exact parameter cases, 14,400 exact probe inequalities, and 10,125 full-strip source identities; no violation | Checks include protected heights outside the evaluated source slab. |
| Separate primary proof implementation | 324 configurations; 11,988 exact probe checks, 11,988 full-strip identities, and 11,988 simplified-strip checks; no violation | Also checked the additive four-residual identity with nonlinear \(A(v)=\mu v+v^3\). |
| Varying-coefficient exact implementation | 324 configurations, 11,988 exact edge checks and 11,988 full-strip source identities; no violation; seed 314159 | Supports the stronger simple comparator, not a need for complex models. |
| Separate varying-coefficient implementation | 320 configurations, 10,240 exact probes, 4,800 full-strip identities; float64 maximum positive excess \(7.81\times10^{-16}\) in the modest-scale suite | Separate code corroboration; extreme-scale failures are reported below. |
| Primary/additive finite LPs | Eleven parameter instances, each solved once in the free-node class and once in the additive class; values in Section 5.4 | Numerical finite corroboration of the class separation, with all protected represented heights linked. |

The root rational check used parameters \(L\in\{0,1,2\}\), \(r\in\{0,1/2,1,2,5/2,3,4,5,9\}\), and \(H\in\{0,1/8,1/2,1,2,3,4,5,6,8,10,20\}\), with 31 dyadic source/amplitude triples per case. Section 13 supplies a standalone reference kernel and enough test instructions to recreate this check without the research workspace.

### 5.1 Exact exterior witness checked now

For \(L=1,r=4,H=5,\mu=1\), use sources at horizontal coordinates \(\pm1\) and heights 0 and 5. Amplitudes \(7/2,3/2\) reach \((5/2,-9/8)\); their reflected amplitudes reach \((-5/2,-9/8)\). Exact rational arithmetic confirmed both endpoint coincidences. The two source semantic gaps have total height 10 and four probe inequalities, giving \(E\ge5/2\). The explicit optimizer attains \(5/2\).

### 5.2 Compact-companion finite LP results, with units and timing limits

The lattice spacing was \(1/4\) horizontally and \(1/32\) vertically; all grid targets were represented exactly. An equivalent vertically translated slab was used internally; the table consistently reports full width \(H\). Times are one Linux solver invocation per case, exclude some construction/reporting work, and are **not** a performance comparison or a forecast for the Mac.

| \(r\), with \(L=\mu=1\) | \(H\) | State nodes | Inequalities | LP defect | Solver seconds |
|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 6,717 | 38,016 | 2 | 0.661 |
| 4 | 2.5 | 7,373 | 47,376 | 2 | 1.162 |
| 4 | 3 | 8,029 | 56,736 | 2 | 1.335 |
| 4 | 4 | 9,341 | 75,456 | 2 | 3.652 |
| 4 | 5 | 10,653 | 94,176 | 2.5 | 9.577 |
| 4 | 6 | 11,965 | 112,896 | 3 | 13.690 |
| 4 | 8 | 14,589 | 150,336 | 3 | 21.226 |
| 5 | 2 | 9,541 | 47,376 | 2 | 1.109 |
| 5 | 2.5 | 10,325 | 59,040 | 2 | 1.488 |
| 5 | 3 | 11,109 | 70,704 | 2 | 2.088 |
| 5 | 4 | 12,677 | 94,032 | 2 | 5.079 |
| 5 | 5 | 14,245 | 117,360 | 2.5 | 6.124 |
| 5 | 6 | 15,813 | 140,688 | 3 | 17.099 |
| 5 | 8 | 18,949 | 187,344 | 4 | 26.891 |

### 5.3 A negative diagnostic that changed the plan

For the **compact-semantic companion**, a coarse grid at \(L=r=1,H=0.05,\mu=1\) returned 0.045, while its continuum optimum is 0.05. The grid omitted the critical amplitude \(\sqrt{0.1}\). Similar companion relaxations returned 0.08 for \(H=0.1\) and 0.18 for \(H=0.2\). These are discretization gaps, not evidence that nonlinear readouts beat the theorem. The primary optimum at these same \(L,r,\mu\) is 0.5 for every height; do not assign the companion fixture to its problem ID.

The executable plan therefore requires exact witness insertion or rigorously bracketed algebraic amplitudes, and never promotes a mesh LP result to a continuum upper bound. It also uses the explicit optimizer rather than building a large two-dimensional fitting system merely to rediscover the formula.

### 5.4 Primary finite LP separation checks

All rows use \(L=\mu=1\). The free-node LP preserves full-height strip semantics at every represented eligible height; the additive LP uses variables for all represented horizontal coordinates and heights and enforces the global height slope of \(A\). These are diagnostic finite restrictions, not a substitution for the continuum proofs. The largest free graph had 1,085 state nodes and 3,754 inequalities; the largest additive LP had 104 variables and 3,648 inequalities. Maximum reported primal violation was approximately \(9.77\times10^{-15}\).

| \(r\) | \(H\) | Free-node LP | Additive LP | Exact predicted pair \((T,S)\) |
|---:|---:|---:|---:|---|
| 4 | 0 | 2 | 3 | \((2,3)\) |
| 4 | 1 | 2 | 3 | \((2,3)\) |
| 4 | 3 | 2 | 3 | \((2,3)\) |
| 4 | 4 | 2 | 3 | \((2,3)\) |
| 4 | 5 | 2.5 | 3.000000000000002 | \((2.5,3)\) |
| 4 | 6 | 3 | 3 | \((3,3)\) |
| 4 | 8 | 3 | 3 | \((3,3)\) |
| 5 | 0 | 2 | 4 | \((2,4)\) |
| 5 | 4 | 2 | 4 | \((2,4)\) |
| 5 | 5 | 2.5 | 4 | \((2.5,4)\) |
| 5 | 8 | 4 | 4 | \((4,4)\) |

### 5.5 Executed numerical cancellation stress test

Use \(L=1,H=4,\mu=1\), source \((u,v)=(-1,4)\), and \(t=r\). The exact defect of both primary readouts is 2 at \(r=10^6,10^8,10^9,10^{10}\). At the first two radii, the tested float64 interval implementation also returned 2; at the last two it returned **6**. For example, the exact target height at \(r=10^9\) is \(-499999999999999996\); float64 rounds it to \(-5\times10^{17}\), losing the original height offset. This is a numerical failure, not a theorem counterexample.

The varying-coefficient implementation returned 2 on that fixture at all four radii, but it did **not** solve the general numerical problem. At source \((+1,4)\), \(r=10^9\) or \(10^{10}\), its tested float64 defect was 4 instead of the exact 2; the interval implementation returned 6. Thin layers near this outer boundary also expose cancellation. A factored range-width calculation sometimes helps, but one tested thin-layer case at \(r=10^6\) produced approximately 2.0000305176 with factoring while the unfactored expression happened to produce 2.

Retain both source signs and the exact rational thin-layer fixture \(u=1-3/r\). Treat rounded-input evaluation, rounded target construction, and rounded readout evaluation as separate error sources. No universal float64 accuracy claim follows from the modest-scale random tests. The core exact certificates avoid this issue; a production floating implementation would need its own supported-range specification and error enclosure.

## 6. Fair evaluation protocol

### 6.1 Claim hierarchy, metrics, and interpretation

The primary claim is **the exact value (P1) for (7) under (6P), together with the exact all-additive value \(\mu S\) and an elementary varying-coefficient attainer**. The weaker compact value (1) is a separately labeled companion. Primary evidence consists of legal lower certificates and analytic upper constructions with matching bounds. The numerical summary is the interval \([L_B,U_B]\) in output units, accompanied by the exact expression when available. Report the certification gap \(U_B-L_B\), maximum observed edge residual, semantic violations, and certificate type separately for each contract and class.

For every rational primary instance, its finite witnesses are rational: require exact lower/upper equality. The compact companion can require an irrational amplitude; retain a symbolic argument or a rational lower approximation with a rigorously stated gap. A small floating solver residual is neither a proof of optimality nor a semantic tolerance that weakens (6) or (6P).

A second claim is explanatory: which restrictions create a gap? Report each admissible baseline's absolute defect and, only when its value is positive, absolute and relative difference from the optimum. Report every parameter regime, including equality. Do not average away regimes with no gain. The primary comparison is additive versus varying-coefficient/interval, with equality against the strongest expressive comparators. The companion's strongest simple portfolio also ties its unrestricted optimum.

A third, optional claim concerns engineering cost. Measure setup, per-evaluation cost, graph construction, solving, verification, storage, and peak RSS separately and end to end. An explicit closed-form evaluation and a generic finite optimizer offer different information: the former embeds a derived theorem. Any runtime comparison must disclose that difference and must not be sold as a new general LP algorithm.

### 6.2 Baselines, versions, and reproduction checks

All direct comparators use the same \(L,r,H,\mu\), exact source/target policy, arithmetic precision, and coordinate convention. There are no pretrained assets or project-specific training expenses to hide. The primary algorithms are the mathematical definitions below; their executable versions are identified by the PoC repository commit and configuration hash.

| ID / method ID | Method and contract | Exact defect or role | Required check |
|---|---|---|---|
| B0 / `tangent` | \(G=\mu v\); both contracts | \(\mu r^2/2\) | Residual exactly \(-\mu t^2/2\) |
| B1 / `original_y` | \(F=\mu y\), equivalently \(G=\mu(v+u^2/2)\); both | \(\mu Lr\) | Residual \(\mu ut\); zero at \(L=0\) |
| B2 / `strip_pdf` | Supplied strip optimizer (9)–(10); both | \(\mu S\) | Source-strip bound, transitions, exterior extension |
| B2S / `strip_simple` | Simplified strip optimizer (P4); both | \(\mu S\); exact optimum of **all additive readouts for the primary** | Full-strip source identity and complete additive lower certificate |
| B3 / `clip` | Plain clipping; **compact only** | \(\mu\min(H,r^2/2)\) | Source slope and endpoint residual; reject as a primary method |
| B4 / `clip_strip` | Better of B2S/B3; **compact only** | \(\mu\min(H,S)\) | Restricted reference; not the strongest comparator |
| B5 / `sat_optimal` | Optimized saturation-plus-potential family; **compact only** | \(\mu e_{\rm sat}\) | Class lower bound and (14)–(15) |
| B6 / `compact_optimal` | Better of B2S/B5; **compact only** | Exactly \(R_{\rm compact}\) | Strongest simple companion baseline; source identity holds for either selected member |
| B7 / `free_node_lp` | Free-node sparse LP; contract-aware | Unrestricted finite lower relaxation | Exact endpoints, all eligible semantic nodes, free bounds, independent certificate |
| B8 / `difference_constraints` | Exact fixed-threshold verifier; contract-aware | Finite feasibility or negative-cycle witness | Independent signs, source tests, and probe denominator |
| B9 / `protected_interval` | Interval construction (P2); primary | Exactly \(R_{\rm protected}\) | Full-strip identity, clamp ordering on the whole plane, predecessor bounds |
| B10 / `protected_affine` | Varying-coefficient readout (P5)–(P6); primary | Exactly \(R_{\rm protected}\) | Equality branch before division; coefficient continuity and residual interval |
| B11 / `fixed_label_lp` | Established minimax graph extension with protected labels \(\mu v\); primary diagnostic | Strong general extension comparator, with continuum optimum matched by B9/B10 | Correct fixed labels; no incompatible extra zero gauges; finite values are not a continuum interpolant |
| B12 / `additive_lp` | Finite additive \(A(v)+B(u)\) LP; primary diagnostic | Lower relaxation of the additive class, whose continuum value is \(\mu S\) | Global slope on every represented height; four-residual witness |

B0–B6, B9 and B10 have constant-size evaluation formulas and no tuning budget. Class restrictions are checked before any objective comparison. B3–B6 are registered as companion methods; do not admit them to a primary run merely because a particular branch happens to choose the strip readout. B9/B10 also satisfy the weaker contract but need not be optimal there.

B7, B11 and B12 use SciPy/HiGHS if available; lock the installed version after the target smoke test and preserve solver options. Use `highs-ds` initially, and use `highs-ipm` only for a concrete discrepancy or an explicitly planned cost comparison. Both are documented choices. Default `linprog` bounds are nonnegative; set readout/potential variables to `(None, None)` and only the defect variable to `(0, None)`, except explicit fixed-label equalities. [R15]

B8 is an independent implementation of established difference-constraint feasibility. Implementing it is a correctness cross-check, not a proposed novel solver. A tropical common-potential implementation is an optional equivalent control for that restricted problem; it is not required in addition to both B7 and B8 unless it resolves a remaining discrepancy. [R3–R6]

No stronger nonlocal baseline is needed to compare (7): the exact bounds and elementary attainers are already local. The published graph extension mechanism is represented directly by B11, but no published generic runtime is imported. Neural quantization results and higher-dimensional optimal-design examples remain contextual because their tasks and information differ. Do not import their numbers into a head-to-head table.

### 6.3 Workloads, provenance, and splits

The core data are generated mathematical instances, not a sample from a real application. Their provenance is the public equations and the versioned generator. The generator and fixtures should have an explicit project license chosen for the repository; do not assume the attached PDF's prose or figures are redistributable. There are no external dataset licenses, accounts, access tokens, or model-weight terms in the core path.

Use three separately recorded sets:

**Development fixtures.** Include the exact examples and all executed cases in Section 5. These are already seen and must not be called a protected confirmation set. Use them to debug signs, deduplication, branch boundaries, and solver plumbing.

**Analytical branch suite.** Cover every cell and boundary of the phase diagram. For \(L=\mu=1\), include \(\rho=0,1,2,5/2,3,4,5\); for each positive \(\rho\), include the distinct relevant heights among \(0,I,2I,2S,r^2/2\), their midpoints, and a height above the last transition. Add rational points on both sides of each nonzero boundary. The offset size is chosen to be representable in the selected arithmetic and small enough to test the same branch transition; it is a numerical stress parameter, not an effect-size target. Deduplicate exact duplicates and retain the derivation of each case label.

**Frozen independent implementation confirmation.** After the code, generators, baselines, and analysis schema pass the development gates, commit them and record their hashes. Then generate a fresh seed using `secrets.randbits(64)` and instantiate the recorded standard-library `random.Random(seed)`. Create a read-only manifest with this default rule, fixed before reading results:

1. Set \(q=4096\). Each fresh \(U\) below is an independently drawn integer from 1 through \(q-1\), divided by \(q\). At \(L=\mu=1\), use the radii \(0,2,3,2-1/q,2+1/q,3-1/q,3+1/q,2U_1,2+U_2,3+U_3,3+1/U_4\) in exactly this order, removing duplicates by first occurrence. These cover the zero case, both radial transitions and their sides, every open radial regime, and a larger-radius tail.
2. For each radius compute \(Q,I,S\) exactly. Sort and deduplicate the height boundaries \(\{0,I,2I,2S,Q\}\). Include every boundary; for each adjacent distinct pair \(h_0<h_1\), include \(h_0+U(h_1-h_0)\); add \(h_{\max}(1+U)+1\) above the last boundary. All randomness uses the same recorded RNG in this specified loop order.
3. Apply each scale pair \((a,b)=(1,1),(1/8,4),(8,1/4)\): map parameters to \((aL,ar,a^2H,b\mu)\). Add the dedicated \(L=0,r=1,H\in\{0,1\},\mu=1\) cases. Evaluate **both contracts** at every retained instance, with separate semantic predicates and admissible method rosters.
4. For each parameter instance, include the deduplicated triples with \(u\in\{-L,0,L\},v\in\{0,H\},t\in\{-r,0,r\}\), all applicable rational certificate probes, and 31 further dyadic triples \(u=L(2j_1/q-1),v=Hj_2/q,t=r(2j_3/q-1)\), where each \(j_i\) is uniform on integers 0 through \(q\). For an irrational companion internal witness, the default manifest uses Section 6.5's rational lower approximation with \(k=12\), matching the generator's dyadic granularity, and records its exact bound gap. This is an implementation fixture rather than an effect-resolution target; a claimed nonzero effect requires the separately frozen precision rule in Section 6.5. The primary certificate is already exact. These are fixed regression checks, not a statistical sample size or a cap on investigation. The proof supplies universal coverage.
5. Add known negative controls and large-radius cancellation fixtures as a separately labeled, previously seen stress suite. Every fixture carries its own problem ID: in particular, the coarse-mesh discrepancy is companion-only. Freeze any optional LP subset using graph size and branch coverage from pilot inputs, before seeing objective results. Retain exact scalar/certificate checks for all instances; a skipped optional LP is explicitly marked.

Generate the entire parameter list in steps 1–3 before drawing step 4's edge triples. Traverse radii in the stated first-occurrence order, height boundaries and their intervening points in ascending order, then scale pairs in the stated order; append the dedicated zero-width cases last. Deduplicate parameter tuples by first occurrence. Generate one edge list per parameter tuple and reuse it for both contracts, with no extra random draws between contracts. This fixes the RNG-consumption order. Store the complete manifest, not just its seed, and report its resulting case and edge counts. The dyadic denominator provides repeatable rational arithmetic and boundary resolution; it is not a claim about physical resolution. Changes to this default are allowed before confirmation when a measured cost or missing branch justifies them, with an amendment. Do not append cases until a desired score appears. If inspected confirmation results lead to a code or methodological change, preserve the failed manifest and use a fresh seed for the revised confirmation.

This separation protects against tailoring implementation to familiar fixtures. It does not turn finite sampling into evidence for a universal theorem. The proof addresses all permitted real parameters; the confirmation run addresses the coded realization.

### 6.4 Exact finite graph and LP specification

Use exact rational coordinate pairs as node keys, reduced to canonical numerator/denominator form before merging grids and witnesses. Integer lattice keys are only a local construction convenience: carry their grid denominator and convert them to canonical rationals before cross-grid deduplication. Insert every source, every probe target, and every witness endpoint before deduplication. A target that also lies in \(D\) is a source-eligible point with the same readout variable; it must not acquire a second identity. A source grid need not be closed under probes.

Let \(z_i\) be the value at node \(i\) and \(E\ge0\). For each sampled permitted source probe \(i\to j\), add

\[
z_j-z_i-E\le0,\qquad z_i-z_j-E\le0.
\tag{16}
\]

Use separate predicates: `probe_source_contains(u,v)` means \(|u|\le L,0\le v\le H\); `semantic_node_contains(u,v,problem_id)` means the same for the compact companion but only \(|u|\le L\), at **any height**, for the primary. The full-plane semantic negative control admits every node. A target can be semantically protected without being an eligible probe base.

For each exact horizontal coordinate, sort all represented **semantically eligible** nodes by height. Consecutive heights \(v_l<v_h\) suffice to impose all sampled semantic comparisons by transitivity:

\[
z_l-z_h\le-\mu(v_h-v_l).
\tag{17}
\]

Minimize \(E\), fixing one additive value to zero in each weakly connected constraint component of the **readout-node graph, excluding the shared defect variable**. This gauge removes only an irrelevant additive constant. Self-probes at \(t=0\) can be omitted after a separate identity test; do not mistakenly turn them into a positive lower bound. Build sparse matrices directly, never a dense constraints-by-variables array.

For floating LPs with \(L>0\), normalize \(u'=u/L\), \(v'=v/L^2\), \(z'=z/(\mu L^2)\), and \(E'=E/(\mu L^2)\). This avoids presenting tiny physical-unit right-hand sides to an absolute solver tolerance. Record the normalization, rescale bounds exactly, and verify the extracted certificate in the original rational coordinates. Extremely small normalized heights can still require exact arithmetic or further diagnosed scaling; overflow/underflow is a numerical failure, not an optimum of zero. Handle \(L=0\) and \(r=0\) analytically; handle \(H=0\) by contract: primary \(\mu I\), companion zero.

For rational-lattice replication, take \(u=p/n\), \(t=k/n\), \(v=q/(2n^2)\); the target key is exactly \((p+k,q-k^2)\). Keep \(|p/n|\le L\), \(|k/n|\le r\), and \(0\le q/(2n^2)\le H\), adding nonlattice witnesses separately when necessary. This lattice is convenient for the Section 5 fixtures, not a mandate to refine it indefinitely.

The free-node finite feasible class is a relaxation of the continuous problem: any admissible continuous function gives feasible values at these nodes. Therefore a verified finite optimum is a **lower bound**. Adding a bilinear interpolation constraint would restrict the class and destroy this lower-bound interpretation.

**B11, fixed-label extension.** Use the same probe edges, fix \(z_i=\mu v_i\) at every represented protected-strip node, and minimize maximal absolute probe difference. Strip semantic inequalities then hold automatically. Add a zero gauge only to a connected readout component with no fixed label; do not additionally force an arbitrary labeled node to zero. The labels already determine that component's additive offset. This is the standard fixed-label minimax extension formulation [R7]. The continuous proof establishes that these labels lose no optimum for this geometry. Without invoking that proved reduction, this more constrained finite solve is not an independent unrestricted lower-bound proof. Report its role explicitly, and compare its finite value to the restrictions of B9/B10; a finite value below the continuum formula can simply reflect omitted probes.

**B12, additive LP.** Introduce a variable \(a_j\) for each distinct represented height and \(b_k\) for each distinct represented horizontal coordinate, and substitute \(z_i=a_{j(i)}+b_{k(i)}\) in (16). Sort **all represented heights**, including heights appearing only at exterior target columns, and impose \(a_{j+1}-a_j\ge\mu(v_{j+1}-v_j)\). This follows from (6P) for an additive readout. Use free variable bounds. Fix one selected \(a_j\) and one selected \(b_k\) to zero to remove the internal decomposition shift and overall output shift; choose them from a represented state. Additional unidentifiable directions may remain on disconnected designs, but must not be removed by constraints that change probe differences. Insert the four probes from Section 4.7. This is a lower relaxation of the additive class, whose exact continuum upper is B2S; it is not a universal lower bound on interacting functions.

For a fixed proposed \(E\), convert (16)–(17) into difference-constraint arcs. Probe arcs have weight \(E\); semantic arcs point from high to low and have weight \(-\mu\Delta v\). A cycle with \(p\) probe arcs and total semantic height \(d\) requires

\[
pE-\mu d\ge0.
\tag{18}
\]

Thus \(\mu d/p\) is a certified lower bound for \(p>0\). Semantic arcs contribute **zero** to the denominator. Ordinary cycle mean, which counts every arc, gives the wrong quantity. A negative cycle containing only semantic arcs indicates invalid graph construction because true vertical order cannot strictly descend and return to the same point. [R6]

Implement exact Bellman–Ford feasibility for small witness graphs: initialize every node distance to zero (equivalent to a super-source), relax all arcs, and if a relaxation occurs on the \(N\)-th pass, follow predecessor links \(N\) steps and extract a cycle. Independently sum its rational weights and recheck every probe/source/semantic incidence. For larger graphs use B7 to suggest active constraints, then verify a small extracted certificate exactly. An approximate dual multiplier or guessed rational reconstruction is not a certificate until all identities and signs are checked.

For an exact graph lower bound \(\ell>0\), test infeasibility at a rational threshold strictly below \(\ell\), such as \(\ell/2\), and feasibility at the value of a proven admissible upper readout restricted to that graph. Use the **certified graph lower bound**, not the continuum formula, when constructing an infeasible threshold: a dyadic approximation to an irrational witness can leave a legitimate graph gap. At zero optimum, check feasibility at zero and the trivial bound \(E\ge0\); do not manufacture a negative-error instance.

### 6.5 Irrational amplitudes, numerical conventions, and proof boundaries

The **primary** internal amplitude \(a=\min(r,2L)\) and exterior coordinate \(w=\min(H,2L(r-L))/(2L)\), when that witness applies, are rational for rational parameters. Its complete finite certificate needs no square root or algebraic package.

For the **compact companion's** internal witness, \(a=\sqrt{2\min(H,I)}\) can be irrational. The mathematical proof uses this real number exactly. The machine verifier has two faithful options: represent it algebraically with its defining equation and ordered root, or construct a dyadic rational \(a_-\le a\) and certify its squared inequality using integers. If \(q=\min(H,I)=n/d\), choose precision \(k\), set

\[
m=\left\lfloor\sqrt{\left\lfloor2n\,2^{2k}/d\right\rfloor}\right\rfloor,
\qquad a_-=m/2^k.
\]

Python's integer `math.isqrt` supplies the inner operation. Check \(a_-\le r,2L\) exactly and use lower bound \(\mu a_-^2/2\). Increase \(k\) only until the certified gap to the analytic upper meets the predeclared numerical resolution. If testing a proposed nonzero difference, resolution must be tight enough to separate the intervals; otherwise report inconclusive. Choosing \(H=a^2/2\) with rational \(a\) gives exact fixtures without algebraic dependencies.

Parse rational inputs from integer pairs or strings, not binary floats. `Fraction(1.1)` and `Fraction("1.1")` denote different exact numbers. Rationalizing an approximate solver output is a hypothesis, not an exactness guarantee. [R16]

Use float64 for vectorized diagnostics. Do not introduce reduced precision or quantization: they are unnecessary and change the numerical question. Branch decisions for rational configuration parameters should be made exactly before vector evaluation. Test arbitrary negative target heights, zeros, ties, source/target overlap, and values just across clamp thresholds.

The analytic proof certifies the real-arithmetic function. A floating implementation's continuum error needs an additional rounding enclosure before being advertised as machine-certified for all inputs. If a rigorous uniform point-evaluation bound \(\delta\) is established on the relevant compact rectangle, add \(2\delta\) to the probe defect bound. The corresponding semantic guarantee is only \(G_{\rm float}(u,v_2)-G_{\rm float}(u,v_1)\ge\mu(v_2-v_1)-2\delta\). It does not recover a strictly positive lower slope for arbitrarily close real pairs in finite precision. Exact admissibility belongs to the mathematical function/rational reference; a finite-input implementation must state its supported input spacing and numerical semantic error. Without a rounding enclosure, label float results as numerical diagnostics.

### 6.6 Statistical and performance design

The primary result is deterministic. Do not compute a p-value from millions of edge samples, count seeds as independent users, or call the fraction of sampled inequalities passed a probability of theorem correctness. One valid counterexample or one invalid proof step matters more than a high pass fraction.

Use a pilot only to estimate numerical conditioning, construction cost, memory growth, and timing variability. Warm imports separately; use `time.perf_counter_ns`; batch scalar evaluation until timer/call overhead is negligible relative to the measurement, as determined by an empty-loop/control measurement. Record cold start separately. All array operations here are CPU operations; do not add a meaningless GPU synchronization step.

For descriptive calibration, report raw per-instance timings and repeated process-block summaries without significance claims. If a runtime advantage becomes a scientific claim, first declare the deployment workload and the smallest time difference that would affect that decision. Use pilot variance of paired log times to fix the number of independent run blocks for the desired confidence/precision, then freeze it before confirmation. Randomize comparator order within blocks; keep versions, thread settings, power mode, and cache policy fixed. Use a paired interval across independent blocks and report instance heterogeneity. Repeated inner-loop timings are not independent workload samples. Amend and reconfirm if the target changes; do not run until significance.

When effects are tiny or unstable, state that practical importance is unresolved. There is no arbitrary required percentage robustness improvement: the exact theorems set the available headroom, and the strongest simple expressive comparator for each contract has none.

## 7. M3 Max feasibility and hardware-adaptive execution

### 7.1 Hardware discovery and actual capability gates

Apple's M3 Max configurations differ in CPU/GPU counts and unified memory. Do not assume the largest memory option. [R17] The primary workload needs a native CPU interpreter and exact integers, not an Apple-specific acceleration kernel.

These are **existing system commands**, to be captured by the proposed `doctor` utility:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu
df -k .
python3 --version
```

Also record `platform.machine()`, Python's executable and build, available memory estimate, current swap/memory pressure, and whether the interpreter is translated. Avoid recording hardware serial numbers. A native arm64 interpreter is preferred; an x86_64/Rosetta run must be labeled, not called a native M3 benchmark. If Python is absent, install a supported native/universal CPython through the user's existing trusted setup route. No compiler toolchain is necessary for the exact core.

### 7.2 Selected stack, version policy, and licensing

**Required:** Python standard library, especially `fractions`, `math`, `json`, `hashlib`, `pathlib`, `time`, `subprocess`, and `unittest` or an already available test runner. The scalar mechanism can be reproduced with no third-party dependency.

**Optional diagnostic stack:** NumPy for batches and SciPy/HiGHS for sparse LPs. On the research date, the official distribution pages listed NumPy 2.5.4 and SciPy 1.18.1 with CPython 3.12 macOS 14+ arm64 wheels; matching alternatives are available for other supported macOS/Python combinations. These distributions were inspected individually; their joint installation was not tested here. [R18, R19] Use native Python 3.12 or 3.13 if available. Try these releases when compatible, but accept an older supported binary pair passing the resolver and smoke tests; record the reason and exact lock.

Example **existing package-management commands**, after creating the project directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: numpy==2.5.4 scipy==1.18.1
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze --all > requirements.lock.txt
```

The optional install is a compatibility attempt, not an unconditional promise. If resolution fails, inspect supported wheel tags and select a compatible stable pair before freezing. Do not silently compile a large stack or change the scientific model to make an install work. Preserve exact wheel filenames and SHA-256 values, plus the resolver report if supported by the installed pip. A plain `pip freeze` records versions but is not by itself a complete hash lock.

Direct `highspy` is optional only if documented thread control, basis reuse, or solver export is needed. Its inspected version was 1.15.1 with macOS arm64 distribution support. [R20] SymPy is optional for algebraic amplitudes; the dyadic witness method above avoids making it indispensable. Do not add a learning framework.

SciPy has BSD licensing, HiGHS/highspy MIT licensing, and NumPy's distribution includes a compound license expression for bundled components. Preserve the installed notices. The core has no weights or dataset downloads. Do not redistribute the attached PDF without a known permission basis; the standalone equations and newly written implementation are sufficient for execution. [R18–R20]

### 7.3 Resource model and estimates

Estimates below are derived allocation models, not M3 measurements. Measure the interpreter's resident baseline \(M_0\) and every substantial optional stage locally.

| Stage | Cost model | Initial practical shape and uncertainty |
|---|---|---|
| Formula and scalar certificate | Fixed arithmetic operation count; rational cost depends on numerator/denominator bit length | A few coordinate records per witness. Store bit lengths. The mathematical result does not grow with dataset size. |
| Exact parameter/edge tests | Linear in cases and sampled edges; stream each case | Core suite can use the Section 5 scale; no need to retain all evaluated arrays or grow until a desired result appears. |
| Vector diagnostic | \(8kN\) bytes for \(k\) live float64 arrays of batch length \(N\), plus runtime/temporary overhead | With 12 arrays and 100,000 edges: 9.6 MB payload. Determine actual live arrays and RSS before enlarging batches. |
| Free-node / fixed-label sparse LP (B7/B11) | Before gauges, \(m=2P+V_s\) inequality rows and at most \(6P+2V_s\) nonzeros for \(P\) probes and \(V_s\) semantic constraints | Encode fixed labels as equality bounds, or add the actual equality-matrix payload. Solver factors, presolve, dictionaries and assembly copies can dominate. |
| Additive sparse LP (B12) | Before gauges, \(m=2P+V_a\) inequality rows and at most \(10P+2V_a\) nonzeros; \(V_a\) counts adjacent-height constraints | Each probe row can contain four separate \(a,b\) coefficients plus \(E\). Count actual coalesced nonzeros and gauge/equality storage; calibrate separately. |
| Reports/checkpoints | Case-level JSON, compact certificate graphs, aggregate tables and plots | Store complete failure cases, but avoid logging every successful random edge unless needed to reproduce a defect. |

For a CSR matrix with `nnz` nonzeros, index size \(b_i\in\{4,8\}\), and \(m\) rows,

\[
M_{\rm CSR}=(8+b_i)\,\mathrm{nnz}+b_i(m+1),\qquad
M_{\rm rhs}=8m.
\]

Add source/target keys, readout vectors, duals, assembly copies, and measured solver workspace. The largest executed diagnostic had 18,949 state nodes and 187,344 inequalities. A dense float64 matrix with the additional defect variable would require about **28.4 GB just for that matrix**; sparse coefficient payload is orders of magnitude smaller. This is why dense construction is prohibited, not evidence that a large LP is needed.

A closed lattice with horizontal spacing \(h\) requires vertical spacing \(h^2/2\), creating approximately \(h^{-3}\) source nodes before counting probes. Do not confuse a modest horizontal refinement with a modest memory increase. Prefer exact endpoint deduplication and insertion of the few decisive witnesses; the explicit theorem needs neither a fine mesh nor a large solver.

### 7.4 Calibration, allocation, and fallbacks

Run the exact kernel first. Next measure one small completed vector batch and, if enabled, one small completed LP. Measure a second graph size only to estimate a real growth trend. Predict the next case's payload and peak RSS from actual node/edge counts, adding observed solver growth and background-memory variation. Set a documented working budget below currently available memory with room for those measured uncertainties. Do not fill all unified memory simply because it exists.

Use one solver process initially. Stream independent cases and release temporary matrices after recording each result. Parallelize only after measurement shows that additional processes fit the budget and improve a declared throughput objective. CPU parallelism changes performance evidence, not the mathematical feasible set.

If available memory is low, reduce diagnostic batch size, use witness-only graphs, and retain scalar exact checking. These are faithful changes because they do not alter the theorem or certificates. If SciPy is unavailable, use the exact small-graph verifier and report the missing numerical cross-check. Do not describe a reduced grid as a failed mechanism or replace the local requirement with cloud compute.

No honest M3 runtime promise can be made from the Linux timing table. Measure local time per exact edge, per vector batch, and per graph stage, then calculate the remaining run forecast from the frozen manifest. Longer runs are justified only if they resolve a stated uncertainty that the exact formula and small witnesses have not already settled.

### 7.5 Interruption and recovery

Each case is an independent transaction. Keep `manifest.json` immutable: it contains input cases, method rosters and hashes. Store mutable progress separately in `state.json` or an append-only `attempts.jsonl`, keyed by immutable case ID, with states `pending`, `running`, `passed`, `failed`, `inconclusive`, or `resource_limited`. Write results to a temporary file, validate the schema, then atomically replace the final case output. Mark a case complete only after its certificate and timing records are present. Never mutate the input manifest to record progress or silently drop a case.

After interruption, compare input, code, dependency, and schema hashes. Reuse only completed outputs with matching hashes. Restart a partial case from its immutable input; basis warm starts are unnecessary for the core and must not be trusted across unverified solver versions. Record the interrupted attempt and reason. Do not append new numbers to a half-written result or reuse old certificates after a domain change.

## 8. Repository, interfaces, artifacts, and logging contract

Everything in this section is a **proposed project interface to implement**, not a claim that these files or commands already exist. The supplied handoff file is the only required input. The original PDF may be retained as provenance if available, but no essential equation or implementation decision depends on opening it.

### 8.1 Minimal layout

| Path | Responsibility |
|---|---|
| `README.md` | Exact problem, command sequence, environment, claim limits |
| `research_poc_plan.md` | This frozen handoff |
| `pyproject.toml`, `requirements.lock.txt`, `provenance/wheels.json` | Minimal package metadata and resolved environment |
| `src/straightprobe/model.py` | Parameters, domain predicates, forward probe, coordinate conversion |
| `src/straightprobe/exact.py` | Rational constants, strip, interval, varying-coefficient and saturation kernels; exact selectors |
| `src/straightprobe/certificates.py` | Internal/exterior witnesses; exact incidence and bound checks |
| `src/straightprobe/graph.py` | Rational node deduplication, semantic arcs, sparse constraint assembly |
| `src/straightprobe/solvers.py` | Optional SciPy interface and independently coded exact feasibility verifier |
| `src/straightprobe/experiments.py`, `cli.py` | Manifest generation, case execution, resumability, proposed CLI |
| `tests/` | Mathematical fixtures, negative controls, independent-coordinate and restart tests |
| `configs/` | Versioned rational parameter manifests and numeric/performance policies |
| `provenance/`, `decisions/`, `audits/` | Source ledger, environment, amendments, stage gates, challenge records |
| `runs/<run_id>/` | Immutable manifest, per-case results, certificates, failures, timing/resource records |
| `reports/` | Rebuilt findings, phase tables/plots, claim ledger, reproduction instructions |

Do not create abstractions for models, datasets, distributed jobs, or training optimizers that this project does not use.

### 8.2 Shared method and certificate interfaces

Use explicit numeric types and input modes. A minimal conceptual contract is:

```python
@dataclass(frozen=True)
class Parameters:
    L: Fraction
    r: Fraction
    H: Fraction
    mu: Fraction

class Readout:
    method_id: str
    # scalar exact reference; vector float evaluation is a separate implementation
    def value(self, u: Fraction, v: Fraction) -> Fraction: ...
    def analytic_bound(self) -> Fraction: ...
    def semantic_certificate(self) -> dict: ...

def make_readout(method_id: str, parameters: Parameters,
                 problem_id: str) -> Readout:
    # Validate the contract/method combination, then bind exact parameters.
    ...

def build_graph(parameters: Parameters, problem_id: str,
                sources, amplitudes, witnesses) -> dict:
    # Use separate probe-base and semantic-node predicates.
    # Canonical nodes plus arcs; every coordinate is rational.
    ...

def solve_finite_graph(graph: dict, solver_config: dict) -> dict:
    # Values/status plus a proposed certificate; never a continuum readout.
    ...

def probe(point, t):
    u, v = point
    return u + t, v - t*t/2

def verify_certificate(parameters, certificate, problem_id) -> dict: ...
```

Readouts B0–B6, B9 and B10 implement the same source/target interface. The Section 13 kernel supplies their essential exact core; add the factory's contract check and any explicitly omitted control from the equations. Its `Readout(parameters, method_id)` constructor is a numerical core, not permission to bypass the factory's admissibility check. Register B3–B6 as companion-only. Reject unknown problem IDs and incompatible `semantic_scope` strings rather than guessing. The LP produces finite node values and a role-specific certificate, not a callable continuum readout unless a separate valid extension is supplied. All methods receive the same configuration; none may inspect protected result files to choose an implementation.

A certificate record contains its type and problem ID, exact rational node coordinates, every underlying forward probe with source and amplitude, semantic links with lower/upper endpoint IDs, directed inequality traversal, probe count, total semantic height, resulting rational bound, and verifier version. Store rationals as numerator/denominator pairs or canonical strings. A traversal of an inverse edge refers to the original permitted forward edge; it is not a new `t -> -t` operation. An additive-class certificate is labeled with that extra class restriction and is never presented as a universal interacting-class bound.

### 8.3 Configuration and experiment records

Example proposed configuration:

```json
{
  "schema_version": 1,
  "problem_id": "protected_strip_v1",
  "phase": "development",
  "parameters": {"L": "1", "r": "4", "H": "5", "mu": "1"},
  "source_coordinates": "u_v",
  "target_policy": "retain_all",
  "semantic_scope": "strip_all_heights",
  "methods": ["strip_simple", "protected_interval", "protected_affine", "free_node_lp", "fixed_label_lp", "additive_lp"],
  "arithmetic": "fraction_reference_float64_diagnostic",
  "solver": {"method": "highs-ds"},
  "seed": 28102026
}
```

For the companion, use `problem_id="compact_parabolic_slab_v1"`, `semantic_scope="slab_only"`, and its admissible roster from Section 6.2. Use `plane_semantics_control_v1` only for the explicitly stronger negative control, with probe bases still in \(D\). It requires a separate all-plane semantic predicate and expected value \(\mu S\).

The runtime record must include: run/case IDs; problem ID and function-class restriction; timestamp; phase; input and code hashes; git dirty state or exact patch; Python/package versions; OS/architecture; rational parameters and normalized values; generator/seed and RNG implementation; node/edge counts; solver status/options; exact lower and upper bounds; bound provenance (`analytic`, `exact_cycle`, `numerical_only`); maximum sampled residual; full declared-semantic-scope violations; construction/solve/verify/total wall seconds; peak RSS in **bytes**; array/matrix payload; interruptions; and links relative to the run directory to the full certificate or failure example.

Resource APIs differ by platform: if using `resource.getrusage`, normalize its RSS units explicitly rather than assuming Linux and macOS report the same unit. Record the raw value and unit source too. Never store an unlabelled number called `memory`.

All reports must regenerate from these artifacts. Preserve unsuccessful solver attempts and unfavorable equality cases. `decisions/progress.md` records the last completed work package, unresolved issue, next permitted action, and exact resume command. This enables execution without conversation memory.

## 9. Risk-ordered work packages and decision gates

The following commands are **proposed project commands**. The coding agent must implement them through `src/straightprobe/cli.py` and a package `__main__.py`. Use `.venv/bin/python -m straightprobe ...` after the package is installed or configured on the module path. Do not copy these commands as if a released package called `straightprobe` already existed.

| Package | Dependencies | Primary uncertainty | Gate before proceeding |
|---|---|---|---|
| WP0 | None | Correct problem, provenance, target capabilities | Exact scope recorded; native scalar Python works |
| WP1 | WP0 | Mathematical transcription and certificate legality | Proof obligations and exact minimal witnesses pass |
| WP2 | WP1 | Faithful readouts and strongest baseline | Independent implementations agree; all families present |
| WP3 | WP1, WP2 | Correct finite lower harness | Known positive/negative fixtures and cycle accounting pass |
| WP4 | WP2, WP3 | Original result reproduced and no weak comparator | Strip/global controls and full baseline table verified |
| WP5 | WP4 | Resource capacity and confirmation design | Measured allocation/throughput; immutable protocol/manifest |
| WP6 | WP5 | Independent implementation confirmation | Every case has a valid status and evidence; no contradiction left unexplained |
| WP7 | WP6 | Attribution, boundaries, and scientific stopping | Required robustness checks complete; any new problem separately specified |
| WP8 | WP7 | Reproducibility and honest conclusions | Clean rerun, complete records, final challenge audit |

### WP0 — Scope, provenance, and target discovery

**Objective/risk:** prevent a fresh agent from solving an easier or different problem or assuming nonexistent hardware. **Inputs:** this handoff and the user's actual laptop. **Actions:** create the minimal repository; record equations (5)–(8), (6P), both problem IDs, full-width convention, and source/target policy in `README.md`; create `provenance/sources.json` from Section 12; capture the hardware commands and interpreter metadata from Section 7. If the PDF is present, hash it and compare with [P0]; if absent, continue using the fully restated problem. Add a source-availability note rather than fabricating an attachment.

**Files/command:** `model.py`, `configs/development.json`, `provenance/environment.json`; proposed `doctor --output provenance/environment.json`. Implement input validation and rational parsing before numerical libraries. Optional binary installs follow scalar success.

**Outputs/tests:** JSON with OS/architecture, available resources and dependency status; identity probe test; exact conversion round trip \((x,y)\leftrightarrow(u,v)\); rejection of invalid parameters. **Resources:** one interpreter process; measure baseline RSS. **Acceptance:** the core scalar path executes and the scope fields are explicit. **Next action:** on missing optional SciPy, continue with exact checking and mark the LP gate conditional; on a missing working Python, resolve that prerequisite before dependent work; on a requested domain mismatch, record an amendment instead of silently changing (8).

### WP1 — Proof transcription and small exact certificates

**Objective/risk:** validate the indispensable mechanism before building a larger harness. **Dependencies:** WP0. **Actions:** implement the primary internal witness from Section 4.7, with source height zero and targets at \(-I\), and the exterior witness from Section 4.3. Their rational instances are fully exact. Separately implement the compact internal witness from Section 4.2, with its dyadic square-root approximation only when required. The verifier must recompute targets, test every forward source against \(D\), test amplitudes against \([-r,r]\), validate vertical links against the selected semantic predicate, and derive the bound by summing inequalities. Include the separate \(L=0,r=0,H=0\) branches and the additive four-residual identity.

**Files/command:** `certificates.py`, `tests/test_witnesses.py`, `audits/proof_obligations.md`; proposed `verify-witnesses --config configs/development.json`. **Outputs:** small exact certificate JSON files, including the \(L=1,r=4,H=5\) target coincidence \((\pm5/2,-9/8)\) and lower bound \(5/2\).

**Tests/resources:** independently check (5) in original coordinates and check that substituting \(S_{-t}\) for the inverse fails as expected. Use at most the few nodes each witness requires; rational bit length governs cost. **Acceptance:** legal exact certificates give the expected bounds and deliberate source/amplitude corruption is rejected. **Next action:** a wrong target or sign is an implementation defect; a legal counterexample to a claimed inequality is a proof defect that blocks later efficacy claims. An irrational coordinate representation problem calls for the rational lower approximation, not omission of the case.

### WP2 — Reference readouts and strongest simple comparators

**Objective/risk:** prevent weak-baseline or capacity claims. **Dependencies:** WP1. **Actions:** implement B0–B6, B9 and B10 using the exact reference kernel and equations; independently code any enabled float64/vector forms. Implement parameter-branch decisions exactly once per instance. Include both primary attainers and reject companion saturation methods under the primary contract. Expose semantic-scope and analytic-bound records through Section 8. Do not fit a more complicated model to approximate an available exact baseline.

**Files/command:** `exact.py`, optionally `vectorized.py`, `tests/test_readouts.py`; proposed `check-readouts --config configs/development.json`. **Tests:** source identity for B5/B6 in \(D\); B2S/B9/B10 identity at arbitrary protected heights below zero and above \(H\); continuity in state coordinates at clamp/coefficient seams; residual (13); equality of (1)/(4); equality of B9/B10 bounds to (P1); boundary examples \((L,r,H)=(1,4,0),(1,4,3),(1,4,5),(1,5,4)\). Repeat on rescaled rational parameters. The selected optimizer need not depend continuously on the parameters across a method-selection branch; each state function and the optimal value must have the proved continuity properties.

**Outputs/resources:** exact contract-aware baseline tables plus float discrepancy diagnostics; scalar memory or calibrated vector batches. **Acceptance:** exact values and independent implementations agree; known floating failures are detected and labeled. **Next action:** a purported improvement over B10 in the primary or B6 in the companion requires an admissibility/proof/harness audit before tuning. Ties against these strong comparators are the predicted result, not a reason to hide them or introduce complexity.

### WP3 — Independent finite lower-bound harness

**Objective/risk:** expose sampling, endpoint, semantic, and optimization errors. **Dependencies:** WP1–WP2. **Actions:** implement (16)–(18), exact node deduplication, separate semantic/probe eligibility, and a small exact Bellman–Ford verifier. If SciPy is available, add the B7/B11/B12 formulations from Section 6.4 and independent primal-residual reporting. Provide `mesh_only` and `witness_enriched` graph modes. The latter deliberately receives analytical witnesses; agreement is verification, not independent discovery.

**Files/command:** `graph.py`, `solvers.py`, `tests/test_constraints.py`; proposed `check-graph --fixture three_state`, then `solve-grid --config configs/development.json`. **Fixtures:** \(A=(0,0),B=(1,1/2),C=(0,1)\) with \(C\to B\), \(B\to A\), and semantic gap \(C-A\ge1\), giving \(E=1/2\). Also use the reciprocal compact witness and the exterior common-target witness.

**Negative controls:** in the free-node formulation remove all semantic constraints and require \(E=0\); duplicate a target under two IDs and ensure the identity validator catches it; reverse a semantic arc; count semantic arcs in the cycle denominator; impose default nonnegative bounds and detect changed feasible values where the gauge needs negatives. Attempt to zero-gauge a nonzero fixed label and reject it. Omit negative-height protected semantic nodes and demonstrate the changed primary problem. The correct harness must reject corrupted records or display their expected wrong result.

**Outputs/resources:** exact certificates, sparse matrix sizes, numerical solver status, and discrepancy examples. Start with witness graphs; calibrate before grid growth. **Acceptance:** B7/B8 match on identical finite fixtures, and a lower bound is never mislabeled as a continuous upper bound. **Next action:** if SciPy fails but exact B8 works, record the limitation; if B7/B8 disagree, stop broad sweeps and isolate the smallest disagreeing graph; if a mesh undercuts the theorem, insert the missing witness before interpreting the difference.

### WP4 — Reproduce relevant originals and finalize comparison scope

**Objective/risk:** establish credible baselines before evaluating the refinement. **Dependencies:** WP2–WP3. **Actions:** reproduce the strip's regimes and both transition boundaries, including legal inverse-edge accounting. Check the \(d=3\) global polynomial residual and six-traversal cancellation from Section 4.6 using exact polynomial coefficient arithmetic. A larger dimension sweep is unnecessary unless this check exposes an unresolved implementation issue.

**Files/command:** `tests/test_original_controls.py`, `reports/baseline_reproduction.md`; proposed `reproduce-controls`. **Outputs:** exact strip values, lower witnesses, both strip uppers, the cubic maximum \(\mu r^3/12\), and complete contract-specific baseline tables including B6/B9/B10/B11. Record that the global polynomial control has different domain assumptions.

**Resources/acceptance:** constant-size symbolic/rational expressions plus optional small LPs; every relevant baseline is correct and receives the same parameters. **Next action:** a failure of the original formula blocks attribution until resolved. Finding a stronger established equivalent reclassifies novelty, but does not invalidate correct mathematics. Do not launch an ML baseline to compensate for an unresolved mathematical discrepancy.

### WP5 — Pilot, capacity calibration, and protocol freeze

**Objective/risk:** choose informative work within measured capacity and prevent adaptive confirmation. **Dependencies:** WP4. **Actions:** measure scalar exact throughput, a vector batch if enabled, and a small LP if used. Estimate the next graph's node/edge count and memory before constructing it. Decide which optional diagnostics add information beyond the proof. Freeze the exact method roster, generator rules, arithmetic/certification policy, output schema, and any performance question. Generate the independent confirmation manifest only after this freeze.

**Files/command:** `configs/confirmation.json`, `runs/<id>/manifest.json`, `decisions/protocol_freeze.md`; proposed `pilot`, `freeze-protocol`, and `make-manifest`. These commands must save the code/dependency hashes and never inspect confirmation outputs while selecting the manifest.

**Outputs:** measured resource table, a run forecast on the actual Mac, final case count with a branch-coverage/cost rationale, and a preserved pilot report. **Acceptance:** all core branches are represented, all planned jobs fit the measured working budget, and any required statistical timing study has a fixed design. **Next action:** insufficient capacity reduces optional batch/mesh work while preserving exact witnesses. Inadequate numerical resolution triggers a higher exact precision or an inconclusive label, not a changed effect threshold after seeing the result.

### WP6 — Controlled confirmation

**Objective/risk:** independently confirm the coded theorems and comparisons. **Dependencies:** WP5. **Actions:** execute the Section 6.3 manifest with each contract's admissible roster and declared finite controls. Check mathematical admissibility before comparing defects. Produce exact primary certificates and exact or rigorously bracketed companion evidence; keep sampled maxima in a separate field. For every instance report all admissible baselines and explicit inadmissibility labels, not only the winning method.

**Files/command:** `runs/<id>/cases/`, `certificates/`, `resources.jsonl`; proposed `run --manifest ...`, with `--resume` validating hashes. **Expected outputs:** primary B9/B10 equal (P1), additive optimum equals \(\mu S\), strict separation occurs exactly in the proved regime, and general fixed-label extension can match. Companion B6 equals (1), can improve on B4, and B5 can lose to the strip readout at large heights. No admissible method falls below its valid universal lower certificate.

**Resources:** the measured WP5 forecast, sequential case execution initially. **Acceptance:** every manifest item has a preserved terminal status; all exact contradictions are resolved; no unreported exclusion or post hoc baseline change. **Next action:** a valid counterexample stops the corresponding theorem claim and triggers WP1; a float/solver discrepancy triggers a minimal exact reproduction; an interrupted/resource-limited run resumes or remains explicitly incomplete; ties against B10 and B6 are the expected strong-baseline conclusions.

### WP7 — Attribution, robustness, and optional geometry transfer

**Objective/risk:** explain the effect and avoid overstating its scope. **Dependencies:** WP6. **Actions:** run Section 10's mandatory tests. In the primary, compare the entire additive class with B10, strengthen semantics to the whole plane to recover \(\mu S\), and test the numerical cancellation fixtures. In the companion, remove the horizontal saturation offset to recover clipping. Hold the probe population fixed when changing a function class; label semantic-contract changes as different tasks. Test both sides of every phase transition and the large-height limit. Compare known-equivalent implementations to detect manufactured differences.

**Files/command:** `configs/ablations.json`, `reports/attribution.md`; proposed `run-ablations`. **Outputs:** mechanistic explanation with legal witnesses, explicit no-advantage conclusions against B10/general extension for the primary and B6 for the companion, and assumptions whose violation changes the problem.

**Conditional follow-on:** if there is a reason to assess a usual calibration rectangle, first create `compact_original_rectangle_v1` with source \(|x|\le L,0\le y\le H\), semantic comparisons only within those original-coordinate vertical intervals, the original \(T_t\), and all targets retained. This is explicitly a compact-semantic transfer diagnostic. Its straightened source fibers are \([-u^2/2,H-u^2/2]\). Recheck admissibility before reusing any readout. Compare \(F=\mu y\), the strip optimizer, and \(F=\mu\operatorname{clip}(y,0,H)\), whose exact defect is \(\mu\min(H,Lr)\). A new free-node lower graph and separately verified continuous upper are required. Neither (1) nor (P1) applies to this geometry. A full-height protected-strip version of this rectangle would be another explicitly specified problem; clipped \(y\) would be inadmissible there.

**Acceptance/next action:** the primary result remains explicitly limited to (8). If the ordinary rectangle invalidates the saturating readout's semantic slope, that is a failed transfer of the method, not a counterexample to (1). Increase complexity only when the new problem has a verified nonzero gap that simple admissible baselines do not close, or a concrete unresolved optimality question. Otherwise finish with the narrow result.

### WP8 — Reproducibility, final audits, and disposition

**Objective/risk:** deliver a result another coding agent can reproduce without chat history. **Dependencies:** WP7. **Actions:** rebuild tables from immutable run records; perform a clean local environment replay of at least one case in every distinct branch, including the strongest-baseline equality and an intentionally invalid certificate. Verify interruption/resume equivalence with matching hashes. Re-audit the theorem, claim labels, baseline scope, source ledger, and hardware measurements. Perform a targeted literature follow-up on the exact final theorem statement if any new formulation emerged.

**Files/command:** `reports/findings.md`, `reports/claims.json`, `audits/final.md`, `decisions/progress.md`; proposed `report --run ...` and `reproduce --manifest ...`. **Outputs:** exact theorem and implementation status, full baseline table, resources, failure cases, provenance, and a decision of continue/revise/narrow/stop with reasons.

**Acceptance:** a fresh local run regenerates the cited evidence; unsupported novelty or practical claims have been removed; all remaining unknowns are named. **Next action:** if a prior equivalent exists, publish only a replication/worked-example characterization if publication is later desired. If no application mapping exists, conclude the mathematical PoC without starting a quantization project. A deployment, public release, or contact with outside researchers is not part of this research assignment.

## 10. Ablations, boundary tests, failure interpretation, and continuation

### 10.1 Mandatory adversarial tests

| Test | Expected outcome / reason | Diagnosis if violated |
|---|---|---|
| Restrict primary to all \(A(v)+B(u)\) | Exact value \(\mu S\), including nonlinear \(A\) | Comparing only a few additive polynomials is an incomplete class baseline |
| Allow \(k(u)v+b(u)\) in primary | Exact value \(\mu T\), equal to B9 and general extension | A purported further robustness gain needs an immediate certificate/admissibility audit |
| Strengthen primary semantics to every horizontal column | Exact value returns to \(\mu S\) | Omitting exterior semantic links falsely preserves a gain |
| Remove \(h_*\) from companion B5 | Exactly B3; isolates horizontal target compensation | Different result means populations or arithmetic differ |
| Enforce global exact vertical covariance in either contract | Class optimum \(\mu S\) | A smaller result likely weakened covariance, sampled amplitudes, or changed targets |
| Preserve exact covariance only on the protected strip | Primary B9/B10 remain admissible | Rejecting them confuses protected-strip with all-plane constraints |
| Compare B6 or B10 with a separately coded identical formula | Identical exact objective and admissibility | Any reported gain is a harness artifact |
| Give the LP all legal critical witness nodes | Lower certificate reaches the predicted branch value, exactly or with an explicit algebraic approximation gap | Inspect endpoints, directions, missing semantic comparisons, and variable bounds |
| Use the companion-only coarse mesh at \(L=r=1,H=0.05,\mu=1\) | Companion LP 0.045 versus true 0.05; primary has different value 0.5 | A missing amplitude is a discretization gap; using the wrong problem ID invalidates evaluation |
| Delete semantic constraints | Constant readout achieves zero | A positive lower bound is a graph/gauge construction error |
| Reject probes with targets outside \(D\) | Detect and reject as a changed problem | Smaller defect has no bearing on either stated theorem |
| \(L=0\) or \(r=0\) | Zero optimum for both contracts | Division-by-zero or identity-probe bug |
| \(H=0\), with \(L,r>0\) | Primary \(\mu I\); companion zero | Treating primary semantics as vacuous silently weakens the task |
| Exact phase equalities and neighbors | Optimal values agree at boundaries; predicted branches on either side | Float branch logic or width/scale error; the selected optimizer itself need not be parameter-continuous |
| Rescale \(u,t\) by \(a\), \(v\) by \(a^2\), output slope by \(b\) | Defect scales by \(ba^2\) | Unit or normalization defect |
| Evaluate the original-coordinate probe independently | Same residual after the coordinate transformation | A missing \(t^2/2\) term or accidental exact-flow substitution |
| Increase actual semantic domain beyond design height | Admissibility must be rechecked; saturation may fail the new lower slope | Do not report only its smaller error while hiding lost semantics |
| Change \(r\) after selecting the readout | Recompute or explicitly test a mismatched-domain problem | Extra information or post hoc parameter retuning confounds comparison |
| Original-coordinate rectangle | Unmodified compact-band saturation can fail semantic preservation | A transfer failure, not a theorem contradiction |
| \(r=10^6,10^8,10^9,10^{10}\), \(u=\pm1\) and \(u=1-3/r\), \(v=4,L=1,H=4,t=r\) | Exact reference stays within 2; naive float evaluators can fail as in Section 5.5 | Diagnose input/target/output rounding separately; never discard the failure as an optimization result |
| Feed a reached point outside \(D\) back as a new probe base | It is outside the evaluated one-step guarantee | Do not reinterpret the theorem as arbitrary multi-step robustness |
| Resume after interrupt or changed code/config | Same-hash rerun agrees; mismatched state is refused | Stale certificate reuse invalidates reproducibility |

The original-coordinate rectangle supplies a concrete adversarial case: at \(u=1\), original source heights \(y=0\) and \(y=1/4\) map to negative \(v\). A plain \(v\)-clipped readout can be constant at both, violating the required positive source difference for that rectangle. This illustrates why the geometric distinction is substantive.

### 10.2 Concrete semantic-scope counterchecks

At \(L=1,r=5,H=4,\mu=1\), the primary affine optimizer has \(k(4)=0,b(4)=2\). It is constant as a function of height at horizontal coordinate 4. This is allowed outside the protected strip and directly violates a hypothetical all-plane positive lower slope. On \(|u|\le1\), the same optimizer is exactly \(v\) at every height. Test both facts; neither one can substitute for the other.

For the compact saturation optimizer at these parameters, the horizontal offset at \(u=0\) is zero, and its value is constant for \(v<0\). Thus its comparisons at \(v=-1,0\) violate the primary promise while its comparisons within \([0,4]\) satisfy the companion. A harness that reports this lower error without flagging primary inadmissibility is invalid.

The interval and varying-coefficient uppers in Sections 4.7 and 4.9 are the independent primary constructions. Their pointwise values need not match away from the source strip; their admissibility and worst-case bounds must match. Do not compare different chosen optimizers using pointwise equality as the success criterion.

### 10.3 Safe lower/upper separation for an unsolved follow-on

Only if WP7 opens a genuinely different domain, use a finite free-node graph for lower bounds and an explicitly continuous model for upper bounds. For a rectangular mesh, a bilinear readout is continuous; imposing adjacent vertical increments at least \(\mu\Delta v\) at both horizontal endpoints of every source cell enforces the semantic lower slope throughout that cell. This condition applies only when cells match the declared semantic domain. A curved or varying-height source requires a compatible mesh or separate proof. A finite mesh alone does not certify full-height strip semantics: that stronger contract also needs a proved continuous extension beyond the meshed heights.

For (5), if such a readout has rigorously bounded coordinate Lipschitz constants \(K_u,K_v\) on all evaluated sources and targets, its residual satisfies the derived inequality

\[
|\Delta R|\le2K_u|\Delta u|+2K_v|\Delta v|
+(K_u+rK_v)|\Delta t|.
\tag{19}
\]

A complete covering grid with half-spacing radii therefore gives a continuum allowance

\[
K_u h_u+K_v h_v+(K_u+rK_v)h_t/2
\]

above its sampled maximum. Evaluate the allowance exactly or with outward rounding and use actual boundary cover radii. A restricted interpolant LP is not an unrestricted lower bound. A valid but wide enclosure is inconclusive; refine where the enclosure is loose, not merely where a plot looks interesting. Do not claim that arbitrary finite feasible values admit a monotone Lipschitz extension. [R11, R12]

### 10.4 Outcome classification and stopping rules

| Observation | Classification | Next action |
|---|---|---|
| Invalid source, merged/doubled endpoint, wrong semantic sign, or missing exterior target | Broken implementation/evaluation | Minimize the fixture, fix, rerun development and fresh confirmation if outputs were inspected |
| Exact legal witness contradicts a proposed upper | Mathematical defect or incorrectly stated admissible class | Suspend the theorem claim; audit the conflicting inequalities before further scaling |
| Float LP undercuts the theorem but witness is absent | Insensitive finite relaxation | Add the witness or retain a lower-only result; no efficacy claim |
| Bounds overlap a proposed small improvement | Inconclusive numerical resolution | Improve certification if affordable and scientifically useful; otherwise report the interval |
| Primary improves the additive optimum but ties B10/general extension | Expected class-separation result | Explain the simple varying coefficient and retain the no-gain conclusion against stronger methods |
| Companion improves B2/B3 but ties B6 | Expected scope-limited result | Explain saturation/offset compensation and preserve the no-gain conclusion against its strongest baseline |
| No gain in a predicted equality regime | Evidence consistent with the theorem | Report it; do not remove that regime |
| A new method is worse than B10 in primary or B6 in companion | Dominated for that contract | Stop tuning unless a separate cost/regularity constraint justifies it |
| Memory or local dependency limit | Resource/prerequisite limit | Use the faithful scalar/witness path; mark optional work unavailable |
| Exact prior-art equivalent found | Replication/new exposition rather than methodological novelty | Cite and reclassify; retain valid checks and consider a new problem only for a stated reason |
| Compact success fails on another source geometry | Boundary of applicability | Narrow the claim; formulate a new domain before testing a new conjecture |
| Quantizer or application mapping cannot be established | Broader real-world claim unsupported | Stop at the mathematical result |

The core stopping point is reached when the proof obligations, all strong comparators, exact certificates, independent local implementation, and prescribed robustness checks agree within their declared arithmetic status, and the final source search does not materially change the novelty classification. Known float failures must be reproduced and scoped; they must not be hidden behind exact success. Additional millions of random edges cannot improve the universal proof. Do not increase dimension, train a network, or run a larger LP merely to produce a more impressive workload.

Escalation is justified by a new, explicit constraint that the current theorem does not cover; a concrete counterexample demanding explanation; a demonstrated application mapping with remaining headroom; or a certificate gap caused by a genuinely unresolved geometry. A hypothetical future scaling benefit does not excuse a failure in a setting that already contains the proposed mechanism. Conversely, failure to transfer to a different geometry does not refute the proved parabolic-band statement.

## 11. Audit record, reporting requirements, and residual limitations

### 11.1 Material defects found and revisions made

Three distinct AI-assisted review tracks were used during this investigation: mathematical/prior-art challenge, experimental-validity challenge, and implementation/feasibility challenge. They were separate agent analyses within this research session, not external peer review or proof-assistant verification. The following changes were made in response to concrete problems:

| Defect or challenge | Revision | Remaining limit |
|---|---|---|
| A finite graph/common-potential solver looked like a contribution | Located direct log-Chebyshev/tropical prior art; reclassified it as a baseline | Exact compact formula's historical priority still unresolved |
| Compact improvement could be explained by elementary saturation | Added exact clipping baseline, then optimized the entire saturation-plus-potential family | Gains against weaker classes must retain their class labels |
| A complex joint model seemed necessary for the companion | Simplified its upper to a scalar clip plus a horizontal offset | No interaction requirement for the compact-semantic problem |
| Strongest baseline erased the companion algorithmic win | Proved the strip/saturation portfolio attains its unrestricted optimum | The contribution is a characterization, not superiority over that portfolio |
| Initial compact refinement weakened both probe population and semantics | Derived the protected-strip theorem retaining the PDF's full semantic promise | Probe evaluation is still finite-height and one-step |
| Primary additive-class gap could be mistaken for a need for complex models | Derived an exact varying-coefficient attainer and identified the established model family | Only the specifically defined additive class is separated |
| Generic extension looked like adjacent rather than direct prior art | Mapped fixed protected labels to standard minimax Lipschitz extension | No claim of a new general extension algorithm |
| A sequential two-probe loop lost a factor of two in short slabs | Replaced it with reciprocal probes joined by two semantic comparisons | Certificate verifier must distinguish probe edges from semantic links |
| Original strip loops could leave the compact source range | Built common-target pairs with every source height checked | Different probe/source policies require a new proof |
| Grid LP values appeared smaller than the continuum optimum | Reproduced the missing-amplitude failure; separated lower graphs from upper constructions | Sampling alone never establishes the supremum |
| Ordinary cycle mean would count semantic arcs | Specified cycle ratios or exact fixed-threshold feasibility | Any extracted numerical dual still needs exact verification |
| Probe eligibility and semantic eligibility were easy to conflate | Separate contract predicates; primary includes negative-height protected targets | All-plane semantics are stronger and return the floor to \(\mu S\) |
| \(H=0\) looked uniformly vacuous | Split its value by contract: primary \(\mu I\), companion zero | Configuration/admissibility checks must preserve this distinction |
| Finite precision appeared harmless after modest random checks | Executed large-radius failures for interval and affine readouts | No universal floating implementation certificate has been established |
| Fixed labels plus arbitrary zero gauges can contradict each other | Gauge only unlabelled fixed-extension components; specify additive LP heights globally | Numerical solver status still requires independent interpretation |
| Fresh confirmation cases lacked a reproducible generation rule | Added an exact seeded, stratified manifest contract with fixed default branch coverage | Finite confirmation verifies implementation, not theorem probability |
| A rectangle after straightening could be mistaken for an ordinary calibration rectangle | Stated the parabolic original-coordinate set and added a concrete transfer failure | Broader application relevance remains open |
| Floating solver success and old PDF tests could be overstated | Separated exact/analytic/numerical evidence, original reported tests, and newly executed checks | No formal floating continuum certificate or M3 test has been completed here |
| The key equality of two compact formulas was left implicit | Added the complete three-case portfolio identity argument | Human mathematical review can still find a missed issue |
| A large mesh would waste resources and risk dense allocation | Primary path now uses O(1) readouts and tiny certificates; sparse LP is secondary | Optional graph peak memory still needs target measurement |
| Additive LP rows are wider than free-node rows | Separate \(10P+2V_a\) versus \(6P+2V_s\) sparse payload estimates and require actual nonzero counts | Solver workspace remains measured rather than predicted from payload alone |
| An immutable confirmation manifest appeared to contain mutable progress | Separate immutable inputs from `state.json`/`attempts.jsonl` | Resume must validate the unchanged manifest and code/dependency hashes |

After these revisions, the final scientific/prior-art, mathematical/experimental-validity, and fresh-laptop implementation reviews found no further substantive correction. The last targeted rechecks covered the companion-only coarse-grid example, full-height semantic eligibility, fixed-label gauges, additive LP storage, RNG ordering, and immutable resume inputs. The Python block was extracted directly from this Markdown and executed again; its hash and counts matched Section 13. UTF-8 decoding, equation-delimiter balance, reference identifiers, code fences, and table column counts were also checked. These are completed AI-assisted challenge and consistency checks, not an assertion of formal proof verification or external peer review.

### 11.2 Required final report from the coding agent

`reports/findings.md` must lead with the actual outcome and contain the precise problem, theorem/verification status, complete baseline comparison, proof/certificate links, local hardware/dependency lock, raw case coverage, performance boundaries, unfavorable results, methodological amendments, and a continuation decision. Keep these evidence labels separate:

- **Source-verified:** facts read in primary publications or official documentation.
- **Derived here:** mathematical consequences with their stated assumptions and proofs.
- **Executed now:** tests actually run, including exact environment and measurement boundaries.
- **Proposed/unperformed:** remaining M3 checks or optional follow-ons.
- **Unresolved:** historical novelty, application mapping, and any unclosed numerical interval.

A claim ledger must pair every claimed contribution with its closest prior art, baseline, metric, certificate or measurement, and permitted scope. If a practical accuracy/latency claim appears, require the missing task, real data, quantizer, deployment measurement, and matched comparison; otherwise remove the claim. Do not infer that a finite mathematical source range captures real semantic importance merely because it is called semantic in the equations.

For figures, generate ordinary reproducible plots from exact formula tables: primary and companion height frontiers at \(r/L=4\) and 5, the additive/strip frontier, and a two-parameter phase map if useful. Label analytical curves, LP points, and numerical intervals distinctly. Include B10 on the primary optimum and B6 on the companion optimum, even though they coincide. Mark the different \(H=0\) values. Retain the source table alongside every plot; plots explain the result and do not prove it.

### 11.3 Residual uncertainty

The theorem is narrow and the proof is not formally machine-checked. Search coverage cannot establish priority. Sources behind access barriers and unindexed equivalents remain possible. Native M3 installation, measured memory, and full clean-room replay are still future work. The core proof has no real dataset, quantizer, application loss, or demonstrated deployment value. Smooth attainment under additional fixed derivative bounds, asymmetric amplitudes, ordinary original-coordinate rectangles, nonlinear fields, and multiple protected directions are not established by this document.

These limitations do not require a large speculative implementation. They define the boundary of the completed research and the gates for any subsequent project.

## 12. References, inspection record, and search coverage

**Research cutoff and access date: 28 October 2026.** “Inspected” means that the relevant primary text, theorem, definition, or documentation section was opened and read during this investigation. It does not mean every proof was formally verified, every implementation was executed, or a target-machine installation was tested. Abstract-only and inaccessible sources are marked. The classification distinguishes sources that directly support established machinery from sources that merely resemble the present problem.

### 12.1 Supplied source and mathematical references

**[P0] User-supplied report.** *Robustness to straight probes while preserving meaningful change: An autonomous mathematical investigation inspired by quantization*, dated 28 October 2026; attachment `straight_probe_research.pdf`. No named author or redistribution license was supplied. The PDF's statements, equations, proof constructions, figures, references, and reported computational evidence were inspected. Its strip theorem is the direct mathematical starting point; its global model supplies controls. Its original code and raw experimental outputs were **not attached**. Newly executed checks in this plan are therefore separate reconstructions, not execution of its original repository. Attachment SHA-256: `7a3e5b2ffb0a3395a553f3c1e2c23261ba9633588d2791b542b039093cf47c3f`.

**[R1] Holger Dette, Viatcheslav B. Melas, and Petr Shpilev.** *Optimal designs for estimating individual coefficients in polynomial regression with no intercept*. Statistics & Probability Letters **158** (2020), 108636; preprint 2019. DOI: [10.1016/j.spl.2019.108636](https://doi.org/10.1016/j.spl.2019.108636). [Full preprint](https://arxiv.org/pdf/1906.08343). **Inspected:** Theorem 2.1, equation (3.2), Theorem 3.1, and low-degree examples. **Direct support:** established no-intercept extremal polynomials, Elfving geometry, support points and weights underlying the global controls. **Limit:** neither the protected-strip nor compact-height probe theorem is stated here; preserving the no-intercept assumption is essential.

**[R2] Stephen P. L. Diliberto and Ernst G. Straus.** *On the approximation of a function of several variables by the sum of functions of fewer variables*. Pacific Journal of Mathematics **1**(2) (1951), 195–210. [Original publisher PDF](https://msp.org/pjm/1951/1-2/pjm-v1-n2-p04-s.pdf). **Inspected:** Problems A/B, permissible paths, Theorem 1, and levelling construction. **Related established mechanism:** closed-path obstructions in uniform approximation. **Limit:** its separate coordinate functions are independent; this is not automatically the shared-potential or partially ordered probe problem. It supplies ancestry, not the exact formula claimed in this handoff.

**[R3] Richard M. Karp.** *A Characterization of the Minimum Cycle Mean in a Digraph*. Berkeley report UCB/ERL M77/47, 28 June 1977; Discrete Mathematics **23** (1978), 309–311. [Original report PDF](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1977/Archive/ERL-m-77-47.pdf). **Inspected:** the full short report, Theorem 1 and dynamic-programming recurrence. **Direct support:** classical finite cycle-mean characterization and computation. **Limit:** this plan's semantic edges have zero probe-count denominator, so its mixed graph needs a cycle ratio or threshold feasibility, not an ordinary mean over all edges. Finite graph results do not establish Euclidean continuity.

**[R4] Nikolai Krivulin.** *Using Tropical Optimization Techniques to Evaluate Alternatives via Pairwise Comparisons*. Proceedings of the Seventh SIAM Workshop on Combinatorial Scientific Computing (2016), 62–72. DOI: [10.1137/1.9781611974690.ch7](https://doi.org/10.1137/1.9781611974690.ch7). [Full author preprint](https://arxiv.org/pdf/1503.04003), version 3 dated 28 August 2016. **Inspected:** Sections 3.3–3.5 and 4.1, Lemma 1 and Theorem 2. **Direct support:** common-potential Chebyshev fitting on additive/log scales, spectral-radius optimal values and Kleene-star minimizers. **Limit:** finite pairwise matrices do not evaluate this continuum geometry or enforce its semantic condition. This source defeats a generic potential-fitting novelty claim.

**[R5] Nikolai Krivulin.** *Application of log-Chebyshev approximation and tropical algebra to multicriteria problems of pairwise comparisons*. arXiv:2601.17999v1, 25 January 2026. [Full HTML](https://arxiv.org/html/2601.17999v1). **Inspected:** Section 2.1 equation (1), Sections 4.1–4.2 and Lemma 1. **Direct support/current followup:** explicit log-Chebyshev objectives and tropical solutions in a wider comparison framework. **Limit:** a preprint and a different finite problem; no protected-strip formula. Its broad informal description of optimization difficulty is not imported: the basic log-variable Chebyshev formulation is a convex LP.

**[R6] Classical difference-constraint networks; inspected accessible statement.** Rina Dechter, Itay Meiri, and Judea Pearl, *Temporal constraint networks*, Artificial Intelligence **49**(1–3) (1991), 61–95, DOI [10.1016/0004-3702(91)90006-6](https://doi.org/10.1016/0004-3702(91)90006-6). The original publisher abstract and author bibliography were inspected; the [university full-text mirror](https://cse.unl.edu/~choueiry/Documents/DechterMeiri-AIJ.pdf) repeatedly timed out. Luke Hunsberger, *Algorithms for a Temporal Decoupling Problem in Multi-Agent Planning*, AAAI 2002, provides an [accessible primary PDF](https://cdn.aaai.org/AAAI/2002/AAAI02-071.pdf): Definitions 1–3, Theorem 1 and Fact 2 were inspected. **Direct support:** finite inequalities between node values are feasible exactly when the corresponding distance graph has no negative cycle; the accessible paper attributes that standard theorem to Dechter–Meiri–Pearl. **Limit:** scheduling applications differ, and neither source proves continuum attainment here.

**[R7] Rasmus Kyng, Anup Rao, Sushant Sachdeva, and Daniel A. Spielman.** *Algorithms for Lipschitz Learning on Graphs*. COLT 2015, PMLR **40**, 1190–1223. [Paper page](https://proceedings.mlr.press/v40/Kyng15.html); [full PDF](https://proceedings.mlr.press/v40/Kyng15.pdf). **Inspected:** equation (1), definitions of minimal versus lexicographic extension, Sections 4.3 and 5, and directed-graph results. **Direct comparator:** once protected-strip labels are fixed at \(G=\mu v\), the finite unit-edge problem is exactly a minimal Lipschitz extension problem. **Limit:** proving those fixed labels lose nothing and producing a continuous geometric extension remain separate work. The paper's [YINSlex code link](https://github.com/danspielman/YINSlex) was identified; its implementation was not executed or audited. Absolutely minimal/lexicographic extension is a stronger objective than this plan needs.

**[R8] Mahmood Ettehad and Simon Foucart.** *Instances of Computational Optimal Recovery: Dealing with Observation Errors*. SIAM/ASA Journal on Uncertainty Quantification **9**(4) (2021), 1438–1456. DOI: [10.1137/20M1328476](https://doi.org/10.1137/20M1328476). [Full preprint](https://arxiv.org/pdf/2004.00192). **Inspected:** local/global error definitions, compatible-output sets, Section 2 equation (9), and Theorem 1. **Direct support:** local minimax recovery is a Chebyshev-center problem, with constructive LP formulations for polyhedral models. **Limit:** its observation model differs. Scalar predecessor-interval centering is established machinery; the particular geometric optimum and lower bound in this plan are derived here.

**[R9] Michiel de Bondt.** *Homogeneous quasi-translations in dimension 5*. Contributions to Algebra and Geometry **59** (2018), 295–326; online 13 October 2017. DOI: [10.1007/s13366-017-0358-2](https://doi.org/10.1007/s13366-017-0358-2). [Publisher full text](https://link.springer.com/article/10.1007/s13366-017-0358-2). **Inspected:** Definition 1.1 and Section 2 equivalences. **Related background:** special algebraic maps whose inverse has the expected straight-step form. **Limit:** the probes here fail that special property; this source supplies terminology and contrast, not a minimax formula.

**[R10] Ernst Hairer.** *Geometric Numerical Integration*, TU München lectures, January–February 2010: [Lecture 1: Hamiltonian systems](https://www.unige.ch/~hairer/poly_geoint/week1.pdf) and [Lecture 2: Symplectic integrators](https://www.unige.ch/~hairer/poly_geoint/week2.pdf). **Inspected:** the first-integral definition and conservation context in Lecture 1, and numerical integration, Euler-type schemes, quadratic-integral and symplecticity context in Lecture 2. **Related background:** exact-flow structure and its possible loss under discretization. **Limit:** the specific drift, inverse identities and source-domain claims in this plan are directly calculated; no lecture theorem is being presented as this problem's exact solution.

**[R11] Efe A. Ok.** *Order-preserving extensions of Lipschitz maps*. Journal of the Australian Mathematical Society **118**(1) (2025), 91–107; online 17 December 2024. DOI: [10.1017/S1446788724000120](https://doi.org/10.1017/S1446788724000120). [Full preprint, titled *Monotonic Extensions of Lipschitz Maps*](https://arxiv.org/pdf/2305.00511). **Inspected:** Theorems 2 and 4, the two-point counterexample, Remark 5 and later corollaries. **Direct caution:** simultaneous preservation of order and Lipschitz constants requires geometric compatibility. **Limit:** no conclusion about this special slab follows without checking hypotheses; the explicit optimizer, not a blanket extension theorem, provides continuity here.

**[R12] Edoardo Gargiulo and Efe A. Ok.** *Order-Preserving Extensions of Hadamard Space-Valued Lipschitz Maps*. arXiv:2603.03549v1, submitted 3 March 2026. [Full HTML](https://arxiv.org/html/2603.03549v1). **Inspected:** introduction, nonextension result, Theorems 2–3 and Section 5. **Related current followup:** positive and negative ordered-extension results under specified domain/codomain conditions. **Limit:** a preprint with different hypotheses; it warns against an unjustified general interpolation argument, not against the explicit constructions proved here.

**[R13] Andrea Davini, Albert Fathi, Renato Iturriaga, and Maxime Zavidovique.** *Convergence of the solutions of the discounted equation: the discrete case*. Mathematische Zeitschrift **284** (2016), 1021–1034. [Author-repository record](https://cvgmt.sns.it/paper/6143/); [full PDF](https://cvgmt.sns.it/media/doc/paper/6143/DFIZb_2016.pdf). **Inspected:** introduction, Theorems 1.1 and 2.1, and closed-measure context. **Related established mechanism:** critical costs, potential inequalities and measures that annihilate coboundaries. **Limit:** its compact-space/cost hypotheses are not automatically those of a source-restricted relation with stripwise semantics. It does not supply the exact height frontier.

**[R14] Jerry Chee, Arturs Backurs, Rainie Heck, Li Zhang, Janardhan Kulkarni, Thomas Rothvoss, and Sivakanth Gopi.** *DiscQuant: A Quantization Method for Neural Networks Inspired by Discrepancy Theory*. COLT 2025, PMLR **291**, 924–951. [Publisher paper page](https://proceedings.mlr.press/v291/chee25a.html); [linked implementation](https://github.com/jerry-chee/DiscQuant). **Inspected:** publisher abstract and implementation link only; full paper and code were not audited or executed. **Contextual comparator:** actual fixed-grid neural-network quantization has a different task and metric. **Limit:** no model-accuracy, speed, or compression result is imported as evidence for the straight-probe objective.

### 12.2 Implementation and hardware references

The following are official documentation or maintainer-uploaded distribution records inspected on the cutoff date. Availability of a wheel or API is distinct from a successful joint installation and from performance measured on the user's Mac.

**[R15] SciPy developers.** [`linprog(method='highs')` API documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html), live documentation identifying SciPy 1.18.0 when inspected. **Inspected/direct support:** solver choices, arguments, default nonnegative bounds, status, residual and tolerance documentation. The executing research environment used SciPy 1.17.0; the proposed local release must pass its own locked smoke test. Solver success and floating residuals are not exact optimality certificates.

**[R16] Python Software Foundation.** [`fractions` — Rational numbers](https://docs.python.org/3/library/fractions.html), live Python 3.15.2 documentation when inspected. **Inspected/direct support:** exact rational construction, string versus binary-float input, and denominator approximation. The kernel uses longstanding operations available in Python 3.12; it does not depend on newly added 3.15 features. `limit_denominator` suggests a rational candidate and does not prove it satisfies an exact certificate.

**[R17] Apple.** [*MacBook Pro (14-inch, M3 Pro or M3 Max, Nov 2023) — Tech Specs*](https://support.apple.com/en-us/117736), model introduced in 2023; official page inspected 28 October 2026. **Inspected/direct support:** the available M3 Max CPU/GPU and unified-memory configurations. **Limit:** this establishes that configurations vary. It neither identifies the user's exact model nor measures its present available memory, software, thermal behavior, or performance.

**[R18] NumPy developers and maintainers.** [Official NumPy distribution metadata](https://pypi.org/project/numpy/), release **2.5.4**, 10 October 2026. **Inspected/direct support:** Python/platform distribution entries, macOS arm64 wheels including a CPython 3.12/macOS 14+ build, and the distribution's compound license expression. **Limit:** NumPy 2.3.5 executed in the research workspace. The newer release was not jointly installed with the proposed SciPy release or tested on the target Mac.

**[R19] SciPy developers and maintainers.** [Official SciPy distribution metadata](https://pypi.org/project/scipy/), release **1.18.1**, 21 August 2026. **Inspected/direct support:** Python requirements, macOS arm64 distributions, including CPython 3.12/macOS 14+ and other supported builds, and BSD license metadata. **Limit:** release availability is not local compatibility or a runtime forecast. Lock versions and wheel hashes only after the target resolver and numerical smoke checks pass.

**[R20] HiGHS/ERGO-Code developers.** [HiGHS project](https://highs.dev/); [official `highspy` distribution](https://pypi.org/project/highspy/), release **1.15.1**, 2 July 2026; [stable Python interface](https://ergo-code.github.io/HiGHS/stable/interfaces/python/); [stable options definitions](https://ergo-code.github.io/HiGHS/stable/options/definitions/). **Inspected/direct support:** native macOS arm64 distribution support, MIT license, initialization, option setting, status/solution access, thread and tolerance controls, and basis-related facilities. **Limit:** direct `highspy` was not executed here; the local sparse LP smoke test used SciPy's interface. Optional facilities are not required for the exact core and must be checked against the installed version if used.

### 12.3 Established varying-coefficient class

**[R21] Lan Xue and Annie Qu.** *Variable Selection in High-dimensional Varying-coefficient Models with Global Optimality*. Journal of Machine Learning Research **13** (2012), 1973–1998. [Full publisher PDF](https://jmlr.csail.mit.edu/papers/volume13/xue12a/xue12a.pdf). **Inspected:** introduction and Section 2 equation (1). **Direct support:** the established class in which a covariate enters linearly while its coefficient depends on another variable. The form \(k(u)v+b(u)\) is a simple instance. **Limit:** the paper's statistical estimation, variable selection, consistency and convergence-rate results are different questions and are not imported. The exact coefficients attaining this probe problem are derived in this investigation, not attributed to this source.

**[R22] Trevor Hastie and Robert Tibshirani.** *Varying-Coefficient Models*. Journal of the Royal Statistical Society, Series B **55**(4) (1993), 757–779; discussion continues through page 796. DOI: [10.1111/j.2517-6161.1993.tb01939.x](https://doi.org/10.1111/j.2517-6161.1993.tb01939.x). [Publisher page](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1993.tb01939.x). **Inspected:** publisher abstract and metadata; original full-PDF access failed. **Historical class attribution:** coefficients that vary with other predictors are established modeling machinery. **Limit:** no detailed theorem or probe-specific optimality result is asserted from this abstract-only inspection; [R21] supplies the inspected explicit mathematical definition.

### 12.4 Search coverage, adjacent work, and remaining blind spots

Searches used two web search systems and followed consequential results to original publisher pages, author repositories or full primary preprints. Search engines and secondary indexes helped locate material; their snippets were not treated as theorem proofs. Exact source inspection statuses above are part of the evidence, not a claim of exhaustive historical review.

| Search family | Terms and reformulations covered | Consequential result |
|---|---|---|
| Original geometry | Straight probes, Euler steps, approximate invariance, `x+t,y+xt`, strip minimax, quadratic/plateau behavior | No exact duplicate identified; [P0] remains the direct source. |
| Common potentials | Uniform approximation by coboundaries, common potential differences, log-Chebyshev fitting, tropical optimization | [R4–R5] directly establish the finite fitting machinery. |
| Dual certificates | Cycle means and ratios, difference constraints, shortest-path potentials, temporal networks | [R3, R6] establish the finite graph mechanism. |
| Approximation/extension | Diliberto–Straus closed paths and bolts, sums of two algebras, scalar minimax recovery, graph Lipschitz extension | [R2, R7–R8] explain the known lower/upper mechanisms; continuity cannot be inferred from a mesh alone. |
| Variational and order formulations | Weak KAM critical values, closed measures, ergodic optimization, monotone/ordered metric extension, quotient metrics | [R11–R13] supply related frameworks and hypothesis checks, not an exact equivalent formula. |
| Compact and protected refinements | Bounded source heights with global-in-height strip semantics, predecessor/common-target conflicts, parabolic extension, additive separability, conditional monotonicity | No exact same-assumption phase diagram or separation located; the decisive upper mechanism is established extension. |
| Simple expressive alternatives | Varying coefficients, affine dependence on height with a horizontal coefficient, sums of univariate functions | [R21–R22] prevent claiming a new architecture; the simple varying-coefficient comparator already attains the optimum. |
| Global and practical branches | No-intercept c-optimal design, polynomial/trigonometric followups, quasi-translations, discrepancy quantization | [R1, R9–R10, R14] distinguish known ingredients and different application objectives. |
| Local execution | Official package distributions, sparse solver APIs, exact arithmetic, Apple specifications | [R15–R20] support a conditional native CPU execution path, not an M3 benchmark claim. |

The forward check included 2026 work. [R5] and [R12] were inspected. Two adjacent 2026 items remain access limitations: Petr Shpilev and Viatcheslav B. Melas, *Optimal designs for estimating individual coefficients in trigonometric regression with no intercept*, Statistics & Probability Letters **237** (2026), 110825, DOI [10.1016/j.spl.2026.110825](https://doi.org/10.1016/j.spl.2026.110825), was visible through publisher abstract/intro search material but full access returned 403; Arindam Mandal, *Group Actions, Fixed Points and Orbit Quotients in Lipschitz Spaces*, [arXiv:2609.25931](https://arxiv.org/abs/2609.25931), September 2026, was a metadata/abstract alert whose primary full text could not be retrieved. Neither is used for a technical conclusion.

Additional inspected neighboring material included Asgarova–Ismailov's [2016 generalization of Diliberto–Straus](https://arxiv.org/pdf/1603.07073), whose convergence assumptions reinforced the need for explicit continuum certificates. McShane's original 1934 paper, DOI [10.1090/S0002-9904-1934-05978-0](https://doi.org/10.1090/S0002-9904-1934-05978-0), was identified but the AMS full text returned 403; it is not counted as inspected. The primary sources [R7–R8] already support the extension machinery needed here.

The stopping point was reached when geometry-specific searches returned no further material exact match, while adjacent searches converged on already inspected extension, potential-fitting and varying-coefficient mechanisms. The ensuing work was proof challenge and construction of the strongest simple comparator. This is sufficient to reject broad solver/architecture novelty claims and to define a narrow candidate representation theorem. It does **not** establish historical priority. Inaccessible papers, unindexed or unpublished work, non-English monographs, and equivalent formulations under other coordinates remain blind spots. A prior equivalent result would reclassify the contribution as replication; an immediate specialization of known machinery would reduce its significance without, by itself, proving prior publication. No patentability or freedom-to-operate opinion is offered.

## 13. Standalone exact reference kernel

This is the actual small standard-library reference used in the investigation, supplied in full so the handoff does not depend on intermediate files. Save the following Python block as `reference_kernel.py` and run the existing interpreter command:

```bash
python3 reference_kernel.py --output kernel_results.json
```

Use ordinary Python execution, not `python -O`: the verifier deliberately refuses execution when assertions are disabled. It accepts exact integer, rational-string, decimal-string, or `Fraction` inputs and rejects binary-float parameters. The project factory in Section 8 must enforce the method/contract metadata; this numerical core does not accept a contract argument in its readout constructor.

The kernel includes B0, B1, B2S, B3, B5, B6, B9 and B10, primary internal and shared exterior certificate builders, exact incidence/cancellation checks, coordinate identities, and a deterministic development regression suite. B2's original construction, B4's trivial bound selector, the companion's irrational internal witness, graph LPs, and the final confirmation-manifest generator are specified elsewhere in this document and remain work packages. The included suite is already seen development evidence, not the fresh confirmation protocol.

The saved kernel SHA-256 is `8870136f65e6cfde1f9c5261b8edf8979dee08fb2ebc4d20ff17ff67c9afd777`. Its executed Linux/Python 3.12.14 result was: 324 parameter cases, 10,044 dyadic probe triples, 80,352 exact method inequalities, 30,132 full-strip source identities, 30,132 compact source identities, 438 exact certificate checks, 10,044 coordinate identities, four rejected invalid-parameter fixtures, and zero failures. These counts describe finite code checks; the continuum claims rest on Section 4's proofs. A clean target run should reproduce the counts and exact values, with its own environment metadata.

```python
"""Exact reference for straight-probe robustness; standard library only.
Finite regression corroborates this code. The handoff's inequalities prove the continuum claims.
Run: python3 reference_kernel.py --output kernel_results.json
"""
from dataclasses import dataclass
from fractions import Fraction as F
from collections import defaultdict
from pathlib import Path
import argparse
import json
import platform
import random

if not __debug__:
    raise RuntimeError("Do not run this exact verifier with Python -O or PYTHONOPTIMIZE")

PROTECTED = "protected_strip_v1"
COMPACT = "compact_parabolic_slab_v1"

def rational(value):
    if isinstance(value, bool) or not isinstance(value, (F, int, str)):
        raise TypeError("Use Fraction, integer, or exact rational/decimal string; no floats")
    return F(value)

@dataclass(frozen=True)
class Parameters:
    L: F
    r: F
    H: F
    mu: F = F(1)
    def __post_init__(self):
        for name in ("L", "r", "H", "mu"):
            object.__setattr__(self, name, rational(getattr(self, name)))
        if min(self.L, self.r, self.H) < 0 or self.mu <= 0:
            raise ValueError("Require finite L,r,H>=0 and mu>0")

def constants(p):
    Q = p.r*p.r/2
    I = min(Q, 2*p.L*p.L)
    S = Q if p.r <= 2*p.L else max(2*p.L*p.L, p.L*(p.r-p.L))
    T = min(S, max(I, p.H/2))
    C = min(p.H, S, max(I, p.H/2))
    esat = max(min(p.H, I), min(p.H, Q)/2)
    assert C == min(S, esat)
    return dict(Q=Q, I=I, S=S, T=T, C=C, esat=esat,
                R_protected=p.mu*T, R_compact=p.mu*C)

def clip(value, low, high):
    if low > high:
        raise ValueError("Unordered clamp endpoints")
    return min(max(value, low), high)

def probe(point, amplitude):
    u, v = map(rational, point)
    t = rational(amplitude)
    return (u+t, v-t*t/2)

def probe_source(p, point):
    u, v = point
    return abs(u) <= p.L and 0 <= v <= p.H

def semantic_point(p, point, contract):
    u, v = point
    if contract == PROTECTED:
        return abs(u) <= p.L
    if contract == COMPACT:
        return abs(u) <= p.L and 0 <= v <= p.H
    raise ValueError("Unknown semantic contract")

def semantic_pair(p, low, high, contract):
    return (low[0] == high[0] and low[1] <= high[1]
            and semantic_point(p, low, contract)
            and semantic_point(p, high, contract))

def q_bounds(p, w):
    return (max(abs(w)-p.L, F(0))**2/2,
            min(p.r, abs(w)+p.L)**2/2)

METHODS = {
    "tangent": (PROTECTED, COMPACT),
    "original_y": (PROTECTED, COMPACT),
    "strip_simple": (PROTECTED, COMPACT),
    "protected_interval": (PROTECTED, COMPACT),
    "protected_affine": (PROTECTED, COMPACT),
    "clip": (COMPACT,),
    "sat_optimal": (COMPACT,),
    "compact_optimal": (COMPACT,),
}

@dataclass(frozen=True)
class Readout:
    parameters: Parameters
    method_id: str
    def __post_init__(self):
        if self.method_id not in METHODS:
            raise ValueError("Unknown readout")
    def value(self, u, v):
        p = self.parameters
        u, v = rational(u), rational(v)
        c = constants(p)
        qm, qp = q_bounds(p, u)
        m = self.method_id
        if m == "tangent":
            return p.mu*v
        if m == "original_y":
            return p.mu*(v+u*u/2)
        if m == "compact_optimal":
            m = "strip_simple" if c["S"] <= c["esat"] else "sat_optimal"
        if m == "strip_simple":
            return p.mu*(v+max(F(0), qp-c["S"]))
        if m == "clip":
            return p.mu*clip(v, F(0), p.H)
        if m == "sat_optimal":
            return p.mu*(clip(v, F(0), p.H)+max(F(0), min(p.H, qp)-c["esat"]))
        e = c["T"]
        if m == "protected_interval":
            a, b = max(F(0), v+qm), min(p.H, v+qp)
            return p.mu*clip(v, b-e, a+e)
        if e == c["S"]:
            k, b = F(1), max(F(0), qp-e)
        else:
            assert c["I"] > 0 and 2*e-c["I"] > 0 and p.H <= 2*e
            D = max(F(0), qp-qm)
            k = clip((2*e-D)/(2*e-c["I"]), F(0), F(1))
            b = max(F(0), (1-k)*p.H+k*qp-e)
        return p.mu*(k*v+b)
    def analytic_bound(self):
        p, c, m = self.parameters, constants(self.parameters), self.method_id
        if m == "original_y":
            return p.mu*p.L*p.r
        key = {"tangent": "Q", "strip_simple": "S", "sat_optimal": "esat",
               "compact_optimal": "C", "protected_interval": "T",
               "protected_affine": "T"}.get(m)
        return p.mu*(min(p.H, c["Q"]) if m == "clip" else c[key])
    def semantic_certificate(self):
        return {"method_id": self.method_id,
                "admissible_contracts": METHODS[self.method_id],
                "type": "analytic_formula", "slope": str(self.parameters.mu)}

def certificate(p, nodes, probes, semantic_links):
    # Each probe is (source_id, amplitude, target_id, sign), representing
    # sign*(G(target)-G(source)) <= E. Semantic links are (low_id,high_id).
    return dict(nodes=nodes, probes=probes, semantic_links=semantic_links)

def verify_certificate(p, cert, contract):
    coeff = defaultdict(F)
    count, height = 0, F(0)
    nodes = {name: tuple(map(rational, point)) for name, point in cert["nodes"].items()}
    for source, t, target, sign in cert["probes"]:
        t = rational(t)
        a, b = nodes[source], nodes[target]
        assert sign in (-1, 1)
        assert probe_source(p, a) and abs(t) <= p.r and probe(a, t) == b
        coeff[b] += sign
        coeff[a] -= sign
        count += 1
    for low, high in cert["semantic_links"]:
        a, b = nodes[low], nodes[high]
        assert semantic_pair(p, a, b, contract)
        coeff[a] += 1
        coeff[b] -= 1
        height += b[1]-a[1]
    # Keys are exact coordinate pairs, so coincident named nodes cannot evade cancellation.
    assert count > 0 and all(value == 0 for value in coeff.values())
    return dict(probe_count=count, semantic_height=height, lower_bound=p.mu*height/count)

def protected_internal_witness(p):
    a = min(p.r, 2*p.L)
    d = a*a/2
    nodes = {"A": (-a/2, F(0)), "B": (a/2, F(0)),
             "C": (a/2, -d), "D": (-a/2, -d)}
    return certificate(p, nodes, [("A", a, "C", -1), ("B", -a, "D", -1)],
                       [("D", "A"), ("C", "B")])

def exterior_witness(p):
    if p.L == 0 or p.r < 2*p.L or p.H < 2*p.L*p.L:
        return None
    d = min(p.H, 2*p.L*(p.r-p.L))
    w = d/(2*p.L)
    z = p.H-(w+p.L)**2/2
    nodes = {"A": (-p.L, p.H), "B": (p.L, p.H-d),
             "C": (p.L, p.H), "D": (-p.L, p.H-d),
             "P": (w, z), "M": (-w, z)}
    probes = [("A", w+p.L, "P", -1), ("B", w-p.L, "P", 1),
              ("C", -w-p.L, "M", -1), ("D", -w+p.L, "M", 1)]
    return certificate(p, nodes, probes, [("D", "A"), ("B", "C")])

def exercise():
    rng = random.Random(28102026)
    counts = dict(parameter_cases=0, sampled_edges=0, readout_inequalities=0,
                  protected_source_identities=0, compact_source_identities=0,
                  certificate_verifications=0, coordinate_identities=0)
    for L in map(F, [0, 1, 2]):
      for r in map(F, ["0", "1/2", "1", "2", "5/2", "3", "4", "5", "9"]):
        for H in map(F, ["0", "1/8", "1/2", "1", "2", "3", "4", "5", "6", "8", "10", "20"]):
          p = Parameters(L, r, H, F(3, 2)); c = constants(p)
          lower = verify_certificate(p, protected_internal_witness(p), PROTECTED)["lower_bound"]
          counts["certificate_verifications"] += 1
          ext = exterior_witness(p)
          if ext is not None:
              for contract in (PROTECTED, COMPACT):
                  result = verify_certificate(p, ext, contract)
                  counts["certificate_verifications"] += 1
              lower = max(lower, result["lower_bound"])
          assert lower == c["R_protected"]
          assert (c["T"] < c["S"]) == (L > 0 and r > 3*L and H < 2*c["S"])
          methods = [Readout(p, name) for name in METHODS]
          for _ in range(31):
              u, v, t = L*F(rng.randrange(-32, 33), 32), H*F(rng.randrange(33), 32), r*F(rng.randrange(-32, 33), 32)
              target = probe((u, v), t)
              assert probe_source(p, (u, v))
              for method in methods:
                  assert abs(method.value(*target)-method.value(u, v)) <= method.analytic_bound()
                  counts["readout_inequalities"] += 1
              outside_v = F(rng.randrange(-3200, 3201), 32)
              for name in ("protected_interval", "protected_affine", "strip_simple"):
                  assert Readout(p, name).value(u, outside_v) == p.mu*outside_v
                  counts["protected_source_identities"] += 1
              for name in ("clip", "sat_optimal", "compact_optimal"):
                  assert Readout(p, name).value(u, v) == p.mu*v
                  counts["compact_source_identities"] += 1
              x, y = u, v+u*u/2
              xx, yy = x+t, y+x*t
              assert (xx, yy-xx*xx/2) == target
              counts["coordinate_identities"] += 1
              counts["sampled_edges"] += 1
          counts["parameter_cases"] += 1
    # Additive four-residual cancellation: nonlinear A and B, full-height semantic slope.
    p = Parameters(1, 4, 0, "3/2"); w = p.r-p.L
    Q, q = p.r*p.r/2, (p.r-2*p.L)**2/2
    A = lambda v: p.mu*v+v**3
    B = lambda u: u**3-2*u
    G = lambda u, v: A(v)+B(u)
    residuals = [G(w,-Q)-G(-p.L,0), G(w,-q)-G(p.L,0),
                 G(-w,-Q)-G(p.L,0), G(-w,-q)-G(-p.L,0)]
    alternating = residuals[0]-residuals[1]+residuals[2]-residuals[3]
    assert alternating == 2*(A(-Q)-A(-q)) <= -4*p.mu*p.L*(p.r-p.L)
    # Contract-sensitive zero-height fixture; neither contract may be silently substituted.
    c = constants(Parameters(1, 4, 0))
    assert c["R_protected"] == 2 and c["R_compact"] == 0
    invalid = [lambda: Parameters(-1,1,1), lambda: Parameters(1,1,1,0),
               lambda: Parameters(1.0,1,1), lambda: Parameters("NaN",1,1)]
    for make in invalid:
        try:
            make()
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        else:
            raise AssertionError("Invalid parameters accepted")
    return dict(environment=platform.platform(), python=platform.python_version(),
                seed=28102026, counts=counts, additive_alternating_sum=str(alternating),
                validation_rejections=len(invalid), failures=0,
                note="Finite exact checks; continuum claims use the handoff proofs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="kernel_results.json")
    args = parser.parse_args()
    result = exercise()
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
```
