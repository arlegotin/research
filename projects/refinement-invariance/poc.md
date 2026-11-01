# Research and proof-of-concept plan: refinement-invariant learning for operator circuits

**Research date:** 31 October 2026  
**Target:** A MacBook Pro with an M3 Max; actual RAM, storage and macOS are not yet known  
**Status:** Research and planning completed; small verification experiments executed on Linux, not on the target Mac  
**Primary deliverable:** A reproducible mathematical and numerical study of the obstruction to local refinement-invariant learning, including a sharp quantitative relaxation  
**Priority claim:** Unconfirmed. No new generally superior optimizer is established.

## 1. Executive assessment and selected direction

The supplied notebook [R0] contains a mathematically credible obstruction, not evidence that a proposed optimizer outperforms existing methods. Independent examination found no gap in its main classification, nilpotent-stalling result, polynomial degree bound, Lipschitz slowdown, or exponential update identity under the stated assumptions. Every executable notebook cell was rerun successfully.

**Select a theory-and-diagnostics project:** reproduce the obstruction, show that fewer compatibility assumptions suffice, establish the quantitative identity-insertion result below, and test whether the resulting local compromises have useful behavior on more general rewrites. Treat contextual natural-gradient methods as established escape controls. Do not start by implementing a large neural architecture, training a language model, or presenting the trace-polynomial rule as a competitive optimizer.

Three findings determine this recommendation.

1. **The obstruction is stronger than the notebook's formulation suggests.** Compatibility with inserting an identity factor on either side, at any one fixed positive mass ratio, already implies its full refinement compatibility. Arbitrary mass divisions need not be assumed separately.
2. **A sharp quantitative relaxation is available.** Once the identity response is fixed, an explicit midpoint rule minimizes the worst discrepancy between those two identity insertions. Under Euclidean identity calibration, it is the average of the classical left- and right-invariant matrix gradients.
3. **The improvement has a strict boundary.** The midpoint rule halves the worst defect on the normalized shear witness, but an exact two-dimensional example makes its defect positive while ordinary local GD has zero defect. A claim of uniform improvement over general factorizations is already false.

The main potentially differentiated contribution is the **precise obstruction, weaker sufficient assumptions, quantitative defect characterization, and reproducible evaluation**, rather than a new matrix-gradient formula. Polynomial concomitants and trace invariants are classical; one-sided matrix metrics are classical; contextual deep-linear natural gradients, quotient methods, ScaledGD, and full invertible fixed-depth gauge methods such as LoRA-RITE already exist [R1–R12].

### What this project can establish

- A self-contained theorem package, its exact scope, and small independently reproducible checks.
- A sharp minimum defect for two identity insertions with an explicitly prescribed identity response.
- A measured tradeoff among general rewrite sensitivity, descent, numerical conditioning, runtime, and contextual information.
- A positive, negative, or inconclusive finding about a declared distribution of general rewrites.
- If earlier evidence warrants it, whether observed rewrite sensitivity affects controlled training trajectories.

It cannot establish historical priority by search alone, minimal contextual information by exhibiting one sufficient construction, or practical superiority of structured models from dense synthetic examples. No evidence presently supports a universal replacement for matrices or a breakthrough optimizer.

### What was actually checked during this investigation

The execution environment was Linux x86_64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0. These are execution facts, not a verified Mac installation recipe. The notebook SHA-256 was:

~~~text
a50d87b15fcaa3a9d4bf7bb49cc18844750fe1f697c461ff5eea3e9510cc8758
~~~

| Executed check | Result | Interpretation |
|---|---:|---|
| Notebook: 300 polynomial refinement checks | Maximum relative residual 6.3250e-16 | Numerical consistency |
| Notebook: 300 simultaneous exponential checks | Maximum relative residual 4.7142e-16 | Numerical consistency |
| Notebook: trace-square descent identity | Maximum relative residual 2.2204e-16 | Numerical consistency |
| Notebook: shear at identity | Gradient norm 1; rule update exactly 0 | Reproduces the obstruction witness |
| Same shear, dense GD with step 0.1 | Loss 0.5 to 0.405 | Nonstationarity is genuine |
| Notebook: cycle, n=10 and epsilon=0.1 | Dissipation about 1e-18; dense value 0.1 | Fixed-normalization slowdown, not training benchmark |
| Notebook: contextual dense-flow identity | Residual 2.3551e-16 | Extra context removes this obstruction |
| New midpoint proposition: 400 random cases | Maximum relative equality error 3.3307e-16 | Checks the derived formula |
| New arbitrary-split residual formula: 400 random cases | Maximum relative error 1.7820e-15 | Independent algebraic implementation oracle |
| Required adverse general split | GD defect 0; midpoint defect 1.5 | Refutes universal midpoint dominance |
| Secondary notebook branch | 3,279 words / 9,837 pair checks passed | Separate word-curvature result, outside primary project |

The new random checks used seed 3102026 and dimensions 2, 3, 5, 8, with 100 cases per dimension. Within each case, draws occurred in this order: A=I+0.2Z, G=Z, B=I+0.2Z, C=I+0.2Z, and t uniform on (0.01,0.99), with independent standard-normal matrices Z. A,G checked the midpoint formula; B,C,G,t checked equation (18). The notebook used its supplied seeds, including 20261031. The notebook's main relative-error convention was norm(error)/max(1,norm(reference)); companion-coordinate errors were absolute. Numerical checks supplement the proofs. No confirmatory workload comparison, target-Mac measurement, or application benchmark has been performed.

### Intended beneficiaries and practical bottleneck

The inferred beneficiaries are researchers and implementers of trainable linear operator objects, adaptive matrix factorizations, and function-preserving architecture rewrites. The relevant workflow represents a linear map by a product, rewrites that product during training, and wants predictable subsequent learning.

The existence of a mathematical mismatch is established: the same represented map can have different output velocities under local parameter updates. Its frequency and economic importance in a deployed rewrite engine remain unknown. There is no supplied workload, user study, or production trace demonstrating that this is the dominant practical bottleneck. The plan therefore separates mathematical characterization from application-value claims.

## 2. Problem statement, assumptions, and hypotheses

### 2.1 Objects and conventions

Fix an integer n >= 2. All matrices are real; every factor is square and invertible. Use the Frobenius pairing

\[
\langle X,Y\rangle_F=\operatorname{tr}(X^\top Y).
\]

A circuit represents

\[
W=A_1\cdots A_\ell,\quad
P_i=A_1\cdots A_{i-1},\quad
Q_i=A_{i+1}\cdots A_\ell.
\]

Empty products are identities. For a differentiable scalar loss L(W), write G = gradient_W L and

\[
G_i=P_i^\top GQ_i^\top.
\]

A shared local rule receives only its own factor and its ordinary local gradient:

\[
F:GL_n(\mathbb R)\times M_n(\mathbb R)\to M_n(\mathbb R),
\qquad \dot A_i=-m_iF(A_i,G_i).
\]

The positive dimensionless masses m_i sum to M; normalize M=1 in experiments. A split conserves mass. F is continuous and memoryless. It cannot inspect neighboring factors, global gradients, circuit depth, a cached metric, or optimizer history. Such extensions are allowed as separate contextual or stateful methods, with their information explicitly recorded.

For a two-factor representation W=BC and a mass fraction t in (0,1), define the **positive-update residual**

\[
\mathcal R_F(B,C,G,t)
=
F(BC,G)-tF(B,GC^\top)C-(1-t)BF(C,B^\top G).
\tag{1}
\]

The output-velocity discrepancy has the opposite sign and the same norm. Exact refinement compatibility is \(\mathcal R_F=0\) for all admissible inputs. Universal instantaneous descent means

\[
\langle G,F(A,G)\rangle_F\ge0
\quad\text{for every }A,G.
\tag{2}
\]

Strict descent means positive pairing for G != 0. Every matrix G is a possible gradient at a chosen point of a suitable smooth loss, so this is a meaningful universal requirement.

### 2.2 Distinguish four different requirements

| Requirement | What is preserved | Why it matters here |
|---|---|---|
| Algebraic rewrite | W before any update | Necessary, but says nothing about learning |
| Fixed-depth gauge equivariance | Updates under a change of internal basis at fixed factor count | Covered by substantial existing work |
| Refinement compatibility | Output vector field when a factor is split and mass conserved | Notebook's central requirement |
| Exact finite-step compatibility | Product after a specified discretized update | Depends on integrator, old versus refreshed gradients, and state transport |

Do not infer one row from another without a proof. A stateful method's gauge experiment must transform its state as specified by the method. Resetting its moments during one rewrite silently changes the algorithm.

### 2.3 Assumptions that must remain explicit

- The theorem is for all real gradients and all invertible factors, not an average over a dataset.
- The same F is used at every gate in a fixed dimension. F has no mass input other than the external multiplier.
- Continuity establishes zero vector field at the obstruction. Local Lipschitz continuity is needed for the standard uniqueness conclusion for a stationary ODE solution.
- Rectangular factors, singular models, changes of rank, constrained sparsity patterns, and nonlinearities inserted between factors are outside this theorem.
- A structured update must remain in its declared parameter class. Projecting a dense update onto a sparsity mask changes the mechanism and requires a new analysis.
- Floating-point near-nilpotency is not equivalent to exact nilpotency. Test fixtures must be algebraically constructed.
- Exact exponential compatibility is not an unconditional finite-step decrease or global convergence theorem.
- Requiring exact covariance under every bounded-condition basis change, with unrestricted compositions and domain, still generates full GL covariance. A condition cap alone is not an escape.
- Fixing a single mass ratio does not remove the obstruction when both identity-insertion orientations remain admissible.

### 2.4 Hypotheses and discriminating observations

| ID | Statement and status | Support / alternative explanation | Test and weakening observation |
|---|---|---|---|
| H1 | The notebook's exact local obstruction is correct. Proof-level claim, supported by audit. | Companion-matrix proof; apparent numerical violations could be conditioning, transposes, or out-of-scope context. | Independent proof review and exact fixtures. A genuine counterexample satisfying every assumption invalidates it. |
| H2 | The identity-insertion midpoint bound is sharp under prescribed K. Derived and proved here. | Triangle inequality and explicit attaining descent rule. It may be previously known. | Check exact formulas, rotated fixtures, and mass variants. An algebraic mismatch invalidates the formulation; a prior source narrows novelty. |
| H3 | A development-selected local metric compromise reduces general rewrite discrepancy versus the strongest fixed simple local comparator on the declared population. Unproved empirical hypothesis. | The midpoint suggests a useful compromise; the adverse split proves no universal guarantee. | Independent confirmatory cases outside identity insertions. An interval excluding useful improvement, or a gain erased by normalization/tuning, rejects this population-specific claim. |
| H4 | Context can restore invariant descent at tolerable cost for a specified workload. Correctness is established; cost-value is unknown. | Exact lifts and transported metrics. A dense collapsed representation may do the same work more cheaply. | Compare all setup, solves, metric transport, and rewrite costs with anchor and collapse/refactor controls. Dominance by the simpler baseline removes the engineering case. |
| H5 | Lower instantaneous rewrite discrepancy predicts less trajectory disruption without harmful task regressions. Conditional empirical hypothesis. | First-order sensitivity suggests a relation; integrator effects, implicit bias, and rescaling can explain apparent gains. | Paired rewritten/unrewritten training with integrator controls and test loss. Failure to transfer prevents a training-benefit claim. |

A successful proof project does not require H3–H5 to be positive. A practical optimizer claim requires evidence beyond H1–H2.

## 3. Mathematical findings and the selected quantitative extension

### 3.1 Classification from the notebook

Define

\[
K(H)=F(I,H),\quad H=A^\top G,\quad
p_k(H)=\operatorname{tr}(H^k),\quad
p(H)=(p_1,\ldots,p_n).
\]

Exact refinement implies

\[
F(A,G)=AK(A^\top G)=K(GA^\top)A
\tag{3}
\]

and

\[
K(S^{-1}HS)=S^\top K(H)S^{-\top}.
\tag{4}
\]

Equivalently, \(\Psi(H)=K(H)^\top\) is a usual similarity-equivariant matrix concomitant:

\[
\Psi(S^{-1}HS)=S^{-1}\Psi(H)S.
\]

The complete continuous representation is

\[
F(A,G)
=
A\left[\sum_{k=1}^{n}a_k(p(A^\top G))(A^\top G)^{k-1}\right]^\top,
\tag{5}
\]

where a: R^n -> R^n is continuous and unique. Universal descent is equivalent to

\[
a(p)\cdot p\ge0\quad\text{for every }p\in\mathbb R^n.
\tag{6}
\]

**Self-contained proof of the nontrivial representation step.** For any real vector p, Newton's recursion

\[
c_0=1,\qquad
kc_k+\sum_{i=1}^{k}c_{k-i}p_i=0
\]

constructs a real companion matrix C(p) with superdiagonal ones and last row
\((-c_n,-c_{n-1},\ldots,-c_1)\). Its first n power traces equal p. Thus every p is realized, even though some eigenvalues may be nonreal.

Equivariance under the invertible matrices I+sH implies that \(\Psi(H)\) commutes with H. On C(p), the commuting matrix is uniquely determined by its first row because \(e_1^\top C^j=e_{j+1}^\top\) for j<n. Consequently it equals \(\sum_k a_k(p)C^{k-1}\), with \(a_k(p)=\Psi(C(p))_{1k}\). These coefficients are continuous. Every cyclic real matrix is similar to its companion; cyclic matrices are dense because a suitable Krylov determinant is a nonzero polynomial. Similarity and continuity extend the formula to all matrices. Evaluating on companions proves uniqueness. Pairing with H gives equation (6).

This proof avoids importing a compact-group smooth-equivariant theorem into a noncompact continuous setting. Polynomial concomitants themselves are established prior art [R1–R3].

### 3.2 Nilpotent stalling and its exact scope

If a(0)=v != 0, choose p=-sv for small positive s. Continuity gives \(a(-sv)\cdot(-sv)<0\), contradicting equation (6). Therefore a(0)=0.

Newton's identities and Cayley–Hamilton give

\[
p(H)=0\iff H\text{ is nilpotent}.
\]

It follows that

\[
A^\top G\text{ nilpotent}\Longrightarrow F(A,G)=0.
\tag{7}
\]

This is an entire zero update, not just zero loss derivative.

For

\[
N=E_{12},\quad T=I-N,\quad
L(W)=\tfrac12\|W-T\|_F^2,
\]

the point W=I has G=N != 0 and loss 1/2; T is invertible. Every admissible local rule returns zero. Dense GD yields \(W^+=I-\eta N\) and loss \((1-\eta)^2/2\).

For any factorization,

\[
A_i^\top G_i=Q_i^{-\top}(W^\top G)Q_i^\top.
\]

Thus all gate arguments are similar, and all gate updates vanish at the same obstruction. This is not a defect caused only by the one-factor representation.

### 3.3 Matching construction, degree, and slowdown

The notebook's witness is

\[
K_*(H)=\sum_{k=1}^{n}\operatorname{tr}(H^k)(H^{k-1})^\top,
\qquad F_*(A,G)=AK_*(A^\top G).
\tag{8}
\]

Its dissipation is exactly

\[
\langle G,F_*(A,G)\rangle_F
=\sum_{k=1}^{n}\operatorname{tr}((A^\top G)^k)^2,
\tag{9}
\]

strictly positive precisely off the nilpotent cone.

If K is polynomial and strictly descends at every nonnilpotent argument, its total degree in H is at least 2n-1. To see this, restrict the invariant dissipation \(q(H)=\langle H,K(H)\rangle_F\) to a weighted n-cycle \(B(z)=\sum_{i<n}z_iE_{i,i+1}+z_nE_{n,1}\). Diagonal similarity forces every invariant monomial to be a power of \(\prod_i z_i\). Degree below 2n in q leaves at most a linear polynomial of this product. Its constant is zero, and nonnegativity for both signs forces the linear coefficient to vanish. That contradicts strict descent for an invertible cycle. Equation (8) attains degree 2n-1.

For the cyclic permutation P, a locally Lipschitz compatible descent rule satisfies

\[
q(\varepsilon P)=O(\varepsilon^{2n}),\qquad
q_*(\varepsilon P)=n^2\varepsilon^{2n}.
\tag{10}
\]

Proof: the companion-type matrix \(H_s=N_{\rm Jordan}+sE_{n1}\) has only its nth power trace nonzero, equal to ns. The relevant continuous coefficient vanishes at s=0; Lipschitz continuity makes it O(s). Hence q(H_s)=O(s^2). Similarity between \(\varepsilon P\) and \(H_{\varepsilon^n}\) gives equation (10).

This is a fixed-rule normalization statement. Multiplying a rule by an arbitrarily large constant is not a remedy for its zero set, and any normalization by vanishing invariants must be checked for continuity. Directly computing all powers costs O(n^4) with conventional dense multiplication, despite O(n^2) streaming workspace.

### 3.4 Exact finite-step identity

With old gradients and all factors updated simultaneously,

\[
A_i^+=A_i\exp[-\eta m_i K(A_i^\top G_i)]
\]

gives

\[
\prod_i A_i^+=W\exp[-\eta M K(W^\top G)].
\tag{11}
\]

Equation (4) yields \(K(A_i^\top G_i)=Q_iK(W^\top G)Q_i^{-1}\). Conjugating the exponentials through suffixes makes the product telescope; the remaining exponentials commute because they are scalar multiples of the same matrix.

This identity holds in exact arithmetic at every real eta. Descent holds only for sufficiently small positive eta when the directional pairing is positive. Sequentially recomputing gradients, replacing the exponential by Cayley, or allowing each representation to select a different step size changes the claim.

### 3.5 Strengthening: two identity insertions already suffice

This deduction was made during the present audit.

Fix any t in (0,1). If exact equation (1) is required only for the two splits (A,I) and (I,A), for every A,G, it gives respectively

\[
F(A,G)=tF(A,G)+(1-t)AK(A^\top G),
\]
\[
F(A,G)=tK(GA^\top)A+(1-t)F(A,G).
\]

These imply equation (3), then (4), then all the original refinement identities for every positive mass allocation.

**Implication:** permitting only a fixed mass split does not evade the obstruction. Requiring only one orientation does: the two one-sided metric gradients below satisfy the corresponding individual identities and have strict descent.

### 3.6 Sharp quantitative identity-insertion theorem

Prescribe an identity response K, so every considered rule satisfies F(I,H)=K(H). For fixed A,G, define

\[
U=AK(A^\top G),\qquad V=K(GA^\top)A.
\]

For equal child masses the two residuals are

\[
D_{(A,I)}=\tfrac12(F(A,G)-U),\qquad
D_{(I,A)}=\tfrac12(F(A,G)-V).
\]

The triangle inequality gives

\[
\max(\|D_{(A,I)}\|_F,\|D_{(I,A)}\|_F)
\ge\tfrac14\|U-V\|_F.
\tag{12}
\]

The rule

\[
F_{\rm mid}(A,G)=\tfrac12\{AK(A^\top G)+K(GA^\top)A\}
\tag{13}
\]

attains equality pointwise for every A,G and preserves the prescribed K at identity. If K is continuous, so is this rule. If \(\langle H,K(H)\rangle_F\ge0\), then

\[
\langle G,F_{\rm mid}(A,G)\rangle_F
=
\tfrac12\{\langle A^\top G,K(A^\top G)\rangle_F
+\langle GA^\top,K(GA^\top)\rangle_F\}\ge0.
\]

Strict descent of K away from zero gives strict descent for invertible A.

Under **Euclidean identity calibration**, K(H)=H,

\[
F_{\rm sym}(A,G)=\tfrac12(AA^\top G+GA^\top A),
\tag{14}
\]
\[
\inf_{F:\,F(I,H)=H}
\max(\|D_{(A,I)}\|_F,\|D_{(I,A)}\|_F)
=
\tfrac14\|AA^\top G-GA^\top A\|_F.
\tag{15}
\]

The infimum remains attainable among continuous universally strictly descending rules because

\[
\langle G,F_{\rm sym}(A,G)\rangle_F
=\tfrac12(\|A^\top G\|_F^2+\|GA^\top\|_F^2)>0
\]

for nonzero G.

This is an elementary exact minimax theorem for **two identity insertions**. The gradients in equation (14) are classical [R4,R5]. Do not label their average a newly invented optimizer. The candidate differentiation is the specific quantitative refinement characterization; its historical novelty is unconfirmed.

For a fixed positional ratio t:(1-t), the optimum is \((1-t)U+tV\) and the value is \(t(1-t)\|U-V\|_F\). For the supremum over all positive mass allocations and both orientations, the midpoint attains \(\|U-V\|_F/2\). Equal-mass experiments must not be described as testing that latter supremum.

### 3.7 A normalization-safe sharp witness

Let

\[
A=\operatorname{diag}(1,\kappa^{-1},1,\ldots,1),\quad
G=N=E_{12},\quad \kappa\ge1.
\]

Then \(\|A\|_2=\|G\|_F=1\), condition(A)=kappa, and U=N, V=kappa^{-2}N for K(H)=H. The optimum is

\[
d_{\min}=(1-\kappa^{-2})/4.
\tag{16}
\]

| Kappa | Sharp bound / symmetric rule | GD and either one-sided rule |
|---:|---:|---:|
| 1 | 0 | 0 |
| 2 | 0.1875 | 0.375 |
| 10 | 0.2475 | 0.495 |
| 100 | 0.249975 | 0.49995 |

The factor-of-two improvement is exact on this stated metric, and was numerically reproduced. It is not an empirical benchmark discovery.

More generally, if \(\langle N,K(N)\rangle_F\ge\mu\) and
\(\|K(N/\kappa)\|_F\le L/\kappa\), the same normalized fixture gives

\[
\max(\|D_{(A,I)}\|_F,\|D_{(I,A)}\|_F)
\ge(\mu-L/\kappa^2)_+/4.
\tag{17}
\]

This makes the role of nonvanishing descent and finite gain explicit. Without a fixed K or such a constraint, K=0 trivially wins an invariance-only objective.

### 3.8 Required negative control: the improvement is not universal

For W=BC, equation (14) has residual

\[
\mathcal R_{\rm sym}
=\tfrac12\{tWW^\top G+(1-t)GW^\top W-BB^\top GC^\top C\}.
\tag{18}
\]

Use

\[
B=\operatorname{diag}(1,2),\quad
C=\operatorname{diag}(2,1),\quad
W=2I,\quad G=E_{12},\quad t=1/2.
\]

For ordinary GD, both the parent and split positive updates equal G, so the defect is zero. For the symmetric rule, the parent update is 4G and the split update is 2.5G, giving defect 1.5G.

This exact example is mandatory in every report. A scalar-normalized symmetric rule removes this particular discrepancy, which is why normalized variants belong among the strong simple comparisons. Neither that correction nor the identity-insertion theorem proves superiority on all other splits.

### 3.9 Contextual escapes and the available headroom

The notebook's contextual dense-flow lift is

\[
F_i^{\rm context}=P_i^{-1}GQ_i^{-1},
\qquad \dot A_i=-m_iF_i^{\rm context},
\qquad \dot W=-MG.
\tag{19}
\]

This is closely related to established exact natural-gradient and quotient constructions [R6–R10]. It uses extra information and is a correctness/cost baseline.

For an exact dense finite step, change only one anchor factor j:

\[
A_j^+=A_j-\eta P_j^{-1}GQ_j^{-1}.
\tag{20}
\]

The product equals W-eta G exactly. This illustrates why exponentials are not uniquely necessary once context is allowed.

A single suffix SPD matrix also suffices for a different strict-descent geometry. Put \(R_i=Q_iQ_i^\top\), \(H_i=A_i^\top G_i\), and \(X_i=R_iH_iR_i^{-1}\). Then

\[
X_i=Q_i(W^\top G)Q_i^{-1},\quad
\dot A_i=-m_iA_iX_i,\quad
\dot W=-MWW^\top G.
\tag{21}
\]

The pairing is \(M\|W^\top G\|_F^2\), and simultaneous old-gradient exponentials give \(W^+=W\exp[-\eta M W^\top G]\). This establishes sufficiency of one SPD context, not minimality. Forming a Gram matrix squares its condition number; compare factor-based solves against the explicit Gram realization.

At a fixed output update norm, dense Euclidean descent already maximizes instantaneous loss reduction by Cauchy–Schwarz. A meaningful gain must therefore concern rewrite robustness, conditioning, representation cost, numerical stability, or a stated task tradeoff—not an unexplained larger gradient step.

A dense product of ell dense n-by-n factors stores ell times as many parameters and usually adds work relative to a single dense W. No storage or arithmetic saving is presumed. A structured-library application is justified only if a genuine structural benefit survives context and update costs.

## 4. Research landscape and precise novelty assessment

### 4.1 Prior-art comparison

The table separates resemblance from exact antecedence. Complexity is for the relevant basic operation, not a claim about published end-to-end speed.

| Work / family | Problem and mechanism | Assumptions and evidence | Cost / reusable implementation | Exact overlap and boundary |
|---|---|---|---|---|
| Procesi; trace algebras [R1,R2] | Polynomial invariants and matrix concomitants | Classical algebraic results | Reuse theory; no training benchmark | Polynomial algebraic ingredients are established; inspected source does not supply the whole continuous descent package |
| Malgrange/Schwarz account [R3] | Construct equivariants from invariants | Polynomial framework; cited smooth extension uses compact groups | Explicit constructions | Do not transfer compact assumptions silently to real GL |
| GL invariant metrics [R4,R5] | Left/right Riemannian matrix gradients | Established geometry and applications | Dense O(n^3), simple formulas | Both terms of equation (14) are old; the defect characterization is the candidate extension |
| Exact deep-linear NG [R6,R7] | Contextual parameterization-invariant learning | Deep linear models; also documented loss of useful implicit bias | Prefix/suffix algebra, O(ell n^3) dense implementation | Strong precedent for equation (19); not limited to isolated F(A,G) |
| Quotient low-rank optimization [R8] | Optimize a fixed-rank matrix through equivalent factors | Full GL(r) gauge, contextual metric | Author MATLAB implementation; first/second-order methods | Fixed-depth geometry, not arbitrary-depth local refinement |
| ScaledGD [R9] | Inverse-Gram preconditioning for low-rank estimation | Convergence guarantees under initialization and problem assumptions | Official repository; Gram operations in rank r | Square two-factor specialization reproduces dense output gradient flow up to mass/time scaling |
| LoRA-RITE [R10] | Invariant adaptive updates of a factor pair | Arbitrary invertible latent transforms; paired-factor data and transported state | Official PyTorch reimplementation; QR/eigensolves; license and recipe caveats | Much closer than orthogonal-only work, but violates strict local/memoryless assumptions |
| Lau–Su; Singh [R11,R12] | Orthogonal and other architecture-specific optimizer symmetries | Current 2026 primary texts; fixed architecture and specified state behavior | Some CPU diagnostics reusable | Neither is an arbitrary-GL local refinement theorem |
| Path-SGD [R13] | Rescaling-invariant neural optimization | Smaller rescaling symmetry | Established implementable algorithm | Preserving a smaller group is an existing strategy |
| Net2Net [R14] | Function-preserving network expansion | Algebraic equality at rewrite | Existing construction | Equality of the current function does not ensure equality of later learning |
| CoLA; butterfly transforms [R15,R16] | Compositional linear algebra and efficient structured transforms | Established frameworks and learned structures | Existing libraries/implementations | Operator objects and structured factorization are not new; arbitrary dense updates may destroy structure |
| RoM, September 2026 [R23] | Fixed-rank manifold and rotated-normalization optimization | Primary text retrieved; introduction and method organization inspected | Implementation not reproduced | Current contextual comparator for a future fixed-rank claim; outside the isolated local rule |
| Related patent publication [R24] | Transformation-invariant adaptive matrix preconditioning | Public technical disclosure | Not an implementation license | Further evidence that broad invariant-preconditioning claims are occupied |

### 4.2 What exists, what to reuse, and what may differ

**Already exists:** operator abstractions, function-preserving transformations, invariant metrics, polynomial concomitants, natural-gradient lifts, fixed-depth GL quotient optimization, and full-GL adaptive factor-pair methods.

**Reuse:** NumPy/SciPy dense algebra; classical gradient formulas; exact contextual controls; known quotient/ScaledGD equations where their domain matches; published diagnostic designs. Reuse licensed code only after pinning and checking its declared license. Avoid building a Fisher matrix with (ell*n^2)^2 entries when the required lift is available through solves.

**Potential differentiation:** the exact continuous local information model and its forced nilpotent zero set; the sharp polynomial degree and regularity consequences; the weaker identity-insertion assumption; the sharp, normalized two-insertion defect theorem; and empirical findings about which compromises survive general rewrites.

**Possible value:** these results can reject an impossible optimizer interface early and identify precisely which extra state, restricted rewrite family, or permitted inconsistency is needed. That is a design benefit even if no optimizer benchmark improves.

**Discriminating test:** reproduce the theorem witnesses, the exact minimax equality, and the adverse split; then test general rewrites chosen independently of the witness. A method tested only on identity insertion cannot support a broader claim.

**Novelty-invalidating evidence:** an earlier source with the same local-information/refinement conditions and conclusion; an existing continuous concomitant theorem already specialized to the positive pairing and nilpotent zero set; or an earlier midpoint refinement characterization. If found, reclassify the work as replication, unification, or empirical diagnosis. Do not rename an old optimizer.

### 4.3 Direction choice and alternatives

The selected primary direction is the quantitative obstruction study. Its mathematical mechanism is present at n=2, its general theorem remains dimension-dependent, and the core is feasible without external data or a GPU. The separate symmetric-gate branch has established SPD-product and path-signature precedents [R25,R26].

A new contextual optimizer is not selected: the main escape formulas already have strong precedents, and anchor/collapse controls eliminate several superficially attractive claims. A large benchmark campaign is not selected before a method demonstrates a relevant tradeoff. The secondary gate-word Hessian result is retained as a separate possible theorem note; it has a different mechanism and would dilute this project's evaluation.

If the exact obstruction is already known, complete the diagnostic benchmark only if it identifies a reproducible unresolved behavior or implementation issue. If the diagnostic offers no advantage over existing comparisons, conclude with a valid negative or replication result rather than expanding the architecture.


## 5. Implementation specification

### 5.1 Minimal repository

The coding agent should create the following project. The package name and commands below are **proposed project interfaces to implement**, not an existing downloadable package.

~~~text
refinement_learning/
  README.md
  pyproject.toml
  requirements.lock.txt
  THIRD_PARTY_NOTICES.md
  protocol/
    claims.md
    theory.md
    confirmatory.json
    amendments.jsonl
    decisions.jsonl
    source_ledger.jsonl
  configs/
    smoke.json
    development.json
    confirmatory.json
    stress.json
    trajectories.json
  src/rilo/
    __init__.py
    __main__.py
    environment.py
    circuits.py
    rules.py
    context.py
    integrators.py
    generators.py
    metrics.py
    tuning.py
    runner.py
    analysis.py
    reporting.py
  tests/
    test_algebra.py
    test_gradients.py
    test_controls.py
    test_resume.py
  data/
    manifests/
  runs/
    development/
    pilot/
    confirmatory/
    stress/
  reports/
    findings.md
    tables/
    figures/
~~~

Keep mathematical proofs and machine-readable claims separate from measured results. The README must say which stages were actually executed and whether the target machine was used.

### 5.2 Shared interfaces and information boundaries

Use explicit positive directions and apply the minus sign once.

~~~python
from dataclasses import dataclass
import numpy as np

@dataclass
class Circuit:
    factors: list[np.ndarray]   # float64, each n by n
    masses: np.ndarray         # positive, sum equals 1

@dataclass
class CircuitSnapshot:
    W: np.ndarray
    prefix: list[np.ndarray]   # prefix[i] = A[0] ... A[i-1]
    suffix: list[np.ndarray]   # suffix[i] = A[i] ... A[ell-1]
    global_grad: np.ndarray
    local_grads: list[np.ndarray]

# Core local method contract:
# direction(A, G, config) -> unweighted positive F(A,G)
#
# Contextual method contract:
# directions(circuit, snapshot, config) -> positive F_i
#
# Core represented velocity:
# V = -sum_i masses[i] * prefix[i] @ F_i @ suffix[i+1]
#
# Integrator:
# propose(circuit, snapshot, directions, eta, state)
#     -> proposed_circuit, proposed_state, diagnostics
#
# Rewrite:
# split_gate(circuit, index, B, C, t)
#     -> new_circuit, event_record
# Preconditions: B @ C equals old factor within audited roundoff,
# B and C invertible, new masses = old_mass * (t, 1-t).
~~~

Local methods must not receive the snapshot or a global normalization factor. Keep contextual and stateful implementations in separate adapters. An output-level normalization used for analysis is a diagnostic, not a free local operation.

Build prefix and suffix products once per snapshot:

- prefix[0]=I; prefix[i+1]=prefix[i] @ A[i].
- suffix[ell]=I; suffix[i]=A[i] @ suffix[i+1].
- W=prefix[ell]; G_i=prefix[i].T @ G @ suffix[i+1].T.

All simultaneous updates use the same immutable snapshot. A rejected line-search trial must not advance moments or change the snapshot. This prevents old-gradient versus refreshed-gradient errors.

### 5.3 Reference formulas

These snippets are executable with the named dependencies. They are small reference components, not a completed experiment runner.

~~~python
import numpy as np
from scipy.linalg import expm, solve

def gd(A, G):
    return G.copy()

def left_metric(A, G):
    return A @ (A.T @ G)

def right_metric(A, G):
    return (G @ A.T) @ A

def symmetric_metric(A, G):
    return 0.5 * (left_metric(A, G) + right_metric(A, G))

def mixed_metric(A, G, weights):
    alpha, beta, gamma = np.asarray(weights, dtype=float)
    assert min(alpha, beta, gamma) >= 0
    assert np.isclose(alpha + beta + gamma, 1.0)
    return alpha * G + beta * left_metric(A, G) + gamma * right_metric(A, G)

def scalar_normalized(rule, A, G):
    scale = np.sum(A * A) / A.shape[0]
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError("Invalid scale; do not silently damp.")
    return rule(A, G) / scale

def trace_K(H):
    n = H.shape[0]
    previous = np.eye(n, dtype=H.dtype)
    answer = np.zeros_like(H)
    traces = []
    for k in range(1, n + 1):
        current = previous @ H
        p = np.trace(current)
        answer += p * previous.T
        traces.append(p)
        previous = current
    if not np.isfinite(answer).all():
        raise FloatingPointError("Trace polynomial overflow.")
    return answer, np.asarray(traces)

def trace_rule(A, G):
    return A @ trace_K(A.T @ G)[0]

def refinement_residual(rule, B, C, G, t=0.5):
    parent = rule(B @ C, G)
    split = t * rule(B, G @ C.T) @ C
    split += (1 - t) * B @ rule(C, B.T @ G)
    return parent - split

def right_solve(X, Q):
    # Returns X @ inv(Q), without explicitly constructing inv(Q).
    return solve(Q.T, X.T, assume_a="gen").T

def contextual_dense_direction(P, G, Q):
    return right_solve(solve(P, G, assume_a="gen"), Q)

def compatible_exponential_step(A, local_G, mass, eta):
    # All inputs must come from the same OLD circuit snapshot.
    return A @ expm(-eta * mass * trace_K(A.T @ local_G)[0])
~~~

The scalar-normalized variant divides by \(\|A\|_F^2/n\). Adding an epsilon, clipping singular values, or replacing a solve by a pseudoinverse changes the rule; expose such choices in configuration and label them as separate methods.

For the one-metric control, implement both routes:

1. Algebraic oracle: \(H=W^\top G\), \(X_i=Q_iH Q_i^{-1}\), \(F_i=A_iX_i\).
2. Local-with-context route: \(R_i=Q_iQ_i^\top\), \(X_i=R_iH_iR_i^{-1}\).

Require agreement at well-conditioned inputs. Count both the cost of computing R_i and the cost of maintaining it after every rewrite. A compressed or stale metric is a new approximation, not the exact control.

### 5.4 Minimum independent smoke program

Copy the following into an executable smoke module after importing the preceding functions. It does not require the original notebook.

~~~python
def smoke():
    I = np.eye(2)
    N = np.array([[0., 1.], [0., 0.]])
    assert np.array_equal(trace_rule(I, N), np.zeros((2, 2)))

    eta = 0.1
    T = I - N
    old_loss = 0.5 * np.sum((I - T) ** 2)
    new_loss = 0.5 * np.sum((I - eta * N - T) ** 2)
    assert old_loss == 0.5
    assert abs(new_loss - 0.405) < 1e-14

    for kappa in (1., 2., 10., 100.):
        A = np.diag([1., 1. / kappa])
        bound = (1. - kappa ** -2) / 4.
        gaps = [
            np.linalg.norm(refinement_residual(symmetric_metric, A, I, N)),
            np.linalg.norm(refinement_residual(symmetric_metric, I, A, N)),
        ]
        assert abs(max(gaps) - bound) < 1e-13

    B = np.diag([1., 2.])
    C = np.diag([2., 1.])
    assert np.linalg.norm(refinement_residual(gd, B, C, N)) == 0.
    assert abs(np.linalg.norm(
        refinement_residual(symmetric_metric, B, C, N)) - 1.5) < 1e-14

    rng = np.random.default_rng(20261031)
    for n in (2, 3, 5):
        A = np.eye(n) + 0.1 * rng.normal(size=(n, n))
        B = np.eye(n) + 0.1 * rng.normal(size=(n, n))
        G = 0.1 * rng.normal(size=(n, n))
        W = A @ B
        H = W.T @ G
        K, p = trace_K(H)
        assert np.allclose(np.sum(G * (W @ K)), np.sum(p*p),
                           rtol=1e-11, atol=1e-13)
        Ap = compatible_exponential_step(A, G @ B.T, .3, .07)
        Bp = compatible_exponential_step(B, A.T @ G, .7, .07)
        assert np.allclose(Ap @ Bp, W @ expm(-.07 * K),
                           rtol=1e-11, atol=1e-13)
        Fa = contextual_dense_direction(np.eye(n), G, B)
        Fb = contextual_dense_direction(A, G, np.eye(n))
        assert np.allclose(.3 * Fa @ B + .7 * A @ Fb, G,
                           rtol=1e-11, atol=1e-13)
    return {"smoke": "passed"}
~~~

These fixed tolerances are for these small, well-conditioned fixtures. Do not reuse them as universal tolerances for a condition-number sweep.

### 5.5 Numerical conventions and correctness

Use float64 CPU arithmetic as the reference. Record matrix layout and BLAS implementation. Do not detect nilpotency by asking whether floating-point eigenvalues are exactly zero. Construct exact triangular or cycle fixtures, or certify a small rational example symbolically.

For algebra tests, record both:

\[
e_{\rm scaled}=\frac{\|X-Y\|_F}{\max(1,\|X\|_F,\|Y\|_F)}
\]

and the absolute residual. For cancellation-prone formulas also record a backward-style denominator equal to the sum of the norms of the evaluated terms. Do not enlarge a tolerance until a failed comparison passes. Estimate the expected roundoff from dimension, operation count, and conditioning; resolve material violations with an alternate formula or higher precision on a small instance.

Use finite differences only to check gradients, not to estimate the principal refinement residual. For a direction E, compare the analytic directional derivative with centered loss differences over a geometric step-size ladder. Seek the expected decrease-then-roundoff curve; retain the curve if no stable range exists.

For extremely small cycle dissipations, plot analytic log-values and measured values separately. A zero due to underflow is not evidence of exact stalling at a nonnilpotent matrix. Scaling the entire gradient may be useful as a diagnostic, but it changes the experiment unless analytically accounted for.

Use solves rather than explicit inverses. Log relative solve residuals, matrix condition estimates, singularity failures, and any regularization. Factor-based implementations and Gram-based implementations may have different numerical stability even when algebraically equivalent.

## 6. Baselines, workloads, metrics, and confirmation

### 6.1 Comparison tiers

| Tier | Methods and configuration | Role / admission rule |
|---|---|---|
| Local, memoryless, identity-calibrated | GD; left metric; right metric; symmetric metric; nonnegative three-term blends; scalar-normalized variants | Mandatory main comparisons |
| Exact compatible local witnesses | Trace rule; central trace-only rule | Diagnose the obstruction; different identity response, so do not rank them as calibrated competitors |
| Deliberately invalid controls | Zero rule; transpose-compatible ascending example; Cayley replacement | Harness must detect stalled learning, ascent, and finite-step incompatibility |
| Contextual exact controls | Dense-flow lift, one-metric lift, anchor finite step, collapse/update/refactor | Mandatory correctness and cost bounds; explicitly more information |
| Stateful practical comparators | Dense and factor Adam; momentum; licensed current orthogonal diagnostics if making optimizer claims | Optional for field-only theorem; mandatory before broad training-efficiency claims |
| Fixed-rank contextual comparators | ScaledGD, quotient geometry, LoRA-RITE; inspect RoM before such a claim | Required only if scope expands to low-rank contextual optimization |
| Task-level controls | Dense least squares for fully observed regression; dense GD and L-BFGS | Prevent an unnecessary factor optimizer from looking competitive against a weak baseline |

The trace rule must be run only where its direct cost and conditioning are informative; its tiny loss decrease is predicted, not a tuning failure to hide. The zero rule is assigned a failed-descent label even if its absolute rewrite residual is zero.

For factor Adam, implement the standard moment recurrences with explicitly frozen beta1, beta2, epsilon, bias correction, and no weight decay in the unregularized comparison. The first development search should cover beta1 in {0,0.9}, beta2 in {0.99,0.999}, and learning rates located by a stability bracket, not only a default rate. Extend the search if its best configuration is on a meaningful boundary. Report the full tuning cost. Adam is stateful and need not give universal instantaneous descent; do not treat that as a violation of the theorem.

The remaining required diagnostic controls are fully specified by:

\[
F_{\rm ctr}(A,G)=\operatorname{tr}(A^\top G)A,\qquad
F_{\rm bad}(A,G)=AG^\top A,\qquad F_0(A,G)=0.
\]

The central rule has pairing \(\operatorname{tr}(A^\top G)^2\) and stalls on trace-zero arguments. The bad rule is compatible but has pairing \(\operatorname{tr}((A^\top G)^2)\); at A=I and \(G=\begin{psmallmatrix}0&1\\-1&0\end{psmallmatrix}\), the pairing is -2 and the negative update ascends. Define Cayley by \(C(X)=(I-X/2)^{-1}(I+X/2)\), implemented by a solve. The exact fixture \(C(-I/2)=3I/5\) differs from \(C(-I/4)^2=49I/81\), exposing its failed semigroup identity.

For an unconstrained dense collapse control, use the cheapest admissible representation: train W directly, or use (W,I,...,I) if a depth-preserving interface is required. Do not force an SVD after every update. Balanced or structured refactorization is charged only when an explicit factor constraint requires it.

Under dimension-changing rewrites there is no default universally correct Adam state transport. Report a named reset policy only as that policy; do not call it a test of an invariant optimizer. A fixed-depth gauge test must use the comparator's documented state transformations.

### 6.2 Strong simple baseline tuning

Use

\[
F_w(A,G)=w_0G+w_LAA^\top G+w_RGA^\top A,\quad
w\ge0,\quad w_0+w_L+w_R=1.
\tag{22}
\]

Also consider equation (22) divided by \(\|A\|_F^2/n\), as a separate family. The raw symmetric rule and its normalized variant belong to the respective families. These are conventional combinations of established positive geometries; selecting their coefficients is engineering/tuning, not methodological novelty.

The fixed comparator set consists of the raw and scalar-normalized forms of GD, left metric, right metric, and symmetric metric: eight nominal methods, with any exact duplicates recorded. Select the best fixed comparator using the same development cases and objective as the searched candidate.

Search both families only on development cases, using the primary discrepancy metric below. Evaluate simplex lattices with denominators q=8,16,32,..., including all fixed-comparator weights and all previous best points. At each refinement run constrained SLSQP optimization from those fixed points, the previous distinct numerical optima, and each lattice local minimum. A lattice neighbor transfers one unit of 1/q between two coordinates; for a connected plateau of equal local minima within numerical resolution, use its lexicographically first point. Record every start, solver status, feasible final coefficients, objective, and boundary optimum. An unsuccessful solver result is not a certificate.

Freeze a development-resolution tolerance \(\tau_{\rm tune}\) before confirmation, with its relationship to development sampling uncertainty and floating-point resolution recorded. Stop when two consecutive lattice refinements together with their local searches improve the best objective by less than this tolerance. If the resource allowance ends first, report the best-found rule and unresolved tuning error. This search does not prove a continuum global optimum or identify the best possible local rule.

For efficient tuning, cache for each case the 3-by-3 Gram matrices of the three component residuals, parent directions, and split directions. For example,

\[
\|\mathcal R_{F_w}\|_F^2=w^\top C_{\mathcal R}w.
\]

This permits dense coefficient searches without repeatedly multiplying n-by-n matrices. Do not infer convexity of a ratio objective from convexity of its numerator. Treat objective values within the frozen \(\tau_{\rm tune}\) as ties. Break ties by fewer dense matrix products in the reference implementation, then fewer nonzero coefficients, then raw before scalar-normalized, then lexicographic coefficient order \((w_0,w_L,w_R)\). Apply the same rule to candidate and comparator selection. If the selected family member equals the best fixed rule, record no differentiated candidate and omit a redundant confirmatory superiority claim.

An optional finite-development-set certificate is available. For component j on one case, write \(P_j,S_j,R_j=P_j-S_j\) for parent, split, and residual directions. Set

\[
b_{\min}=\min_j\frac{\langle G,P_j+S_j\rangle_F}{\|G\|_F}>0,\qquad
L_{\rm case}=
\frac{2\{\max_j\|R_j\|_F+\max_j\|P_j\|_F+\max_j\|S_j\|_F\}}{b_{\min}}.
\]

Strict descent makes \(b_{\min}\) a lower bound on the denominator of r throughout the simplex. The triangle inequality and the bound \(r\le2\) then give an \(\ell_1\)-Lipschitz constant \(L_{\rm case}\). Average these constants with the declared development weights. A denominator-q simplex lattice has an elementary \(\ell_1\) covering bound of at most 4/q, giving a missed-improvement bound \(4\bar L/q\). This certificate may be conservative; it is optional and is not a gate for the theorem project. Diagnose a nonpositive or numerically unresolved \(b_{\min}\); do not fabricate a certificate.

Compare the selected family member with the development-selected best fixed simple comparator. Never compare it only with the weakest endpoint. All methods receive the same development cases and tuning objective. A broad invariance win remains a tradeoff if descent angles or computation worsen.

### 6.3 Mandatory deterministic suite

Use independently implemented formulas and include:

1. Scalar n=1, to distinguish scalar compatibility from the matrix obstruction.
2. Nonzero shear N=E12 at W=I.
3. Higher nilpotent Jordan blocks and orthogonally rotated nilpotent matrices.
4. Noncyclic matrices: zero, repeated diagonal eigenvalues, and block-diagonal repeated Jordan blocks.
5. Companion matrices from prescribed real power traces, including traces requiring nonreal eigenvalues.
6. Weighted directed cycles of both product signs.
7. The cycle epsilon scaling, with analytic log-values and a precision audit.
8. Both identity-insertion orientations, fixed and unequal masses.
9. The adverse general split in Section 3.8.
10. Orthogonal/conformal factors, where several residuals should vanish.
11. Simultaneous old-gradient exponentials and a deliberately refreshed-gradient implementation.
12. Cayley semigroup failure.
13. Dense-flow, one-metric, anchor, and collapse controls.
14. Opposite determinant components and a proposed step approaching singularity.
15. Exact mass conservation and equality of the represented W before every rewrite.

Do not attach confidence intervals or p-values to these deterministic truths. A randomized consistency test is not a proof and cannot falsify a theorem unless the tested inputs meet its hypotheses and numerical error is resolved.

### 6.4 Generated empirical population

No external dataset is needed for the primary claim. Generated matrices have transparent provenance and avoid leakage from a curated benchmark. The population below is a research distribution, not a claim about production rewrite frequencies.

The initial main population uses n in {2,8,32}, covering the minimal obstruction, a multi-directional small operator, and a larger dense check. These are coverage points, not an upper size limit. Additional dimensions are justified if dimension changes the ranking or estimated cost enough to affect the conclusion.

For each dimension, use all ordered prescribed-condition pairs from {1,2,10} squared. The first regime samples B,C with those conditions. The second samples W,S with those conditions and derives C=S^{-1}W; its realized C can have condition up to 100 and remains in that main population. Do not apply an unannounced rejection to it. A separate stress population uses prescribed conditions 100 and above. This distinguishes sampled input conditions from derived factor conditions.

Generate an invertible matrix as follows:

1. Draw two independent Gaussian matrices, QR-factor each, and correct Q's column signs using the signs of R's diagonal. This produces orthogonally randomized orientations.
2. For kappa>1, choose log singular values with endpoints \(-\tfrac12\log\kappa\) and \(+\tfrac12\log\kappa\); draw any interior values uniformly between them and permute the positions. For kappa=1 use zero logs.
3. Draw log scale uniformly between log(1/2) and log(2).
4. Return U diag(scale*exp(log_singular_values)) V.T.
5. Record actual singular values and condition estimates.

Draw B and C independently with this rule and set W=BC in the first regime. Record conditions and norms of B,C,W, including the possibility that condition(W) differs markedly from either factor. In the second regime, draw W and S independently, set B=S and solve C=S^{-1}W. The generator is repeated across independent suite indices; comparisons across methods always use the identical sampled parent and split.

Normalize G to Frobenius norm one after selecting equally weighted gradient families:

- dense Gaussian;
- symmetric Gaussian \((Z+Z^\top)/2\);
- skew-symmetric Gaussian \((Z-Z^\top)/2\);
- scalar trace direction I/sqrt(n);
- rank-one nilpotent uv.T, where u,v are the first two columns of a sign-corrected Gaussian QR draw.

If a generated norm is zero in finite arithmetic, regenerate and record the event. Add near-nilpotent gradients N+epsilon E and reciprocal factorizations of I as robustness populations, not post hoc favorable slices.

Use t in {0.1,0.5,0.9} for each instance. The main stratum definition is exactly the Cartesian product of three dimensions, two generation regimes, nine ordered condition pairs, and five gradient families. Generate base matrices independently across these 270 strata; reuse each base case only across masses and methods. Give each stratum equal weight, then average its three masses equally: 810 evaluated cases per suite index. Endpoint-near values and repeated rewrite chains belong in the stress suite. Both identity insertion and the adverse diagonal split are separate deterministic strata, not the whole empirical distribution.

**Independent unit:** a newly generated suite index produces one independently sampled base case in each declared stratum. The suite-level average is one observation. Different masses, orientations, rewrite repetitions, and timings within that suite are nested observations. This makes aggregation and uncertainty unambiguous.

Record all generated seeds, generator version, stratum weights, and rejection counts. Do not reject a case because a method performs poorly, has an inconvenient condition number not excluded by the frozen generator, or takes longer.

### 6.5 Metrics

For positive parent and split output directions Vp and Vs, store:

\[
d_{\rm abs}=\|V_p-V_s\|_F,\qquad
d_G=\frac{\|V_p-V_s\|_F}{\|G\|_F},
\]
\[
r=\frac{2\|V_p-V_s\|_F}{\|V_p\|_F+\|V_s\|_F}\in[0,2],
\tag{23}
\]
\[
c(V)=\frac{\langle G,V\rangle_F}{\|G\|_F\|V\|_F},\qquad
g(V)=\frac{\langle G,V\rangle_F}{\|G\|_F^2},\qquad
s(V)=\frac{\|V\|_F}{\|G\|_F}.
\tag{24}
\]

Also store the angle between Vp and Vs, dissipation difference, conditions, runtime, and peak memory.

**Primary deterministic metric:** the absolute worst two-insertion defect of equation (12), with the prescribed K and matrix normalization.

**Primary empirical metric:** the suite-level mean r over the frozen general-rewrite population, with equal declared stratum weights. It measures fractional field distortion. It is not the same quantity as the theorem's absolute defect.

If both Vp and Vs are zero for nonzero G, set r to null and mark a stall; never set it to zero and include it as an invariance success. If one direction is zero, r=2. If G=0, log a zero-gradient fixture and exclude it from positive-gradient aggregate metrics.

Never compute the paired primary comparison on method-specific successful subsets. By default, a stall, nonfinite result, or unresolved arithmetic failure in either primary method on any planned main-population case suspends an unqualified discrepancy-superiority conclusion. Retain the planned case count, failure rates, and a clearly conditional description on jointly valid cases. Repair an implementation defect before a fresh confirmation. A failure-aware primary score or missing-case bound is permissible only through a prospective protocol amendment; do not invent one after seeing failures.

**Guardrails:** stall frequency; ascent frequency beyond numerical uncertainty; cosine/descent gain distribution; upper-tail r; condition-number growth; arithmetic failures; per-update wall time; end-to-end amortized cost; memory. Report both parent and split values. An r improvement with lower descent or higher cost is a tradeoff, not an unqualified optimizer improvement.

A globally rescaled direction has unchanged r if the same scale is applied to both representations. A scale chosen independently in each representation can alter r; that is why scalar-normalized methods must be treated as explicit algorithms.

### 6.6 Pilot, freeze, sample size, and inference

Use disjoint seed namespaces for development, pilot, confirmatory, and robustness, with separate timing-order, trajectory, and bootstrap namespaces as specified in Section 10.1. Do not reuse a failure discovered during development as an unannounced random confirmatory observation; retain it as a named adversarial case.

The pilot estimates runtime, memory, numerical noise, and the variance of the selected paired difference. It must not select the final claimed effect from many inspected metrics. Obtain enough pilot suite indices to estimate its standard deviation with useful precision. For orientation, a normal-approximation relative standard-error target of about 20% for the SD suggests roughly 14 independent suites; use a bootstrap SD interval to decide if more pilot work is needed. This is calibration, not evidence for a positive result.

Before opening confirmatory results, freeze:

- generator source and stratum weights;
- method definitions, configurations, and tuning outcomes;
- primary comparison and metric;
- independent unit, sample count, interval procedure, missing/failure handling;
- all decision thresholds or interval precision targets;
- timing boundaries and hardware/backend;
- source revision and configuration hashes.

For a paired difference \(d_j=r_{\rm selected,j}-r_{\rm comparator,j}\), use the pilot SD estimate s and a precision target h to select a fixed N, initially

\[
N\approx (1.96s/h)^2.
\tag{25}
\]

Use an upper uncertainty estimate for s. Equation (25) is a normal-approximation initializer: round up to an integer N of at least two and iterate the Student-t quantile in place of 1.96 until the proposed half-width is at most h. Pilot-resampling simulations assess approximate precision and interval behavior under the pilot empirical distribution; they do not establish coverage for the actual population or provide a universal power guarantee.

If a downstream use case supplies a smallest consequential effect, choose h small enough to distinguish it. If no such use case exists, choose h from an explicitly reported precision-versus-compute curve and report the smallest distinguishable effect; do not call a statistically detectable difference practically important by default. If the affordable N is insufficient, label that hypothesis inconclusive.

The primary 95% interval is the paired Student-t interval

\[
\bar d\ \pm\ t_{0.975,N-1}\frac{s_d}{\sqrt N},
\]

computed from the N independent suite differences. Its coverage is approximate, not a distribution-free finite-sample guarantee. Always compute a percentile bootstrap interval by resampling whole paired suite differences as a sensitivity analysis. Never substitute whichever interval is narrower or more favorable. If the intervals disagree on whether the mean difference is below zero, report inference sensitivity and withhold a robust superiority conclusion.

Use the frozen bootstrap RNG namespace. Increase resamples in repeated independent batches until estimated Monte Carlo variability of each interval endpoint is below h/10. This controls computational precision and adds no independent experimental observations. If all observed differences are identical, report the degeneracy explicitly; do not fabricate a standard error or treat a degenerate empirical bootstrap as proof of zero population uncertainty.

The primary comparison is one frozen comparison after development selection. Label all other pairwise comparisons exploratory, or apply Holm correction if making a family of confirmatory claims. Do not select the favorable sign, normalization, dimension, or loss afterward. Do not stop a fixed-sample run early because its result looks significant.

Report absolute difference in r, each group's mean, uncertainty, per-stratum effects, and failures. Relative percentages may be undefined or misleading when the comparator discrepancy is near zero; preserve the absolute values.

For the fixed adverse suite, report exact or numerical maxima without population confidence claims. A bootstrap over random cases cannot certify an unrestricted worst-case guarantee.

### 6.7 Optional controlled training diagnostic

Enter this stage only after the algebra and numerical harness are valid and a general-rewrite result or a concrete unresolved trajectory question justifies it. It is not required to validate the mathematical contribution.

Use

\[
Y=TX+\Xi,\qquad
L(W)=\frac{1}{2b}\|WX-Y\|_F^2,\qquad
G=(WX-Y)X^\top/b.
\tag{26}
\]

Use the following concrete synthetic default, then freeze its separate trajectory protocol before protected runs:

- Set \(W_0=I\). For the spectral target family, draw \(T=U\Sigma V^\top\), with independent \(U,V\in SO(n)\) and log-spaced positive singular values of geometric mean one and target condition in {1,10}. Obtain an SO(n) draw from sign-corrected Gaussian QR by flipping its last column if its determinant is negative.
- Add a separate nonnormal family \(T=I-a uv^\top\), with orthonormal u,v drawn as above and a in {0.25,1}. Every such target has determinant one. The exact shear fixture remains a control, not the entire task population.
- Draw input covariance \(\Sigma_x=Q\operatorname{diag}(\lambda)Q^\top\), with orthogonal Q and log-spaced positive eigenvalues of geometric mean one and condition in {1,10}. Cross the target and covariance choices.
- Generate b=4n independent training inputs \(x\sim N(0,\Sigma_x)\). Add independent Gaussian label noise with per-coordinate standard deviation either zero or \(0.01\sqrt{\operatorname{tr}(T\Sigma_xT^\top)/n}\).
- Methods receive only X,Y and their declared state/context. The evaluator computes exact population prediction risk
  \[
  \mathcal E(W)=
  \tfrac12\operatorname{tr}((W-T)\Sigma_x(W-T)^\top)+\tfrac n2\sigma^2.
  \]
  This evaluation oracle is unavailable to the optimizer. Development tasks may select settings, but protected task parameters and risks remain unopened until the protocol is frozen.

These numerical choices define a synthetic research population; they do not estimate deployment prevalence. Use n from the validated hardware ladder. The fully observed training problem admits a direct least-squares solution: compute it with a QR/SVD-based solver and include its total cost, training loss, and population risk. For an underdetermined implicit-bias study, explicitly change and label the task; do not silently infer that greater invariance improves generalization [R7].

Initialize all compared representations with the same W0. Construct depth changes by inserting identities or using recorded, controlled splits; do not obtain W0 by independently randomizing every method's factors. Preserve total mass one. Before WP06, record the actual depths, run horizon, rewrite milestones, minibatch policy, Adam epsilon and moment settings, line-search constants, trial limit, and numerical-invertibility threshold in a separate frozen trajectory configuration. An incomplete configuration blocks only this optional stage.

For the primary trajectory population, choose W0 and the noiseless target T in the same determinant component. In noisy cases also record the determinant component of the unconstrained fitted solution when invertible; a mismatch is a separately reported geometry stress case. Log determinant-sign changes. Real exponential updates preserve sign, whereas Euler or anchor steps can jump across a singular straight segment despite invertible endpoints. Do not pool an opposite-component reachability problem with ordinary optimizer failures.

For each method run a paired twin:

- one trajectory with a frozen rewrite schedule;
- one without rewrites, with identical data and stochastic batch order.

Schedule rewrites at prespecified effective-time or gradient-evaluation milestones, not when one method looks favorable. Record field discrepancy at the exact pre-rewrite state, then subsequent trajectory separation. A schedule triggered by a norm threshold is a different policy and must be declared before confirmation.

Use a common Euler-with-backtracking adapter for the declared vector-field comparison. Standard Adam and L-BFGS retain their native update, state and line-search semantics in separately labeled algorithm comparisons; do not impose this adapter on them. For direction components -m_i F_i, the initial derivative is
\(-\sum_i m_i\langle G_i,F_i\rangle_F\). Trial all factors simultaneously from the same old snapshot. Backtrack on actual loss and numerical invertibility; record every rejected trial and include its cost. If a step cannot be accepted above the numerical resolution floor, report a numerical/conditioning failure rather than silently changing the rule.

Use the exact trace exponential and contextual anchor as separately labeled integrator controls. Validate an eta ladder: the finite-difference output velocity must approach the analytic field as eta decreases until roundoff dominates. Do not attribute an O(eta^2) Euler product error to failure of vector-field invariance.

Compare dense GD, dense Adam, L-BFGS, direct least squares, factor GD, the selected local mix, exact contextual flow, anchor, and collapse/refactor where applicable. Tune learning rates through a stable bracket and refine around the best development values; include tuning cost separately. If the optimum lies at the searched boundary, expand before freezing.

Primary trajectory outputs are rewrite-induced product separation, exact population risk at matched gradient evaluations, and end-to-end time to prespecified training-loss/risk levels above the estimated numerical/noise floor. Report right-censored failures rather than dropping runs that never reach a threshold. An optimization-only win does not establish a generalization benefit or a useful structured-computing advantage.


## 7. M3 Max feasibility and hardware-adaptive execution

### 7.1 Primary stack

Use native arm64 Python with NumPy and SciPy on the CPU. The mandatory path needs matrix products, QR/SVD, general solves, symmetric eigensolves for optional diagnostics, and matrix exponentials. It does not need remote compute, a paid API, pretrained weights, an accelerator, or an external dataset.

NumPy and SciPy use permissive BSD-style licenses; preserve their notices and inspect bundled dependency notices when redistributing an environment [R21,R22]. A public research repository without a clear license is not automatically reusable code. The core formulas are short enough to implement independently from the mathematics.

The investigation ran NumPy 2.3.5 and SciPy 1.17.0 on Linux. Official release-file pages confirm native CPython 3.12/macOS 14+ ARM64 wheels for both versions [R27]; this establishes an available installation route, not successful execution on the target. Prefer an available native arm64 wheel set with those versions for the first reproduction, then lock the exact resolved environment. If that combination is unavailable on the discovered Python/macOS version, select a compatible stable wheel set, record the difference, and pass the same golden tests before proceeding. Do not assume the newest documentation version is the installed version.

SciPy documents Accelerate support introduced with its 1.14 release, including wheels for recent macOS [R19]. Verify the actual BLAS backend through NumPy/SciPy configuration output. PyTorch's official MPS documentation explicitly excludes float64 and complex128 allocation; an MPS fallback setting does not make double-precision GPU tensors work [R20]. Consequently CPU float64 is the faithful default. Optional float32 MPS experiments are separate numerical-sensitivity/performance studies after the CPU claim is secure, not replacements for it.

Use the standard-library unittest runner initially; optional plotting and high-precision packages are installed and locked only when their stage needs them. The plan does not rely on an unverified GPU matrix-exponential or QR implementation.

### 7.2 Initial discovery

These are existing operating-system/package commands, unlike the proposed project CLI:

~~~bash
uname -m
sw_vers
sysctl -n machdep.cpu.brand_string
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
df -h .
python3 --version
~~~

Select a native arm64 interpreter explicitly; do not benchmark under an unrecognized x86 emulation. Do not collect serial numbers or unrelated machine identifiers. Save machine model/chip, OS build, Python architecture, memory bytes, available storage, power mode, thread configuration, and package/BLAS versions.

If no suitable interpreter is installed, obtain a supported CPython build with compatible binary wheels through the official macOS download page, or an already trusted package manager. Official universal2 installers support native Apple Silicon execution [R28]. Verify the selected executable's architecture and exact version before creating the environment; do not modify the Apple-controlled system Python. A fresh laptop is not assumed to contain the example python3.12 executable below.

A conditional bootstrap, after selecting an available interpreter, is:

~~~bash
# Replace this executable name only after interpreter discovery.
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: numpy==2.3.5 scipy==1.17.0
.venv/bin/python -m pip freeze > requirements.lock.txt
.venv/bin/python -c "import platform,numpy,scipy; print(platform.machine()); numpy.show_config(); scipy.show_config()"
~~~

If native wheels cannot be resolved, do not launch a source build or silently downgrade precision. Choose a supported stable CPU wheel set, rerun the capability tests, and record the amendment. The dependency lock should include package hashes or a retained wheel manifest in addition to exact versions.

After creating pyproject.toml and the src/rilo package, install the proposed project with:

~~~bash
.venv/bin/python -m pip install -e .
~~~

All later project commands written as "python -m rilo" mean this environment's interpreter, either through activation or the explicit .venv/bin/python path. Implement the command-line parsers before invoking those commands. The repository and CLI are deliverables of the work packages, not assets already supplied by this handoff.

Critical capability tests are multiplication, QR, SVD, solve, expm, and the smoke program, all with float64. Native library/thread initialization and representative operations must pass before any performance estimate is adopted.

### 7.3 Working-memory model

For an intentionally conservative dense implementation, budget

\[
B_{\rm arrays}\approx
8n^2(10\ell+40)+8nb(2\ell+6)\quad\text{bytes},
\tag{27}
\]

plus interpreter, BLAS workspace, reporting buffers, and optional copied arrays. The first coefficient allows factors, gradients, directions, prefix/suffix products, trial factors, two optimizer moments, and temporary matrices. The second allows inputs/targets, activations, and backward buffers. A stateless field-only experiment uses much less.

For ell=8 and b=4n, equation (27) gives:

| n | Estimated array storage | Scientific role |
|---:|---:|---|
| 32 | 1.6 MiB | General field diagnostics |
| 128 | 26 MiB | Moderate training/cost calibration |
| 256 | 104 MiB | Optional size sensitivity |
| 512 | 416 MiB | Only if cost/scale evidence warrants it |
| 1024 | 1,664 MiB | A conditional resource example, not a mandated workload |

These are calculations, not measurements. They exclude process overhead and do not imply that the O(n^4) trace rule is computationally sensible at the larger sizes. SciPy documents quadratic-order workspace for expm with data-dependent algorithmic choices [R18]; measure peak usage rather than interpreting that statement as a byte-accurate total process bound.

Never materialize the full circuit Jacobian or Fisher matrix for this problem. A Jacobian with n^2 rows and ell*n^2 columns already costs O(ell*n^4) storage. The contextual identities avoid it.

### 7.4 Time and scaling model

| Operation | Conventional dense work | Calibration measurement |
|---|---:|---|
| Local GD direction | O(n^2) | Copy/update overhead and allocation |
| One-sided metric | About two dense matrix products, O(n^3) | Associated product ordering |
| Symmetric metric | About four dense products, O(n^3) | Shared/reused intermediates where valid |
| Naive trace K | n power products, O(n^4) | Dimension and norm dependence |
| Prefix/suffix construction | O(ell*n^3) | Include only when method needs it |
| Contextual directions | O(ell*n^3) solves/products | Factorization reuse and solve residuals |
| Exponential update | O(n^3), data-dependent constants | eta, norm, and nonnormality dependence |
| Factor training forward/backward | O(ell*n^2*b) | Actual minibatch and activation storage |
| Dense collapse/refactor | O(ell*n^3) plus chosen factorization | One-time versus per-rewrite amortization |

In the field diagnostic, full products can be constructed to evaluate the oracle. In a training timing, local methods should use their normal activation/backpropagation path; do not charge them unnecessary context-building work solely because the diagnostic harness happens to compute prefixes. Charge contextual methods the quantities they actually need. Report diagnostic overhead separately and also report total experiment cost.

Measure a dimension/depth ladder beginning with the smallest mechanism-preserving cases. Expand it only to resolve an uncertainty: dimension-dependent ranking, conditioning, amortization, or relevance to a declared structured workload. If increasing n merely reproduces an already exact two-dimensional fact, there is no scientific reason to continue.

Fit a simple cost model to measured matrix products, solves and exponentials; validate it at the next dimension before extrapolating. Record both warm per-step time and setup time. Forecast full-stage runtime as measured case cost times planned cases/methods, including tuning, failed steps, and repetitions.

### 7.5 Memory and runtime admission rules

Start with one worker and small dimensions. Determine a working-memory allowance from actual available memory and observed memory pressure, preserving room for the OS and other applications. As an initial conservative engineering allowance, use the smaller of half physical RAM and 70% of a trustworthy available-memory estimate. Adjust only from observed peaks and pressure; this is allocator/OS headroom, not a scientific limit.

Admit a larger configuration only if:

- measured previous-scale peak plus estimated next-scale arrays/workspace fits that allowance;
- the projected runtime resolves a named uncertainty;
- the operations remain numerically faithful;
- free storage covers checkpoints, raw results, failures, and environment assets.

If memory is lower than expected, stream independent cases, reduce batch size where the statistical task remains unchanged, and avoid concurrent method runs. Reduce n only if the claim still concerns the smaller domain; retain and label any lost scale question. The core n=2 obstruction and small-matrix diagnostics remain valid under very small working-memory budgets.

If an operation lacks a faithful native implementation, use CPU reference algebra or reduce a diagnostic to an analytically equivalent small fixture. If no faithful path exists, block the dependent claim and complete unaffected stages. Do not replace expm with Cayley or a GPU approximation without changing the claim.

### 7.6 Timing protocol

Use perf_counter_ns or an equivalent monotonic timer. Warm native libraries and representative kernels before recording. Batch tiny operations until timer/dispatch measurement overhead is small relative to the measured block, while preserving the same allocation policy across methods.

Randomize paired method order within timing blocks. Record power mode, thread settings, and thermal/time drift. Treat repeated timings as nested measurements of the same instance, not independent scientific samples. Use a per-instance robust timing summary and a separate distribution of timings across independent instances.

For substantive per-method memory comparisons, run each method/configuration in a fresh subprocess with the same imports, inputs, and warm-up policy, one subprocess at a time. Record the common baseline and total child-process peak RSS separately, with the collector, its original units, and conversion to bytes. A high-water RSS counter in one long-lived process cannot identify later methods' individual peaks; subtracting two high-water readings does not establish peak incremental memory. If only a shared-process measurement is available, label it whole-run peak RSS and use array accounting for the method comparison. Keep memory-profiling overhead separate from update timings, and retain required setup/context allocations in the reported total.

For any optional asynchronous accelerator path, synchronize before and after the timed region and include transfers and fallback costs. Record preprocessing, adaptation, update, context maintenance, rewriting, and inference separately, then combine them for an explicit number of steps/rewrites. A method cannot hide its expensive setup outside every comparison window.

### 7.7 Storage, checkpoints, and recovery

For a checkpoint containing factors and s state arrays per factor, a first estimate is \(8(1+s)\ell n^2\) bytes, plus configuration and data/batch position. Keep two completed checkpoints during atomic replacement. Estimate result-table storage from actual pilot row size times all planned rows. Preserve exact small failing fixtures; otherwise retain seeds, generator code hashes, and selected arrays sufficient to reproduce the case.

Save recoverable working checkpoints after each case or complete accepted training step. These checkpoints do not count as completed statistical suites. Save:

- factor arrays, masses, optimizer state and dtype;
- all RNG states and data iterator position;
- accepted step, effective time and rewrite schedule position;
- configuration/source/dependency hashes;
- cumulative timings and failure status.

Use one immutable completed-suite JSON file per `(run_id, replication_id, phase, suite_id)` as the canonical result, plus an atomically replaced manifest indexing each suite key, file path, SHA-256, row count, and frozen provenance hashes. A suite file contains all planned rows, including terminal failure records; no pending row is allowed. Validate the expected row-key set from Section 10.1 before committing. JSONL tables are regenerable exports of these canonical files, never an additional source of completed observations.

Use a single writer per run. Flush, fsync, and atomically publish referenced arrays first, recording their dtype, shape, and hash. Write the complete suite JSON to a temporary file in the destination directory, flush and fsync it, then atomically rename it to its final path and synchronize the directory where supported. If that final suite path already exists, validate and reuse the completed file rather than overwrite it. Next write, flush, fsync, and atomically replace the manifest, synchronizing its directory where supported. Record any filesystem durability limitation; atomic visibility alone is not a promise against every storage failure.

Recovery must cover each interruption window:

- **During a temporary write:** ignore the incomplete temporary file, restore the last valid working checkpoint, and finish the uncommitted suite. Its partial rows contribute zero completed suites.
- **After the suite rename but before the manifest update:** validate the final suite file, referenced arrays, expected row keys, and provenance; adopt it into the manifest exactly once without rerunning or appending its observations.
- **During or after a manifest replacement:** rebuild or reconcile the manifest from validated final suite files, keyed by the unique suite tuple. Never sum duplicate manifest entries. A manifest entry pointing to a missing/corrupt file, or conflicting records for one key, is a reported integrity failure that blocks the affected result rather than silently losing or double-counting it.

Before resuming computation, validate run/replication IDs and configuration, generator, source, dependency, and environment hashes against the frozen record. Verify the recomputed W against the stored product before accepting new training steps, and rebuild derived prefix/suffix caches rather than trusting stale copies. If the environment changed, run a differential smoke check and start a linked run if numerical behavior changes materially. Do not combine pre-change and post-change confirmatory trajectories silently.

A checkpoint round trip must reproduce the next step within the declared deterministic tolerance. Inject interruptions at the three commit windows above on a tiny fixture and verify that recovery yields the same unique row-key set and exactly the same number of completed suites as an uninterrupted run.

## 8. Ordered work packages and decision gates

Every project command in this section is **to be implemented**. The coding agent has authority to choose routine non-destructive implementation details. Changes to hypotheses, evaluation populations, primary metrics, baseline access, or interpretation require a recorded methodological amendment before new confirmation.

### WP00 — Establish provenance and target capability

**Objective / risk:** prevent a valid theorem study from being confused with an unverified target-machine benchmark.

**Dependencies / inputs:** this handoff; the target Mac; optional original notebook for provenance comparison. The plan remains executable without that notebook because essential formulas and fixtures are included.

**Actions and files:** implement environment.py and the command:

~~~bash
python -m rilo discover --output runs/environment.json
~~~

Record hardware, native architecture, OS, RAM, disk, dependency versions, BLAS/thread settings, and capability results. Copy the reference functions and smoke program from Sections 5.3–5.4 into a standalone bootstrap_smoke.py before invoking the full package CLI, so this gate does not depend on WP01's later implementation. Create the dependency lock and THIRD_PARTY_NOTICES.md. Preserve the notebook hash if the notebook is supplied. Store source retrieval dates and access limitations.

**Tests / outputs:** valid float64 operations and Section 5.4; a machine-readable capability record; no implicit precision fallback.

**Resources:** tiny matrices and dependency downloads only. Measure interpreter/library baseline RSS.

**Acceptance / next action:** proceed to WP01 when CPU reference operations work. On wheel incompatibility, select and verify a compatible native CPU set. On target unavailability, complete theorem/code work but mark all Mac timing claims blocked. Do not infer an M3 throughput from this investigation's Linux timings.

### WP01 — Reproduce the theorem package and new quantitative propositions

**Objective / risk:** detect a false theorem, incorrect scope, or formula/transposition error before building a benchmark.

**Dependencies:** WP00 capability path; equations in Section 3.

**Actions and files:** write protocol/theory.md with the companion proof, zero-update proof, degree argument, Lipschitz condition, identity-insertion strengthening, midpoint theorem, and exact adverse split. Implement test_algebra.py and the deterministic fixture generator.

~~~bash
python -m rilo verify-theory --config configs/smoke.json
python -m unittest discover -s tests
~~~

**Tests / outputs:** exact rational checks for the 2-by-2 examples; numerical companion/cyclic/noncyclic checks; correct exponential versus Cayley distinction; one JSON result per fixture. Reproduce the notebook's qualitative results without depending on saved outputs.

**Resources:** begin with n=1,2,3,5; use n=2,3,5,10 for the cycle fixtures. Higher precision is only needed to resolve an actual numerical ambiguity.

**Acceptance / next action:** proceed on a coherent proof and correct controls. If a proof gap appears, distinguish a missing hypothesis from a genuine counterexample; repair and document, or stop the affected claim. Numerical disagreement alone triggers diagnosis, not a theorem rejection.

### WP02 — Build and validate the measurement harness

**Objective / risk:** make sure an apparent invariance improvement is not a mass, sign, gradient, or metric bug.

**Dependencies:** WP01.

**Actions and files:** implement circuits.py, generators.py, metrics.py, and the immutable snapshot. Implement both direct residual evaluation and equation (18). Add centered-gradient checks and split/merge event records.

~~~bash
python -m rilo validate-harness --config configs/smoke.json
~~~

**Tests / outputs:** zero loss gradient; zero-rule stall; known ascending rule; nilpotent shear; exact contextual velocity; one-sided insertion; midpoint equality; adverse general split; finite-step eta convergence; simultaneous-versus-refreshed gradient distinction. On a tiny fixture, inject interruption during suite writing, after suite rename/before manifest update, and during manifest replacement; recovery must preserve the expected unique row keys and completed-suite count without duplicate observations (Section 7.7).

**Resources:** small CPU arrays. Cache only what the tested method may use.

**Acceptance / next action:** proceed only when the harness reports both known equalities and known failures. If all methods look equally invariant, inspect the product gradient, mass normalization, and denominator. If the zero rule ranks first, correct the stall handling before any tuning.

### WP03 — Implement strong baseline families and contextual controls

**Objective / risk:** avoid manufactured gains from a weak or incorrectly scoped comparator.

**Dependencies:** WP02.

**Actions and files:** implement rules.py and context.py using the provided formulas; add scalar-normalized variants and coefficient tuning. Implement anchor and collapse/update/refactor controls. Use the original old-gradient snapshot for all simultaneous steps.

~~~bash
python -m rilo baseline-checks --config configs/development.json
~~~

**Tests / outputs:** every calibrated local method has F(I,G)=G; positive pairing for local metric methods; exact contextual output velocities; correct anchor product; any damping or normalization appears in method metadata. Save per-method computational primitives and information access.

**Resources:** calibrate n=8 and n=32 before increasing. Trace rule uses only the algebraically informative range.

**Acceptance / next action:** proceed when a competent implementation of each relevant simple competitor is verified. If baseline tuning or generic caching removes a gain, keep that result. If a newer contextual method cannot be reproduced, restrict claims to the local-information regime or block the contextual-performance comparison.

### WP04 — Pilot, calibrate, and freeze the protocol

**Objective / risk:** resolve numerical sensitivity, cost, and sample-size uncertainty without using protected evaluation outcomes.

**Dependencies:** WP03.

**Actions and files:** generate development and pilot suite indices; tune local coefficients and practical optimizer settings if needed; estimate time/RSS and paired-difference variance; choose a precision target and fixed confirmation count. Write and hash protocol/confirmatory.json and configs/confirmatory.json.

~~~bash
python -m rilo run --phase development --config configs/development.json
python -m rilo run --phase pilot --config configs/development.json
python -m rilo freeze --protocol protocol/confirmatory.json
~~~

**Tests / outputs:** baseline search includes the boundary and records solver outcomes and its stopping criterion; the selected fixed comparator is best among all eight declared fixed methods under the frozen tie-break; the candidate is labeled best-found unless a certificate is computed; variance estimates and resource forecast are recorded; confirmatory seed manifest is unconsumed.

**Resources:** derive total demand from measured case costs and equation (27). Stream cases. Pilot size grows only to improve a named variance/cost estimate.

**Acceptance / next action:** proceed to WP05 with a frozen, feasible protocol. If candidate and comparator coincide, report no distinct empirical candidate; a descriptive frontier can still be completed. If required precision exceeds capacity, record the detectable effect and declare the ambitious empirical hypothesis locally inconclusive; keep deterministic results.

### WP05 — Execute the general-rewrite confirmation

**Objective / hypothesis:** evaluate H3 on cases that did not select the method.

**Dependencies:** valid freeze from WP04.

**Actions and files:** run the exact planned suite indices, paired across methods; preserve failures; compute the frozen primary interval and secondary diagnostics.

~~~bash
python -m rilo run --phase confirmatory --config configs/confirmatory.json
python -m rilo analyze --phase confirmatory --protocol protocol/confirmatory.json
~~~

**Tests / outputs:** correct instance counts; no duplicated seeds; configuration/source hashes match; method order randomized for timings; all planned strata represented. Produce effect table, per-stratum table, failure table, and raw rows.

**Resources:** forecast from WP04; checkpoint after every suite. No optional early stopping for a favorable result.

**Acceptance / next action:** a valid negative comparison completes H3. A positive discrepancy result advances only as a metric-specific finding unless guardrails and costs also support the proposed use. A numerical or implementation defect invalidates affected observations and requires a documented fresh confirmation after repair. An interval too wide is inconclusive.

### WP06 — Test trajectory relevance when justified

**Objective / hypothesis:** evaluate H5 without confusing an integrator benefit with a learning-rule benefit.

**Dependencies:** WP05 or a specifically recorded unresolved trajectory question from WP02. Does not depend on a favorable p-value; does depend on a scientifically meaningful question.

**Actions and files:** implement integrators.py and the fully specified synthetic calibration population in Section 6.7; create paired rewrite/no-rewrite schedules; include dense task-level controls. Freeze a separate trajectory protocol containing every required configuration field before its protected evaluation; a missing field blocks this stage.

~~~bash
python -m rilo run --phase trajectories --config configs/trajectories.json
~~~

**Tests / outputs:** correct gradients; same W0 and data order; mass conservation; exact pre-rewrite W; step-halving consistency; line-search cost; training loss and exact population risk; censored time-to-target; condition and determinant-component trajectories.

**Resources:** select n, depth, and batch from measured capability and the question. The small shear mechanism does not require large n; a scale claim does.

**Acceptance / next action:** if field improvements do not transfer, narrow the result to diagnostics. If a direct dense solve or collapse baseline dominates, do not claim a better training workflow. Escalate toward a real structured library only when an actual workload and structural saving have been identified.

### WP07 — Run targeted robustness and independent confirmation

**Objective / risk:** expose attribution errors and boundary failures.

**Dependencies:** WP01–WP05; WP06 only for trajectory-specific checks.

**Actions and files:** run the prespecified stress suite: rotations, transpose, determinant components, unequal/near-endpoint masses, condition and scale sweeps, near-nilpotents, repeated splits/merges, numerical precision, and context transport. Repeat any newly claimed empirical finding on fresh suite seeds and an independently implemented residual formula.

~~~bash
python -m rilo run --phase stress --config configs/stress.json
python -m rilo confirm-replication --protocol protocol/confirmatory.json
~~~

**Tests / outputs:** robustness matrix with every failure retained; exact adverse split present; residual-versus-condition and cost plots; updated interpretation boundaries.

**Resources:** target only concrete remaining risks. Do not rerun an expansive sweep when a simple exact counterexample answers the question.

**Acceptance / next action:** a result that depends on one orientation, precision artifact, or favorable slice is narrowed accordingly. If the scope changes, write an amendment and generate fresh independent confirmation rather than repurposing the old test set.

### WP08 — Assemble an auditable final research record

**Objective:** allow another coding agent or researcher to reproduce and assess the result without chat history.

**Dependencies:** all applicable earlier packages; negative/inconclusive branches are legitimate inputs.

**Actions and files:** generate reports/findings.md, machine-readable tables, static plots, an environment manifest, and a claim-to-evidence table. Every figure caption identifies deterministic theory, exploratory data, or protected confirmation.

~~~bash
python -m rilo report --output reports/
python -m rilo audit-artifacts --report reports/findings.md
~~~

**Tests / outputs:** all referenced files exist; no claimed run is missing raw data; source/config/dependency hashes resolve; rejected and failed runs appear; report regeneration from saved artifacts succeeds; resume test passes.

**Resources:** ordinary CPU reporting; avoid loading all matrices when scalar tables suffice.

**Acceptance / next action:** conclude with one of the interpretations in Section 11. If no practical advantage survives, report that outcome explicitly. A valid limitation theorem plus a negative empirical result is a completed project.

## 9. Attribution, robustness, and failure analysis

### 9.1 Required ablations

| Question | Controlled comparison | What would change the interpretation |
|---|---|---|
| Is the gain just a changed step magnitude? | Raw, scalar-normalized, and direction-only diagnostics; report c, g and s | Gain disappears after legitimate baseline calibration |
| Is it specific to a convenient basis? | Orthogonal conjugations, independent left/right orientations, transpose | Benefit confined to canonical E12 alignment |
| Is identity insertion an unrepresentative case? | General splits and exact adverse diagonal example | No advantage beyond the theorem fixture |
| Does extra information explain the result? | Strict local API versus one-metric, full context, and anchor | Contextual advantage cannot be attributed to a local rule |
| Is it an integrator effect? | Analytic fields, common Euler eta ladder, exact exponential/anchor controls | Difference vanishes in the small-step field limit |
| Does matrix size matter? | Mechanism-preserving dimension ladder | Ranking/cost changes justify additional scale study |
| Is it numerical cancellation? | Alternate algebra, exact small examples, selective higher precision | Apparent violation disappears under faithful arithmetic |
| Is regularization responsible? | Damped versus undamped methods, with damping logged | Claimed invariance only holds in an altered geometry |
| Does history/state transport matter? | Correctly transported state versus explicitly named reset policy | A reset failure does not establish an invariant method's failure |
| Is the context cost hidden? | Recompute, transport, and amortize across recorded rewrite frequencies | Gain erased by maintenance or collapse baseline |
| Does invariance remove useful bias? | Matched training loss and held-out error in an explicitly underdetermined task | Better consistency accompanies worse prediction |
| Is tuning unfair? | Strong fixed comparator plus tuned simple blends, same development data | Improvement disappears after comparable tuning |

### 9.2 Domain-specific adversarial cases

**Bounded-condition composition.** Decompose a large diagonal change of basis into many small-condition roots. With an unrestricted domain, exact covariance under each step implies exact covariance under their product. For approximate methods, measure error accumulation and amplification. A fixed per-step tolerance is not a path-independent guarantee.

**Tiny identity response.** A near-zero K can make absolute residuals small while learning stops. Require the prescribed K or report descent/gain explicitly. Never compare the trace rule's tiny field to an identity-calibrated rule without acknowledging the difference.

**Equal condition number, different scale.** Condition number does not control matrix norm. Vary overall scale independently; retain the normalization-safe witness of Section 3.7.

**Well-conditioned product, ill-conditioned factors.** Reciprocal factors can make W benign while local updates or context solves are unstable. Log factor, prefix, suffix and product conditions.

**Nearly singular iterates.** A finite Euler step can leave GL; exponential updates preserve mathematical invertibility but may become numerically singular. Do not silently switch to a pseudoinverse and claim the same theorem domain.

**Disconnected invertible domain.** Opposite determinant signs are separate components. Distinguish failure to reach another component under a continuous/exponential flow from poor optimization within a component.

**Lost structure.** Full matrix exponentials, inverse-Gram corrections, and dense local directions can fill sparse/butterfly factors. Any structured-runtime claim must include sparsity/fill-in and the cost of reprojecting/refactoring.

**Finite precision versus algebraic similarity.** Very ill-conditioned conjugation can change traces numerically. Diagnose backward error before attributing a discrepancy to a new phenomenon.

**Noncommuting update sequencing.** Sequentially refreshed gates, stale contexts, and different rewrite order can all break exact finite-step statements. Log update ordering.

**Cancellation across gates.** Small global residual can conceal very large opposing local updates. Report local update norms and condition growth alongside the output residual.

### 9.3 Failure taxonomy

| Code | Meaning | Required action |
|---|---|---|
| IMPLEMENTATION_INVALID | Sign, transpose, gradient, mass, state, or serialization defect | Fix; invalidate affected runs; rerun protected evaluation on fresh instances if conclusions were inspected |
| NUMERICAL_UNRESOLVED | Condition/roundoff prevents a faithful distinction | Use alternate formulas or small high-precision checks; otherwise mark unresolved |
| PROXY_INADEQUATE | Workload cannot express the proposed mechanism or claimed benefit | Change scope transparently; freeze a new test population |
| RESOURCE_BLOCKED | Faithful workload exceeds measured local capacity | Stream/reduce only if claim preserved; report untested scale |
| HYPOTHESIS_REJECTED | Valid evidence contradicts the stated claim | Retain result and stop pursuing that claim without new evidence |
| NO_ADVANTAGE | Mechanism works but an existing simpler method matches or dominates | Report replication/tradeoff; do not rebrand as novelty |
| INCONCLUSIVE | Uncertainty overlaps the relevant range | State detectable effects and what additional evidence is needed |
| SUCCESS_NARROW | Positive result within explicit assumptions and metrics | Report boundaries and independent confirmation; do not extrapolate |

## 10. Reproducibility and reporting requirements

### 10.1 Experiment record

Use the canonical completed-suite JSON files and recoverable manifest described in Section 7.7, with arrays in a standard format carrying dtype, shape, and hash. Export JSONL from the validated canonical files when needed. The unique row key is `(run_id, replication_id, phase, suite_id, case_id, method_id, mass_id)`; `case_id` identifies the frozen stratum and any within-stratum data repetition. Store timing repetitions within that row rather than counting them as new scientific observations. A minimum per-case record is:

~~~json
{
  "schema_version": 1,
  "run_id": 0,
  "replication_id": 0,
  "phase": "confirmatory",
  "suite_id": 0,
  "stratum_id": 0,
  "data_repetition_id": 0,
  "case_id": "generated-stable-id",
  "method_id": "mixed_metric",
  "mass_id": 1,
  "config_sha256": "computed",
  "source_revision": "actual-git-commit-or-source-tree-hash",
  "environment_id": "computed",
  "generator_revision": "computed",
  "seed_derivation_version": 1,
  "rng_bit_generator": "PCG64",
  "seed_namespace": "confirmatory",
  "seed_material": [20261031, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0],
  "n": 8,
  "depth": 2,
  "mass_fraction": 0.5,
  "gradient_family": "nilpotent_rank_one",
  "factor_conditions": [],
  "prefix_conditions": [],
  "suffix_conditions": [],
  "product_condition": null,
  "gradient_norm": null,
  "parent_update_norm": null,
  "split_update_norm": null,
  "residual_absolute": null,
  "residual_relative": null,
  "parent_pairing": null,
  "split_pairing": null,
  "parent_cosine": null,
  "split_cosine": null,
  "timing_seconds": {},
  "peak_rss_bytes": null,
  "solve_residuals": [],
  "status": "pending",
  "failure_code": null,
  "artifact_paths": [],
  "timestamp_utc": "generated"
}
~~~

The strings "computed" and "generated" are schema placeholders, never acceptable values in completed records. JSON does not support NaN or Infinity; write null plus an explicit status/reason.

Use master seed **20261031** and the fixed integer namespaces `development=1`, `pilot=2`, `confirmatory=3`, `robustness=4`, `timing_order=5`, `trajectories=6`, and `bootstrap=7`. The CLI phase `stress` maps to `robustness`. Allocate and persist nonnegative integer run and replication IDs before drawing data, taking the next unused IDs from the project registry: resume reuses them; a new independent run receives a new run ID; a fresh replication receives a new replication ID. Freeze the stratum, mass, and auxiliary-stream integer maps with the protocol. Never use Python's process-dependent `hash()` for identifiers or seeds.

The following seed derivation is part of the reproducibility contract; store the returned integer list in the corresponding record:

~~~python
import numpy as np

MASTER_SEED = 20261031
SEED_NAMESPACE = {
    "development": 1, "pilot": 2, "confirmatory": 3,
    "robustness": 4, "timing_order": 5, "trajectories": 6,
    "bootstrap": 7,
}

def child_rng(namespace, *, run_id, replication_id, phase,
              suite_id=0, stratum_id=0, data_repetition_id=0, repetition_id=0,
              mass_id=None, stream_id=0):
    namespace = "robustness" if namespace == "stress" else namespace
    phase = "robustness" if phase == "stress" else phase
    if mass_id is not None and (not isinstance(mass_id, int)
                               or isinstance(mass_id, bool) or mass_id < 0):
        raise ValueError("mass_id must be None or a nonnegative integer")
    mass_slot = 0 if mass_id is None else 1 + mass_id
    words = [MASTER_SEED, SEED_NAMESPACE[namespace], run_id,
             replication_id, SEED_NAMESPACE[phase], suite_id,
             stratum_id, data_repetition_id, repetition_id, mass_slot, stream_id]
    if any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < 2**32
           for x in words):
        raise ValueError("seed words must be integers in [0, 2**32)")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words))), words
~~~

Generate each base matrix/gradient instance once using its data-phase namespace, `mass_id=None`, `repetition_id=0`, and `stream_id=0`; reuse those exact matrices for all methods and all mass fractions. Pass the case's `data_repetition_id` to both data and auxiliary streams. A row's `seed_material` records this base seed, so it intentionally has mass slot 0 even when that row has a concrete `mass_id`. Use the frozen zero-based mass ID only for mass-specific auxiliary randomness, where the function encodes it as `1 + mass_id`. For example, timing permutations use namespace `timing_order`, the underlying data phase, the actual mass ID, and timing-block index as `repetition_id`; they cannot change the data draws. Store their seed material with the timing measurements.

Bootstrap resampling uses namespace `bootstrap`, the analyzed data phase, and an independently frozen seed specification; use its resample index as `repetition_id` and mass slot 0 for the suite-averaged primary analysis. Auxiliary data-order/rewrite streams use separately frozen `stream_id` values. Retain full RNG state and iterator position for trajectory resumes, and save failures with the same provenance detail as successes. Changing method order, adding timing repetitions, or resuming a run must not change a base case's generated matrices.

### 10.2 Source and methodological records

For each external source save title, authors/organization, date/version, URL/DOI, retrieval date, inspected sections, technical claim informed, limitations, access status, and whether it supports, contradicts, or merely resembles the proposal. Pin code revisions before execution; "main" is not a reproducible version.

For each methodological amendment save old and new configuration hashes, reason, affected claims, whether protected results had been inspected, and the fresh confirmation namespace. Do not retroactively rewrite a preregistration.

The experiment ledger must distinguish:

- theorem or algebraic deduction;
- executed consistency test;
- exploratory observation;
- protected confirmation;
- resource estimate;
- unperformed proposal;
- access-limited or unresolved claim.

### 10.3 Required report contents

The report must include the exact counterexample and minimax scope, the strongest baseline, the full tuning procedure, per-stratum results, all material regressions, setup and steady-state costs, and target hardware details.

Recommended figures are standard exported plots:

1. Sharp shear bound versus kappa, with each rule and the exact curve.
2. Cycle dissipation versus epsilon on log axes, with analytic slopes and underflow markers.
3. General rewrite discrepancy versus factor/product condition, separated by gradient family.
4. Descent-angle / discrepancy / runtime tradeoff, with contextual controls labeled.
5. If WP06 runs, rewritten versus unrevised trajectory separation, training loss, and exact population prediction risk.

Use visible uncertainty for random-workload means. Do not add uncertainty bands to exact theorem curves. Plots should be regenerated from saved tables rather than manually edited.

A final claim table should state: claim; population/assumptions; strongest comparator; metric; effect and uncertainty; regressions; information/compute differences; evidence status; allowed interpretation. Published numbers from other workloads belong in the literature discussion, never in the same controlled comparison column.

## 11. Outcome interpretation and evidence-triggered pivots

| Observed outcome | Valid conclusion | Next action |
|---|---|---|
| Proofs and checks hold; an exact earlier theorem is found | Correct replication/unification | Attribute priority; retain useful diagnostics if they add value |
| Midpoint bound is sharp; general rewrite comparison is negative | Exact local tradeoff, no broad midpoint advantage | Publish/report the narrow theorem and negative control |
| A tuned local blend improves mean discrepancy but worsens descent or runtime | A measured tradeoff | State who could value that tradeoff; no blanket superiority claim |
| A local blend improves the declared practical frontier on fresh cases | Population-specific empirical improvement | Independently confirm; test a real rewrite workload before deployment claims |
| Exact context works but costs more than anchor/collapse | Contextual correctness without engineering advantage | Prefer simpler existing representation/update in that workload |
| One-metric context is cheaper and adequate for a real structure | Engineering opportunity | Compare with existing quotient/ScaledGD/LoRA-RITE methods in the matching domain |
| Synthetic consistency does not improve training behavior | Component result only | Stop training-superiority claims; retain diagnostic utility |
| Larger scale is scientifically required but locally infeasible | Local evidence plus an explicit unverified scale claim | Complete local result; specify what measured bottleneck would justify other resources |
| No differentiated result survives | Negative research outcome | Deliver the reproducible diagnosis; do not invent a new name or architecture |

Possible later directions must be evidence-triggered:

- **Restricted rewrite groups:** pursue when an actual operator library uses a smaller closed family, such as orthogonal or conformal changes. State the group and prove what is preserved.
- **Compressed or transported context:** pursue when exact context is demonstrably useful but its measured maintenance cost is the bottleneck. Compare against exact anchor/collapse and established contextual methods; derive approximation errors before broad claims.
- **Stateful/rectangular/rank-changing extensions:** pursue only after specifying state transport and the represented tangent space. The current proof does not cover them.
- **Secondary word-curvature theorem:** pursue as a separate mathematical project if its precise Hessian trichotomy remains distinct after its own literature comparison. Four runs means an ABAB subsequence, not four letter switches.

Do not infer impossibility for all practical learning from the local theorem. Equally, do not invoke possible future scale benefits to excuse an exact counterexample to a universal claim.

## 12. Audit record and remaining limitations

After drafting, the investigation conducted three distinct audit passes: scientific/prior-art review of the theorem, equivalences and novelty; experimental-validity review of population construction, tuning, inference and baseline fairness; and implementation/feasibility review from the perspective of a coding agent with only this plan and a fresh laptop. A separate mathematical check supported the first pass. Findings were cross-checked and the affected sections revised. These were research-assistant audit passes, not journal peer review or a formal proof-assistant certification. The computational checks listed in Section 1 were actually executed.

| Material defect or overclaim challenged | Correction incorporated | Residual risk |
|---|---|---|
| Treating every positive mass division as essential | Proved that one fixed-ratio pair of identity insertions suffices | A different mass-dependent/stateful interface requires a new theorem |
| Marketing the contextual lift as new | Added exact NG, quotient, ScaledGD and LoRA-RITE precedents | Historical equivalences may be broader |
| Missing full-GL fixed-depth prior art | Added LoRA-RITE and current forward-citation check; later recovered RoM primary text | These contextual implementations were not reproduced |
| Calling the symmetric metric a new optimizer | Identified classical one-sided metric ingredients | Exact defect characterization may also be known |
| Inferring general superiority from the sharp witness | Derived and executed the adverse general split | Population-specific benefits remain untested |
| Trivial invariance through zero/tiny updates | Fixed identity response; added stall and descent metrics | Different normalizations answer different questions |
| Treating condition caps as an exact escape | Added composition argument and cumulative-path tests | Restricted-domain variants need precise definitions |
| Confounding flow with discretization | Required analytic fields, common integrator checks, and exact anchor control | Near-singular finite steps remain difficult |
| Overstating minimum context | Added one-SPD sufficient construction and removed necessity claim | Minimal sufficient information remains open |
| Inferring target performance from Linux checks | Separated execution environment and required Mac calibration | Mac compatibility/throughput not measured here |
| Assuming Apple GPU double precision | Selected CPU float64 from official documentation | Optional reduced-precision studies need separate validation |
| A nominal “strong baseline” with weak tuning | Added eight fixed comparators, reproducible best-found search and an optional certificate | Continuum optimality is not presumed |
| A favorable average created by missing failures | Fixed the 270-stratum/810-case population and prohibited method-specific deletion | Unresolved failures make confirmation inconclusive |
| Analyst choice of a favorable interval | Fixed paired-t inference and mandatory bootstrap sensitivity | Coverage remains approximate |
| Changing practical optimizers during comparison | Preserved native Adam/L-BFGS semantics and their complete costs | Their state needs explicit handling under rewrites |
| A determinant-component reachability confound | Specified same-component primary tasks and separate opposite-component stress tests | Noisy fitted solutions can have a different component |
| Synthetic regression presented as a practical advance | Added least-squares and collapse/task-level controls | No real operator-library workload is supplied |
| Unclear reproduction without the notebook | Included proofs, formulas, fixtures, interfaces, reference code, editable installation and a standalone bootstrap gate | Full benchmark implementation is still proposed |
| Hidden state or duplicate observations on interruption | Added atomic completed-suite commits, deterministic namespaces, hash checks and interrupted-write tests | Bitwise portability across BLAS builds is not promised |

Targeted closure reviews of the revised mathematics, experimental protocol, and implementation/recovery sections found no further substantive correction. The handoff's Python reference snippets and smoke program were executed; its JSON example, seed reproducibility, UTF-8 encoding, citation definitions, and code-block structure were checked. This closes the planning audit, not the proposed confirmatory experiments.

Remaining novelty blind spots include older continuous matrix-function classifications, invariant vector-field literature indexed under different terminology, and inaccessible historical texts. The limited patent search is a technical landscape check, not a patentability or freedom-to-operate opinion.

The confirmatory comparisons, real application transfer, and target-Mac measurements remain unperformed. The appropriate current verdict is a supported mathematical obstruction and a sharp limited relaxation, with practical optimizer value still to be tested.


## 13. References and source-inspection ledger

All web sources below were checked on **31 October 2026**. An inspected abstract, metadata record, or repository page is identified as such; it is not described as a full reproduction. URLs identify primary publications, author material, official documentation, or a public patent disclosure. The prose and formulas in this handoff contain the essential implementation decisions.

**[R0] Supplied notebook.** *Operator circuits: exact refinement invariance and nilpotent stalling*, supplied as *Refinement-Invariant Learning Obstruction.ipynb*, dated 31 October 2026; authorship not specified. All markdown and code cells inspected; every code cell executed in order. No public archival URL was supplied. SHA-256 and the execution environment appear in Section 1. The theorem package is the project input, not independently established historical priority.

**[R1] Claudio Procesi (1976).** *The invariant theory of n × n matrices*. Advances in Mathematics 19(3), 306–381. [DOI](https://doi.org/10.1016/0001-8708(76)90027-X). Bibliography and precise polynomial-concomitant attribution inspected through [R2]; original publisher text was inaccessible. Establishes historical background for the algebraic ingredients; no assertion that the original paper was fully inspected.

**[R2] Erin Griesenauer, Paul S. Muhly, Baruch Solel (2018).** *Boundaries, bundles, and trace algebras*. New York Journal of Mathematics 24a, 136–154. [Primary PDF](https://nyjm.albany.edu/j/2018/24a-5v.pdf). Inspected introduction, concomitant definition, trace-algebra discussion and references. Supports established conjugation-equivariant terminology; its holomorphic multiple-complex-matrix setting differs from this continuous real single-matrix theorem.

**[R3] Ben Blum-Smith, Soledad Villar (2023).** *Machine learning and invariant theory*. Notices of the AMS 70(8), 1205–1213. [Primary text](https://arxiv.org/pdf/2209.14991); [DOI](https://doi.org/10.1090/noti2760). Inspected invariant-to-equivariant construction and compactness qualification. Informs the distinction between classical polynomial machinery and the assumptions needed for smooth extensions.

**[R4] Simone Fiori (2005).** *Quasi-Geodesic Neural Learning Algorithms Over the Orthogonal Group: A Tutorial*. JMLR 6, 743–781. [Primary PDF](https://www.jmlr.org/papers/volume6/fiori05a/fiori05a.pdf). Inspected the GL discussion on pp. 753–754: it derives the left-metric gradient and discusses right translation. Supports the established provenance of the one-sided gradient ingredients.

**[R5] Florent Bouchard, Bijan Afsari, Jérôme Malick, Marco Congedo (2020).** *Approximate Joint Diagonalization with Riemannian Optimization on the General Linear Group*. SIAM J. Matrix Anal. Appl. 41(1), 152–170. [DOI and publisher page](https://doi.org/10.1137/18M1232838). Inspected publisher abstract, references, and publication history; author full text was bot-blocked. Supports a paired treatment of standard invariant metrics; use [R4] for the fully inspected formula derivation.

**[R6] Alberto Bernacchia, Máté Lengyel, Guillaume Hennequin (2018).** *Exact natural gradient in deep linear networks and its application to the nonlinear case*. NeurIPS 2018. [Primary PDF](https://papers.nips.cc/paper_files/paper/2018/file/7f018eb7b301a66658931cb8a93fd6e8-Paper.pdf). Inspected abstract, introduction and derivation discussion. Establishes tractable exact deep-linear natural gradients; nonlinear approximations and its experiments are not controlled results for this project.

**[R7] Anna Kerekes, Anna Mészáros, Ferenc Huszár (2021).** *Depth Without the Magic: Inductive Bias of Natural Gradient Descent*. arXiv:2111.11542, 22 November 2021. [Primary PDF](https://arxiv.org/pdf/2111.11542). Inspected parameterization invariance, matrix-factorization discussion, prefix/suffix formula, and Appendix B.4. Closely overlaps the contextual escape and supplies negative evidence about removing useful implicit bias. Its task assumptions are not a universal statement that invariance harms generalization.

**[R8] Bamdev Mishra, K. Adithya Apuroop, Rodolphe Sepulchre (2012).** *A Riemannian geometry for low-rank matrix completion*. arXiv:1211.1550. [Primary PDF](https://arxiv.org/pdf/1211.1550); [author implementation page](https://bamdevmishra.in/codes/qgeommc/). Inspected quotient setup, metric and gradient equations, plus implementation page. Full-GL fixed-rank contextual geometry is established; the MATLAB implementation was not executed here.

**[R9] Tian Tong, Cong Ma, Yuejie Chi (2021).** *Accelerating Ill-Conditioned Low-Rank Matrix Estimation via Scaled Gradient Descent*. JMLR 22(150), 1–63. [Primary PDF](https://www.jmlr.org/papers/volume22/20-1067/20-1067.pdf); [official code](https://github.com/Titan-Tong/ScaledGD). Inspected problem setup, scaled updates, guarantee assumptions, and repository landing page. Relevant inverse-Gram baseline for matching low-rank tasks. Code was not executed and its reuse license was not verified.

**[R10] Jui-Nan Yen, Si Si, Zhao Meng, Felix Yu, Sai Surya Duvvuri, Inderjit S. Dhillon, Cho-Jui Hsieh, Sanjiv Kumar (2025).** *LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization*. ICLR 2025; arXiv:2410.20625v2, 16 July 2025. [Primary full text](https://arxiv.org/html/2410.20625v2). Inspected Definition 1, diagonal-preconditioning limitation, algorithm structure, and proofs' setup. Full invertible fixed-pair invariance is a close prior mechanism with additional information/state.

**[R10-code] LoRA-RITE author repository, inspected 2026.** [Repository](https://github.com/gkevinyen5418/LoRA-RITE); [implementation](https://github.com/gkevinyen5418/LoRA-RITE/blob/main/lora_rite.py). README and code inspected, not executed. README identifies a PyTorch reimplementation with a different recipe from the original JAX experiments. QR, pseudoinverses, eigensolves and state transport require careful reproduction. No clear root/source license was visible. A nondefault relative-epsilon path appears to reference an undefined variable; verify against a pinned revision if used. None of this blocks the independent core mathematics.

**[R11] Tim Tsz-Kit Lau, Weijie Su (2026).** *Symmetry-Compatible Principle for Optimizer Design: Embeddings, LM Heads, SwiGLU MLPs, and MoE Routers*. arXiv:2605.18106v5, 14 October 2026. [Primary full text](https://arxiv.org/html/2605.18106v5). Inspected scope and Section 3. It explicitly does not require all invertible coordinate changes. Supplies current adjacent orthogonal/permutation/shift context, not a contradiction of the refinement theorem.

**[R12] Devender Singh (2026).** *The Loss Does Not See the Basis, but Adam Does*. arXiv:2608.05136v1, 5 August 2026. [Primary full text](https://arxiv.org/html/2608.05136v1); [author code](https://github.com/idevender/loss-basis-adam). Inspected setup, orthogonal gauge, structure-theorem scope, limitations and README. Code exposes CPU diagnostics and identifies an MIT license; no experiment was executed here. Stateful comparisons require the specified state transport.

**[R13] Behnam Neyshabur, Ruslan Salakhutdinov, Nathan Srebro (2015).** *Path-SGD: Path-Normalized Optimization in Deep Neural Networks*. NeurIPS 2015. [Primary PDF](https://papers.neurips.cc/paper_files/paper/2015/file/eaa32c96f620053cf442ad32258076b9-Paper.pdf). Inspected problem, rescaling symmetry and method description. Establishes optimization under a smaller symmetry; not an arbitrary-GL refinement method.

**[R14] Tianqi Chen, Ian Goodfellow, Jonathon Shlens (2016).** *Net2Net: Accelerating Learning via Knowledge Transfer*. ICLR 2016. [Primary PDF](https://arxiv.org/pdf/1511.05641). Inspected introduction and function-preserving construction. Establishes widening/deepening without changing the current represented function; does not supply the local field invariance required here.

**[R15] Andres Potapczynski, Marc Finzi, Geoff Pleiss, Andrew G. Wilson (2023).** *CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra*. NeurIPS 2023. [Primary proceedings page](https://papers.neurips.cc/paper_files/paper/2023/hash/88c3c482430a62d35e03926a22e4b67e-Abstract-Conference.html). Proceedings abstract inspected. Supports established compositional operator abstractions and structural dispatch; no CoLA implementation or training-invariance benchmark was executed.

**[R16] Tri Dao, Albert Gu, Matthew Eichhorn, Atri Rudra, Christopher Ré (2019).** *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations*. ICML, PMLR 97, 1517–1527. [Primary proceedings page](https://proceedings.mlr.press/v97/dao19a.html). Proceedings description inspected. Establishes learned efficient structured transforms. This study does not reproduce its application numbers or infer that dense GL updates preserve butterfly structure.

**[R17] Awad H. Al-Mohy, Nicholas J. Higham (2009).** *A New Scaling and Squaring Algorithm for the Matrix Exponential*. SIAM J. Matrix Anal. Appl. 31(3), 970–989. [DOI](https://doi.org/10.1137/09074721X); [author-hosted PDF](https://eprints.maths.manchester.ac.uk/1442/2/alhi09a.pdf). Inspected abstract/method motivation and its use in [R18]. Informs the established numerical exponential backend and nonnormality/overscaling cautions; this is not a new integrator implementation.

**[R18] SciPy project, current documentation inspected 2026.** *scipy.linalg.expm*. [Official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html). Signature, numerical method and memory notes inspected. Documents variable-order Padé/scaling-and-squaring and quadratic workspace order. Live documentation identified SciPy 1.18.0; actual checks here used 1.17.0, so installation must be separately recorded.

**[R19] SciPy project (2024).** *SciPy 1.14.0 release notes*. [Official notes](https://docs.scipy.org/doc/scipy/release/1.14.0-notes.html). Accelerate/macOS wheel notes inspected. Supports the CPU feasibility route; not a target-machine measurement.

**[R20] PyTorch project, documentation inspected 2026.** *MPS backend*, double-precision section. [Official documentation](https://docs.pytorch.org/docs/main/notes/mps.html). Inspected float64/complex128 restriction and fallback explanation. Determines the faithful default precision/backend decision; no MPS timing was performed.

**[R21] NumPy project, license inspected 2026.** [Official license](https://github.com/numpy/numpy/blob/main/LICENSE.txt). Redistribution conditions inspected. Supports permissive use with preserved notices; lock the actual distribution and its bundled notices.

**[R22] SciPy project, license inspected 2026.** [Official license](https://github.com/scipy/scipy/blob/main/LICENSE.txt). Redistribution conditions inspected. Same role and limitation as [R21].

**[R23] Yuhui Ding, Javier Zazo, James Hensman (2026).** *Rotated Manifold Optimization for Low-Rank Adaptation*. arXiv:2609.34264v1, 28 September 2026. [Primary full text](https://arxiv.org/html/2609.34264v1); [PDF](https://arxiv.org/pdf/2609.34264). Initial access failed during one review, but subsequent retrieval succeeded. Abstract, introduction and method organization were inspected: fixed-rank product optimization, tangent projection, rotated normalization, retraction and momentum transport. Full invariance details and experiments were not independently reproduced. Required adjacent comparator if the project expands to that domain.

**[R24] Google LLC, named applicant (2026).** *Training neural networks using adaptive matrix preconditioning with transformation invariance*. Publication WO2026076211A1, published 9 April 2026; listed priority 2 October 2024. [Public technical disclosure](https://patents.google.com/patent/WO2026076211A1/en). Inspected metadata and adapter/preconditioning description, including paragraphs 00108–00110. Relevant landscape evidence, not an assessment of legal status, claim validity or freedom to operate.

**[R25] Mahmoud Abdelgalil, Tryphon T. Georgiou (2025).** *On the factorization of matrices into products of positive definite ones*. arXiv:2507.12560v1, 16 July 2025. [Primary full text](https://arxiv.org/html/2507.12560v1). Inspected rotation constructions and historical Ballantine references. Establishes antecedents of the notebook's separate symmetric-gate branch; no exact Hessian-trichotomy antecedent was located in the inspected material.

**[R26] Laura Colmenarejo, Francesco Galuppi, Mateusz Michałek (2020).** *Toric geometry of path signature varieties*. [Primary PDF](https://arxiv.org/pdf/1903.03779); [DOI](https://doi.org/10.1016/j.aam.2020.102102). Inspected Section 3, axis-parallel word description and combinatorial signature formulas. Relevant antecedent for the secondary word/area mechanism, not evidence for the primary optimizer obstruction.

**[R27] NumPy and SciPy projects, release files inspected 2026.** [NumPy 2.3.5 files](https://pypi.org/project/numpy/2.3.5/#files); [SciPy 1.17.0 files](https://pypi.org/project/scipy/1.17.0/#files). Verified the respective cp312-cp312-macosx_14_0_arm64 wheel names and release metadata. Their SHA-256 values are 612a95a17655e213502f60cfb9bf9408efdc9eb1d5f50535cc6eb365d11b42b5 and f3cd947f20fe17013d401b64e857c6b2da83cae567adbb75b9dcba865abc66d8. These identify those specific wheel files only; retain the actual downloaded wheel manifest for the discovered environment. No target-Mac installation was executed during this investigation.

**[R28] Python Software Foundation, documentation inspected 2026.** *Using Python on macOS*. [Official documentation](https://docs.python.org/3/using/mac.html); [official macOS downloads](https://www.python.org/downloads/macos/). Inspected installation, universal2 architecture support, and the distinction from Apple-controlled system Python. Supports the fresh-machine acquisition path; exact interpreter and compatible wheels remain a discovery decision.

### 13.1 Search coverage and stopping rationale

Searches began with the notebook's claims and references, then expanded to:

- operator/factor refinement, identity insertion, function-preserving deepening, factor splitting, and optimizer representation dependence;
- matrix concomitants, single-matrix similarity-equivariant maps, polynomial trace invariants, companion matrices, nilpotent cones, invariant vector fields, and positive/dissipative pairings;
- approximate equivariance, noncompact symmetry, bounded-condition transformations, left/right invariant gradients, and quantitative descent tradeoffs;
- natural gradients in deep linear networks, quotient low-rank geometry, ScaledGD, LoRA transformation invariance, orthogonal gauge, and transported optimizer state;
- current 2026 optimizer-symmetry work and forward references, including RoM;
- official macOS/Apple Silicon numerical support, matrix exponentials, package licensing, and a targeted public patent query;
- the secondary branch's SPD products, commutator motion and axis-parallel path signatures.

Backward tracing identified classical invariant theory and matrix-group gradients. Forward/adjacent searches identified the consequential full-GL fixed-depth comparator LoRA-RITE and the newer fixed-rank RoM direction. These findings changed the novelty and baseline assessment. Later targeted searches for the exact local-refinement/nilpotent combination and midpoint identity-insertion characterization returned repeated adjacent approaches or unrelated uses of “refinement” and “descent”; no additional inspected source changed the selected narrow project.

Search coverage is substantive but not exhaustive. In particular, an inaccessible historical full text, an obscure equivalent formulation, or later work could change the priority assessment. Search failure does not prove originality. The target-Mac feasibility gate and future conditional comparator reproductions are explicitly unperformed prerequisites, not research results withheld from this handoff.

### 13.2 Completion checklist for the coding agent

Before declaring the PoC complete, verify that the report:

- preserves the original local-refinement question and identifies every relaxation;
- contains the theorem hypotheses, quantitative bound and required adverse split;
- distinguishes classical ingredients from candidate mathematical differentiation;
- uses the strongest relevant simple comparison and the correct information regime;
- includes exact and general-rewrite tests, all failures, and numerical diagnostics;
- separates exploratory tuning from protected confirmation;
- records actual target hardware, versions, time and memory;
- includes setup, context, rewrite and integrator costs in any practical claim;
- reproduces from saved configurations and survives interruption/resume;
- states a negative or inconclusive outcome whenever that is what the evidence supports.




