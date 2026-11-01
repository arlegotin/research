# Coordinate-gated residual circuits: research and executable proof-of-concept plan

**Research date:** 31 October 2026.  
**Target:** one MacBook Pro with an M3 Max; installed memory, storage and macOS version are not yet known.  
**Deliverable type:** standalone mathematical research and computational validation handoff.  
**Primary direction:** exact linear-repair classification and sharp repair-size/conditioning trade-offs for the elementary cross-coordinate sigmoid-gated residual map.  
**Status:** the derivations below were developed and challenged during this investigation. Numerical formula checks were run on Linux x86_64. Historical novelty, rigorous numerical certification, and target-laptop execution remain separate gates.

## How to use this plan

Read Sections 1–4 before implementing anything. Execute work packages WP0–WP8 in order, subject to their gates. The core does not require training a network, downloading a dataset, using a remote GPU, or purchasing an API. It produces a proof dossier, exact constructions, independently checkable numerical certificates, and an honest comparison with the strongest applicable mathematical alternatives. Optional extensions have separate entry conditions.

The source proposal is the supplied 15-page research note *Coordinate-gated residual maps: a complete injectivity criterion*, dated 31 October 2026, referred to as **[S0]**. Its SHA-256 is `9bbf33d97956aa68c7adde164d607d9966c2d2b5a2dcdc2bfef79288e4f6de8c`. It is an unpublished user-supplied document without a verified public publication record. This plan restates the definitions, proofs, and implementation details needed for the selected work; the coding agent need not possess the original PDF or this conversation.

Evidence labels used throughout:

- **Source-verified:** an inspected primary paper, official documentation, or implementation supports the statement.
- **Derived here:** a mathematical deduction, with its argument included; this is not a claim of historical priority or external peer review.
- **Measured here:** a finite computation actually performed in the stated Linux environment.
- **Proposed:** work for the coding agent; no result or runtime is asserted in advance.
- **Unknown:** a consequential unresolved fact with an explicit discovery or decision gate.

## 1. Executive assessment and recommended direction

### 1.1 Verdict

The supplied note identifies a real mathematical obstruction in a precisely defined family. For

\[
F(x)=x+\operatorname{diag}(\sigma(a_i x_i+b_i))Bx,
\qquad \sigma(t)=\frac1{1+e^{-t}},
\]

with every slope nonzero, its global injectivity theorem forces **all cross-coordinate entries of \(B\) to vanish**. The proof survived a fresh mathematical audit here. This is not evidence that every multiplicatively gated neural layer is useless, nor that bounded-data performance is poor. It concerns this self-gating rule, this identity skip, and the entire space \(\mathbb R^n\). The original note itself acknowledges these limits. [S0]

The strongest feasible next step is to answer a specific open question in its Section 9: **what constant linear corrections can restore injectivity, and what do those corrections cost in rank, size, and numerical conditioning?** The smallest faithful model is already informative:

\[
F_M(x,y)=\big(Ax+[p+c\sigma(ax+b)]y,\;qx+dy\big),
\qquad ac\ne0,
\quad M=\begin{pmatrix}A&p\\q&d\end{pmatrix},\quad R=M-I_2.
\tag{1}
\]

This keeps the original nonlinear gate and its cross-coordinate product unchanged; only the linear skip is modified. It is a deliberate restriction to an elementary mechanism, not a claim that a two-dimensional example validates an arbitrary learned butterfly network.

The investigation derived the following exact statements, with proofs in Section 4:

1. **Complete repair classification:** \(F_M\) is globally injective exactly when \(d=0\), \(q\ne0\), and \(p(p+c)\ge0\). Every such map is a smooth global diffeomorphism.
2. **Sharp correction threshold:** every repair satisfies \(\|R\|_2>1\), but the infimum is 1 even for rank-one corrections. It is never attained. This sharpens the source's sufficient obstruction \(\|R\|_2+|c|<1\) for this two-dimensional family.
3. **Sharp conditioning frontiers:** at correction budget \(r>1\), the best worst-of-two-saturation-tail minimum singular value is \(\min(1,r-1)\) for rank-one repairs, versus \(\sqrt{r^2-1}\) for unrestricted repairs. Explicit constructions attain both. A separate formula handles orientation-preserving rank-one repairs.
4. **Exact finite-box unrestricted optimum:** on \([-X,X]\times[-Y,Y]\), a scaled quarter-turn linear skip is optimal among *all* admissible linear repairs with \(\|R\|_2\le r\) for the worst local Jacobian minimum singular value. Its value reduces to a one-dimensional minimization and can be certified through a quartic polynomial.

These results give a meaningful mathematical comparison: the source's norm-2 swap is only one point in an exact design space, while a rank-two correction is an essential stronger comparator. A smaller correction is not automatically preferable: near the infimum, rank-one tail conditioning collapses linearly in \(r-1\).

### 1.2 What may be new, and what is already known

Every admissible repair in (1), after a fixed output swap and rescaling, is an **affine coupling layer**. Coupling constructions, their explicit inverses, and their triangular Jacobians are established by NICE/Real NVP. Inverse instability of affine coupling is also established, including the distinction between global invertibility and numerical stability. Rank-one updates of invertible *linear* layers are established in P4Inv. [R2–R5]

The plausible narrow contribution is the **exhaustive necessity theorem within all constant linear repairs of a fixed self-gated shear**, together with its sharp norm/rank/conditioning envelope. No exact matching statement was located in the targeted searches described in Section 12. Its proof is elementary enough that it may be an unnoticed corollary of existing results. Do not label it a new architecture, a breakthrough, or a priority-established theorem.

The finite-box spectral reduction is useful engineering and mathematical specialization. Treat its novelty conservatively. Generic interval injectivity analysis, local invertibility radii, and nonlinear neural-network verification already exist. [R8–R11]

### 1.3 Scope of success

The core can establish a sharper, reproducible component-level characterization without making an unsupported end-to-end performance claim. It should succeed as a research artifact if the proofs withstand the specified audits and the implementation faithfully certifies their consequences. A novelty-negative outcome remains possible and publishable only as replication, exposition, or a benchmark contribution, as appropriate.

The core **cannot establish** improved prediction, density modeling, training convergence, generalization, high-dimensional expressivity, accelerator speed, or superiority over optimized coupling architectures. There is no specified application or dataset that would justify choosing an arbitrary machine-learning benchmark now. If the mathematical classification is already known, do not manufacture originality by renaming the construction. Follow the explicit alternatives in Section 10.

### 1.4 Why this direction was selected

| Direction | Evidence and value | Main problem | Decision |
|---|---|---|---|
| Train the original self-gated butterfly plus low-rank layer immediately | Preserves the broad seed architecture | No demonstrated task bottleneck; global invertibility fails without a structural change; butterfly plus low rank is already established | Defer until a concrete application and mechanism advantage exist |
| Repackage conditional gates as a new invertible layer | Explicit inversion and easy implementation | This is affine coupling prior art | Use as an equivalence control |
| Build a general bounded-domain injectivity verifier | Useful for practical operating regions | Interval, IQC, MIP and general-nonlinearity verification are established; general comparison is much larger | Do not select as the primary novelty claim |
| Classify and optimize linear repairs of the elementary gated shear | Directly answers a source open question; exact strongest comparisons; no scale or dataset confound | Narrow result, novelty unconfirmed, limited high-dimensional implication | **Primary** |
| Study butterfly border approximation and norm blow-up | Independent mathematical question; source gives a nonclosedness example | Broader algebraic geometry and topology dependence; less direct to the gate mechanism | Preserve a small replication and a conditional secondary research track |

## 2. Problem statement, assumptions, and hypotheses

### 2.1 Original mechanism and actual bottleneck

The seed architecture is

\[
F_{B,R,a,b}(x)=(I+R)x+D_{g(x)}Bx,\qquad R=UV^\top,
\quad g_i(x)=\sigma(a_i x_i+b_i).
\tag{2}
\]

Here \(x\in\mathbb R^n\), \(B\) is a mixing operator, \(R\) is a constant linear correction, and \(D_v\) denotes a diagonal matrix. A butterfly representation of \(B\) can reduce arithmetic count, but does not by itself guarantee invertibility, numerical stability, or practical speed. The actual Jacobian is

\[
J(x)=I+R+D_g B+
\operatorname{diag}\big(a\odot g\odot(1-g)\odot Bx\big).
\tag{3}
\]

The last term is the causal mechanism. Bounding only the frozen matrix \(I+R+D_gB\) omits the sensitivity of gate selection to its own input. Even a small \(B\) can give a large derivative when a different input coordinate is large.

Potential beneficiaries are researchers designing reversible representation maps, numerical inverses, or structured neural circuits. The proven bottleneck is a global mathematical obstruction. The existence of a commercially useful task bottleneck, and whether invertibility is desirable for that task, remain unknown. Information loss may be useful in discriminative prediction; this project must not equate injectivity with accuracy.

For the primary study, choose \(B=cE_{12}\), retain the nonconstant first gate, and let \(M=I+R\) range over all real \(2\times2\) matrices. The second gate has no effect because the second row of \(B\) is zero. This is exactly (1), a subfamily of (2).

### 2.2 Definitions and units

- \(r=\|R\|_2\): spectral/operator norm of the correction, dimensionless once the input/output coordinate units are fixed.
- \(\operatorname{rank}R\): exact algebraic rank, not an SVD threshold label.
- \(M_0=M\), \(M_1=M+cE_{12}\): limiting Jacobians as the first gate switches fully off/on while \(y\) stays fixed.
- \(\gamma(M)=\min\{\sigma_{\min}(M_0),\sigma_{\min}(M_1)\}\): worst of these two tail margins. It is not a global Jacobian lower bound.
- \(D_{X,Y}=[-X,X]\times[-Y,Y]\), with \(X,Y>0\): an explicitly declared operating box.
- \(\mu_D(M)=\min_{(x,y)\in D}\sigma_{\min}(J(x,y))\): worst local inverse-sensitivity margin, in output units per input unit. Its reciprocal is the largest inverse-derivative norm at images of points in \(D\).
- \(L_D=\max_D\sigma_{\max}(J)\): local forward-sensitivity envelope. The ratio \(L_D/\mu_D\) is an envelope; report the actual maximum pointwise condition number separately if needed.
- \(\ell_D=\inf_{z\ne z'\in D}\|F(z)-F(z')\|_2/\|z-z'\|_2\): global secant lower bound on the box. In general \(\ell_D\) is **not** established merely by evaluating \(\mu_D\).

Fix coordinate units before comparing methods. Orthogonal reflections preserve the norm and singular values; arbitrary standardization or output rescaling does not. An output rescaling used to recognize coupling prior art is not free preprocessing in the repair-cost evaluation.

The sign of \(c\) can be changed by conjugation with \(\operatorname{diag}(1,-1)\), an orthogonal coordinate reflection. Thus constructions may be explained for \(c>0\) without changing the norm objective. Retain both signs in tests. The slope \(a\) and bias \(b\) remain explicit; a nonorthogonal normalization of \(a\) is not free for conditioning comparisons.

### 2.3 Assumptions and consequential unknowns

| Item | Core assumption or discovery rule | Consequence if different |
|---|---|---|
| Gate | Exact logistic sigmoid, \(ac\ne0\) | Constant gates reduce to linear algebra; different gates need a new theorem |
| Repair | Constant real \(2\times2\) linear correction; no input-dependent \(R\) | Input-dependent repairs are outside the classification |
| Domain of injectivity | All of \(\mathbb R^2\) | Failure outside a finite box does not prove finite-box failure |
| Conditioning domain | Fixed rectangle with \(y\) centered at zero | The rotation optimality proof may fail for shifted/asymmetric \(y\) domains |
| Metric | Euclidean coordinates fixed across all methods | Rescaling changes the problem and needs a methodological amendment |
| Precision | Exact parameters for statements, float64 for search, validated arithmetic for numerical certificates | A floating-point sign is not a proof near a boundary |
| Actual laptop | Unknown RAM, disk, macOS, native Python | Discover at WP0; use serial CPU execution if necessary |
| Application | None specified | No claim of prediction or commercial value; task study is conditional |
| Novelty | Not established by absence of a search hit | Any exact antecedent changes attribution and continuation decisions |

### 2.4 Hypothesis and falsification register

| ID | Claim and current status | Supporting evidence | Alternative explanation or decisive challenge |
|---|---|---|---|
| H0 | Source's all-active identity-skip classification is correct; audited but not externally peer reviewed | Proof restated in Appendix A; exact shear collisions | A valid counterexample, incorrect saturation limit, or missed singular graph case invalidates reliance on it |
| H1 | All linear repairs of (1) have exactly the classification in Section 4.1; **derived here** | Explicit determinant sign witnesses and inverse | Any injective example with \(d\ne0\), or failure of the inverse in an admissible case, refutes it |
| H2 | The stated norm/rank/tail frontiers are sharp; **derived here** | Universal lower/upper bounds plus attaining constructions | A symbolic counterexample exceeding the bound at the same norm/rank/metric refutes it; a numerical excess first triggers a measurement audit |
| H3 | The scaled rotation maximizes \(\mu_D\) over every admissible linear repair at the same norm cap for centered boxes; **derived here** | Pointwise singular-value dominance, Section 4.4 | Counterexample under the exact assumptions refutes it; a shifted box is not such a counterexample |
| H4 | Some finite-box rank-one optima differ usefully from the balanced tail-optimal family; **open empirical/optimization question** | Small exploratory optimizer examples in Section 5.5 | Gain may disappear under global certification, stronger baseline optimization, fixed units, or different boxes; it is not evidence of a new architecture |
| H5 | The specialized certificate is useful enough to justify software reuse; **proposed engineering question** | Quartic degree and one-dimensional reduction | General interval verification or an equivalent implementation may match it; certificate overhead may erase any practical speed benefit |
| H6 | High-dimensional low-rank repair retains the two-dimensional trade-off; **unproved extension** | Direct sums inherit the result only under block-diagonal repair restrictions | Cross-block repairs may invalidate rank extrapolation; no unrestricted high-dimensional claim is authorized |

### 2.5 Strongest case for and against the direction

The strongest case is unusually clean attribution: a specific derivative term causes global failure; a complete linear-repair theorem shows what structural change is necessary; exact extremizers expose the cost of restricting correction rank; and a validated scalar calculation measures finite-domain conditioning without training noise. All of this is locally testable.

The strongest case against a major contribution is that the repaired family is ordinary affine coupling and the sharp two-dimensional matrix inequalities may already be implicit in prior art. A circuit designer allowed to change the architecture or coordinate routing might simply choose a coupling, additive shear, or certified bi-Lipschitz network from the outset. Moreover, rank one versus rank two at dimension two is not a meaningful proof of global low-rank efficiency at large \(n\). These limitations constrain the claim even if every derivation is correct.

## 3. Research findings, prior art, and novelty boundary

### 3.1 Comparison matrix

References are ordinary bibliography identifiers defined in Section 12. “Code available” is distinct from “code run”; no external research implementation was executed in this investigation.

| Approach | Problem/mechanism and assumptions | Evidence, cost, implementation | Exact overlap and implication |
|---|---|---|---|
| Gale–Nikaidô [R1] | Global univalence from P-matrix **actual Jacobians** on rectangles | Classical theorem; principal-minor checking can be exponential | Supplies background; frozen-gate matrices are insufficient; no selected repair frontier located |
| NICE / Real NVP [R2,R3] | Preserve a controller block; add or affinely scale the other block | Explicit inverse and triangular determinant; implementations are widely documented; core scalar formulas need no external training code | Every admissible repaired shear is an affine coupling after fixed output operations; construction is not novel |
| Exploding Inverses [R4] | Stability of invertible architectures in finite precision | Theory plus trained-model failures; affine scaling generally lacks global bi-Lipschitz control | The instability mechanism is known; a sharp fixed-coordinate repair trade-off is a narrower question |
| P4Inv [R5] | Train invertible **linear** layers using rank-one parameter updates | Sherman–Morrison/determinant updates and numerical safeguards; paper inspected, maintained code not established | Rank-one inverse maintenance is known; does not classify repairs of the nonlinear self-gated map |
| iResNet / Residual Flows [R6,R7] | Make the entire residual contractive; compute/invert residual flows | Published theory and experiments; generic network training costs more than this PoC | Small \(B\) does not bound the multiplicative residual's global Lipschitz constant; a safe alternative architecture |
| Interval injectivity / IAVIA [R8] | Guaranteed multiplicity and injectivity analysis using interval subdivision | Low-dimensional algorithm; potentially exponential cost; author manuscript inspected, old C++ download not tested | Generic certified injectivity/multiplicity is established; useful independent methodology if a verifier claim is pursued |
| Certified invertibility MIP [R9] | Largest local invertible regions/alternative preimages for ReLU models | Exact piecewise-linear encodings, solver cost; not a faithful sigmoid implementation | Broad “first invertibility radius” claims are untenable; replacing the gate by ReLU would change the research problem |
| IQC/LMI certificates; BiLipNet [R10,R12] | Bound incremental sensitivity or parameterize strong monotonicity and Lipschitzness | Convex certificates; BiLipNet has official code, iterative inverse | Stronger safe-network context than spectral normalization alone; outside fixed nonlinear-repair class |
| GenBaB / auto_LiRPA [R11] | Bound general nonlinear computational graphs including sigmoid and multiplication | Branch-and-bound/relaxation; official CPU path documented, target behavior untested | Required contemporary comparator for a *general verifier speed* claim; not needed to refute an exact optimum theorem |
| Learned butterfly / fixed-support identification [R13,R14,R23] | Fast structured linear maps and complementary low-rank blocks | Sparse-factor products; CUDA and pure-framework paths exist; topology and kernels matter | Structured transforms and their identification are established; reuse definitions and test patterns |
| Butterfly error guarantees [R15] | Chainable sparse factorizations with approximation guarantees and attained minima | Theory and algorithms; full paper inspected | The butterfly family can be closed while its sum with a rank-restricted family is nonclosed; no contradiction with [S0] |
| Pixelated Butterfly / Monarch [R16,R17] | Hardware-aware butterfly variants, low-rank additions, block products | Published task-specific speed/quality results; optimized kernels and architectures differ | Butterfly plus low rank is not novel; published GPU speedups cannot be imported as M3 measurements |
| Linear-circuit geometry [R18] | Matrix rigidity, circuit varieties, butterfly dimension | Algebraic theory, not a trainable-gate result | Dimension \(n(\log_2n+1)\) is known; source's fixed-topology rank-distance question is narrower |
| Coupling depth/conditioning [R19,R20] | Representations by compositions of affine/block-triangular maps | Exact representation and conditioning results | Adding routing/depth changes a single-layer obstruction; do not claim a fixed-topology lower bound for a freely permuted multilayer class |
| Recent rotations/analytic bijections [R21,R22] | Sphere shear universality; smooth analytic invertible scalar families | 2026 primary sources inspected | These directions have substantial prior art; a new contribution would require a specific additional result |

### 3.2 Six explicit prior-art answers

1. **What already exists?** Couplings, contractive residual flows, numerical inverse-instability analysis, rank-one linear inverse updates, local injectivity certification, and structured butterfly plus low-rank architectures.
2. **What can be reused?** NumPy/SciPy linear algebra and stable sigmoid, SymPy exact polynomial algebra, FLINT/Arb enclosures, coupling identities, interval verification principles, and published butterfly definitions. Reimplement only the short family-specific formulas and certificate translation.
3. **What is substantively different?** Potentially the proof that *every* constant linear repair of the fixed self-gated shear must satisfy a precise structural constraint, and the exact fixed-coordinate repair norm versus rank and conditioning optimum. This is a theorem-level specialization, not architectural novelty.
4. **Why could that difference matter?** It turns an existence example into a design limit, shows a discontinuous global repair requirement as nonzero mixing tends to zero, and prevents selecting a near-minimum correction whose inverse is poorly conditioned.
5. **What isolates the contribution?** Proof of necessity, analytic attainable bounds against the entire feasible comparator class, exact norm matching, coupling-equivalence tests, and certified finite-box values.
6. **What invalidates novelty?** A prior theorem containing the same classification/frontier, an explicit reduction making it an already-stated corollary, or a prior implementation/paper with the same optimization result. Merely changing notation or attaching a butterfly implementation does not restore novelty.

### 3.3 Reconciliation of apparently conflicting evidence

An invertible frozen shear and a noninjective self-gated map are compatible because their Jacobians differ by the last term of (3). A successful contractive residual network is compatible with the source obstruction because it constrains the *whole residual*, which the self-gated shear does not satisfy globally. A globally invertible coupling can have a badly conditioned inverse because bijectivity and uniform sensitivity bounds are different properties. Finally, exact rank distance and approximation distance need not agree when parameters can diverge; the source's butterfly border construction exploits such divergence. [S0,R4,R6,R15,R18]

### 3.4 Novelty disposition

Maintain three separate entries in `claims.json`:

- `source_classification`: source theorem reproduced/audited; historical novelty unresolved; not invented by this plan.
- `repair_classification_and_frontiers`: derivation in this investigation; candidate narrow mathematical contribution, pending a targeted antecedent check and proof review.
- `implementation_and_certificates`: reproducible engineering specialization; only label a method innovation if a material new guarantee or demonstrated efficiency advantage survives the appropriate baselines.

Search coverage is substantial enough to select the direction. It is not a proof that no prior art exists, and no patentability or freedom-to-operate conclusion is made.

## 4. Selected mechanism and complete implementation mathematics

### 4.1 All constant linear repairs: classification and proof

Write \(s=\sigma(ax+b)\), \(k(x)=p+cs\). The Jacobian of (1) is

\[
J(x,y)=\begin{pmatrix}A+ca\,y\,s(1-s)&p+cs\\q&d\end{pmatrix},
\quad
\det J=dA-q(p+cs)+dca\,y\,s(1-s).
\tag{4}
\]

**Theorem T1 (derived here).** For \(ac\ne0\), the following are equivalent: global injectivity of \(F_M\); a smooth global inverse; and

\[
d=0,\qquad q\ne0,\qquad p(p+c)\ge0.
\tag{5}
\]

**Necessity.** If \(d\ne0\), set \(x_0=-b/a\). Then

\[
\det J(x_0,y)=K+\lambda y,
\quad K=dA-q(p+c/2),\quad \lambda=dca/4\ne0.
\]

The two finite choices \(y_\pm=(\pm1-K)/\lambda\) give determinants \(+1\) and \(-1\). A continuously differentiable injective map on a connected open subset of \(\mathbb R^2\) cannot have regular points of opposite orientation: invariance of domain makes it a homeomorphism onto its image, and local topological orientation is consistent. Thus \(d\ne0\) is impossible. This uses a global theorem, not an assertion that one negative determinant alone proves noninjectivity on an arbitrary box.

If \(d=q=0\), the second output is identically zero, and a continuous map from \(\mathbb R^2\) to that line cannot be injective; one can also construct collisions by varying \(y\) along an appropriate first-output level. If \(d=0\), \(q\ne0\), the second output fixes \(x\). At that fixed \(x\), the first output fixes \(y\) exactly when \(p+c\sigma(ax+b)\ne0\). Since the gate ranges over the *open* interval \((0,1)\), this is equivalent to \(p(p+c)\ge0\). If the product is negative, choose the unique \(x\) with \(\sigma(ax+b)=-p/c\), and every \(y\) has the same image.

**Sufficiency and onto property.** For any output \((u,v)\),

\[
x=v/q,\qquad
y=\frac{u-A(v/q)}{p+c\sigma(a(v/q)+b)}.
\tag{6}
\]

Under (5) the denominator never vanishes at a finite point. This is a smooth two-sided inverse on all of \(\mathbb R^2\). The endpoint cases \(p=0\) and \(p=-c\) are included; a denominator may tend to zero only at infinity.

**Known construction boundary.** The output map \((u,v)\mapsto(v/q,u)\) transforms \(F_M\) into \((x,k(x)y+Ax)\). This is an affine coupling, with a fixed reflection if \(k<0\). Its construction is established prior art. T1 concerns necessity within the original repair class. [R3]

### 4.2 Exact rank-one parameterization and minimum correction size

Under (5),

\[
R=\begin{pmatrix}A-1&p\\q&-1\end{pmatrix}.
\]

Its second row gives \(\|R\|_2\ge\sqrt{1+q^2}>1\). Every globally injective repair therefore costs strictly more than 1, independent of the size of the nonzero \(c\).

Rank one is equivalent to \(A=1-pq\). In that case,

\[
R=\begin{pmatrix}p\\-1\end{pmatrix}
\begin{pmatrix}-q&1\end{pmatrix},
\qquad
\|R\|_2=\sqrt{(1+p^2)(1+q^2)}.
\tag{7}
\]

For \(c>0\), take \(p=q=\delta>0\). Then \(\|R\|_2=1+\delta^2\to1\), proving that the infimum is exactly 1 and is unattained. For \(c<0\), take \(p=q=-\delta\). The source swap corresponds to \(p=q=1,A=0\) when \(c>0\), with rank one and correction norm 2. For \(c<0\), use its orthogonally reflected counterpart.

The discontinuity at \(c=0\) is genuine: when \(c=0\), the identity is invertible without correction; for every nonzero \(c\), (5) applies. This is a whole-space phenomenon, not a discontinuity of finite-box behavior.

### 4.3 Sharp tail-conditioning frontiers

For a correction cap \(r>1\), define the supremum of \(\gamma\) among admissible repairs, with and without a rank-one restriction.

**Theorem T2 (derived here).**

\[
\Gamma_1(r)=\min(1,r-1),\qquad
\Gamma_{\mathrm{all}}(r)=\sqrt{r^2-1}.
\tag{8}
\]

Both optima are attained; the correction-size infimum at \(r=1\) is not.

**Rank-one upper bound.** A nonzero vector in \(\ker R\) is fixed by \(M=I+R\), so \(\sigma_{\min}(M)\le1\le\sigma_{\max}(M)\). Also \(|\det M|=|pq|\), and

\[
\|R\|_2^2-(1+|pq|)^2=(|p|-|q|)^2\ge0.
\]

Consequently \(\sigma_{\min}(M)=|pq|/\sigma_{\max}(M)\le r-1\). Since \(\gamma\le\sigma_{\min}(M)\), (8)'s first bound follows. Set \(p=q=\operatorname{sign}(c)\sqrt{r-1}\), \(A=2-r\). The resulting symmetric \(M\) has eigenvalues \(1\) and \(-(r-1)\), attaining the bound. As the gate increases, \(|p+cs|\) increases; at fixed \(A,q\), increasing this magnitude cannot decrease the smallest singular value. Thus the on-tail is no worse, and the two-tail bound is attained.

**Unrestricted upper bound and construction.** From the second column of \(R\), \(|p|\le\sqrt{r^2-1}\). Since \(Me_2=(p,0)^\top\), \(\sigma_{\min}(M)\le|p|\). Let

\[
\kappa=\sqrt{r^2-1},\quad p=\operatorname{sign}(c)\kappa,
\quad q=-p,\quad A=d=0.
\tag{9}
\]

Then \(M\) is a scaled quarter-turn, \(\|M-I\|_2=r\), \(R\) has rank two, and both-tail margin is \(\kappa\). This is a mandatory comparator, not an optional weak control.

**Orientation constraint.** The balanced rank-one optimum reverses orientation. If orientation preservation is required, impose it explicitly. The sharp rank-one frontier becomes

\[
\Gamma_{1,+}(r)=\frac{\sqrt{r^2+4r-4}-r}{2}
=\frac{2(r-1)}{\sqrt{r^2+4r-4}+r}.
\tag{10}
\]

It is attained by \(|p|=|q|=\sqrt{r-1}\), with \(p\) having the sign of \(c\) and \(q\) the opposite sign. To verify the upper bound, distinguish the actual norm \(\rho=\|R\|_2\le r\) from its cap, and put \(t=|pq|\le\rho-1\). For the positive-orientation branch with nonzero tail margin, \(pq<0\), \(\|M\|_F^2=\rho^2+2t\), and \(\sigma_{\max}(M)-\sigma_{\min}(M)=\rho\). Thus the smaller singular value is \((\sqrt{\rho^2+4t}-\rho)/2\), bounded by \(f(\rho)=(\sqrt{\rho^2+4\rho-4}-\rho)/2\). This function increases for \(\rho>1\), so \(f(\rho)\le f(r)\), proving (10) for the norm cap. Endpoint scales have zero worst-tail margin. The unrestricted rotation is orientation preserving.

Near \(r=1+\varepsilon\), unrestricted tail margin is approximately \(\sqrt{2\varepsilon}\), while the best rank-one margin is \(\varepsilon\). This is a proved rank/conditioning trade-off, not a statistical effect.

### 4.4 Exact local conditioning on a centered box

For an admissible repair, \(J=\begin{pmatrix}h&k\\q&0\end{pmatrix}\), where

\[
h=A+ca\,y\,s(1-s),\quad k=p+cs,
\quad s\in[s_L,s_U]=[\sigma(b-|a|X),\sigma(b+|a|X)].
\]

At fixed \(s\), the determinant magnitude \(|qk|\) is independent of \(y\). Increasing \(|h|\) increases the larger singular value and decreases the smaller one. Thus the worst \(y\) is an endpoint, and

\[
H(s)=|A|+|ca|Y s(1-s),\qquad
\mu_D(M)=\min_{s\in[s_L,s_U]}\phi(H(s),p+cs,q),
\tag{11}
\]

where \(\phi(h,k,q)=\sigma_{\min}\begin{pmatrix}h&k\\q&0\end{pmatrix}\).

**Stable floating-point evaluation:**

\[
T=h^2+k^2+q^2,\quad
\Delta=\operatorname{hypot}(h^2+k^2-q^2,\,2hq),
\quad \sigma_{\max}=\sqrt{(T+\Delta)/2},\quad
\phi=|qk|/\sigma_{\max}.
\tag{12}
\]

Do not compute the smaller squared singular value by subtracting nearly equal terms. If all entries vanish, define \(\phi=0\); admissible core repairs have \(q\ne0\), so that special case is only a harness control.

For extreme inputs also prevent overflow/underflow in squares or \(qk\): let \(v=\max(|h|,|k|,|q|)\), evaluate the larger singular value \(w\) of the matrix divided by \(v\), then compute \(\phi=\min(|q|,|k|)\,[\max(|q|,|k|)/v]/w\). Handle \(v=0\) separately. Report overflow of \(\sigma_{\max}=vw\) as a numerical status or evaluate with Arb. This scaled implementation complements the cancellation-safe identity (12).

**Theorem T3 (derived here).** For the fixed \(a,b,c,X,Y\), the rotation (9) maximizes \(\mu_D\) among all globally injective repairs satisfying \(\|R\|_2\le r\).

This optimizes absolute inverse sensitivity: the worst inverse-derivative norm on the declared box is \(1/\mu_D\). It does not claim to minimize the Jacobian condition-number ratio, the forward envelope, or floating-point inversion error.

**Proof.** Row and column norm bounds give \(|p|,|q|\le\kappa\). At each gate value,

\[
H(s)\ge |ca|Y s(1-s),\qquad |p+cs|\le\kappa+|c|s.
\]

The function \(\phi\) decreases with \(|h|\) and increases with \(|k|\) and \(|q|\). The first fact follows from fixed determinant and increasing trace; the latter two follow by adding positive semidefinite diagonal terms to \(JJ^\top\) or \(J^\top J\). Construction (9) simultaneously attains the best values of all three quantities at every \(s\). It therefore dominates pointwise after taking the worst case over \(y\), and hence after minimizing over \(s\). This proves optimality without relying on a numerical optimizer.

The symmetry of the \(y\) domain is essential. On \([y_0-Y,y_0+Y]\), the worst term becomes \(|A+ca y_0s(1-s)|+|ca|Y s(1-s)\); a nonzero \(A\) may help. Do not extend T3 silently to that setting.

### 4.5 Rigorous quartic certificate for \(\mu_D\)

For a trial threshold \(0\le\tau<|q|\), define

\[
P_\tau(s)=q^2(p+cs)^2
-\tau^2\{H(s)^2+(p+cs)^2+q^2\}+\tau^4.
\tag{13}
\]

Then \(\mu_D\ge\tau\) exactly when \(P_\tau(s)\ge0\) on \([s_L,s_U]\). Indeed the lower-right pivot of \(JJ^\top-\tau^2I\) is \(q^2-\tau^2>0\), so positive semidefiniteness is equivalent to a nonnegative determinant; maximizing \(h^2\) is exact. The polynomial has degree at most four. At \(\tau=|q|\), this strict-pivot argument is unavailable: handle the boundary directly or retain a bracket. In particular, \(P_\tau\ge0\) alone at that boundary is insufficient.

With rational \(A,p,q,a,c,Y,\tau\), the coefficients are exact rationals. Only the gate endpoints require transcendental enclosures. Implement:

1. Construct `sympy.Poly(P, s, domain=QQ)` from exact strings/fractions, never from rounded decimal floats.
2. Isolate all distinct real roots of the derivative's square-free part. The public `Poly.intervals(eps=...)` and `refine_root` APIs are documented. A zero derivative means a constant polynomial; handle it directly. There are at most three distinct stationary points. [R26]
3. Enclose \(s_L,s_U\) with Arb. Evaluate the polynomial at both endpoint balls and at every stationary-root enclosure possibly inside the gate interval. Convert a rational interval \([l,u]\) to midpoint \((l+u)/2\) and radius \((u-l)/2\) using exact rational inputs; `arb(l,u)` is a midpoint/radius constructor, **not** an endpoint constructor. Use outward enclosures, not nearest floating-point conversions. [R24,R25]
4. Strictly positive enclosures at all candidates certify a lower threshold. An exact zero constant is nonnegative. For other zero-containing balls, refine; never guess the sign.
5. A negative upper bound at an actual gate endpoint, or at an exact rational interior point, proves the threshold too large. A negative value at a stationary-root interval outside the actual gate domain is not a valid witness.
6. Maintain a certified bracket \([\mu_L,\mu_U]\), initially \([0,|q|]\), updating only with proved outcomes. Equality can remain unresolved. If a midpoint stays unresolved, test nearby dyadic thresholds separated by less than the requested final width; report the surviving bracket if the resource gate is reached.

Use outward-directed `arb.lower()`/`arb.upper()` and exact dyadic serialization (`man_exp()` on the exact bounds) for persistent certificates; store integer mantissa and exponent, not a display string with omitted radius. A replay checker must verify input hashes, root enclosures, domain membership, endpoint enclosures, and every sign. It must also verify **completeness** of the stationary-root list by independently re-isolating the whole derivative or proving its full real-root count. Validating only the listed roots would allow an omitted minimum to create a false certificate. API support was inspected in official documentation; this rigorous backend was **not run here**. [R24–R26]

**Optional exact-subdivision fallback.** If the root-isolation path is unavailable but validated gate endpoints work, enclose the gate interval by rational endpoints and subdivide that outer interval. On each rational subinterval \([l,u]\), express \(P(l+(u-l)t)=\sum_{k=0}^d a_k t^k\), \(d\le4\), and compute Bernstein coefficients \(b_j=\sum_{k=0}^j a_k\binom{j}{k}/\binom{d}{k}\) exactly. Nonnegative coefficients certify that subinterval; a rational interior point with negative exact polynomial value, whose actual gate-domain membership is proved, rejects the threshold. Bisect undecided intervals, refine the endpoint enclosure if its excess causes uncertainty, and retain unresolved intervals when the resource gate is reached. Handle a constant polynomial directly. A positive result needs a complete covering partition with every interval certified; an omitted interval invalidates replay. This fallback trades exact root isolation for subdivision and may be much slower near a zero. It does not remove the need for sound transcendental endpoint bounds.

**Irrational extremizers are not rational coefficients.** At arbitrary \(r\), formulas (8)–(9) contain square roots. Do not cast them to float and call the resulting certificate exact for the ideal extremizer. For exact common-budget fixtures choose rational \(u>0\) with \(u^2<2\), and set

\[
\delta=\frac1u-\frac u2,\quad w=\frac1u+\frac u2,
\quad\kappa=\delta w,\quad r=1+\delta^2.
\tag{14}
\]

Then \(w^2-\delta^2=2\) and \(\kappa^2=r^2-1\). Both the rank-one \((p,q)=(\delta,\delta)\) and rank-two \((p,q)=(\kappa,-\kappa)\) constructions have rational entries and exactly the same norm \(r\). For example \(u=1\) gives \(r=5/4\), \(\delta=1/2\), \(\kappa=3/4\). For other desired budgets, retain symbolic algebraic numbers or use proven rational inner approximations, report their actual norm interval, and bound the approximation error. An interval-coefficient subdivision backend is a faithful optional fallback; it must not pretend its coefficients lie in \(\mathbb Q\).

The required source budget \(r=2\) has ideal rotation scale \(\sqrt3\). A concrete route using only the rational certificate engine is: choose a dyadic \(\kappa_0\le\sqrt3\) with a rigorous error bound \(e_U\ge\sqrt3-\kappa_0\); certify that rational rotation; and enclose the ideal value by \([\mu_L(\kappa_0),\mu_U(\kappa_0)+e_U]\). Monotonicity gives the lower endpoint. Weyl's singular-value perturbation bound gives the upper endpoint because the Jacobian difference is a constant quarter-turn of norm \(|\sqrt3-\kappa_0|\). Refine until the ideal bracket meets the requested width. More generally, with gate parameters fixed, \(|\Delta A|+\max(|\Delta p|,|\Delta q|)\) bounds the Jacobian perturbation, allowing a rigorous enclosure of an algebraic extremizer from rational approximations. Record separately the ideal theorem value and the actually implemented matrix.

### 4.6 Distinguish local conditioning from global stability

For every admissible repair with \(ac\ne0\), fix a finite \(x\) and send \(|y|\to\infty\). Then \(|h|\to\infty\) while \(|\det J|=|qk(x)|\) stays fixed. Hence \(\sigma_{\min}(J)\to0\), and neither a global forward Lipschitz bound nor a global Lipschitz inverse exists. This specializes the known affine-coupling stability issue. [R4]

A separate conservative global secant guarantee on the box is available. Let

\[
K_s=\max_{s\in[s_L,s_U]}s(1-s),\quad
H_{\max}=|A|+|ca|YK_s,\quad
k_{\min}=\min_{s\in[s_L,s_U]}|p+cs|>0.
\]

Then

\[
\ell_D\ge\phi(H_{\max},k_{\min},|q|)>0.
\tag{15}
\]

To verify it, use the exact identity for two inputs:

\[
\Delta F=
\begin{pmatrix}
A+c\,\frac{s(x)-s(x')}{x-x'}\,\frac{y+y'}2
&p+c\,\frac{s(x)+s(x')}2\\q&0
\end{pmatrix}\Delta(x,y),
\]

with the derivative limit if \(x=x'\). The scalar mean-value theorem bounds the first entry by \(H_{\max}\); the scale retains its fixed sign and magnitude at least \(k_{\min}\). Singular-value monotonicity gives (15). Report this conservative bound separately from the exact local quantity (11).

For the required forward-sensitivity guardrail, let \(k_{\max}=\max(|p+cs_L|,|p+cs_U|)\). A cheap certified upper bound is \(L_D\le\sigma_{\max}\begin{pmatrix}H_{\max}&k_{\max}\\|q|&0\end{pmatrix}\), evaluated with outward arithmetic. Label it a bound, not an exact maximum. Exact \(L_D\) maximization is optional and reduces to a one-dimensional gate problem.

### 4.7 Core computational interface and stable conventions

Implement a single map interface used by every representation:

```python
class GatedShear:
    # Parameters are stored exactly; numerical evaluation is a chosen backend.
    # theta = {A, p, q, d, a, b, c}; batch shape (..., 2).
    def forward(self, xy, backend): ...
    def jacobian(self, xy, backend): ...  # (..., 2, 2), actual Jacobian
    def inverse(self, uv, backend): ...   # only after exact admissibility check
    def classify_exact(self): ...        # bijective / noninjective / outside_scope
    def correction_exact(self): ...      # M-I, rank expression, norm enclosure
    def tail_margin(self, backend): ...
    def box_margin_certificate(self, domain, tolerance, policy): ...
```

An `EquivalentCoupling` adapter must implement the same function through the affine-coupling equations. It is a correctness/implementation control, never a different model claimed to be less accurate.

For float64 candidate evaluation, use SciPy `expit`. Evaluate the sigmoid derivative in tails as `exp(-abs(z))/(1+exp(-abs(z)))**2`, not `s*(1-s)` after `s` has rounded to 1. Evaluate both `expit(z)` and `expit(-z)` when needed. For \(p=-c\), use \(-c\sigma(-z)\) for the scale; for \(p=0\), use \(c\sigma(z)\). Any underflow is a numerical status, not a mathematical zero. Final boundary decisions use exact conditions and Arb.

The direct map may have cancellation in `u-A*x` during inversion. Record absolute/relative round-trip errors, scale, condition information, and overflow/underflow separately. Never hide nonfinite results by clipping. Use stable `log|q|+log|k|` for the log-absolute-determinant if that diagnostic is requested; determinant sign is a separate field.

All method implementations must use exactly the same input coordinates, parameter meaning, gate, domain, precision policy, and requested certificate tolerance. The core has no learned parameters or pretrained assets.

## 5. Baselines, data, metrics, and confirmatory protocol

### 5.1 Primary claims and the right kind of evidence

The primary population is the complete parameter class in T1–T3, not a sampled image dataset. Proofs establish universal statements; finite computations check implementation and illustrate consequences. Do not use seed averages, p-values, or successful optimization runs to replace a proof of injectivity or optimality.

The primary comparative quantities are:

| Claim | Metric and aggregation | Strongest valid comparison | Decision evidence |
|---|---|---|---|
| Classification | Exact necessary/sufficient parameter conditions | All constant real linear repairs of the fixed shear | Complete proof, all degenerate cases, symbolic and adversarial fixtures |
| Minimum repair size | Infimum of \(\|R\|_2\), plus attainment status | Entire rank-one and unrestricted classes | Universal lower bound and an admissible converging sequence |
| Tail frontier | \(\gamma\) at each fixed norm cap; report rank and orientation | Analytic complete feasible class, not only the source swap | Upper bound plus attaining construction; exact rank/norm verification |
| Finite-box local optimum | \(\mu_D\), certified interval per domain/budget | T3's unrestricted rotation optimum | Pointwise proof and rigorous value brackets |
| Rank-one finite-box refinement | Difference in \(\mu_D\) at common \(r,D,a,b,c\) | Balanced rank-one, source swap where feasible, globally bounded rank-one optimization, and unrestricted optimum | Certified candidate advantage; optimality claim only with global upper bound |
| Practical numerical behavior | Round-trip errors, nonfinite rate, local sensitivity, forward envelope, separate secant bound | Equivalent coupling and all relevant analytic constructions | Honest error/conditioning trade-off, no clipping or rescaling |

Do not compress this into one invented composite score. A rank-two method has extra correction rank; a smaller norm with worse sensitivity is a trade-off. Report absolute differences, ratios when denominators are nonzero, and each guardrail.

### 5.2 Baseline roster and reproduction requirements

| ID | Method/control | Role and exact configuration | Required reproduction check |
|---|---|---|---|
| B0 | Unrepaired shear | \(M=I_2\), same nonzero \(a,c\) | For \(a=1,b=0\), reproduce the source's exact three-preimage identity in Appendix A |
| B1 | Source swap | \(M=\begin{pmatrix}0&1\\1&0\end{pmatrix}\), \(c>0\); reflected version for \(c<0\) | Rank-one correction, norm 2, inverse correct; \(\gamma=1\) |
| B2 | Complete tail-optimal rank-one family | Formula (8)'s construction for every budget | Exact norm, eigenvalues and tail margin; orientation reversal disclosed |
| B2+ | Orientation-preserving rank-one family | Formula (10)'s construction | Positive determinant, correct constrained frontier |
| B3 | Optimal unrestricted repair | Scaled rotation (9) | Exact norm and rank two, \(\gamma=\sqrt{r^2-1}\); use T3 as finite-box oracle |
| B4 | Equivalent affine-coupling representation | Exactly the same \(A,p,q,a,b,c\), including fixed output operations | Forward, inverse and Jacobian agree; deviations are implementation bugs or arithmetic-order effects |
| B5 | Full admissible rank-one search | Parameterize \(A=1-pq,d=0\) exactly; enforce norm cap and scale sign | Used only for H4; optimizer must match known tail optimum before searching box objective |
| C1 | Constant-gate/zero-mixing limits | \(a=0\) or \(c=0\), with generic linear skips | Classifier returns `outside_scope` then applies linear invertibility, not T1 |
| C2 | Broken frozen-gate shortcut | Check only \(M+D_gB\) | Must incorrectly appear safe on the known shear, demonstrating why this is only a negative control |
| C3 | Near-repair perturbation | Change exactly \(d=0\) to rational \(d=\pm\epsilon\ne0\) | Reproduce opposite-orientation witnesses from (4); finite random tests may miss them |

There is no legitimate model-quality competition between B2/B3 and B4: they are two implementations of identical functions. No method may be “tuned” by changing the gate, coordinate metric, or problem definition. No expensive published model needs to be run for the primary theorem. If a later claim concerns a general verifier or a learned architecture, the corresponding stronger sources in Section 3 become mandatory direct baselines, subject to faithful local compatibility.

B1 has correction norm 2. At budgets \(r<2\), retain it only as an explicitly out-of-budget reference, never as a feasible matched-budget competitor. Apply this feasibility flag in every table and in WP6.

### 5.3 Exact fixtures and generated workloads

Use generated parameter/domain instances. There are no external dataset licenses, training labels, or pretrained assets in the core. The project should license its own code and generator metadata explicitly; package and source-document licenses remain separate.

**Deterministic branch suite.** Include at least one analytically specified case from every distinct logical branch, rather than choosing a random case count:

- \(a,c\) positive/negative; nonzero bias; nonzero but tiny slope/mixing; exact zero slope/mixing as out-of-scope controls.
- Admissible scales \(p=0\), \(p=-c\), and both open admissible branches; inadmissible scale between the endpoints.
- \(q=0\), both nonzero signs of \(q\), \(d=0\), and both nonzero signs of \(d\).
- Rank-one, rank-two, orientation-preserving and orientation-reversing repairs.
- Gate intervals containing the transition, entirely in a tail, narrow intervals, and large boxes; centered and deliberately shifted \(y\) domains with the latter excluded from T3 claims.
- Repeated stationary polynomial roots, a constant polynomial, a positive polynomial, a negative witness, and an equality case that must remain unresolved or be handled exactly.
- A numerical cancellation case such as balanced rank-one scale \(10^{-8}\), for which tail margin is approximately \(10^{-16}\).

**Exact matched-budget core fixtures.** Use (14) with rational values of \(u\) sampling three mathematically different regimes: \(r\downarrow1\), \(1<r<2\), and \(r>2\). Rational approximants to \(\sqrt2\) from below generate the first regime. Include \(u=1\) as the simplest exact fixture. Add the source point \(r=2\) symbolically; if its irrational rank-two construction is represented numerically, use a proper algebraic or enclosure path as described in Section 4.5.

**Exploration domain.** Begin with \(a=1,b=0,c=1,X=Y=4\) as a transparent worked example, not a representative application population. Then vary the dimensionless quantities \(|a|X\), \(|ca|Y\), bias placement relative to the transition, and \(c\)'s magnitude. Select additional points adaptively only to resolve a change in minimizer location, a suspected instability, or an uncertain comparison. Preserve all exploratory points.

**Confirmation suite.** After pilots, freeze exact generator code, parameter strata, fixture list, tolerance policy, methods, and analyses in `protocol/confirmatory.json`, with a cryptographic hash. Include every logical branch above and all claimed domains. Generate fresh rational instances within predeclared strata using a fresh stored seed only if a distributional generalization is intended. A seed alone is not reproducibility: store the realized rational parameters, because generator/library revisions may change outputs.

A fixed deterministic suite supports statements about that suite and the accompanying theorems. If a later paper wants “typical random repair” or timing population claims, explicitly define the parameter distribution and independent instance unit before observing confirmation results. There is no natural universal distribution of matrices to silently assume.

### 5.4 Freeze, sensitivity, and stopping rules

For the mathematical core, the stopping condition is logical completeness plus successful replay of the required certificates, not statistical significance. Failed proofs, invalid witnesses, unresolved intervals, and deviations from known frontiers remain in the report.

For certified numerical values, start with a requested relative bracket width of \(2^{-30}\) for positive \(\mu_D\). This is an engineering resolution target of roughly nine decimal digits, not a minimum improvement requirement. Use `abs_width=0` unless an application supplies a meaningful absolute scale. Stop only once \(\mu_L>0\) and \(\mu_U-\mu_L\le2^{-30}\mu_L\), or return an explicit incomplete bracket when the calibrated resource gate is reached. If a claimed difference is smaller than the combined bracket widths, refine until it is resolved or declare the comparison inconclusive. Never continue selecting domains until a favorable difference appears.

For H4, an exact or certified separation of two values shows a component-level difference on that declared instance. It does not establish typical performance. A practical benefit requires an application-defined inverse-error requirement, acceptable correction rank, and meaningful operating domain; no arbitrary percentage threshold is imposed here.

If distributional inference is later added, use independent full parameter/domain instances as the unit. Pair methods on the same instances, retain strata, and estimate uncertainty by resampling instances, not individual input points or timing repeats. Pilot variances determine a fixed sample size for a predeclared precision target; for a mean paired difference a planning approximation is \(N\approx(z_{1-\alpha/2}s/h)^2\), followed by a check that its assumptions are reasonable. Lock \(N\) before confirmation. Multiple prespecified claims use a familywise adjustment or are explicitly exploratory. Do not repeatedly inspect p-values to choose when to stop.

### 5.5 Checks actually performed during this investigation

These are **finite float64 checks**, not formal certificates, exhaustive tests, or M3 measurements. Environment: Linux x86_64 (`Linux-6.18.57-x86_64-with-glibc2.39`), Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0. No neural network was trained. Neither SymPy nor python-flint was installed in this research container; their proposed APIs were checked against documentation.

| Executed check | Observed result | Interpretation |
|---|---|---|
| Four nonzero \(d\) values down to magnitude \(10^{-9}\), with explicit \(\det J=\pm1\) witnesses | Maximum determinant discrepancy \(4.44\times10^{-16}\) | Formula arithmetic agrees; proof supplies the global conclusion |
| 1,000 seeded rank-one parameter checks, positive \(p,q\) drawn log-uniformly from \(e^{-4}\) to \(e^2\) | Relative norm-formula discrepancy at most \(6.58\times10^{-16}\); no sampled tail-bound violation | Finite diagnostic, not a proof of the universal optimum |
| Closed-form forward/inverse round trip on those 1,000 cases with inputs in \([-4,4]^2\) | Maximum absolute error \(1.31\times10^{-14}\) | Consistent implementation; does not bound worst-case error |
| Stable two-by-two singular-value formula against NumPy SVD on 1,000 random triples | Maximum absolute discrepancy \(4.44\times10^{-16}\) | Formula implementation check |
| Independently coded 64 parameter cases, 101 by 33 input grids | SVD discrepancy at most \(1.11\times10^{-15}\); full-grid and exact-\(y\) reduction minima differ by at most \(2.22\times10^{-16}\) | Two computational paths agree on sampled points; no certified continuous minimum |
| 6,464 quartic/singular-value threshold comparisons in that independent check | Zero disagreements | Translation check away from numerical ambiguity; not interval verification |
| Balanced scale \(10^{-8}\) | Stable formula resolves a tail margin near \(10^{-16}\); naïve subtraction returns zero | Demonstrates a material cancellation failure |
| Equivalent coupling on 1,024 inputs | Forward discrepancy \(8.88\times10^{-16}\) | Algebraically identical map, arithmetic ordering only |
| Source's size-16 butterfly border construction at \(t=2,3,10\) | Squared errors \(1.125,0.4691358025,0.0402\), agreeing with \(4/t^2+2/t^4\) | Checks the displayed example; does not establish historical novelty |

For a transparent exploratory example with \(a=c=1,b=0,X=Y=4\), the following values were obtained by gate-grid candidate discovery plus scalar local refinement. The tail entries agree with analytic formulas; the box entries are **uncertified optimizer estimates**:

| Correction norm \(r\) | Rank-one tail \(\gamma\) | Rank-two tail \(\gamma\) | Balanced rank-one estimated \(\mu_D\) | Rotation estimated \(\mu_D\) |
|---:|---:|---:|---:|---:|
| 1.10 | 0.100000 | 0.458258 | 0.0967321 | 0.292172 |
| 1.50 | 0.500000 | 1.118034 | 0.401313 | 0.882253 |
| 2.00 | 1.000000 | 1.732051 | 0.771809 | 1.470705 |

These are intentionally unfavorable to a “rank one is better conditioned” narrative. They illustrate why rank must be recorded as a cost and why a tail margin cannot be substituted for a box margin.

A separate exploratory rank-one search at \(a=1,b=0,c=1/2,X=Y=2,r=2\) found \(p\approx0.9445516,q\approx1.0554461\), estimated \(\mu_D\approx0.9711977\), versus \(0.9498817\) for the balanced family. This is only a lead for H4. It was not certified globally and cannot be described as a confirmed improvement. The coding agent must use rational admissible candidates, verify the norm cap, certify both values, and compare to the unrestricted optimum.

To reproduce the essential checked formulas without this conversation, implement (4), (7), and (12); use `numpy.random.default_rng(20261031)` for the first audit, draw the stated \(p,q\) and input distributions, and compare to `numpy.linalg.norm(..., 2)` and `numpy.linalg.svd(..., compute_uv=False)`. The independent 64-case audit is a separate verification path, not a required bit-for-bit fixture; its reported outputs are historical observations only. All future exact confirmation inputs are saved explicitly by WP5.

## 6. M3 Max feasibility and hardware-adaptive configuration

### 6.1 Stack selection

Use **native arm64 CPython + NumPy + SciPy + SymPy + python-flint/Arb**, with `pytest` for the meaningful scientific correctness tests. Use standard-library JSON, `fractions.Fraction`, `hashlib`, `pathlib`, and `time` for metadata and exact parameter transport. Matplotlib is optional for static plots in the execution report.

The core matrices are two by two and the certificate polynomial has degree four. GPU acceleration offers no necessary scientific capability here. Official Python-FLINT documentation lists macOS ARM binary distributions; official SymPy documentation exposes rational polynomial root isolation. This supports feasibility, conditional on a successful target installation and tests. [R24–R26]

MPS does not support float64/complex128; the current PyTorch documentation states that CPU fallback cannot make double tensors allocatable on MPS. MLX likewise documents float64 as CPU-only. Optional accelerator evaluation can use float32 for candidate generation or throughput studies, but all scientific boundary/certificate decisions remain on the CPU reference path. Do not replace a failed float64 accelerator operation by float32 without labeling the resulting change. [R27,R28]

The core uses public packages with no pretrained weights. Python-FLINT's binding is MIT-licensed; its numerical dependencies have their own licenses, with current FLINT documentation indicating LGPL-3-or-later. Inspect the actual installed notices before redistributing a bundled environment; do not infer dependency licensing solely from the wrapper's license. [R29] Installing and using the packages locally is distinct from redistributing binaries.

### 6.2 Initial discovery: existing commands

These are real macOS/system/Python commands, not invented project interfaces:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.physicalcpu
sysctl -n hw.logicalcpu
df -k .
vm_stat
python3 -c 'import platform, sys; print(platform.machine()); print(sys.version)'
```

Save outputs with timestamps. Require native `arm64` in both OS and Python. If the existing interpreter runs under Rosetta, select a native supported CPython; do not silently mix architectures. Discover actual free disk and available memory; do not assume the largest advertised M3 Max configuration.

Create an isolated environment and resolve dependencies only after selecting a supported interpreter:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy scipy sympy python-flint pytest
.venv/bin/python -m flint.test
.venv/bin/python -m pip freeze --all
.venv/bin/python -c 'import numpy; numpy.show_config()'
```

The latest package release is not inherently the right scientific version. If wheel compatibility fails, select a currently supported arm64 CPython from the inspected compatibility table and retry in a new environment; preserve failures. After successful smoke tests, lock exact resolved versions and wheel/source hashes. Use that lock for all compared runs. Record BLAS vendor/thread settings. Do not force an OS upgrade solely to enable an optional GPU path. [R24,R30]

### 6.3 Memory and storage model

**Core:** interpreter overhead plus a handful of degree-four rational polynomials and arbitrary-precision balls. Process cases sequentially by default. Peak usage depends mainly on integer bit lengths, library overhead, and the number of simultaneous workers. There is no need to load a large model or retain every subdivision node in RAM.

For an optional batch of \(N\) two-dimensional samples in float64, inputs and outputs together occupy approximately \(32N\) bytes; explicit two-by-two Jacobians add \(32N\) bytes. Temporary gate/singular-value arrays and NumPy/LAPACK workspace are additional. For \(N=10^6\), these base arrays total about 61 MiB before temporaries. This is an estimate, not a measured peak.

For a direct product with dimension \(n=2k\), store \(k\) blocks and evaluate in chunks: parameter and certificate storage is \(O(k)\). A full dense float64 Jacobian costs \(8n^2\) bytes by itself: 8 MiB at \(n=1024\), 128 MiB at \(n=4096\). Avoid materializing it when a block minimum suffices.

For the optional butterfly track, an \(n\)-coordinate dyadic butterfly stores \(2n\log_2 n\) scalar parameters, and \(U,V\) add \(2nr\). Slopes/biases add \(2n\). Training would additionally require gradients, optimizer states, activations, and allocator overhead; these are not part of the core and must be calibrated before any training extension. A low parameter count alone is not a measured memory or speed improvement.

Certificate storage per instance is a short record of parameters, coefficients, isolated roots, endpoint enclosures, precision, bounds and timings. Measure its actual byte size on pilot cases; project total disk consumption from the frozen case count, plus two checkpoint generations and environment files. Stream JSONL records and keep large exploratory point clouds optional. No multi-gigabyte dataset download is justified by the primary task.

### 6.4 Calibration and resource decisions

WP0 measures import/idle RSS. WP4 profiles one ordinary certificate, one near the repair threshold, one near a repeated/equality root, and one saturated-gate case, at two arithmetic precision levels. These cases probe different costs rather than forming an arbitrary performance quota. Record wall time, CPU time, peak RSS, coefficient bit lengths, root refinements, and certificate status.

Derive a memory allowance from measured available memory after normal OS/application use, retaining enough headroom to avoid swap. A conservative initial operating policy is at most half of currently available memory for the experiment, adjusted downward if pressure appears; this is a safety envelope, not an assumption about total installed RAM. Select worker count from measured per-worker peak memory and physical CPU count. The serial core is the faithful fallback.

Predict a sweep with measured per-case timing quantiles and the actual frozen case count. Report the range and its uncertainty; no wall-clock duration is promised in this plan. Precision and root separation may make boundary cases much slower than ordinary cases. Increase work only to resolve a named uncertainty. If a numerical equality will not resolve, return a valid bracket rather than treating longer runtime as proof of eventual success.

For long runs, checkpoint after every completed case and on interruption. On memory pressure reduce batch size/workers before changing precision or the method. If the required rigorous package cannot run locally, CPU float64 exploration may continue, but label every affected certification result `blocked_backend` or `unresolved`; it cannot count as confirmed.

### 6.5 Timing protocol if performance is claimed

The primary result is mathematical, so benchmarking is optional unless H5 is pursued. For any claimed computational improvement, include input preparation, coefficient construction, candidate search, all failed attempts, certification, and output serialization when these are required to obtain the result. Report search and verification subtotals as well as the total.

Separate cold startup from warm repeated work. Alternate/randomize method order, hold CPU thread settings fixed, use `time.perf_counter_ns()`, record power/thermal context, and repeat only until the timing uncertainty can resolve the declared comparison. Do not treat repeated timings as independent scientific examples. Include timeouts and unresolved cases in completion-rate and total-cost summaries. If an optional MPS path is used, synchronize before and after the measured region and disclose CPU transfers and fallback operations. No speedup claim may be based solely on operation counts or a CUDA paper's reported numbers.

## 7. Ordered work packages and decision gates

### 7.1 Proposed repository and command contract

The following layout and commands are **to be implemented**, not preexisting tools. Keep the project small enough that the mathematical logic is visible.

```text
gated-repair/
  README.md
  pyproject.toml
  requirements.lock
  environment.json
  claims.json
  protocol/
    exploratory.json
    confirmatory.json
    amendments.jsonl
  src/gated_repair/
    __init__.py
    exact.py
    map.py
    baselines.py
    spectra.py
    certify.py
    replay.py
    fixtures.py
    rank1_search.py
    resources.py
    cli.py
  tests/
    test_classification.py
    test_frontiers.py
    test_numerics.py
    test_certificates.py
    test_resume.py
  proofs/
    source_audit.md
    repair_classification.md
    frontiers.md
    box_optimum.md
  artifacts/
    pilot/
    confirmation/
    certificates/
    failed/
  reports/
    decisions.jsonl
    progress.md
    results.md
    results.json
```

WP0 creates the minimal `pyproject.toml`, package, and discovery command after dependency imports succeed. Because this layout uses `src/`, install that project into the environment with `.venv/bin/python -m pip install -e .` after those files exist. Then use a module entry point, for example `.venv/bin/python -m gated_repair.cli doctor --out environment.json`. WP2 creates and documents the remaining CLI. Every command accepts a JSON config file and output directory, records the resolved config/hash, and refuses accidental overwrite of a completed run. Entry points planned below are `doctor`, `fixtures`, `audit`, `calibrate`, `certify`, `replay`, `rank1-search`, `confirm`, and `report`. They must be implemented before anyone is instructed to run them as if they exist.

### WP0 — Target prerequisites and exact arithmetic gate

**Objective/risk:** establish a faithful local execution path before claiming M3 feasibility. Addresses backend, precision, storage and license risks.  
**Inputs/dependencies:** this plan; target laptop access; no earlier implementation.  
**Actions:** run Section 6.2 discovery; select native Python; install the core packages in isolation; run the documented FLINT installation tests; inspect installed license notices; save exact package versions and architecture. Test an Arb expression with a known exact result and one with a nonzero radius, directed endpoint serialization/reload, and a rational polynomial with known simple/repeated roots.  
**Files/interfaces:** `environment.json`, provisional lock, `reports/progress.md`; implement minimal `doctor` after basic imports succeed.  
**Outputs/units:** RAM/disk in bytes, architecture, OS/build, Python/package versions, CPU/BLAS configuration, test outcomes, timestamp.  
**Resource rule:** one process; inspect idle/peak memory; no dataset/model download.  
**Acceptance:** native arm64 environment and successful exact/ball/root-isolation smoke tests; a reproducible dependency lock.  
**Failure/next action:** change interpreter within documented wheel support or use documented source-build prerequisites if appropriate; preserve the failed environment report. If rigorous arithmetic remains unavailable, mark dependent certificate work blocked while continuing proof review. Do not label float64 exploration certified.

### WP1 — Independent proof and source audit

**Objective/risk:** invalidate the mathematics or novelty claim before infrastructure grows. Addresses H0–H3.  
**Dependencies:** none for proof reading; WP0 for optional exact algebra checks.  
**Actions:** rewrite the necessity proof of T1 using both (i) opposite determinant signs and (ii) scalar fibers at fixed second output; check all degenerate cases. Verify rank-one factorization, norm, both tail bounds, orientation restriction, and the pointwise proof of T3. Reconstruct the source's main proof and sharp scalar endpoints from Appendix A. Inspect the closest sources in Section 12 and record whether each contains an antecedent, a related construction, or only background.  
**Files:** `proofs/*.md`, `claims.json`, `reports/decisions.jsonl`, source ledger with exact URLs/access dates.  
**Expected outputs:** complete logical arguments with assumptions and counterexample attempts, including \(p=0,-c\), \(d\ne0\), \(q=0\), \(c=0\), shifted domains and metric changes.  
**Resources:** small symbolic calculations; no high-dimensional optimization.  
**Acceptance:** no unresolved proof dependency in T1–T3; narrower claims explicitly marked; novelty status accurately attributed.  
**Branches:** a counterexample pauses the affected theorem and dependent experiments; a missing proof step is a blocking gate, not an empirical hypothesis to “verify” randomly. Prior art may change attribution while leaving replication work valid. The PDF's assertion of prior independent checking cannot satisfy this gate.

### WP2 — Minimal faithful map and evaluation harness

**Objective/risk:** implement exactly the mechanism and catch convention or numerical mistakes. Addresses H1–H3 implementation.  
**Dependencies:** WP0; theorem statements fixed by WP1.  
**Actions:** implement Section 4.7, exact admissibility/rank classification, stable sigmoid derivative, stable singular-value formula, all analytic baselines, and equivalent coupling. Parse rational strings into exact types; provide explicit conversion to float64/Arb. Implement the proposed CLI and test harness.  
**Files:** `exact.py`, `map.py`, `baselines.py`, `spectra.py`, first three test modules.  
**Outputs:** per-fixture forward/Jacobian/inverse/rank/norm results; known collision and opposite-orientation witnesses.  
**Tests:** compare analytic Jacobians to an independently coded finite-difference path away from singular/tail limits; compare singular values to SVD; verify exact algebraic identities and coupling equivalence; assert B0 fails injectivity and valid branches invert. Test that constant-gate controls dispatch to linear algebra.  
**Resources:** scalar/batched two-dimensional inputs only; calibration determines batch size.  
**Acceptance:** discrepancies fit documented floating-point conditioning/error budgets; no exact logical decision depends on a numeric tolerance; invalid cases fail for the expected reason.  
**Failure/next action:** debug the harness before any comparison. A mismatched formula or a hidden float32 cast is an implementation failure, not evidence against the mechanism.

### WP3 — Rigorous margin certificates and replay

**Objective/risk:** convert continuous-domain conditioning claims into independently checkable numerical bounds. Addresses H3/H4 measurement validity.  
**Dependencies:** WP0–WP2; rational fixtures or explicit algebraic/enclosure parameter policy.  
**Actions:** implement (13) using exact rational coefficients, derivative root isolation, Arb gate endpoints, candidate classification and threshold bracketing. Add the strict \(\tau<|q|\) guard. Implement directed serialization and a separate replay checker that recomputes rather than trusting stored verdicts. Include exact common-budget generator (14).  
**Files:** `certify.py`, `replay.py`, `test_certificates.py`, certificate JSON schema.  
**Expected outputs:** \([\mu_L,\mu_U]\), rational threshold decisions, polynomial coefficients, isolated root intervals, gate endpoint balls, precision, and statuses `certified`, `unresolved`, or `invalid`.  
**Tests:** known constant/positive/negative quartics, repeated roots, boundary \(\tau=|q|\), endpoint-scale maps, intentionally corrupted certificates including deletion of a stationary root, and coefficient/domain conversion errors. The checker must reject the corrupted records.  
**Resources:** at most three stationary roots per threshold; serialize work by default; escalate precision based on unresolved signs.  
**Acceptance:** every claimed bound replays from exact inputs; negative witnesses lie in the actual gate interval; unresolved equality is preserved.  
**Failure/next action:** fix interval construction/root filtering; if a package path fails, use rational Bernstein subdivision with outward gate bounds as a documented fallback. Do not accept floating optimization as a certificate.

### WP4 — Pilot, conditioning sensitivity and resource calibration

**Objective/risk:** determine precision/runtime requirements and expose wrong proxy choices. Addresses H4/H5 and resource uncertainty.  
**Dependencies:** WP2–WP3.  
**Actions:** run the regime-based pilot in Sections 5.3 and 6.4. Compare exact tail values, certified box values, conservative secant bounds, forward envelopes and round trips. Reproduce a balanced/asymmetric rank-one lead only as exploration. Vary transition placement and box size. Profile entire certificate paths, including failures.  
**Files:** `protocol/exploratory.json`, `artifacts/pilot/*.jsonl`, `resources.py`, `reports/decisions.jsonl`.  
**Expected outputs:** resource estimates with uncertainty, numerical-resolution policy, lists of sensitive branches, and a decision about whether H4 or H5 is worth additional work.  
**Acceptance:** known analytic frontiers reproduced, realistic per-case budgets measured, and no unexplained disagreement between exact/float/ball paths.  
**Branches:** a tiny H4 difference triggers more precise certification rather than immediate scaling; absent or inconsistent advantage is recorded. If only the analytic results survive, proceed to confirm and report them without adding training. If resources fail, reduce concurrency/chunking first and report any unresolved cases.

### WP5 — Freeze the confirmation contract

**Objective/risk:** prevent adaptive case selection and baseline weakening.  
**Dependencies:** WP4; all implementation bugs resolved.  
**Actions:** freeze H0–H3 statements, any selected H4/H5 claim, exact fixture/generator specs, metric definitions, orientation policy, domain/coordinate units, methods, tolerance, analysis, and resource accounting. Save code revision, lock hash, protocol hash, and protected case manifest. Do not change a primary metric after viewing confirmation.  
**Files:** `protocol/confirmatory.json`, `claims.json`, `artifacts/confirmation/manifest.json`.  
**Outputs:** a standalone configuration whose cases and decisions are reproducible.  
**Acceptance:** a fresh process can reconstruct all cases and identify each primary/secondary/exploratory output.  
**Branches:** an unresolved central design choice blocks freezing. Later methodological changes create an amendment and fresh confirmation instances where selection could matter; technical reruns preserve failed originals.

### WP6 — Confirm exact frontiers and bounded-domain consequences

**Objective/risk:** deliver the core reproducible result.  
**Dependencies:** WP5.  
**Actions:** execute the frozen suite with B0–B4 and the selected diagnostic controls; produce certificates for every positive conditioning claim; verify exact rank/norm matching; compare all primary quantities. Include source swap, best rank-one, orientation-preserving rank-one, and optimal unrestricted repairs. Report that the coupling-equivalence control has identical mathematical behavior.  
**Files:** `artifacts/confirmation`, `artifacts/certificates`, `reports/results.json`.  
**Outputs:** per-case records plus a frontier table; no dropped failures.  
**Resource rule:** use WP4 measured budgets; checkpoint per case; run serially if needed.  
**Acceptance:** all proof claims match exact computations, certificates replay, and no baseline violates its own feasible constraints.  
**Branches:** any observed violation first triggers precision/configuration audit; a valid exact counterexample reopens the theorem. Unresolved certificates are inconclusive values, not failed injectivity. A rank-two advantage is expected from the proven optimum and must be retained.

### WP7 — Attribution, adversarial cases and clean-process reproduction

**Objective/risk:** eliminate positive-looking artifacts and ensure results survive interruption and independent computation.  
**Dependencies:** WP6.  
**Actions:** execute Section 8's core controls and only those optional controls whose work-package entry gates have passed; replay all certificates from a clean process with locked dependencies; regenerate a subset through an independently coded formula path; interrupt and resume a run, then compare manifests and outputs with uninterrupted execution. Recheck exact constraint preservation after serialization.  
**Files:** `test_resume.py`, replay logs, `reports/audit.md`.  
**Expected outputs:** explicit pass/fail and discrepancy records for every material adversarial scenario.  
**Acceptance:** no methodological correction remains unpropagated; successful records retain exact provenance; failed records remain visible.  
**Branches:** a discovered defect revises its code, protocol and dependent outputs, with a new run ID. Stop adding optional tests once concrete residual risks are resolved.

### WP8 — Synthesis, attribution and continuation decision

**Objective:** produce an auditable conclusion without overstating scope.  
**Dependencies:** WP1–WP7 outcomes, including failures.  
**Actions:** generate a report separating theorem/proof status, novelty status, certified values, exploratory observations, target hardware measurements and unresolved cases. Include strongest comparisons, orientation/rank costs, numerical errors, search coverage and methodological amendments. Apply Section 10's outcome rules.  
**Files:** `reports/results.md`, `reports/results.json`, `reports/progress.md`, final decision entry.  
**Acceptance:** a reader can reproduce the computation from the repository and understand why each claim is supported, narrowed, or rejected. No unperformed training or hardware benchmark is implied.  
**Branches:** deliver a narrow mathematical result if that is all the evidence supports; retain a useful negative/replication result if novelty or benefit disappears; authorize an extension only when its specific entry condition is met.

### Optional WP9 — Certified rank-one finite-box optimization

**Entry condition:** H4 survives pilot certification and a fixed rank-one correction budget has a meaningful use case or a mathematically distinct conjecture. This work is not necessary to establish T1–T3.

For \(c>0\) on a centered \(y\) box **when orientation reversal is permitted**, it suffices to search \(p,q\ge0\): reflecting a negative \(p\) to its magnitude increases the useful scale, and choosing \(q\) with the sign of \(p\) minimizes \(|1-pq|\) at fixed magnitudes. The norm constraint is \((1+p^2)(1+q^2)\le r^2\). Under an orientation-preserving policy instead use \(p\ge0,v=-q\ge0\), \(A=1+pv\), with \((1+p^2)(1+v^2)\le r^2\); do not apply the orientation-unconstrained reduction. Include boundary points for upper bounds; \(q=0\) yields zero margin and is not admissible. Do not assume the budget is exhausted or that \(p=q\) is finite-box optimal.

Implement a deterministic branch-and-bound over the compact \((p,q)\) rectangle \([0,\sqrt{r^2-1}]^2\), discarding regions proved infeasible. An incumbent rational feasible point has a certified **lower bound** on its worst-case margin from WP3. For a parameter rectangle, select one or more fixed rational gate values known to lie in the actual gate interval; a validated upper bound on \(\phi\) there is an **upper bound** on the best possible minimum over that rectangle. Tighten with interval arithmetic and subdivision; do not reverse these bounds. The unrestricted optimum from T3 supplies a global initial ceiling. Retain a verified best feasible lower bound and remaining global upper bound.

For the orientation-preserving policy replace the second parameter by \(v=-q\) in that rectangle. If the global upper gap stalls, enrich the common gate-witness set using rational interior points proposed by the one-dimensional minimization at promising parameter centers, verifying membership before use. Refine gate coverage as well as parameter boxes: a permanently fixed finite gate set can otherwise leave an irreducible sampled-objective gap even as parameter boxes shrink.

Stop when the optimum bracket meets the frozen tolerance or when its resource gate is reached. The latter yields a certified candidate and an optimality gap, not a global optimum. Share all generic numerical improvements with baseline searches. Compare a claimed candidate gain to the balanced/source references by disjoint verified value intervals at the same norm cap. A candidate cannot beat T3 under its assumptions; any apparent excess is a critical defect.

### Optional WP10 — High-dimensional or butterfly continuation

**Entry condition:** the narrow result is correct, its novelty/utility disposition is known, and a specific unsolved question justifies expansion.

First use direct sums of independent two-dimensional gates. Under **block-diagonal repairs**, correction ranks add, spectral correction norm is the maximum block norm, and the block-Jacobian minimum singular value is the minimum block value. This is an exact mechanism/scaling check and requires no dense Jacobian. It says nothing about optimal unrestricted cross-block repairs. Investigate those first at dimension four with symbolic conditions/counterexample search; treat any rank lower bound as a conjecture until proved.

The separate butterfly track is specified in Appendix B. Begin with the exact small fixed-topology theorem and the size-16 degeneration. Add factor-norm bounds and conditioning diagnostics before optimizing approximate representations. A freely available permutation, extra butterfly depth, or a different topology changes the feasible class and must be a separate baseline. No large-scale training starts until the target task, known bottleneck, baselines, data license/splits, and local resource budget are independently specified.

## 8. Ablation, robustness, and failure-analysis program

The following are scientific controls; each resolves a specific interpretation risk. They are not an invitation to run an indiscriminate hyperparameter grid. Rows marked optional apply only when their stated extension has been selected; they cannot block completion of the core PoC.

| Challenge | Required action | Expected interpretation and failure response |
|---|---|---|
| Frozen-gate safety | Compare actual Jacobian with frozen matrices on the exact shear | Frozen matrices may all be nonsingular while the actual map folds; a harness that misses this is invalid |
| Exact repair constraint | Set \(d=0\), then \(d=\pm2^{-j}\) for increasing \(j\), using exact rational parameters | Global failure persists for every nonzero perturbation; witnesses can move far away; finite-box success does not imply global repair |
| Endpoint scale | Test \(p=0\), \(p=-c\), and points on both sides | Endpoints are globally invertible but tail margin is zero; a tolerance-based endpoint classifier is unacceptable |
| Constant-gate limit | Compare exact \(a=0\) with small nonzero \(a\), and exact \(c=0\) with small nonzero \(c\) | The whole-space theorem is nonuniform in these limits; do not reinterpret it as a finite-domain discontinuity |
| Rank and orientation | Compare B2, B2+, B3 at matched cost, reporting determinant signs | Orientation reversal or extra rank may explain a gain; neither may be hidden |
| Norm objective gaming | Repeat only with preregistered coordinate scalings and report transformed budgets | Results in rescaled units are a different metric; post-hoc rescaling cannot count as improvement |
| Gate placement | Center the transition, move it outside the operating interval, change slope sign | Confirms whether a box result is driven by transition sensitivity or a saturated almost-linear regime |
| Domain symmetry | Test a shifted \(y\) box separately | T3 is scoped to centered boxes; any shifted-domain gain does not refute it and does not inherit its optimum |
| Local/global confusion | Compare \(\mu_D\), bound (15), random secants, and exact inversion | Random secants are diagnostics; \(\mu_D\) alone is not a global co-Lipschitz certificate |
| Unbounded domain | Increase \(Y\) with \(x\) in transition | \(\mu_D\) tends toward zero even for a valid repair; no global numerical stability claim survives |
| Numerical precision | Compare float32, float64 and Arb on near-threshold/tail cases | Low precision may erase a positive margin or a remote witness; final labels use exact/validated paths |
| Equivalent functions | Compare direct map and equivalent coupling under equal batching/precision | A mathematical advantage between them indicates an error; a speed difference is implementation-specific |
| Optimizer strength | Reproduce analytic tail optima in the core; compare H4 searches under equal budgets only if H4/WP9 is selected | Failure to reach a known optimum is a baseline weakness, not a new method advantage |
| Confirmation selection | Retain all frozen cases, failures, timeouts and changed hypotheses | Favorable subset reporting or adapting the primary claim requires new confirmation |
| Restart integrity | Interrupt between certificate construction and commit, then resume | Atomic records and hash checks must prevent duplicate or incompatible results |
| High-dimensional extrapolation — optional WP10 | Only if this WP10 branch is selected, compare block-diagonal direct sums with unrestricted cross-block repairs as distinct classes | Per-block rank requirements do not imply a global unrestricted rank lower bound |
| Butterfly cancellation — optional WP10 | Only if this WP10 branch is selected, track factor/operator norms, residual norm, and precision as \(t\) increases | Approximation by diverging parameters is different from a stable low-rank representation |

For numerical witnesses, save the actual inputs, outputs, Jacobians, and exact generating formulas. A pair with small floating residual is only a candidate collision. If a constructive collision certificate is needed, use an exact identity or interval root existence in disjoint input boxes with the same certified target output. Do not impose a minimum separation threshold chosen after seeing results.

For performance or trained extensions, account for preprocessing, search, tuning, certification, inversion, storage and material deployment costs. Apply useful generic optimizations to all methods. A speedup confined to an isolated arithmetic kernel is not an end-to-end gain without the corresponding accounting.

## 9. Reproducibility, logging, and reporting requirements

### 9.1 Configuration conventions

Use UTF-8 JSON configurations parsed with the standard-library `json` module; no additional configuration parser is required. Store all scientific parameters as rational strings or an explicitly tagged algebraic representation. Separate them from resource controls. The following is an illustrative **proposed project configuration**, not an already executable package schema:

```json
{
  "schema_version": 1,
  "phase": "confirmation",
  "claim_ids": ["H1", "H2", "H3"],
  "map_family": "linear_repair_sigmoid_shear",
  "coordinate_metric": "fixed_euclidean",
  "gate": {"a": "1", "b": "0", "c": "1"},
  "domain": {"x_lo": "-4", "x_hi": "4", "y_center": "0", "y_radius": "4"},
  "budget": {"generator": "matched_rational", "u": "1"},
  "methods": ["rank1_balanced", "rank1_orientation_preserving", "rank2_rotation", "equivalent_coupling"],
  "certificate": {
    "relative_width": "1/1073741824",
    "absolute_width": "0",
    "initial_precision_bits": 128,
    "precision_policy": "double_on_unresolved",
    "parameter_policy": "exact_rational"
  },
  "resources": {"workers": 1, "case_budget_source": "artifacts/pilot/calibration.json"}
}
```

Initial 128-bit precision is an implementation starting point, not a proof guarantee or a cap. Refine based on enclosure widths and exact root separation. If precision/resource policy changes after confirmation, record whether it changes only computation or also the estimand and feasible method comparison.

### 9.2 Shared output schema

Every run record must contain these fields, with explicit units and nulls rather than fabricated defaults:

```json
{
  "schema_version": 1,
  "run_id": "generated-unique-id",
  "case_id": "hash-of-canonical-exact-instance",
  "phase": "exploratory|confirmation|replay",
  "claim_ids": ["H3"],
  "method_id": "rank2_rotation",
  "source_revision": "actual-commit-hash",
  "protocol_sha256": "actual-hash",
  "environment_sha256": "actual-hash",
  "dependency_lock_sha256": "actual-hash",
  "exact_parameters": {"A":"0","p":"3/4","q":"-3/4","d":"0","a":"1","b":"0","c":"1"},
  "domain": {"x_lo":"-4","x_hi":"4","y_lo":"-4","y_hi":"4"},
  "metric": "minimum_local_jacobian_singular_value",
  "rank_exact": 2,
  "correction_norm": {"kind":"exact_rational","value":"5/4"},
  "orientation": "positive",
  "classification": "globally_bijective",
  "mu_interval": {"lower_dyadic": null,"upper_dyadic": null},
  "certificate_status": "pending",
  "certificate_sha256": null,
  "precision_bits": null,
  "wall_seconds": null,
  "cpu_seconds": null,
  "peak_rss_bytes": null,
  "failure_kind": null,
  "started_at_utc": null,
  "finished_at_utc": null
}
```

The shown parameter example is a real exact fixture; placeholder hashes/IDs/times must be populated by the implementation, never reported as actual provenance. A certificate includes its rational polynomial coefficients, tested thresholds, square-free derivative/root enclosures, endpoint ball bounds, gate-domain membership decisions, sign enclosures, and replay version. Save search candidates and rejected cases separately from verified outcomes.

For float checks record dtype, array shape, input-generation seed, realized input file/hash when needed, stable-function implementation, absolute/relative error definitions, and whether error is forward or backward. For timing record cold/warm status, thread count, process affinity if used, synchronization boundaries, sample order, and all failures.

### 9.3 Resume protocol

1. Write a complete result/certificate to a temporary sibling path, flush, and atomically rename it after validation. Append its hash to the run manifest only after the rename.
2. On resume, verify schema, code revision, lock, exact input, protocol, and precision policy. A mismatch creates a new run or an explicit migration/amendment; do not append incompatible results to the old run.
3. Reuse completed records only after hash/checker validation. Keep failed and interrupted attempts with statuses rather than deleting them.
4. Store interval-search queue, incumbent bounds, candidate parameter boxes, root-refinement state where serializable, and RNG state for optional stochastic work. Recomputing a small certificate from exact inputs is acceptable and often safer than restoring library-internal objects.
5. Never treat a pending or partially written record as a completed proof. A crash during serialization must be detectable by the manifest/checksum.

### 9.4 Progress, amendments and final reporting

`reports/progress.md` records completed work packages, current blockers, next action and exact commands needed to resume. `reports/decisions.jsonl` records date, evidence, decision and affected claim IDs. The coding agent may make routine nondestructive choices about packaging, data structures and chunk sizes autonomously. Changes to the hypothesis, population, primary metric, baseline fairness, coordinate metric or interpretation require a recorded methodological amendment and appropriate fresh confirmation.

The final execution report must contain:

- A claim table separating proved, independently replayed, exploratory, contradicted, inconclusive and blocked results.
- Exact problem assumptions and the prior-art/attribution disposition.
- The source swap, optimal rank-one, orientation-preserving rank-one and optimal unrestricted frontiers, with rank and cost visible.
- Certified finite-box values with intervals; no replacement by point estimates in headline claims.
- Round-trip errors and conditioning guardrails, including global instability and boundary cases.
- Target-laptop environment and **measured** time/memory only where actually obtained.
- All failed/unresolved confirmation cases and reasons; exploratory findings visibly separated.
- Provenance, protocol changes, source limitations, and a continuation/stop decision.

No report should describe the number of random tests as proof, zero observed collisions as injectivity, a favorable optimizer trajectory as a global optimum, or an algebraic identity as an end-to-end training improvement.

## 10. Outcome interpretation and evidence-triggered alternatives

| Observed outcome | Correct conclusion | Authorized next action |
|---|---|---|
| T1–T3 hold; exact antecedent found | Correct replication/specialization; priority claim withdrawn | Attribute the antecedent; deliver implementation/certificates if useful; pursue only a concrete remaining question |
| Proof gap or exact counterexample | Affected theorem unsupported/false | Preserve counterexample, revise proof/claim, rerun only dependent work; no scaling as a substitute |
| Proofs hold and no exact antecedent found | Candidate narrow mathematical contribution with unresolved historical novelty | Prepare a concise proof note and reproducibility package; seek qualified external review before claiming a research breakthrough |
| Rank-one uses less correction norm but loses conditioning | Quantified structural trade-off | Choose only if the application accepts the loss; do not call it overall superiority |
| Rank-two rotation dominates local conditioning | Expected strongest baseline result, not a project failure | Report the exact envelope; consider whether restricting rank is justified |
| Asymmetric rank-one gain vanishes after certification or better search | No confirmed H4 benefit | Keep the negative result; stop tuning that claim unless a new evidence-based hypothesis is specified |
| Asymmetric gain is certified on some boxes | Narrow component improvement for those boxes | Confirm on preregistered domains; report its optimum gap and orientation/rank; do not claim universal advantage |
| Candidate beats T3 on a centered box | Mathematical or implementation inconsistency | Audit constraints, exact norm, domain, metrics and rounding immediately; a valid exact counterexample supersedes the theorem |
| Only shifted-domain gains exist | Different optimization problem | Amend scope and derive the shifted-domain theorem; do not use T3 as a claimed baseline guarantee there |
| GPU path unsupported or memory low | Hardware limitation of an optional route | Use serial CPU float64/Arb; core remains scientifically faithful |
| Rigorous backend unavailable | Numerical certification blocked | Deliver proofs plus clearly provisional floats; record exact missing prerequisite; no certified-value claim |
| Synthetic mechanism works but a later real task does not improve | Mathematical component result lacks demonstrated task value | Retain the theorem; stop or revise the application rather than hiding negative transfer |
| High-dimensional cross-block repair violates a blockwise conjecture | Block restriction mattered | Develop the counterexample/classification; withdraw extrapolation |

### 10.1 Preferred practical alternative when global stable inversion is required

If an application requires a globally bounded inverse sensitivity on all of \(\mathbb R^n\), preserving a nonconstant multiplicative scale of an unbounded coordinate is incompatible with that requirement in this family. The appropriate evidence-triggered pivot is to additive coupling, a genuinely contractive residual, or a certified bi-Lipschitz architecture such as BiLipNet, with a newly specified task. This changes the mechanism and must be recorded as such. [R2,R4,R6,R12]

### 10.2 Most promising mathematical extensions

The following are ordered by evidence and scope, not an undifferentiated menu:

1. **Shifted/asymmetric operating domains:** only if real input ranges motivate them. Derive the corrected \(H(s)\); test whether a nonzero linear diagonal improves the optimum. This preserves the gate mechanism and directly challenges a stated T3 assumption.
2. **Cross-block repair rank at dimension four:** only after direct-sum controls pass. Search for counterexamples to naive additive-rank intuition before formulating a general lower bound. Exact symbolic conditions are preferable to broad neural training.
3. **Partially constant gates with an invertible constant-output block:** a clean supporting theorem is given in Appendix A. It may be a routine corollary; use it to guide a harder singular-block classification, not to create a new name.
4. **Norm-bounded butterfly approximation closure:** only if the independent algebraic question remains interesting after comparing [R14,R15,R18]. Track factor norms, operator norms and arithmetic precision; the source already shows why unconstrained optimization can exploit blow-up.

A mechanism needing scale unavailable on the laptop is not rejected merely because a toy task lacks that scale. State the locally testable necessary prediction, test it, and distinguish a failure of that prediction from an underpowered proxy. Conversely, a conjectured future scale advantage cannot excuse a violation of an exact small-dimensional theorem.

## 11. Audit record, limitations, and remaining risks

Three distinct review passes were conducted during this investigation using separate research agents and direct checks: mathematical/prior-art challenge, experimental-validity challenge, and implementation/feasibility challenge. They are AI-assisted internal reviews, **not external peer review**. The source document's claim of earlier independent reviews was not treated as evidence that these checks had already been done.

| Material weakness found | Revision incorporated | Residual risk |
|---|---|---|
| A plausible “new invertible gate” is ordinary affine coupling | Reframed contribution as exhaustive repair classification and fixed-coordinate optimality; added exact equivalence control | The narrow theorem may still be a known or routine corollary |
| Source's norm-2 repair is a weak standalone comparator | Derived the full rank-one and unrestricted tail frontiers; added rank-two scaled rotation | Rank cost has limited practical meaning at dimension two |
| Smaller correction can conceal severe inverse instability | Made conditioning primary alongside rank/norm; exposed near-infimum collapse and global instability | Application significance requires actual domain/error requirements |
| Local Jacobian margin could be mislabeled a global inverse-Lipschitz guarantee | Separated \(\mu_D\), \(\gamma\), \(\ell_D\); supplied independent secant bound | Secant bound is conservative |
| Tail-optimal balanced rank-one family need not be box-optimal | Added optional certified full rank-one search and exact unrestricted box optimum | Full rank-one box optimum may be costly near equality |
| Orientation can change during repair | Added orientation-preserving frontier and determinant-sign reporting | Application may forbid orientation reversal, requiring the constrained comparison |
| A proof could substitute the norm cap for the actual correction norm | Used the actual norm first and proved monotonicity in the cap | Numerical implementations must report actual norms and exact feasibility |
| Symmetric-domain proof could be overextended | Stated centered-\(y\) assumption; shifted domains are a separate problem | General domain optimum remains open |
| Irrational optimum rounded into a “rational” certificate | Added exact common-budget rational generator and explicit algebraic/enclosure policy | Implementer must preserve exact types through serialization |
| Quartic nonnegativity alone fails at a zero Schur pivot | Required \(\tau<\lvert q\rvert\), with explicit boundary handling | Equality may remain unresolved numerically |
| A certificate could omit a stationary minimum | Required independent full-root enumeration or root counting; added deleted-root corruption test and complete-cover check for subdivision | Replay must verify completeness as well as listed signs |
| Naïve singular-value subtraction loses small positive margins | Used stable product/hypot formula and Arb certification | Extremely ill-conditioned inversion still loses practical precision |
| MPS nominal support could imply nonexistent float64 capability | Chose CPU reference and official capability gates | Target installation and runtime are unmeasured |
| Exact rank obstruction could be misused as approximation gap | Preserved norm-bound and nonclosedness distinctions in Appendix B | General low-rank closure questions remain open |
| Appendix derivative and norm bounds lacked explicit hypotheses | Required gate continuity, nonzero target scale and \(r<n\) for the denominator bound | These statements remain topology-specific |
| Optional extensions accidentally became core prerequisites | Made WP7 controls conditional on WP9/WP10 entry gates | A later claim expansion requires an amended protocol |
| Configuration format required an unlisted parser | Standardized all configuration files on standard-library JSON | Schema validation and exact rational parsing still need implementation |
| Reproducibility could depend on chat or transient scratch notes | Restated mathematics, interfaces, fixtures, logging, provenance and commands here | Future code and certificates must still be produced and reviewed |

The final targeted rechecks of the changed proofs, certificate logic, configuration/install path, and conditional work-package gates found no further substantive correction within this plan's scope. The coding agent must repeat targeted audits after implementation. Such a stopping decision is not a claim of absolute perfection. Report unresolved proof, source-access, package and resource issues explicitly.

**Remaining limits:** no external human proof review; no exhaustive literature/patent search; no formal proof-assistant verification; no rigorous interval execution yet; no M3 runtime or memory measurement yet; no learned high-dimensional model, real-world task result, or production performance gain. These are limitations of the present evidence, not unreported successful work.

## Appendix A. Standalone reproduction of the source's central mathematics

### A.1 Identity-skip theorem and scalar endpoints

For (2) with \(R=0\) and every \(a_i\ne0\), define

\[
h_b(t)=\sigma(t)+(t-b)\sigma'(t),\quad
m_b=\min_t h_b(t)<0,\quad M_b=\max_t h_b(t)>1.
\]

The source theorem states equivalence of injectivity, being a homeomorphism onto \(\mathbb R^n\), nonnegative Jacobian determinant everywhere, and

\[
B\text{ diagonal},\qquad -1/M_{b_i}\le B_{ii}\le-1/m_{b_i}.
\tag{A1}
\]

Strict coefficient inequalities give a smooth diffeomorphism; endpoints remain homeomorphisms with singular Jacobian points. [S0]

Here is a complete structural proof outline sufficient to reconstruct the argument:

1. Let \(T=\operatorname{diag}(a)\), \(z=Tx+b\), \(C=TBT^{-1}\). Affine conjugation gives \(G(z)=z+D_{\sigma(z)}C(z-b)\), with Jacobian \(I+D_\sigma C+\operatorname{diag}(\sigma'(z)\odot C(z-b))\). Along \(z=-t\mathbf1\), it tends to \(I\). Injectivity therefore forces nonnegative determinant everywhere by consistent orientation.
2. At every saturation corner, with coordinates \(\pm t\), the derivative diagonal tends to zero. For each subset \(S\), the determinant limit is the principal minor \(\det(I+C)_{SS}\ge0\). Thus \(I+C\) is a P0-matrix.
3. If \(C_{ij}>0\), keep \(z_i=0\) and set other coordinates \(-tv_k\), with positive \(v_k\) and \(v_j\) large enough that \(\sum_{k\ne i}C_{ik}v_k>0\). The other rows tend to identity rows; the \(i,i\) entry has a negative linear term. The determinant becomes negative. Hence every off-diagonal \(C_{ij}\le0\).
4. If \(C_{ij}<0\), keep \(z_i=0\), set \(z_j=t\) and the remaining coordinates \(-\delta t\) for sufficiently small positive \(\delta\). The growing diagonal term is negative, and its limiting cofactor is \(1+C_{jj}\ge0\). Nonnegativity forces \(1+C_{jj}=0\).
5. Draw an edge \(j\to i\) for \(C_{ij}<0\). Every vertex with an outgoing edge has zero diagonal in \(I+C\). A shortest directed cycle has no chord that could create a shorter cycle; its principal determinant is negative, contradicting P0. If the graph is acyclic with an edge, it has a source with an outgoing edge. Its row has no off-diagonal entries and \(C_{jj}=-1\). Choose its coordinate so that \(h_{b_j}>1\), send all others off, and the determinant tends to \(1-h_{b_j}<0\). Therefore there are no edges, proving diagonality.
6. For a scalar diagonal component, \(f'(x)=1+\beta h_b(ax+b)\). Nonnegative derivative is equivalent to (A1). The derivative of \(h_b\) is \(\sigma'(t)[2-(t-b)\tanh(t/2)]\). There are exactly two stationary points, one below \(\min(0,b)\), one above \(\max(0,b)\); each is unique on its interval. At a stationary point \(h_b=1/2+1/(2\tanh(t/2))\), establishing the signs of its extrema. At an endpoint coefficient the derivative vanishes only at the corresponding extremum, so the scalar map stays strictly increasing. Its asymptotic slopes \(1\) and \(1+\beta\) are positive, proving onto-ness. Products finish the equivalence.

For \(b=0\), let \(q_0\tanh(q_0/2)=2\). Then \(q_0\approx2.3993572805\), \(m_0=(2-q_0)/4\), \(M_0=(2+q_0)/4\), and the interval is approximately \([-0.9092237218,10.0160938467]\). Recompute it using a bracketed root and validated arithmetic; the printed decimals are not exact thresholds. Scalar Swish/LipSwish analysis is established prior art. [R7]

### A.2 Exact collisions and bounded-region control

For \(F_c(x,y)=(x+c\sigma(x)y,y)\), let \(L=\log3\) and \(y_*=-4L/c\). Then the three distinct inputs \((-L,y_*),(0,y_*),(L,y_*)\) all map to \((-2L,y_*)\). Their determinants are \(1-3L/4,1-L,1-3L/4\), hence all three are regular. The derivative has only two zeros, so that output has exactly three preimages. Direct products yield exactly \(3^k\) preimages in dimension \(2k\), even with arbitrarily small nonzero \(|c|\). [S0]

On the centered box \([-\rho,\rho]^2\), the same shear is injective precisely for \(\rho\le4/|c|\). At and below this threshold every fixed-\(y\) scalar fiber is strictly increasing; at equality its derivative can vanish only at an isolated point. Above it, choose an interior \(y\) just beyond the threshold, producing a local fold near \(x=0\) and distinct preimages within the box. The statement also holds for the centered Euclidean ball with the same radius, by taking the fold sufficiently close to zero. This elementary calibration result is also reached by the actual-Jacobian triangular/P-matrix approach; improving over only a crude contraction bound here is not a new certification result.

The source's general sufficient finite-ball bound is

\[
\|D(F-I)\|_2\le\|R\|_2+\|B\|_2(1+a_*\rho/4),\quad
a_*=\max_i|a_i|,
\]

on \(\|x\|_2\le\rho\). A value below 1 gives a bi-Lipschitz bound on that convex ball. It can be conservative. Do not compare this ball certificate to a box result without matching domains.

### A.3 Multiplicity and linear corrections

If \(I+B\) has all positive principal minors, every frozen \(I+DB\), \(0\le D\le I\), is uniformly nonsingular: each principal determinant is a convex combination of the corresponding endpoint minors. The homotopy to the identity is uniformly proper and has degree one. An active mixing row produces a negative-orientation point by a transition-row ray; degree then forces at least two positive-orientation preimages in addition. Smoothness, properness and regular-value arguments give an open set with at least three preimages. The sufficient condition \(\|B\|_2<1\) implies the premise. With a correction, \(\|R\|_2+\|B\|_2<1\) gives the analogous degree-one obstruction. [S0]

This argument does not contradict contraction-based invertibility: it never bounds the entire multiplicative residual's Lipschitz constant on unbounded space. [R6]

### A.4 Supporting partial-gate extension

Split coordinates into active \(u\) and constant-gate \(v\), let \(D\) be the positive constant-gate diagonal, and assume \(L=I+DB_{vv}\) is nonsingular. Put \(K=DB_{vu}\). The constant output is \(w=Ku+Lv\); eliminate \(v\) to get

\[
u+D_g(u)\{Su+Tw\},\quad
S=B_{uu}-B_{uv}L^{-1}K,\quad T=B_{uv}L^{-1}.
\]

A nonzero row of \(T\) permits one active gate in transition and a freely varied \(w\) to force a negative determinant, while all other active gates are off; for that fixed \(w\), an all-off ray supplies positive orientation. Hence injectivity requires \(B_{uv}=0\). The original theorem then forces \(B_{uu}\) diagonal in its scalar intervals. These conditions, arbitrary \(B_{vu}\), and nonsingular \(L\) are also sufficient by triangular inversion. This is a supporting derivation, likely a routine corollary rather than the main novelty candidate. Singular \(L\) is outside this argument and can permit coupling-style repairs; do not discard that regime.

## Appendix B. Fixed butterfly topology: exact rank and approximation safeguards

This secondary track preserves the supplied note's separate result. It is not evidence for or against T1–T3.

Let \(m=2^k\), \(n=m^2\), index coordinates by \((a,b)\) with linear index \(am+b\), and define \(P e_{(a,b)}=e_{(b,a)}\). A fixed butterfly is \(B=M_{2k-1}\cdots M_0\), where stage \(j\) mixes pairs differing only in binary bit \(j\), with arbitrary real two-by-two blocks, including singular ones. The order is part of the model. Then

\[
\min_{B\in\mathcal B_n}\operatorname{rank}(P-B)
=\frac{m(m-1)}2=\frac{n-\sqrt n}{2}.
\tag{B1}
\]

To reproduce the proof, group the first \(k\) stages into \(R^{(a)}\) and the last \(k\) into \(L^{(d)}\). The block \((d,a)\) of \(PB\) is \(L^{(d)}_{:,a}R^{(a)}_{d,:}\), so has rank at most one. For a \(p\times p\) matrix of blocks of size \(d\), each rank at most \(q\), the filtration of \(\ker(I-A)\) by initial zero block coordinates gives

\[
\operatorname{rank}(I-A)\ge\sum_{t=1}^p(d-qt)_+.
\]

Explicitly, let \(K_i=\ker(I-A)\cap\{x_1=\cdots=x_i=0\}\). For \(x\in K_{i-1}\), the equation \(x_i=\sum_{j=i}^p A_{ij}x_j\) restricts the possible \(i\)-th block to a space of dimension at most \(\min(d,q(p-i+1))\). The kernel of that block projection is \(K_i\). Summing the resulting bounds on \(\dim K_{i-1}-\dim K_i\) proves the displayed inequality.

Use \(p=d=m,q=1\); equality in (B1) is attained by \(B=I\), since each off-diagonal swap pair contributes rank one to \(P-I\). [S0]

For \(\operatorname{rank}R\le r\le m-2\),

\[
\inf\|P-B-R\|_2=1.
\tag{B2}
\]

Each diagonal block of \(PB+PR\) has rank at most \(1+r<m\); its difference from \(I_m\) has norm at least one. The upper value one is attained by \(B=R=0\). Separately, for \(\operatorname{rank}R\le r<n\), \(K\ge0\), and \(\|B\|_2\le K\), the broader bound is

\[
\|P-B-R\|_2\ge
\frac{[n-Km-(K+1)r]_+}{n-r}.
\tag{B3}
\]

It follows by bounding \(\operatorname{Re}\operatorname{tr}(PB)\le mK\) and evaluating on an \((n-r)\)-dimensional subspace of \(\ker(PR)\). These are topology- and norm-specific statements. If the target permutation itself is a freely available primitive, it is an exact cheap baseline and the fixed-topology obstruction is not an architecture-wide lower bound.

For \(F(x)=x+D_{g(x)}Bx+Rx\) with the **gate** \(g\) continuous at zero, \(DF(0)=I+D_{g(0)}B+R\); row scaling can be absorbed into the final butterfly stage. Thus matching \(I+\alpha P\) locally for \(0<|\alpha|<1\) inherits the rank obstruction. Nonzero \(\alpha\) permits division by that scale; \(\alpha=0\) would be a trivial exception. At \(n=1024\), the exact threshold is 496, and rank eight is within (B2)'s range. This is a derivative/operator-norm statement, not average prediction error. [S0]

### B.1 Reconstructible size-16 approximation example

For \(t>1\), use zero-based indices \(0,1,2,3\), elementary matrices \(E_{ab}\), and swaps \(S_{ab}\). Define

\[
\begin{aligned}
L^{(0)}&=E_{00}+E_{22}-tE_{31}-t^2E_{33},&R^{(0)}&=E_{00}+E_{11}+E_{22}+E_{31},\\
L^{(1)}&=S_{13}\operatorname{diag}(-t,1,1,-t),&R^{(1)}&=S_{13}S_{01},\\
L^{(2)}&=I_4,&R^{(2)}&=I_4,\\
L^{(3)}&=-t^2E_{00}-tE_{01}+E_{22}+E_{33},&R^{(3)}&=E_{01}+E_{11}+E_{22}+E_{33}.
\end{aligned}
\]

Set \((A_t)_{ij}=L^{(i)}_{:,j}R^{(j)}_{i,:}\) and \(B_t=PA_t\). These factors lie in the stated bit-order butterfly class. With ordered set \(J=(0,1,3)\), frames \(Q_0=(e_1,e_3),Q_1=(e_0,e_3),Q_3=(e_0,e_1)\), and \(i=J_a\), set

\[
U_i=Q_i\operatorname{diag}(t^a,t^{-a}),\quad
V_i=Q_i\operatorname{diag}(t^{-a},t^a),\quad U_2=V_2=0.
\]

Stack these block rows into \(U,V\). Let \(w_i=e_{(i,2)}-e_{(2,i)}\) and \(G=\operatorname{diag}(I_4,S_{13},I_4,I_4)\). Define

\[
C_t=PG\left(UV^\top+\sum_{i\in J}w_iw_i^\top\right)G^\top.
\]

Then \(\operatorname{rank}C_t\le5\),

\[
\|P-B_t-C_t\|_F^2=4t^{-2}+2t^{-4}\to0,
\qquad \|B_t\|_2\ge t^2.
\tag{B4}
\]

Exact representation would require rank at least six by (B1). This demonstrates nonclosedness of the butterfly-plus-rank-five sum, while exposing parameter/operator blow-up. It is consistent with closedness results for the butterfly family itself under chainability. [S0,R15]

**Executable optional mini-program:** implement binary-bit stages and a separate dense construction; test indexing at \(n=4,16\); verify (B1)'s equality by exact integer rank of \(P-I\); construct (B4) with rational \(t\) and verify error/rank symbolically or with validated arithmetic; compare dense float64 residuals as \(t\) grows; report factor norms and loss of digits. A numerical SVD “rank five” label is insufficient to prove algebraic rank. Reproduce [R15]'s fixed-support approximation method only if extending beyond these supplied identities. Do not spend effort training a full network for this secondary replication.

## 12. References and search-coverage record

### 12.1 Annotated primary-source bibliography

All web sources were inspected on 31 October 2026. Dates below are publication/submission dates where established; documentation is identified as moving material. The inspection notes distinguish full text, specific sections, metadata, and repository inspection. No source's published benchmark number is treated as a controlled result for this project.

**[S0] Supplied research note.** *Coordinate-gated residual maps: a complete injectivity criterion*, 31 October 2026, 15 pages, author not identified in the supplied document. No public URL verified. PDF text and selected rendered pages inspected; proofs and exceptional cases rederived; limited independent float64 checks performed. Source for the original classification and butterfly appendix. Its own novelty/review assertions are not independently established. The identity hash appears at the start of this plan.

**[R1] David Gale and Hukukane Nikaidô.** *The Jacobian Matrix and Global Univalence of Mappings*. Mathematische Annalen 159, 81–93 (1965). [Primary paper PDF](https://fig.if.usp.br/~marchett/fluidos/jacobian-matrix-global-univalence-mapping_gale-nikaido.pdf). Inspected theorem setup and P-matrix discussion. Supplies actual-Jacobian univalence background; its domain and matrix hypotheses matter.

**[R2] Laurent Dinh, David Krueger and Yoshua Bengio.** *NICE: Non-linear Independent Components Estimation*. Submitted 30 October 2014, revised April 2015. [arXiv:1410.8516](https://arxiv.org/abs/1410.8516). Official metadata/abstract inspected; additive-coupling context corroborated by [R3,R4]. A reference for additive-coupling constructions; stability conditions are discussed in [R4].

**[R3] Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio.** *Density estimation using Real NVP*. Submitted 2016; ICLR 2017. [Full text, arXiv:1605.08803v3](https://arxiv.org/html/1605.08803v3). Sections 3.2–3.3 and inversion equations inspected. Closest constructive prior art; establishes the coupling equivalence boundary. Its original density-estimation experiments do not evaluate this fixed-coordinate repair objective.

**[R4] Jens Behrmann, Paul Vicol, Kuan-Chieh Wang, Roger Grosse and Jörn-Henrik Jacobsen.** *Understanding and Mitigating Exploding Inverses in Invertible Neural Networks*. AISTATS 2021, PMLR 130. [Paper](https://proceedings.mlr.press/v130/behrmann21a/behrmann21a.pdf); [arXiv version](https://arxiv.org/pdf/2006.09347). Theorem 2, stability sections and appendix arguments inspected. Direct prior art for inverse instability and the local/global distinction; motivates sensitivity guardrails. Trained-model observations belong to its own workloads.

**[R5] Andreas Krämer, Jonas Köhler and Frank Noé.** *Training Invertible Linear Layers through Rank-One Perturbations*. arXiv:2010.07033v2, 1 December 2020. [Full text](https://arxiv.org/html/2010.07033v2). Sections 2.1 and 3.2–3.4, inverse updates and numerical safeguards inspected. The initial version had a broader property-preserving title. P4Inv concerns linear-layer training, not the selected nonlinear repair necessity theorem. No maintained implementation was assumed.

**[R6] Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud and Jörn-Henrik Jacobsen.** *Invertible Residual Networks*. ICML 2019, PMLR 97:573–582. [Paper](https://proceedings.mlr.press/v97/behrmann19a/behrmann19a.pdf). Residual contraction/inversion conditions inspected. Establishes a useful sufficient design principle; it requires control of the full residual branch.

**[R7] Ricky T. Q. Chen, Jens Behrmann, David Duvenaud and Jörn-Henrik Jacobsen.** *Residual Flows for Invertible Generative Modeling*. NeurIPS 2019. [Paper](https://proceedings.neurips.cc/paper/2019/file/5d0d5594d24f0f955548f0fc0ff83d10-Paper.pdf). Residual-flow and LipSwish sections inspected. Scalar derivative bounds and contraction-based generative modeling are existing work; no selected repair classification was located there.

**[R8] Sébastien Lagrange, Nicolas Delanoue and Luc Jaulin.** *Injectivity analysis using interval analysis: Application to structural identifiability*. Automatica 44 (2008), DOI [10.1016/j.automatica.2008.04.018](https://doi.org/10.1016/j.automatica.2008.04.018). [Author manuscript](https://webperso.ensta.fr/jaulin/paper_lagrange_automatica_2008.pdf). Full four-page manuscript, algorithms and limitations inspected. Guaranteed interval multiplicity/injectivity analysis is established; exponential dimensional cost and unresolved singular cases limit it. Publisher access was inconsistent; the author manuscript was accessible. Old IAVIA binaries were not tested.

**[R9] Tianqi Cui, Thomas Bertalan, George Pappas, Manfred Morari, Yannis Kevrekidis and Mahyar Fazlyab.** *Certified Invertibility in Neural Networks via Mixed-Integer Programming*. L4DC 2023, PMLR 211. [Paper](https://proceedings.mlr.press/v211/cui23b/cui23b.pdf). Formal optimization problems and ReLU encoding inspected. Prior art for local invertibility regions; no faithful sigmoid replacement is implied.

**[R10] Navid Hashemi, Justin Ruths and Mahyar Fazlyab.** *Certifying Incremental Quadratic Constraints for Neural Networks via Convex Optimization*. L4DC 2021, PMLR 144:842–853. [Paper](https://proceedings.mlr.press/v144/hashemi21a/hashemi21a.pdf). Incremental/local certificate setup inspected. Stronger comparison family than crude product norms if generic certification is pursued; multiplicative gates need a faithful formulation. Linked implementation was not run.

**[R11] Zhouxing Shi, Qirui Jin, Zico Kolter, Suman Jana, Cho-Jui Hsieh and Huan Zhang.** *Neural Network Verification with Branch-and-Bound for General Nonlinearities*. TACAS 2025; arXiv:2405.21063v3. [Full text](https://arxiv.org/html/2405.21063v3). General nonlinear and multiplication relaxations inspected. Official [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) READMEs inspected for supported operations/CPU path. Strong contemporary general-verifier comparator; no Mac certificate soundness or runtime was established by documentation alone.

**[R12] Ruigang Wang, Krishnamurthy (Dj) Dvijotham and Ian Manchester.** *Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks*. ICML 2024, PMLR 235:50379–50399. [Venue page](https://proceedings.mlr.press/v235/wang24p.html); [full text](https://arxiv.org/html/2402.01344v4); [official code](https://github.com/acfr/PLNet). Definitions, architecture and inverse method inspected; repository README/license inspected, not executed. BiLipNet is relevant when stable global inversion becomes the actual application requirement.

**[R13] Tri Dao, Albert Gu, Matthew Eichhorn, Atri Rudra and Christopher Ré.** *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations*. ICML 2019, PMLR 97:1517–1527. [Paper](https://proceedings.mlr.press/v97/dao19a/dao19a.pdf). Factorization definitions and implementation context inspected. Establishes learned butterfly transforms; reported training/inference results use that paper's platforms and tasks.

**[R14] Léon Zheng, Gilles Puy, Elisa Riccietti, Patrick Pérez and Rémi Gribonval.** *Butterfly factorization by algorithmic identification of rank-one blocks*. arXiv:2307.00820, 3 July 2023. [Metadata and paper access](https://arxiv.org/abs/2307.00820); [official reproduction repository](https://github.com/leonzheng2/permutation-butterfly). Abstract, problem statement and repository README inspected; the paper is in French. Supports the established complementary-block/identification landscape; no performance reproduction or exact rank-distance antecedent was established.

**[R15] Quoc-Tung Le, Léon Zheng, Elisa Riccietti and Rémi Gribonval.** *Butterfly factorization with error guarantees*. arXiv:2411.04506v2, 27 March 2025. [Full text](https://arxiv.org/html/2411.04506v2). Chainability, approximation algorithm, complementary low-rank characterization and existence/closedness sections inspected. Distinguishes a fixed butterfly family's approximation guarantees from adding low-rank corrections or changing topology.

**[R16] Tri Dao, Beidi Chen, Kaizhao Liang, Jiaming Yang, Zhao Song, Atri Rudra and Christopher Ré.** *Pixelated Butterfly: Simple and Efficient Sparse Training for Neural Network Models*. ICLR 2022. [Full text, arXiv:2112.00029v2](https://arxiv.org/html/2112.00029v2). Flat/block butterfly plus low-rank parameterization and hardware discussion inspected. Direct prior art against novelty of the broad structured-plus-low-rank combination; no M3 speed implication follows.

**[R17] Tri Dao, Beidi Chen, Nimit S. Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra and Christopher Ré.** *Monarch: Expressive Structured Matrices for Efficient and Accurate Training*. ICML 2022, PMLR 162:4690–4721. [Paper](https://proceedings.mlr.press/v162/dao22a/dao22a.pdf). Block-product parameterization, analytic projection result and related hierarchy inspected. Stronger hardware-aware structured-transform context if the project returns to dense-layer replacement.

**[R18] Fulvio Gesmundo, Jonathan D. Hauenstein, Christian Ikenmeyer and J. M. Landsberg.** *Complexity of linear circuits and geometry*. arXiv:1310.1362v2, 10 March 2015 revision. [Paper](https://arxiv.org/pdf/1310.1362). Butterfly dimension/circuit proposition and border-rigidity context inspected. The dimension result is established; no novelty is claimed for it. An author-site redirect failed; arXiv full text succeeded.

**[R19] Frederic Koehler, Viraj Mehta and Andrej Risteski.** *Representational aspects of depth and conditioning in normalizing flows*. ICML 2021, PMLR 139. [Paper](https://proceedings.mlr.press/v139/koehler21a/koehler21a.pdf). Coupling-depth, permutation/linear representation and conditioning discussions inspected, with an accessible longer manuscript consulted during the targeted follow-up. Distinct from a single fixed shear's additive-norm repair objective.

**[R20] John Urschel.** *Representing the special linear group with block unitriangular matrices*. PNAS Nexus 2(10), pgad311 (2023), DOI [10.1093/pnasnexus/pgad311](https://doi.org/10.1093/pnasnexus/pgad311). [Author manuscript](https://math.mit.edu/~urschel/preprints/sln.pdf); [archived publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558096/). Published theorem/coupling consequences inspected; a later archive request encountered a browser wall. Relevant to multilayer routing/depth alternatives, not a claim that this two-dimensional repair result is new.

**[R21] Dennis DeTurck.** *Rotations, Shears, and Two-Generation on Spheres*. arXiv:2610.27507v1, submitted 20 October 2026. [Official record](https://arxiv.org/abs/2610.27507); [full text](https://arxiv.org/html/2610.27507v1). Metadata and theorem/rotational-shear statement inspected. The source PDF's recent citation is genuine. It is a preprint, and its sphere-universality result is contextual prior art rather than a baseline for the repair objective.

**[R22] Mathis Gerdes and Miranda C. N. Cheng.** *Analytic Bijections for Smooth and Interpretable Normalizing Flows*. ICML 2026, PMLR 306:34588–34610. [Venue page](https://proceedings.mlr.press/v306/gerdes26a.html); [full text](https://arxiv.org/html/2601.10774v2). Scalar-bijection construction and limitations inspected. Prevents an unsupported pivot to claiming that smooth analytic invertible scalar transforms are new. Its experimental results are not compared numerically here.

**[R23] HazyResearch.** *Butterfly matrix multiplication in PyTorch*, official repository, moving revision inspected 31 October 2026. [Repository](https://github.com/HazyResearch/butterfly). README, extension/pure-framework distinction and license listing inspected. The fast documented path uses C++/CUDA; a pure PyTorch fallback exists but does not establish MPS speed. If reused, record the exact commit and validate bit-stage conventions independently. No repository code was executed here.

**[R24] Python-FLINT contributors.** *Installation documentation*, moving stable documentation inspected 31 October 2026. [Install guide](https://python-flint.readthedocs.io/en/stable/install.html). macOS ARM binary support, supported interpreter range, and `python -m flint.test` inspected. Supports the target stack selection; not evidence of a successful installation on the user's laptop.

**[R25] Python-FLINT and FLINT contributors.** *Arb scalar API and arithmetic guarantees*, moving documentation inspected 31 October 2026. [Python API](https://python-flint.readthedocs.io/en/stable/arb.html); [FLINT source documentation](https://github.com/flintlib/flint/blob/main/doc/src/arb/index.md). Rational construction, midpoint/radius semantics, directed bounds, exact dyadic serialization and inclusion/error propagation inspected. Correct translation and complete root enumeration remain the project's responsibility.

**[R26] SymPy contributors.** *Polynomials Manipulation Module Reference*, documentation inspected 31 October 2026. [Reference](https://docs.sympy.org/latest/modules/polys/reference.html). Public rational root isolation, square-free operations and refinement APIs inspected. The documented API does not authorize silently treating rounded algebraic coefficients as rational exact inputs.

**[R27] PyTorch contributors.** *MPS backend*, documentation inspected 31 October 2026; stable route resolved to a 2.15 documentation page. [Documentation](https://docs.pytorch.org/docs/2.15/notes/mps.html). Double-precision limitations and fallback behavior inspected. This motivates CPU correctness work; the actual local version must be discovered and locked.

**[R28] MLX contributors / Apple.** *Data Types*, moving documentation inspected 31 October 2026. [Documentation](https://ml-explore.github.io/mlx/build/html/python/data_types.html). Float64 CPU-only note inspected. MLX is optional and not a prerequisite for this PoC.

**[R29] Python-FLINT and FLINT contributors.** *Licenses and project introduction*, inspected 31 October 2026. [Python-FLINT license](https://github.com/flintlib/python-flint/blob/main/LICENSE); [FLINT introduction](https://github.com/flintlib/flint/blob/main/doc/src/guide/getting-started/introduction.md). Binding MIT and current FLINT licensing inspected. Older summaries can differ across versions; actual installed notices control any redistribution decision.

**[R30] NumPy contributors.** *NumPy 2.0.0 Release Notes* (2024), inspected 31 October 2026. [Release notes](https://numpy.org/doc/stable/release/2.0.0-notes.html). macOS Accelerate/wheel support discussed in the official release notes. This is compatibility context, not proof of which BLAS the target install uses; save `numpy.show_config()`.

**[R31] Hongxiang Fan, Chun Pong Chau, Stylianos Venieris, Alexandros Kouris, Royson Lee and Mohamed S. Abdelfattah; listed applicant Samsung Electronics.** *Apparatus for accelerating neural network computations*, publication EP4261745A1, 18 October 2023. [Published patent document](https://patents.google.com/patent/EP4261745A1/en). Technical butterfly/sparsity discussion and publication metadata inspected. It cites butterfly-plus-low-rank work as context. This narrow technical search provides no claim-construction, legal-status, patentability or freedom-to-operate opinion.

### 12.2 Search scope, sequence, and stopping rationale

The investigation used two search engines, direct primary-paper retrieval, official documentation, and repository inspection. Search terms expanded beyond the source's terminology:

| Search family | Representative queries and citation tracing | What changed the plan |
|---|---|---|
| Original mechanism | `coordinate-gated injectivity sigmoid`; `self-gated injectivity`; `sigmoid Hadamard residual Jacobian`; `global univalence P matrix` | Separated actual from frozen Jacobians; audited source proof instead of assuming a norm bound |
| Safe alternatives | `affine coupling inverse stability`; `invertible residual contraction`; backward references from Real NVP/Residual Flows; forward stability work | Recognized repaired family as known coupling; added inverse-conditioning guardrails and contemporary BiLipNet context |
| Repair specificity | `sigmoid rank-one injective`; `linear repair sigmoid invertibility`; `skip conditioning invertible rank-one`; `m22 sigmoid injective`; additive spectral norm and optimal coupling/rotation searches | Located P4Inv and conditioning literature; no exact all-repair classification/frontier match found |
| Bounded-domain verification | `injectivity analysis interval`; `certified invertibility radius neural`; `sigmoid multiplication verification`; forward general-nonlinearity/Jacobian support | Rejected generic “new local verifier” framing; found IAVIA, MIP, IQC and GenBaB alternatives |
| Structured circuits | `butterfly rank permutation rigidity`; `butterfly border low rank approximation`; references from error-guarantee and geometry papers | Preserved fixed-topology/norm assumptions; recognized existing butterfly-plus-low-rank methods and closedness distinctions |
| Current research | 2025–2026 filters and direct verification of supplied recent references | Verified DeTurck rather than treating its date as evidence of fabrication; identified analytic-bijection context |
| Implementation | Official MPS/MLX precision docs, Python-FLINT ARM/API/license docs, SymPy root isolation, NumPy macOS support | Selected CPU validated arithmetic; added exact root/interval and dependency gates |
| Patents/application disclosures | `invertible gated low rank` and `butterfly low-rank` restricted to published patent documents | Found adjacent coupling/acceleration disclosures, no selected exact theorem; no legal conclusion drawn |

After narrowing to T1–T3, additional targeted searches repeatedly returned affine coupling, inverse instability, rank-one linear updates, and coupling-depth representations rather than a materially different direct competitor. Relevant primary hits were inspected. This was a mechanism-driven stopping decision, not a paper-count quota. The execution-stage antecedent gate remains important because the candidate theorem is short and may exist under different mathematical terminology.

**Blind spots and access limitations:** unindexed or non-English mathematical work, older books/theses, unpublished notes, and a nonexhaustive patent landscape. Some repository raw-file, publisher, author-site redirect and archive requests failed; accessible official versions or author manuscripts were used where available, and uninspected implementation behavior is not asserted. Apple's target-specification page was checked through the feasibility review, but a later direct request was rate-limited; this plan therefore relies on local hardware discovery rather than an assumed hardware configuration. No external trained baseline, Mac installation, or rigorous certificate backend was executed during the research phase.

### 12.3 Handoff completion checklist

Before beginning confirmation, the coding agent should be able to answer **yes** to each of the following from project artifacts alone:

- The original gate and repair class are fixed, and every deviation has an explicit scope label.
- All central proofs include boundary cases, norm-cap versus actual-norm distinctions, and coordinate/domain assumptions.
- The strongest feasible mathematical comparators are included, with source swap marked out of budget below 2.
- All numerical certificates use exact/enclosed inputs, a complete stationary-root enumeration, and replayed signs.
- Every expensive extension has passed its prior evidence and resource gate.
- The target environment, parameters, raw failures, timing boundaries, source revisions and dependency locks are recorded.
- Numerical uncertainty, practical significance, historical novelty and mathematical validity are separate conclusions.
- A negative or inconclusive result has a documented interpretation and does not trigger unbounded favorable-case searching.

The research-phase handoff is complete as a plan. The proposed repository, rigorous certificates and M3 measurements remain execution work, not files or results implicitly delivered with this Markdown document.
