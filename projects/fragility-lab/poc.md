# POC_EXECUTION_BRIEF

## Joint-information capacity bounds for a Financial Fragility Radar

**Prepared:** 2 November 2026.  
**Recipient:** autonomous coding agent configured as **gpt-6.1-sol, xhigh reasoning**.  
**Deliverable you must produce:** an executed, reproducible research result, following the gates below.  
**Status of this document:** research and execution design; the proposed PoC has not been implemented or benchmarked. All performance budgets below are planning limits, not measured M3 Max performance.

## 1. Project decision and contribution

Build one prerequisite study for the proposed **U.S. Equity & Macro Financial Fragility Radar**: determine the practical benefit of jointly using a capped leverage sample and independently known leverage moments when computing a conditional lower bound on upward demand-absorption capacity.

The product ambition is to measure vulnerability to self-reinforcing buying. The supplied mathematics establishes a capacity functional under a particular rebalancing model. Its strongest public-data forecasting interpretation is unsupported. Therefore, do not begin with a historical stock-ranking system, macro model, dashboard, or live feed. First resolve whether the proposed information combination adds enough useful capacity certification to justify its numerical and data requirements.

### 1.1 Primary, falsifiable question

> In the specified synthetic institutional-information setting, with an enforceable cap \(A=10\) on leverage minus one, exact first two initial-holdings-weighted population moments, and \(n=128\) or \(512\) independent holdings-weighted observations, can joint optimization recover, on average, at least **0.10 of the true modeled capacity** beyond the strongest inexpensive certificate that uses the same information separately, while retaining a numerically verified continuous-support lower bound within the stated M3 Max budget?

The comparison is against

\[
B=\max\{C_{\mathrm{mom}},C_{\mathrm{DKW}}\},
\]

not against either constituent alone. The primary effect is the incremental value of **combining constraints**. Also report improvement over the moment-only bound as a secondary measure of the value of obtaining samples.

The fixed primary regime uses a conservative structural cap: the evaluator's synthetic populations have support at most \(a=2\), but the estimator is told only \(a\le10\). This is a deliberate information regime, not an estimate of typical U.S. investor leverage. Tighter caps \(A=2,4\) are secondary controls. No claim about general superiority across caps is authorized.

The 0.10 threshold is a **research decision threshold**, chosen before confirmation: recover an additional tenth of the modeled shock-absorption budget to justify joint optimization over two cheap established calculations. It is not a calibrated trading threshold, financial return, probability, or demonstrated customer willingness to pay. A numerical enclosure of order one percent of capacity is needed to distinguish this effect from computational uncertainty.

### 1.2 Smallest defensible contribution

A useful positive result would be a reproducible characterization of when joint sample-and-moment information gives material, rigorously bounded improvement for this particular nonlinear capacity functional, at affordable local cost. A useful negative result would show that the strongest separated bounds already recover essentially all useful information in the tested regime, or that closing the remaining identification gap is computationally impractical within the budget.

**Do not claim a new confidence inequality, new robust-optimization framework, new leverage-feedback mechanism, or demonstrated forecasting skill.** Goodness-of-fit regions intersected with moment information already appear in Bertsimas, Gupta and Kallus, especially §3.2, Eq. (11) [R3]. Canonical moment extrema and worst-case exponential transforms have close precedent in Pandit and Meyn [R2]. The specific quantitative study proposed here was not located in the targeted search; that is not exhaustive novelty clearance.

A positive sign alone is uninformative: set intersection automatically gives a weakly stronger ideal bound. The scientific question concerns the **size, reliability, conditioning, and cost** of the improvement.

### 1.3 Decision this study informs

Decide whether a future institutional version should implement the joint calculation or retain the inexpensive separated bounds. The plausible information arrangement is an institution releasing audited aggregate moments and an enforceable cap, plus a small holdings-proportional sample, while withholding its complete distribution. If the complete relevant position census is available, calculate its capacity directly; sampling is unnecessary.

Public U.S. financial data does not automatically provide this observation model. For example, SEC Form 13F has reporting lags and excludes short positions; it does not supply the complete collateral and rebalancing information required here [R9]. A cap is not a target-leverage policy. Those are substantive limitations, not missing software features.

### 1.4 Scope and exclusions

Implement a single-asset, long-only, continuously targeted-leverage mathematical experiment. Exclude crypto entirely.

Do not implement live ingestion, stock selection, causal depth estimation, macro forecasting, network contagion, short positions, derivative gamma, cross-margin portfolios, a composite risk score, trading, or a dashboard. Do not buy data or use remote experimental compute. These exclusions preserve the product's practical purpose by resolving an indispensable certification question first.

| Alternative considered | Reason not selected |
|---|---|
| Public-data equity/crash prediction | Required leverage weights, behavior, cap and supply response are not identified by the available inputs; a backtest would mainly test unvalidated proxies. |
| Finite-round extension alone | Already developed in notebook §11; its endpoint extrema lack the continuous model's established stability interpretation. |
| Joint capacity certification | Explicit gap in notebook §7, constructive local computation, credible strong baseline, and useful positive or negative outcomes without proprietary data. |

Execute the authorized stages autonomously. Repair ordinary implementation problems within the cumulative limits. A justified early stop, accompanied by its derivation or diagnostic and a reproducible report, is a completed outcome.

## 2. Inputs, evidence status, and mathematical contract

### 2.1 Supplied sources and provenance

Two supplied attachments were read completely, including all 26 notebook cell sources and its embedded figure.

| File | Location in the planning workspace | SHA-256 |
|---|---|---|
| instructions.md | /workspace/scratch/59b5e43ab315/upload/instructions.md | a4b58d085796ad8bc0055bee7a5ba9ce4ed5efa295ebc9726f042fc75467698d |
| bubble_ignition_research(4).ipynb | /workspace/scratch/59b5e43ab315/upload/bubble_ignition_research(4).ipynb | 1f79c449b2ce0789cf06d99d709d71388105535ff2034ce3cd4d5412ff65b2c0 |

On the Mac, archive any supplied copies under inputs/. Those workspace paths describe the planning environment and are not assumed to exist on the Mac. The notebook is the supplied mathematical source; no separate paper PDF was attached. Its embedded code is available, but its code cells have null execution counts. Its tables describe an earlier recorded synthetic run, not an independently verified market study. The original separately named reproduce.py and results.json are not attached. Do not claim to have checked their original byte hashes.

All mathematics and experiment definitions needed for this PoC are restated below. A missing local notebook copy does not block implementation from this brief; document that archival limitation. External papers provide attribution and context, not an indispensable data dependency.

Source locations below use notebook section numbers, equation numbers, and **zero-based cell indices**.

### 2.2 Objects, units, and assumptions

For one asset, let price be \(p>0\), reference price \(p_0>0\), investor shares \(q_i\ge0\), equity \(E_i>0\), and fixed target leverage

\[
L_i=\frac{p q_i}{E_i}\ge1,\qquad a_i=L_i-1\ge0.
\]

Define \(x=\log(p/p_0)\), initial total holdings \(Q=\sum_iq_i(p_0)>0\), and

\[
\mu=\sum_i\frac{q_i(p_0)}{Q}\delta_{a_i}.
\]

The weights are **initial shares**, not investor counts, equity weights, current shares, or arbitrary portfolio weights. Let \(m_1=E_\mu[a]\), \(m_2=E_\mu[a^2]\), and let the remainder of the market supply \(Dx\) additional shares, with \(D>0\).

| Quantity | Meaning and units |
|---|---|
| \(a,L,x,m_j,A\) | Dimensionless; \(A\) caps \(a\), so maximum target leverage is \(A+1\). |
| \(Q,D,u,H,C\) | Shares; \(D\) is shares per unit log-price move. |
| \(D/Q\) | Dimensionless depth relative to the modeled sector's initial shares. |
| \(\mu\) | Probability distribution weighted by initial shares. |
| \(x^*\) | Prospective log-price increase at the continuous-model fold. |
| \(C\) | Maximum external net buying that the stylized clearing curve absorbs. |

The source model assumes finite-variation price adjustment, self-financing continuous target-leverage rebalancing, no financing costs or dividends during adjustment, sufficient borrowing, no defaults, and a fixed logarithmic background supply curve. From \(dE_i=q_i\,dp\),

\[
d\log q_i=a_i\,d\log p,\qquad q_i(p)=q_i(p_0)e^{a_i x}.
\]

Consequently,

\[
H_\mu(x)=Dx-QE_\mu[\operatorname{expm1}(a x)],\quad x\ge0. \tag{1}
\]

Under the auxiliary adjustment rule \(\tau\dot x=u-H_\mu(x)\), \(\tau>0\), a clearing equilibrium is locally stable where \(H_\mu'(x)>0\). This interpretation depends on that rule; it is not an estimated equation for real markets. Notebook §§2–3, Eqs. (1)–(3), cells 2–3 [N].

For \(m_1>0\) and \(\Delta=D-Qm_1>0\),

\[
H_\mu'(x)=D-QE_\mu[a e^{a x}],\quad
H_\mu''(x)=-QE_\mu[a^2e^{a x}]<0,
\]

\[
QE_\mu[a e^{a x^*}]=D,\qquad C(\mu)=H_\mu(x^*)=\max_{x\ge0}H_\mu(x). \tag{2}
\]

For \(D\le Qm_1\), use the source's boundary convention \(x^*=C=0\). For \(\mu=\delta_0\), \(D>0\), there is no finite fold and \(C=x^*=+\infty\). Handle these branches before root solving. Reject invalid/nonfinite inputs; do not silently repair an infeasible distribution.

A fold is conditional loss of upward absorption capacity. It does not establish fundamental mispricing, a future crash, an event date, or a market-event probability. A low lower bound can indicate insufficient information rather than low true capacity.

### 2.3 What the supplied results establish

| Result | Source and status | Consequence for this project |
|---|---|---|
| Continuous clearing curve and unique fold | Notebook §§2–3; derivation valid under stated assumptions. | Defines the sole target functional. |
| Exact-moment hidden-tail collapse | §4, Eqs. (4)–(10), cells 4–8; constructive proof audited. Interior feasible moment fibers permit TV-small, exact-moment perturbations with \(C\to0\) when no common cap exists. Boundary feasible fibers are singleton. | Moments without a verified cap cannot generally certify positive capacity uniformly. Do not infer a cap from the sample maximum. |
| Finite-sample impossibility | §5, Eqs. (11)–(12), cell 9; direct TV argument audited. | Applies to the broad uncapped class and the specified sampling model; not a universal impossibility of estimation. |
| Sharp capped moment bounds | §6, Eqs. (13)–(15), cell 10; classical extremal-expectation specialization [R1], [R2]. | Supplies the strongest inexpensive moment comparator. |
| Exact DKW worst law | §7, Eq. (16), cell 11; stochastic-order construction valid with known cap and IID holdings-weighted draws. | Supplies the strongest inexpensive sample comparator; generally violates additionally imposed moments. |
| Compound-Poisson identity | §8, cell 12; algebraic convex-conjugate identity. | Does not turn \(e^{-C/Q}\) into a bubble probability. |
| Recorded computations | §§9–10, cells 13–18; source reports synthetic checks. | Numerical illustrations, not proof of general inequalities or empirical predictive performance. |
| Finite rounds | §11, Eqs. (17)–(19), cells 19–22. | Moments through order \(n\) determine its \(n\)-round endpoint curve. For \(n=1\), initial stability implies no upward endpoint fold. Do not transfer its extrema into real-time stability claims. |
| Economic qualifications | §12, cell 23. | One-sided caps, stochastic paths, funding changes and endogenous depth can change the mechanism. |

The arbitrary-baseline tail proof invokes an auxiliary representing law whose existence follows from moment geometry. The notebook code constructs only explicit uniform-baseline examples. This study needs no generic moment-representation oracle and must not pretend that the existence proof provides one.

The underlying leverage/depth instability mechanism already appears in MinMaSS [R6]; application to historical U.S. equities also predates this project [R7]. One-sided leverage constraints differ from symmetric targeting [R8]. These precedents narrow the contribution.

## 3. The statistical target and credible baselines

### 3.1 Information supplied to every primary method

Each scenario supplies only:

- \(Q,D,A\), with the same verified economic model;
- exact population \(m_1,m_2\), supplied independently of the sample;
- \(n\) IID observations drawn from \(\mu\);
- confidence level \(1-\alpha=0.95\).

Do not constrain population moments to sample estimates. That substitution changes the coverage argument. Exact aggregate moments are hypothetical privileged inputs, not facts obtainable from public market prices. Independence is required for the observations; independently known moments need not be statistically independent of the observations because they are treated as known constants.

Use \(Q=1\) for the experimental normalization. Thus reported capacities are \(C/Q\); multiplying by a real \(Q\) would require actual share holdings. Keep \(Q\) in the reference interface and verify scale invariance.

Set

\[
\epsilon=\min\left(1,\sqrt{\frac{\log(2/\alpha)}{2n}}\right),
\quad
\mathcal P=\{\nu\in\mathcal P([0,A]):
E_\nu a=m_1,\ E_\nu a^2=m_2,\
\|F_\nu-\widehat F_n\|_\infty\le\epsilon\}. \tag{3}
\]

The primary ideal joint value, when \(\mathcal P\ne\varnothing\), is

\[
C_J^*=\inf_{\nu\in\mathcal P}C(\nu). \tag{4}
\]

On the single DKW event that the true law belongs to the band, it also belongs to \(\mathcal P\), so every verified lower bound on (4) is a lower confidence bound on \(C(\mu)\). No extra confidence penalty is needed for intersecting exact known moments or for taking the maximum of lower bounds valid on the same event. No independence between competing estimators is assumed.

This is one-snapshot coverage. It is not a simultaneous guarantee over repeated daily monitoring, many assets, or optional stopping.

### 3.2 Sharp moment-only baseline

Let \(0<m_1<A\), \(m_1^2<m_2<Am_1\), \(r=D/(Qm_1)>1\), and

\[
\Phi=D\log r-D+Qm_1.
\]

The mean-and-cap baseline is \(C_{\mathrm{mean}}=\Phi/A\), achieved by
\((1-m_1/A)\delta_0+(m_1/A)\delta_A\). The mean-only upper bound is \(\Phi/m_1\).

For exact first two moments define

\[
b=\frac{m_2}{m_1},\quad
\nu_-=\left(1-\frac{m_1^2}{m_2}\right)\delta_0+
\frac{m_1^2}{m_2}\delta_b,
\]

\[
c=\frac{Am_1-m_2}{A-m_1},\quad
\theta=\frac{m_2-m_1^2}{A^2-2Am_1+m_2},\quad
\nu_+=(1-\theta)\delta_c+\theta\delta_A.
\]

Then

\[
C_{\mathrm{mom}}=C(\nu_+)\le C(\mu)\le
C_{\mathrm{mom}}^{U}=C(\nu_-)=\frac{m_1}{m_2}\Phi. \tag{5}
\]

Compute the lower endpoint by its scalar fold root. Handle \(m_2=m_1^2\) as the unique law \(\delta_{m_1}\), \(m_2=Am_1\) as the unique endpoint mixture, and \(m_1=0,A\) explicitly. Check feasibility \(0\le m_1\le A\), \(m_1^2\le m_2\le Am_1\). These formulas come from notebook §6 and classical three-convex expectation ordering [N], [R1], [R2].

### 3.3 Exact DKW-only baseline

Remove the lowest \(\epsilon\) of empirical probability mass, splitting a tied mass if necessary, and place it at \(A\). Equivalently,

\[
F_w(a)=\max(\widehat F_n(a)-\epsilon,0)\quad(0\le a<A),
\qquad F_w(A)=1. \tag{6}
\]

Set \(C_{\mathrm{DKW}}=C(\nu_w)\), including the zero-capacity initial-instability branch. This is the stochastically largest law in the entire capped DKW band, so it minimizes capacity. Use outward numerical enclosures for a reported lower endpoint.

For numerical release, construct the DKW worst law using a rational outward upper radius \(\epsilon_{\mathrm{hi}}\ge\epsilon\), capped at 1. The larger radius moves at least as much low mass upward and therefore gives a conservative lower endpoint. Use exact rational mass transfer. An outward lower radius may supply an upper enclosure for this baseline, but must not be used to release its lower bound.

The primary comparator \(B\) is the maximum of the two verified lower endpoints from (5) and (6). Also report each component, the mean-and-cap bound, and the empirical plug-in capacity. The plug-in is descriptive and carries no distribution-free lower-bound label.

### 3.4 Why the joint problem is legitimate but not a closed-form DKW modification

The moment-plus-band set is convex and weakly compact when nonempty. At each empirical jump \(t\), the band is equivalently enforced by

\[
\nu([0,t])\ge \widehat F_n(t)-\epsilon,\qquad
\nu([0,t))\le \widehat F_n(t-)+\epsilon,
\]

together with support and probability constraints. The closed/open Portmanteau inequalities make these restrictions weakly closed. Include endpoints and ties correctly.

Jensen's inequality bounds every fold by

\[
X=\frac{\log(D/(Qm_1))}{m_1}.
\]

On \([0,X]\), \(H\) is continuous and concave in \(x\), and continuous affine in the measure. Sion's theorem therefore gives

\[
\inf_{\nu\in\mathcal P}\max_{x\in[0,X]}H_\nu(x)
=
\max_{x\in[0,X]}
\left\{Dx-Q\sup_{\nu\in\mathcal P}
E_\nu[\operatorname{expm1}(a x)]\right\}. \tag{7}
\]

This is a derivation for this problem using [R4], not a novel minimax theorem. A common optimizing distribution for all \(x\) is not assumed. Fixed-mean distinct laws cannot strictly dominate one another in first-order stochastic order.

An empty intersection has no value under (4). A floating-point solver declaring infeasibility does not prove the continuous intersection empty. Use the explicit failure policy in §5.6; never report \(+\infty\) as a capacity certificate because an infimum over an empty set is formally \(+\infty\).

## 4. Constructive continuous-support algorithm

Implement the following method. It uses finite sparse LPs, scalar root solves, and independently checked numerical bounds. It does not assume the true support equals the computational grid.

### 4.1 Normalize and bound the search interval

For \(A>0\), set

\[
y=a/A,\quad t=Ax,\quad \bar m=m_1/A,\quad
\bar s=m_2/A^2,\quad \delta=D/(QA),
\]

\[
h_\nu(t)=\delta t-E_\nu[\operatorname{expm1}(yt)],\qquad
C(\nu)/Q=\max_{t\ge0}h_\nu(t).
\]

**Internal units contract:** every solver, baseline and comparison interface returns normalized capacity \(c=C/Q\). Convert every baseline and analytic upper bound to these units before combining it with an LP value. The report may additionally emit shares as \(C=Qc\), multiplying exactly once at the output boundary. The theoretical equations in §§2–3 retain \(C\) in shares. Test \(Q=2,D=3\) against \(Q=1,D=1.5\): the normalized capacity and fold agree, and the share capacity doubles. In general use \(C_{\mathrm{scale}}=\Phi/(Qm_1)\) internally.

For initially stable nondegenerate inputs, a valid common fold bound is

\[
T=\min\left\{
\frac{\log(\delta/\bar m)}{\bar m},
\frac{\delta-\bar m}{\bar s}
\right\}. \tag{8}
\]

The second bound follows from \(E[y e^{yt}]\ge\bar m+\bar s t\). For the mean-only ablation use only the first bound. Construct a rational upper enclosure \(T_{\mathrm{hi}}\ge T\). Search on \([0,T_{\mathrm{hi}}]\). Use \(0\le z\le \delta T_{\mathrm{hi}}\) as an explicit epigraph-variable bound.

### 4.2 Upward rounding gives a safe outer approximation

Let \(G=\{0=g_0<\cdots<g_M=1\}\) contain:

1. every distinct observed normalized sample value, exactly;
2. a nested uniform mesh;
3. moment-extremizer nodes that lie in the interval, computed from supplied moments.

Do not merge nearby unequal sample values. Store rational node values for exact bookkeeping; floats are only solver views. Define \(h=\max_j(g_{j+1}-g_j)\) and \(T_G(y)=\min\{g\in G:g\ge y\}\).

For every grid knot \(g\), \(T_G(y)\le g\) iff \(y\le g\). Thus
\(F_{T_G\#\nu}(g)=F_\nu(g)\). Because the empirical CDF has no jump between grid knots, upward rounding preserves the entire DKW band. Moreover,

\[
\bar m\le E[T_GY]\le\min(1,\bar m+h),
\]

\[
\bar s\le E[(T_GY)^2]\le
\min(1,\bar s+2h\bar m+h^2). \tag{9}
\]

For \(t\ge0\), exponential demand increases under rounding, so
\(C(T_G\#\nu)\le C(\nu)\).

Define \(\mathcal P_{\mathrm{out}}(G)\) as grid-supported probability laws satisfying the original band and the relaxed moment intervals (9). Then

\[
C_{\mathrm{out}}^*(G):=
\inf_{w\in\mathcal P_{\mathrm{out}}(G)}C(w)
\ \le\ C_J^*. \tag{10}
\]

This inequality is the essential continuum-safety argument. A grid law constrained to exact moments alone gives the opposite approximation direction and cannot supply the required lower certificate.

Nested meshes shrink \(h\), but the sequence of individual outer optima need not be monotone because the support set also expands. Retain the maximum of all verified lower bounds. Compactness supplies a limiting justification when the continuous set is nonempty; no finite convergence rate is assumed near degenerate moment constraints.

### 4.3 Sparse LP representation and exchange algorithm

Use grid weights \(w_0,\ldots,w_M\), cumulative masses \(v_0,\ldots,v_{M-1}\), and epigraph \(z\). Require

\[
v_j=\sum_{i=0}^{j}w_i,\qquad \sum_iw_i=1,\qquad
0\le w_i,v_j\le1.
\]

Implement the cumulative equalities using adjacent differences, not a dense triangular matrix. At every knot \(g_j<1\),

\[
\max(0,\widehat F_n(g_j)-\epsilon)\le v_j
\le\min(1,\widehat F_n(g_j)+\epsilon).
\]

The predecessor cumulative mass checks the left limit at the next sample knot. Because both measures are grid-supported and the grid includes every sample value, these constraints cover the whole CDF band. Add the two moment intervals from (9).

For a finite set \(\mathcal T\subset[0,T_{\mathrm{hi}}]\), solve the master LP

\[
\min z,\qquad
z\ge \delta t-\sum_jw_j\operatorname{expm1}(g_jt)
\quad\text{for each }t\in\mathcal T. \tag{11}
\]

Initialize \(\mathcal T\) with 0, \(T_{\mathrm{hi}}\), and the normalized moment-only and DKW fold locations when finite and within the interval. Deduplicate exactly represented cut locations.

Every master optimum is no greater than the outer optimum and hence no greater than \(C_J^*/Q\). Given the LP's candidate weights, solve their scalar feedback equation to obtain a candidate maximizing \(t\), then add that cut. A separation candidate is only a way to choose another \(t\); its feasibility is not a certificate. Reject grossly invalid candidates. Tiny negative solver weights may be clipped and renormalized only for choosing a cut, with the repair logged, never to establish feasibility or a bound.

### 4.4 An independent lower certificate for each master

A successful LP status and its approximate objective are insufficient.

Write the mathematical master as

\[
\min c^\top u,\qquad Au\le b,\ Eu=e,\ 0\le u\le U.
\]

For any \(\lambda\ge0\), unrestricted \(\eta\), let

\[
r=c+A^\top\lambda+E^\top\eta.
\]

Then weak duality gives the valid lower bound

\[
\boxed{
L_{\mathrm{LP}}=
-\lambda^\top b-\eta^\top e+
\sum_i U_i\min(r_i,0).
} \tag{12}
\]

Proof: \(c^\top u+\lambda^\top(Au-b)+\eta^\top(Eu-e)\le c^\top u\) on the feasible set; minimize that affine expression over the box. The residual term accounts for imperfect stationarity.

Use the solver's marginals as proposals: for SciPy's convention, try
\(\lambda=\max(-\mathrm{ineqlin.marginals},0)\),
\(\eta=-\mathrm{eqlin.marginals}\). Verify the convention on exact tiny LPs [R10]. Treat these finite floats as exact rational candidates for the verifier.

**Recompute (12) against the intended mathematical LP**, including exact rational grid, cuts and moments and outward exponential/CDF-radius enclosures. Verifying only the rounded matrix passed to HiGHS is not enough. The solver may use ordinary float64 coefficients: the certificate remains valid because any nonnegative multipliers are admissible and the verifier evaluates the intended problem.

Implement outward arithmetic with the standard-library Decimal module at 80 digits initially. Its exp and ln are documented as correctly rounded with half-even rounding [R11]; enclose an endpoint by next_minus/next_plus in the same context. Other interval operations and rational conversions require directed rounding. For the square-root radius, verify bracketing endpoints by rational squaring. Do not treat arbitrary-precision agreement alone as interval certification. Avoid noninteger Decimal.power.

Use 120 digits only on a bounded verification retry. If the interval cannot establish the required inequality, return an uncertified status. Serialize the lower endpoint downward, preserving enough digits to reproduce it. Include the proof, coefficient reconstruction, and a certificate replay command in the repository.

### 4.5 Original-set feasible witnesses and upper bounds

An outer-grid law is not an upper witness for the original intersection. Obtain a separate candidate by solving the same master with **exact** moments on the same grid. It may use a slightly tightened CDF band, radius
\(\max(0,\epsilon_{\mathrm{lo}}-10^{-8})\), solely to leave numerical repair room. This is an inner restriction; failure does not prove infeasibility.

Repair a candidate using exact rational arithmetic:

1. Select a deterministic pool of at most eight distinct positive-mass nodes, favoring large masses and separation. Enumerate at most 56 anchor triples.
2. Fix the non-anchor weights as nonnegative rational values derived from the candidate. Log any removal of tiny negative numerical weights; this is a proposed repair only.
3. Compute the residual exact mass and moments \(R_0,R_1,R_2\).
4. For anchors \(a,b,c\), set
   \[
   w_a=\frac{R_2-(b+c)R_1+bcR_0}{(a-b)(a-c)}
   \]
   and cyclic counterparts.
5. Accept only if all weights are nonnegative, total mass and both moments are exact, and every CDF constraint holds against the conservative original radius \(\epsilon_{\mathrm{lo}}\). Store the repaired law.
6. If no triple works, try the analytic moment extremizers only if they pass the same exact CDF checks. Otherwise report NO_WITNESS.

For the mean-only ablation, use two anchors and exact mass/mean repair.

A feasible witness \(\nu_{\mathrm{in}}\) gives \(C_J^*\le C(\nu_{\mathrm{in}})\). For a verified fold bracket \([t_l,t_h]\), a simple capacity enclosure is

\[
\delta t_l-\sum_jw_j\operatorname{expm1}(g_jt_h)
\ \le C(\nu_{\mathrm{in}})/Q\le\
\delta t_h-\sum_jw_j\operatorname{expm1}(g_jt_l). \tag{13}
\]

Use outward arithmetic and tighten the bracket. Also retain the analytic upper bound \(C_{\mathrm{mom}}^U\) for any nonempty intersection. Take the smallest verified applicable upper bound.

A releaseable joint result must have both a verified lower certificate and a verified original-set witness. This deliberately conservative release rule makes nonemptiness reviewable. The raw lower certificate can still be retained when witness construction fails.

### 4.6 Scalar capacity solver

Implement a float64 reference using log-sum-exp for
\(\log\sum_jw_jg_j e^{g_jt}-\log\delta\), with zero-weight/zero-node terms excluded. Bracket monotonically, then use Brent's method or bisection. Impose iteration and domain limits.

Use at most 64 bracket expansions, 128 float64 root iterations, and 256 interval-bisection iterations, subject to the scenario deadline. Every standalone law, including the DKW extremal law, gets a bracket from its **own** moments or a monotone sign search. The hybrid common bound depends on the supplied exact moments and must not truncate a DKW-only law that violates them. Ignore or clamp a baseline fold outside the hybrid domain only when proposing a hybrid cut, never when computing the baseline itself.

The independent verifier uses interval arithmetic to bracket the derivative sign and evaluates (13), rather than trusting a residual or root tolerance. Near \(t=0\), use expm1 and compensated summation; the Decimal verifier supplies a separate check against cancellation in \(\delta t-\sum w\,\operatorname{expm1}(gt)\). Never evaluate an overflowing exponential merely because the root was located in the log domain. Handle initial instability, zero feedback, invalid weights and tiny stability margins explicitly.

Cache outward exponential coefficients by exact grid-node, cut and precision identifiers; reuse them across master verifications and refinements. Charge their construction time and memory. Store replayable sparse multipliers, grids/cuts and rational inputs, not repeated dense high-precision coefficient matrices. Use compressed result files where helpful; any dropped nonzero multiplier requires re-verification of the changed certificate. These measures address a real potential verification/preprocessing bottleneck, not an assumed speedup.

Accuracy targets for primary cases:

- scalar capacity enclosure width at most \(10^{-10}\max(C_{\mathrm{scale}},10^{-12})\), where \(C_{\mathrm{scale}}=\Phi/m_1\) in normalized \(Q=1\) units;
- master certificate rounding width at most the same amount;
- desired continuum bracket width
  \(\tau=0.005\,C_{\mathrm{scale}}\).

The last tolerance is a planning target for resolving a 0.10 normalized gain; always report its actual ratio to true capacity. A wider bracket remains valid but cannot be described as a sharp joint solution. Near-degenerate stress fixtures use their own high-precision relative check; do not silently pass them using an absolute tolerance larger than their capacity.

## 5. Execution details, limits, and failure behavior

### 5.1 Fixed algorithmic limits

Use nested mesh sizes \(J=256,1024,4096,16384\), where the uniform component has \(J+1\) points. Preserve sample and previously added nodes across refinement. No full data-by-grid dense intermediates are needed.

Use SciPy highs-ds initially, with primal and dual feasibility tolerances \(10^{-9}\). Validate availability/options in calibration. A switch to highs-ipm is allowed only as the one documented numerical fallback for a failed solve, consumes an LP slot and the same time allowance, and must be recorded.

Per scenario:

- at most four mesh levels;
- at most 32 distinct price cuts per master;
- at most 96 total LP calls across both outer and inner problems and all mesh levels;
- at most one 120-digit verification retry;
- at most 56 anchor triples per candidate, with at most two candidate repairs per mesh level;
- LP time limit equal to the smaller of 5 seconds and the remaining scenario allowance, and complete-scenario limit 30 seconds. Check the deadline between root iterations, verification batches and anchor repairs.

Calibration may lower the mesh ceiling or select a uniform scenario limit of 15 or 30 seconds **before freezing**, using costs only. It may not change the mathematical tolerances or silently treat a looser bound as a solved instance. Do not raise these caps during confirmation. A 60-second scenario limit is permitted only as a preregistered tiny diagnostic, not as a hidden main-study rescue.

Allocate at most 24 LP calls per mesh level, in at most 12 outer/inner pairs. A numerical fallback consumes one of these calls. The outer call comes first, then the inner call; if the inner problem returns no candidate, do not repeatedly solve the same infeasible/failed inner system at that mesh without a specific numerical repair. Try at most two witness repairs per mesh: the first usable candidate and, on mesh exit if time remains, the retained candidate with the smallest nominal scalar capacity (ties go to first appearance). Nominal ranking only chooses a candidate to validate. A valid witness found earlier remains usable.

Use a shared set of at most 32 distinct price cuts for both masters. Add a candidate fold cut when its nominal master violation exceeds \(\tau/8\); nominal gaps guide work allocation only and have no certificate status. If two consecutive pairs add no cut, the cut limit is reached, or the 24-call mesh allocation is exhausted, advance to the next mesh unless the verified original gap already meets \(\tau\). Stop immediately when that verified gap meets \(\tau\). Preserve the strongest verified lower and smallest verified original upper across refinements. No hidden extra call allocation is available after the fourth mesh.

### 5.2 Pseudocode

~~~text
solve_joint(sample, A, m1, m2, D, Q, alpha, fixed_limits):
    validate inputs; compute the outward epsilon bracket
    compute verified separate bounds, including boundary branches
    B = max(their normalized lower endpoints)
    for a unique moment law, test its exact CDF-band membership:
        if outside the band, report inconsistent-band fallback, not a joint value
        if inside, return its verified capacity with the unique-law certificate
    handle any remaining zero-capacity boundary before nondegenerate normalization
    normalize to rational y, mean, second moment, delta
    compute the common time-domain upper bound
    raw_lower = 0; original_upper = moment_upper
    best_witness = none
    initialize price cuts from 0, common upper endpoint, baseline folds

    for J in frozen mesh sequence:
        form exact nested grid including all sample knots
        construct sparse outer master using rounding moment envelopes
        construct inner master using exact moments and tightened CDF radius

        for at most 12 outer/inner pairs, respecting 24 calls per mesh:
            solve outer master in float64
            if usable marginals:
                replay mathematical dual bound with outward arithmetic
                raw_lower = max(raw_lower, verified dual lower, 0)
            add a distinct bounded cut from candidate fold when available

            solve inner master after the outer call, subject to the fixed skip rule
            repair/validate the first usable candidate
            retain the best nominal-capacity candidate, ties to first appearance
            if verified feasible:
                compute capacity enclosure
                update best witness and original_upper
                add its maximizing cut

            if best_witness exists and
               original_upper - max(B, raw_lower) <= target_gap:
                return released result with full certificates

            if two pairs add no new cut, or cut/mesh-call allowance is exhausted:
                break
        on mesh exit, use the second repair slot on the best retained candidate
        if time remains; update verified upper and witness only after validation

    if best_witness exists and raw_lower is certified:
        return max(B, raw_lower), upper, status VALID_WIDE or VALID_TIGHT
    else:
        return B, diagnostic upper if applicable, status BASELINE_FALLBACK
        retain raw certificates and exact reason for no joint release
~~~

Any strictly negative verified gap \(U<L\), beyond outward arithmetic, is a correctness failure. Stop interpreting results and investigate within the repair budget.

### 5.3 Baselines and ablation

| Method | Information | Role |
|---|---|---|
| Mean-and-cap formula | \(m_1,A,D,Q\) | Simple reference and information ceiling context. |
| Sharp two-moment bound | \(m_1,m_2,A,D,Q\) | Strong analytic constituent. |
| Exact DKW envelope | Sample, \(A,D,Q,\alpha\) | Strong sample constituent. |
| Maximum of the last two | All primary information | Primary strong comparator. |
| Joint outer/inner certificate | Exactly the same primary information | Proposed computational procedure. |
| Joint mean-only variant | Same sample/cap/depth, only \(m_1\) | Targeted second-moment ablation. |
| True-law capacity | Full population, evaluator only | Correctness oracle, never a deployable comparator. |

Run baselines with the same numerical-verification standard. No tuning on confirmatory outcomes is allowed.

**Mean-only ablation contract:** its fallback is \(B_1=\max(C_{\mathrm{mean}},C_{\mathrm{DKW}})\), its universal upper bound is \(\Phi/m_1\) before normalization, and its time bound is the first term of (8) only. It gets no \(m_2\), no second-moment-derived grid nodes, no second-moment feasibility checks, and no second-moment-derived tolerance. Use the common mean-only scale \(C_{\mathrm{scale}}=\Phi/(Qm_1)\), the same nominal meshes and wall limits, and exact mass/mean witness repair. The principal descriptive ablation contrast is \(J_{\mathrm{two\ moments}}-J_{\mathrm{mean\ only}}\), normalized by true capacity. Also report each method's gain over its own separated fallback; those are different contrasts.

MinMaSS is conceptual prior art, not a fair numerical comparator for this different target. Volatility, momentum, credit spreads and macro stress indexes do not estimate the same conditional capacity. No claim of outperforming them is made. Comparing their published forecast scores with this synthetic certificate experiment would be invalid.

### 5.4 Interfaces and information firewall

Implement ordinary Python interfaces equivalent to:

~~~text
capacity(law, D, Q) -> CapacityInterval
moment_bounds(m1, m2, A, D, Q) -> Bounds
dkw_worst(sample, A, alpha) -> LawWithRadius
solve_joint(public_case, limits) -> JointResult
verify_certificate(case, certificate) -> VerificationResult
generate_bundle(law_id, n, seed) -> public_case + evaluator_truth
evaluate(results, evaluator_truth, frozen_protocol) -> tables
~~~

The optimizer must not receive a law identifier, true support, true probabilities, true capacity, or held-out generator description. Pass only numerical public-case inputs to solving functions. Keep truth in a separate evaluator file/module. Numerical tolerance and timing settings must not depend on realized comparative gains.

Result fields must include scenario/bundle IDs, method, all public input identifiers, status, lower/upper normalized capacity, optional share capacity, raw lower, baseline fallback, mesh size and maximum gap, cuts, LP calls, primal residuals, certificate correction magnitude, witness hash, root bracket, solve/certification/total time, peak RSS, and termination/failure reason. Represent infinity or undefined values with explicit statuses and null fields, not nonstandard JSON NaN/Infinity numbers.

### 5.5 Runtime accounting

Primary timing is input-to-result time, including grid construction, LP setup, scalar solves, witness repair, verification and serialization. Also record those components. Record a separate cold interpreter/import/setup cost and warmed scenario times. If sample preprocessing is reused across two depth ratios, report complete bundle time plus the stated amortization; do not make preprocessing disappear.

Use CPU float64 for solver proposals and CPU Decimal for verification. No GPU, MPS, CUDA, or remote acceleration is needed. Use one worker for primary measurements; cap BLAS/OpenMP threads consistently. Record the actual thread settings and backend. Timing uses a monotonic clock. Measure whole-process peak RSS, including child processes if any; on macOS the units of resource.getrusage differ from some other systems, so record conversion explicitly and cross-check with psutil.

Every executed correctness or experiment command, including pytest and ad hoc development probes, must run through the budget ledger wrapper. Nested timed commands are charged once for elapsed wall time, not double-counted. Installation/import setup is reported separately only where explicitly allowed; computational verification is never excluded merely because it ran during development.

Do not repeat deterministic correctness fixtures to create artificial sample sizes. A handful of cold/warm timing repeats belongs only in calibration. Preserve failed calls and their costs.

### 5.6 Failure and missing-run policy

| Condition | Required behavior |
|---|---|
| Invalid cap, unsupported sample, impossible moments, nonpositive \(Q,D\), nonfinite values | Reject the scenario as invalid input; do not fabricate a bound. Such deliberate stress fixtures are not primary population draws. |
| Initially unstable or zero-feedback law | Return the explicit mathematical boundary status; not a numerical failure. |
| LP time limit, nonconvergence, unavailable marginals, or inability to certify a new candidate | Preserve diagnostics and every earlier verified result. Within the bounded repair allowance retry only the predeclared alternative. If a verified lower plus original-set witness already exists, return its best pair as VALID_WIDE (or VALID_TIGHT) with the termination reason. Return \(B\), with zero operational incremental gain and BASELINE_FALLBACK, only when no such pair exists. |
| No original-set witness | Preserve raw lower bound but release \(B\); label NO_WITNESS. A computational failure to construct a witness is not proof of an empty continuous set. |
| Solver says infeasible | First distinguish numerical failure from a mathematically checked contradiction. Do not call it proven continuous infeasibility without an exact/verified certificate. Preserve an earlier valid pair if one exists; otherwise release \(B\) without a joint-set value. A claimed infeasibility certificate contradicting an existing valid witness is a correctness defect to investigate. |
| Verified empty moment-band intersection | Record INCONSISTENT_BAND and release the separated fallback only. The ideal joint procedure is defined to fall back to \(B\) on this event; never use the empty-set \(+\infty\). |
| A released lower bound exceeds known true capacity on a verified true-law-in-band event | Invalid experiment. Stop comparative interpretation, preserve evidence, repair within the global allowance, and rerun every affected result. |
| Budget stops a scheduled main run | Count it as unfinished, not as an adverse scientific result. Report the balanced partial data and classify the planned comparison as inconclusive unless a separate verified analytic bound already resolves it. |

A fallback is still a recorded algorithm outcome, not an exclusion. Coverage guarantees apply under the specified model and sampling assumptions; an inconsistent-band event is outside the true-law-in-band event if the supplied exact moments are correct.

The preservation rule applies to local solve failures and deadlines, not to evidence that a previously accepted certificate is wrong. A proven certificate/implementation defect invalidates every affected result until repaired and reverified. Unlaunched cases after a global stop remain unfinished; do not manufacture fallback runs for them.


## 6. Data generation and the locked experiment

### 6.1 Exact synthetic populations

Use four finite rational populations. They resemble ordinary broad, centrally concentrated, U-shaped and coarse exposure distributions, but are **not fitted U.S. market populations**. Their purpose is mechanism isolation and exact evaluation.

The estimator's ambiguity class still contains every probability distribution on \([0,A]\). The finite true populations are hidden evaluator fixtures, not a license to restrict the adversary to their support.

| ID | Exact law for \(a\) | Integer probability construction | \(m_1\) | \(m_2\) |
|---|---|---|---:|---:|
| U65 | \(a=k/32,\ k=0,\ldots,64\) | All 65 integer weights equal 1; denominator 65. | 1 | \(43/32\) |
| TRI65 | \(a=k/32,\ k=0,\ldots,64\) | Weights \(k+1\) for \(k\le32\), \(65-k\) for \(k\ge33\); denominator \(33^2=1089\). Equivalent to the sum of two independent discrete uniforms on \(0,\ldots,32\), divided by 32. | 1 | \(113/96\) |
| UEDGE65 | \(a=k/32,\ k=0,\ldots,64\) | Weight \(\binom{2k}{k}\binom{128-2k}{64-k}\); denominator \(4^{64}\). A symmetric beta-binomial construction with parameters \(64,1/2,1/2\). | 1 | \(193/128\) |
| ATOM3 | \(a=0,1,2\) | Integer weights \(1,1,1\); denominator 3. | 1 | \(5/3\) |

Verify normalization and both moments exactly with Fraction before any numerical experiment. The formulas and moment identities were checked during planning; the coding agent must independently reproduce these tiny checks.

Sample integer node IDs using a uniform integer in the exact denominator range and cumulative integer weights, for example Python random.Random(seed).randrange(denominator). This avoids silently moving probability atoms when converting rational nodes to floats. Preserve node values exactly and normalize by exact \(A\). Use floats only for solver proposals.

Generate seeds from SHA-256 of a documented string containing the stage, law ID, sample size, and replicate number. Suggested distinct namespace strings are fragility-dev-20261102, fragility-calibration-20261102, fragility-pilot-20261102, and fragility-confirmation-v1-20261102. Store the resulting integer seeds and generator/version details. Do not use Python's process-randomized hash().

Save each sample bundle once. The same bundle is reused across its depth and cap scenarios. Separate cases are not independent observations merely because their depth differs.

### 6.2 Primary matrix

Fix \(\alpha=0.05\), \(Q=1\), \(A=10\), and \(D=r\,m_1\).

| Axis | Values |
|---|---|
| Population | U65, TRI65, UEDGE65, ATOM3 |
| Sample size | \(n=128,512\) |
| Depth ratio | \(r=6/5,2\) |
| Independent bundles per law/sample-size stratum | \(B_{\mathrm{rep}}=24\), or the resource-selected 16 or 8 |
| Main scenarios | \(16B_{\mathrm{rep}}\) |
| Actual independent sampling units | \(8B_{\mathrm{rep}}\) bundles |

The parameter choices are intended to examine a moderately small stability margin and a more comfortable one, with finite sample sizes that an institutional aggregate-sharing pilot could plausibly process. They are scenario assumptions, not empirical claims about institutional feeds.

The primary mechanism is complementary information: moments limit how much mass can be moved to high leverage, while the CDF band constrains where the compensating bulk mass can move. The effect should be weak when moments already identify a nearly unique law, or the cap is tight. It may also shrink when a very accurate sample makes the DKW-only bound nearly sufficient. Do not require gains to increase monotonically with sample size.

### 6.3 Secondary controls and ablation

Use the same first four confirmatory bundles in every law/sample-size stratum for both tighter caps \(A=2,4\) and both depth ratios: **128 secondary scenarios**.

Use the same first six bundles per stratum at \(A=10\), both depth ratios, for the mean-only joint ablation: **96 scenarios**.

| Chosen replicate count | Primary | Cap controls | Ablation | Total planned comparative scenarios |
|---:|---:|---:|---:|---:|
| 24 | 384 | 128 | 96 | 608 |
| 16 | 256 | 128 | 96 | 480 |
| 8 | 128 | 128 | 96 | 352 |

Controls and ablation are descriptive mechanism checks; four or six bundles do not support precise subgroup claims. Do not substitute a successful control subgroup for an unsuccessful primary test.

Before allocating LP work, compute the simple available-information ceiling from the moment-only lower/upper bounds. If the separated lower bound is already close to the moment upper bound, record that little additional improvement is mathematically possible. This can save work on controls without creating a favorable-case selection rule: retain every scheduled row and record the analytic enclosure.

### 6.4 Independent correctness checks

Implement these before comparative interpretation.

1. **Closed-form scalar cases.** For \(\delta_a\), \(a>0\), use
   \[
   x^*=\log(D/(Qa))/a,\quad
   C=[D\log(D/(Qa))-D+Qa]/a.
   \]
   Verify the endpoint-mixture formula, zero feedback, and initial-instability branches.
2. **Notebook anchors.** For \(\mu=(\delta_0+\delta_1+\delta_2)/3\), \(D=1.5,Q=1\), reproduce \(x^*=0.2395605578388631\), \(C=0.06422573132920234\). With \(A=2,m_1=1,m_2=5/3\), reproduce the capacity interval approximately \([0.0636150764352122,0.0649185972973480]\).
3. **Tiny independently soluble LP.** In a declared finite-support test on \(\{0,1/3,2/3,1\}\), empirical weights \(1/4\), exact mean \(1/2\), second moment \(7/18\), and fixed CDF radius \(1/10\), all feasible weights are
   \[
   (1/4-q,\ 1/4+3q,\ 1/4-3q,\ 1/4+q),\quad |q|\le1/20.
   \]
   For positive \(t\), their exponential expectation increases with \(q\), since its coefficient is \((e^{t/3}-1)^3\). Thus at \(\delta=3/4\) the robust capacity is attained at \(q=1/20\). Compare the finite-support master/certificate with this independently derived result. This test does not assert that finite support is the continuous-support ambiguity set.
4. **Continuous feasible fixture.** Use Uniform\([0,2]\), exact \(m_1=1,m_2=4/3\), \(A=4,D=1.5,Q=1\), and the 20 midpoint quantiles \(a_i=(2i-1)/20\). Their CDF discrepancy is \(1/40\); set a deterministic radius \(1/20\). Check the true continuous capacity via \(M(x)=\operatorname{expm1}(2x)/(2x)\), with the limit at zero and independent high-precision quadrature. The certificate must not exceed it. This fixture is not IID sampling and makes no DKW-coverage claim.
5. **Rounding lemma checks.** On small rational laws whose atoms lie between mesh points, verify exactly the CDF identities and moment-inflation inequalities and compare rounded versus original capacities. Include observed atoms at 0 and \(A\), repeated observations, and unequal close rational knots.
6. **Dual sign and residual controls.** Use tiny LPs with analytically known optima. Deliberately perturb a returned multiplier, coefficient, or marginal sign. The verifier must penalize residuals or reject the proposed certificate; it must never accept a spurious higher lower bound.
7. **Inner witness checks.** Deliberately perturb mass, mean, second moment, or a CDF prefix. The exact validator must reject each invalid witness. No silent clipping establishes feasibility.
8. **Positive complementarity control.** Use the deterministic balanced sample of 40 observations at each of 0,1,2, \(n=120,A=10,D=1.5,Q=1,m_1=1,m_2=5/3,\alpha=0.05\). Demonstrate a verified improvement exceeding \(10^{-4}\) in normalized capacity over \(B\), or identify the specific numerical obstruction. This detects whether the calculation can show the expected effect; it is excluded from confirmation.
9. **Null controls.** A point-mass law with its exact first two moments leaves no room for information gain. Check that the joint interval agrees with its known capacity and does not manufacture improvement.
10. **Assumption controls.** Document the analytic one-sided-policy counterexample: shares that remain fixed after a gain yield \(H(x)=Dx\), not exponential induced buying. Also reproduce the source's \(n=1\) finite-round derivative \(D-Qm_1e^{-x}>0\). These refute unqualified transfer of the continuous-model interpretation; they are not competing estimates of the primary target.

Use no more than 80 tiny correctness scenario attempts. Exact rational identities need one evaluation, not statistical repetitions. Inspect the numerical output and failed controls; passing exit codes alone are insufficient.

### 6.5 Targeted stress checks

Reserve at most 32 small stress scenarios, selected before confirmation:

- stability ratios approaching one, including \(1+10^{-6}\), using high precision;
- degenerate/near-degenerate moment pairs;
- rare high-leverage tails within a verified cap;
- three explicit uncapped hidden-tail instances from notebook §4, as a conditional no-cap diagnostic only;
- sample ties/endpoints and off-grid rational mass;
- falsely specified caps or moments, to test rejection;
- one deterministic biased-sampling or dependent-duplicate example showing that the IID guarantee cannot simply be relabeled.

These are robustness and assumption tests, not additional observations supporting the primary effect. Do not run a full economic simulator or a finite-round benchmark. Do not treat successful synthetic tests as validation of actual investor behavior.

## 7. Analysis, uncertainty, and interpretation rules

### 7.1 Quantities to retain

For every scenario record:

- verified separated bound \(B_i\);
- raw joint lower, released lower \(J_i\), verified upper \(U_i\), and their gap;
- true capacity interval \(C_i\), computed by the evaluator only;
- operational gain \(g_i=(J_i-B_i)/C_i\);
- acquisition-value secondary \((J_i-C_{\mathrm{mom},i})/C_i\);
- bound tightness \(J_i/C_i\), non-vacuity, and each constituent's tightness;
- failures, fallbacks, infeasibility/ambiguity status, cost, and numerical correction magnitude.

Use interval endpoints when computing effect enclosures. All primary true capacities are strictly positive. Do not remove large relative errors in near-zero stress cases; give absolute and relative errors and distinguish them from the primary matrix.

A method returning fallback \(B_i\) has operational gain zero and remains in the denominator. The exact maximization in the release rule ensures nonnegative operational gain; that fact is not evidence of a contribution.

For information-value diagnosis define the ideal procedure to return \(C_J^*\) when the intersection is nonempty, and \(B\) when it is empty. An applicable original-set witness gives an upper bound on its gain. Without a witness, the universal moment upper bound supplies a conservative ceiling:
\[
g_i^{U}=\frac{\max(B_i,C_{\mathrm{mom},i}^{U})-B_i}{C_i}.
\]
Use a tighter verified witness upper when available. This ceiling also covers the empty-set fallback. A failure to close the continuum gap cannot be converted into evidence that the information has no value.

### 7.2 Primary estimand and uncertainty

Within each bundle, average \(g_i\) across its two depth ratios. Compute the mean within each of the eight fixed population/sample-size strata, then average those eight means equally. That is the primary estimand.

Use 10,000 stratified bootstrap resamples of **whole bundles within each stratum** for an approximate 95% interval. Reuse the same resampled bundle indices for lower- and upper-gain analyses. Do not bootstrap individual depth/cap rows as independent. Do not resample the four chosen populations as if they were randomly drawn markets.

Report every cell, the aggregate, and raw bundle gains. The planned interval-halfwidth target is 0.025 relative to the 0.10 threshold; the chosen replication counts do not guarantee it. At eight bundles per stratum, explicitly qualify bootstrap accuracy. A wide interval is inconclusive, not permission to keep sampling until the result is favorable.

The only primary confirmatory claim concerns the fixed average. Cap comparisons, sample-size patterns, ablation and rare-tail results are secondary/descriptive. If one of them motivates a changed hypothesis, label that exploratory and require a future fresh study.

Prespecify an evaluator-only robustness gate: recompute the gain average with outside-band or ambiguous-band bundle contributions set to zero, retaining the original denominator. Use identical stratified bootstrap indices. A supported conditional information-value contribution requires both this conservative sensitivity and the raw primary lower interval to exceed 0.10. The sensitivity is not a deployable algorithm or an extra confidence guarantee. If the raw primary passes but this gate does not, report the useful-certification claim as inconclusive and show the paired decomposition; this alone does not prove that the result depends on confidence misses. Do not describe the two intervals as jointly calibrated 95% intervals.

### 7.3 Coverage and correctness are separate from effect size

For each exact finite population, compute
\(\sup|F_{\mathrm{true}}-\widehat F_n|\) exactly at the union of true and sample atoms, including left and right limits. Compare against the outward DKW-radius bracket. An ambiguous equality gets a boundary status, not an invented membership result.

On every verified true-law-in-band event, a released lower bound must lie below the true capacity within the certified interval arithmetic. A violation invalidates the numerical study even if average coverage looks good.

Outside-band observations remain in all effect and coverage denominators. Their failures do not refute the DKW theorem. Report the separate contributions to gain from inside-band, outside-band and ambiguous bundles, including the prespecified zero-outside sensitivity. Report lower-bound coverage and the true-law-in-band rate with bundle-level uncertainty; the six possible cap/depth scenarios share a sampling event and are not six independent trials. Do not require each finite sample's observed coverage to be at least exactly 95%.

Any simulated coverage result only checks for bugs and gross misspecification. The finite-sample guarantee comes from the mathematical event argument and valid numerical enclosures.

### 7.4 Outcome classification

| Outcome | Evidence required and allowed conclusion |
|---|---|
| **Supported contribution** | All validity gates pass, the complete selected primary manifest is accounted for, and both the raw primary and prespecified zero-outside sensitivity lower 95% effect intervals exceed 0.10 within budget. Report a conditional computational information-value contribution for the specified synthetic regime. |
| **Informative negative: little additional information value** | Valid computation or analytic ceilings give an upper 95% interval for the ideal gain below 0.10. Upper enclosures must include unresolved optimization/discretization error. Conclude the strong separated bounds suffice to the chosen resolution in this regime. |
| **Informative negative about the implemented approach** | The upper endpoint of the prespecified 95% interval for operational gain is below 0.10, but an ideal-gain ceiling remains wide. State that this bounded implementation did not deliver useful improvement; the information-value question remains unresolved. |
| **Practical infeasibility** | Measured scaling or resource limits prevent even the minimum meaningful protocol. Identify whether the obstruction is solver cost, certificate verification, witness construction, precision, memory, or setup. Do not call a timeout a refutation of the mathematics. |
| **Inconclusive evidence** | The effect interval crosses 0.10, the continuum gap is too broad, too few independent draws fit, necessary target hardware is unavailable, or the planned run is incomplete. |
| **Invalid experiment** | An inside-band bound violation, baseline defect, true-support leakage, inappropriate weighting, wrong CDF handling, or invalid measurement remains unrepaired. Preserve the attempted results but make no comparative claim. |

A useful positive result does not justify starting a public-data crash predictor. The next institutional step would be to verify the actual observation and rebalancing model. A strong negative may justify using the separated analytic/sample bounds. Identify at most one concrete next research step warranted by the result.

## 8. Mac feasibility, cumulative budgets, and evidence gates

### 8.1 Inspect the actual target first

Run and record the equivalents of:

~~~bash
uname -m
sw_vers
system_profiler SPHardwareDataType
sysctl -n hw.memsize
df -h .
python3 --version
~~~

Confirm the actual machine is a MacBook with an M3 Max. Do not infer RAM, free storage, macOS version or package/backend support from the chip name. Record available memory with psutil, CPU architecture, power mode if observable, Python architecture, NumPy/SciPy versions, BLAS details, HiGHS availability, Decimal backend, and thread limits.

The planning calculations were performed in a separate Linux tool environment, using NumPy 2.3.5 and SciPy 1.17.0. They establish no M3 Max performance.

Use a project-local virtual environment and CPU libraries. If the actual Mac is unavailable, finish derivations, fixtures, code and portable correctness work that is authorized and possible. Mark all M3 timing and feasibility conclusions unperformed; do not substitute another machine's benchmark.

### 8.2 Resource ceilings

These limits apply cumulatively, including failed attempts, development trials, tuning, repairs, repeats and fallback branches. A new stage does not reset them.

| Resource | Ceiling / rule |
|---|---|
| Total executed experimental computation | **240 minutes** wall time, with one primary worker. Includes preprocessing, correctness scripts, all solver calls, certification, diagnostics and report-generation computation. |
| Informative pilot | Target within **50 cumulative compute minutes** from the first executable check; typical cost may be lower, but is unmeasured. |
| Peak task RSS | Minimum of **6 GiB**, 25% of installed RAM, and 60% of memory available at start. If this is below 1 GiB, run only tiny diagnostics before reassessing feasibility. |
| Task-owned disk | **5 GiB** including environment/download cache; results/checkpoints target below 1 GiB. Do not begin larger work without this space plus reasonable system headroom. |
| Total scenario attempts | **1,000**, including correctness, pilot, calibration, secondary work and repairs. Root iterations/LP calls are additionally capped. |
| Total LP calls | **100,000**, including failed calls and retries. The wall-clock ceiling takes precedence. |
| Repairs | At most **25 compute minutes**, 120 scenario attempts, and two repair cycles for any one material defect; all consume the same global totals. |
| Reporting reserve | Protect **30 compute minutes** and at least 30 minutes of human/agent analysis effort. Stop launching new experiments before consuming it. |

Budgeted computation is distinct from engineering effort. A planning estimate is 30–60 minutes of setup, 6–12 hours of implementation and review, and 1–2 hours of analysis/writing, plus up to four hours of executed experiments: roughly **8–19 active hours**, depending on repairs. These are unmeasured effort estimates. Limit implementation/review to 14 active hours; if numerical certification is still incomplete, finish a scoped feasibility report instead of expanding the project.

Normal dependency and source downloads are setup, not remote experimental computation. Record their elapsed time separately. Do not use hosted compute, paid solvers, proprietary feeds, CUDA-only packages or large financial datasets.

### 8.3 Calibration and scaling

Calibrate on 12 representative scenario attempts, including ATOM3 and UEDGE65 at \(n=512,A=10,r=6/5\), both outer and inner work, and at least two mesh sizes. Include a complete attempt at the **largest mesh permitted for confirmation**, including coefficient verification and witness repair. If that largest mesh cannot be measured within calibration limits, freeze the ceiling at the largest actually measured size. Include difficult witness repair and verification, not just a warm scalar root.

Measure complete scenario cost, peak RSS, matrix nonzeros, cut count, support size and certificate time. With \(M\) knots and \(K\) cuts, stored coefficients should scale approximately as \(O(MK+M)\); dense CDF matrices would instead create avoidable \(O(M^2)\) storage. LP factorization can exceed matrix storage, so extrapolate from actual peak memory and costs, not just array sizes.

Forecast the remaining matrix with a safety factor of at least 1.5 applied to representative measured costs. Choose the largest replicate count in \(\{24,16,8\}\) that fits the **remaining** locked-comparison allocation and global reserves. Select using costs, not observed gains. Freeze the selected count, limits, tolerances and order before generating confirmatory samples.

The largest comparative matrix has 608 scenarios; adding the maxima of 80 correctness, 12 calibration, 32 pilot, 32 stress and 120 repair attempts gives 884, within the 1,000-attempt ceiling. The corresponding LP cap also fits the nominal call-count allowance, but worst-case time limits cannot all be spent: time, memory and remaining repair reserves take precedence. For the full 120-minute comparison allocation, the 1.5 safety factor requires measured complete-scenario averages of roughly 7.9, 10.0 or 13.6 seconds for the 608-, 480- or 352-scenario designs. Recompute using actual remaining time and heterogeneous measured costs.

If eight bundles per stratum do not fit, do not reduce the four populations, sample sizes, depth regimes or meaningful effect threshold to manufacture a result. Stop the full comparison as practically infeasible. A smaller descriptive pilot may be reported as such.

A watchdog must stop new work before a projected time or memory overrun. If RSS reaches 90% of its cap, checkpoint and stop the responsible scenario. Do not silently switch to float32, drop CDF knots, coarsen tolerances or ignore verification.

### 8.4 Ordered evidence gates

Stage budgets below sum to 240 compute minutes. They are ceilings, not targets to exhaust. Unused time may move to a later stage only through the remaining global budget; the protected reporting reserve remains intact.

| Gate | Question / prerequisites | Minimum work and evidence | Compute budget | Continue, repair, fallback, or stop |
|---|---|---|---:|---|
| G0 Constructive feasibility | Are the mathematical assumptions and local resources explicit? | Environment record; exact input contract; derivations of rounding and dual bounds; dependency smoke check; no required inaccessible data. | 5 min | Continue if construction and local stack are usable. If only hardware is unavailable, complete portable work and mark target validation blocked. An essential mathematical defect triggers analytic-stop reporting. |
| G1 Tiny correctness | Does the implementation have trustworthy expected behavior? | Scalar anchors, exact population identities, tiny LP oracle, CDF/tie tests, witness rejection, interval/dual controls, continuous fixture. | 12 min | All central invariants must pass. At most two bounded repair cycles; no empirical interpretation while a certificate defect persists. |
| G2 Resource calibration | Can meaningful instances fit? | Representative costs, memory/nonzero counts, scale forecast, chosen limits and tentative replicate count. | 8 min | Continue if minimum design fits with 1.5 safety factor and reserves. Otherwise retain evidence and stop as practical infeasibility. |
| G3 Mechanism pilot | Can the design detect complementarity and distinguish numerical looseness? | Positive and null controls; at most 32 fresh development scenarios; interval widths, witness rate and baseline ceilings. | 25 min | Positive control must show verified gain, null must not. A noisy or small ordinary-case gain is not rejection. Repair only a specific numerical/design defect within allowance; do not change primary populations or threshold to chase gain. |
| G4 Locked comparison | Does useful improvement survive the strong baseline? | Freeze protocol/hash, generate fresh bundles, execute balanced complete blocks and secondary manifests, retain all outcomes. | 120 min | No outcome-driven tuning or early favorable stopping. If a material defect appears, invalidate affected results and use bounded repair; any scientific protocol change makes a later run exploratory unless fresh confirmation fits. |
| G5 Diagnosis / limited robustness | Which assumption or numerical limitation explains the result? | At most 32 predeclared stress cases; bracket/error decomposition; failure review. | 15 min | Run only checks addressing a concrete remaining interpretation question. No open-ended rescue campaign or new benchmark family. |
| Repair reserve | Can a specific defect be corrected without changing the scientific question? | Versioned defect record, fix, affected-case list, regression check, rerun ledger. | 25 min | Two cycles per defect maximum; count failed and replacement attempts. Stop invalid/inconclusive if exhausted. |
| G6 Report and reproducibility | What is supported, and can another person regenerate it? | Tables, essential plots, report, raw results/certificates, one-command verification and regeneration. | 30 min | Complete for whatever stages were reached, including early stops. Never spend the reserve on a larger experiment. |

Do not build later-stage infrastructure before the preceding gate passes. If a verified analytic obstruction resolves a question, report the derivation and minimal diagnostic; an unnecessary large benchmark is not required.

Run confirmation in complete replicate blocks across all eight law/sample-size strata, with both depth ratios. Complete the primary matrix first, then the secondary cap controls and ablation. Within blocks, use a fixed balanced order. An unplanned interruption must not preferentially retain easy or favorable populations. Preselected lower replication from calibration is valid; unplanned truncation of the frozen primary manifest is not a completed confirmatory study. Missing secondary work prevents the corresponding secondary claims but does not invalidate an otherwise complete primary result; report the shortfall explicitly.

## 9. Minimal implementation and commands

### 9.1 Stack and repository

Use Python 3.11–3.13 if available natively on the actual Mac. Inspect first; do not install or upgrade system Python silently. Dependencies: NumPy, SciPy/HiGHS, mpmath for an independent high-precision cross-check, Matplotlib, psutil, pytest. Fractions, Decimal, JSON, CSV, hashing and timing come from the standard library. No licensed optimizer, database, service or browser application is necessary.

Use current compatible package wheels verified on the target, then freeze the exact resolved versions. Do not cite the planning environment's versions as target requirements. If a required wheel/backend is unavailable, allow one documented compatible-version retry; otherwise report the dependency block and finish unaffected analytic work.

Suggested minimal layout:

~~~text
fragility-capacity-poc/
  README.md
  pyproject.toml
  requirements-lock.txt
  configs/
    base.json
    locked.json
  inputs/
    POC_EXECUTION_BRIEF.md
    source_manifest.json
  fragility_poc/
    __main__.py
    core.py
    bounds.py
    joint.py
    verify.py
    fixtures.py
    protocol.py
    report.py
  tests/
    test_math_and_certificates.py
    test_protocol_and_information.py
  results/
    environment.json
    budget_ledger.jsonl
    protocol_sha256.txt
    samples/
    truth/
    rows.jsonl
    certificates/
    figures/
    report.md
~~~

Module responsibilities: core handles laws and roots; bounds implements the established comparators; joint implements sparse masters and refinement; verify independently reconstructs intervals/dual/witness checks; fixtures owns exact generators and private truth; protocol owns manifests, seeds, freezing and budgets; report regenerates tables and plots.

Do not copy the notebook's permissive weight clipping or unbounded root loops. Its code is a reference, not a production-quality certificate implementation.

### 9.2 Intended commands to create

**The following project commands are interfaces you must implement; they have not been run or verified during preparation of this brief.** Standard shell setup commands are examples contingent on the inspected environment.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip freeze > requirements-lock.txt

python -m fragility_poc environment --out results/environment.json
python -m fragility_poc budget-run -- python -m pytest -q
python -m fragility_poc smoke --config configs/base.json
python -m fragility_poc calibrate --config configs/base.json
python -m fragility_poc pilot --config configs/base.json
python -m fragility_poc freeze --config configs/base.json --out configs/locked.json
python -m fragility_poc run --config configs/locked.json
python -m fragility_poc verify-results --config configs/locked.json
python -m fragility_poc report --config configs/locked.json
~~~

The budget-run wrapper and all project subcommands must update the same ledger, including test sessions, certificate replay and report generation, without double-counting nested wall time. The freeze command must verify prior gate evidence, select only the cost-authorized replication count, write an immutable configuration/manifest hash, and refuse to overwrite it silently. The run command must refuse an altered hash. A resume command may continue unfinished IDs without changing existing results; repairs append versioned replacements with links to invalidated rows.

A report-only rerun must regenerate essential figures/tables without solving any LP or regenerating samples.

### 9.3 Configuration contract

Include at least: study version, exact rational constants, alpha, population definitions, sample sizes, caps, depth ratios, stage-specific seed namespaces, selected replicates, secondary subset indices, meshes, cut/LP/root limits, Decimal precisions, requested capacity gap, confidence-analysis settings, wall/RSS/disk/attempt budgets, solver options, thread settings, and all output paths.

Record which changes are engineering choices and which are fixed science. Engineering choices may include data-class style, sparse matrix assembly and plot colors. They may not alter the observation model, baselines, generators, information access, effect threshold, estimand, primary cap, confirmation seeds, or acceptance rules after freezing.

## 10. Required deliverables and claims

Deliver artifacts proportionate to the stages actually completed:

1. Runnable code and concise setup/usage instructions if implementation occurred.
2. Source/input manifest and exact environment/dependency record.
3. Frozen protocol, sample bundles and private evaluator truths, seeds and hashes.
4. Raw per-scenario and per-call results, including failures, timeouts, fallbacks and replacements.
5. Verifiable lower certificates and feasible witnesses for released joint results, with a replay command.
6. Essential analysis: a primary effect table/interval; complete cell table; bound-tightness and gap table; coverage-event diagnostics; runtime/memory/failure summary.
7. At most four essential figures: bound tightness/gain by sample size and depth; secondary cap sensitivity; identification/numerical gap versus cost; failure/coverage diagnostic if informative. Display uncertainty and label synthetic model assumptions.
8. A standalone research report stating the question, prior art, assumptions, method, actual stage decisions, measured resource use, results, limitations, and outcome classification.
9. For an analytic or resource stop, the checked derivation/counterexample or measured scaling evidence and the minimal code needed to reproduce it. Do not invent absent benchmark artifacts.

The report must separate:

- what the supplied notebook proves;
- what this implementation verifies numerically;
- what the synthetic experiment supports;
- what remains assumed about real institutions and markets;
- what was not performed.

Explain whether combined information is useful beyond separated bounds, whether samples themselves add value over moments, whether the cost is acceptable, and whether a specific next step is warranted. These are distinct questions.

Do not use terminology such as certified market safety, validated bubble prediction, guaranteed crash warning, or real-time hidden-leverage measurement. A valid capacity lower confidence bound remains conditional on known model inputs and behavior.


## 11. Supporting research and work performed during planning

### 11.1 Search scope and attribution

The research search was conducted on **2 November 2026**, using primary papers, author manuscripts and official technical documentation. It covered constant-leverage feedback and MinMaSS; one-sided leverage constraints; finite rebalancing and leveraged-ETF path dependence; bounded moment problems and canonical distributions; worst-case large deviations; DKW/KS confidence regions; robust sample-average approximation; moment-plus-CDF ambiguity sets; scalar numerical optimization; LP dual output; and directed decimal arithmetic.

Backward investigation followed the notebook's moment and large-deviation references. Forward/contextual checking located the historical U.S. equity application [R7] and the established goodness-of-fit-plus-moment construction [R3]. An additional finite-support KS LP treatment is Luo and Mehrotra, §3.5, Theorem 5 [R13]. Its finite-support formulation does not by itself justify a continuous-support discretization.

No exact published characterization of the proposed quantitative capacity-intersection experiment was located. This search is not exhaustive, and equivalent formulations may exist. If a precise prior match is encountered during implementation, acknowledge it and classify the work as reproduction plus diagnostic evaluation; do not relabel established mathematics as a new theorem.

### 11.2 Central claims and resolving checks

| Central claim | Evidence status / source | Remaining uncertainty | Resolving check |
|---|---|---|---|
| The source capacity is a well-defined continuous-model fold under initial stability. | Source derivation audited, notebook §§2–3. | Economic behavior and depth are assumed. | Analytic/scalar checks; no claim of market identification. |
| Uncapped finite moments do not generally give a uniformly positive capacity guarantee. | Notebook §§4–5; exact perturbation and TV argument checked. | Historical novelty of the precise combined theorem remains unresolved. | Small rational tail diagnostic; no generic moment-oracle implementation. |
| Capped moment and DKW baselines are strong and cheap. | Notebook §§6–7; classical sources [R1], [R2], [R5]. | Numerical robustness in the target implementation. | Closed-form anchors, exact CDF transform, interval capacity checks. |
| Combining confidence regions and moments is known methodology. | [R3], §3.2, Eq. (11). | Quantitative benefit for this particular capacity functional. | Frozen comparison against the maximum of separated bounds. |
| The ideal joint value has a minimax formulation. | Derivation in §3.4 using [R4]. | Empty-set handling and executable numerical enclosure. | Compactness/domain audit; safe LP and original-set witness. |
| Upward rounding provides a continuum-valid lower approximation. | Derivation in §4.2, independently reviewed during planning. | Finite mesh cost and conditioning. | Exact rounding fixtures, measured refinement and gap. |
| Approximate LP multipliers can yield a valid bound with residual correction. | Direct weak-duality derivation in §4.4; marginal convention checked against [R10]. | Correct directed implementation. | Independent verifier and intentionally corrupted certificates. |
| Joint optimization materially improves the operational certificate. | Hypothesis only. | Size, sample variation and verification cost. | Primary effect interval, failure accounting and ideal-gain upper ceiling. |
| The specified workloads fit an M3 Max. | Planning estimate only. | Actual RAM, packages, timing and witness cost. | Mandatory environment inspection and calibration gate. |
| Results apply to a real U.S. equity or macro radar. | Not established. | Actual holdings weights, behavior, cap, depth, sampling and external shocks. | Outside this PoC; require a future institutional observation study. |

The cap-bounded calculations naturally extend from finite-support examples to all probability laws on the compact interval: the exponential and its relevant derivatives are bounded continuous functions there. This extension is part of the present construction, not a numerical assumption that unknown support points happen to lie on a mesh.

### 11.3 Small checks actually performed

The research lead and independent mathematical reviewer performed limited planning calculations, not the PoC:

- independently regenerated the notebook's baseline \(x^*\) and \(C\);
- regenerated the \(A=2\) two-moment capacity endpoints;
- checked the exact moment identities for the four rational generators;
- audited the upward-rounding argument, minimax conditions, bounded-variable dual formula and exact three-anchor witness repair;
- inspected the notebook's embedded synthetic figure;
- ran a tiny, deterministic, fixed-\(x\) outer-LP probe to check that the proposed information combination was not obviously vacuous.

For that last probe, the input was the balanced 120-observation ATOM3 sample, \(A=10,D=1.5,Q=1,m_1=1,m_2=5/3,\alpha=0.05\). At the single fixed point \(x=0.2\):

| Quantity | Planning result |
|---|---:|
| True ATOM3 capacity | 0.06422573132920234 |
| Sharp two-moment lower, \(A=10\) | 0.049441100247039016 |
| DKW-only lower | 0 |
| DKW radius | 0.12397713925884912 |
| Nominal rounded outer-LP lower at mesh \(J=256\) | 0.04651705636554304 |
| Nominal rounded outer-LP lower at mesh \(J=1024\) | 0.053746110344044584 |

These LP numbers used ordinary float64 optimization and **were not independently interval-certified**. They are exploratory plausibility evidence only. The coarser result is below the simple baseline, illustrating why resolution and fallback matter. No joint optimum, coverage study, main experiment, time-series backtest, or M3 performance measurement was performed.

The focus on a conservative primary cap and the exact rational generators was settled during this planning phase, before generating confirmatory draws. The final protocol must be frozen again by the coding agent before its fresh confirmation samples are generated.

### 11.4 Material application limitations

The finite-variation rebalancing law is not valid unchanged on a general stochastic price path. Leveraged-portfolio returns can depend on realized variance [R12]. A maximum leverage restriction alone does not establish symmetric purchases after gains [R8]. Market depth is also not simply visible order-book volume.

Even correct point-in-time inputs would not identify future exogenous buying, changing margins or funding interventions. Therefore this study can evaluate conditional capacity information, but cannot answer when a crash will occur or whether a valuation bubble exists.

A favorable conditional result can be overwhelmed by errors in \(D,Q,A\) or the behavior model. Do not add estimated parameter uncertainty to the reported confidence level without a separate joint coverage argument. The present alpha covers sampling of the leverage law under known inputs only.

## 12. References and precise source locations

Reference links are ordinary Markdown links and remain usable outside this chat. Access dates below are 2 November 2026 unless otherwise stated. Read locations indicate the content checked during preparation, not a claim that every page of every cited work was audited.

### Source N

**Supplied notebook, author not identified in the attachment.** *Hidden leverage and the limits of bubble prediction: A theorem about the fragility of upward market-clearing capacity*. Research date 2 November 2026. File: **bubble_ignition_research(4).ipynb**. Locations and hashes are in §2.1. Critical material: §§2–8, Eqs. (1)–(16); §11, Eqs. (17)–(19); §12 caveats; embedded code cells 17 and 21. All cell sources and the embedded figure were inspected. This is the source of the particular capacity functional and claimed exact-moment collapse results.

### Source R1

**E. Seneta and N. C. Weber (1982).** “Attainable bounds for expectations.” *Journal of the Australian Mathematical Society, Series A* 33(3), 411–420. [DOI 10.1017/S144678870001884X](https://doi.org/10.1017/S144678870001884X). [Publisher PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F2D9D5BB90840A4C10A044C66D48877F/S144678870001884Xa.pdf/attainable-bounds-for-expectations.pdf). Relevant locations read: Lemmas 1–2, Theorem 1 on printed p.414, Corollary 1 on p.415. Establishes the classical sharp expectation-bound background.

### Source R2

**Charuhas Pandit and Sean Meyn (2006).** “Worst-case large-deviation asymptotics with application to queueing and information theory.” *Stochastic Processes and their Applications* 116(5), 724–756. [DOI 10.1016/j.spa.2005.11.003](https://doi.org/10.1016/j.spa.2005.11.003). [Author manuscript dated 16 November 2007](https://faculty.eng.ufl.edu/meyn/wp-content/uploads/sites/671/archive/spm_files/Papers_pdf/PM05.pdf). Relevant locations read: §1.1, Theorems 1.1–1.2; §2.1, Propositions 2.1–2.2 and expanding-support example. The manuscript date is not the publication year. This is close prior work on exponential extrema and rare-tail sensitivity; the notebook's compound-Poisson formulation is not identical to an ordinary log-MGF rate function.

### Source R3

**Dimitris Bertsimas, Vishal Gupta and Nathan Kallus (2018; online 2017).** “Robust sample average approximation.” *Mathematical Programming* 171, 217–282. [DOI 10.1007/s10107-017-1174-z](https://doi.org/10.1007/s10107-017-1174-z). [Author-hosted full paper](https://web.mit.edu/~dbertsim/www/papers/Robust%20Optimization/Robust%20Sample%20Average%20Approximation.pdf). Relevant locations read: §§2.1–2.2; §3.2, Proposition 2 and Eqs. (9)–(11); selected tractability results. Equation (11) is the important direct precedent for intersecting goodness-of-fit and moment information. Its particular approximate moment-test examples are not the coverage argument used in this brief.

### Source R4

**Maurice Sion (1958).** “On general minimax theorems.” *Pacific Journal of Mathematics* 8(1), 171–176. [DOI 10.2140/pjm.1958.8.171](https://doi.org/10.2140/pjm.1958.8.171). [Publisher PDF](https://msp.org/pjm/1958/8-1/pjm-v8-n1-p14-p.pdf). Relevant location read: Theorem 3.4, printed p.174, with definitions in §2. Applies after the compactness/convexity/continuity conditions established in §3.4 of this brief.

### Source R5

**Pascal Massart (1990).** “The Tight Constant in the Dvoretzky–Kiefer–Wolfowitz Inequality.” *The Annals of Probability* 18(3), 1269–1283. [DOI 10.1214/aop/1176990746](https://doi.org/10.1214/aop/1176990746). [Author-uploaded record](https://www.researchgate.net/publication/38360933_The_Tight_Constant_in_the_Dvoretzky-Kiefer-Wolfowitz_Inequality). The explicit two-sided inequality was verified in the paper's abstract on the author-uploaded record. The publisher retrieval returned an iframe; the complete original proof was not independently retrieved in this review. This access limitation does not substitute a different confidence inequality.

### Source R6

**Tobias Adrian, Karol Jan Borowiecki and Alexander Tepper (2014; revised February 2021).** “A Leverage-Based Measure of Financial Instability.” Federal Reserve Bank of New York Staff Report 688. [Official paper](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr688.pdf). Relevant locations read: introduction, single-asset demand Eq. (6), aggregate stability Eq. (32) and accompanying interpretation. MinMaSS is direct financial-mechanism precedent. Use report number 688 from the official cover; an inner abstract page contains a conflicting typographical number.

### Source R7

**Karol Jan Borowiecki, Michał Dzieliński and Alexander Tepper (2023; online 19 October 2022).** “The great margin call: The role of leverage in the 1929 Wall Street crash.” *The Economic History Review* 76(3), 807–826. [DOI and open article](https://doi.org/10.1111/ehr.13213). Relevant locations read: introduction and §§III–V. Establishes that application of leverage-based stability analysis to historical U.S. equities is not new; it does not validate the notebook's particular capacity certificates.

### Source R8

**Rama Cont and Eric Schaanning (2017).** “Fire sales, indirect contagion and systemic stress testing.” Norges Bank Working Paper 2/2017, 17 March 2017. [Official publication page](https://www.norges-bank.no/aktuelt/publikasjoner/Working-Papers/2017/22017/). [Official PDF](https://www.norges-bank.no/contentassets/bb47f56979fe4adf9249d1c0ab55c7d1/working_paper_2_17.pdf?v=17032017132952). Relevant locations read: introduction, §2.1 and §2.6, printed p.15. Distinguishes one-sided constraints from symmetric targeting. Different circulated versions have different section numbering.

### Source R9

**U.S. Securities and Exchange Commission, Division of Investment Management.** “Frequently Asked Questions About Form 13F.” Page updated 6 March 2026. [Official FAQ](https://www.sec.gov/rules-regulations/staff-guidance/frequently-asked-questions-about-form-13f). Relevant locations checked: Question 25 on filing timing, Question 41 on short positions, Question 43 on options. These facts delimit the observation model.

### Source R10

**SciPy developers.** “linprog(method='highs').” Official SciPy reference documentation, accessed 2 November 2026. [Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html). Relevant sections read: problem convention, result marginals, time_limit, primal/dual feasibility tolerances, and marginal-sign note. Runtime package versions must be inspected and frozen; a documentation page's displayed version is not proof of target installation.

### Source R11

**Python Software Foundation.** “decimal — Decimal fixed-point and floating-point arithmetic.” Official Python documentation, accessed 2 November 2026. [Documentation](https://docs.python.org/3/library/decimal.html). Relevant sections read: Decimal.exp, ln, from_float, next_minus, next_plus, contexts and rounding modes. Used to specify a replayable outward-arithmetic verifier rather than merely more decimal digits.

### Source R12

**Marco Avellaneda and Stanley Zhang (2010).** “Path-Dependence of Leveraged ETF Returns.” *SIAM Journal on Financial Mathematics* 1, 586–603. [DOI 10.1137/090760805](https://doi.org/10.1137/090760805). [Author manuscript, 19 May 2009](https://math.nyu.edu/~avellane/LeveragedETF20090515.pdf). Relevant material read: §2, Eq. (10), and derivation of realized-variance dependence. Provides a practical boundary on extending a price-only continuous finite-variation response over stochastic horizons.

### Source R13

**Fengqiao Luo and Sanjay Mehrotra (2018 preprint; 2020 publication).** “Distributionally Robust Optimization with Decision Dependent Ambiguity Sets.” *Optimization Letters* 14, 2565–2594. [DOI 10.1007/s11590-020-01574-3](https://doi.org/10.1007/s11590-020-01574-3). [arXiv:1806.09215](https://arxiv.org/abs/1806.09215). Relevant locations read: Assumption 1; §3.1, Eq. (2); §3.5, Eqs. (29)–(34), Theorem 5. The finite-support LP result is methodological context, not a justification for ignoring continuum approximation error.

[N]: #source-n
[R1]: #source-r1
[R2]: #source-r2
[R3]: #source-r3
[R4]: #source-r4
[R5]: #source-r5
[R6]: #source-r6
[R7]: #source-r7
[R8]: #source-r8
[R9]: #source-r9
[R10]: #source-r10
[R11]: #source-r11
[R12]: #source-r12
[R13]: #source-r13


## 13. Review findings, repairs, and residual risks

Three distinct independent-agent passes reviewed the actual draft: mathematical/novelty validity, adversarial experimental validity, and execution from an empty project on the target laptop. The research lead reconciled their findings, revised the brief, and requested targeted rechecks of affected mathematical and execution sections. These reviews were of the design; they are not implementation tests or hardware benchmarks.

| Material finding or stress scenario | Disposition in the final brief |
|---|---|
| The proposed mechanism may already be known. | Credit MinMaSS, canonical moment bounds, robust SAA and KS LP work. Limit the contribution to a quantitative conditional information-value result. |
| A source proof contains a nonconstructive/existence step. | Avoid a generic moment-representation oracle; use explicit rational fixtures and finite LPs with a proved rounding enclosure. |
| An assumption fails for realistic investor behavior. | Preserve the conditional target-leverage model; include one-sided-policy and one-round analytic controls. No public-data forecasting claim survives by renaming a proxy. |
| Ordinary grid restriction can falsely increase guaranteed capacity. | Use upward rounding with explicit moment inflation, and independently verified lower dual bounds. |
| Numerical objectives or high precision can be mistaken for certificates. | Verify the mathematical master with outward arithmetic and bounded-variable residual penalties; validate witnesses exactly. |
| Capacity units could mix \(C\) and \(C/Q\). | All internal interfaces and comparisons use \(C/Q\); convert once at the output boundary and test \(Q\ne1\). |
| Irrational DKW radius and a wrong common root domain could distort the baseline. | Use rational outward radius endpoints and each standalone law's own root bracket. |
| Mean-only ablation could silently receive second-moment information. | Explicit separate fallback, upper bound, grid, domain, repair and contrast contracts exclude \(m_2\). |
| Only favorable synthetic cases might work. | Four fixed ordinary rational populations, cap controls, an exact null, off-grid/continuous fixtures, fresh confirmation, and a narrowly stated conservative-cap claim. |
| The strongest baseline removes the gain. | Compare against the maximum of the exact separated constituents; a resolved lack of material gain is a valid outcome. |
| Permitted confidence misses could inflate mean improvement. | Preserve all draws, report event contributions, and require the prespecified zero-outside sensitivity gate for a supported conditional-information claim. |
| Preprocessing or certification dominates. | Include setup/verification in scenario timing, cache reconstructible coefficients, store sparse certificates, and measure the largest permitted mesh. |
| Time/memory exceeds expectations. | Inspect the actual Mac, select replication by measured costs, enforce cumulative ceilings, and retain a resource-stop outcome. Counts include development and repair allowances. |
| A small pilot is inconclusive. | Use pilot controls to check detectability; do not use noisy pilot gain to retune the scientific question. Freeze fresh confirmation or report insufficient resources. |
| A source or indispensable dataset is inaccessible. | Essential mathematics and synthetic generators are restated. The full Massart proof access limitation is disclosed. No market dataset is required. Missing target hardware blocks its performance claims, not unaffected work. |
| Timeout handling could discard a previously useful result. | Preserve an existing verified lower/witness pair and return its status plus termination reason; fallback only when no pair exists. Genuine correctness defects invalidate affected evidence. |
| Main hypothesis is false. | Distinguish an ideal-information upper ceiling below threshold, a bounded-method negative, unresolved precision, and invalid execution. No indefinite rescue or favorable-subgroup substitution. |

Residual risks remain: this is a deliberately synthetic, conservative-cap regime; exact aggregate moments and the sampling model may be unavailable in practice; modest replication can leave a threshold decision uncertain; witness recovery near feasibility boundaries can be difficult; and Decimal verification may dominate local runtime. These risks have explicit gates or scoped conclusions. No claim of global novelty, absolute correctness of an unimplemented program, or actual M3 feasibility is made.

**Final execution instruction:** carry out the authorized gates, inspect the evidence, preserve adverse outcomes, and deliver the scoped research result. Do not expand this brief into the full radar unless a separate later task supplies and validates the necessary real-market observation model.




