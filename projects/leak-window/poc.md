# PoC execution brief: reliable detection windows in a resonant amplifier with spontaneous reporter activation

**Prepared:** 27 October 2026. **Executor:** an autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning. **Experimental hardware:** one locally available MacBook with an M3 Max; inspect its actual configuration before allocating work. **Status:** researched execution instructions, including explicitly identified planning calculations; the PoC and its Mac benchmarks have not been completed.

## 1. Project decision and smallest defensible contribution

Implement and evaluate **one finite-parameter feasibility diagnostic for the supplied three-stage amplification circuit when its final reporter can activate spontaneously throughout the assay**. Produce safe-design bounds, model-specific impossibility certificates, and an honest unresolved region. Determine how much the supplied asymptotic horizon helps beyond ordinary finite-time variation of constants.

The practical decision is whether this mathematical approach warrants a subsequent, chemically grounded stochastic study. It is premature to optimize a diagnostic device, search unrestricted network architectures, or claim a clinical detection limit. The present study can nevertheless establish a useful limitation: under an explicit ongoing-background model, an attractive clean-signal timing prediction may coexist with an extremely restrictive permissible initiation rate.

### 1.1 Primary research question

> For the specified three-stage circuit, can a scalar calculation derived from the paper's positive-comparison argument and Poisson event conditioning locate the maximum permissible reporter-initiation rate at the clean target deadline to within a 20% multiplicative bracket, while correctly distinguishing safe, impossible, and unresolved detection settings at false-positive probability 0.01 and false-negative probability 0.05?

Test this on the locked finite-parameter grid in §7. The **primary quantity** is the interval width

\[
W_\nu=\nu_N/\nu_S-1,
\]

where \(\nu_S\) is a provably sufficient initiation-rate limit and \(\nu_N\) is a necessary upper limit for reporting at the clean target crossing. Smaller is better. When the rare-initiation condition below holds, that crossing is also the earliest deadline satisfying the sensitivity constraint, so the boundary has a stronger interpretation.

The screening-use criterion is: valid finite bounds, no contradicted safe/impossible certificates, and \(W_\nu\le0.20\) in at least 90% of the locked core cases, with the fractions and worst widths also reported separately for every signal-strength and detuning stratum. A passing case must have positive finite verified rate endpoints, a safe certificate at \(0.8\nu_S\), and an all-deadline impossibility certificate at \(1.2\nu_N\). Failure to compute or validate these primary outputs counts against this fraction. A wide auxiliary numerical reference alone does not invalidate a proved analytic interval; it limits the corresponding point-predictor comparison. This is a **planning utility threshold**, not a clinical standard: a screen should resolve a 20% engineering margin before it is useful for deciding which settings deserve a more expensive model. The percentage summarizes this specified grid; it is not a population confidence statement.

A supported screen does **not** automatically support a distinctive advantage for the asymptotic theorem. Separately assess whether its point prediction adds accuracy or computational value against the stronger finite-time comparator. A valid result may be: “the elementary finite-time screen is useful, but the asymptotic horizon adds explanation rather than practical predictive advantage.”

### 1.2 What changes from the original opportunity

Keep the original purpose—separating a real weak input from background activation—but fix the circuit, specify a tractable background process, and analyze a conditional target response. Background is a homogeneous Poisson stream of positive reporter seeds, present with the **same law under both labels**. Between those seeds, concentrations follow the supplied ODE.

This is a piecewise deterministic model of **random initiation followed by deterministic amplification**. It is not a chemical master equation, a Langevin approximation, or a single-molecule model of all reactions. It includes ongoing spontaneous activation; it excludes intrinsic birth/death fluctuations, uncertain capture, measurement noise, random manufacturing distributions, shared reagent depletion, feedback, and target recognition chemistry. Rate uncertainty is represented by explicit fixed scenarios, not by an invented manufacturing distribution. These exclusions are substantive, especially the deterministic positive arm; see §4.4 and §10.

Do not spend the PoC budget on wet-lab protocols, molecular sequence design, clinical data, a web application, large graph enumeration, or a general reaction-network compiler. Do not infer minutes, concentrations, assay costs, or tuberculosis performance from dimensionless results.

### 1.3 Contribution and novelty boundary

The minimum contribution is a **reproducible finite-parameter regime map, supported by explicit sufficient and impossibility inequalities**, for a well-defined resonant circuit with ongoing reporter nucleation. It includes an audit of whether the theorem actually improves the decision compared with standard calculations. The derivations in §4 are elementary combinations and extensions of established tools; do not advertise them as a historically new stochastic theory without a stronger priority review.

Accounting for background in autocatalytic amplification is already established by Branca et al. [R2]. Programmable molecular kinetics is established by Zhu et al. [R3], whose mechanism differs materially from this circuit. Stable-zero leak suppression is an important alternative [R4]. The paper's degree recursion, random entrance ideas, and nonlinear resonance also have mathematical neighbors [A1, R5, R6]. If closer prior work proves equivalent, retain the validation/limitation study and remove the originality claim. Do not manufacture a larger project to rescue novelty.

## 2. Evidence, source audit, and what is actually established

### 2.1 Supplied materials and provenance

| ID | Supplied item | Location inspected during planning | Status |
|---|---|---|---|
| A1 | `resonant_horizon_research(1).pdf`, *Separation horizons in positive resonant cascades: A nonlinear saturation theorem and its critical-detuning extension*, 27 October 2026, 17 pages; no author identified | `/workspace/scratch/77fb9895c0ac/upload/resonant_horizon_research(1).pdf` | All prose read; displayed equations and relevant proofs checked visually. |
| A2 | `Pasted text(20261027-161608).txt` | `/workspace/scratch/77fb9895c0ac/upload/Pasted text(20261027-161608).txt` | Full request and application argument read. Numerical/application claims treated as claims to assess. |

Those are planning-workspace paths, not paths assumed to exist on the Mac. If provided with A1, copy it without modification to `inputs/resonant_horizon_research.pdf` and record SHA-256. The requested study is executable from the definitions and proofs in this brief; if A1 is absent, report that its source audit was not independently repeated. A1 mentions accompanying numerical code and JSON, but neither was supplied. They are not dependencies of this PoC. Do not invent them or imply that they were inspected.

Ordinary text extraction omits A1's displayed mathematics. Equation (4) on PDF page 2 is visibly clipped on the right; its clock is recoverable consistently from equation (11), page 5, and equation (20), page 15. This is a presentation defect, not missing essential mathematics.

### 2.2 The source theorem

For fixed finite dimension \(n\), \(\lambda>0\), positive integer weights \(w_1=1,w_j\), and triangular polynomials

\[
P_1=0,\qquad P_j(x_{<j})=\sum_{p\in E_j}c_{jp}\prod_{i<j}x_i^{p_i},
\]

the assumptions are: finite nonempty \(E_j\subset\mathbb Z_{\ge0}^{j-1}\), strictly positive coefficients, \(|p|\ge2\), and exact weighted resonance \(\sum_{i<j}w_i p_i=w_j\). The dynamics and initial data are

\[
\dot x_j=\lambda w_j(x_j-x_j^3)+P_j(x_{<j}),\qquad
x(0)=(\epsilon,0,\ldots,0),\quad0<\epsilon<1.
\tag{1}
\]

Define \(Q_1=1\), \(Q_j(t)=\int_0^tP_j(Q_{<j}(s))\,ds\), and compute its degree and leading coefficient without expanding the entire polynomial:

\[
d_1=0,\quad a_1=1,\qquad
d_j=1+\max_{p\in E_j}\sum_{i<j}p_i d_i,
\]
\[
a_j=\frac1{d_j}\sum_{\substack{p\in E_j\\\sum_{i<j}p_i d_i=d_j-1}}
c_{jp}\prod_{i<j}(a_i)^{p_i}.
\]

Here \(p\) is a monomial exponent vector and \(a_i\) is a leading coefficient. Implement the product as `leading_coefficient[i] ** p[i]`, sum over degree-maximizing monomials, then divide by `degree[j]`.

Put \(L=\log(1/\epsilon)\), \(\rho_j=d_j/w_j\), \(A_j=a_j\lambda^{-d_j}\), \(\rho=\max_j\rho_j\), and \(I=\{j:\rho_j=\rho\}\). A1 proves bounded forward solutions and the first global transition profile; only coordinates in \(I\) remain macroscopic at \(T=(L-\rho\log L)/\lambda\). Its §12 additionally proves each component's own first transition:

\[
T_j=\frac{L-\rho_j\log L}{\lambda},\qquad
x_j(T_j+s)\longrightarrow
\frac{A_j e^{\lambda w_js}}{\sqrt{1+A_j^2e^{2\lambda w_js}}},
\]
\[
t_{j,\eta}=T_j+\frac1{\lambda w_j}
\log\frac{\eta}{A_j\sqrt{1-\eta^2}}+o(1),\quad0<\eta<1.
\tag{2}
\]

Source locations: A1 equations (1)–(6), pp. 2–3; positive-comparison proof, pp. 3–5; individual transitions, equations (20)–(27), pp. 15–17. No essential defect was found in an independent planning audit of these arguments. This is not formal proof verification or a historical-priority determination.

The proof uses compactness to identify an asymptotic limit, but computation of the support recursion and amplitudes is constructive. **The theorem does not supply a usable finite-\(L\) error bar for equation (2).** A small residual in one example is not such an error bar. The finite bounds in §4 deliberately use the proof's explicit comparison inequality instead.

### 2.3 Existing extensions and limits

| Source result | What may be used | What must not be inferred |
|---|---|---|
| A1 §9, pp. 11–13, equations (13)–(18) | Rates \(\lambda w_j+\gamma_j/L\), with linear and cubic coefficients changed together; uniformity for \(\gamma\) in a fixed compact set; a constructive amplitude ODE | Fixed nonzero detuning is not covered uniformly as \(L\to\infty\). |
| A1 §10, pp. 13–14 | Bounded weighted downstream initial seeds \(b_j(\epsilon)\epsilon^{w_j}\); random source amplitude as a random initial condition | This extension is already present. It is not continuous leakage, intrinsic reaction noise, or a new result of this PoC. |
| A1 §5, p. 6 | Conditional extension of the clock to suitable scalar saturation laws | The cubic sigmoid is not universal. Coercivity and entrance-orbit conditions must be checked. |
| A1 §8, p. 10 | Explicit signed-cancellation counterexample and scope warnings | Positive support selection does not survive arbitrary signed coupling, feedback, or changing network size. |
| A1 §6, pp. 7–8 | Reported scaled numerical checks and sampled profile errors | The table entries are not certified uniform error bounds or Mac timings; source code was not supplied. |

Do not implement a large generic polynomial engine. For this project the exact degree recursion returns \(d=(0,1,2)\), \(w=(1,2,3)\), \(I=\{3\}\), and \(A_3=k_2k_3/(2\lambda^2)\). A tiny independent recursion check is enough to verify the connection to A1.

### 2.4 Closest evidence and contribution constraints

Branca et al. model background amplification and its effect on analytical detection limits [R2]. Its abstract and bibliographic record were verified, but full text was not recovered. Equation-level overlap remains unresolved; this blocks a strong priority claim, not the proposed synthetic diagnostic.

Zhu et al.'s 2025 experiments concern aptamer sequestration and competing hybridization, with a favorable hierarchy \(k_1\gg k_2\gg k_3\) [R3, Fig. 1 and pp. 39453–39454]. They do not demonstrate autonomous resonant growth rates \(1:2:3\). Montagne et al. demonstrate stable-zero behavior through saturable deactivation [R4, Results and Figs. 2–3]; that mechanism changes the theorem's unstable-zero premise.

Random branching-to-ODE time shifts predate this project [R5, §1.1 and Theorem 1.1]. The hypotheses of that result do not automatically fit a diagonal linearization with nonlinear downstream initiation. Dominant-monomial and tropical reductions also predate it [R6, §2 and §3]. None of these observations invalidates A1; they restrict what can be claimed as original.

The WHO's March 2026 announcement verifies the broader need for more accessible molecular TB diagnosis [R7]. It provides no evidence that this circuit will improve a diagnostic. No matching experimental dataset for the stipulated circuit was found or is required for this computational study.

## 3. Fixed mathematical and observation model

Use dimensionless time, setting \(\lambda=1\). Physical conversion would require a measured rate: \(t_{\rm physical}=t/\lambda_{\rm physical}\). All concentrations are dimensionless. The clean positive model is

\[
\dot x_1=x_1-x_1^3,
\quad \dot x_2=2(x_2-x_2^3)+k_2x_1^2,
\quad \dot x_3=r(x_3-x_3^3)+k_3x_1x_2,
\tag{3}
\]

with \(k_2,k_3>0\), \(r=3+\delta>0\), and \(x(0)=(e^{-L},0,0)\). Under \(H_0\), start at \((0,0,0)\). Under **both** hypotheses, reporter jumps occur at times of a homogeneous Poisson process of rate \(\nu\ge0\):

\[
x_3(S_m^+)=x_3(S_m^-)+q,\quad 0<q<\eta<1.
\tag{4}
\]

The leak process is independent of the target and has identical parameters in both arms. It does not seed stages 1 or 2. In particular, a stage-2 seed alone under \(H_0\) would not activate stage 3, because the reporter coupling contains \(x_1x_2\). Do not accidentally invent that path.

The observation is the latent, noise-free endpoint: report positive if \(x_3(\tau)\ge\eta\). It is equivalent here to crossing by \(\tau\): the drift at \(x_3=\eta\) is positive and all jumps are positive, so a crossed trajectory cannot fall below \(\eta\). This equivalence would fail for a noisy measurement rule or a different reset/inhibition model.

Fix \(\alpha=0.01\), \(\beta=0.05\); these are synthetic study targets. Let \(t_S\) denote the clean positive reporter's first crossing. Let \(F_0(\tau;\nu,q,r,\eta)\) be the negative false-positive probability. Define

\[
\nu_*:=\sup\{\nu:F_0(t_S;\nu,q,r,\eta)\le\alpha\}.
\tag{5}
\]

This rate limit refers to the clean target deadline. It becomes the reliable-detection boundary in the rare-initiation regime established in §4.4. Do not extend a local boundary argument to arbitrary very high initiation rates without checking that argument's conditions.

## 4. Derived finite-time guarantees and computational mechanism

Equations (6)–(15) below are **derivations made for this brief**, using A1's comparison argument and elementary event conditioning. They are not quoted claims of A1 or experimentally fitted laws. Reproduce the proofs in `analysis/derivations.md` and check their implementation before benchmarking.

### 4.1 Exact clean comparison and asymptotic predictors

Remove cubic damping from (3). Variation of constants gives

\[
u_1(t)=e^{t-L},\quad u_2(t)=k_2t e^{2(t-L)},
\quad u_3(t)=k_2k_3t^2e^{3(t-L)}\phi_2(\delta t),
\tag{6}
\]

where

\[
\phi_2(z)=\frac{e^z-1-z}{z^2},\quad\phi_2(0)=\frac12.
\]

All \(u_j\) are nondecreasing, \(u_j'\ge r_j u_j\), and \(0\le x_j\le u_j\), with \(r_j=(1,2,r)\). Use logarithmic evaluation. At exact resonance,

\[
\log u_3=\log(k_2k_3/2)+2\log t+3(t-L).
\]

The theorem point prediction is

\[
t_H=L-\tfrac23\log L+\tfrac13\log\frac{\eta}{A\sqrt{1-\eta^2}},
\tag{7}
\]

with \(A=k_2k_3/2\) at \(\delta=0\), or \(A=k_2k_3\phi_2(\gamma)\) for \(\delta=\gamma/L\). The offset denominator remains the limiting rate 3. No finite-time certificate may use \(t_H\) as an exact crossing time.

Define the strong inexpensive point comparator \(t_V\) as the unique positive root

\[
u_3(t_V)=q_\eta,\qquad q_\eta=\eta/\sqrt{1-\eta^2}.
\tag{8}
\]

The transform accounts for the scalar cubic saturation profile while retaining the full finite-time forcing integral. It is a point approximation, not an exact nonlinear solution or a guaranteed bound. Also compute the rigorous lower time bound \(t_L\) from \(u_3(t_L)=\eta\). This distinction matters: using \(q_\eta\) instead of \(\eta\) in the impossibility proof would be unjustified.

### 4.2 A constructive upper bound on the clean crossing

For any chosen \(t_0>0\), put

\[
E(t_0)=\tfrac12\max_i u_i(t_0)^2,\qquad z=e^{-5E(t_0)}u_3(t_0).
\]

A1's damping comparison yields \(x_j(t_0)\ge e^{-K_jE}u_j(t_0)\), where \(K=(1,3,5)\). The finite calculation does not require resonance: \(u_j'\ge r_ju_j\) implies \(r_j\int_0^{t_0}u_j(s)^2ds\le u_j(t_0)^2/2\). Apply variation of constants inductively; each forcing product inherits the sum of the predecessor exponents, and each scalar damping factor adds one.

For \(0<a<b<1\), define the exact scalar cubic delay

\[
d_r(a,b)=\frac1r\left[\log b-\log a+
\tfrac12\log(1-a^2)-\tfrac12\log(1-b^2)\right].
\tag{9}
\]

The scalar unforced reporter starting at \(z\) is a lower solution after \(t_0\). Therefore

\[
t_L\le t_S\le t_U(t_0):=t_0+d_r(z,\eta),\quad0<z<\eta.
\tag{10}
\]

If \(z\ge\eta\), use \(t_U=t_0\). In the prescribed calculation choose four cutoffs \(m\in\{\eta/20,\eta/10,\eta/5,2\eta/5\}\), solve \(\max_i u_i(t_0)=m\), evaluate each upper bound, and take their minimum. Omit a cutoff if \(m\le e^{-L}\), recording the reason. If all are omitted, use the positive point \(t_0=t_L/2\); the general inequality is still valid, though possibly loose. This is a fixed four-choice procedure, not outcome-driven optimization.

In exact arithmetic this gives a finite bound. Implement the last decisions using outward interval arithmetic as specified in §6; ODE tolerances alone do not certify it.

### 4.3 False-positive bounds for ongoing reporter activation

Under \(H_0\) only the scalar reporter evolves. Let \(d=d_r(q,\eta)\). At a deadline \(\tau\ge0\), define \(a=(\tau-d)_+\), \(b=\min(\tau,d)\). A jump in the early interval of length \(a\) is sufficient for detection by \(\tau\). If there is no early jump, zero or one jump in the remaining interval cannot suffice. Independent Poisson counts give

\[
F_{\rm lo}(\tau,\nu)=1-e^{-\nu a}
\le F_0(\tau;\nu)
\le F_{\rm hi}(\tau,\nu)=1-e^{-\nu\tau}(1+\nu b).
\tag{11}
\]

The gap is \(e^{-\nu a}[1-e^{-\nu b}(1+\nu b)]\), of order \((\nu b)^2\) when \(\nu b\) is small. This accounts for ongoing activation, not just an initial offset or a mean leakage flux. For \(q\ge\eta\), a separate scalar correctness control has \(d=0\) and exact \(F_0=1-e^{-\nu\tau}\); the core grid retains \(q<\eta\).

At \(\nu=0\) return zero risk directly. Avoid cancellation: use `expm1` or high-precision evaluation for small \(\nu b\). Clamp only a final interval intersection with \([0,1]\), never a negative physical state or a broken formula.

### 4.4 Safe, impossible, and unresolved decisions

**Safe:** if \(F_{\rm hi}(t_U,\nu)\le\alpha\), reporting at \(t_U\) has conditional target sensitivity one and false-positive probability at most \(\alpha\). Positive reporter jumps cannot delay a target crossing. This is a stronger sensitivity condition than required for the stipulated deterministic positive arm, not an unconditional diagnostic sensitivity statement.

**Earliest-deadline lemma:** if \(\nu t_S<\log(1/\beta)\), then no \(\tau<t_S\) can meet sensitivity \(1-\beta\). On the no-arrival event, whose probability is \(e^{-\nu\tau}>\beta\), the target is still below threshold. At \(t_S\), every target path has crossed. Thus \(t_S\) is the earliest sensitivity-feasible deadline under this condition. A sufficient computable check is \(\nu t_U<\log(1/\beta)\). The atom at the deterministic clean crossing is why this reduction works; do not transfer it to intrinsically stochastic or noisy positives.

**All-deadline impossibility:** a useful alternative needs no numerical ODE oracle. For \(\nu>0\), put

\[
c_\alpha=-\log(1-\alpha),\quad c_\beta=\log(1/\beta),\quad
D_{\max}=d+c_\alpha/\nu.
\]

FPR feasibility necessarily gives \(\tau\le D_{\max}\) by (11). If

\[
u_3(D_{\max})<\eta\quad\text{and}\quad e^{-\nu D_{\max}}>\beta,
\tag{12}
\]

then all such deadlines precede the clean target crossing and leave too many no-arrival targets negative. **No deadline satisfies both constraints within model (3)–(4).** Leakage is allowed to help targets; the proof does not assume otherwise. Test these strict inequalities using interval signs. At an equality or unresolved interval, return unresolved.

**Rate bracket:** find the largest safe lower endpoint \(\nu_S\) by the monotone root \(F_{\rm hi}(t_U,\nu)=\alpha\), selecting an interval-verified endpoint below it. If \(t_L>d\), define

\[
\nu_N=c_\alpha/(t_L-d).
\tag{13}
\]

Choose a conservative numerical upper endpoint for this expression. Then \(\nu_S\le\nu_*\le\nu_N\). If \(t_L\le d\), this simple necessary limit is infinite; record an uninformative bracket. A rate greater than \(\nu_N\) is a candidate all-deadline no-go, but **still evaluate both conditions in (12)**. Failure of a sufficient condition alone never proves impossibility.

Return one of `safe`, `impossible`, `unresolved`, `outside_model`, or `numerical_failure`, with the evaluated inequalities and their intervals. Keep `outside_model` distinct from an unfavorable result.

### 4.5 A stronger probability reference without rare-event Monte Carlo

The numerical reference uses exact scalar flow, event-count conditioning, and a bounded two-arrival area. This avoids trying to measure a 1% probability to high relative precision with a small random sample.

The scalar flow for positive \(x\) is

\[
\Phi_h(x)=\frac{x}{\sqrt{x^2+(1-x^2)e^{-2rh}}},\quad h\ge0,
\]

and, for a second arrival at time \(s\in[0,\tau]\), the threshold pulled backward is

\[
y(s)=\Phi_{s-\tau}(\eta)
=\frac{\eta e^{-r(\tau-s)}}{\sqrt{1-\eta^2+\eta^2e^{-2r(\tau-s)}}}.
\]

Define \(D(s)=0\) when \(y(s)\le2q\), otherwise \(D(s)=d_r(q,y(s)-q)\). Let \(g(s)=(s-D(s))_+\), and \(I_2=\int_0^\tau g(s)ds\). For second arrival at \(s\), exactly the first-arrival times in an interval of length \(g(s)\) produce detection. Therefore

\[
F_2=e^{-\nu\tau}\{\nu(\tau-d)_++\nu^2I_2\}
\le F_0(\tau;\nu)
\le F_2+\Pr\{\operatorname{Pois}(\nu\tau)\ge3\}.
\tag{14}
\]

There is **no extra factor \(1/2\)** in front of \(I_2\); ordered arrival times already account for it.

Use a bounded quadrature, not an adaptive solver's estimated error as a proof. The function \(D\) is continuous and nondecreasing. On a panel \([a,b]\),

\[
(a-D(b))_+\le g(s)\le(b-D(a))_+.
\tag{15}
\]

Summing panel widths times these endpoints gives lower and upper areas. Intersect with \([0,\tau^2/2]\). For uniform panel width \(h=\tau/N\), the area gap is at most \(h[\tau+D(\tau)-D(0)]\). Use this to select an initial dyadic \(N\); double only while needed, up to 16,384 panels. Evaluate endpoints with outward arithmetic. Normalize time by \(\tau\) when convenient; then cache \(I_2/\tau^2\) for reuse across every \(\nu\) root evaluation. The geometry does not depend on \(\nu\).

Intersect the two-event interval with (11). The probability of three or more arrivals is an explicit truncation remainder, not a fitting error. If this remainder prevents the precision goal, do not add an unbounded higher-order integrator. Report a reference interval or unresolved case; the single predeclared refinement is (14).

### 4.6 Algorithm pseudocode

```text
analyze_case(case, cumulative_budget):
    validate positive rates/couplings, L > 0, 0 < q < eta < 1,
             zero H0 initial state, reporter-only homogeneous Poisson leak,
             cubic drift, endpoint latent observation
    if unsupported: return outside_model with exact reason
    construct log(u1,u2,u3) and the scaled clean ODE
    obtain verified tL below the u3 = eta root
    for each of four prescribed cutoff levels greater than exp(-L):
        find an ordinary floating candidate t0 > 0
        interval-evaluate E, a lower z, and an upper t0 + d_r(z,eta)
    if no eligible cutoff: evaluate the stated tL/2 fallback
    tU = smallest verified candidate upper bound
    compute tH where its assumptions apply; compute tV everywhere in this family
    compute interval d(q,eta), nuS, nuN; preserve infinite/unresolved endpoints
    record safe/no-go decisions at specified trial rates via (11) and (12)
    if stage permits reference work:
        compute and validate clean nonlinear crossing tS numerically
        compute probability reference at its time uncertainty endpoints
        locate a reference rate bracket using (11) intersected with (14)
        distinguish numerical tS uncertainty from analytic time bounds [tL,tU]
    append raw record, failures, costs and budget counters atomically
    stop the queue if any stage or cumulative ceiling is reached
```

## 5. Physical interpretation and early application checks

### 5.1 One molecule versus the critical deterministic seed

If a normalization capacity is \(\Omega\) molecules and the target count is \(n_0\), then \(\epsilon=n_0/\Omega\) and one reporter molecule corresponds to \(q=1/\Omega=\epsilon/n_0\). The clean resonant reporter has effective seed \(A_3\epsilon^3L^2\). Consequently

\[
\frac{q}{A_3\epsilon^3L^2}
=\frac{\Omega^2}{A_3 n_0^3L^2}.
\tag{16}
\]

The requirement that a continuous direct reporter seed be negligible can lie far below one molecule. This makes **the probability of initiation**, not an arbitrarily tiny continuous background concentration, the meaningful variable in this model. It does not establish that an actual chemistry has any particular \(\nu\). Do not interpret deterministic growth after a one-molecule jump as experimentally validated.

The grid's \(n_0=1,10,100\) values are normalization scenarios, not evidence of single-copy performance. Large \(L\), especially 80 and 160, are asymptotic numerical stress tests; their implied capacities may be physically unreasonable. Keep results separated by \(L\) rather than allowing extremely small inputs to dominate the interpretation.

### 5.2 Capture imposes a separate information limit

If captured target count is \(N\sim\operatorname{Pois}(\mu)\), and a target sample with \(N=0\) has the same observation law as \(H_0\), then

\[
\operatorname{FNR}\ge e^{-\mu}(1-\alpha),\qquad
\mu\ge\log\frac{1-\alpha}{\beta}
\tag{17}
\]

is necessary for the requested sensitivity. For this study's \(\alpha,\beta\), the lower mean is approximately 2.98568. This is an elementary conditioning argument, independent of A1. Do not quietly condition away zero-capture samples when discussing eventual clinical sensitivity. The computational study remains conditional on the specified nonzero deterministic input.

### 5.3 Architecture control

A directly seeded scalar amplifier with drift \(r(x-x^3)\), input \(\epsilon\), the same \(q,\nu,\eta\), has clean deadline \(d_r(\epsilon,\eta)\), approximately \(L/r\). Include this counterfactual control. It tests whether an apparent cascade speed benefit merely compares different autonomous rates. Direct recognition of the fast stage is an added architectural assumption; its chemical feasibility and cost are unknown. Report the comparison in this mathematical model without claiming a fair hardware-cost comparison or global architectural optimality.

## 6. Implementation and numerical specification

### 6.1 Minimal stack and repository

Use native ARM64 Python 3.12 or 3.13 if available, NumPy, SciPy, Matplotlib, mpmath, and pytest. All experimental work uses a single CPU process and float64 for ODEs; interval scalar decisions use mpmath's `iv` context. No GPU, Metal/MPS, CUDA, cloud compute, paid data, services, or databases are needed. Version ranges are not an assertion of compatibility: install available compatible native wheels, run the checks below, then freeze exact versions. Keep compiled acceleration out unless the measured calibration identifies a concrete bottleneck; it is not expected to be necessary.

Create this small structure:

```text
leak-window/
  README.md
  requirements.lock.txt
  inputs/                         # optional unchanged source PDF and hashes
  configs/{development,core,stress}.json
  protocol.lock.json
  leak_window/
    __main__.py                   # CLI and cumulative budget ledger
    models.py                     # exact comparator, scalar flow, horizon, validation
    certificates.py               # interval scalar bounds and decisions
    probability.py                # negative flow, event geometry, risk intervals
    reference.py                  # scaled ODE and independent tiny checks
    study.py                      # fixed case generation, caching and timing
    report.py                     # regenerate tables/plots/report
  tests/test_oracles.py
  analysis/derivations.md
  results/{environment.json,budget.json,cases.jsonl,failures.jsonl}
  results/tables/
  results/figures/
  REPORT.md
```

A module may be merged if it makes the implementation simpler. Do not build an extensible framework. Required interfaces are `clean_comparator(t, case)`, `clean_time_bounds(case)`, `risk_bounds(tau, nu, case)`, `risk_reference(tau, nu, case)`, `clean_reference(case)`, and `classify(case, nu)`. Return structured records with status, numeric intervals, tolerances, method, elapsed time, and error messages. Never encode failure as a plausible zero or finite crossing.

### 6.2 Stable ODE reference

Directly initializing \(e^{-L}\) in a solver with ordinary absolute tolerance can erase the signal. Evolve the exactly equivalent variables \(z_j=e^{j(L-t)}x_j\):

\[
z_1'=-e^{2(t-L)}z_1^3,
\]
\[
z_2'=k_2z_1^2-2e^{4(t-L)}z_2^3,
\]
\[
z_3'=\delta z_3+k_3z_1z_2-r e^{6(t-L)}z_3^3,
\quad z(0)=(1,0,0).
\tag{18}
\]

Use DOP853 initially with `rtol=2e-11`, `atol=2e-12`, `max_step=0.25`, terminal positive-direction reporter event. Evaluate the event through a stable reconstruction or \(3(t-L)+\log z_3-\log\eta\); at \(z_3=0\), return a finite negative event value. The integration endpoint is the verified \(t_U+1\). Failure to find the event by then is a correctness failure, not an infinite biological detection time. [R8] documents the solver and event semantics.

At \(L=160\), some exponentially negligible damping coefficients underflow near the initial time. Record this explicitly; compare representative cases with a logarithmically evaluated product and tighter/alternative integration. Never underflow the actual initial source or discard a non-negligible forcing term. Do not integrate to very late times that cause irrelevant large exponential intermediates.

Validate an initial subset with Radau and 10-fold tighter tolerances; use exact scalar \(x_1\), comparison bounds, and small-\(L\) unscaled integration with component-specific tiny tolerances as independent checks. No independent method should inherit an approximate asymptotic initial condition. For \(L\le20\), an unscaled check can use absolute tolerances proportional to \((e^{-L},e^{-2L},e^{-3L})\); verify their values are representable.

Accept a numerical reference time only when discrepancy in the prescribed checks is at most \(10^{-7}\) time units and at most one twentieth of the timing difference being interpreted. The initial precision target is much smaller than the 20% rate-bracket utility margin. If these conditions fail, refine once or label the comparison unresolved. A solver tolerance is a local error controller, not a rigorous enclosure of \(t_S\).

### 6.3 Scalar arithmetic, root finding, and certification

Use log amplitudes throughout. Evaluate \(\phi_2\) by its convergent series near zero (beginning \(1/2+z/6+z^2/24+\cdots\)) or a tested stable special-function expression; outside that regime use `expm1` and log forms as needed. Test both signs of detuning and the exact zero branch. For floating estimates, bracket monotone roots and use `brentq`; do not use an unbracketed Newton iteration near vanishing denominators.

For **certified labels**, evaluate only elementary scalar expressions with `mpmath.iv` at 50 decimal digits. Preserve primitive inputs such as \(L,n_0,\gamma,k_2,k_3,\eta\) as exact integer/rational/decimal strings, and construct \(\epsilon=e^{-L}\), the case's \(q\) and \(\delta\), \(r=3+\delta\), and the cutoffs inside the interval context. For core/development normalization cases use \(q=e^{-L}/n_0\), \(\delta=\gamma/L\); fixed-detuning stress uses its declared \(\delta\) directly, while critical-seed stress constructs \(q=\kappa A_3e^{-3L}L^2\) from its declared exact primitives. Preserve these different case definitions. Do not reuse a float-rounded `q`, `log_q`, or `delta` from a result record: that would certify a nearby model rather than the defined case. Store defining expressions/inputs alongside display approximations. Its interval support is documented as experimental [R9]; the early gate must verify the elementary operations actually used, endpoint containment, zero branches, and behavior at ambiguous comparisons. Do not call unsupported interval ODE or optimization routines. Increase to 80 digits once if a decision is ambiguous; otherwise abstain.

The certified \(\phi_2\) path uses \((\exp z-1-z)/z^2\) in interval arithmetic when the interval excludes zero, and exactly \(1/2\) when \(z=0\). A series is permitted in that path only with an explicit outward remainder bound; a point-accuracy Taylor stopping rule is insufficient. If a nonzero-width interval straddles zero, subdivide it or abstain unless a verified series enclosure is available. The same restriction applies to any series for the small Poisson tail. Ordinary series remain acceptable for the separate floating point-prediction path.

A floating root finder may suggest points, but retain interval-verified signs at bracketing endpoints. For \(t_L\), choose a point where the **upper** evaluation of \(u_3\) is below \(\eta\). A candidate \(t_0\) need not equal its cutoff root exactly, because (10) holds at any positive point. For \(t_U\), use an upper bound on \(E\), a lower positive bound on \(z\), and an upper scalar delay. Evaluate \(F_{\rm hi}\) at this outward upper time. For no-go, require strict separated intervals in both inequalities (12).

Check the elementary interval layer against 80-digit point evaluations and analytically solvable special cases. This is software validation, not a formal verification of mpmath. If elementary interval support cannot be made reliable within the repair allowance, retain the exact mathematical derivations and report computations as **numerically checked bounds**; remove machine-certificate claims. Never replace outward arithmetic by extra digits while keeping the same certification label.

Root goals: time brackets of width at most \(10^{-8}\) for scalar estimates; rate endpoint relative width at most \(10^{-4}\), unless the probability truncation remainder prevents it. Interval loops are capped at 100 bisections. At a boundary whose sign cannot be separated, output unresolved rather than rounding into a favorable classification.

### 6.4 Reference risk precision

At fixed numerical \(t_S\), target a total probability-reference width of \(10^{-5}\), one thousandth of \(\alpha\). Seek a reference rate bracket of at most 0.5% relative width. These serve different claims: the former is the target for a fine probability comparison, while the latter determines whether a point-rate error comparison is sufficiently resolved. If only one target is met, report which comparisons it supports; neither substitutes for the analytic primary interval. Use (14) with its analytic tail, the bounded area calculation, and outward scalar arithmetic; intersect with (11). Evaluate at \(t_S\pm10^{-7}\), expanded if validation indicates greater timing uncertainty. These are **validated numerical reference intervals conditional on that timing uncertainty**, not formal ODE enclosures. The primary safe/no-go outputs instead use rigorous \([t_L,t_U]\).

Invert the reference by interval decisions, not by blindly root-finding the truncated expression \(F_2\), which need not be globally monotone in \(\nu\). Start from the finite analytic bracket \([\nu_S,\nu_N]\). At a trial rate, evaluate the combined probability interval over the numerical time uncertainty. Raise the reference rate's lower endpoint only if the probability upper endpoint is at most \(\alpha\); lower its upper endpoint only if the probability lower endpoint exceeds \(\alpha\). Monotonicity of the true \(F_0\), obtained by coupling extra Poisson arrivals, justifies these updates. If the midpoint straddles \(\alpha\), refine the geometry within its cap if that can help; then permit the two quartile points to bracket the ambiguous region. If the tail or time uncertainty still prevents the rate-width goal, return the remaining interval as unresolved. Stop at 100 rate evaluations. If the initial necessary limit is infinite, retain an unresolved boundary and the specified fixed-rate stress results; no unbounded rate search is required. Store this numerical reference bracket separately and never use it to overwrite an analytic certificate.

Do not continue refining panels when the Poisson tail alone exceeds the goal. Record which uncertainty dominates: clean-time enclosure, numerical time validation, two-event geometry, three-or-more-event tail, or roundoff. This attribution determines whether the method is limited or merely insufficiently evaluated.

### 6.5 Planned commands

The following CLI interfaces must be created; they have not been run on the user's Mac. The venv commands are conventional setup commands, subject to actual Python availability.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy matplotlib mpmath pytest
.venv/bin/python -m pip freeze > requirements.lock.txt
.venv/bin/python -m leak_window inspect
.venv/bin/python -m pytest -q
.venv/bin/python -m leak_window run --suite development --phase smoke
.venv/bin/python -m leak_window run --suite development --phase calibration
.venv/bin/python -m leak_window run --suite development --phase pilot
.venv/bin/python -m leak_window freeze
.venv/bin/python -m leak_window run --suite core --phase confirmation
.venv/bin/python -m leak_window run --suite stress --phase robustness
.venv/bin/python -m leak_window report
.venv/bin/python -m leak_window reproduce --from results
```

`freeze` must record configuration and code hashes, versions, selected full/reduced grid, the arithmetic settings, metric definitions, resource limits, and a timestamp before opening confirmatory results. `reproduce` regenerates the essential plots/tables from raw results without rerunning experiments. Provide a separate documented switch to rerun the authorized computation; it must honor the existing cumulative ledger.

## 7. Experiments, baselines, and frozen evaluation

### 7.1 Methods and fairness

| Role | Method | Information and limitation |
|---|---|---|
| Proposed decision mechanism | Finite comparison bounds (10), Poisson envelope (11), safe/no-go tests (12) | Uses the known model and exact parameters; no fitted reference timings. |
| Theorem point predictor | \(t_H\), then \(\widehat\nu_H=c_\alpha/(t_H-d)\) when positive | Single-early-event approximation; not a certificate. Report invalid denominator explicitly. |
| Closest strong inexpensive comparator | Full finite-time \(u_3\), saturation-transformed root \(t_V\), corresponding point rate | Uses exactly the same parameters and scalar root budget; not an exact nonlinear guarantee. |
| Simple reference/ablation | Delete only the \(-\tfrac23\log L\) term from \(t_H\), retaining its offset and detuning amplitude | Isolates the log-log contribution; not the only comparator. |
| Targeted risk ablation | Replace both probability bounds by the old-event expression \(F_{\rm lo}\) | Shows whether omitting multiple arrivals creates unsafe optimism. |
| Nonlinear and probability reference | Scaled clean ODE plus exact negative scalar event-count conditioning | Same equations, input, leak, threshold and parameters; includes setup and precision costs. |
| Architecture counterfactual | Direct scalar rate-\(r\) amplifier | Same normalization and reporter leak; recognition rewiring is an explicit unvalidated assumption. |

No method receives reference-derived calibration offsets. Use one fixed four-cutoff procedure throughout. A timing advantage is not credited if it comes from a looser error target or an uncertified decision. Count case preparation, comparator construction, roots, interval checks, reference geometry, and any tuning in total cost. Show standalone point-predictor times and the cost of the complete safe/no-go workflow separately.

### 7.2 Development and core cases

All inputs are deterministic synthetic cases chosen to isolate the model, not experimental assay measurements. Save the exact generated case list and its hash.

**Development only:** \(L\in\{6,16,64\}\), \((k_2,k_3)=(1,1)\), \(\delta=0\), \(\eta=0.5\), and \(n_0\in\{1,10\}\), with \(\log q=-L-\log n_0\). Use these for debugging, tolerance selection and resource calibration. They are not part of the confirmatory score.

**Locked core:** the Cartesian product below, with \(\eta=0.5\), \(\alpha=0.01\), \(\beta=0.05\).

| Parameter | Values | Purpose |
|---|---|---|
| \(L\) | 8, 12, 20, 40, 80 | Finite inputs through asymptotic stress, reported separately. |
| \((k_2,k_3)\) | (0.5,0.5), (1,1), (2,2), (0.25,4), (4,0.25) | Different gain and equal-product controls exposing intermediate saturation. |
| \(\gamma\), with \(\delta=\gamma/L\) | -3, 0, 3 | Source theorem's compact critical-detuning regime. |
| \(n_0\), with \(q=e^{-L}/n_0\) | 1, 10, 100 | Discrete-seed normalization scenarios. |

This is **75 distinct clean ODE cases and 225 risk-boundary cases**. Reuse clean times across \(q\), and cache probability geometry across rate evaluations. For each case report \(\nu_S,\nu_N\), the numerical reference interval for \(\nu_*\) if resolved, \(t_L,t_U,t_S,t_H,t_V\), and classifications at \(0.8\nu_S\), \(1.2\nu_N\) when finite, and the horizon's proposed rate when defined. The use of computed boundary-relative trial rates is predeclared; it tests certificates and is not an independently sampled application distribution.

If calibration requires the reduced core, use exactly \(L\in\{8,20,80\}\), retaining every coupling, detuning and seed-count value: 45 clean cases and 135 risk cases. Decide before freezing or inspecting core outcomes. Report the reduced scope explicitly. Do not remove only unfavorable or expensive parameter regions.

### 7.3 Targeted stress cases

Run only after core validity is established and within the reserved allocation. They test failure boundaries; they do not enter the primary success fraction.

| Stress | Fixed cases | Decision it informs |
|---|---|---|
| Very finite and more asymptotic input | \(L=4,160\), all five coupling pairs, \(\delta=0,\eta=0.5\), all three \(n_0\) | Whether usefulness depends on unattainably asymptotic inputs; finite precision at large \(L\). |
| Fixed rate mismatch | \(L=8,20,80\), \(\delta=-0.15,+0.15\), \(k_2=k_3=1,\eta=0.5\), all three \(n_0\) | Finite comparison remains valid, while the compact-detuning theorem must not be silently extrapolated. |
| Threshold choice | \(L=8,20,80\), \(\eta=0.25,0.75\), exact resonance and unit couplings, all three \(n_0\) | Whether the result depends on the particular reporter threshold. |
| Critical-size continuous seeds | \(L=20\), unit couplings and exact resonance, \(q=\kappa(1/2)e^{-60}20^2\), \(\kappa=0.1,1,10\), with \(\nu t_S=0.01,0.1,1,3\) | A regime where early-event risk bounds and the no-arrival reduction can lose usefulness; explicitly submolecular under some normalizations. |
| Direct fast-amplifier control | \(L=8,20,80\), unit couplings for cascade, \(r=3,\eta=0.5,n_0=1\) | Whether a speed claim survives a directly seeded fast scalar comparator. Report scalar clean time and its safe envelope at cascade trial rates. |

The first three rows add 22 clean cases and 66 risk-boundary cases. The fourth adds 12 fixed-rate risk cases, not an unbounded new boundary search. The scalar control is analytic. No growing architecture search, stochastic-chemistry fallback, or high-order Poisson expansion is authorized by this plan.

### 7.4 Independent correctness controls

Before interpreting comparative results, require the following substantive checks:

1. Exact scalar cubic flow, delay and threshold recovery, including \(\nu=0\), \(q\ge\eta\), \(\tau<d\), and both small and moderate seed sizes.
2. Direct differentiation or independent quadrature of (6) for zero, positive and negative detuning; correct \(\phi_2(0)=1/2\). Verify the three-stage degree recursion and constants separately.
3. Clean numerical trajectories inside the positive-comparison bounds; exact \(x_1\) agreement; \(t_L\le t_S\le t_U\) beyond numerical uncertainty. Negative components or a crossing outside the bound stop interpretation.
4. Small-\(L\) scaled versus unscaled ODE, and DOP853 versus Radau on representative extreme parameter cases. No clipping of negative solver states to conceal an error.
5. Deterministic zero-, one-, and two-arrival traces checked by successive exact scalar flow. Independently integrate the two-arrival region on a small two-dimensional mesh, and confirm convergence into the one-dimensional area enclosure. This catches an ordering or factor-of-two mistake.
6. A handful of small-\(L\) positive/negative paths with identical prescribed arrival times, including late and closely spaced jumps, to verify target dominance and persistence above threshold. These are pathwise checks, not estimates of FPR.
7. A model validator that rejects negative coupling, nonpositive drift, feedback, nonzero initial H0 reporter, source-stage leakage, or noisy observation when asked to apply this certificate. A rejected assumption is not a simulated scientific refutation.

### 7.5 Metrics and analysis rules

Report every case, including failures and infinite endpoints. Primary metrics are \(W_\nu\), the safe/impossible/unresolved counts, and certificate consistency. Also report relative uncertainty of the nonlinear probability reference, safe deadline overhead \((t_U-t_S)/t_S\), point timing errors, and point rate errors against the entire reference interval. If the reference interval does not establish an ordering between two errors, call the comparison unresolved.

For the asymptotic predictor, record how often its proposed rate is shown unsafe by a risk lower bound; “small timing error” is not an adequate risk metric. Freeze the point-rate error as \(|\widehat\nu/\nu_*-1|\), and the timing error as \(|\widehat t-t_S|\). Propagate the full reference interval into an error interval: the minimum is zero if the point lies inside, otherwise the smaller endpoint error; the maximum is the larger endpoint error. Report medians and maxima of these error intervals, plus casewise wins/losses only where intervals separate. Report the log-log ablation effect separately from the comparison to \(t_V\). Equal-product coupling cases are especially informative: identical leading amplitudes need not imply identical finite saturation behavior.

Issue separate conclusions for **screen usefulness** and **incremental value of the asymptotic predictor**. A passing primary screen does not imply asymptotic superiority. Only describe better aggregate predictor accuracy if the median error intervals separate and the stratum-level results disclose exceptions. Only claim a computational advantage at comparable attained accuracy and with end-to-end costs included; disclose the absolute saved time even when a relative speed ratio is large. For fixed nonzero detuning, label \(t_H\) `outside_theorem` rather than silently substituting \(\gamma=\delta L\) into a compact-parameter guarantee.

These are deterministic evaluations of probabilities, so independent random seeds and conventional Monte Carlo confidence intervals are not needed for the core. Use analytic truncation/interval bounds and numerical-refinement discrepancies. Do not treat 225 grid cells as independent clinical samples. Timing repetitions quantify runtime variability only, not scientific uncertainty about the model.

Produce at most four essential figures: (1) leakage-rate safe/necessary/reference bands against \(L\), with separate detuning panels; (2) timing/rate error of horizon, finite-time comparator and ablation; (3) safe/impossible/unresolved map including selected stress cases; (4) total-cost/accuracy comparison if timing resolution supports it. Include dimensions and model assumptions in captions. Show failures, gaps and out-of-scope cases explicitly.

## 8. Mac feasibility, budgets, and evidence gates

### 8.1 Initial inspection and calibration

Inspect `sw_vers`, `uname -m`, the CPU description, installed RAM, available memory, free disk, Python architecture/version, actual NumPy/SciPy versions and BLAS configuration. Save results. The M3 Max name does not establish RAM, free storage, OS version or installed backends. Use native ARM64 where possible; document translation if unavoidable. The planning calculations in §9 were performed on hosted Linux x86-64 and establish no Mac performance claim.

Set a single-process RSS ceiling of `min(2 GiB, 20% of installed RAM)`, and require enough currently available memory for that reservation before the main run. Keep generated results below 500 MiB and all project/dependency/cache storage below 3 GiB. Do not save dense trajectories for every case; retain crossing summaries and only designated diagnostic traces.

Use one CPU worker and one BLAS thread for comparable timings. Measure complete subprocess wall time, CPU time, and peak resident memory with macOS `/usr/bin/time -l`; supplement with in-process `perf_counter_ns` for phases. Record the units reported by the platform. Do not assume Linux `ru_maxrss` units on macOS.

Calibration uses the three development clean cases and both seed sizes. Measure ODE, scalar certificate, interval endpoint, and probability-geometry costs separately. Predict full cost from observed representative and worst-case costs, with a factor-of-two reserve; account for both time-uncertainty endpoints and caching. If the full grid exceeds the 110-minute locked-comparison allocation, select the predeclared reduced grid before freeze. If that also exceeds the allocation, stop as practically infeasible or deliver only the valid analytic diagnostic. Do not lower accuracy to make the timing estimate fit.

### 8.2 Hard cumulative limits

The experimental-compute ceiling is **240 minutes on the Mac**, including failed runs, development, calibration, tuning, timing repetitions, checks, repairs, fallback work and report generation. A restart or new approach does not reset it. These are ceilings, not targets to exhaust. Expected compute is minutes to roughly an hour for a working implementation, but this is an unmeasured planning estimate until calibration.

| Allocation | Maximum local compute | Maximum work |
|---|---:|---|
| Constructive/environment gate | 5 min | Inspect environment; verify scalar arithmetic and data independence. |
| Tiny correctness | 15 min | Analytic oracles and independent numerical checks. |
| Resource calibration | 10 min | Representative cases and full/reduced cost prediction. |
| Mechanism pilot | 20 min | Development cases and informative safe/no-go/refinement examples. |
| Locked core plus permitted stress | 110 min | At most 400 distinct risk cases and 400 ODE invocations cumulatively across the whole project. |
| Diagnosis, figures and report | 35 min | Preserve all evidence and deliver conclusions. |
| Bounded repair reserve | 30 min | At most two repair cycles, each addressing a named defect. |
| Unallocated evidence-preservation reserve | 15 min | Checkpoints and graceful completion, not more hypothesis search. |
| **Total** | **240 min** | All branches counted together. |

Count every adaptive reference attempt and method repeat in the ledger. Scalar root evaluations and quadrature panels are not separate experimental cases, but their time is counted and their numerical caps apply. Record them as operation counters. Limit a clean ODE invocation to 30 seconds and a probability-geometry attempt to 20 seconds; a timeout triggers one diagnosed repair or an unresolved record, not an open retry loop. The first unexpected timeout stops automatic allocation of similar work until calibration is updated.

For runtime comparisons use one cold subprocess run and five warmed timing batches on each of three representative development cases. A batch may repeat cheap scalar work to exceed 0.1 seconds, up to 1,000 iterations. Time only after imports for warm results; include imports, preparation and all scalar validation in cold end-to-end results. ODE batches use at most five solves each and count toward the 400-invocation cap. Change method order deterministically between batches to reduce thermal/order effects. Record medians and ranges, power state and obvious interruptions; do not publish a precision speedup from sub-clock-resolution measurements.

### 8.3 Stage decisions

| Gate | Question and minimum evidence | Continue | Bounded repair / fallback / stop |
|---|---|---|---|
| G0: constructive feasibility | Can equations, elementary interval operations and required local dependencies be executed? Produce environment record, derivation audit and source-access ledger. | All essential expressions and model assumptions are explicit; no missing dataset or theorem oracle. | Repair an ordinary installation/notation defect within reserve. If interval support fails, downgrade machine certification as stated. If Mac absent, complete derivations/code preparation and mark all Mac experiments blocked. |
| G1: tiny correctness | Do scalar oracles, detuned comparator, scaled ODE and event-count geometry agree? | All checks in §7.4 pass at declared precision. | At most two defect-driven repair cycles globally. Persistent contradiction means invalid experiment; preserve reproducer and stop comparative claims. |
| G2: calibration | Is the frozen accuracy attainable within memory/time/disk limits? | Full grid or predeclared reduced grid fits with factor-two reserve. | Select reduced grid before seeing results. If neither fits, report practical infeasibility; do not substitute a scientifically irrelevant toy. |
| G3: pilot | Can the design distinguish a safe point, an impossible point, and a deliberately unresolved regime? Does the reference resolve the planned 20% effect? | At least one independently checked safe and impossible development case; core-like reference rate uncertainty at most 0.5%; arithmetic stable. | Apply (14) if it has not yet been used. Once its exact three-or-more-arrival tail dominates, extra panels cannot remove it: retain the interval and report the precision limitation. No additional event-order integrator is authorized. A wide pilot alone is not evidence that the theorem is false. |
| G4: locked comparison | Does the diagnostic meet the utility criterion, and does the theorem add value against the strong comparator? | Run frozen core, then only the listed stress cases within budget. | Do not tune on core outcomes. A pure implementation correction requires documented affected-case reruns; changing the hypothesis, metric, baseline or grid makes subsequent findings exploratory. |
| G5: diagnosis/report | What conclusion is supported, and what remains blocked? | Deliver reproducible evidence with one of the categories in §10. | Stop optional testing when the decision is sufficiently supported. If budgets expire, report achieved precision and unresolved cells, not a negative scientific conclusion. |

Reserve the report allocation before launching confirmation. Write resumable JSONL results after every case. At 80% of a stage allocation, launch another case only if its measured worst-case cost fits with a 20% margin. At the cumulative cap, stop computation, preserve results, and report.

### 8.4 End-to-end effort estimate

Allow approximately 30–60 minutes for setup/access inspection, 3–6 hours of implementation and review effort, 0.5–2 hours of expected local experiments, and 1–2 hours of interpretation/reporting. These are planning estimates, not measured promises; the hard local-compute cap remains four hours even if implementation takes longer. The coding agent's hosted reasoning is not experimental compute. Do not turn an inconclusive study into a days-long rescue effort.

## 9. Planning calculations already performed

These observations helped design the study. They are not confirmatory evidence and must not be mixed into its locked results. They used a hosted Linux x86-64 environment, Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0. They include an independent mathematical review, small scaled-ODE probes, and scalar root/quadrature calculations; no M3 Max timings or completed PoC are claimed.

For \(L=20,k_2=k_3=1,r=3,\eta=0.5,q=e^{-20}\), a single clean DOP853 check with `rtol=2e-11`, `atol=2e-12`, `max_step=0.25` gave:

| Quantity | Planning value, dimensionless |
|---|---:|
| Clean comparison lower time \(t_L\) | 18.0704802137 |
| Scaled nonlinear reporter crossing \(t_S\) | 18.1148427067 |
| Four-cutoff upper time \(t_U\) | 18.1505129628 |
| Theorem point time \(t_H\) | 18.0507921630 |
| Finite-time saturation comparator \(t_V\) | 18.1167233753 |
| One reporter-seed maturation delay \(d\) | 6.4835646186 |
| Sufficient rate \(\nu_S\), ordinary floating calculation | 0.0008601087494 |
| Necessary upper rate \(\nu_N\) | 0.0008673866458 |
| Relative bracket width | 0.00846160, about 0.85% |

The selected cutoff was \(m=0.1\); the other three candidates were also evaluated as specified. These floating calculations did not use outward intervals. Treat their decimals as reproducibility targets with tolerances, not certified endpoints. The useful warning is that \(t_H<t_L\): in this finite example, treating the theorem's point estimate as an exact target crossing would violate an independent bound. The strong finite-time point comparator was much closer to the numerical crossing.

At the displayed sufficient rate, an ordinary adaptive two-event quadrature gave a probability interval of approximately [0.00995376, 0.00995438] after adding the exact three-arrival tail; the quadrature error estimate itself was not certified. The locked implementation must use the bounded geometry in §4.5. This check only established that a sharper deterministic reference appears feasible.

An independent planning audit also checked the proposed initial-seed scales \(b_2=c_2\epsilon^2L\), \(b_3=c_3\epsilon^3L^2\), beyond A1's bounded weighted seeds, using twelve small scaled integrations. The comparator gives reporter entrance amplitude \(k_2k_3/2+k_3c_2+c_3\), and the observed finite-time residuals decreased with \(L\). This is supporting analysis, not a second project: **do not implement a broad multi-source extension in this PoC**.

For the same \(L=20\) normalization with one target molecule, (16) is approximately \(1.18\times10^{15}\). This is a model-scale calculation, not a measured contamination rate. It explains why background event probabilities deserve explicit treatment.

## 10. Interpretation, stopping, and final deliverables

### 10.1 Required outcome categories

Distinguish a **method-level conclusion** from a **detector-level conclusion**. If valid calculations show that fewer than 90% of core cases meet the specified width/classification criterion, the primary screening-use hypothesis fails on this grid: that is an informative negative result about this screen. A wide bound can simultaneously leave the detector's actual feasibility unresolved. If a numerical defect or missing reference prevents validating the attempted comparison, use the invalid/inconclusive category instead. A supported finite screen and a negative or unresolved comparison for the asymptotic predictor may coexist.

| Outcome | Evidence required | Permitted conclusion |
|---|---|---|
| Supported contribution | Correct finite bounds and numerical references; utility criterion met across the declared core; source and numerical limitations disclosed | The diagnostic resolves useful safe/no-go decisions in this model. Attribute any distinct benefit of the horizon only if its strong-comparator comparison supports it. |
| Informative negative result | Valid implementation showing failure of the primary utility criterion, adequately resolved unfavorable predictor comparisons, or a verified inequality (12) | This screen is insufficiently informative on the declared grid, the predictor adds no supported advantage, or the specified detector setting cannot meet the probability constraints. Identify which conclusion is established. |
| Practical infeasibility | Measured resource costs or a supported scaling estimate exceed the caps while preserving necessary accuracy | This implementation/study is not feasible under the stated local budget. This does not refute the mathematics. |
| Inconclusive evidence | Probability tail, timing uncertainty, missing access or budget prevents a comparison; analytic bounds may leave an individual detector's feasibility unresolved | State exactly which interval or missing condition prevents the decision. Do not relabel a demonstrated failure of the screen's predeclared width criterion as merely inconclusive. |
| Invalid experiment | Unrepaired contradiction, incorrect model/observation, inconsistent baseline or numerical artifact | Preserve the failure reproducer; withdraw empirical comparisons affected by it. A bug is not a scientific negative result. |

If the inexpensive finite-time comparator matches or beats the horizon at negligible cost, state that plainly. If only very large \(L\) looks favorable, the practical bridge remains weak. If the scalar fast-amplifier control is better under its assumptions, report it without inventing cost or recognition restrictions. A clinically useful claim would additionally require a justified count-level reaction model, measured leakage and rate variation, recognition/capture data, readout noise and external validation; this study supplies none of those.

A justified analytic stop is a completed research outcome. Do not force a large benchmark after finding a decisive defect or obstruction. Conversely, one particular no-go case is not a reason to skip the finite regime map when other authorized cases remain informative and affordable.

### 10.2 Required artifacts for stages actually reached

Deliver runnable local code and a concise README if implementation occurred; exact configuration/protocol hashes; environment and dependency records; derivations with assumptions; complete raw case records including failed/time-limited runs; essential tables and figures; and `REPORT.md`. Include a command that regenerates the report's quantitative tables from saved raw results.

The report must lead with the research decision, then give the mathematical model, evidence, comparisons, resource use and limitations. It must separate supplied theorems, new derivations, numerical findings, planning assumptions, and unperformed work. Include the actual number of attempted/completed/failed cases, cumulative compute and setup costs, actual hardware/backend/precision, and any deviations from the frozen protocol.

At an analytic stop, the necessary deliverable is the verifiable derivation or counterexample plus a minimal reproducer if computation was involved. At a blocked-hardware stop, deliver completed code/analysis and identify exactly which Mac checks remain unperformed. No source dataset, remote service, paid computation or patent clearance is a prerequisite for the scoped numerical study; publication-level novelty remains provisional because of the access and coverage limits below.

Recommend at most one specific follow-up, justified by the outcome. The likely next step, if useful finite bounds survive, is obtaining a chemically defensible count-level model and measured reporter-initiation process. Do not automatically propose a full diagnostic platform.

## 11. Claim ledger and review dispositions

### 11.1 Central claim-to-evidence map

| Claim | Evidence status | Remaining uncertainty / resolving check |
|---|---|---|
| Support recursion selects first-transition clocks and cubic profiles | Established in A1 under its assumptions; inspected proof | No historical-priority conclusion; independently reproduce essential constants and finite examples. |
| The unmodified horizon is accurate at finite inputs | Hypothesis, not a supplied guarantee | Locked comparison with nonlinear reference and \(t_V\); do not infer accuracy from asymptotics alone. |
| Equations (10)–(12) give finite safe/no-go conditions | Derived here with complete comparison/event arguments | Review algebra; verify elementary interval implementation and pathwise controls. |
| Two-event probability conditioning is a trustworthy reference | Derived exact event decomposition plus explicit remainder | Validate arrival ordering, area enclosures, numerical clean-time uncertainty, and truncation width. |
| The resulting screen is useful | Primary empirical/computational hypothesis | Predeclared 20% bracket and coverage criterion, resource measurement, failures included. |
| A real molecular detector can match this ODE and leak law | Unverified model assumption | Requires a separate chemical/experimental study; no conclusion here. |
| Background and sampling constrain low-input detection | Elementary model derivations (16)–(17), supported by relevant prior literature context | Actual rates, capacities and capture law unknown; no conversion to clinical performance. |
| The project is historically novel | Unresolved | Restricted claim only; full equation-level comparison to inaccessible R2 and broader priority review would be needed. |

### 11.2 Material review findings and how the plan handles them

Three distinct planning reviews covered mathematics/novelty, experimental validity, and execution. Material issues were reconciled in the specification; this is a disposition summary, not a claim that no further defect is possible.

| Stress or defect | Disposition |
|---|---|
| Source proof uses compactness / appears nonconstructive | Coefficients are explicitly computable; finite decisions use an explicit damping sandwich instead of an unknown asymptotic error oracle. |
| Proposed extension already exists | A1 §10's weighted seeds are acknowledged; ongoing reporter initiation and finite risk decisions are the scoped subject. |
| A strongly related mechanism is already known | R2/R5/R6 constrain novelty. Equivalent prior work changes attribution, not the validity of a replication or limitation study. |
| Realistic assumptions fail | Discrete-count audit, fixed-detuning stress, and strict model validation; unstable-zero conclusions do not transfer to stable-zero circuits. |
| Favorable synthetic inputs manufacture success | Fixed finite and asymptotic strata, equal-product controls, threshold/drift stresses, and explicit no-real-data scope. |
| Tuned strong baseline removes the gain | Full finite-time comparator is mandatory; no theorem-superiority claim if it removes the advantage. |
| Background differs between labels | Identical reporter initiation law under both labels; safe/no-go proofs allow leakage to help the target. |
| Failure at clean deadline is mistaken for global infeasibility | All-deadline certificate (12) and the no-arrival lemma state the necessary extra condition. |
| Finite precision erases microscopic signals | Exactly scaled ODE, log calculations, independent small cases and scalar interval validation. |
| Interval code certifies rounded inputs or an unbounded series truncation | Derive normalized parameters inside the interval context from primitive exact inputs; use elementary expressions or explicit series remainders. |
| A truncated probability expression is treated as globally monotone | Invert the true-probability enclosure by verified inequalities; retain ambiguous rate intervals. |
| Preprocessing or interval arithmetic dominates | Include construction/validation/cache costs and cold totals; no hidden amortization. |
| Runtime/memory exceeds estimates | Actual Mac calibration, fixed reduced grid, cumulative caps and graceful stop. |
| Small pilot is inconclusive | Separate reference precision failure from method failure; allow only the declared bounded refinement. |
| Source or original code unavailable | Essential equations reproduced here; missing original code is disclosed. R2 full text remains a novelty limitation. |
| Primary hypothesis is false | Report the limitation or informative negative result; no alternative-project rescue campaign. |

## 12. Research coverage and references

Research and source inspection were performed on **27 October 2026**. Searches covered autocatalytic background amplification/detection limits, the specific 2025 kinetic-programming paper, stable-zero/bistable molecular leak absorption, branching-process escape with random time shifts, and dominant-monomial/tropical reduction. Backward and forward neighborhoods were followed where they affected the project choice: earlier branching/ODE theory identified by R5, and the continuation of deactivation-based circuits into molecular detection. Those older originals and all supplementary materials were not exhaustively audited. This was a targeted source and novelty screen, not a systematic review or proof of global priority.

Accessible source content was read, not just titles. Some publisher/PMC pages intermittently returned access checks; author-hosted or indexed primary content recovered the relevant mechanism descriptions. R2's full text remained inaccessible after publisher and repository attempts. No experimental data for the exact resonant cascade were supplied or verified. Avoid numerical comparisons to incompatible published assays.

**[A1] Supplied mathematical note.** Author not identified. *Separation horizons in positive resonant cascades: A nonlinear saturation theorem and its critical-detuning extension*. 27 October 2026. Attachment `resonant_horizon_research(1).pdf`, 17 pp. Critical locations: assumptions and recursion, p. 2, equations (1)–(4); theorem/comparison, pp. 3–5, equations (5)–(11); saturation scope, p. 6; numerical checks, pp. 7–8; novelty/limits, pp. 9–10; detuning, pp. 11–13, equations (13)–(18); weighted initial data and random source, pp. 13–14; every-coordinate transition proof, pp. 15–17, equations (20)–(27). No public stable URL was supplied. Preserve the attachment and its hash when available.

**[A2] Supplied application/request.** Author not identified. `Pasted text(20261027-161608).txt`, supplied 27 October 2026. Contains the proposed molecular-detection application, illustrative three-stage timings, leakage argument, and execution-brief requirements. No external public URL supplied.

**[R2]** Mathieu Branca, Corentin Calvet, Benoît Limoges, François Mavré. *Specific Versus Non-specific Response in Exponential Molecular Amplification from Cross-Catalysis: Modeling the Influence of Background Amplifications on the Analytical Performances.* **ChemPhysChem 22**(15), 1611–1621, 4 August 2021. DOI: [10.1002/cphc.202100342](https://doi.org/10.1002/cphc.202100342). [PubMed record and abstract](https://pubmed.ncbi.nlm.nih.gov/34038617/). Access scope: abstract and metadata; full equations not verified.

**[R3]** Guichi Zhu, Dominic Lauzon, Carl Prévost-Tremblay, Arnaud Desrosiers, Bal-Ram Adhikari, Alexis Vallée-Bélisle. *Kinetically Programmed Signaling Cascades for Molecular Detection.* **Journal of the American Chemical Society 147**(43), 39452–39463, online 16 October 2025, issue 29 October 2025. DOI: [10.1021/jacs.5c12059](https://doi.org/10.1021/jacs.5c12059). [Author-hosted full text](https://www.nanomachineslab.org/wp-content/uploads/2025/11/AVB064.pdf). Critical location: Fig. 1 and Results, pp. 39453–39454, for the actual inhibitory/competitive mechanism and kinetic hierarchy. Relevant full-text passages read.

**[R4]** Kevin Montagne, Guillaume Gines, Teruo Fujii, Yannick Rondelez. *Boosting functionality of synthetic DNA circuits with tailored deactivation.* **Nature Communications 7**, 13474, 15 November 2016. DOI: [10.1038/ncomms13474](https://doi.org/10.1038/ncomms13474). [Full article](https://www.nature.com/articles/ncomms13474); [PMC copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC5116077/). Critical locations: Results, Figs. 2–3 and discussion of unstable versus stabilized zero. Relevant primary content read; no full supplementary-model replication undertaken.

**[R5]** A. D. Barbour, K. Hamza, Haya Kaspi, F. C. Klebaner. *Escape from the boundary in Markov population processes.* **Advances in Applied Probability 47**(4), 1190–1211, December 2015. DOI: [10.1239/aap/1449859806](https://doi.org/10.1239/aap/1449859806). [Open preprint](https://arxiv.org/abs/1312.5788), v3 dated 14 October 2014. Critical locations in the open PDF: §1.1, pp. 5–6, and Theorem 1.1 with interpretation, pp. 8–9. Read assumptions and result; no assertion that its irreducible branching hypotheses cover (3).

**[R6]** Vincent Noël, Dima Grigoriev, Sergei Vakulenko, Ovidiu Radulescu. *Tropicalization and tropical equilibration of chemical reactions.* In **Tropical and Idempotent Mathematics and Applications**, Contemporary Mathematics 616, 261–277, 2014. DOI: [10.1090/conm/616/12316](https://doi.org/10.1090/conm/616/12316). [Open preprint](https://arxiv.org/abs/1303.3963), v2 dated 27 May 2013. Critical locations: §2, Definitions 2.1–2.2, and §3's distinction between heuristic reduction and controlled regimes. These relevant sections were read.

**[R7]** World Health Organization. *WHO recommends near point-of-care tests, tongue swabs, and sputum pooling for TB diagnosis.* Departmental update, 9 March 2026. [Official announcement](https://www.who.int/news/item/09-03-2026-who-recommends-near-point-of-care-tests--tongue-swabs--and-sputum-pooling-for-tb-diagnosis). Used only to verify the application motivation; not a performance source for this PoC.

**[R8]** SciPy community. *scipy.integrate.solve_ivp*, online reference documentation, accessed 27 October 2026. [Official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). Relevant entries: DOP853/Radau, `rtol`/`atol`, terminal/directional events and the sign-change limitation. The accessed documentation advertised v1.18.0; planning probes used v1.17.0. Freeze and report the Mac's actual tested version.

**[R9]** mpmath developers. *Contexts: arbitrary-precision interval arithmetic (`iv`)*, mpmath 1.3.0 documentation, accessed 27 October 2026. [Official documentation](https://mpmath.org/doc/current/contexts.html). Relevant entries: interval containment, exact decimal-string construction, and the warning that interval support is experimental. Its elementary subset is an explicitly tested dependency, not an assumed formal-verification service.

---

**Execution instruction:** perform the authorized gates autonomously, preserve unfavorable findings, repair ordinary implementation defects within the stated limits, and deliver the best justified research conclusion reached. The successful outcome is reliable knowledge about this model and this mathematical approach, whether that knowledge supports further investment or explains why it should stop.
