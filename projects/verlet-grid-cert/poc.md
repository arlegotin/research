# POC EXECUTION BRIEF

## Certified finite-horizon energy amplification under bounded time-grid displacement

**Prepared:** 27 October 2026. **Executing agent:** `gpt-6.1-sol`, reasoning effort `xhigh`. **Target:** one local MacBook with an M3 Max; its actual memory, operating system and available storage must be inspected. **Status:** research and execution design, with explicitly identified small planning probes. The PoC and Mac benchmarks have not been completed.

You are to implement, verify, run and interpret the bounded study below. Follow its gates autonomously. A verified limitation or justified early stop is a completed research outcome. Do not turn this into an industrial solver integration, an open-ended search for favorable examples, or a new general optimization framework.

## 1. Project decision and smallest defensible contribution

### 1.1 Selected question

**Can a simple, rigorously evaluated upper bound tightly enclose the maximum artificial endpoint energy growth of a fixed harmonic oscillator integrated with velocity Verlet, over every time grid whose nodes move by at most `C/N²`, and does this provide information beyond sampled jitter and conventional accuracy checks?**

The main computational quantity is a bracket on the maximum **physical energy amplification**, for a fixed initial state, frequency and terminal time. Its lower end is the validated energy of a specific admissible grid; its upper end covers every admissible grid. Smaller bracket width is better. The principal decision is whether a 10% energy increase can be produced or ruled out under the stated grid tolerance. This is a dimensionless numerical diagnostic, not an engineering safety certification.

Use the supplied oscillatory partition theorem as a small positive control and motivation. The actual bridge is the standard harmonic oscillator and the established velocity-Verlet method. This preserves the purpose of distinguishing grid artifacts from physical predictions while making every input, reference and guarantee inspectable.

### 1.2 What is already known, and what this study may add

Variable time steps can destabilize Verlet, including period-two perturbations around dimensionless step `sqrt(2)`. This is established prior art, including Skeel [R2] and Piché [R3]. A period-two step sequence already has bounded cumulative node displacement. Implicit midpoint's exact energy preservation for this oscillator is also known. Neither the instability, its resonance center, the node rephrasing, nor its prevention is a novelty claim.

The candidate quantitative contribution is the following tightly scoped result, derived in this brief and to be independently checked in the PoC:

1. A closed, computable finite-`N` upper bound for **all** grids in the prescribed maximum-node box, with endpoint constraints included.
2. A matching asymptotic energy-amplification law in an explicitly declared high-frequency family, supported by an exact finite-`N` witness and a measured, validated finite-`N` gap.
3. A clear limitation of transfer from the attachment: at fixed physics, uniformly convergent solvers cannot have its nonvanishing asymptotic output range. A finite-resolution resonance example may already be rejected by ordinary state and refinement checks.

The bound is a specialization of established induced-norm/product reasoning, not a new generic certification algorithm. No exact equivalent of the combined finite-horizon envelope was located in the bounded literature search, but priority and publishable novelty remain unconfirmed. If close prior art makes the specialization routine, report a **verified benchmark and transfer limitation**. Do not relabel it as an original numerical method. There is sufficient value in an inexpensive, reproducible answer to this application test; a large benchmark is unnecessary to preserve a novelty premise.

### 1.3 Fixed scope and exclusions

Study one scalar oscillator, its two-dimensional first-order state, one node uncertainty set, velocity Verlet, and implicit midpoint. Use exact rotation as the solution oracle. All data are deterministically generated; no proprietary data or external simulation package is needed.

The study does not establish a vulnerability in CFD, molecular dynamics software, finite-element packages, aircraft, reactors or other deployed systems. It does not establish a full signed-output reachable interval, a complete trajectory reachable set, a sharp minimum-energy law, or robustness of all initial phases. It does not certify physical-model adequacy. It does not optimize a controller, change the physical initial condition, or treat frequency changes as grid changes. There is no GPU requirement, cloud experimental compute, dashboard, service, distributed scheduler or commercial optimizer.

The main conclusion is about **endpoint energy**, accompanied by mandatory state and trajectory diagnostics. Energy robustness does not imply trajectory accuracy.

There is already a decisive limitation on the “hidden hazard” interpretation: at the exact resonance center, the nominal Verlet trajectory has nodewise energies alternating between 1 and 1/2, despite endpoint energy 1. Routine trajectory energy monitoring therefore exposes a 50% error. Proceed to characterize the finite all-grid envelope and its limits; do not build a general auditor on the premise that this example defeats competent conventional verification.

## 2. Inputs, evidence status and source audit

### 2.1 Supplied documents and independent execution

The planning investigation read the complete nine-page PDF, including its proofs, computational claims, discussion and references. Pages containing the main construction, power extension, stochastic theorem and figure were visually checked where text extraction obscured exponents. It also read the full request document.

| Input | Original supplied location | Suggested project copy | Identification |
|---|---|---|---|
| `oscillatory_partition_threshold.pdf` | `/workspace/scratch/2571c209dae8/upload/oscillatory_partition_threshold.pdf` | `inputs/oscillatory_partition_threshold.pdf` | *A sharp π threshold for oscillatory partition sums: A mathematical investigation of resolution sensitivity*, dated 27 October 2026; no named author on the supplied document; not peer reviewed |
| `Pasted text(20261027-161632).txt` | `/workspace/scratch/2571c209dae8/upload/Pasted text(20261027-161632).txt` | `inputs/problem_request.txt` | User's application proposal and requirements for this execution brief |

SHA-256 values of the supplied bytes:

```text
oscillatory_partition_threshold.pdf
7234e242c354dfff8218ffb50e10f78aa698c5cfc7fc2f9367dd1c9f7af4c4c4

Pasted text(20261027-161632).txt
e85e6cdeee842233299ccd2265b174332cb6c060739dbf6b0f16a7319e7c788f
```

These original workspace paths describe provenance; they are not assumed to exist on the Mac. Copy the attachments if provided and record their hashes. The PDF mentions accompanying scripts, raw results, figures and LaTeX sources, but that bundle was **not attached**. Do not claim to have run its original script or reproduced its exact random stream. The equations and algorithms needed for this study are fully specified here, so missing bundle files do not block execution. If the PDF itself is unavailable at handoff, label source-file reinspection unavailable and proceed from this self-contained specification; do not invent a retrieved original.

### 2.2 Central claim-to-evidence register

| Claim | Source or derivation | Status at handoff | Resolving check or limitation |
|---|---|---|---|
| The unweighted increment prescription cannot approximate increments of one function with uniform error below 1 | [R1], Proposition 1, p.1 | Proof inspected; not the object used in the PoC | Keep it distinct from the length-weighted accumulator |
| `π/N²` permits every anchored 1-Lipschitz target, with a uniform `O(1/N)` construction error | [R1], Theorem 2 and construction, pp.2–4 | Proof inspected; finite recursion executable | Reconstruct targets and check the explicit bound below |
| `π` is sharp in the maximum-node norm | [R1], Theorem 2(2), p.4 | Asymptotic lower-bound proof inspected | Do not infer a finite-`N` jump or the full reachable set at every `0<C<π` |
| Smaller-order node shifts preserve uniform-grid harmonic behavior | [R1], Theorem 2(3), p.5 | Proof inspected | Source scope only; no separate large experiment needed |
| Power phase has radius `π/(p N^(p+1))` | [R1], Theorem 3, p.5 | Fixed-`p>0` proof inspected | Not an oscillator threshold and not an extra benchmark branch |
| Independent uniform node errors give Brownian fluctuations with variance parameter `1/2` | [R1], Theorem 4, pp.5–6 | Proof inspected; actual slopes are not exactly independent | Endpoint moment check only; no finite sample proves a functional limit |
| A uniformly convergent fixed-physics solver cannot retain an order-one reachable width | Section 3.3 below | Elementary derivation supplied | Verify assumptions before applying it to any other solver |
| Alternating-step oscillator resonance and midpoint's energy control are established | [R2–R4] | Relevant primary content inspected | Credit the known mechanism and use it as the strong comparison |
| The finite-`N` all-grid energy upper bound is valid | Sections 4–5 and Appendix A | New derivation in this planning work; independently checked algebraically, not peer reviewed | Re-derive, exact tiny checks, ball-arithmetic evaluation, witness containment |
| The changing-frequency family has maximum energy tending to `exp(4C)` | Section 5.3 and Appendix A | Matching proof supplied; small binary64 probes consistent | Separate it from fixed-frequency refinement throughout |
| Bound and witness are practically cheap and numerically reliable on the target | Implementation design | Unmeasured planning expectation | Mac calibration and precision gates |
| An auditor improves real engineering decisions | Original application proposal | Unsupported by this study | No market, deployment or industrial safety claim |

### 2.3 Literature decision, not an assumed research gap

Read [R2] pp.173–174 and [R3] §§3–5 before coding the oscillator. Their matrix products, periodic-step resonance and common metrics are the closest mechanisms. [R4] §2 explains related prescribed-step parametric instabilities. [R5] supplies the important stochastic comparison: its random **steps** and convergence assumptions are not the same probability law as independent bounded **nodes** here. General matrix-product and common-quadratic bounds are established [R6, R7].

The sewing literature [R8] supports the distinction between additive consistency and partition dependence; it does not transfer the attachment's threshold to a stateful numerical flow. Standard verification already includes consistency, refinement and accurate reference solutions [R9]. The proposed test must disclose whether those checks already reveal the bad regime.

Recent NIST guidance [R10] and nodewise topology-optimization work [R11] support the relevance of simulation credibility and mesh uncertainty. They do not validate this particular application. For [R11], only primary metadata and abstract were accessible; do not cite its uninspected methods as evidence. This study deliberately chooses a small diagnostic over a claim about industrial prevalence.

## 3. What the attachment actually supports

### 3.1 Mathematical objects and constructive algorithm

For `0<=s<t<=1`, define

\[
A(s,t)=(t-s)\cos\left(s+\frac{1}{t-s}\right),\qquad A(s,s)=0.
\]

For a strictly increasing partition `P=(t_0,...,t_N)` with endpoints 0 and 1, let `h_i=t_(i+1)-t_i`. The polygonal path starts at zero, has increment `h_i cos(t_i+1/h_i)` and is affine on each cell. Its slopes have absolute value at most 1. Define

\[
J_N(P)=N^2\max_{0\le i\le N}|t_i-i/N|.
\]

For a target `F(0)=0` with Lipschitz constant at most 1, set

\[
q_i=N[F((i+1)/N)-F(i/N)],\quad \alpha_i=\arccos q_i.
\]

Use the following construction exactly; the last cell is deliberately uncontrolled:

```text
require integer N > 2*pi and a target known to be 1-Lipschitz
u[0] = 0
for i = 0,...,N-2:
    q = N*(F((i+1)/N)-F(i/N))
    reject |q| > 1 by more than the calibrated rounding tolerance
    alpha = acos(q)  # only tiny rounding excess may be clipped, and log it
    u[i+1] = wrap_to_half_open_minus_pi_pi(u[i] + N + i/N - alpha)
u[N] = 0
t[i] = i/N + u[i]/N**2
require positive cell widths and the specified node bound
evaluate increments again from these nodes; do not substitute target slopes
return nodes, increments, cumulative polygon, geometry checks
```

`wrap(z)=z-2π floor((z+π)/(2π))` takes values in `[-π,π)`. Preserve the chosen integer wrap branches in diagnostics. If higher precision cannot decide a branch near the cut, increase precision within the stated cap. At the exact cut, `-π` and `+π` are equivalent on the circle and both satisfy the construction's bounds, but they produce different finite meshes; use `-π` under the stated convention and log any changed mesh.

With `v_i=u_(i+1)-u_i`, the exact phase identity is

\[
t_i+\frac1{h_i}=N+\frac{i}{N}-v_i+\frac{u_i}{N^2}
                      +\frac{v_i^2}{N+v_i}.
\]

This supplies an independent evaluation route. The source's explicit bound is

\[
\|X_P-F\|_\infty\le B_N:={4\pi^2\over N-2\pi}
                          +{2\pi+2\over N}+{6\pi\over N^2}.
\]

This bound is conservative and need not decrease in step with observed errors. Do not demand monotone empirical errors, exact finite-`N` endpoint attainment, or equality of positive and negative errors. The source's sharp lower bound says that a sequence whose final output tends to 1 must have `liminf J_N>=π`. It does not compute all finite-resolution extrema.

For the source probability law, sample interior `u_i` independently and uniformly on `[-π,π]`, reset both endpoints to zero, and use the same nodes. The leading cosine increments, apart from the last one, become independent through a triangular, Haar-measure-preserving transformation of circle differences. The actual accumulated path differs uniformly by `O(1/N)`. Consequently `sqrt(N) X_P` converges weakly to standard Brownian motion divided by `sqrt(2)`, and `E||X_P||∞²=O(1/N)`. Do not claim that the finite-`N` actual increments are independent or exactly centered [R1, Theorem 4].

### 3.2 Why the source is a positive control, not a consistent physical solver

The singular local phase `1/h` has derivative `-1/h²`, so `O(N^-2)` changes in widths near `1/N` change phase by order one. A physical oscillator has phase `Ω h`, whose derivative with respect to `h` is the fixed frequency `Ω`; these are different mechanisms.

The source's additivity defect is

\[
D(s,u,t)=A(s,t)-A(s,u)-A(u,t).
\]

For `s=0,u=h,t=2h` and `h=[2(2k+1)π]^-1`, direct substitution gives

\[
D(0,h,2h)=-h(3+\cos h)\sim-4h.
\]

Thus a uniform superlinear defect estimate is unavailable. The classical sewing criterion with defect bounded by `K|t-s|^(1+α)`, `α>0`, is sufficient for a unique additive accumulation, but failure of that sufficient condition is not by itself a proof of controllable instability. Stateful solvers require a composition defect such as `Ψ_(h+k)-Ψ_k∘Ψ_h` together with stability estimates; subtracting arbitrary state increments and invoking scalar sewing is invalid [R8].

### 3.3 Fixed-physics obstruction

If a fixed model, fixed terminal time and fixed output admit a uniform error estimate

\[
\sup_{P\in\mathcal P_N(C/N^2)}|Y_N(P;\theta)-Y^*(\theta)|
 \le E(h_{\max})\longrightarrow0,
\]

then

\[
R_N\le2E(1/N+2C/N^2)\longrightarrow0.
\]

This is a triangle-inequality argument, not a claim that every practical simulation has such a bound. It rules out direct transfer of the attachment's nonzero limiting range to a solver satisfying those hypotheses. A diagonal experiment with `Ω=Ω_N` or `T=T_N` changing with refinement does not contradict it. Every table and plot must distinguish those experiments.

## 4. Fixed physical problem, solvers and uncertainty set

### 4.1 Problem and output

Use nondimensional time `t in [0,1]` and

\[
q''+\Omega^2q=0,\qquad q(0)=1,\quad q'(0)=0,\quad \Omega>0.
\]

Write `w=q'/Ω`, `z=(q,w)^T`. Then

\[
z'=\Omega\begin{pmatrix}0&1\\-1&0\end{pmatrix}z,
\quad z^*(t)=(\cos(\Omega t),-\sin(\Omega t))^T.
\]

The physical energy is proportional to `q²+w²`; normalize by its initial value:

\[
G_N(P)=\frac{E_N}{E_0}=q_N^2+w_N^2,
\quad G^*(t)=1,
\quad M_N(C;\Omega)=\sup_{P\in\mathcal P_N(C/N^2)}G_N(P).
\]

The initial state remains `(1,0)` for every main comparison. A singular-vector maximizing initial state would be a different problem and is not permitted in the primary results.

### 4.2 Exactly admissible grids

Parameterize

\[
t_i={i\over N}+{u_i\over N^2},\quad u_0=u_N=0,\quad |u_i|\le C.
\]

Then

\[
h_i={1\over N}+{u_{i+1}-u_i\over N^2},\qquad
r_i=\Omega h_i=a\left(1+{u_{i+1}-u_i\over N}\right),\quad a=\Omega/N.
\]

Require integer `N>=2`, `C>=0`, and `N>2C`. This ensures positivity for every member of the box. End cells satisfy `|r_i-a|<=a C/N`; other cells satisfy `|r_i-a|<=2a C/N`. The full endpoint time is exactly 1. The two-boundary-cell formula is not defined for `N=1`; reject that input rather than extrapolating it.

The two positive and negative extremal node values differ by `2C/N²`. Always state that `C/N²` is a radius from the nominal grid, not the maximum distance between two admissible grids. Do not sample independent widths, sort nodes, reject and renormalize widths, or rescale the final time: those operations alter the set or probability law. No smoothness constraint on `u` is assumed. Alternating nodes are valid, but whether a real workflow permits them remains outside this study.

For the usual individual-step stability comparison, separately check

\[
a(1+2C/N)<2.
\]

The certificate below needs `0<a<2` and positive steps; it does not require every perturbed step to be individually stable. Nevertheless all main resonance demonstrations must meet the stronger inequality, so their growth cannot be attributed to one step crossing the usual constant-step boundary.

### 4.3 Velocity Verlet and independent implementation

For each step, use the kick–drift–kick update in scaled variables:

```text
w_half = w - (r/2)*q
q_new  = q + r*w_half
w_new  = w_half - (r/2)*q_new
```

Its exact matrix is

\[
V(r)=\begin{pmatrix}
1-r^2/2&r\\
-r+r^3/4&1-r^2/2
\end{pmatrix},\qquad\det V(r)=1.
\]

Implement the three-operation recurrence as the main evaluator and the explicit matrix as a separately written reference. Matrix products act from right to left: `z_N=V(r_(N-1))...V(r_0) z_0`. A reversed product is a substantive bug.

### 4.4 Strong corrective control and exact oracle

Implicit midpoint for this same system has the closed Cayley matrix

\[
I(r)={1\over1+r^2/4}
\begin{pmatrix}1-r^2/4&r\\-r&1-r^2/4\end{pmatrix}.
\]

It is orthogonal in exact arithmetic, so `G_N=1` for every positive-step grid. Its phase advance per step is `2 atan(r/2)`, not `r`; it can have substantial phase error. Implement both its explicit update and the independent total-angle reference. This is an appropriate strong, cheap comparison because it is established and exact for the energy property under investigation [R3, R4]. It is not proposed as a new general remedy for nonlinear oscillators.

Use exact rotation at each actual node as the trajectory oracle. Record the endpoint state error, the maximum state error over simulated nodes, the maximum absolute energy error over nodes, and the signed accumulated phase discrepancy where a reliable unwrapped reference exists. For uniform Verlet, compare `N*2 asin(a/2)` with `Ω`; for midpoint compare `sum 2 atan(r_i/2)` with `Ω`. For general Verlet, use the state-error norm as the primary accuracy control and avoid ambiguous phase-unwrapping claims.

## 5. Proposed finite-`N` bound and constructive lower bound

### 5.1 All-grid upper certificate

For `0<a<2`, define

\[
d=1-a^2/4>0,\qquad S=\operatorname{diag}(\sqrt d,1).
\]

The nominal step `S V(a) S^-1` is orthogonal. A direct calculation gives, for positive `r`,

\[
\log\|S V(r) S^{-1}\|_2
=\operatorname{asinh}\left({|r(r^2-a^2)|\over8\sqrt d}\right).
\]

For `|r-a|<=η<a`, the maximum of this expression is attained at `r=a+η`. Therefore set

\[
L(a,\eta)=\operatorname{asinh}
\left({(a+\eta)\eta(2a+\eta)\over8\sqrt d}\right),
\]

\[
\eta_b={aC\over N},\quad\eta_i={2aC\over N},\quad
\Lambda_N=2L(a,\eta_b)+(N-2)L(a,\eta_i).
\]

Submultiplicativity gives `||S z_N||<=exp(Λ_N)||S z_0||`. Because `z_0=(1,0)`, `||S z_0||²=d`, and `||z_N||²<=||S z_N||²/d`, the factors cancel:

\[
\boxed{M_N(C;\Omega)\le U_N:=\exp(2\Lambda_N).}
\]

Evaluate `2Λ_N` as the primary log upper bound; exponentiate only when safe. This computation is constant cost in `N`, apart from arithmetic precision. It needs neither a continuous optimizer nor sampled grids. It deliberately relaxes correlations between successive widths, so it need not be tight away from resonance.

For context, the same argument and determinant-one singular values give the conservative enclosure

\[
d\,e^{-2\Lambda_N}\le G_N(P)\le e^{2\Lambda_N}.
\]

Do not present this as the exact full reachable range or as an exact minimum. At `C=0` its lower end can remain loose even though there is only one grid. The one-sided maximum bracket is the primary object.

For an arbitrary unit initial phase `z_0=(cos φ,-sin φ)`, the upper prefactor would be `(d cos²φ+sin²φ)/d`. This is why the prescribed initial state must be retained.

### 5.2 Exact alternating witness with correct boundary cells

For even `N`, take interior `u_i=C(-1)^i`, and set both endpoint shifts to zero. Put

\[
a_\pm=a(1\pm2C/N),\qquad b_\pm=a(1\pm C/N),\qquad k=(N-2)/2.
\]

The full product is

\[
P_{\mathrm{alt}}=V(b_+)\,[V(a_-)V(a_+)]^{k}\,V(b_-).
\]

Thus `G_alt=||P_alt (1,0)^T||²` is an achieved lower bound on `M_N`. The opposite alternating sign swaps the plus/minus roles. Evaluate both, plus the nominal grid; take the largest **validated lower endpoint** of their energy balls as `L_N`. Use the full products, not just a spectral radius. Ignoring the first and last half-perturbed cells gives a different admissible sequence and can materially affect finite-`N` values.

The period-two matrix `Q=V(a-δ)V(a+δ)`, `δ=2aC/N`, has

\[
\operatorname{tr}Q=(a^2-2)^2-2-a^2\delta^2.
\]

Its hyperbolic region is exactly

\[
|a^2-2|<2a^2C/N,
\quad\text{or}\quad
{2\over1+2C/N}<a^2<{2\over1-2C/N}.
\]

This is the credited Floquet comparison, linked to [R2, R3], not a new attack algorithm. It describes the repeated interior pair. Finite endpoint energy still requires the boundary product and the fixed initial state. Outside this tongue, failure of this witness does **not** rule out growth from other admissible node patterns.

### 5.3 Matching high-frequency family and its limitation

For fixed `C>0`, `N` a multiple of four, and **changing** frequency `Ω_N=sqrt(2) N`, the upper bound has `Λ_N -> 2C`. The alternating product converges to

\[
P_{\mathrm{alt}}\longrightarrow
\operatorname{diag}(e^{2C},e^{-2C}),
\]

so

\[
\boxed{\lim_{N\to\infty}M_N(C;\sqrt2N)=e^{4C}.}
\]

Appendix A supplies the short matching argument. This is a candidate sharp specialization to the node uncertainty model, not evidence against fixed-physics convergence. `π` has no critical role here: every `C>0` produces an amplification limit above 1 in this resonant family. A 10% energy increase corresponds asymptotically to `C>log(1.1)/4≈0.02383`; the main budget `C=0.1` is therefore a moderate predetermined effect test, not a fitted threshold.

For each fixed `Ω`, `a=Ω/N -> 0`. The upper-bound exponent tends to zero, with leading `2Λ_N ~ C Ω³/N³`; together with `d->1`, the conservative energy enclosure collapses to 1. Fixed-frequency refinement is mandatory and is a separate result.

### 5.4 Audit outputs and decisions

Return `L_N <= M_N <= U_N`, retaining rigorous numerical enclosures for both reported ends. Define the log gap `Δ_N=log(U_N)-log(L_N)`. Use interval arithmetic to decide comparisons, not rounded printed decimals.

| Evidence | Endpoint-energy decision | Meaning |
|---|---|---|
| Validated witness lower bound exceeds `1.1` | Demonstrated violation | At least one allowed grid causes more than 10% artificial energy growth |
| Validated all-grid upper bound is at most `1.1` | Certified exclusion of this violation | Every allowed grid meets this one-sided endpoint-energy tolerance for the exact discrete rule |
| Neither condition holds | Unresolved maximum at this tolerance | A loose bound or weak witness cannot establish robustness or vulnerability |
| Source, admissibility, arithmetic or invariant check fails | Invalid pending repair | Comparative outcomes must not be interpreted |

For the two exact-resonance main cases, require `Δ_N<=log(1.05)` to call the maximum **tightly enclosed**. This is a 5% multiplicative uncertainty target for a 10% decision tolerance, fixed before evaluation. It is not a universal engineering accuracy standard. The broad result is still informative if a case is correctly classified but the bracket is loose; report those conclusions separately.

The true reference is available here, so also give the accuracy verdict. No row may be called a trustworthy simulation simply because its energy is stable.

## 6. Implementation specification

### 6.1 Minimal local stack and project layout

Use a native Apple Silicon CPython virtual environment, preferably an already available supported Python 3.11–3.13. Use NumPy for binary64 arrays, python-flint/Arb for validated scalar and small-matrix arithmetic, Matplotlib for static figures, psutil for process measurements, and pytest for the scientific correctness checks. A pure Python two-by-two matrix multiplier is sufficient for ball calculations. No automatic differentiation, SciPy optimizer, symbolic package, GPU framework or semidefinite solver is needed.

Official python-flint documentation provides Apple Silicon wheels and describes rigorous ball arithmetic [R12]. Availability on the actual Python/macOS combination is still an early gate, not an assumption. Install compatible releases, then record and freeze the versions actually used. Do not silently substitute ordinary arbitrary-precision floating point for rigorous enclosures.

Create this small layout; these are interfaces you must implement, not preexisting commands verified by the planning investigation:

```text
grid_robustness/
  README.md
  config.toml
  requirements.lock
  run.py
  mesh.py
  partition_control.py
  oscillator.py
  certificates.py
  analyze.py
  tests/test_science.py
  inputs/                    # optional copies of supplied documents
  results/
    manifest.json
    budget.json
    protocol_frozen.json
    runs.jsonl
    certificates.jsonl
    failures.jsonl
    witnesses/
    figures/
    tables/
  REPORT.md
  DERIVATIONS.md
```

Module responsibilities:

| Module | Required responsibility |
|---|---|
| `mesh.py` | Exact specification of `u`, endpoint enforcement, positive widths, nominal/alternating/opposite/shuffled/random patterns, node and width validation |
| `partition_control.py` | Source construction, direct and identity evaluations, target errors and endpoint moments |
| `oscillator.py` | Independently written kick–drift–kick, explicit matrix, midpoint, exact rotation, streaming trajectory diagnostics |
| `certificates.py` | Scalar upper enclosure, exact periodic product and nominal product by repeated squaring, initial-state checks, rigorous comparisons |
| `run.py` | CLI stages, deterministic case IDs, seeds, process isolation where useful, cumulative budget and raw status records |
| `analyze.py` | Regenerate tables/plots/report inputs strictly from saved results, classify outcomes and show missing runs |
| `tests/test_science.py` | The independent mathematical and implementation checks in Section 8 |

Use standard JSON/JSONL/CSV and compressed NumPy witness files. Do not save every random path or a giant `(trials,N,N)` intermediate. Store each trial's scalar results and the exact seed identifiers; retain actual nodes for selected witnesses and failures. Archive a handful of complete diagnostic trajectories only.

### 6.2 Core callable interfaces

Implement equivalent explicit interfaces; small naming changes are allowed if documented in the README:

```text
make_mesh(N, C_spec, pattern, seed_id=None) -> Mesh
validate_mesh(mesh, requested_C) -> geometry_record
evaluate_partition(mesh, target_id=None) -> control_record
solve_oscillator(mesh, omega_spec, method, trace=False) -> run_record
upper_energy_certificate(N, C_spec, omega_spec, bits) -> certificate_record
periodic_witness_certificate(N, C_spec, omega_spec, sign, bits) -> certificate_record
classify_maximum(lower_ball, upper_ball, threshold="1.1") -> decision
```

Represent scientific constants by exact specifications: `C="1/10"`, `"1"` or `"pi"`; detunings by rational decimal strings; `omega_spec={"base_N":N0,"detuning":"..."}` means `sqrt(2)*N0+eta`. At refinements keep that object unchanged. Do not recompute it from the current `N`. Record both the exact specification and the binary64 approximation in every result.

Initialize every trajectory at `(1,0)`. Terminate after exactly `N` steps. Reject nonpositive widths, invalid node bounds, unsupported certificate domains, nonfinite arithmetic and a changed terminal time. Return an explicit failure record rather than a numerical zero or dropped row. A certificate outside its domain is `not_applicable`, not an assertion of instability.

### 6.3 Auditor pseudocode

```text
audit_case(N, fixed_omega_spec, C_spec):
    establish integer N >= 2, C >= 0, N > 2*C, and 0 < a=fixed_omega/N < 2
    evaluate whether a*(1+2*C/N) < 2
    U_ball = validated exp(2*Lambda(N,C,a))
    candidates = []
    for schedule in [nominal, alternating_plus, alternating_minus]:
        mesh = construct prescribed nodes with exact endpoints
        validate the mesh; log all geometry
        result = binary64 kick-drift-kick(mesh)
        check independent matrix recurrence and, for this schedule,
            the validated repeated-product evaluation
        candidates.append(validated energy ball)
    L = maximum of the lower endpoints of validated candidate energies
    assert L <= upper endpoint of U_ball, otherwise mark invalid
    compute log-gap conservatively from those bounds
    apply the three-way energy decision
    run the same grids through midpoint and exact state diagnostics
    run only the random/ablation work authorized for this case
    persist evidence and status before starting another case
```

Use the lower end of a witness energy interval for an attained-value lower bound and the upper end of the all-grid bound interval for the universal upper bound. A narrow ball around an approximate optimizer's value would not prove that an optimizer is global; no such inference is used here.

### 6.4 Intended commands

From an empty writable project directory on the Mac, create the files first. Then make the following commands work as documented. The shell setup is conventional; project-specific CLIs are required deliverables and have not been executed in this planning task.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib python-flint psutil pytest
python -m pip freeze > requirements.lock

python run.py inspect --config config.toml
python -m pytest -q
python run.py smoke --config config.toml
python run.py calibrate --config config.toml
python run.py pilot --config config.toml
python run.py freeze --config config.toml
python run.py confirm --config config.toml
python analyze.py --results results --report REPORT.md
```

`confirm` must refuse to run without a frozen protocol hash, passing correctness gate and remaining budget. `freeze` records case specifications, seeds, numerical tolerances, package versions, resource estimates and the code revision/hash. A post-freeze change to scientific settings requires a new protocol version and preserved old results. A bug fix may rerun affected cases within the cumulative caps; document the cause and invalidated rows.

Provide `python run.py reproduce --config config.toml` for a clean rerun of the completed authorized protocol, and `python analyze.py ...` for analysis without rerunning experiments. A resumable run must skip only records with matching configuration and code hashes, never silently mix versions.

## 7. Locked experiment specification

### 7.1 Primary hypothesis, expected failures and numerical meaning

The primary quantitative hypothesis is that, at the two prescribed exact resonance centers and `C=0.1`, the validated constructive lower bound exceeds `1.1` and the all-grid upper/lower bracket has log gap at most `log(1.05)`. For each corresponding fixed physical problem, the upper certificate at `2N0` should fall below `1.1`. These predictions follow from the supplied derivation; the experiments test correct finite implementation, practical evaluation and interpretation, not the existence of a statistical surprise.

A secondary descriptive question is how often the prescribed independent-node sampling detects a 10% energy violation and how its distribution compares with the validated attainable witness. Do not transfer the source's Brownian theorem to this oscillator. Any oscillator random-law asymptotics would require a separate proof and are outside the authorized study.

At detuned frequencies the simple alternating witness may weaken while the general certificate remains loose. That is an expected limitation of this particular finite bracket, not evidence that all other grids are safe. Fixed-frequency refinement and midpoint are the clean robust controls. Ordinary full-state verification may already reject the resonance center; that would limit the incremental value of the auditor rather than invalidate the resonance calculation.

### 7.2 Development, prior planning probes and confirmation

Development uses `N0=128` and `512`. The main matrix uses `N0=1024` and `2048`. The planning work already explored some center cases at `C=π`, the `N0=512,C=0.1` witness, some other radii and detunings, and evaluated several closed-form upper bounds. Those are disclosed in Section 12. They are exploratory anchors, not independent confirmatory observations. The `N0=1024` center and the main `C=0.1` detuning matrix were not trajectory-tested during planning.

This is an analytic verification study with deliberately chosen stress cases, not a random sample of engineering problems. The freeze protects against selective reporting, budget inflation and retrospective threshold changes. Do not claim that these cases estimate industrial prevalence or that deterministic cases need repeated random seeds to become valid.

### 7.3 Exact case matrix

For each main physical instance set

\[
\Omega=\sqrt2 N_0+\eta,\quad T=1,\quad z_0=(1,0).
\]

Use `N0 in {1024,2048}`, `eta in {-0.5,-0.05,0,0.05,0.5}`, `C=0.1`, and current `N in {N0,2N0,4N0,8N0}`. There are ten physical instances and forty resolution cases. No parameter is tuned after seeing their outcomes.

| Experiment | Exact workload | Claim or uncertainty addressed | Required comparison |
|---|---|---|---|
| Source positive control | `N={128,512,2048,8192}`, five targets; `1024` random grids at each `N` | Constructive implementation; typical-vs-attainable distinction for the supplied rule | Direct-node recomputation, phase identity, explicit error bound, endpoint variance |
| Main all-grid audit | All 40 main resolution cases | Bound validity, gap, violation/exclusion and fixed-physics refinement | Nominal, both exact alternating witnesses, validated upper, exact solution |
| Main random audit | First two resolutions of each of the ten physical instances; `1024` independent grids per case | Detection under the specified probability law | Same nodes through Verlet and midpoint; same accuracy diagnostics |
| Ordering ablation | All ten main base-resolution cases; `64` independent permutations of the alternating interior node values | Effect of temporal correlation at the same node amplitude and histogram | Alternating versus shuffled order; endpoints remain zero |
| Known remedy | All deterministic main grids plus the paired random grids | Energy-specific prevention and remaining phase error | Midpoint, Verlet, exact rotation at equal `N` |
| Radius stress | `N0={1024,2048}`, `eta=0`, `C={1,π}`, `N/N0={1,2,4,8}`; deterministic only | Precision, finite-gap deterioration, stronger perturbation and eventual refinement | Same certificates and full boundary witnesses; no random campaign |
| Fixed-physics accuracy extension | Centers only, nominal Verlet and midpoint, continuing `N/N0=16,32,...` if needed, at most 6 extra levels and `N<=2^20` | Whether full-state accuracy requires much more resolution than energy control | Exact trajectory at nodes; both methods get the same levels |

The source targets are `F(t)=t`, `-t`, `0`, `min(t,1-t)`, and `0.4t+(0.6/7)sin(7t)`. For piecewise-linear targets, compute the exact polygon/target supremum from the union of their breakpoints. For the smooth target, also evaluate interior critical points where `0.4+0.6 cos(7t)` equals the cell slope; enumerating the finitely many inverse-cosine roots on `[0,1]` suffices. A dense plot alone is not a supremum-error calculation.

For the accuracy extension, use maximum nodewise state error `<=0.01` as the declared 1% normalized-state target. This is a diagnostic tolerance, not a physical safety requirement. Stop when both methods meet it, a resource limit is reached, or all six extra levels are used. If only one meets it, report that difference. Do not change the `C` uncertainty law or use this extension to retune the main cases. No random trials are needed at these large resolutions.

The detunings `±0.05` fall inside the leading period-two tongue at `C=0.1`; `±0.5` lie outside it. These are mechanism controls, not an assertion that the global maximum changes to one outside the tongue. Frequencies are frozen individually throughout their refinements.

### 7.4 Comparators and fairness

| Role | Implementation | Interpretation and fairness constraint |
|---|---|---|
| Exact physical reference | Analytical rotation | Used for validation and reporting, not hidden information granted only to one numerical method |
| Simple numerical reference | Uniform-grid Verlet | Same equation, initial state, terminal time and `N`; endpoint energy alone is inadequate |
| Closest strong failure construction | Exact periodic/Floquet witness from Section 5.2 | Mandatory; a generic search is not allowed to claim discovery of this established effect |
| All-grid upper analysis | Common nominal modified-energy norm | This is the proposed specialization and already-standard machinery, not a different baseline to be counted twice |
| Stochastic stress | Independent uniform interior node shifts | Same radius, endpoints and number of cells; count complete paths as samples |
| Correlation ablation | Random permutation of alternating interior shifts | Preserves the values and count of positive/negative shifts; changes ordering, not amplitude |
| Corrective method | Implicit midpoint | Same grid and physical inputs; report both equal-step cost and full-state accuracy |
| Conventional verification | Fixed-frequency refinement and exact state error | Do not extrapolate a convergence order through a resonance or near-zero energy differences |

There is no general-purpose optimizer in the main study. The explicit witness plus universal bound already brackets the quantity of interest near resonance. Off-resonance unresolved brackets should be reported as such. Adding an optimizer, mesh DP, SDP or nonlinear oscillator is a possible future project, not an automatic rescue branch. Therefore this study makes no claim of outperforming global optimization, arbitrary switched-system certification, industrial UQ or probabilistic ODE methods.

Compare diagnostic costs honestly: the certificate answers an all-grid question; Monte Carlo estimates a distribution; midpoint changes the numerical method. They are not interchangeable algorithms with the same output guarantee. Report cost and information together rather than advertising a speedup over a method answering another question. Charge construction, preprocessing, validation and analysis to their respective totals; report both cold setup and warmed kernel timings.

## 8. Independent correctness and numerical precision

### 8.1 Mandatory checks before comparative interpretation

1. **Exact tiny algebra.** With Python `fractions.Fraction`, verify `det V(r)=1` and the Gram-trace identity in Appendix A at the fixed rational pairs `(a,r)={(3/2,7/5),(1/2,2/3),(7/4,9/5)}`. Use the weighted expression without square roots. Verify the pair-trace formula for three rational `(a,δ)` pairs. These checks cannot prove identities universally; also write the algebraic derivation.
2. **Independent recurrence.** At `N={8,16,32}`, compare the three-stage Verlet implementation with explicit two-by-two products on nominal, alternating and random valid grids. Use `C=0.1` and `Ω={1,4}` subject to the domain checks. Compare sequential and repeated-power products for periodic grids, including both boundary matrices. This catches indexing and multiplication-order errors.
3. **Exact flows and invariants.** Verify midpoint orthogonality algebraically, its numerical agreement with total-angle rotation, and its energy conservation. Verify the uniform-Verlet invariant `d q²+w²` at every step. With `C=0`, the all-grid exponent must be exactly zero, while the actual uniform trajectory need not match the physical one. At `a=sqrt(2)`, require the nominal energies to alternate between 1 and 1/2 within arithmetic tolerance. Record maximum **absolute** energy error, not just maximum positive growth, to expose this defect.
4. **Box and endpoint checks.** Test the first/last cell radii, interior radius, positivity, fixed total time, and rejected invalid input. Exhaustively enumerate `u_i in {-C,0,C}` for `N=8,C=0.1,Ω=8*sqrt(2)`; `3^7=2187` sequences are cheap. Every achieved value must lie below the bound. This is an implementation test on a finite subset, not a global certificate by enumeration.
5. **Source constructor.** Recompute from nodes, check the explicit `B_N` error bound and the direct/identity phase agreement; check all five target derivative bounds analytically. At `N=8192`, source-like positive/negative endpoints should be near `±0.999877`, but the theorem's bound and independent high-precision values govern correctness, not exact decimal matching.
6. **Certificate containment.** Every validated witness must lie beneath the universal upper enclosure, and every binary64 deterministic result must agree with its independent reference within the calibrated tolerance. A contradiction is an invalid experiment until explained, never evidence of a stronger instability.

### 8.2 Binary64 versus validated arithmetic

Use binary64 for batches and ordinary timing. Use python-flint `arb` with initial `ctx.prec=128` for the scalar certificate and periodic/nominal witness products. Parse rational specifications into integer division, for example `arb(1)/10`, or a documented rational type; do not assume the string grammar accepts `"1/10"`. Form `π` from `arb.pi()` and the frequency from a ball square root of 2. `arb(0.1)` represents the binary float, not exactly one tenth. Test the decimal/rational input route against `arb(1)/10` [R12].

For a witness, perform the two-by-two matrix power by repeated squaring in ball arithmetic. This is `O(log N)` and avoids propagating an interval through thousands of rotating steps. Sequential intervals can suffer severe wrapping; raising precision without bounding that cost is not an acceptable plan. Test small sequential ball products against the power result, but do not require long sequential ball integration for every sample.

Increase precision through `128,256,512` bits only if the required decision or enclosure width is unresolved. Require the ball radius in reported `log G` and `2Λ` to be at most `1e-10`, and enough separation to make a claimed comparison with `log(1.1)` decisive. A 10% energy decision is far larger than this arithmetic uncertainty. If the interval remains too wide after 512 bits, retain the valid broad enclosure and mark the numerical decision unresolved; do not print its midpoint as a certificate.

Take a logarithm of a witness lower bound only after proving that endpoint strictly positive. If a broad enclosure includes zero, the log-gap is unresolved (or infinite for display), and the precision rule applies; do not take the logarithm of a negative interval endpoint or replace it with an arbitrary small constant.

Preserve interval radius in serialized output. Use documented outward endpoint methods or an exact ball representation, and reparse it once in a test to ensure it still encloses the original value. Decimal printing without a radius is presentation only. The proof and rigorous arithmetic certify the **exact discrete rule with the exact specified inputs**. They do not automatically certify every binary64 instruction or hardware fault.

On development cases, compare binary64 ideal-width recurrence with the validated periodic reference at the **same exact scientific inputs**. Define

```text
tau_num = max(1e-12, 20 * largest observed development discrepancy in log energy)
```

Freeze this tolerance only if `tau_num<=1e-8`. The factor 20 is a conservative verification margin, not a proved floating-point error bound. Use the interval result, not this empirical tolerance, for certification. If binary64 exceeds the cap, investigate node cancellation, operation ordering and input rounding; use validated deterministic evaluation and reduce sampling scope if needed. Never loosen the 10% scientific threshold to accommodate arithmetic error. Guarded and unguarded meshes are different inputs: compare a guarded ideal mesh with a periodic reference at its exact guarded radius, and report its difference from the original-radius mesh as an input change, not an arithmetic failure.

### 8.3 Mesh representation and stored-node roundoff

The canonical scientific mesh is the exact specification `t_i=i/N+u_i/N²`. The stable formula for widths and dimensionless steps avoids subtracting nearby times. Implement a second path that explicitly forms binary64 nodes and recomputes widths by subtraction, because that is how a conventional mesh-based solver may receive them. Compare these two paths on every deterministic case and a fixed first eight random paths per random case.

For any stored binary64 node vector, compute its realized normalized radius `C_eff=N² max|t_i-i/N|`, treating the stored floats as exact inputs when validating. Use `float.as_integer_ratio()` and exact rational differences, or outward-rounded Arb enclosing those exact dyadic values; compare with the exact rational `C` or a validated `π` ball. A naive binary64 comparison is insufficient at a saturated boundary. Record `C_eff` and endpoint/positivity checks. Roundoff can move a boundary node slightly outside the ideal box. If that happens, the vector is not a witness at the original radius: either generate an inward-guarded vector and revalidate it, or evaluate an upper bound for `C_eff` and label the different radius. Never silently clip a failed geometry check.

For a simple inward guard, start from node shifts of magnitude `max(0,C-16*eps*N²)`, with `eps=np.finfo(float).eps`, then verify the actual stored nodes against the original box. If verification fails, permit one doubled guard and report it. This guard is a construction aid, not a proof by itself. The primary exact alternating witness and its matrix-power certificate remain defined at the exact requested `C`; separately identify any stored-node reproduction and its small guard. Do not pretend the periodic formula exactly describes the rounded node differences.

Separate stored-node/ideal-mesh differences from arithmetic discrepancies. For same-input verification of the stored-node recurrence, use a separately written standard-library `decimal` implementation on exact dyadic nodes and the same frequency specification, at 60 and 120 decimal digits on the tiny/development cases and one main deterministic case per base frequency. Its precision agreement is an independent numerical check, not an interval certificate. Do not require long sequential Arb products on arbitrary rounded nodes. If this check fails to stabilize, mark that reproduction path unresolved while retaining an independently valid exact periodic witness.

For the source accumulator, an absolute width error of order machine epsilon can produce phase error of order `eps*N²`. Preserve both direct-node and phase-identity calculations, and check selected cases at higher precision. Do not extrapolate binary64 behavior to arbitrarily large `N`.

### 8.4 Failures and exclusions

Never omit a failed case from denominators or timing totals. Record `invalid_geometry`, `outside_certificate_domain`, `nonfinite`, `precision_unresolved`, `timeout`, `dependency_blocked`, `interrupted`, and `ok` distinctly. All main cases are specified to fall in the mathematical domain; a main-case domain failure usually indicates an implementation or configuration defect. An unresolved broad interval is a valid but insufficient bound, whereas a violated invariant is an invalid experiment. Rerun only for a documented cause within the repair budget.

## 9. Measurements, statistics and analysis

### 9.1 Required row-level records

Each result must identify the case family, fixed physical instance, current `N`, exact `Ω` specification, exact `C`, realized radius if a stored vector was used, method, schedule, seed/trial ID, precision, code/configuration hashes, status and wall time. Include minimum width, total time, maximum `Ω h_i`, endpoint `(q,w)`, `G`, `log G`, endpoint state error and maximum nodewise state/energy error. For the source accumulator, store its final value and target error instead of oscillator fields.

Certificate records contain the input specifications, domain check, `d`, `η_b`, `η_i`, `2Λ`, upper ball, witness energy balls, conservative maximum bracket, log gap, classification, arithmetic precision and computation time. Keep an explicit `exact_discrete_rule` scope field. Store certificate and binary64 records separately, linked by case ID, so their evidence cannot be confused.

For paired methods use the identical node vector. For timings report mesh generation, numerical integration, certificate/reference calculation, output writing and total elapsed time separately. The end-to-end total includes them all. There is no preprocessing cost to amortize across a hidden number of users; any reused matrix power or scalar quantity must be disclosed.

### 9.2 Random generation and independent units

Use NumPy `Generator(PCG64(SeedSequence(...)))`. Development master seed is `20261027`; confirmation master seed is `7312049`. Derive each stream from a documented tuple of nonnegative integer IDs for stage, experiment family, physical instance, current resolution and trial. Do not use Python's process-randomized `hash()`. Generating a batch of 32 instead of 128 must not change a given trial's nodes.

Random-node paths have `u_1,...,u_(N-1)` independently uniform on `[-C,C]`; endpoints are fixed. Their widths are correlated. The independent statistical unit is an entire sampled grid, not a node, a time point, a solver's paired evaluation, or a different resolution of the same physical instance.

For each of the twenty main random cases, report mean and standard deviation of `G` and `log G`, median, empirical 5th/95th/99th percentiles, minimum/maximum, the fraction with `G>1.1`, and invalid-run count. Let `k` count paths with `log G > log(1.1)+tau_num`; let `b` count valid paths within `tau_num` of `log(1.1)`. Keep all `m` valid paths in the denominator and report the possible event fraction `[k/m,(k+b)/m]`. For nonzero counts, give the envelope of 95% Wilson score intervals over counts from `k` through `k+b`; compute the normal quantile with Python's `statistics.NormalDist`. These are inference intervals conditional on the stated numerical classification accuracy, not rigorous per-grid certificates.

Only when `k=0` and `b=0` use the exact one-sided 95% zero-exceedance probability upper bound `1-0.05^(1/m)`. If `b>0`, do not call the result zero exceedances: retain the count envelope, using its upper count for a conservative upper inference. This is a statement about this distribution only. Use `0.05/20` in the zero-exceedance formula if making a simultaneous statement across all twenty unambiguous zero-event cases. Do not turn an observed sample range into a confidence interval for all possible outputs.

At `m=1024`, zero observed exceedances still permits approximately 0.29% exceedance probability at one-sided 95% confidence for one case. A claim that a vulnerable grid is impossible would be false. If some runs fail, report the failure fraction and the probability statement conditional on valid runs; do not silently discard them. Classify arithmetic-near-threshold cases using the fixed numerical tolerance and mark ambiguous events separately.

For the source control, report `N*sample_variance(Y)` against the theoretical limit `1/2`, the mean, standard deviation, sample range and constructive endpoint values. Finite-`N` means need not equal zero, and finite variances need not equal their limit. Do not make correctness depend on a mean confidence interval covering zero or the value `1/2` lying in every variance interval. An optional fixed `2000` bootstrap resamples of complete endpoint observations can quantify finite-sample moment uncertainty, charged to the analysis budget. Use it only if it resolves a large persistent discrepancy after deterministic checks. It is not a proof of the functional central limit theorem.

Do not infer a population-level superiority claim from ten deliberately chosen physical instances. Report every case. Statistical repetition measures the sampled grid law; it does not establish novelty or validate the all-grid proof.

### 9.3 Essential figures and tables

Produce four concise figures plus their underlying CSV tables:

1. **Source calibration:** constructive positive/negative endpoints and random dispersion versus `N`; label the analytic asymptote and observed quantities separately.
2. **Maximum-energy bracket:** validated lower/upper bounds at `C=0.1`, including detuned cases; draw the `1.1` decision line and display unresolved gaps. Use a logarithmic energy axis if needed.
3. **Fixed-physics refinement:** separate panels per frozen frequency, showing energy certificates, observed witnesses and exact-reference state errors. Do not connect cases with `Ω_N` changing as if they were convergence data.
4. **Correlation and remedy:** random/permuted energy distributions versus periodic witness, with midpoint energy and state accuracy displayed together. Show `C=π` stress results in a separate panel or table so they do not compress the main 10% decision scale.

Provide a small cost table with cold start, warmed certificate time, witness evaluation, sampled-grid audit time, complete run time and measured peak resident memory. Do not repeat identical deterministic runs to create scientific confidence intervals. Timing repetition has a separate purpose.

The report must answer: What is mathematically certified? What was merely observed? Which cases were already rejected by state/refinement verification? What uncertainty assumptions make the adversarial grids admissible? What remains unresolved away from the selected resonance?

## 10. Mac feasibility, cumulative budgets and execution gates

### 10.1 Inspect the actual target first

`run.py inspect` must record the OS version, machine architecture, CPU/chip model, total and currently available memory, free space in the project filesystem, Python executable/version/architecture, dependency versions, NumPy configuration, process thread settings and precision/backend actually used. Use macOS `sw_vers`, `uname -m`, `sysctl -n hw.memsize` and a suitable chip/model query; retain relevant fields rather than hardware serial numbers. Confirm native `arm64` execution. Record a translated/x86 environment if encountered and repair it within setup limits if feasible.

Do not infer RAM, free storage or precision from “M3 Max.” The planning probes ran on hosted Linux, not this Mac. If the requested Mac is unavailable, complete mathematical review, code and portable correctness work where authorized, but mark M3 Max validation blocked and do not present other-machine timings as target measurements.

Use CPU binary64 and Arb. No accelerator is needed; leaving GPU/Metal unused is an intentional simplicity choice. Set NumPy/BLAS thread counts to one for comparability where applicable. Record thermal/power conditions and substantial background activity. For runtime claims, use AC power when available and report the actual condition; do not require an uncontrolled machine-setting change.

### 10.2 Concrete ceilings and expected scale

These are planning caps, not measured performance promises:

| Resource | Ceiling and rule |
|---|---|
| Total experimental and analysis-command wall time | `180 minutes`, cumulative across development, calibration, repeats, failed attempts, repairs, confirmation and fallbacks; keep a persistent ledger |
| Final preservation/report reserve | The last `20 minutes` of that cap are reserved; stop starting scientific work at minute 160 |
| Dependency setup/download effort | At most `30 minutes` before using the documented blocked-dependency consequence; separate from experimental wall time |
| Expected implementation/review effort | Approximately `3–5 hours`, unmeasured; total end-to-end planning expectation about `5–8 hours` including experiments and reporting |
| Peak memory of the process tree | `min(2 GiB, 20% of installed RAM, 50% of initially available memory)`; sample actual RSS, reduce batches before swapping |
| Disk growth | `1 GiB` for project results and copied inputs; at least `2 GiB` free before running; dependency environment allowance separately at most `2 GiB` |
| Total scalar trajectory evaluations | `65,000` across methods, tests, repeats, failed attempts and all branches; each method applied to one grid counts once |
| Random/permuted trajectory evaluations | `50,000` of the above maximum; the main paired random block uses `40,960` evaluations, source `4,096`, paired ablation `1,280` |
| Main random matrix | At most `20 cases × 1024 grids`; no unplanned resampling after looking at results |
| Main deterministic/stress maximum `N` | `16384`; only the declared accuracy extension may reach `2^20` |
| Validated precision | `128`, then `256`, then `512` bits; no unbounded precision escalation |
| One worker/task chunk | `10 minutes`; checkpoint and return control at its limit. A stage controller such as `confirm` may resume chunks automatically under the same frozen hash up to its cumulative allocation |
| Repairs | At most `2` repair cycles and `20 minutes` cumulative experiment reruns; engineering edits remain within the overall effort estimate |

The large sampling jobs should stream batches of at most 128 paths and keep only current states and scalar diagnostics. The largest random mesh here has `N=8192` in the source control or `N=4096` in the main random oscillator block. A `128×8193` binary64 node array is about 8 MiB before other arrays. This is an array-size calculation, not a peak-memory measurement. Avoid allocating every trial's trajectory. The accuracy extension is a single trajectory at a time with online maxima.

At the full sample tier, the locked random/source/ablation work consumes `46,336` evaluations, leaving `3,664` of the random/permuted cap for all earlier work and repairs. Predeclare at most `1,024` pilot evaluations, `960` calibration/timing evaluations and `128` tiny random-check evaluations, leaving at least `1,552` for repairs. The remaining deterministic/exhaustive work also counts against the overall `65,000` ceiling. Before freeze, reserve the exact remaining locked workload and the report time; if those totals do not fit, apply the allowed whole-matrix sample-tier reduction before collecting confirmation data.

For main batches, cost is `O(number_of_grids × N)` and state memory is `O(batch_size)` plus mesh storage. The scalar certificate is `O(1)` and periodic witness certification is `O(log N)` two-by-two operations. This makes a pilot in roughly 15–30 minutes and a locked run in about an hour plausible, but only local calibration can justify those allocations. Small overheads can dominate certificate timings.

### 10.3 Calibration and workload resizing

Before freezing, measure one ordinary trajectory, one 128-path batch, the scalar certificate, the periodic witness and one analysis/write pass at representative `N={512,2048,8192}`. Use the development physical instance `Ω=sqrt(2)*512`, `C=0.1` unchanged across these oscillator calibration resolutions. Time one deterministic source-control evaluation at `N=8192` as well and charge it to deterministic work. Measure a second batch size, 32, as specified below. Use the actual methods and diagnostics; a bare arithmetic microbenchmark is insufficient.

Record one fresh-process cold invocation including imports. For scalar-certificate and periodic-witness kernels use one warm-up and five timed repetitions on the same deterministic inputs; report median and range. If scalar calls are too short, choose a batch length once to exceed 0.1 seconds, capped at 1000 calls, and record it; otherwise report that cost is below useful timing resolution. Do not batch extra witness trajectories for this purpose. Use one 128-path paired Verlet/midpoint batch at each of the three calibration resolutions (`768` trajectory evaluations total), then three warmed paired 32-path batches at `N=2048` (`192` more). Repeating those inputs supplies timing information, not additional independent scientific samples. Mesh generation and I/O remain included in the separately reported end-to-end measurement. Charge cold-run trajectory work as part of this fixed allocation or use a deterministic cold input.

Estimate remaining time from measured per-cell cost and fixed overhead, and require a factor-of-two headroom within the remaining scientific budget. Measure memory with psutil process-tree RSS at approximately 10 Hz; also record OS peak usage where available. On macOS `ru_maxrss` is in bytes, unlike Linux's common KiB convention; normalize and report the source/unit. Do not use Python allocation tracing as the only native-array memory measurement.

Allowed calibration-only resizing, in order:

1. Reduce batch size from 128 to 32, then 8, without changing case content or seeds.
2. Reduce the main Monte Carlo count from 1024 to 512 for **every** main random case, and source random count from 1024 to 512. Recompute probability intervals using actual `m`. This affects stochastic precision, not the deterministic certificate.
3. Reduce main and source random counts to 256 if needed. Below 256, drop the random-detection comparison and label that component inconclusive; preserve the deterministic study.
4. Omit the accuracy extension after the required first four resolutions if its measured cost threatens the reserve. Report the unresolved state-accuracy target. Do not shrink the main resonant `N` to an irrelevant tiny case.

Do not remove difficult detunings, change frequencies, increase `C`, lower the 10% threshold or alter solver tolerances to fit a budget. No size reduction is allowed after seeing confirmation outcomes except stopping whole remaining blocks with missing-run accounting. If the analytic/certified core itself is infeasible, stop with that measured limitation.

### 10.4 Stages and evidence gates

The allocations below sum to 180 minutes. They are command/experiment/analysis budgets; implementation work is estimated separately. Unused time can be moved to a justified later stage, but the total, repair limit and final reserve remain fixed.

| Gate and allocation | Question and minimum evidence | Continue, repair, fallback or stop |
|---|---|---|
| **G0: constructive and source feasibility — 10 min** | Verify supplied definitions, reproduce the all-grid derivation in `DERIVATIONS.md`, inspect closest prior art, establish local environment and dependency availability | If an essential derivation fails, stop that claim with a counterexample or precise gap. Missing original bundle is nonblocking. If exact finite-envelope equivalence is found in prior art, use the minimal diagnostic fallback below and drop novelty. Ordinary known resonance alone is already anticipated |
| **G1: tiny correctness — 20 min** | Pass the independent checks in Section 8, including exhaustive `N=8` enumeration, source reconstruction and ball serialization | Continue only after contradictions are resolved. Permit bounded repairs. A correctness failure is invalid evidence, not a scientific negative result |
| **G2: resource calibration — 5 min** | Record representative costs, RSS and estimated full count/time; choose allowed batch/sample tier and freeze numeric tolerance | Require twofold time headroom and memory compliance. Apply only Section 10.3 resizing. If no meaningful deterministic scope fits, report practical infeasibility |
| **G3: mechanism pilot — 15 min** | At development `N0=128,512`, `eta=0`, `C=0.1`, compare witness/upper, nominal, midpoint, fixed-frequency `2N0`, and up to 256 random grids at the base resolutions | Check the 10% separation and certificate gap. If a valid pilot contradicts the analytic prediction, investigate implementation/derivation first. Wide intervals or noisy samples alone do not refute the mechanism. Permit one targeted extension within repairs; otherwise classify unresolved/invalid as appropriate |
| **G4: locked comparison — 60 min** | Freeze and execute Section 7's main matrix, ablation, stress and feasible accuracy extension, persisting every record | No favorable-case selection. Stop initiating runs before the reserve. Tight-center failure with valid arithmetic means the specified finite-gap target is unsupported; detuned looseness is a scoped limitation, not a reason to invent another project |
| **G5: diagnosis and analysis — 30 min** | Generate all tables, plots, claim classifications, uncertainty intervals, cost and missingness audit | Repair analysis-only defects if possible without rerunning valid data. Distinguish certificates, observations and hypotheses |
| **Repair allowance — 20 min** | At most two documented fixes with affected-case reruns | Charge all failed work; do not restart budgets or indefinitely change hypotheses |
| **Preservation and final report reserve — 20 min** | Finish reproducible artifacts and a complete report for reached stages | Always complete; a partial experiment with explicit limitations is preferable to spending the reserve on more trials |

The principal fatal mathematical gate is the certificate derivation, not whether a random pilot looks dramatic. A pilot's failure to see a rare grid is not evidence of robustness. Conversely, a sampled failure is evidence of vulnerability only after geometry and numerical correctness are checked.

### 10.5 Predeclared limited fallbacks

**Exact prior-art equivalence:** If a source supplies the same all-grid finite bound and matching maximum law, do not spend the full budget claiming to validate a new contribution. Complete the derivation attribution, one fixed instance `Ω=sqrt(2)*1024`, `C=0.1`, its four prescribed resolutions, both methods, both periodic signs, the certificate and at most 256 random grids at the base resolution. Reproduce source lines `±t` at `N=512,8192`. Report a known-mechanism benchmark and a limitation of the original novelty/industrial-transfer premise. This is not triggered merely by the already-known resonance center.

**Validated arithmetic unavailable:** Try installation in a compatible local Python once, within setup allowance. If still unavailable, retain the analytic proof, exact rational checks and reproducible binary64 diagnostics, and label numerical upper values as evaluations of a proved formula with unvalidated rounding. Do not make a numerically certified near-threshold decision. This yields an analytic/diagnostic result with numerical certification blocked, not permission to substitute a nonrigorous library and keep the label.

**Target Mac unavailable:** Complete unaffected derivations and portable implementation, preserve any other-platform checks, and clearly state that target feasibility is untested. No remote experimental compute is authorized as a replacement.

**Main hypothesis false after valid verification:** Preserve the failed tightness or decision outcome, explain whether the problem is a loose bound, a weaker finite witness, incorrect proposed theorem or resource limitation, and stop the promised contribution. Do not pivot to a different oscillator, different uncertainty magnitude or favorable threshold. An optional future direction can be recommended, but not executed as an unbounded branch.

## 11. Honest outcome interpretation and final deliverables

### 11.1 Classification rules

Use a primary outcome category and separate component statuses; different parts can legitimately have different outcomes.

| Category | Evidence required here | Permitted conclusion |
|---|---|---|
| **Supported contribution** | Verified derivation; validated finite bounds; tight center brackets at both main centers; fixed-physics controls; credible prior-art distinction if novelty is claimed | A restricted finite-horizon quantitative result for this oscillator and node model. If novelty remains uncertain, say “supported quantitative result, novelty unconfirmed” |
| **Informative negative result** | Valid mathematics/implementation shows a proposed transfer or incremental-benefit claim fails; e.g. uniform fixed-physics convergence blocks the attachment's asymptotic pathology, or existing verification already rejects all demonstrated bad regimes | The tested application premise is limited in the stated scope. Do not infer that every adversarial-grid audit is useless |
| **Practical infeasibility** | Measured local costs, unavailable indispensable resources, or supported scaling prevent a meaningful promised stage within caps | Identify the exact resource and unfinished claim. A timeout alone is not a refutation of the mathematical mechanism |
| **Inconclusive evidence** | Valid but loose bounds, insufficient sampling, inaccessible comparison, missing target hardware, or incomplete confirmation | State which observation/certificate would resolve the uncertainty |
| **Invalid experiment** | Bug, wrong grid law, wrong physical parameters, reference error, bound violation or unaccounted rounding invalidates a comparison | Withdraw affected numerical claims, preserve failure records and describe repairs or the unresolved defect |

Separate three conclusions explicitly:

1. **Existence:** the known resonance can produce large finite-resolution artificial energy changes under this node budget.
2. **Certification:** the upper bound covers the continuous admissible family and is quantitatively tight or loose in each declared regime.
3. **Incremental usefulness:** whether the proposed energy audit changes the conclusion after a credible state/refinement check. A spectacular energy plot does not establish this third point.

A successful midpoint control supports an energy-specific remedy for the linear oscillator. It does not establish accurate phase, preserved energy for arbitrary nonlinear Hamiltonians, or a general safety solution. A narrow Monte Carlo distribution supports only a statement about the chosen law. A proved upper bound above the threshold demonstrates neither an actual failure nor a failure probability.

### 11.2 Required final report

Write `REPORT.md` so it stands alone. Begin with the research decision and outcome category, the achieved versus certified bounds, and the strongest limitation. Include the fixed physical problem and uncertainty assumptions; equations and attribution; completed/blocked stages; frozen and deviated protocol; correctness evidence; every required case and failure; numerical precision; comparisons and state-error controls; actual Mac configuration and resource totals; costs and uncertainty estimates; source access/novelty limitations; and one justified next step, if any.

If all harmful cases are plainly underresolved in state and ordinary verification already rejects them, say so prominently. The meaningful result may then be an inexpensive illustrative certificate and a negative finding about incremental industrial value, rather than an undiscovered numerical hazard.

In particular, the exact center's known 50% nodewise energy defect already makes “hidden from full conventional verification” unsupported for that case. No favorable endpoint-only result may reverse that verdict. The tight all-grid envelope can still be supported independently.

Possible future work, only if evidence justifies it, is a sharper bound for the specific detuned cases that remain unresolved. Do not suggest an unbounded tour of industrial solvers. If no next step is justified within the observed regime, say that.

### 11.3 Artifacts proportionate to reached stages

Always provide `REPORT.md`, `DERIVATIONS.md`, the frozen/configured protocol or explicit reason no freeze occurred, environment record, cumulative budget, source references and failure/blocker record. If code was written, provide runnable source, tests, exact dependency lock and concise setup/reproduction commands. If experiments occurred, provide raw JSONL records, selected witness arrays, certificate balls, analysis scripts and essential figures/tables. For an analytic stop, provide the checkable proof or counterexample and the minimal diagnostic actually used; no empty benchmark scaffolding is required.

Do not ship fabricated “expected results” as measured data. Preserve failed and unfavorable results. Regenerating the essential figures from saved raw files must not require rerunning the whole experiment. Check all output files are readable and their stated counts match the manifest before finishing.

You are authorized to execute the specified stages, repair ordinary implementation defects, install local dependencies in the project environment and generate the report within these bounds. Do not wait for another prompt at each gate. If access, hardware or permissions block an indispensable step, finish unaffected work and identify precisely what remains unperformed.

## 12. Work already performed during planning

This section is provenance, not a completed PoC report. Reproduce any result you rely on in the target project, using the independent checks and validated arithmetic specified above.

### 12.1 Document, research and mathematical work

The planning work inspected both supplied documents, recovered and visually checked ambiguous PDF formulas, audited the source's proofs and constructive indexing, researched the primary sources listed below using two search systems, and commissioned separate mathematical, prior-art and experimental analyses. It derived the all-grid certificate, its physical-energy conversion, the boundary-correct alternating product, the pair trace and the matching changing-frequency limit. Three exact rational checks of the Gram identity and three of the pair trace were executed with Python's `Fraction`; these support algebra checking but are not substitutes for the general derivation in Appendix A.

No essential defect was found in the supplied theorem proofs within the scope of this audit. That is not peer review, a formal proof check, or a guarantee of historical originality. The principal application limitations were identified independently of numerical performance: lack of an underlying fixed physical equation for the supplied rule, established oscillator resonance prior art, and the difference between energy and trajectory accuracy.

### 12.2 Small computations actually executed

The source construction was independently implemented for targets `+t` and `-t` at four resolutions and then recomputed from binary64 node positions. Representative results were:

| `N` | Positive endpoint | Negative endpoint |
|---:|---:|---:|
| 128 | 0.9868884131 | -0.9877742162 |
| 512 | 0.9986768216 | -0.9987730475 |
| 2048 | 0.9998823874 | -0.9998753728 |
| 8192 | 0.9998771472 | -0.9998771115 |

These reproduce the scale of the supplied deterministic table. The source's original random bundle was unavailable; its full random experiment was not independently rerun during planning.

A separate design probe executed 38 small oscillator trajectories in hosted Linux with Python 3.12.14 and NumPy 2.3.5. It considered selected nominal, alternating, opposite and single-seed random grids, several radii, fixed-frequency refinements and a few detunings. Additional small independent calculations checked the scalar norm formula against two-by-two singular values and evaluated its finite bound. These were mechanism and design checks, not a powered random-law study or target-machine benchmark.

| Planning check | Observed or calculated value | Evidence status |
|---|---:|---|
| `N=512, Ω=sqrt(2)*512, C=0.1`, alternating endpoint `G` | 1.4906596900 | Binary64 trajectory |
| Same case, all-grid formula `U` | 1.4910080855 | Binary64 evaluation of proved formula; not outward-rounded yet |
| Same fixed `Ω,C`, `N=1024`, formula `U` | 1.0384929694 | Binary64 formula evaluation |
| Same fixed `Ω,C`, `N=2048`, formula `U` | 1.0044986862 | Binary64 formula evaluation |
| `N=512, Ω=sqrt(2)*512, C=π`, alternating `G` | 279736.86065 | Binary64 trajectory; largest individual `Ω h≈1.43157<2` |
| `N=2048, Ω=sqrt(2)*2048, C=π`, alternating amplitude `sqrt(G)` | 533.84760939 | Binary64 trajectory, a different physical frequency |
| Limit amplitude for the changing-frequency family at `C=π` | `exp(2π)≈535.49165552` | Analytic limit, not a finite-`N` measurement |

One opposite-sign schedule at `C=π,N=512` produced amplitude approximately `2.30881`, not `exp(-2π)`. This exposed the need to retain boundary matrices and avoid asserting a reciprocal fixed-state minimum. The 10% main threshold and `C=0.1` were chosen as a clear dimensionless diagnostic before main confirmation; neither is a measured tolerance from a real engineering grid pipeline.

All actual Mac performance, rigorous numerical endpoints, full locked comparisons, random exceedance rates, and final research classifications remain work for the executing agent. Do not copy the numbers above into results files as if they had been measured on the Mac.

## 13. Review findings and disposition

The brief underwent distinct mathematical/novelty, experimental and execution review passes. The following material issues were raised and addressed in the design; unresolved matters are explicit gates rather than hidden assumptions.

| Stress test or finding | Disposition |
|---|---|
| The theorem might be nonconstructive or require an inaccessible oracle | The explicit `O(N)` feedback recursion, target slopes and phase identity are included; missing companion bundle is disclosed and nonessential |
| The original mathematical model might not approximate fixed physics | The inconsistency/defect and uniform-convergence obstruction are stated; a genuine oscillator is used without claiming the same singular-phase mechanism |
| The mechanism might already be known | Skeel, Piché and Richardson–Finn are central references and the strongest baseline; novelty is limited to a conditional quantitative specialization, with an exact-equivalence fallback |
| Only specially favorable synthetic cases might work | Main exact and detuned frequencies plus fixed-physics refinements are predetermined; no prevalence claim; weak detuned witnesses remain unresolved |
| A competent baseline might remove the apparent advantage | Midpoint and conventional state/energy-trace verification are mandatory; the report must withdraw any claim of hidden failure when these checks already reject the run |
| A high-energy plot could conceal inaccurate nominal phase or energy trace | Exact state, maximum absolute trajectory-energy error and refinement are required, not just endpoint energy |
| Upper bound might have a hidden norm factor or apply to another initial state | The fixed `(1,0)` initial condition and conversion are explicit; other phases need a different prefactor |
| Endpoint corrections might be dropped | Full boundary-correct periodic product is specified and checked independently |
| Finite precision might change the result | Binary64 and exact-rule certificates are separated; Arb enclosures, same-input comparisons, direct-node geometry and bounded precision escalation are required |
| Preprocessing or interval wrapping might dominate | Scalar certificate and repeated two-by-two powers avoid large optimization and long interval propagation; all setup costs are measured |
| Memory or runtime might exceed the laptop budget | Cumulative caps, calibration, streamed batches, sample tiers and a preserved report reserve are fixed |
| A noisy pilot might be mistaken for scientific failure | Analytic/correctness failures, interval looseness and sampling uncertainty have different decisions; repairs and extensions are bounded |
| An essential package, paper or target machine might be inaccessible | Self-contained definitions, accessible source alternatives and precise blocked-stage consequences are specified |
| The main hypothesis might be false | Preserve results and classify the failing claim; no threshold change, solver hunt or indefinite rescue campaign |

Residual risks are limited but real: historical priority of the exact envelope is unconfirmed; the selected resonance is strongly structured and underresolved; the grid uncertainty has no demonstrated industrial provenance; interval-library availability and Mac throughput are unmeasured; and the simple upper bound can be loose away from resonance. None is permission to overstate a conclusion.

## Appendix A. Checkable derivations for the coding agent

### A.1 Exact step norm and universal finite bound

Let `A=1-r²/2` and write

\[
B(r)=S V(r)S^{-1}=\begin{pmatrix}A&b\\c&A\end{pmatrix},
\quad b=\sqrt d\,r,\quad c={-r+r^3/4\over\sqrt d}.
\]

Since `det B=A²-bc=1`,

\[
\|B\|_F^2=2A^2+b^2+c^2=2+(b+c)^2,
\quad b+c={r(r^2-a^2)\over4\sqrt d}.
\]

The singular values are `e^λ,e^-λ` for `λ>=0`, because their product is 1. Thus

\[
\|B\|_F^2=2+4\sinh^2\lambda,
\quad \lambda=\operatorname{asinh}{|r(r^2-a^2)|\over8\sqrt d}.
\]

For positive `r` with `|r-a|<=η<a`,

\[
|r(r^2-a^2)|=r(r+a)|r-a|
 \le(a+\eta)(2a+\eta)\eta,
\]

with equality at `r=a+η`. Counting the two endpoint cells and `N-2` interior cells gives `Λ`. Repeated application of the norm bound gives the upper result in Section 5.1. The reciprocal smallest singular values give the lower modified-norm bound. As `d||z||²<=||Sz||²<=||z||²` and `||Sz_0||²=d`, the stated physical-energy enclosure follows.

This is a real-arithmetic proof over a continuous compact node box. It does not enumerate nodes or assume that a grid optimizer finds the maximum. It also does not assert that all per-step norm maximizers can be realized simultaneously: their independent maximization is a relaxation used only for the upper bound.

For the exact rational test, avoid square roots by checking

\[
V_{00}^2+d V_{01}^2+V_{10}^2/d+V_{11}^2-2
 ={r^2(r^2-a^2)^2\over16d}.
\]

### A.2 Matching changing-frequency limit

Set `a=sqrt(2)`, `δ=2aC/N`, and `D=diag(sqrt(2),-sqrt(2))`. Direct differentiation gives

\[
[V(a),V'(a)]=\operatorname{diag}(a^3/2,-a^3/2)=D,
\quad V(a)^2=-I.
\]

Consequently

\[
Q_\delta=V(a-\delta)V(a+\delta)=-I+\delta D+O(\delta^2).
\]

For fixed `C`, `N=4m`, and `k=(N-2)/2`, the number `k` is odd, `kδD -> diag(2C,-2C)` and `kδ² -> 0`. Hence

\[
Q_\delta^k\longrightarrow-\operatorname{diag}(e^{-2C},e^{2C}).
\]

The endpoint matrices tend to `V(a)`. Multiplying them on the two sides gives

\[
V(a)\left[-\operatorname{diag}(e^{-2C},e^{2C})\right]V(a)
=\operatorname{diag}(e^{2C},e^{-2C}).
\]

The witness therefore has `G_alt -> e^(4C)`. Meanwhile the expansion of the exact scalar bound gives `Λ_N -> 2C`. Sandwiching the supremum between this witness and the upper bound establishes the claimed maximum law. The argument uses `C` fixed, `N` a multiple of four and `Ω_N=sqrt(2)N`. It is not a proof for a fixed physical frequency.

The elementary pair-trace identity in Section 5.2 can be checked by expanding the four polynomial entries of `V(a-δ)V(a+δ)`. Because the determinant is one, trace below `-2` gives one eigenvalue with magnitude above one. This spectral fact describes the repeated pair, while the complete achieved output includes boundaries and initial state.

### A.3 Fixed-physics limit and elementary warning about endpoint accuracy

For fixed positive `Ω,C`, `a=Ω/N` and `d->1`. Expanding the positive argument of `asinh` gives

\[
\Lambda_N\sim {C a^3\over2\sqrt d}
\sim {C\Omega^3\over2N^3}.
\]

Thus both ends of the conservative energy enclosure tend to 1, as required. At the resonance center, by contrast,

\[
V(\sqrt2)=\begin{pmatrix}0&\sqrt2\\-1/\sqrt2&0\end{pmatrix}.
\]

From `(1,0)`, nominal nodewise energy alternates between 1 at even indices and `1/2` at odd indices. The endpoint at even `N` has energy 1 despite a **50% maximum absolute energy error along the numerical trajectory**. This exact observation is already enough to reject a claim that the resonance center hides from competent trajectory verification. It is an explicit limitation of the proposed application, not an empirical question to be suppressed by favorable endpoint plots.

## Appendix B. Research coverage and references

### B.1 Search coverage and access limitations

Research date: **27 October 2026**. Queries covered the exact oscillatory rule and partition dependence; sharp maximum-node constraints; sewing and error-feedback interpretations; random time-step and random-mesh UQ; oscillatory quadrature; variable-step Verlet/Runge–Kutta–Nyström resonance; cumulative timing jitter; common quadratic norms, joint spectral radius and constrained matrix products; conventional verification; and validated arithmetic on Apple Silicon. Backward references from time-step resonance literature led to Skeel and Piché. Forward searches around random-mesh FEM located work on interpolation error and bias [R13, R14].

Primary content was read for the central theorem, resonance/contractivity, random-time-step method, norm-product context, sewing and implementation sources. Recent topology-optimization and random-mesh-bias material was accessible only at abstract/metadata level and is used solely to constrain broad novelty claims. No exhaustive global-priority or patent search was performed. No unsupported industrial prevalence, price, regulatory or commercial-return estimate is used. Web renderers sometimes inject a recent rendering date into older arXiv HTML; bibliography dates below use publication/submission records instead.

The references below are ordinary standalone citations. `[R1]` refers to the attached PDF; no chat-only citation tokens or unprovided companion code are necessary to use the brief.

### B.2 Reference list

**[R1] Supplied mathematical note.** Author not named. *A sharp π threshold for oscillatory partition sums: A mathematical investigation of resolution sensitivity.* 27 October 2026. File `oscillatory_partition_threshold.pdf`, identified in Section 2.1. Critical locations: Proposition 1 p.1; object and Theorem 2 p.2; construction and finite bound p.3; sharp lower bound p.4; rigidity and Theorem 3 p.5; Theorem 4 pp.5–6; computational evidence pp.6–7; novelty and limitations pp.7–9. Supplied, non-peer-reviewed source; no public URL asserted.

**[R2] Skeel, Robert D.** *Variable step size destabilizes the Störmer/leapfrog/Verlet method.* BIT Numerical Mathematics **33**, 172–175 (1993). DOI: [10.1007/BF01990352](https://doi.org/10.1007/BF01990352). Accessible full text: [University of Vermont PDF](https://tlakoba.w3.uvm.edu/math6737/paper_Verlet_VariableStepSize.pdf). Critical locations: p.173, velocity-Verlet form and oscillator matrix; p.174, rotation similarity transform and Eq.(4), two-block trace. Establishes variable-step destabilization despite bounds on individual sizes or ratios. Full paper inspected.

**[R3] Piché, Robert.** *On the Parametric Instability Caused by Step Size Variation in Runge–Kutta–Nyström Methods.* Mathematics Report 73, Tampere University of Technology (1998); arXiv version submitted 24 September 2012. [arXiv:1209.5173](https://arxiv.org/abs/1209.5173); [full text](https://arxiv.org/html/1209.5173v1). Critical locations: §1, period-two center `sqrt(2)`; §3, contractivity and Theorem 1; §4, periodic transition products and step modulation; §5/Table 1, critical steps and wedge slopes. An official repository alternative is [Tampere PDF](https://trepo.tuni.fi/bitstream/handle/10024/116743/piche_on_the_parametric_instability_caused_by_step_size_variation.pdf?isAllowed=y&sequence=2). Relevant full content inspected; matrices in this brief were independently derived rather than copied from ambiguous extraction.

**[R4] Richardson, A. S., and Finn, J. M.** *Symplectic integrators with adaptive time steps.* Plasma Physics and Controlled Fusion **54**(1), 014004 (2012); preprint 2011. DOI: [10.1088/0741-3335/54/1/014004](https://doi.org/10.1088/0741-3335/54/1/014004). [arXiv:1108.0322](https://arxiv.org/abs/1108.0322); [full text](https://arxiv.org/html/1108.0322v1). Critical locations: §2.1, Eqs.(9),(14), parametric instability under prescribed time-step modulation; Eqs.(15)–(17), Crank–Nicolson action conservation; §2.2, numerical demonstrations. Relevant primary content inspected.

**[R5] Abdulle, Assyr, and Garegnani, Giacomo.** *Random time step probabilistic methods for uncertainty quantification in chaotic and geometric numerical integration.* Statistics and Computing **30**, 907–932 (2020). DOI: [10.1007/s11222-020-09926-w](https://doi.org/10.1007/s11222-020-09926-w). [arXiv:1801.01340](https://arxiv.org/abs/1801.01340); [full text, version 4](https://arxiv.org/html/1801.01340v4). Critical locations: §2, Eq.(7) and Assumptions 1–3; §§3–5, convergence and Monte Carlo analysis; §§6–7, geometric properties. Its independent positive random steps do not imply this brief's bounded cumulative-node law. Relevant primary sections inspected.

**[R6] Blondel, Vincent D., and Nesterov, Yurii.** *Computationally efficient approximations of the joint spectral radius.* SIAM Journal on Matrix Analysis and Applications **27**(1), 256–272 (2005). DOI: [10.1137/040607009](https://doi.org/10.1137/040607009). [arXiv:math/0407485](https://arxiv.org/abs/math/0407485); [full text](https://arxiv.org/html/math/0407485v1). Relevant locations: §1, Eqs.(1.1)–(1.2), matrix-product growth and discussion of common quadratic/ellipsoid bounds. Used to delimit the originality of norm-based certification; the present finite-horizon scalar formula is derived directly in Appendix A.

**[R7] Philippe, Matthew; Essick, Ray; Dullerud, Geir E.; and Jungers, Raphaël M.** *Stability of discrete-time switching systems with constrained switching sequences.* Automatica **72**, 242–250 (2016). DOI: [10.1016/j.automatica.2016.05.015](https://doi.org/10.1016/j.automatica.2016.05.015). [arXiv:1503.06984](https://arxiv.org/abs/1503.06984); [PDF](https://arxiv.org/pdf/1503.06984). Relevant locations: §§1–3, constrained products and multinorm framework. Broader prior art, not a required implementation or a source of a claim that the continuous node box has already been solved by this brief's formula.

**[R8] Broux, Lucas, and Zambotti, Lorenzo.** *The Sewing lemma for 0<γ≤1.* Journal of Functional Analysis **283**, 109644 (2022); preprint 2021. [arXiv:2110.06928](https://arxiv.org/abs/2110.06928); [full text, version 2](https://arxiv.org/html/2110.06928v2). Critical locations: introduction, contrast with the unique superlinear sewing result; §2, low-exponent construction; §3, nonuniqueness/critical behavior. Used to distinguish the source rule's additive defect from a stateful solver's composition problem. Relevant primary content inspected.

**[R9] NASA Glenn Research Center / NPARC Alliance.** *Verification Assessment* and *Examining Spatial (Grid) Convergence.* Online verification-and-validation tutorial, no publication date relied upon; accessed 27 October 2026. [Verification assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html); [grid convergence](https://www.grc.nasa.gov/WWW/wind/valid/tutorial/spatconv.html). Relevant locations: verification process items 3–6; asymptotic range and grid-convergence discussion. The project uses the principles of accurate-reference comparison and refinement, not a claim to apply CFD GCI formulas directly to a nonasymptotic oscillator resonance.

**[R10] Srinivasan, Vijay.** *On Methods of Uncertainty Quantification in Computational Modeling and Simulation: Harmonizing Modeling and Metrology Practices.* NIST Advanced Manufacturing Series **100-85**, published 5 October 2026. DOI: [10.6028/NIST.AMS.100-85](https://doi.org/10.6028/NIST.AMS.100-85). [Official publication page](https://www.nist.gov/publications/methods-uncertainty-quantification-computational-modeling-and-simulation-harmonizing). Relevant locations: abstract and report §§2.1–2.3, 4, credibility/UQ context. Publication details and official abstract verified; the research team also inspected the report, though the direct PDF endpoint was not consistently accessible. Used only for relevance, not a theorem or regulatory approval claim.

**[R11] Karnath, Justus; Kriegesmann, Benedikt; and Pedersen, Claus B. W.** *Numerical and physical robustness with respect to nodewise geometrical uncertainty in topology optimization.* Computer Methods in Applied Mechanics and Engineering **451**, 118651 (April 2026). DOI: [10.1016/j.cma.2025.118651](https://doi.org/10.1016/j.cma.2025.118651). [Primary repository record and abstract](https://tore.tuhh.de/entities/publication/eb4cf970-4a3e-4597-b9be-41940da916ca). Abstract/metadata only; full methods were inaccessible in this investigation. Establishes related activity, not an equivalent result or an implemented comparator.

**[R12] Python-FLINT and FLINT contributors.** *python-flint documentation: Install; General concepts; arb—real numbers*, and *FLINT arb.h documentation*. Living official documentation, accessed 27 October 2026. [Installation](https://python-flint.readthedocs.io/en/latest/install.html); [ball arithmetic and input conventions](https://python-flint.readthedocs.io/en/latest/general.html); [Arb Python API](https://python-flint.readthedocs.io/en/stable/arb.html); [underlying ball-arithmetic semantics](https://flintlib.org/doc/arb.html). Relevant locations: Apple Silicon support; real/complex error bounds; float-versus-decimal construction; directed endpoint extraction. Actual installed versions must be recorded locally.

**[R13] Abdulle, Assyr, and Garegnani, Giacomo.** *A probabilistic finite element method based on random meshes: A posteriori error estimators and Bayesian inverse problems.* Computer Methods in Applied Mechanics and Engineering **384**, 113961 (2021). DOI: [10.1016/j.cma.2021.113961](https://doi.org/10.1016/j.cma.2021.113961). [arXiv:2103.06204](https://arxiv.org/abs/2103.06204). Introduction and primary method summary inspected for novelty context. No FEM experiment or comparative performance claim is included here.

**[R14] Poot, Anne; Rocha, Iuri; Kerfriden, Pierre; and van der Meer, Frans.** *Effects of Interpolation Error and Bias on the Random Mesh Finite Element Method for Inverse Problems.* Proceedings of Machine Learning Research **271**, 95–102 (2025). [Official paper page](https://proceedings.mlr.press/v271/poot25a.html); [arXiv:2504.03393](https://arxiv.org/abs/2504.03393). Primary abstract inspected; full PDF not successfully retrieved during this investigation. Used only to note that random-mesh overconfidence and bias are already studied, not to assert a detailed equivalence or performance comparison.

## Completion check for the executing agent

Before handing off the finished research result, verify that every claimed experiment has a raw record; every numerical certificate retains an enclosure; all physical parameters stayed fixed within refinement series; all required controls and failures are visible; actual budget and device data are present; all citations are usable outside this chat; and the report makes its supported, negative, infeasible, inconclusive or invalid findings explicit. Finish the bounded project and report its evidence, even if the main application premise does not survive.
