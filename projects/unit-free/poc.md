# POC EXECUTION BRIEF
## Certified single-comparison fault detection: an optimal redundancy benchmark and a limit on the proposed algebraic mechanism

**Addressed to:** an autonomous coding agent configured as **gpt-6.1-sol**, reasoning **xhigh**.  
**Research and source-access date:** 29 October 2026.  
**Target experimental machine:** one MacBook with an M3 Max; its RAM, available storage, operating system, and numerical backend must be inspected.  
**Deliverable from this assignment:** execute the applicable stages below and produce a reproducible research result. This document is the plan, not a claim that the PoC or Mac validation has already been completed.

---

## 1. Project decision and the contribution to pursue

### 1.1 Execute this project

Build and verify a **100-node, 300-comparison network with two positive latent quantities per node**, using the three comparison signatures
\[
A=(1,0),\qquad B=(0,1),\qquad C=(1,1).
\]
Use a periodic triangular grid with the precise signature assignment in Section 4. It is simple, connected, degree six, has exactly 100 comparisons of each type, and identifies all 198 state coordinates after removing the two unavoidable global scale freedoms.

The proposed construction attains the universal upper bound on the smallest diagonal of the Gaussian residual projector:
\[
q_{\min}=\frac{300-198}{300}=\frac{17}{50}=0.34.
\]
This is an exact mathematical candidate certificate, derived in this brief and independently reviewed during planning. Verify the proof and its implementation. Do not substitute a graph-search tournament for an available sharp bound.

Use a **single-pass, Bonferroni-calibrated maximum standardized residual detector**. Under the assumptions below, its probability of any false alarm in one network snapshot is at most 5%. For a fixed but unknown comparison containing an additive log-error of magnitude \(10\sigma\), the construction certifies an alarm probability of at least **0.98059**; the analytic lower bound, rounded for display, is approximately
\[
0.9805918141.
\]
This is a lower bound on detecting that some comparison is inconsistent; it is not a guarantee of identifying which comparison is wrong.

### 1.2 One primary research question

> Under independent, equal-variance Gaussian noise in directly observed log comparisons, can a fully identifiable, simple 100-node network with 300 comparisons and maximum degree six meet a 95% worst-location alarm-probability requirement for one \(10\sigma\) comparison error, while controlling per-snapshot network-wide false alarms at 5%, and attain the sharp minimum-residual-redundancy certificate under that budget?

This is a certificate and implementation study. The analytic answer proposed here is positive; the coding agent must independently verify the derivation, construction, numerical implementation, probability calibration, and local feasibility. A discrepancy is a finding to investigate, not something to hide by changing the question.

The **primary mathematical quantity** is \(q_{\min}\); larger is better. The **primary operational requirement** is a certified probability of alarm of at least 0.95 at standardized fault magnitude \(\rho=10\), with false-alarm probability at most 0.05. The 95% requirement is an explicit engineering planning target for “reliable,” not a measured requirement from a deployed application, a threshold reported by the source, or a hypothesis selected from confirmatory simulations.

### 1.3 What is and is not a contribution

The smallest defensible result is a **source-specific connection and limitation, accompanied by an explicit optimal certificate**:

1. Explain the attachment's exponentially weak examples exactly through normalized residual geometry.
2. Establish which original sensor-fault claims are unidentifiable.
3. Give a fully specified 100-node construction attaining the classical maximum-minimum redundancy bound, with an operational single-fault guarantee.
4. Validate a small, reproducible implementation on the actual laptop.

The minimum-redundancy objective, residual projection, balanced redundancy, and statistical outlier testing are established methods. Amiri-Simkooei explicitly gives the objective \(\max\min_i r_i\) in 2004, Eq. (5) [R1]. Rofatto, Matsuoka, and Klein already design networks using synthetic outliers and detection/identification probabilities [R2]. Do not claim to have invented these principles.

The exact specialization to this finite signature palette and periodic construction was not located in the targeted searches. Its historical novelty remains **unestablished**; periodic triangular networks and self-stress ideas have substantial antecedents. A successful PoC supports a useful certified benchmark and application limitation, not an automatic claim of publication-level priority. If the precise construction is found in prior work, attribute it and retain the reproducibility/limitation result.

There is no justified theorem-derived optimizer to compare against established optimization algorithms in the supplied materials. Do not invent one to preserve a novelty narrative. The exact upper bound is stronger evidence about this objective than outperforming a finite selection of heuristics.

### 1.4 Why the original problem has been refined

The motivating narrative combines three different problems:

| Original wording or implication | Executable interpretation and consequence |
|---|---|
| “Sensor A is twice B” | Ordinary scalar ratios have one projective signature direction. They do not exhibit the attachment's three-direction amplification mechanism. This study retains the paper's two-coordinate, typed model explicitly. |
| “One sensor reports incorrectly” | A coherent change to a node's reported state is indistinguishable from a legitimate different state. The positive guarantee concerns one independently corrupted **comparison channel**. |
| “A limited number of connections” | No numerical budget or physical connection restrictions were supplied. Fix 300 comparisons, at most six incident connections per node, equal cost, and arbitrary permitted node pairs. This is a disclosed benchmark choice. |
| “The theorem chooses better networks” | The source constructs adverse examples but gives no competitive statistical network-design algorithm. Use its examples as diagnostics and classical residual geometry as the correct design criterion. |
| “Adding comparisons made detection worse” | The source chain simultaneously adds nodes, changes connections, and keeps only one residual degree of freedom. It is not a controlled experiment adding observations to the same fixed state model. |
| “Detect every false measurement” | The claim is uniform over each possible fixed single fault location and sign, with probability over noise. It does not mean all counterfactual faults are detected on every noise realization. |

The study preserves the practical aim of reliable comparison-only consistency detection. It does **not** solve arbitrary scalar-sensor design, physically constrained placement, unknown variance estimation, multiple coordinated corruptions, fault isolation, state recovery under attacks, online time-series monitoring, or an application with realistic installation costs. The graph is periodic in its mathematical indexing; no physical wraparound links are assumed to be cheap in a real installation.

The budget is not chosen solely to make a weak comparator lose. The analytic upper envelope for the same certificate, evaluated numerically at every integer budget from 199 through 300 with rank fixed at 198, first exceeds 0.95 at 280 comparisons; at 279 it is approximately 0.94965405. The explicit construction uses 300. Recompute this finite envelope during execution; budgets below 198 cannot attain the required rank, and 198 leaves no residual information. This is a necessary-count bound for this particular sufficient certification strategy, **not** a theorem that 280 is the actual minimum number of measurements for fault detection.

---

## 2. Evidence, inputs, and source fidelity

### 2.1 Supplied materials

The notebook is the mathematical paper; no additional paper is needed to understand the proposed mechanism. Preserve original files without editing them.

| Input | Location within input | Role |
|---|---|---|
| **unit_free_comparison_research(1).ipynb** | Zero-based cells 1–6: Sections 1–6; cell 7: computational evidence; cell 8: executable checks; cell 9: prior-art assessment; cells 10–12: supporting results and limitations | Primary supplied research note, titled *When unit-free comparisons require large exponents: A sharp degree threshold for fixed rational comparison signatures*, dated 29 October 2026; no author identity supplied. |
| **research_checks(1).py** | Functions primitive, right_kernel, comparison_matrix, chain, lift, reduce_degree, check_circuit, change_palette, trimmed_checks, main | Exact arithmetic construction checks. Its contents match notebook code cell 8. |
| **Pasted text(20261029-164446).txt** | Problem description and requested execution-brief specification | Motivation and reported detection percentages. It contains no simulation source, seeds, trial counts, or raw detection outcomes. |

Expected SHA-256 hashes:

~~~text
research_checks(1).py
d199430024b9820230fcd3c66156a8e7cf4a5800ff4b104155a8c29ba1626a67

unit_free_comparison_research(1).ipynb
a25fb8a88eb39d73c1c99c11952a04d044afd54cabc2e95ed753867638835b8d

Pasted text(20261029-164446).txt
a2debf5ab0de1f3a84df73dea5a4852a22fa10ab164d68df049e3989b6d9f6e8
~~~

The planning workspace paths were under **upload/** with those exact filenames. For execution, put supplied originals in **source_materials/**. This brief includes all mathematical and algorithmic ingredients needed for the selected construction. If an original is not delivered, continue independent construction, proofs, and implementation; mark the attachment-reproduction portion unavailable and request only the missing original if it is needed to resolve a discrepancy. Do not invent replacement source text or claim an unread attachment was verified. A hash mismatch requires documenting the difference, not silently accepting a changed source.

### 2.2 What the supplied mathematics actually says

For a fixed finite nonempty palette \(S\subset\mathbb Q^d\setminus\{0\}\), identify proportional signatures. Let \(g(S)\) be the smallest number of dependent distinct projective directions, or infinity if those directions are independent. A primitive circuit is a support-minimal nonzero integer vector in \(\ker M^\mathsf T\), divided by the gcd of its entries.

The note's Section 1 asserts, with proofs in Sections 2–5, that the largest primitive circuit coefficient among simple \(S\)-labelled graphs with at most \(N\) vertices and maximum degree \(\Delta\):

- is bounded independently of \(N\) when \(\Delta<g(S)\);
- has worst-case growth \(2^{\Theta_S(N)}\) when \(g(S)<\infty\) and \(\Delta\ge g(S)\).

For \(S_0=\{A,B,C\}\), \(g(S_0)=3\). The note gives both a planar maximum-degree-five chain and a degree-three family. Section 6 proves a polynomial-degree obstruction for its one-dimensional relation spaces. These results concern arithmetic and algebraic complexity, not statistical power, floating-point conditioning, or exponential runtime.

The constructive bridge is explicit: a base gadget, repeated gluing with a rank argument, lifting to larger dependence palettes, and degree splitting. There is no hidden nonconstructive existence oracle for generating these witnesses. Conversely, the note gives no inexpensive algorithm for enumerating all circuits or maximizing circuit imbalance on arbitrary 100-node graphs. Do not introduce that subproblem.

The supporting robust scalar recovery result in Section 9 uses bounded errors, up to \(k\) corrupted edges, dual flows, and deletion of up to \(2k\) edges. The monomial-dynamics result in Section 10 concerns elimination of physical scale. Both were read during planning; neither supplies the Gaussian detector or design guarantee used here. Do not import their minimax guarantees into this experiment.

### 2.3 Evidence status and resolving checks

| Central claim | Source or derivation | Current evidence status | Required execution check |
|---|---|---|---|
| Fixed-palette degree threshold | Supplied note, Sections 1–5 | Source contains a constructive proof; mathematical review found no essential defect. Not formal verification or established historical novelty. | Immutable source checks; independent tiny exact cases and bounded counterexample search. |
| High-degree polynomial identities | Supplied note, Section 6 | Proof supplied, restricted to its positive one-relation variety. | Review assumptions; do not turn it into a runtime claim. No Gröbner-basis computation is required. |
| Reported 85%, 11%, 5%, 5% detection | Prompt; derivation in Section 3.5 below | Reproduced as analytic Gaussian probabilities under a specific weak-edge interpretation. Original simulation itself is unverified. | Recompute exact geometry and distribution probabilities; label provenance correctly. |
| Coherent node changes are invisible | \(Q M h=0\), Section 3.2 | Analytic identifiability obstruction | Exact tiny identity and numerical invariant check. |
| \(q_{\min}\le(m-r)/m\) | Projection trace, Section 3.3; established redundancy literature [R1] | Exact linear-algebra bound | Independent exact and numerical projector checks. |
| Torus rank and uniform \(q\) | Derivations in Section 4 | Independently reviewed; small planning probes agree | Exact small rank/projector; integer symmetry identity; mandatory 100-node check. |
| Scan false-alarm and power bounds | Gaussian marginals and union bound, Section 3.4 | Analytic under stated model | Independent residual implementation and fixed-protocol simulation. |
| Advantage over classical design | None | Not supported; objective is prior art | Do not claim one. Check certificate against the sharp bound. |
| Laptop resource feasibility | Matrix sizes and bounded workload | Estimates only; no M3 Max measurements performed during planning | Actual Mac inspection and calibration gate. |

### 2.4 Work actually performed during planning

All notebook cells and the complete script were read. Three independent agents performed separable mathematical, literature, and experimental audits, with findings reconciled in this brief.

The supplied exact-check script was executed successfully in the hosted planning environment: 12 planar chains, 20 lifted/degree-reduced cases, 9 rational-palette transformations, and 274,362 trimmed-sum configurations, with no failed assertions. These are finite checks, not a proof of the infinite theorem.

Small calculations were performed using Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0 on an x86-64 Linux host:

- the chain probabilities in Section 3.5;
- torus SVDs for \(L=3,4,5,10\);
- an independent integer check of torus symmetry transformations at \(L=10\);
- the Bonferroni bound and 279/280-comparison envelope;
- one fixed two-label-swap control: rank 198 and \(q_{\min}\approx0.2850390940\).

For \(L=10\), a full SVD gave rank 198 and residual diagonals between approximately 0.3399999999999995 and 0.3400000000000004. These are planning probes, not confirmatory data, performance benchmarks, or validation on the user's Mac. No end-to-end PoC, detector Monte Carlo study, or optimizer competition was executed during planning.

---

## 3. Mathematical model and guarantees

### 3.1 Inputs, dimensions, and noise

Each node \(v\) has \(q_v=(q_{v1},q_{v2})>0\), with log state \(x_v=\log q_v\in\mathbb R^2\). For an oriented edge \(e=u\to v\) with signature \(a_e\in S_0\),
\[
R_e=\prod_{j=1}^2(q_{vj}/q_{uj})^{a_{ej}},\qquad
(Mx)_e=a_e^\mathsf T(x_v-x_u).
\tag{1}
\]
Thus \(M\in\mathbb R^{m\times 2n}\), \(x\in\mathbb R^{2n}\), and the noiseless log observation is \(Mx\in\mathbb R^m\). Store \(M\) as integers before conversion to float64.

The primary stochastic model is
\[
y=Mx+\sigma\epsilon+\rho\sigma e_f,\qquad
\epsilon\sim N(0,I_m),\quad \sigma>0.
\tag{2}
\]
Under the null, \(\rho=0\). Under the alternative, exactly one edge \(f\) is corrupted. The fault location, sign, and amplitude are fixed before noise is drawn, or selected independently of that noise. The detector receives \(M,\sigma,y\), not \(x,f,\rho\).

The positive claim requires **independent, directly measured channels**, including the product comparison \(C\). A product channel synthesized from the same noisy component readings is not an extra independent observation. Noise from shared erroneous node reports can be coherent, correlated, or singular and is outside (2).

Gaussian noise is assumed in log space. This corresponds to multiplicative noise on positive ratios; additive Gaussian errors on raw ratios do not produce this model exactly. Compute in standardized units with \(\sigma=1\); a physical illustration is \(\sigma_{\log}=0.01\), where \(10\sigma_{\log}=0.1\) corresponds to multiplication by \(e^{0.1}\), not a raw value becoming ten times larger. Do not claim hardware realism from this illustration.

There is no state prior, anchor measurement, or temporal model. Global translations of each log coordinate are gauge freedoms. Full identifiability means
\[
r=\operatorname{rank}M=2(n-1).
\tag{3}
\]
Connectivity alone is insufficient for a typed graph. Enforce rank as well as graph and signature constraints.

### 3.2 Identifiability obstruction

A coherent sensor corruption changing its reported log state by \(h\) produces
\[
y=M(x+h)+\sigma\epsilon.
\]
This has exactly the same distribution as a fault-free observation at another admissible state. Therefore no test controlling false alarms uniformly over all \(x\) can distinguish every such fault. This is an identifiability argument for any observation-only test, not merely for the proposed residual implementation.

For deterministic validation, choose a nonzero two-coordinate change at one node and prove/check \(Q M h=0\). At finite precision compare identical-noise residuals with and without \(Mh\). Do not interpret their unchanged alarm rate as a bug.

### 3.3 Residual geometry and the exact design bound

Let \(M^\dagger\) be the Moore–Penrose pseudoinverse and define
\[
Q=I_m-MM^\dagger,\quad \nu=m-r,\quad q_e=Q_{ee}.
\tag{4}
\]
\(Q\) is the orthogonal projector onto \(\ker M^\mathsf T\). If \(Z\in\mathbb R^{m\times\nu}\) has orthonormal columns spanning that kernel, then
\[
Q=ZZ^\mathsf T,\quad q_e=\|Z_{e,:}\|_2^2,\quad
0\le q_e\le1,\quad \sum_e q_e=\nu.
\tag{5}
\]
In particular,
\[
q_{\min}\le\frac{\nu}{m}.
\tag{6}
\]
An edge is undetectable by consistency residuals exactly when \(q_e=0\), equivalently when its coordinate vector belongs to \(\operatorname{col}M\).

This is the established redundancy-number criterion [R1]. For scalar weighted incidence matrices it is equivalently \(q_e=1-w_eR_{\mathrm{eff}}(e)\), connecting the problem to effective-resistance leverage [R3]. That scalar identity does not turn the typed model into an ordinary scalar graph Laplacian.

A nonorthogonal integer stress basis \(C\in\mathbb R^{m\times\nu}\) gives
\[
Q=C(C^\mathsf TC)^{-1}C^\mathsf T
\]
when it has independent columns. Raw basis coefficients alone are not an invariant statistical score. Implement with orthonormal bases, not this inverse.

For the secondary diagonal-noise case, let \(D=\operatorname{diag}(\sigma_e)>0\). Whiten correctly:
\[
\widetilde M=D^{-1}M,\qquad \widetilde y=D^{-1}y,
\]
and form the projector from \(\widetilde M\). A fault of \(10\sigma_e\) on edge \(e\) is again a standardized shift of 10 on that edge. Do not use \(D\) instead of \(D^{-1}\); the weight matrix is \(\Sigma^{-1}\). This equation is derived here independently, including correction of an apparent missing inverse in [R2, p.156].

### 3.4 Primary detector and its certificate

For \(q_e>0\), define
\[
S_e=\frac{(Qy)_e}{\sigma\sqrt{q_e}},\quad
t_m=\Phi^{-1}\!\left(1-\frac{\alpha}{2m}\right),\quad \alpha=0.05.
\tag{7}
\]
Raise a single network alarm if \(\max_e|S_e|>t_m\). This is a one-pass scan; do not delete a suspected observation and refit.

Under the null, every defined \(S_e\) is marginally \(N(0,1)\), although the statistics are correlated. The union bound proves
\[
P_0(\text{alarm})\le m\cdot 2\overline\Phi(t_m)=\alpha.
\tag{8}
\]
For a fixed fault at \(f\), \(S_f\sim N(\rho\sqrt{q_f},1)\). Let
\[
g(\mu,t)=\overline\Phi(t-\mu)+\Phi(-t-\mu),\qquad \mu\ge0.
\]
Then
\[
P_f(\text{alarm})\ge
P_f(|S_f|>t_m)
=g(|\rho|\sqrt{q_f},t_m)
\ge g(|\rho|\sqrt{q_{\min}},t_m).
\tag{9}
\]
The function increases with \(q_{\min}\) for \(\rho>0\). At fixed \(m,r,\rho,\alpha\), (6) gives an upper bound on this **certified worst-fault-coordinate crossing probability**, attained by the proposed design.

Do not call (9) the exact probability of a scan alarm. Other coordinates can also trigger an alarm. Do not claim the torus maximizes actual scan power across all topologies or all tests. Off-diagonal correlations matter for that different problem.

At \(m=300,q_{\min}=0.34,\rho=10\):
\[
t_{300}=3.7648236495,\quad
g(10\sqrt{0.34},t_{300})=0.9805918141.
\]
The amplitude sufficient for a 0.95 bound is approximately \(9.27751999\sigma\).

These guarantees are per snapshot, with known variance, fixed design, correct linear model, and noise-independent fault selection. Monitoring indefinitely at 5% per snapshot does not control a timewise false-alarm probability. An adversary choosing a cancelling fault after observing noise is not covered.

### 3.5 Source example and the omnibus reference

For the supplied unsplit chain with \(k\) gadgets [S1, Section 3],
\[
n_k=3k+2,\quad m_k=6k+2,\quad r_k=m_k-1=2n_k-3.
\]
Its primitive stress \(c_k\) has seven coefficients of magnitude 1, six of magnitude \(2^j\) for each \(1\le j<k\), and one of magnitude \(2^k\). Consequently
\[
\|c_k\|_2^2=3\cdot4^k-1,\quad
Q=\frac{c_kc_k^\mathsf T}{\|c_k\|_2^2},\quad
q_{\min}=\frac{1}{3\cdot4^k-1}.
\tag{10}
\]

The standard overall residual-energy test uses
\[
T=\frac{\|Qy\|_2^2}{\sigma^2}
=\frac{\|Z^\mathsf Ty\|_2^2}{\sigma^2}.
\]
Under the null \(T\sim\chi^2_\nu\); for a single fault at \(f\),
\[
T\sim\chi^2_\nu(\rho^2q_f).
\tag{11}
\]
With threshold \(\chi^2_{\nu,0.95}\), compute power with a survival function, not by subtracting a CDF near one.

For a magnitude-one stress edge and \(\rho=10\), (10)–(11) explain the prompt's table:

| \(k\) | Nodes | Comparisons | \(\|c_k\|_2^2\) | Analytic 5% omnibus detection probability |
|---:|---:|---:|---:|---:|
| 1 | 5 | 8 | 11 | 0.8543218392 |
| 3 | 11 | 20 | 191 | 0.1117993094 |
| 5 | 17 | 32 | 3071 | 0.0537384851 |
| 8 | 26 | 50 | 196607 | 0.0500582656 |

This is a reconstruction of a plausible model underlying the reported percentages, not a recovered simulation. The supplied files do not establish the original experiment's detector, injected edge, repetitions, or uncertainty.

For the proposed 100-node network, \(\nu=102\), and this omnibus test has power approximately **0.6840836652** at \(10\sigma\). That is an instructive detector-choice limitation despite optimal \(q_{\min}\). Retain it as a classical reference sharing exactly the same observations and geometry. Its null probability is exactly 0.05, whereas the Bonferroni scan's is at most 0.05; report actual simulated false-alarm rates and do not advertise an equal-realized-false-alarm superiority benchmark.

### 3.6 Two cheap controls against incorrect interpretation

**Large circuit coefficients do not determine whole-network sensitivity.** Retain every edge \((u,v,a)\) of a chain and add a distinct new node \(w_e\) and edges \((u,w_e,a),(w_e,v,a)\). The original circuit remains a circuit, but every edge now lies in a same-signature triangle. Its unit stress has squared coefficient \(1/3\) on that edge. Since that stress belongs to the full residual space, \(q_e\ge1/3\) everywhere. Verify this analytically and on \(k=1,3\).

This is a mechanism control, not a competitive design: it changes the node/edge budget and leaves additional perpendicular states unobservable. It establishes a residual-leverage bound, not size-independent scan or omnibus power.

**Bounded coefficients do not guarantee strong sensitivity.** A scalar cycle of length \(\ell\) has primitive coefficients \(\pm1\), but \(q_e=1/\ell\). Conversely, the source's degree-three witnesses have too few edges to identify two coordinates at every node. In particular, 100 nodes of degree at most three supply at most 150 comparisons, below rank 198.

The chain does not show that adding independent measurements to an unchanged parameter model inherently destroys information. For old coordinate \(e\), the squared distance of \(e_e\) from an augmented column space is
\[
\min_h\{\|e_e-Mh\|^2+\|Nh\|^2\}
\ge \min_h\|e_e-Mh\|^2.
\]
Thus old-edge residual leverage cannot decrease when rows \(N\) are added with the same old measurement scales. A chosen test can nevertheless lose power through a changed threshold or more degrees of freedom, and new weak edges can change a worst-edge objective.

---

## 4. Exact construction and proof to verify

### 4.1 Network generator

For an integer \(L\ge3\), set \(V=\mathbb Z_L^2\), \(n=L^2\), and node ID \(v(i,j)=Li+j\). For each \(i,j\), in lexicographic order, create these three oriented edges in order \(A,B,C\):

| Type | Tail | Head, indices modulo \(L\) | Signature |
|---|---|---|---|
| A | \((i,j)\) | \((i+1,j)\) | \((1,0)\) |
| B | \((i,j)\) | \((i,j+1)\) | \((0,1)\) |
| C | \((i,j)\) | \((i+1,j+1)\) | \((1,1)\) |

There are no self loops or repeated unordered node pairs for \(L\ge3\). Every vertex has degree six, \(m=3L^2\), and each type occurs \(L^2\) times. Use \(L=10\) for the primary experiment; use \(L=3,4\) for exact correctness and \(L=5\) for development.

To assemble \(M\), put \(-a_e\) in the two columns of the tail, \(+a_e\) in the two columns of the head, and zeros elsewhere. Preserve the generator's edge order in all artifacts. Reversing an edge also negates its log observation; orientation is not an extra measurement.

### 4.2 Full-rank proof

Write state components as \(X(i,j),Y(i,j)\). If \(Mx=0\):

- A constraints imply \(X(i,j)=f(j)\).
- B constraints imply \(Y(i,j)=g(i)\).
- C constraints imply
  \[
  f(j+1)-f(j)+g(i+1)-g(i)=0
  \]
  for every \(i,j\).

Both increments must therefore be constant, with opposite signs. Their sums around a periodic coordinate are zero; over real state values this forces both constants to zero. Only constant \(X\) and constant \(Y\) remain. Hence \(\dim\ker M=2\), \(r=2n-2\), and \(\nu=n+2\).

This proof does not require \(L\) to be prime or odd.

### 4.3 Uniform residual-diagonal proof

Translations of the node torus permute observations and are transitive within each type. Two additional spatial transformations generate permutations of all three types:
\[
F_1=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad
F_2=\begin{pmatrix}1&0\\1&-1\end{pmatrix}.
\]
Both are invertible modulo every \(L\), with determinant \(-1\).

\(F_1\) exchanges A and B. \(F_2\) exchanges A and C, and maps B to \(-B\). For a spatial transformation \(F\), define an invertible state transformation \(T_F\) by
\[
x'(Fv)=F^{-\mathsf T}x(v).
\]
If \(Fd=s_d\pi(d)\), where \(s_d\in\{-1,1\}\) and \(\pi(d)\in\{A,B,C\}\), map an old edge \((v,d)\) to:

- \((Fv,\pi(d))\) if \(s_d=1\);
- \((F(v+d),\pi(d))\) if \(s_d=-1\).

In the second case, canonical edge reversal and signature sign cancel. The observation vector is **permuted without scaling**. If \(P_F\) denotes the permutation with \((P_Fy)_{\phi(e)}=y_e\), then
\[
M T_F=P_F M.
\tag{12}
\]
Therefore the observation column space is invariant under an edge-transitive group of orthogonal permutations. Its orthogonal residual projector commutes with these permutations, so all \(Q_{ee}\) are equal.

Using the trace gives
\[
Q_{ee}=\frac{n+2}{3n}.
\tag{13}
\]
Together with (6), this proves global optimality of \(q_{\min}\) among **all** matrices with the same \(m,r\), hence among the stricter feasible graph class here. Equal noise is essential: the state transform need not be orthogonal, but the resulting observation action must be a pure permutation for the iid statistical symmetry to hold.

Verify (12) exactly with integer matrices for \(L=3\), and by exact integer test states or the matrix identity for \(L=10\). Do not infer symmetry merely from a plot or similar numerical diagonals.

### 4.4 Conditioning and finite precision

A supplementary Fourier argument gives
\[
s_{\min,\mathrm{nonzero}}(M)
\ge2\sin(\pi/L)\sqrt{(3-\sqrt5)/2},\qquad
\|M\|_2\le\sqrt{12}.
\]
To verify that bound, use the unitary discrete Fourier transform on node indices. At frequency \((p,q)\), with \(\omega=e^{2\pi i/L}\), the three rows are \((\omega^p-1)A^\mathsf T\), \((\omega^q-1)B^\mathsf T\), and \((\omega^{p+q}-1)C^\mathsf T\). Every nonzero frequency has at least two independent active rows. Each nonzero scalar factor has magnitude at least \(2\sin(\pi/L)\), and the smallest eigenvalue of the Gram matrix of any two signatures is at least \((3-\sqrt5)/2\). The upper bound follows from scalar factors at most two and \(AA^\mathsf T+BB^\mathsf T+CC^\mathsf T\) having largest eigenvalue three.

At \(L=10\), the first bound is approximately 0.381966 and the resulting condition bound is below 9.07. This applies to the **full** matrix, not to an arbitrary gauge-reduced representation. The large rank gap makes float64 appropriate; verify the actual gap.

Do not infer source-chain detectability from condition number. A planning probe at \(k=32\) had \(q_{\min}\approx1.8070\times10^{-20}\) while the nonzero-singular-value condition number was about 62.3. The tiny residual coordinate, not an exponential condition number, causes the weak fault signal.

---

## 5. Implementation specification

### 5.1 Keep the implementation small

Use ordinary local Python scripts, TOML configuration, JSON/JSONL metadata, compressed NumPy arrays, and static plots. No service, dashboard, optimizer framework, remote execution, paid data, CUDA, GPU, or circuit-enumeration package is needed.

Use Python 3.12 where available; a newer compatible Python is acceptable after dependency checks. Necessary packages are NumPy, SciPy, SymPy for independent tiny exact checks, Matplotlib for final figures, pytest, and psutil. A small threadpool-control package may be used if required to inspect or constrain BLAS.

Create a virtual environment. Resolve compatible binary packages on the actual Mac and freeze exact versions before confirmation. API locations for SciPy's nullspace and distribution functions were checked [R8–R10]; their availability and wheel compatibility on the target Mac remain unverified until setup. The planning host's package versions are not a mandatory Mac lockfile.

A suitable initial dependency specification is:

~~~text
numpy>=2,<3
scipy>=1.13,<2
sympy>=1.12,<2
matplotlib>=3.8,<4
pytest>=8,<10
psutil>=5.9,<8
~~~

If these ranges do not resolve, diagnose once and choose compatible versions providing the specified APIs. Record the reason and final lock. Do not compile large numerical dependencies or substitute float32 merely to satisfy an installation problem; use the setup limit and report the blocked portion if necessary.

### 5.2 Minimal repository and interfaces to create

~~~text
comparison_fault_poc/
  README.md
  requirements.in
  requirements.lock.txt
  protocol.toml
  source_materials/
    research_checks(1).py
    unit_free_comparison_research(1).ipynb
    Pasted text(20261029-164446).txt
  poc/
    __init__.py
    cli.py
    graphs.py
    geometry.py
    exact_checks.py
    experiments.py
    reporting.py
  tests/
    test_exact_and_geometry.py
    test_detector_and_invariants.py
  results/
    environment.json
    budget_ledger.jsonl
    protocol_lock.json
    source_checks.json
    exact_checks.json
    design_certificate.json
    cases.jsonl
    raw/
    figures/
    report.md
~~~

Module responsibilities:

| Module | Required interface/behavior |
|---|---|
| graphs | Generate torus edges and integer \(M\); independently generate the source chain from Appendix A; make precisely declared controls; validate simplicity, connectivity, degrees, quotas. |
| geometry | Float64 full SVD, rank decision, left-null basis, stable \(q\), scan and omnibus evaluation; independent QR/least-squares reference; diagonal whitening. |
| exact_checks | Exact SymPy ranks/projectors for tiny cases; integer symmetry checks; bounded source-theorem diagnostic enumeration; immutable source execution. |
| experiments | Predeclared cases, independent phase RNGs, streaming Monte Carlo, timing, RSS monitoring, budget enforcement, complete failure records. |
| reporting | Regenerate certificates, tables, intervals, and two or three essential plots from saved results, without rerunning experiments. |
| cli | Commands below, prerequisite enforcement, protocol freeze, resumable completed stages, and truthful terminal status. |

Do not create a broad graph abstraction beyond a list of edge records \((u,v,\text{type})\), signature table, and matrices. Source coefficients belong only to diagnostic edge records, never the detector.

### 5.3 Core algorithms

~~~text
generate_torus(L):
    require integer L >= 3
    edges = []
    for i in 0..L-1:
        for j in 0..L-1:
            append A: (i,j) -> (i+1 mod L,j)
            append B: (i,j) -> (i,j+1 mod L)
            append C: (i,j) -> (i+1 mod L,j+1 mod L)
    build exact integer M from signed signature rows
    validate graph and type counts
    return edges, M

decompose(M, sigma_vector):
    require finite M and finite positive sigma_vector
    require m > 0; empty graphs use only the exact-enumeration route
    Awhite = float64(M) / sigma_vector[:, None]
    U, s, Vt = scipy.linalg.svd(Awhite, full_matrices=True,
                                lapack_driver="gesdd")
    tau = max(Awhite.shape) * float64_epsilon * s[0]
    rank = count(s > tau)
    Z = U[:, rank:]               # includes every left-null direction
    q = row_sum(Z * Z)
    validate rank gap, orthogonality, Awhite.T @ Z, trace
    return Awhite, rank, Z, q, tau, singular_values

detect(y, sigma_vector, geometry, alpha):
    if nu == 0:
        return both alarms False, energy 0, maximum_score 0,
               status "no_residual_information"
    yw = y / sigma_vector
    coordinates = Z.T @ yw
    residual = Z @ coordinates
    score_e = abs(residual_e) / sqrt(q_e) for certified positive q_e
    structurally certified q_e == 0 coordinates have score 0;
        exclude them from threshold comparisons and never divide by q_e
    threshold = scipy.stats.norm.isf(alpha / (2*m))
    scan_alarm = any(score_e > threshold)
    energy = dot(coordinates, coordinates)
    energy_alarm = energy > scipy.stats.chi2.isf(alpha, nu)
    return alarms, maximum_score, energy
~~~

**Critical SVD requirement:** for a \(300\times200\) matrix, economy SVD returns only 200 left vectors. Taking its tail after rank 198 loses 100 of the 102 residual directions. Use full SVD, or an independently verified complete nullspace of \(M^\mathsf T\). Do not “fix” the resulting wrong degrees of freedom by changing the expected rank.

For \(\nu=0\), retain \(Z\) with shape \((m,0)\), \(Q=0\), and \(q=0\); do not call a chi-squared quantile with zero degrees of freedom or synthesize positive redundancy. Empty graphs in the tiny enumeration have exact rank zero and no circuits; never pass them to the floating detector.

The independent torus reference fixes both coordinates of node 0 by removing those two columns and uses pivoted QR/least squares on the resulting \(300\times198\) full-column-rank matrix. Compare \(y-M\widehat x\) with \(Z Z^\mathsf Ty\). Normal-equation inversion is not the floating-point reference.

For tiny exact checks, select independent columns \(B\) using SymPy and form
\[
Q_{\rm exact}=I-B(B^\mathsf TB)^{-1}B^\mathsf T
\]
over rationals. This small exact inverse is an oracle, not the production implementation.

### 5.4 Numerical conventions and failure behavior

- Use CPU float64 throughout statistical calculations. Keep integer construction data exact. Use rational arithmetic only for the explicitly small oracle cases.
- For the main torus, compute rank with \(\tau=\max(m,2n)\epsilon_{\rm mach}s_{\max}\). Require rank 198, the smallest retained singular value above \(1000\tau\), and every discarded singular value at most \(\tau\); record the spectrum. For the main unwhitened torus also check the proved Section 4.4 lower bound with relative slack \(10^{-10}\). Do not apply that special lower bound to the label ablation or unequal-noise matrix, or force rank 198 after a contradictory decomposition.
- Let \(A_w\) denote the whitened matrix. Use Frobenius norms for matrix checks: \(\|A_w^\mathsf TZ\|_F/\max(1,\|A_w\|_F\|Z\|_F)\), \(\|Z^\mathsf TZ-I_\nu\|_F/\max(1,\sqrt{\nu})\), and both \(\|Q^2-Q\|_F/\max(1,\|Q\|_F)\) and \(\|Q-Q^\mathsf T\|_F/\max(1,\|Q\|_F)\) must be at most \(10^{-10}\). For each independent residual reference, require \(\|r_{\rm SVD}-r_{\rm QR}\|_2/\max(1,\|y_w\|_2)\le10^{-10}\). Require \(|\sum_e q_e-\nu|/\max(1,\nu)\le10^{-10}\) and \(\max_e|q_e-0.34|\le10^{-10}\) for the main design. This allows substantial margin over dimension-scaled float64 error and is far below any operational effect of interest.
- If a tolerance fails, use the bounded repair route: inspect assembly, dimensions, precision, and the QR reference; at most one switch from the SVD driver gesdd to gesvd is allowed for a documented convergence problem. Preserve the failed result. Do not relax tolerances to accept an apparent gain.
- Squared nullspace row norms are nonnegative by construction. Do not obtain tiny \(q\) by subtracting a nearly-one hat diagonal.
- For the source chain use its exact integer stress to compute \(q_e=c_e^2/\sum c^2\). Normalize by the largest power of two before floating-point norm evaluation. Do not trust relative accuracy of a tiny SVD row norm merely because the global matrix residual is small.
- In exact tree/bridge controls, structural zero \(q_e\) means invisible. A very small positive numerical value without a structural certificate is unresolved, not automatically zero. The primary torus has no small-\(q\) ambiguity.
- Do not clip raw ratios, introduce an artificial noise floor, normalize signatures independently while leaving noise unchanged, or round circuit coefficients before exact checks.
- If raw positive ratios are accepted by an optional input wrapper, reject nonpositive values; do not silently take an absolute value before logging.
- Negative and positive faults have the same power by symmetry. Test sign handling deterministically; separate Monte Carlo for both signs is unnecessary.

### 5.5 Fixed controls and ablation

Implement only these controls:

| Case | Exact definition | Purpose and interpretation |
|---|---|---|
| Scalar tree and cycle | Three-node path; directed three-cycle; cycle lengths 4 and 20 | \(Q=0\) on a tree; \(q=1/\ell\) on a scalar cycle. Independent elementary oracles. |
| Scalar triangle plus bridge | Add a fourth vertex connected by one edge | Bridge must have \(q=0\); triangle edges retain \(q=1/3\). |
| Source chain | \(k=1,3,5,8,12\); optional numerical stress \(k=32\) only if needed to diagnose precision | Reconstruct table and coefficient-versus-normalization mechanism. |
| Triangle-protected chain | Add one same-type two-edge detour per old edge, \(k=1,3\) | Retain large circuit while verifying \(q\ge1/3\); unequal-budget, underidentified diagnostic only. |
| Rank-loss guard | Set all 300 torus signatures to A | Must be rejected as a feasible design, even if its redundancy score looks better. It violates quotas and rank. |
| Two-label-swap ablation | At \(L=10\), exchange the labels of A at \((0,0)\) and B at \((3,4)\), edge indices 0 and 103 | Same topology, counts, degree, budget, and full rank; breaks the joint topology/signature symmetry. Compute \(q_{\min}\) and the analytic bound. No optimization or stochastic tuning. |
| Known unequal channel noise | Original torus; \(\sigma_A=\sigma_B=1,\sigma_C=2\); whiten correctly | Conditional-scope check. Fault amplitude is \(10\sigma_e\) on its chosen channel. Do not claim the equal-noise design is optimal here. |

For the swap ablation, the planning probe gave rank 198 and \(q_{\min}\approx0.2850390940\). Recompute rather than inserting that answer. If the rank is not 198, investigate the implementation once; do not search for a different swap that yields a more attractive contrast.

Add deterministic invariance checks for gauge shifts, node-coordinate transformations with the corresponding signature transform, edge permutations, edge reversals, positive joint row/observation/noise rescaling, and orthonormal changes of residual basis. Negative row sign changes are handled as edge reversals; noise standard deviations remain positive. The detector must not depend on primitive integer normalization or a choice of nullspace basis.

---

## 6. Correctness, experiments, and analysis

### 6.1 Independent checks before statistical interpretation

Perform these in order:

1. Check input hashes and notebook code-cell identity. Run the immutable supplied script once with a 300-second subprocess timeout inside the exact-stage allowance. Save stdout, stderr, exit status, and elapsed time.
2. Independently verify the base chain stress, its exact left-null dimension, and at least \(k=1,3\) with SymPy. The supplied RREF code is a reference, not the sole oracle.
3. For \(L=3,4\), verify exact rank \(2L^2-2\), exact \(Q^2=Q\), \(QM=0\), and every diagonal equal to \((L^2+2)/(3L^2)\).
4. Verify the torus symmetry transformations as exact integer identities, including reorientation of negative-step edges.
5. Run the scalar controls and rank-loss guard.
6. Build the actual \(L=10\) design, validate all 300 rows, rank, quotas, degree, projector invariants, and independent QR residuals.
7. For 200 full-space Gaussian observation vectors with a nonzero deterministic latent state, compare the full least-squares detector path to residual-coordinate calculations on the **same** observations. Do not compare unrelated random samples and call that a numerical identity.
8. Check detector implementation against the analytic normal marginal and chi-squared probabilities before interpreting performance.

Use a deterministic bounded latent field for check 7, for example
\[
X(i,j)=0.3\sin(2\pi i/L)+0.2\cos(2\pi j/L),\qquad
Y(i,j)=0.25\cos(2\pi i/L)-0.15\sin(2\pi j/L).
\]
The main residual-space simulations may take \(x=0\), because \(QMx=0\) has already been independently checked.

### 6.2 Bounded counterexample search for the source theorem

To supplement the positive witness checks, enumerate the \(4^6=4096\) labelled graphs on four labelled vertices, assigning each unordered pair one of absent, A, B, C. Retain those with maximum degree at most two. For each retained graph, enumerate its nonempty edge subsets \(C\), at most 15 subsets for a graph with four edges. A subset is a circuit exactly when \(\operatorname{rank}(M_C)=|C|-1\) and deleting any one row leaves rank \(|C|-1\). For every such subset reconstruct its one-dimensional left kernel, primitive-normalize it, and verify a monochromatic simple cycle with coefficients \(\pm1\). Do not enumerate arbitrary linear combinations of kernel vectors.

The enumeration is finite and deliberately small. It can expose an implementation mistake or a genuine tiny counterexample, but cannot prove the full classification. Preserve the exact matrix and dependence if any claim fails. Stop this enumeration at 180 seconds; record incomplete coverage honestly and continue the independent torus certificate if no relevant defect has been found. Do not expand to larger graph enumeration.

The full supplied script includes an unrelated trimming lemma and prints its consolidated JSON only at completion. A timeout by itself does not show which branch was reached or which checks passed. Unless separately recorded progress establishes completed stages, mark the whole immutable invocation incomplete; retain only independently logged construction checks. This does not refute or block the independent torus certificate.

G2 may pass **for the independent torus route** when the original attachments are unavailable or only the bounded source enumeration is incomplete, provided the exact torus rank/projector, integer symmetries, scalar oracles, and essential detector checks pass. Keep the source-reproduction claim explicitly incomplete. The freeze command must distinguish these noncritical omissions from a failure of a prerequisite for the main certificate.

### 6.3 Reference and comparison roles

| Role | Implementation | Why it is sufficient |
|---|---|---|
| Simple credible reference | Scalar exact controls and least-squares residuals | Detects sign, projection, rank, and threshold bugs. |
| Closest strong design comparison | Sharp trace upper bound, with the classical redundancy interpretation [R1] | Bounds every feasible competing design, so attainment is stronger than a finite optimizer tournament. |
| Closest practical detection mechanism | Standardized residual scan with valid familywise control [R2 plus derivation here] | The proposed detector is an established mechanism; do not rename it as a new one. |
| Classical detector reference | Global residual-energy chi-squared test | Reconstructs the source example and isolates sparse-fault dilution. |
| Targeted ablation | Fixed balanced two-label swap | Holds topology, budget, quotas, and state identifiability fixed while changing the symmetry responsible for equal redundancy. |
| Mechanism counterexample | Triangle-protected chain | Tests whether one circuit's integer coefficients are an adequate full-network score. |

D-optimal sensor selection [R4], robust regression, iterative outlier deletion, and arbitrary local-search optimizers are not implemented. They address different estimation/localization objectives or cannot improve the proven \(q_{\min}\) upper bound here. This omission forbids claims of general algorithmic superiority, but does not weaken the exact certificate for the chosen objective.

There is no tuning allowance for topology, detector thresholds, or primary metrics. All methods use the same observation realization, known variance, and matrix. Numerical implementations get the same precision and tolerance requirements.

### 6.4 Development, freeze, and confirmation

Use explicit NumPy Generator/PCG64 streams with SeedSequence. Fixed base seeds:

- development and pilot: 73001;
- confirmatory primary simulation: 73002;
- unequal-noise scope check: 73003;
- the single allowed audit replication if a stochastic gate fails: 73004.

Record the stream identity and case mapping. Do not reuse pilot samples as confirmatory evidence. Use common random residual noise across detector methods and counterfactual amplitudes in a phase; this is pairing, not additional independent replication.

Development uses \(L=3,4,5\), the specified tiny controls, and at most 2,000 independent main-network pilot noise vectors. Calibration may inspect \(L=10\) dimensions and timings; that deterministic construction is not a held-out random dataset. No unseen network population is being inferred.

Before confirmation, write a protocol lock containing code revision/hash, dependency lock hash, design and edge-order hash, all cases and metrics, tolerances, thresholds, selected Monte Carlo count, seeds, resource limits, and the reason for any pre-confirmation reduction. Freeze the scientific protocol. Later changes to the primary model, algorithm, graph, threshold, outcome criterion, or case set are exploratory and require a fresh locked confirmation if budget permits. Repairs that change results require rerunning affected confirmation with the audit stream and retaining the original; follow the distinct corrected-defect and no-defect audit rules in Section 6.7.

### 6.5 Smallest sufficient experiment matrix

The proof supplies the all-edge claim. Stochastic tests check implementation and calibration; they are not random-network evaluations.

| Stage/case | Work | Primary outputs |
|---|---|---|
| Source bridge | Exact/analytic chain \(k=1,3,5,8,12\); no Monte Carlo required | Counts, rank, stress norms, \(q_{\min}\), analytic omnibus power; original-data provenance caveat. |
| Exact candidate | \(L=3,4\), plus integer symmetries | Exact proof checks and rational diagonal values. |
| Main candidate | \(L=10\), all 300 edges | Feasibility certificate, spectrum, full \(q\) vector, optimality gap to 0.34, analytic scan bound. |
| Main null | No fault | Scan false-alarm rate; omnibus false-alarm rate. |
| Main alternatives | A at \((0,0)\), \(\rho=2,5,10\); B at \((0,0)\), \(\rho=10\); C at \((0,0)\), \(\rho=10\) | Scan alarm rate, faulted-coordinate crossing rate, omnibus power, exact reference probabilities. |
| Fixed ablation | Two-label swap; deterministic analysis only | Full rank, \(q_{\min}\), trace, scan certificate at the weakest edge. |
| Scope check | Known unequal channel noise; deterministic analysis plus optional fixed 5,000-vector null and weakest-edge \(10\sigma_e\) check | Correct whitening, changed \(q\), conditional bounds and observed calibration if run. |
| Structural limitations | Coherent node shift; protected chain; scalar long cycle | Exact identities and scoped conclusions. |

Default confirmatory primary count: **20,000 independent Gaussian vectors**. Reuse the same 20,000 base vectors across the six null/alternative conditions. The five faulted conditions are not five independent network samples, nor are their 300 residual coordinates independent trials.

Because the exact symmetry maps every edge to every other edge without changing the noise distribution or scan, the three type representatives are implementation checks for an all-edge theorem. Do not infer worst-edge coverage from three empirical samples alone. Do not report fault isolation accuracy unless a separate protocol has been defined, which this brief does not authorize.

### 6.6 Streaming simulation

After full-space validation, form the small matrix \(Q=ZZ^\mathsf T\) once for the current graph (or compute a needed column as \(Z Z_{f,:}^\mathsf T\)):

~~~text
for each batch of independent g ~ N(0, I_nu):
    base_residual = g @ Z.T
    for each predeclared alternative (fault index f, amplitude rho):
        residual = base_residual + rho * Q[:, f]
        coordinates = g + rho * Z[f, :]
        compute maximum standardized residual, energy,
                and the signed fault-coordinate statistic
        save raw statistics and alarm indicators
    process the no-fault condition once for the same batch
~~~

All quantities above are whitened. In the main case \(\sigma=1\). The detector evaluation must compute the maximum over **all** 300 edges; it may not use the injected edge to decide whether to alarm. The fault coordinate is accessed only by the offline evaluator to verify (9).

Use batches of 512 or 1,024, selected by calibration. Save base Gaussian coordinates and the decomposition once per phase, plus raw per-condition statistics; this is enough to reconstruct residuals without storing six duplicate observation tensors.

### 6.7 Uncertainty and statistical gates

For each condition report numerator, denominator, point estimate, and a descriptive 95% exact binomial interval. For the fixed primary validation family use **17 simultaneous sanity checks**:

- two null metrics: scan probability at most 0.05; omnibus probability exactly 0.05;
- for each of five alternatives: the exact fault-coordinate crossing probability, exact omnibus power, and lower bound on scan probability.

Use Clopper–Pearson intervals with individual coverage \(1-0.01/17\). The union bound gives at least 99% family coverage even though conditions share random numbers. For exact probabilities, a mismatch means the analytic value is outside the interval. For the scan-null inequality, a mismatch means the interval's lower endpoint exceeds 0.05. For a scan lower bound, a mismatch means the interval's upper endpoint is below that bound.

A point estimate below 0.98059 is not by itself a contradiction. Likewise an upper confidence endpoint above 0.05 does not invalidate an analytic upper bound.

For the engineering outcome, require the simultaneous lower confidence endpoints for the three \(10\sigma\) scan cases to exceed 0.95, in addition to the analytic certificate and deterministic correctness. This is an implementation confirmation threshold, not proof of the mathematical inequality.

An unexplained mismatch triggers a deterministic audit and at most one independent replication, inside the repair reserve. If a deterministic defect is found, mark the original run invalid, freeze the corrected code, and use the sole audit stream for one fresh affected-case confirmation. If no defect is found, the replication is diagnostic: retain the original failed sanity status and report the two runs separately. Do not replace an unfavorable run with the favorable replicate, or pool a result-triggered replication with the original while claiming the fixed-sample confidence coverage above. An unresolved initial discrepancy leaves the affected empirical validation inconclusive; independently verified analytic claims retain their own status. Do not keep adding seeds until the intervals look favorable.

For a resource-reduced confirmation, select the largest count in \(\{20000,10000,5000\}\) that fits the calibrated budget before locking. At 5,000 draws the approximate standard error near probability 0.98 is 0.002; the margin above 0.95 remains useful. If fewer than 5,000 fit, complete and report the deterministic certificate but classify the requested empirical confirmation as incomplete. Do not reduce the 100-node problem to claim target-scale success.

### 6.8 Timing and memory measurements

Runtime is a feasibility measurement, not a claim of superiority over an unimplemented optimizer.

Measure and separately report:

1. cold command startup and imports;
2. graph construction, whitening, decomposition, and validation;
3. online detection for an already factored matrix, with observations prepared outside the timed region;
4. total stage time, including generation, preprocessing, failed attempts, serialization, and analysis.

Use a monotonic high-resolution clock. Perform two warm-ups and at most ten repeated decompositions for timing stability. For online work, record 200 individual snapshot latencies and one batch throughput measurement; report median and 95th percentile, not just the best run. Keep the same backend/thread policy across methods; state if a backend cannot be constrained as requested. CPU execution needs no device synchronization.

Measure process peak RSS with psutil sampling and, where available, a platform resource counter with its units verified. Include child-process memory for exact-check subprocesses. Do not confuse macOS byte-valued maximum RSS with Linux's common kilobyte convention. Record both matrix-only estimates and measured process peaks.

Report one-time preprocessing \(t_{\rm setup}\), per-snapshot detection time, and optional amortized cost
\[
t_{\rm setup}/K+t_{\rm detect}
\]
for explicitly stated \(K=1,1000\). Do not hide setup or assume unlimited reuse.

Every attempted run, timeout, convergence failure, invalid rank, and skipped prerequisite receives a status record. A faster time obtained by omitting a residual direction or changing precision is invalid, not an optimization.

---

## 7. Mac feasibility, cumulative limits, and evidence gates

### 7.1 Inspect the actual target

Before substantive computation, determine and record only relevant machine information:

- chip identity and architecture; whether Python is native arm64 or translated;
- macOS version, Python executable/version, and numerical package versions;
- total and currently available RAM, free storage in the project filesystem;
- NumPy/SciPy BLAS/LAPACK configuration and actual thread settings;
- power state and any evident thermal or competing-load limitation affecting timings.

Use platform inspection, the relevant fields from system_profiler, sw_vers, sysctl, disk usage, and psutil as available. Avoid saving machine serial numbers or unrelated environment variables. Do not infer RAM, precision support, or backend from “M3 Max.”

If the actual Mac is unavailable, complete the code and analytic derivations that do not require it, preserve the plan and source evidence, and identify precisely which target runs remain blocked. Do not execute a remote confirmatory workload and label it Mac validation.

### 7.2 Resource ceilings and estimates

These are **planning ceilings**, not measured M3 Max performance.

| Resource | Ceiling or rule |
|---|---|
| Total experimental workload wall time | **70 minutes cumulative**, including development probes, exact checks, preprocessing, simulations, failures, repairs, and report regeneration. Maintain a persistent ledger. |
| Setup/download elapsed time | Separate maximum **30 minutes**, reported as part of end-to-end effort. Experimental computation during setup also counts against the 70-minute workload cap. |
| Implementation, reasoning, and report-writing effort | Estimate **4–8 hours end to end**. This is distinct from machine workload time; no multi-day rescue project is authorized. |
| Process-tree peak RSS | At most the smaller of **1 GiB** and 25% of available RAM measured before the run; do not begin computational stages if this allowance is below 512 MiB. |
| Generated experimental artifacts | **100 MiB**, excluding the virtual environment and supplied immutable files. Compress raw arrays; preserve failed outcomes before optional plots. |
| Environment and downloads | Target at most **2 GiB**; require at least 3 GiB free before installation or report setup blocked. |
| Generated graph instances | At most **5,000**, including all 4,096 tiny enumeration assignments, development, and failed attempts. No random optimizer search. |
| Independent Gaussian vectors | At most **100,000** cumulatively across phases, full-space checks, optional scope checks, and retries. |
| Condition-specific simulation evaluations | At most **500,000**; reusing one noise vector across alternatives still consumes computation and counts here. |
| Repair allowance | At most **two repair attempts and ten minutes** of workload in total; one stochastic audit replication at most. |

The main matrix contains 60,000 float64 entries, about 0.46 MiB; a full \(300\times300\) matrix is about 0.69 MiB. A \(1024\times300\) residual batch is about 2.34 MiB. A 20,000-by-102 base-noise array is about 15.6 MiB. Python, plotting, SVD workspaces, and exact-arithmetic objects add overhead; measure rather than treating these payload sizes as peak RSS.

For \(p=2n\le m\), a conservative dense full-SVD cost including a complete left basis is \(O(mp^2+m^2p)\); at the fixed aspect ratio here it is cubic in \(n\). Simulation projections cost \(O(Nm\nu)\). The main design needs roughly \(6\times10^8\) multiply-add terms for a 20,000-vector residual projection, plus comparatively small shifted-case evaluations. These sizes suggest a minutes-scale workload, but no specific M3 timing is assumed.

### 7.3 Gates and actions

All stage times below sum to 60 minutes; the separate ten-minute repair reserve brings the total to 70. Unused time is not permission to expand the scientific scope. Check the ledger before every subprocess and simulation batch. Preserve ten minutes for the final reporting stage.

| Gate | Question, prerequisites, and minimum work | Budget | Pass / bounded repair / stop rule |
|---|---|---:|---|
| **G1: Constructive feasibility** | Inspect target; read available originals; verify input identity; review rank, symmetry, and statistical derivations; implement only generators and basic assembly. | 5 min workload | Continue if assumptions and construction are coherent. A missing optional external PDF does not block self-contained mathematics. If independent channels or a real application's link constraints are indispensable but unavailable, restrict to the declared synthetic model; do not claim field applicability. A proven algebraic defect stops the affected positive claim. |
| **G2: Tiny correctness** | Source checks, exact \(L=3,4\), scalar controls, integer symmetries, bounded four-node enumeration. | 10 min | Continue on exact agreement. Preserve any exact counterexample. One implementation repair is allowed from reserve. If the certificate's rank/symmetry proof remains unresolved, do not proceed to a large “victory” benchmark; report the verified diagnostic and an inconclusive positive claim. |
| **G3: Resource calibration** | Actual \(L=10\) SVD/QR, memory logging, 1,000 pilot Gaussian vectors, two warm-ups and short timing measurements. | 5 min | Require rank 198, numerical tolerances, resource cap compliance. Project later time with a safety factor of two. Select batch size and confirmation count before lock. If 100-node correctness cannot fit, report practical infeasibility or a blocked environment; do not substitute smaller \(L\). |
| **G4: Mechanism pilot and protocol freeze** | Complete up to 2,000 total development vectors; source bridge; swap and rank-loss controls; inspect null and fault-coordinate behavior; freeze protocol. | 10 min | Continue if the experiment can observe the effect and deterministic checks are sound. A noisy pilot is not a scientific rejection. Resolve only identified defects; no topology or threshold tuning. An unexplained numerical/model mismatch uses reserve or stops confirmation. |
| **G5: Locked validation** | Default 20,000-vector primary six-condition study; deterministic scope diagnostics; optional 5,000-vector unequal-noise check only if projected time plus report reserve fits. | 20 min | Apply Section 6.7 gates. Save incomplete runs on interruption. Drop the optional stochastic scope check first if needed, then use the prelocked reduced count. Do not alter a locked count after looking at comparative outcomes. No optimizer tournament follows a successful exact certificate. |
| **G6: Diagnosis and report** | Regenerate results, inspect plots and raw failures, write claim-specific conclusions and limitations, verify reproducibility commands. | 10 min | Deliver even after a justified early stop. Distinguish mathematical, implementation, and target-hardware conclusions. Preserve negative results and state remaining blockers. |
| **Shared reserve** | Ordinary implementation fixes or one independent stochastic audit replication; no new research branch. | 10 min | Two repair attempts maximum, counted cumulatively. If repair cannot fit while preserving G6, stop the affected claim and report it. |

Before G5, require
\[
2\,t_{\rm projected\ remaining\ experiments}
+t_{\rm reserved\ report}
+t_{\rm reserved\ repair}
\le t_{\rm remaining}.
\]
If this fails, reduce the optional scope check, batch size for memory, or Monte Carlo count according to the predeclared set. Do not change \(n=100\), \(m=300\), precision, rank requirements, detector threshold, or target effect to fit.

Calibration, failed runs, branches, and repairs never reset the caps. A command must refuse to start if its projected cost cannot fit the remaining reserve.

---

## 8. Configuration, commands, and reproducibility

### 8.1 Required configuration values

Create **protocol.toml** with the following scientific values. Engineering settings such as output paths and batch size may be added; changes to scientific values require a documented deviation.

~~~toml
[model]
latent_dimension = 2
lattice_size = 10
nodes = 100
measurements = 300
max_degree = 6
expected_rank = 198
type_counts = [100, 100, 100]
noise = "independent_gaussian_log"
sigma = 1.0
alpha = 0.05
target_fault_sigma = 10.0
target_alarm_probability = 0.95

[numerics]
dtype = "float64"
svd_driver = "gesdd"
full_matrices = true
identity_tolerance = 1e-10
diagonal_tolerance = 1e-10

[simulation]
development_seed = 73001
confirmation_seed = 73002
scope_seed = 73003
audit_seed = 73004
pilot_vectors_max = 2000
confirmation_vectors = 20000
allowed_confirmation_counts = [20000, 10000, 5000]
scope_vectors_max = 5000
batch_size = 1024
rng = "PCG64"

[budget]
workload_seconds = 4200
setup_seconds = 1800
report_reserve_seconds = 600
repair_reserve_seconds = 600
repair_attempts_max = 2
rss_bytes_max = 1073741824
artifact_bytes_max = 104857600
generated_graphs_max = 5000
gaussian_vectors_max = 100000
condition_evaluations_max = 500000
~~~

The explicit stage and case definitions in this document also belong in the lock. Do not let a missing TOML field silently select a different experiment.

### 8.2 Intended commands to implement

The following CLI interfaces **do not exist yet and were not executed during planning**. Implement them and verify them on the target. The immutable source script is the only supplied runnable program already exercised in planning.

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.in
.venv/bin/python -m pip freeze > requirements.lock.txt

.venv/bin/python -m poc.cli inspect --config protocol.toml
.venv/bin/python -m poc.cli check --config protocol.toml
.venv/bin/python -m pytest -q
.venv/bin/python -m poc.cli smoke --config protocol.toml
.venv/bin/python -m poc.cli calibrate --config protocol.toml
.venv/bin/python -m poc.cli pilot --config protocol.toml
.venv/bin/python -m poc.cli freeze --config protocol.toml
.venv/bin/python -m poc.cli confirm --lock results/protocol_lock.json
.venv/bin/python -m poc.cli report --results results
~~~

Command semantics:

- **inspect:** environment and source hashes, dependency/API checks, resource availability.
- **check:** G2 exact and deterministic checks, including the bounded immutable-script subprocess and one bounded enumeration. The subsequent pytest invocation performs lightweight independent unit checks and reuses recorded immutable-suite and enumeration results; neither large suite is automatically repeated. Every actual generation and retry still consumes the same ledger.
- **smoke:** one complete synthetic observation through generator, decomposition, both detectors, and result serialization; it is not performance evidence.
- **calibrate:** G3 target-scale timings, memory, and at most the first 1,000 pilot vectors.
- **pilot:** complete G4 without exceeding the total 2,000-vector pilot allocation.
- **freeze:** refuse if a prerequisite of the selected main route fails; permit the explicitly documented partial G2 pass in Section 6.2. Write an immutable protocol lock and code/dependency hashes.
- **confirm:** require an unchanged lock, use separate output run IDs, enforce cumulative caps, and checkpoint each completed batch.
- **report:** read saved outcomes and regenerate artifacts; do not perform fresh simulations or silently substitute cached values for missing ones.

Time and bound the initial dependency installation with a Python subprocess timeout or an equivalent available mechanism; the project CLI does not yet exist then, and GNU timeout must not be assumed installed on macOS. Include that elapsed time in the setup record.

All commands should be restartable without repeating completed experiments by accident. For each simulation batch, record its allocated work before starting and atomically commit its output record, PCG64 bit-generator state, and committed draw count at completion. Resume from the committed state; never restart the RNG at its initial seed while retaining earlier counts. An interrupted uncommitted batch remains an attempted cost in the ledger. Every actual retry is a new run ID linked to its predecessor and counted cumulatively. Ordinary implementation choices can be resolved autonomously within these interfaces.

### 8.3 What to record

For every stage and case, record:

- status: completed, incomplete, timeout, numerical_failure, rank_failure, resource_limit, dependency_blocked, source_unavailable, or excluded_by_protocol;
- model/case identifiers; exact graph and edge order; fault location and amplitude used only by the evaluator;
- rank, singular values, tolerance, full \(q\) vector, and exact/analytic expected values;
- code and dependency hashes; source hashes; RNG algorithm, stream seeds, batch sizes, and counts;
- actual device/backend/precision/thread policy, wall time, CPU time when available, peak RSS, and disk use;
- raw alarm counts and per-trial statistics, confidence intervals, and validation decisions;
- failed assertions, stderr, known interruptions, and reasons for any exclusions.

Save the main \(M,Z,q\), graph edge list, and phase base-noise arrays. Save per-condition maximum score, energy, and fault-coordinate statistic, not just success percentages. Raw full-space correctness examples should include \(y\), \(\sigma\), and the relevant expected residuals. Compression is allowed; lossy rounding is not.

No external dataset is required. No downloads beyond dependencies and optional source papers are experimental inputs. Source articles are for attribution; loss of an external link does not justify substituting an unverified theorem.

---

## 9. Outcome interpretation and required final artifacts

### 9.1 Classify each claim, not just the entire project

| Outcome | Evidence required | Permitted conclusion |
|---|---|---|
| **Supported contribution** | Valid rank/symmetry/trace derivation, exact and independent numerical checks, target-scale execution, successful calibration, and adequate confirmatory implementation checks | The specified construction attains the classical redundancy certificate and meets the stated single-comparison detection target under the declared model. Historical novelty still requires separate evidence. |
| **Informative negative result** | A verified mathematical obstruction, or a correct and sufficiently precise evaluation contradicting a scoped empirical hypothesis | Coherent node faults cannot be promised detectable; circuit height is not an independent Gaussian design score; or a particular tested extension loses its guarantee. Do not call these a disproof of the source theorem. |
| **Practical infeasibility** | Measured target costs or a supported scaling/resource argument prevents the intended stage | The relevant implementation/study cannot be completed within this resource envelope. A slow or failed dependency installation is an environment blocker unless it supports a broader cost claim. |
| **Inconclusive evidence** | Missing hardware/source access, unresolved derivation, insufficient simulation precision, or budget exhaustion | State what is verified and exactly what remains untested. A proof may be supported while Mac performance remains unverified. |
| **Invalid experiment** | A rank, covariance, threshold, leakage, numerical, or implementation error undermines the attempted comparison | Preserve the invalid run and diagnostic; report only corrected results if a bounded valid rerun is completed. |

Absence of a new algorithmic advantage is expected from the prior-art and exact-bound analysis; it is not evidence that comparison-only edge detection is useless. Conversely, a 98% model-based bound does not establish a physically deployable sensor security system.

Do not translate an optional unequal-noise failure into a refutation of the homoscedastic certificate. Do not translate the source's valid arithmetic theorem into a guarantee for its statistical application.

### 9.2 Required report content

Write **results/report.md** as a standalone research report with:

1. the exact question and constraint set;
2. the contribution and prior-art boundary;
3. the analytic certificate and coherent-node obstruction, with reproducible derivations;
4. the source-chain reconstruction and what was not recovered from the original simulations;
5. correctness evidence, including rank, symmetry, independent residuals, and the complete residual dimension;
6. all executed cases, analytic versus empirical probabilities, actual false-alarm levels, intervals, and failures;
7. measured Mac environment, setup cost, factorization/detection timing, peak memory, and cumulative resource ledger;
8. limitations and a claim-by-claim outcome classification;
9. at most one justified next step.

A reasonable next step, only if the certificate and calibration hold, is to test a **real, supplied covariance and allowed-link set** using the same redundancy framework. Do not automatically launch it. This would require an actual application contract; unspecified “realistic” random constraints would create a different project.

### 9.3 Deliverables for the stages actually reached

For a completed implementation, deliver runnable code, README instructions, frozen configuration/dependencies, exact source provenance, design certificate, raw and summarized results, and the report.

Produce only essential figures:

- chain \(q_{\min}\) and analytic detection probability versus \(k\), with the model and faulted-edge rule stated;
- main detector probabilities versus \(\rho\), showing analytic omnibus values, the scan lower bound, empirical intervals, and the distinct false-alarm interpretations;
- optionally, all-edge redundancy values for torus, swap, and known-noise scope check.

Use ordinary static scientific plots with readable axes and units. Do not smooth empirical points into unsupported trends, treat different edges as independent samples, or hide numerical failures from plots or tables.

If stopped analytically, deliver the verified derivation or exact counterexample, the minimal diagnostic code actually written, available raw evidence, and a clear list of unexecuted stages. Do not create fake benchmark tables or claim that commands ran.

---

## 10. Research coverage and reference list

### 10.1 Coverage and novelty assessment

Searches on 29 October 2026 used both available search engines and followed terminology across geodetic internal reliability, minimal detectable bias, redundancy numbers, balanced observations, residual leverage, effective resistance, pairwise sensing, circuit imbalance, multicommodity-flow circuits, direction networks, and triangular/periodic networks.

The decisive backward connection is from the proposed Gaussian score to established redundancy design [R1], with later simulation-based network design providing a close practical comparison [R2]. The circuit literature establishes that gadget amplification is substantially precedented [R5–R7]. The exact threshold theorem's priority remains unresolved; proving it is not required for the independent torus certificate, but its specific chain construction is audited.

Relevant primary-source bodies were read. Rofatto's printed pages 156, 157, and 160 were also inspected visually from the retrieved PDF where extraction obscured formulas. Its printed definition of \(W\) after Eq. (3) appears to omit an inverse; the implementation must use the independent whitening derivation here. Some publisher/older PDF routes failed; accessible author or journal copies were used where available. No exhaustive citation census or global novelty determination was performed. No actual sensor dataset, M3 benchmark, or full confirmatory simulation was available during planning.

### 10.2 References and precise locations

**[S1] Supplied, unattributed research note.** *When unit-free comparisons require large exponents: A sharp degree threshold for fixed rational comparison signatures.* 29 October 2026. File **unit_free_comparison_research(1).ipynb**. Critical locations: Section 1/main theorem (cell 1); bounded-regime proof (cell 2); doubling gadget and rank induction (cell 3); lift and degree splitting (cell 4); Hadamard bound (cell 5); polynomial-degree corollary (cell 6); finite evidence and exact code (cells 7–8); prior-art assessment (cell 9); supporting robust inference and scale results (cells 10–11); limitations (cell 12). This is a supplied source without a public URL or verified author attribution.

**[S2] Supplied exact-check implementation.** **research_checks(1).py**, 29 October 2026 attachment. Critical functions: comparison_matrix, chain, lift, reduce_degree, check_circuit, change_palette, and main. Its finite successes verify implementations and examples, not historical priority or universal truth by enumeration.

**[S3] Supplied problem and request.** **Pasted text(20261029-164446).txt**, 29 October 2026. The four detection percentages are reported observations without accompanying simulation code or raw trials.

**[R1] AliReza Amiri-Simkooei.** “A New Method for Second Order Design of Geodetic Networks: Aiming at High Reliability.” *Survey Review* 37(293), 552–560, 2004. DOI: [10.1179/003962604791482306](https://doi.org/10.1179/003962604791482306). [Author-uploaded full text](https://www.researchgate.net/publication/233621754_A_NEW_METHOD_FOR_SECOND_ORDER_DESIGN_OF_GEODETIC_NETWORKS_AIMING_AT_HIGH_RELIABILITY). Read: introduction and “Reliability Criteria of Geodetic Networks,” Eqs. (1)–(5). Critical prior art: minimum detectable bias, residual diagonal, average redundancy, and explicit maximum-minimum redundancy objective. Balancing weights there is not the same engineering construction as the fixed equal-noise topology here.

**[R2] Vinicius Francisco Rofatto, Marcelo Tomio Matsuoka, and Ivandro Klein.** “Design of Geodetic Networks Based on Outlier Identification Criteria: An Example Applied to the Leveling Network.” *Boletim de Ciências Geodésicas* 24(2), 152–170, 2018. DOI: [10.1590/S1982-21702018000200011](https://doi.org/10.1590/S1982-21702018000200011). [Full journal HTML](https://www.scielo.br/j/bcg/a/F5hWZDGfDwyJ4SrFHk7KsBn/?lang=en). [PDF](https://www.scielo.br/j/bcg/a/F5hWZDGfDwyJ4SrFHk7KsBn/?format=pdf&ilang=en&lang=en). Read: Sections 2–4, especially the distinction between detection and identification (p.155), Eqs. (7)–(9) (pp.157–158), and residual formulas (11)–(12) (p.160). This is prior art for simulated reliability-based design, not a source of the torus proof.

**[R3] Daniel A. Spielman and Nikhil Srivastava.** “Graph Sparsification by Effective Resistances.” *SIAM Journal on Computing* 40(6), 1913–1926, 2011. DOI: [10.1137/080734029](https://doi.org/10.1137/080734029). The [STOC 2008 conference version inspected](https://www.cs.cornell.edu/~abrahao/tdg/papers/p563.pdf), Section 2.3 and Section 3, Lemma 3, gives the effective-resistance projection and diagonal identity. Cited here only for the scalar equivalent formulation.

**[R4] Siddharth Joshi and Stephen Boyd.** “Sensor Selection via Convex Optimization.” *IEEE Transactions on Signal Processing* 57(2), 451–462, 2009. DOI: [10.1109/TSP.2008.2007095](https://doi.org/10.1109/TSP.2008.2007095). [Author PDF](https://web.stanford.edu/~boyd/papers/pdf/sensor_selection.pdf). Read: Sections II–III, including the D-optimal objective and selected/unselected swaps in Section III.E. This is a relevant general selection approach, but a different primary objective; no speed or accuracy comparison to its implementation is claimed.

**[R5] Richard Hladík.** *Combinatorial Algorithms for Flow Problems.* Bachelor's thesis, Charles University, Prague, 2021. [Full thesis](https://rihl.cz/bc-thesis.pdf). Critical location: Section 4.5, printed pp.54–57, Figure 4.2, Lemma 4.11, and Corollary 4.12 (PDF indices 61–64). Five-vertex block constructions produce exponential multicommodity circuits. Their existence precedents amplification; equivalence to the complete fixed-palette degree classification was not established.

**[R6] Daniel Dadush, Zhuan Khye Koh, Bento Natura, and László A. Végh.** “On Circuit Diameter Bounds via Circuit Imbalances.” *Mathematical Programming* 206, 631–662, 2024. DOI: [10.1007/s10107-024-02107-x](https://doi.org/10.1007/s10107-024-02107-x). [Repository PDF](https://ir.cwi.nl/pub/34308/34308.pdf). Read: definitions in Sections 1–2 and Section 2.3, printed p.640. Their imbalance is a nonzero coefficient ratio, distinct from primitive integer height. The discussion of general computational hardness warns against an assumed cheap circuit oracle; it does not prove hardness of this restricted comparison class.

**[R7] Steffen Borgwardt, Nicholas Crawford, Sean Kafer, Jon Lee, and Angela Morrison.** “On Circuit Imbalance and 0/1 Circuits for Coloring and Spanning Forest Problems.” Preprint, 4 December 2025, arXiv:2512.05223v1. [Full text](https://arxiv.org/html/2512.05223v1). Read: introduction, definitions, and Section 2, Theorems 2.1–2.4. Establishes broad exponential imbalance in graph-derived constraints; it does not establish the particular statistical application or exact threshold's priority.

**[R8] SciPy developers.** “scipy.linalg.null_space.” Official SciPy API documentation, accessed 29 October 2026. [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.null_space.html). Relevant: orthonormal SVD nullspace, rcond, and numerical rank behavior. Verify against the installed version.

**[R9] SciPy developers.** “scipy.stats.ncx2.” Official SciPy API documentation, accessed 29 October 2026. [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html). Relevant: noncentrality parameter and survival-function evaluation.

**[R10] SciPy developers.** “scipy.linalg.svd.” Official API documentation, accessed 29 October 2026. [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html). Relevant: full_matrices and driver choice. The basic full-SVD API was used in planning calculations; verify the installed target API during setup.

**[R11] Mark A. Atherton, Ronald A. Bates, and Henry P. Wynn.** “Dimensional Analysis Using Toric Ideals: Primitive Invariants.” *PLOS ONE* 9(12), e112827, 1 December 2014. DOI: [10.1371/journal.pone.0112827](https://doi.org/10.1371/journal.pone.0112827). [Full text](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0112827). Read: “Power Products and Toric Ideals” and “The Gröbner Fan, Primitive Invariants and the Graver Basis.” Establishes prior use of toric/primitive-invariant language; not a source of a statistical detectability guarantee.

---

## Appendix A. Self-contained source-chain reconstruction

Use this specification to implement an independent generator without relying on notebook execution. The attached script remains the immutable reference.

The base has nodes 0–4 and these oriented rows and stress coefficients:

| Tail | Head | Type | Coefficient |
|---:|---:|---|---:|
| 0 | 3 | C | -1 |
| 0 | 4 | A | 1 |
| 1 | 3 | A | 1 |
| 1 | 2 | B | 1 |
| 0 | 2 | B | 1 |
| 1 | 4 | C | -1 |
| 2 | 3 | B | 2 |
| 3 | 4 | B | 1 |

Initialize the output edge to \((2,3)\). To extend a chain:

1. Remove its current output edge \((u,v,a,t)\).
2. Start a fresh base gadget, mapping its nodes \(0\mapsto u\), \(4\mapsto v\), and \(1,2,3\) to three fresh node IDs.
3. If the removed type was B, exchange both signature coordinates of every fresh edge so that the omitted fresh A input would have type B.
4. Omit the fresh input edge \((0,4)\).
5. Append the other seven mapped edges with coefficients equal to \(-t\) times their base coefficients.
6. Set the new output to the mapped \((2,3)\) edge.

Each step adds three nodes and six net edges. The old and fresh deleted rows cancel in the equilibrium certificate. Some magnitude-one original coefficients survive, so the recurrence remains primitive. Independently reconstruct the left kernel on small cases; checking \(M^\mathsf Tc=0\) alone does not establish that it is one-dimensional.

The note's rank induction is essential: deletion of a nonzero circuit edge preserves its row span; fresh input endpoints are forced equal, eliminating the old localized right-kernel freedom and leaving only the new output-tail freedom. This establishes \(r=2n-3\). Do not replace that reasoning with the generally false assertion that gluing arbitrary vector-incidence circuits always produces one circuit.

For the full source threshold's lifted/split families, use the supplied functions and their exact checks. Reimplementing all general-palette machinery is not required for the selected detection certificate.

---

## Appendix B. Review findings and resolved stress tests

This brief was reviewed in three distinct passes during planning, including independent mathematical/novelty, experimental, and execution scrutiny. The table records substantive dispositions; it does not claim formal proof verification or freedom from every possible defect.

| Challenge | Disposition built into this brief |
|---|---|
| The paper might be missing or nonconstructive | The notebook is the paper. Its relevant gadget and rank argument are explicit; Appendix A is self-contained. Missing originals block only their reproduction/attribution audit, not independent torus work. |
| The sensor narrative might not match the theorem | Explicitly retain two coordinates and three independently observed types; exclude scalar transfer and coherent sensor-fault promises. |
| A known method might remove the novelty | The classical redundancy objective is acknowledged. Replace a claimed new optimizer with an exact benchmark and source-transfer limitation. |
| Only favorable synthetic inputs work | The main claim is an explicit existence/optimal-certificate result under declared constraints, not typical-case performance. Include source worst cases, scalar controls, a fixed same-budget label ablation, and unequal-noise scope check. |
| A stronger baseline removes the gain | Compare to an analytic bound on every feasible design, and include a standard sparse residual scan rather than relying only on a weak omnibus detector. No heuristic superiority claim remains. |
| Omnibus power is mistaken for best possible detection | Its 68.4% value is detector-specific. The primary scan certificate is about 98.1%; actual scan optimality remains unclaimed. |
| Circuit height is mistaken for a complete statistical score | Use normalized full residual geometry and the protected-chain counterexample. |
| More residual freedom is obtained by dropping state information | Full rank 198 is mandatory; an all-A control must be rejected. |
| Economy SVD loses residual directions | Full left-nullspace construction and independent QR residual checks are mandatory. |
| Empty or zero-residual controls call undefined statistics | Empty graphs remain exact-only; zero residual dimension explicitly returns no alarms, without division by zero or a zero-degree chi-squared quantile. |
| Finite precision fabricates tiny sensitivity | Exact stress normalization for chains, a large torus rank gap, explicit tolerances, and no \(1-H_{ee}\) subtraction for small \(q\). |
| A synthesized product channel is counted twice | Independent direct channels are an explicit prerequisite; covariance changes require correct whitening and a separately scoped interpretation. |
| Preprocessing or backend costs dominate | Measure cold setup, decomposition, online detection, and total cost separately on the Mac; no hidden amortization. |
| Memory/runtime exceeds estimates | Streaming batches, measured RSS, finite caps, fixed reduction ladder, mandatory report reserve; never shrink the main network to claim success. |
| A pilot is noisy or the main hypothesis is false | Separate analytic checks, implementation errors, stochastic tail events, and inadequacy of evidence. Bound repairs and preserve all runs. |
| A favorable rerun replaces a failed check | Corrected-defect confirmation and no-defect diagnostic replication have different dispositions; no result-dependent pooling or pass selection. |
| A timed-out source script is credited with unseen passes | Only separately recorded completed evidence is retained; the independent certificate can receive a partial G2 pass with source reproduction explicitly incomplete. |
| An external source is inaccessible | Core equations and construction are included; record access limits. Do not invent source claims or replace missing user mathematics. |
| The actual Mac is unavailable | Complete unaffected analytic/code work and identify unperformed target validation explicitly. |

**Execution instruction:** proceed through authorized gates autonomously, repair ordinary reversible implementation problems within the stated reserve, and stop only where the evidence or resource rules require it. Deliver a truthful, reproducible result even when the outcome is a limitation, a blocked hardware check, or an early analytic stop. Do not continue searching, tuning, or rerunning merely to obtain a favorable headline.

