# Execution brief: a calibrated challenge for hidden time asymmetry

**Executor:** an autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning.  
**Planning and literature-search date:** 1 November 2026.  
**Target experimental computer:** one MacBook with an M3 Max; inspect its actual configuration before allocating work.  
**Deliverable of your execution:** a reproducible, evidence-backed research result, including a justified early stop if an evidence gate fails. This document is an instruction to implement and execute the study. It is not a report of a completed PoC.

## 1. Project decision and smallest defensible contribution

### 1.1 Execute this project

Build a small reliability challenge for software assessing time-reversal asymmetry in labeled binary multichannel observations. Use stationary **dynamic space packing** on a finite graph: an event activates one vertex and erases every active neighbor. Test a theorem-derived overlap statistic alongside a conventional full-state symmetry test and the current `irreversibility` package's discrete detailed-balance statistic.

The intended decision is whether a particular **method, observation representation, lag, and sample budget** can support a useful time-asymmetry conclusion on a controlled challenge. The result should tell a software maintainer which configuration detects an available signal, which restricted configuration has a proved blind spot, and when even complete observations at the chosen sampling design carry too little evidence.

**Primary falsifiable research question:** With 512 independent stationary pairs of complete labeled snapshots from the unit-rate four-cycle, separated within each pair by \(\tau_* = \log(4/3)\), does a correctly calibrated current full-state software method achieve at least 80% detection probability at nominal 5% type-I error, alongside the overlap witness, while the specified two-bit method remains blind and count-only observations admit an exact reversible explanation?

The primary software method is `irreversibility==1.4.0`, `DiscreteMetrics.DetBalance.GetStatistic(..., conditional=False)`, with the exact conditional calibration specified below. Bowker's full-state symmetry statistic supplies an independent, simple reference. The study also tests whether increasing lag to 2 destroys useful finite-budget detectability, despite mathematically nonzero irreversibility. That latter condition has an information-theoretic upper bound, not merely an anticipated negative p-value.

**Contribution sought:** an independently checkable, graph-specific validation challenge combining a rigorously matched reversible control, an explicitly blind analysis configuration, finite-sample power measurements, and a certificate of insufficient information at a specified budget. This is a diagnostic and research-software contribution. A useful result must include the strong external comparator and translate its behavior into an actionable configuration decision. Merely plotting a nonzero cubic moment is insufficient.

Do not claim a new general irreversibility detector. Dynamic space packing, cubic time-asymmetry tests, higher-order irreversibility, and complete-state transition tests all have prior art [R2–R9]. Historical priority for the supplied theorem and for this exact benchmark construction is **unconfirmed**. Publication-level novelty is not a prerequisite for completing this bounded diagnostic, but must not be asserted without additional evidence. If standard tools already handle every informative fixture well, report that they pass this challenge; do not manufacture a superiority claim for the overlap witness. If the result adds no decision-relevant information beyond existing documented behavior, describe it as a validated replication/diagnostic, not an original methodological advance.

### 1.2 Fixed scope and explicit revision of the original problem

The original practical purpose is retained: prevent an analysis limitation from being mistaken for reversibility of a hidden system. The first experiment uses **independent stationary two-snapshot episodes**, because this makes finite-sample null calibration exact and inexpensive. A separate, mandatory synthetic bridge uses a single sequential stationary process with explicitly separated observation windows and a proved dependence allowance. This does **not** validate inference on arbitrary densely sampled, overlapping windows of an unknown process.

The four suggested explanations for a negative result are not generally identifiable from a finite black-box record, and are not mutually exclusive: a reversible observed count process can simultaneously hide an irreversible full process. Keep separate fields rather than force a single four-way classifier:

| Field | Allowed benchmark conclusion | Basis |
|---|---|---|
| Full dynamics | Reversible / irreversible / unestablished | Known generator and mathematical certificate |
| Observed representation | Reversible under the known model / reversible with a matching irreversible explanation / asymmetric / unestablished | Observation-map proof or exact probability tables; record matching explanations separately where available |
| Analysis restriction | Certified blind for this defined statistic / retains a known signal / unestablished | Theorem plus the actual test definition |
| Finite evidence | Detected / no asymmetry detected / invalid input or failed computation | Calibrated test, uncertainty and provenance |
| Budget limitation | Target power impossible at this budget / measured power adequate / unresolved | Information bound or independent-dataset power study |

On unknown real data, a nonrejection means **“no asymmetry detected by this configuration.”** Never promote it to a claim that the hidden system is reversible. A detector cannot infer the benchmark's oracle annotations merely from its input array.

Exclude AI-memory usefulness or safety, biological model validity, utility/reward inference, thermodynamic energy or dissipation estimates, causal identification of the reset mechanism, nonstationary inference, directed graphs, state-dependent rates, partial erasure, missing-data reconstruction, and universal detector superiority. Do not implement the supplied variance-extremum or deterministic-reset projects: they are separate results and do not improve this study's decision. No neural network, GPU training, external dataset, dashboard, service or distributed infrastructure is needed.

### 1.3 Success and failure are defined before running

Use nominal \(\alpha=0.05\), a two-sided overlap test, and the same observations for every information-preserving method. The 80% power target is an **engineering acceptance target**: this challenge should usually expose a substantial blind spot within a small record, not depend on exceptionally lucky data. It is not a claimed universal standard from the literature. The fixed useful budget is 512 pairs at the reference lag; it was selected using the small probability calculation in §4, before any comparative benchmark.

For the primary alternative, estimate each method's detection probability across 200 independent datasets. Classify its 80% target as supported only if the lower endpoint of a two-sided 95% Wilson interval is at least 0.80; contradicted at that budget if the upper endpoint is below 0.80; otherwise unresolved. Report all three methods separately. Bowker success cannot be substituted silently for external-package success. If the overlap implementation disagrees with its exact power oracle, investigate validity before interpreting that as a scientific negative.

Success at M=2,048 may identify a different useful operating point, but does not retroactively satisfy the primary M=512 target. Keep that distinction even if it makes the headline less favorable.

For the matched primary null, use 400 datasets. Exact conditional arguments establish nominal size; simulations screen for implementation errors and give observed uncertainty. Do not claim that a finite estimated rate proves an exact 5% bound. A 95% Wilson upper limit at most 0.10 is a **gross-calibration screen**, not a 5% guarantee. Also apply the multiplicity-controlled null alarm in §8.4. Known blind methods need mathematical certificates, not a nonsignificant empirical result.

Do not require the overlap witness to outperform established methods. The smallest valuable software conclusion is a validated table such as: full-state configuration detects; the defined two-bit configuration cannot; count-only data cannot identify the hidden arrow; long-lag data are insufficient at the available budget. If a stronger baseline removes any apparent gain, retain and explain that result.

## 2. Inputs, provenance and evidence status

### 2.1 Files to receive and preserve

Place the supplied files, when available, under `inputs/` without altering them:

| File | Role and precise locations |
|---|---|
| `revisable_memory_research.pdf` | Supplied 18-page research note, **An exact test for irreversible local revision**, dated 1 November 2026; no author name is displayed. Model/sign theorem pp. 2–7; two-bit criterion pp. 7–8; three-bit TV pp. 8–9; C4/count example p. 9 and figure p. 10; computational claims pp. 13–14; scope/novelty pp. 14–16; four-bit appendix pp. 16–17. Treat it as an unpublished supplied claim with inspectable proofs, not established priority. |
| `revisable_memory_research.zip` | Reproducibility archive. `research/learning/verify_time_arrow.py` and `time_arrow_results.json`; `research/deliverable/report_blocks.json` is editable full mathematical text; `core_blocks.json` contains the temporal core. The archive PDF is byte-identical to the separately supplied PDF. |
| `Pasted text(20261101-140007).txt` | Original research-design assignment. This brief resolves its central choices; this text is provenance, not a required runtime input. |

Verified attachment SHA-256 values:

```text
revisable_memory_research.pdf
638141eb5b3b5cdd1df74425b601fdbb3ee00b934a9e077888dd995db18cda08
revisable_memory_research.zip
683f2bde9d6e5d41b4042a5664452fa6ec02ea8dd9d31269171d095209fc1f24
```

Extract the ZIP with path-traversal checks. Preserve its original result files; write rerun results elsewhere. Its variance/reset scripts and stored results were inspected during planning but were not rerun in full. The supplied figure contains analytic curves, not Monte Carlo evidence. PDF equation text can be recovered from `report_blocks.json`; check the rendered equation if they disagree. Do not rebuild the PDF merely to run this project.

The essential mathematics and algorithms are included here, so absence of the archive does not prevent the core implementation. If attachments are missing, perform the independent checks specified here, mark source-reproduction steps unavailable, and request the exact missing file only if necessary to resolve a discrepancy. Do not claim to have rerun an unavailable source program.

External resources required for the software comparison are the public PyPI release and official source in [R4], plus its primary paper [R3]. Downloads are allowed; experimental computation remains local. No paid or proprietary data are required.

### 2.2 Central claim-to-evidence map

“Audited” below means bounded proof review and checks, not a formal proof-assistant certification.

| Claim | Source / status | Remaining uncertainty and resolving check |
|---|---|---|
| Insertion plus neighbor erasure and equal-rate stationary local maxima are known | Dandekar–Krapivsky, Appendix D, Eqs. D1–D5 [R2] | No novelty claim; validate the actual generator independently |
| Strictly negative overlap at every positive lag exactly off the union-of-cliques class | Supplied Theorem 1, p. 3, proof pp. 4–7 [R1]; audited, tiny checks passed | Coding implementation, numerical stability; exact tiny matrices and formula comparisons |
| All individual two-bit reversal laws agree iff closed-neighborhood rate sums are componentwise constant | Supplied Theorem 2, pp. 7–8 [R1]; audited | Do not extend to joint distributions of many estimated features; use precisely defined restricted test |
| C4 count trajectory and each individual coordinate trajectory are reversible | [R1], pp. 8–9; audited | Verify strong lumpability / coordinate rates directly |
| Reversible refresh matches snapshots, balanced two-bit laws, and C4 count paths | Derivation in §3.5 introduced for this brief; independent small matrix checks | Reproduce derivation and verify matrices; no historical originality claim |
| Exact finite-step exponential sampler | Constructive derivation in §5.2 introduced for this brief | Not implemented/benchmarked during planning; validate against events and matrix exponentials |
| C4 overlap finite-pair power and long-lag impossibility bound | Derived and numerically evaluated during planning, §4 | Recompute on target with independent count/table implementation |
| Current strong software comparator exists | Zanin 2026 [R3], official v1.4.0 source [R4], directly inspected | Mac import/API and adapter equivalence still unverified |
| Useful finite-record operating regime for all three methods | Research hypothesis | Locked comparison; retain unfavorable results |
| Novel general detector or proven real-system applicability | **Not established and not claimed** | Outside this project |

## 3. Mathematics to implement and audit

### 3.1 Objects, domains and stationary generator

Let \(G=(V,E)\) be a finite simple undirected graph with labeled vertices \(V=\{0,\ldots,n-1\}\). An independent set contains no adjacent vertices. State \(I\subseteq V\) must be independent. Rates \(\lambda_v\) are fixed positive real numbers; the experiment uses the exact integer rates listed in §7. Each vertex has an independent Poisson event clock. With \(N(v)\) the open neighborhood and \(N[v]=N(v)\cup\{v\}\), apply

\[
U_v(I)=(I\setminus N(v))\cup\{v\}.
\]

Events at already active vertices are permitted and do not change state. For row-generator convention,

\[
Q_{IJ}=\sum_{v:U_v(I)=J}\lambda_v\quad(J\ne I),
\qquad Q_{II}=-\sum_{J\ne I}Q_{IJ},\qquad P_\tau=e^{\tau Q}.
\]

Use all independent sets for construction checks, including transient states; the stationary support consists exactly of independent sets meeting every connected component. The empty set has zero stationary mass when \(n>0\). Do not assume irreducibility on the full independent-set space or divide by zero-mass states. Restrict to the recurrent support for stationary joint-table KL calculations.

At stationarity draw independent ages \(A_v\sim\mathrm{Exp}(\lambda_v)\), and retain \(v\) iff its age is strictly smaller than every neighbor's. These are backward ages of the most recent Poisson events. This gives an exact stationary draw without burn-in [R1, pp. 3–4]. Coupling two initial states with common clocks gives

\[
\Pr(I^{(1)}(t)\ne I^{(2)}(t))
\le\sum_v e^{-a_vt},\qquad a_v=\sum_{w\in N[v]}\lambda_w.
\]

Thus the law is unique even though some independent states are transient.

For an independent vertex set \(A\), the monomial and stationary moment satisfy

\[
x_A=\prod_{v\in A}x_v,\quad a_A=\sum_{w\in N[A]}\lambda_w,
\quad Lx_A=\sum_{v\in A}\lambda_vx_{A\setminus\{v\}}-a_Ax_A,
\]

\[
m_\varnothing=1,\qquad m_A=\frac{\sum_{v\in A}\lambda_vm_{A\setminus\{v\}}}{a_A}.
\]

For tiny graphs use rational arithmetic for these moments and Möbius inversion,

\[
\pi(I)=\sum_{A\supseteq I,\ A\text{ independent}}(-1)^{|A|-|I|}m_A.
\]

Independently solve \(\pi Q=0,\sum_I\pi(I)=1\) on the recurrent support and compare. Enumeration is exponential and is an **oracle for tiny graphs**, never a hidden requirement of the overlap detector or the sampler. Cap it at \(n\le8\), recurrent states \(S\le128\). Core experiments have at most five vertices. Counting all independent sets is unnecessary for general observation scoring.

### 3.2 The local and aggregate time-arrow witnesses

Define \(X_v(t)=1_{v\in I(t)}\), \(W(t)=|I(t)|\), and

\[
g(I,J)=(|I|-|J|)|I\cap J|,\qquad
\mathcal A_G(t)=\mathbb E[g(I(0),I(t))].
\]

For distinct nonadjacent \(u,v\), put

\[
\alpha=\lambda_u,\quad\beta=\lambda_v,\quad a=a_u,\quad b=a_v,
\quad c=\sum_{w\in N(u)\cap N(v)}\lambda_w,\quad D=a+b-c.
\]

The single-coordinate mean is \(p_v=\lambda_v/a_v\). The stationary pair moment and covariance are

\[
m_{uv}=\frac{\alpha\beta(a+b)}{abD},\qquad
K_{uv}=\operatorname{Cov}(X_u,X_v)=\frac{\alpha\beta c}{abD}.
\]

For adjacent vertices the covariance is \(-p_up_v\). Define

\[
H_{uv}(t)=\mathbb E[X_u(0)X_v(0)X_u(t)]
-\mathbb E[X_u(0)X_u(t)X_v(t)].
\]

The all-rate source identity is

\[
H_{uv}(t)=-K_{uv}\left[
\frac{a-\alpha}{b-c}(e^{-at}-e^{-Dt})
+\frac{\alpha}{a-c}(e^{-bt}-e^{-Dt})\right].
\tag{1}
\]

Here \(a-c\ge\alpha>0\), \(b-c\ge\beta>0\), and \(D>\max(a,b)\). Consequently \(H<0\) exactly when the endpoints have a common neighbor. Adjacent and identical endpoints contribute zero. Expanding the snapshot statistic gives an **ordered** sum

\[
\mathcal A_G(t)=\sum_{u,v}H_{uv}(t).
\tag{2}
\]

There is no cancellation. A graph has no nonadjacent pair with a shared neighbor iff all components are cliques. The clique process satisfies detailed balance; otherwise (2) is strictly negative at every finite positive lag. This proves the source's sign and reversibility classification within the stated model [R1, Theorem 1 and proof]. The derivative check is

\[
H'_{uv}(0)=-\frac{\alpha\beta c}{bD},\qquad
\mathcal A'_G(0)=-\sum_w\lambda_w\mathbb E\binom{|I\cap N(w)|}{2}.
\]

Do not introduce an extra factor of two in the aggregate derivative. The alternative symmetric-difference formula in the paper is an equality **in expectation after using stationarity**, not a pointwise identity for every pair.

For stable evaluation use

```python
exp_difference = exp(-r*t) * (-expm1(-(D-r)*t))
```

instead of subtracting two nearby exponentials. If a later calculation divides this difference by \(D-r\), use its continuous limit \(t e^{-rt}\) when \((D-r)t\) tends to zero. No such denominator vanishes in the selected integer-rate cases. Classify structural zeros from exact graph/rate identities, not from a floating-point cutoff on an exponentially small signal.

### 3.3 Precisely what is blind

The single-coordinate semigroup gives

\[
\operatorname{Cov}(X_u(0),X_v(t))
=\operatorname{Cov}_\pi(X_u,X_v)e^{-a_vt}.
\tag{3}
\]

Every individual two-bit observation law is invariant under reversal iff \(a_v\) is constant on each connected component [R1, Theorem 2, pp. 7–8]. Necessity follows along an edge, whose stationary covariance is nonzero; equality propagates through a component. Sufficiency follows from (3) and independence between components. Binary two-variable laws are determined by their means and product moment. With unit rates, this is componentwise regularity.

Each individual coordinate's **whole trajectory** is a stationary two-state Markov chain, with rates \(0\to1=\lambda_v\) and \(1\to0=a_v-\lambda_v\), hence reversible. This is stronger than a statement about a particular pairwise statistic.

However, two observations of the whole configuration, or two complete channels observed at two times, include more than two binary samples. Their distributions can be asymmetric. Joint fluctuations/dependencies of many pairwise estimates can also reveal higher-order information. Never label all methods called “pairwise” powerless. The restricted finite-sample test in §6.4 has a narrower, proved blindness guarantee.

For a nonadjacent balanced pair \(a=b\), (1) reduces to

\[
H_{uv}(t)=-\frac{\alpha\beta c}{aD(a-c)}(e^{-at}-e^{-Dt}),
\quad D=2a-c.
\]

For the specified three-bit design and its time reflection,

\[
\nu=\mathcal L(X_u(0),X_v(0),X_u(t)),\quad
\nu^R=\mathcal L(X_u(t),X_v(t),X_u(0)),\quad
\operatorname{TV}(\nu,\nu^R)=4|H|.
\]

For \(Y=(X_u,X_v)\), the four-bit two-snapshot law has TV \(6|H|\) [R1, pp. 16–17]. The maximum of these specified single-observation distances occurs at

\[
t_* = \frac{\log(D/a)}{D-a}.
\]

This maximizes \(|H|\) and the stated TV, **not automatically finite-record power, signal-to-noise, or information per unit recording time**. Do not optimize lag using confirmatory outcomes.

### 3.4 Four-cycle: the mandatory independent oracle

Use edges \(01,12,23,30\) and rates \((1,1,1,1)\). In the state order
\(\{0\},\{1\},\{2\},\{3\},\{0,2\},\{1,3\}\), the recurrent generator and stationary law are

\[
Q=\begin{pmatrix}
-3&1&0&1&1&0\\
1&-3&1&0&0&1\\
0&1&-3&1&1&0\\
1&0&1&-3&0&1\\
0&1&0&1&-2&0\\
1&0&1&0&0&-2
\end{pmatrix},\qquad\pi=(1,1,1,1,1,1)/6.
\tag{4}
\]

Do not confuse this order with numerically sorted bitmasks. Verify (4) by direct update enumeration, not by copying it as the sole generator implementation. Opposite vertices have \(a=b=3,c=2,D=4\), so

\[
H(t)=-\tfrac16(e^{-3t}-e^{-4t}),\qquad
\mathcal A_{C_4}(t)=-\tfrac23(e^{-3t}-e^{-4t}).
\]

At \(t_* =\log(4/3)\), \(H=-9/512\), \(\mathcal A=-9/128\), \(\mathrm{TV}_3=9/128\), and \(\mathrm{TV}_4=27/256\). Full labeled snapshot-pair TV against its reverse is \(9/64\). The last value was derived during planning: the antisymmetric stationary joint matrix equals

\[
(e^{-3t}-e^{-4t})[\operatorname{diag}(\pi)Q-Q^\top\operatorname{diag}(\pi)],
\]

which has eight unordered current edges of magnitude \(1/6\), giving TV \(\frac43(e^{-3t}-e^{-4t})\). Confirm this by enumerating the full 6-by-6 joint table and the projected three-/four-bit tables.

The count map is strongly lumpable: every singleton moves to a pair at total rate 1; every pair moves to a singleton at total rate 2. Thus its entire process has generator

\[
Q_W=\begin{pmatrix}-1&1\\2&-2\end{pmatrix},\qquad
\Pr(W=1)=2/3,\quad\Pr(W=2)=1/3.
\tag{5}
\]

It is reversible. Count-only failure here is **absent information**, not an inadequate detector. Do not generalize (5) to other graphs without a lumpability/path-law argument.

### 3.5 Matched reversible refresh control: a derived construction

For a connected balanced graph with common neighborhood rate \(a\), define the synthetic control

\[
Q_0=a(\mathbf1\pi-I),\qquad
P_0(t)=e^{-at}I+(1-e^{-at})\mathbf1\pi.
\tag{6}
\]

Interpretation: at Poisson rate \(a\), replace the whole state with an independent stationary draw. Null refreshes are allowed. It is reversible because \(\pi_i(Q_0)_{ij}=a\pi_i\pi_j\). It has the same full-snapshot stationary law as DSP. Its coordinate covariance is \(\operatorname{Cov}_\pi(X_u,X_v)e^{-at}\), matching (3), hence **all isolated two-bit laws at every lag** match. Each individual coordinate also has exactly the same two-state trajectory law.

For C4, \(Q_0\) has off-diagonals \(1/2\), diagonals \(-5/2\), and the same count generator (5). Therefore every count record has the same distribution under an irreversible DSP system and this reversible system. No count-only method, however sophisticated or data-rich, can distinguish them. This supplies a concrete nonidentifiability certificate. It also makes the null nontrivial: occupancy varies, and overlap scores are not identically zero. A union-of-cliques control alone would miss bad calibration because its stationary total count is constant.

This refresh process is **outside the local DSP update rule**; it is a matched reversible control, not a claimed extension of the sign theorem. On an unbalanced connected graph use refresh rate \(r=\operatorname{median}_v a_v\) only to match \(\pi\) and provide a reversible control; do not claim matching two-bit laws there. All benchmark graphs are connected. No historical-priority claim is made for (6).

## 4. Planning calculations already performed, and their limits

The planning team read the supplied note including its appendices, examined mathematical source blocks and verification programs, checked relevant rendered equations/figure, and conducted three separable audits. Small computations used hosted **Linux x86-64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0**. They are not Mac timings and are not the requested PoC.

| Actually performed | Result / interpretation |
|---|---|
| Reran supplied exact graph checks only through four vertices | 75 labeled graphs, 150 graph/rate cases, 410 nonadjacent-pair checks, 174 nonzero cubic derivatives; all assertions passed |
| Reran supplied arbitrary-rate finite-lag check through four vertices | 2,460 ordered pair/lag checks and 450 aggregate checks; max residuals \(4.17\times10^{-16}\), \(7.78\times10^{-16}\) |
| Independently assembled and solved C4 generator, then checked five lags | Stationary law, pairwise symmetry, H, overlap, TV3, TV4, full TV, and count lumpability agreed |
| Independently checked matched refresh at six lags | Two-bit moments matched to double-precision error; reversibility/count lumpability confirmed |
| Evaluated exact binomial-mixture power and 6-state KL | Numerical values below; probability calculations, not Monte Carlo benchmarking |
| Read official package source and attempted a minimal import | Source/API inspected; import in the default hosted environment failed because `statsmodels` was absent. **No official-package execution or Mac compatibility is claimed.** |

The supplied archive reports broader exhaustive checks through five vertices and regular-graph/reset enumerations. These are **source-reported results**, not all independently reproduced here. Its temporal script checks H and aggregate A but does not explicitly reconstruct the three-/four-bit probability tables; the independent C4 probe filled that limited gap. The target execution must also check an unequal-rate balanced example.

### 4.1 Exact overlap-test power anchor

For an independent C4 pair at \(t_*\), \(g\in\{-1,0,1\}\), with

\[
\Pr(g=-1)=229/2304,\quad \Pr(g=1)=67/2304,
\quad q=\Pr(|g|=1)=37/288,
\quad r=\Pr(g=-1\mid |g|=1)=229/296.
\tag{7}
\]

Under matched refresh, the same \(|g|\) distribution has equiprobable signs. For \(M\) independent pairs, let \(D_g\sim\mathrm{Binom}(M,q)\), and conditional on \(D_g=d\), let \(K\sim\mathrm{Binom}(d,r)\). Reject when the exact two-sided \(\mathrm{Binom}(d,1/2)\) p-value is at most 0.05; set p=1 at d=0. Summing these two binomial laws gives:

| Pairs M | Exact two-sided power, numerically evaluated |
|---:|---:|
| 128 | 0.534499192567 |
| 256 | approximately 0.86702 |
| 512 | 0.994854169175 |
| 2,048 | within numerical precision of 1 |

The reference lag is therefore an informative software-validation case, not a barely detectable mathematical curiosity. The M=128 cell is deliberately less decisive. Compute tails with survival/CDF routines, not subtraction of almost equal tail probabilities. A numerical probability summation does not prove a general sample-complexity theorem for dependent trajectories.

### 4.2 Certificate that late-lag nonrejection is not necessarily a method defect

Let \(F=\operatorname{diag}(\pi)e^{\tau Q}\) and \(F_0=\operatorname{diag}(\pi)e^{\tau Q_0}\) be distributions of complete stationary pairs. For any test of size at most \(\alpha\) at the reversible null \(F_0\), its power on \(M\) independent pairs obeys

\[
\mathrm{power}\le\alpha+\operatorname{TV}(F^{\otimes M},F_0^{\otimes M})
\le\min\{1,\alpha+\sqrt{M D_{\rm KL}(F\Vert F_0)/2}\}.
\tag{8}
\]

This follows from the definition of TV, additivity of KL for products, and Pinsker's inequality. Logs are natural. At \(\tau=2\), a 6-state matrix calculation gave

\[
D_{\rm KL}(F\Vert F_0)=2.0672304562112625\times10^{-5}\ \text{nats},
\qquad \mathcal A=-0.0014288596991759162.
\]

Equation (8) bounds power by approximately 0.08637, 0.12275 and **0.19549** for M=128, 512 and 2,048 respectively. Thus no universally size-controlled reversibility test can achieve the 80% target in this fixed-lag independent-pair design at M≤2,048. Recompute this on the target; use the bound as a model-specific insufficiency certificate. It does not constrain a detector given densely sampled trajectories, intermediate events, or additional channels. It compares to a reversible control, not merely the reverse of the alternative. Do not confuse this testing bound with the equal-prior single-pair direction-classification accuracies in the source.

No pseudocounts enter (8). On the finite recurrent support, \(F_0\) is positive for positive lag. Handle \(0\log0=0\); report materially negative computed KL or probability entries as errors. For numerical reporting of the bound, round upward with a conservative \(10^{-10}\) allowance after the matrix checks pass. Do not characterize this allowance as a formal interval-arithmetic proof.

## 5. Constructive data generation and observation protocol

### 5.1 Representation and validation

Use NumPy binary arrays for readable reference code and unsigned integer bitmasks for compact storage if convenient. Each dataset contains `x0` and `x1`, shape `(M,n)`, or equivalently M start/end masks with a documented vertex order; timestamp and episode/window identifiers; graph/rate provenance in an oracle-side metadata file; and independent RNG seeds. Detectors receive only observations, timestamps, window boundaries, alpha, and their randomization seed. They must not receive G, rates, Q, pi, true direction, analytic H, or oracle labels.

Reject malformed dimensions, nonbinary values, nonfinite/nonpositive rates in the generator, self-loops, duplicate/asymmetric edges, invalid independent initial states, nonincreasing within-window times, mixed identity maps, and missing values. Do not silently impute, sort time in a way that changes direction, deduplicate repeated states, jitter binary values, or normalize channels separately in the two directions. Repeated snapshots and null events are real observations.

Identity and stationarity guarantees come from the controlled generator and declared metadata. Input validation can detect an explicitly inconsistent identity map; it cannot certify that an unknown recording was never silently relabeled or that its law is stationary. Report those as assumptions for external data, not automatically verified properties.

Cast counts and bit values to **signed integers before subtraction**: unsigned underflow in `W0-W1` or in a coordinate difference is a plausible fatal implementation error. Freeze graph vertices 0 through n−1 and report bit ordering.

### 5.2 Exact stationary draw and finite-step sampler

The following sampler is a derived executable consequence of the last-event construction, not a routine already implemented in the supplied archive:

```text
stationary_draw(G, lambda, rng):
    draw independent ages E[v] ~ Exponential(scale=1/lambda[v])
    reject and redraw an age vector containing an exact tie (at most 3 attempts)
    return x[v] = 1 iff E[v] < E[w] for every w in N(v)

advance_dsp(x, delta, G, lambda, rng):
    require delta >= 0 and x is an independent set
    if delta == 0: return a copy of x
    draw independent E[v] ~ Exponential(scale=1/lambda[v])
    reject/redraw exact ties as above; record any occurrence
    copy x to y
    for each v:
        w = unique minimizer of E on N[v]
        if E[w] < delta:
            y[v] = 1 if w == v else 0
        else:
            y[v] = x[v]
    return y, updating all coordinates simultaneously

advance_refresh(x, delta, r, G, lambda, rng):
    with probability exp(-r*delta): return a copy of x
    otherwise: return stationary_draw(G, lambda, rng)
```

For `advance_dsp`, \(E_v<\delta\) means the last event at v in the interval has backward age \(E_v\); \(E_v\ge\delta\) means no event at v. The most recent event in a closed neighborhood determines its coordinate, or the old value persists if no event occurred. Although the draw is exponential, **do not interpret E as first-event time and choose the wrong chronological event**. Poisson increments on disjoint intervals are independent; repeated steps therefore sample the exact transition semigroup, without Euler time discretization. A fresh vector is required at every transition.

Each draw/step costs \(O(n+|E|)\) with adjacency lists and needs \(O(n+|E|)\) memory, independent of the interval length. Global state enumeration is unnecessary. Exact age ties are a floating-point artifact with negligible frequency in these cases; a third failed redraw is a numerical failure, not permission to impose a biased tie-breaking rule. Strict inequalities and this convention must be shared by reference and optimized versions.

Before using this sampler, implement a separate event-by-event reference: total event rate \(\Lambda=\sum_v\lambda_v\); inter-event time exponential with scale \(1/\Lambda\); event vertex categorical with probability \(\lambda_v/\Lambda\); update via \(U_v\); retain the state at the requested observation times. Include null events. Preserve the next scheduled event correctly when crossing an observation boundary, or explicitly restart using the memoryless property. Use the reference only for tiny checks. It must not share the last-event transition implementation.

The sampler's exact distributional guarantee survives ideal real-arithmetic implementation; finite RNG precision remains. Ordinary float64 and moderate positive rates suffice for the chosen cases. Do not substitute a time-stepping approximate process to gain speed.

### 5.3 Primary independent-pair assay

For every dataset and each of its M episodes, draw a new stationary start, then evolve for the fixed lag using the relevant generator. Starts, future increments and datasets are independent. Each episode is a separate length-two path. Record local times `[0, tau]` and an episode ID. Do not concatenate these into a fictitious continuous trace.

Use independent datasets at different M and lags in the locked study; this simplifies the independent unit and avoids accidental reuse. Methods within a dataset share exactly the same pairs. The default primary simulation uses the age sampler, not draws directly from the matrix-exponential joint table. Exact tables remain an independent oracle.

Independent episodes are a deliberate controlled observation design, not a claim that an experimental laboratory can reset a system to its stationary distribution for free. Report both retained pair count and per-episode lag. Their total within-episode duration is M tau; there is no uniquely defined single recording horizon for independent restarts. The sequential bridge below supplies a real single-process horizon.

### 5.4 Mandatory sequential-window bridge

For C4 only, initialize one process exactly at stationarity. Observe `(X(t_i), X(t_i+tau))`, then advance through a gap g to the next window start:

\[
t_i=i(\tau+g),\quad i=0,\ldots,M-1,
\quad T=M\tau+(M-1)g.
\]

Choose \(\epsilon=10^{-4}\) and

\[
g=\frac{\log(n(M-1)/\epsilon)}{a_{\min}}.
\tag{9}
\]

The distribution of all retained pairs differs in TV from the independent product law by at most

\[
(M-1)\sum_v e^{-a_vg}\le\epsilon.
\]

Reason: after a gap, couple the current process to a fresh stationary start using common clocks; only neighborhoods receiving no event can retain a discrepancy. Couple the subsequent window too, then union-bound over joins. Refresh has no-refresh probability \(e^{-rg}\); (9) is conservative for the C4 refresh rate r=3. The bound applies to these specified synthetic generators, not automatically to an arbitrary record.

Run the same tests at cutoff **0.05−epsilon** in this assay, giving a conservative 0.05 size bound after the coupling allowance. A Monte Carlo p-value may land on a discrete grid; use the cutoff as written without rounding it upward. Report gaps, all retained timestamps, T, and the number of unobserved intervals. Do not fabricate dense observations inside the gaps. Generation skips those intervals using the exact transition sampler; this saves computation, not physical observation time. At the reference lag the resulting horizon is thousands of model time units, and must not be hidden when discussing utility.

This bridge measures the effect of using a sequential process under a justified sampling design. It does not validate randomization of adjacent or overlapping pairs from an ordinary dense trace. There is no generic valid shortcut in which every overlapping window receives an independent random sign. Reversing one whole record provides only two orientations.

## 6. Statistical methods and exact calibration

### 6.1 Shared conditional experiment

Under a reversible two-time law, an independent ordered pair is exchangeable with its reversal. Conditional on each unordered pair of distinct states, its orientation is an independent fair coin. This establishes finite-sample conditional tests without estimating a generator or fitting a reversible model.

Map observed complete masks injectively to consecutive categorical IDs, using the union of start/end states within the dataset and one fixed ordering. This preprocessing is invariant to reversing all pairs. Integer symbol distances have no meaning. Never feed arbitrary state codes to a scalar ordinal or metric-based test and call it a fair multichannel comparison.

Let C be the S-by-S transition count matrix of **within-window** start/end states. For each unordered pair `i<j`, set `h_e=C[i,j]+C[j,i]`, `d_e=C[i,j]-C[j,i]`. Keep diagonal counts unchanged. Conditional null tables are generated by

\[
K_e\sim\mathrm{Binom}(h_e,1/2)\quad\text{independently},\qquad
C^{(b)}_{ij}=K_e,\quad C^{(b)}_{ji}=h_e-K_e.
\]

Use B=999 randomizations, in chunks of at most 128 tables, shared by the omnibus methods and the weighted overlap test when needed. For a statistic whose larger values indicate asymmetry,

\[
p=\frac{1+\#\{b:T_b\ge T_{\rm obs}\}}{B+1}.
\tag{10}
\]

Include ties; never report p=0 or increase B selectively when a result is unfavorable. Randomly sampled conditional tables and the observed table are exchangeable under the null, which is the basis of (10) [R10]. This is a finite Monte Carlo test, not an asymptotic chi-square approximation. For floating JSD comparisons count a value within `32*eps_float64*max(1, abs(Tobs), abs(Tb))` below Tobs as a tie, conservatively. Integer-score comparisons can be exact.

Apply the same conservative tie tolerance to floating Bowker values; rational/equivalent integer comparisons on tiny fixtures are an independent check. For exact relabeling-invariance tests, couple null draws by the physical unordered pairs (or reuse explicit per-pair flip bits mapped with the relabeling). Merely reusing a seed after changing the sorted alphabet can reorder RNG draws and need not yield identical Monte Carlo p-values.

Count setup, alphabet creation, randomization and all statistics in method costs. Sharing random tables is an implementation economy, not extra information. Give isolated-method timings too, with the shared setup allocated in full to each isolated method; do not compare only its incremental cost against another method's full cost.

### 6.2 The overlap witness

Compute signed \(\bar g=M^{-1}\sum_i g(x_i,y_i)\), and test \(T_A=|\sum_i g_i|\). Report the signed mean and that the DSP prediction is negative. Use a **two-sided** test for the primary comparison so it does not receive a direction prior unavailable to an omnibus baseline.

If all nonzero \(|g_i|\) have one common magnitude, use the exact two-sided binomial test on positive versus negative signs, p=1 if none are nonzero. This includes every C4 dataset. Otherwise use (10). It can be computed from the shared null count tables, or from bins of equal absolute score; both must agree on tiny enumerations. Do not discard magnitudes and call the result the same weighted-overlap test.

Raw overlap scoring is O(Mn) with arrays, or O(M) machine-word mask operations for this small n; this says nothing by itself about an overall calibrated test's advantage. No graph/rate information or parameter tuning is needed.

### 6.3 Independent complete-state reference: conditional Bowker

Use the established symmetry statistic [R11]

\[
T_B=\sum_{i<j:C_{ij}+C_{ji}>0}
\frac{(C_{ij}-C_{ji})^2}{C_{ij}+C_{ji}},
\tag{11}
\]

with conditional inference from (10). Terms with `h_e=1` are constant under every orientation and may be removed from both statistics; if all terms are constant, return p=1 exactly. No pseudocount or sparse-cell chi-square approximation is permitted. Diagonal entries carry no direct antisymmetry but remain available to JSD.

This is a legitimate strong reference: the fully observed sampled process is Markov, and its stationary two-time matrix is asymmetric whenever this model's overlap is nonzero. A failure at short length can reflect statistical power; zero empirical currents in sparse data are not an oracle of reversibility.

### 6.4 Explicit restricted two-bit analysis

For each of the fixed `n*(n-1)/2` vertex pairs u<v, compute

\[
D_{uv,i}=x_{i,u}y_{i,v}-x_{i,v}y_{i,u}\in\{-1,0,1\}.
\]

Use an exact two-sided sign/binomial p-value for its positive versus negative values, with p=1 if all zero. The dataset-level p-value is `min(1, number_of_vertex_pairs * min(p_uv))`; declare p=1 if n<2. This is Bonferroni at one fixed lag. Do not select only graph edges or high-contrast pairs using oracle information. Each lag in the experiment is a separately specified configuration; no winner over lags is reported as one uncorrected test.

On a balanced DSP graph, (3) gives \(E[D_{uv}]=0\). Since D only takes −1,0,1, its conditional nonzero sign is exactly fair. Independent episodes make each sign test superuniform **even under this irreversible alternative**; Bonferroni preserves level regardless of dependence between vertex pairs. This is a finite-sample blindness certificate for this exact analysis. Its type-I rate under that alternative is not a measurement of the full process's reversibility.

Do not generalize the certificate to joint classifiers, distributions of all D values together, or methods that compare two-channel snapshots. Those use additional information. A positive control with unequal neighborhood totals must demonstrate nonzero population differences; its finite-sample power remains a measured quantity.

### 6.5 Count-only ablation

Apply the same conditional Bowker test to state sizes instead of labeled states, preserving pair boundaries. This changes the input representation and must be labeled an ablation, not an equally informed competitor. For C4 the full count trajectory is reversible under both laws, so a failure to detect hidden irreversibility is correct behavior. Count-only claims on other graphs are limited to the tested two-time law unless a path-law proof is supplied.

When the count alphabet has at most two observed values, use the exact two-sided binomial reduction of (11), with p=1 if no off-diagonal count is observed. Otherwise obtain each count-null table by aggregating the **same shared full-state null table** by state size, not by launching a second B=999 simulation loop. This has exactly the correct marginal count-null law: independent fair binomial orientations on fine edges sum to independent fair binomials with the aggregated unordered count totals; count-diagonal totals stay fixed. The count statistic sees only the aggregated table. This sharing changes neither its information access nor its null law and is required for the stated randomization budget.

Do not create a count-only substitute for overlap by inventing an intersection from counts. The missing identities are precisely the lost information.

### 6.6 Closest feasible external baseline: `irreversibility` detailed balance

The current official source inspected during planning is `irreversibility==1.4.0`, released 15 September 2026, Python >=3.11, required dependencies NumPy/SciPy/statsmodels, optional Numba, AGPL-3.0-only [R4]. Pin this release, retain attribution/license for any vendored material, and record the dependency resolution. The source-distribution SHA-256 is

```text
e3f16ad3713712b17526ab185d0848ba5808c6fbfdace03531bb242ed62b7ffc
```

Use the inspected API, subject to the target smoke check:

```python
from irreversibility.DiscreteMetrics.DetBalance import GetStatistic
# paths: a Python list of M integer arrays, each containing [start_id, end_id]
score = GetStatistic(paths, conditional=False)
```

**Never pass an M-by-2 NumPy array as if the package necessarily interprets rows as paths.** Its wrapper treats an ndarray as one path; provide a list explicitly. Never flatten/concatenate pairs, which adds nonexistent transitions from one episode to the next. Use nonnegative consecutive IDs rather than sparse bitmask magnitudes: storage is sized by `max(symbol)+1`.

The implemented score can be calculated from C. For each state i with both positive incoming total \(c_i\) and outgoing total \(r_i\), normalize its outgoing row and incoming column to probability vectors \(p_i,q_i\), and compute

\[
I_{\rm DB}(C)=\sum_i(r_i+c_i)\operatorname{JSD}(p_i,q_i),
\quad\operatorname{JSD}(p,q)=\tfrac12\operatorname{KL}(p\Vert(p+q)/2)
+\tfrac12\operatorname{KL}(q\Vert(p+q)/2).
\tag{12}
\]

Use natural logs. The actual v1.4.0 source sets a contribution to **zero** when an empty incoming or outgoing row produces NaN; reproduce that convention in the count adapter and report the frequency of such rows. A single path `[0,1]` therefore has score zero, while three paths `[0,1],[1,2],[2,0]` give `6*log(2)`. These were source-derived predictions during planning and must be executed as checks. Do not “fix” the convention inside the baseline or silently switch to the paper's slightly different occurrence-weight wording.

Implement a counter-based/vectorized evaluation of (12) for conditional null tables to avoid rescanning thousands of paths per surrogate. **Required equivalence check:** compare this adapter to the unmodified official `GetStatistic` on at least 20 fixed small count/path fixtures, including zeros, singleton counts, repeated self-pairs, bidirectional edges, directed cycles, absent IDs, random reversals and a C4 sample. Also compare at least 20 conditional null tables represented as lists of pairs. Require agreement within `1e-11*max(1,abs(score))`. This verifies the external statistic rather than a loosely related reimplementation. Time official and adapter scoring separately on representative inputs.

Use (10) to calibrate this score. The native `GetPValue` shuffles symbols within each path and converts a surrogate z-score to a Gaussian tail. For independent length-two paths, its shuffle **is** correct orientation randomization; the Gaussian tail remains approximate. For longer correlated paths, arbitrary shuffling is not a general reversible null. This project evaluates the **official statistic with specified conditional calibration**, not all of the package's native p-values. Do not claim the package fails, or that its native inference has been validated, without actually testing that separate behavior. No native-p-value benchmark is required here.

`conditional=True` allocates triple counts and is unnecessary for the completely observed first-order sampled process. Do not add it as a costly token competitor. Full discrete PhiID, waiting-time inference, or bispectral implementations are also unnecessary for the selected question: the complete-state reference already receives the full two-time information and tests its symmetry. Their exclusion and scope restrictions are explained in §12.

## 7. Experiment matrix and frozen inputs

### 7.1 Graphs and time normalization

Define \(\theta_*=3\log(4/3)\). In robustness cases use \(\tau=\theta_*/\operatorname{median}_v a_v\); this is a fixed scale convention, not a fitted optimal lag. Refresh rate is r equal to that median. Never rescale rates after observing outcomes.

| ID | Graph and exact rates | Purpose / known truth |
|---|---|---|
| `c4_unit` | Edges 01,12,23,30; rates [1,1,1,1] | Primary balanced irreversible case; all individual two-bit laws symmetric; full count path reversible |
| `c5_unit` | Edges i,(i+1) mod 5; all rates 1 | Additional balanced topology; its count projection is also reversible, as checked below |
| `k23_balanced` | All edges between {0,1} and {2,3,4}; rates [2,2,1,1,1] | Nonregular graph with every a_v=5; prevents confusing degree regularity with the true weighted criterion |
| `p3_unit` | Edges 01,12; rates [1,1,1] | Unequal a=[2,3,2]; positive control restoring some second-order information |
| `c4_unbalanced` | C4 edges; rates [1,2,1,2] | Same topology, altered rate balance; a=[5,4,5,4] |

For each graph, generate its DSP process and its reversible stationary refresh control. Balanced matching claims apply only where a is constant. The reversible controls share the same stationary snapshot distribution in every case.

For unit-rate C5, all recurrent sets have size one or two. Every singleton moves to size two at rate 2; every independent pair moves to size one at rate 1. Thus its count generator is `[[-2,2],[1,-1]]`, also matched by rate-3 refresh. This additional count-path certificate was derived and independently checked over its ten recurrent states during draft review; reproduce the lumpability check and include these count cells in the null-alarm family. C5 is not a count-asymmetry positive control.

Add only deterministic/tiny correctness controls: a clique, isolated vertices, and a union of two cliques (reversible DSP); an empty initial state to show the stationarity assumption matters; time reversal; a fixed global permutation of vertex labels; a fixed bijective relabeling of state symbols; a negative/zero-rate rejection; a malformed identity map and missing timestamp rejection. A global vertex permutation must preserve overlap and all omnibus decisions when random draws are matched. A per-time permutation destroys identities and is **not** an invariance test. No broad noise/nonstationarity benchmark is authorized: those would introduce a new inference problem.

### 7.2 Full locked protocol

Run all five methods in §6 on each dataset, except that structural certificates are reported separately from empirical p-values. Every row below tests a stated uncertainty. Primary decisions concern the specified C4 cell, not a best result selected from the matrix.

| Block | Processes / observations | Lags | M | Independent datasets per cell | Total datasets | Question |
|---|---|---|---|---:|---:|---|
| Core | C4 DSP + matched refresh; independent pairs | 0.02, log(4/3), 2 | 128, 512, 2,048 | 200 | 3,600 | Finite length, short/reference/late sampling, fair comparator power |
| Extra primary null | C4 refresh; independent pairs | log(4/3) | 512 | 200 additional, pooled with its original 200 | 200 | Improve gross-calibration precision at primary configuration |
| Robustness | Four nonprimary graphs in §7.1, each DSP + refresh | theta*/median(a) | 2,048 | 100 | 800 | Balance mechanism, heterogeneous rates, and one nearby regular family |
| Sequential bridge | C4 DSP + refresh; windows from one process, epsilon=1e-4 | log(4/3) | 512, 2,048 | 100 | 400 | Verify the valid bridge and its physical-time cost |
| **Total** | | | | | **5,000** | |

No parameter sweeps, graph searches for favorable cases, or extra detector tuning are needed. The analytic choice of the reference lag is shared with all methods; it is not optimized separately for the witness. Other lags are diagnostic stress cases. Do not pool cells into a single unexplained average power score.

### 7.3 Predeclared resource fallback, selected before confirmation

After calibration, choose the full protocol if its conservative forecast fits the remaining stage and global budgets. Otherwise use this **reduced protocol** without weakening the primary question:

| Block | Cells | Datasets |
|---|---|---:|
| Primary | C4 DSP M512/tau* R200; refresh M512/tau* R400 | 600 |
| Late-lag limit | C4 DSP and refresh M2048/tau2 R200 each | 400 |
| Short-lag stress | C4 DSP and refresh M128/tau0.02 R200 each | 400 |
| Mechanism controls | K23-balanced and P3-unit; each DSP + refresh; fixed normalized lag; M2048 R100 | 400 |
| Sequential bridge | C4 DSP and refresh M512/tau* R100 each | 200 |
| **Total** | | **2,000** |

Record the choice and forecast in the frozen manifest. Reduced coverage cannot support the omitted sample curves or graph-generalization claims. If even this protocol does not fit, do not shrink away exact calibration, the primary sample budget, the strong baseline or independent replication. Stop with a measured feasibility result or an analytic diagnostic, as appropriate.

### 7.4 Development versus confirmation

Use `numpy.random.SeedSequence` with fixed, disjoint root namespaces: development `[20261101,0]`, confirmation `[20261101,1]`, repair/reconfirmation `[20261101,2]`. Derive child streams from recorded integer IDs for graph, process, lag, M, dataset and purpose (stationary draw, transition, conditional randomization). Do not use Python's session-randomized `hash()` for seeds. Record the generator/bit-generator and NumPy version.

Development may use at most 200 scored datasets and the correctness draws in §8. Development data support debugging, cost measurement and adapter equivalence, not selecting which theorem instance “works.” Before opening confirmation results, write and hash `protocol_frozen.json` with graph/rate definitions, full or reduced matrix, every method, cutoffs, B, root seed mapping, numerical tolerances, budgets, null-alarm family and output interpretation. Commit that manifest and code to local Git.

No data-driven tuning is allowed in confirmation. An implementation defect can justify one bounded repair and a new protocol version; preserve all invalid runs, rerun affected confirmations with fresh `[20261101,2]` seeds, and count both attempts against the caps. A changed hypothesis, baseline, lag or sampling distribution is exploratory and requires fresh confirmation if retained. Never keep collecting until the power interval crosses 0.80.

## 8. Correctness, metrics and analysis

### 8.1 Tiny mathematical and deterministic checks

These tests address concrete failure modes and precede comparative claims:

1. Build Q independently by reset actions. Check nonnegative off-diagonals, exact rational row sums, stationary support, normalization and `pi @ Q == 0` in rational arithmetic where possible. Check C4 against (4), and count lumpability against (5).
2. Run exact moment/derivative/reversibility checks on all 75 labeled graphs through four vertices, at unit rates and rates `lambda_v=v+1`. If the archive is available, run its corresponding functions to separate output files; also implement an independent direct generator/linear-solve route. Reusing the source algebra alone is not independent verification.
3. Compare generator-exponential evaluations of (1)–(3), aggregate g, and forward/reverse joint tables at lags 0.02, tau*, 2. Add the source's existing lags when rerunning its function. Confirm TV3 and TV4 on C4 and K23-balanced, including a pair on each bipartition where appropriate. An arbitrary K23 opposite-side adjacent pair has no cubic signal; select same-side nonadjacent endpoints explicitly for this oracle check.
4. Verify the refresh generator's detailed balance, equal stationary law, balanced two-bit matching and C4 count/coordinate generators. Verify the late-lag KL bound numerically from independent table construction.
5. Enumerate every conditional orientation on a tiny fixture with at most 12 nonself pairs. The equally likely outcomes are the elementary per-pair coin flips; if distinct count tables are enumerated instead, weight each by its product of binomial multiplicities. Distinct count tables are not generally equiprobable. Check overlap, Bowker and JSD exact conditional p-values; compare Monte Carlo generation distribution, not equality of a single random p-value. Check all-zero/constant-statistic cases return p=1, a fixed directed cycle is detectable with sufficient repeats, and reversing a dataset negates signed scores but preserves two-sided statistics/p-values with coupled randomness. For a reversed length-two episode, swap endpoint values while retaining ascending timestamp slots `[0,tau]`; do not reverse timestamps into an invalid decreasing record. A whole sequential record, when used for a deterministic reversal check, must have its reflected times reindexed into increasing order.
6. Check state-symbol and fixed vertex relabeling invariance, bit ordering, signed arithmetic, list-of-paths boundaries, no cross-episode transitions, and official-adapter equivalence from §6.6. A proof of the mathematics does not excuse a faulty baseline adapter.

For float64 tiny matrices use initial tolerances `1e-12` absolute and `1e-10` relative, with relative checks only where the reference magnitude exceeds `1e-8`. Check stochastic row sums and stationary residuals to `1e-12`. Clip negative probabilities only if their absolute value is at most `1e-14`, log the total corrected mass, and renormalize only at that scale; otherwise fail. If platform differences breach a tolerance, diagnose conditioning/implementation and use an independent high-precision calculation on the failing tiny matrix. Do not relax thresholds beyond `1e-10` absolute without documenting why the scientific decision remains separated from error. Every threshold is many orders below the core signal; the late-lag KL comparison must remain stable well below `1e-8` nats.

### 8.2 Sampler validation, separate from comparative results

Across a fixed selection of C4, K23-balanced, P3 and clique states, compare stationary/transition frequencies from the fast sampler to exact pi/P and the event reference. Include the empty independent state and an occupied state for transient-transition checks. Use at most 200,000 independent transition draws total, allocated before inspecting counts, at the reference lag and one long gap. Compare all tested cell probabilities using the conservative diagnostic envelope

\[
|\widehat p-p|\le 7\sqrt{p(1-p)/N}+7/N.
\]

This is a practical bug alarm, not a formal distributional equality proof. Report N, cells and the observed maximum standardized discrepancy. Exact support violations, inconsistent independent-set invariants, or failures on deterministic limits are immediate defects regardless of this envelope. If a stochastic alarm occurs, inspect the offending transition and use one fresh bounded diagnostic sample; do not rerun repeatedly until passing. The construction proof, exact matrices and a distinct event algorithm together provide correctness evidence. Matching stationary means alone does not validate temporal dependence.

All these draws, including failed checks, count against the global generated-pair/transition budget. The fast sampler is optional acceleration only after this gate; a correct event reference may run the reduced experiment if the measured resource budget permits.

### 8.3 Measurements to retain

For every dataset/method retain:

- Configuration and data hashes, dataset ID, full RNG seed lineage, method/adapter version, status and failure reason.
- M, n, lag, window design, gap/epsilon/horizon when applicable, retained channels, observed alphabet size, and within-window transition counts.
- Signed mean overlap, exact analytic mean where available in the oracle file, method statistic, p-value, cutoff, decision, B and number of zero/one-count edges; JSD empty-row count.
- Scoring/preprocessing/randomization times, detector total time, generation/oracle setup time, cold import/startup separately, and peak process/tree RSS.
- Oracle fields from §1.2, exact/proved quantities and numerical residuals; never mix these into detector inputs.

Save compact raw pairs and timestamps in compressed NPZ per configuration or shard, plus JSONL result rows and small CSV summaries. Store count matrices as well, so inference can be recomputed without resimulating. Do not store every randomization table; seed/configuration suffices, with a tiny archived fixture for algorithm verification. Save errors and incomplete-run records before exiting.

### 8.4 Statistical interpretation

The independent evaluation unit is **one generated dataset**, not each time point or each surrogate. For a cell with k rejections out of R completed valid datasets, report k/R with a two-sided 95% Wilson interval. Report orientation accuracy only when a direction is defined; a two-sided rejection with the wrong overlap sign is a separate error. A Bernoulli power curve and a signed mean curve answer different questions.

Before confirmation enumerate the null-alarm family: all methods on refresh controls, plus the mathematically null count-only C4/C5 configurations and balanced restricted two-bit configurations under DSP. Let its size be L. For each member, calculate the one-sided binomial tail for at least k rejections under success probability 0.05. An alarm is `tail < 0.01/L`. This is a conservative familywise screen for size inflation. Report all null intervals even without alarms. A rare alarm triggers bounded diagnosis; it is not automatically proof of a package bug or a scientific refutation. The extra primary-null datasets augment that cell rather than create a separate cherry-picked test.

Do not use naive t-tests on dependent time points or count conditional randomizations as independent replications. There is no primary superiority test. If you nevertheless report a difference between methods, use paired dataset-level binary outcomes and a paired bootstrap or an exact paired test with a prespecified contrast, and label any unplanned contrast exploratory. Confidence intervals on individual cells are pointwise; do not imply simultaneous coverage of the whole grid or a globally optimized best lag.

Compare observed C4 overlap power against (7) and compare late-lag rates against (8), allowing the sampling uncertainty of R. Do not read a noisy finite empirical rate above a theoretical power bound as a contradiction without quantifying that uncertainty. A well-calibrated detector may occasionally reject a low-power case. A particular nonrejection on a high-power case is also possible.

### 8.5 Failures, exclusions and runtime accounting

Never remove an unfavorable valid run. Record timeouts, numerical failures, nonfinite scores, violated invariants, missing dependencies and interrupted datasets as explicit statuses. No convergence iteration is expected; if a solver does not deliver a valid tiny stationary distribution, that is a failed oracle, not a zero signal.

For power tables show: valid-run power with its denominator; failure rate; and operational detection fraction over **all planned datasets**, counting failed/uncompleted runs as “no delivered detection” for that operational metric only. On nulls, failures are not false alarms or evidence of correct calibration: show a possible false-alarm range from observed valid rejections/R_planned to `(rejections+failures)/R_planned`. A scientifically decisive primary claim requires no unresolved primary-cell validity failures. Missing optional breadth reduces scope; missing a primary baseline blocks that baseline's conclusion.

Timing uses `time.perf_counter_ns()` in isolated serial runs, with CPU/backend/thread settings recorded. Measure one cold process import/startup per implementation and at least three warm repeated scorings of the same fixed dataset for calibration; include all repeats in budget accounting. Do not regenerate data during detector-only timing. End-to-end timing includes generation, validation, encoding, scoring, conditional calibration and writes. Report total setup and package installation separately. GPU synchronization is inapplicable: the required backend is CPU float64. If a library unexpectedly dispatches elsewhere, record and disable it before comparison.

## 9. Target-machine feasibility and cumulative budgets

### 9.1 Inspect before estimating

Do not infer RAM, free disk, macOS release, installed Python, available instruction set or numerical backend from “M3 Max.” Record `sw_vers`, `uname -m`, `sysctl -n machdep.cpu.brand_string`, `sysctl -n hw.memsize`, CPU count, free project-volume space, Python executable/version/architecture, NumPy/SciPy versions, `numpy.show_config()`, and thread-pool backend information. Check the actual process is native arm64 rather than unintentionally running under Rosetta. Record whether on battery or external power and any obvious concurrent workload that affects timing; do not change system power policies.

Use CPU float64 NumPy/SciPy, standard Python integers/Fraction for tiny rational checks, Matplotlib for static figures, pytest for substantive correctness gates, psutil for process-tree RSS, and threadpoolctl to cap BLAS/OpenMP pools at one thread during comparable timings. Prefer a clean virtual environment without Numba: these tiny matrices do not justify compilation complexity. If Numba is present and the package activates it, record actual dispatch and compilation cost; either remove it from the isolated environment or explicitly compare warmed paths with cold compilation separately. Do not depend on CUDA, MPS, PyTorch, JAX or GPU float64.

Preferred starting versions are Python 3.12, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and `irreversibility==1.4.0`. The first three scientific versions were used in planning checks; **Mac wheels and the full package import have not been verified here**. The package pulls statsmodels and its transitive dependencies. Resolve pytest/psutil/threadpoolctl and transitive packages once, then freeze every version and wheel/hash/source identifier. A compatible existing native Python 3.11–3.13 is acceptable after the same checks. Do not silently upgrade the baseline version during the study.

Allow 45 minutes elapsed for environment setup/downloads and one documented compatibility repair. Prefer binary wheels; do not launch an unbounded local scientific-stack source build. If the preferred numerical pins are unavailable, select one compatible NumPy/SciPy pair retaining the specified scalar APIs, record the deviation, and rerun the numerical gates. This is an engineering allowance, not permission to change algorithms, precision, test cutoffs or statistical assumptions. If the external package cannot be imported or its statistic cannot be reproduced within this allowance, complete the mathematical/reference work and report the external comparison blocked; do not substitute an invented package result.

### 9.2 Hard resource ceilings

All limits below are **cumulative across development, failed runs, tuning, repairs, reruns and fallback branches**. Stage changes do not reset them. Setup/download time is tracked separately; model-hosted reasoning and web research are not PoC experimental compute.

| Resource | Ceiling and enforcement |
|---|---|
| Local experimental/verification/report computation | **240 minutes total elapsed job time**, including failed jobs and timing repeats; serial execution by default |
| Per dataset plus all five tests | 10 seconds hard timeout after calibration. If a required dataset cannot fit, use a bounded implementation repair or report that configuration infeasible; merely dropping other cells does not cure a per-dataset timeout. |
| Scored benchmark datasets/fixtures | 5,500 total; up to 5,000 locked datasets, 200 development/calibration/pilot, 200 replacement/repair datasets, remainder for small adapter fixtures |
| Generated pair/transition samples | 10 million total, including correctness checks and failures; selected full matrix uses about 5.48 million retained pairs before development |
| Conditional null tables | 6 million total across all uses; share a table draw across methods where valid, B=999 per dataset, chunks ≤128 |
| Peak aggregate project process-tree RSS | `min(4 GiB, 20% of measured physical RAM)`; warn at 80%, stop the next allocation at the ceiling |
| Project disk including environment, source downloads and outputs | 4 GiB; raw experiment outputs ≤512 MiB; do not proceed with writes that leave less than 2 GiB free on the volume |
| Tiny enumerated oracle | n≤8 and S≤128 recurrent states; full experiments n≤5; no state-space expansion to make a scaling claim |
| Repairs | At most two diagnosed repair episodes, within a shared **30-minute compute reserve** and at most 2 hours of implementation/debugging effort; at most one changes frozen scientific output logic |

For sample accounting, one stationary start plus its within-pair transition counts as **one generated pair**, not two samples; each extra bridge gap transition counts once separately. Each independently generated validation endpoint transition counts once, including event-reference samples in addition to fast-sampler samples. Underlying exponential draws and internal Poisson events are not additional endpoint samples; track their event count separately as a cost diagnostic. A repeated measurement on existing data consumes elapsed time and any newly generated null tables but creates no new data sample. The full matrix has 5,478,400 retained pairs plus 511,600 bridge gap transitions, before development/repair checks. Exact rational graph calculations are jobs, not scored data replicates.

If the machine is busy or available memory is lower than required, measure and resize chunks first. Do not infer a machine limitation from a preventable `B*M*n` intermediate. For S≤10 in the main graph set, a batch of 128 S-by-S float64 tables is about 0.1 MB before auxiliary arrays. Raw arrays are O(Mn); count tables O(S²). The principal statistical workload is millions of tiny random count tables and JSD evaluations, not matrix exponentiation or simulation of every event.

Rough cost models, to validate empirically:

- Stationary pairs/finite-step sampling: O(M(n+|E|)); event-reference cost additionally scales with total event intensity times lag and is confined to tiny checks.
- Overlap scoring: O(Mn) reference, with scalar/binomial or weighted-randomization cost added.
- Restricted two-bit analysis: O(Mn²), small for n≤5.
- Full-state and JSD conditional calibration: O(M + B S²), with chunked O(S²) storage. A naive package re-scan for every surrogate is avoidable only after exact adapter equivalence is demonstrated.
- Rational stationary enumeration and dense matrix exponential: exponential state enumeration then roughly O(S³) dense numerical work; restricted to small oracles and cached per graph/lag. Cache setup time is still counted and reported.

These are complexity statements and **unmeasured runtime estimates**, not M3 benchmarks. Expected experimental time is minutes for the informative pilot and tens of minutes to a few hours for the comparison if the adapter is efficient. The hard budgets, not optimism, govern execution.

### 9.3 Calibration and resizing rule

Before allocating the locked matrix, measure complete dataset costs for C4 and the largest selected alphabet (C5 or K23), M128 and M2048, every method, B=999, all encoding/I/O included. Use at least three warmed repetitions and one cold startup. Confirm that observed work scales plausibly with M and B and that allocations match the cost model. Include a tiny S=46 C8 table/scoring fixture only if needed to expose a demonstrated alphabet-allocation problem; it is a cost diagnostic, not an additional power benchmark.

Forecast each remaining cell with **twice the maximum measured representative per-dataset cost**, plus measured fixed setup and at least 20 minutes reserved for final analysis/reporting. Choose the full or reduced matrix in §7 before confirmation. If the forecast breaches the locked-stage 130-minute allocation or the global 240-minute remainder, choose reduced; if reduced still breaches, stop empirical expansion. During confirmation, checkpoint at most every 50 datasets and recompute cost forecasts. Finish already valid data, preserve evidence, and stop starting new blocks when the reserve would be consumed. Do not quietly lower B, R, M, numerical precision or baseline coverage mid-study.

If resource costs make only an irrelevant tiny version possible, report practical infeasibility for the intended protocol. The supplied 6-state analytic certificate remains a useful diagnostic outcome, but it is not evidence that the external software comparison ran.

### 9.4 End-to-end effort estimate

Planning estimate, not measured agent performance: roughly 6–10 hours to implement/review the small repository, up to 45 minutes to establish dependencies, up to 4 hours of cumulative local computation, and 1–2 hours to interpret and package results, with some overlap. Allow approximately **8–18 hours end to end**. A clean analytic stop or early validity failure can finish much earlier. This estimate separates implementation/analysis effort from the laptop workload; the computational study is not authorized to run for days.

## 10. Evidence gates and autonomous decisions

Run stages in order. Do not build later-stage infrastructure before its predecessor's evidence is available. Log `continue`, `bounded_repair`, `reduced_protocol`, `blocked`, or `stop` with evidence. A stopped study is complete when its supported diagnostic and reproducible evidence are delivered.

| Stage / compute cap | Question and prerequisites | Minimum work and evidence | Decision and next action |
|---|---|---|---|
| **0. Constructive feasibility — 10 min** | Are mathematics, inputs, target access and baseline constructible? No prerequisite | Environment manifest; hashes; proof/assumption checklist; official import/API smoke; independent C4 Q and refresh construction; resource ledger | Continue if no essential obstruction. One bounded source/dependency repair allowed. If Mac unavailable, complete read-only/code/analytic work elsewhere but mark M3 validation blocked; do not claim performance. A verified theorem obstruction stops empirical expansion. |
| **1. Tiny correctness — 20 min** | Do generator, sampler, oracle and methods agree with trustworthy cases? Stage 0 | §8 checks; source subset rerun where available; exact C4 masses/TV/KL; K23 TV; sampler/event comparison; official JSD adapter checks; tests log | Continue only after discrepancies are resolved. Use repair reserve for a localized defect. No valid generator or baseline means invalid/unexecuted experiment, not evidence against the research hypothesis. |
| **2. Resource calibration — 10 min** | Does a meaningful study fit? Stage 1 | Cold/warm complete-run costs; RSS/disk forecasts; version/backend confirmation; full/reduced choice | Select full if forecast fits, otherwise reduced; final freeze follows the pilot. If even reduced exceeds caps, stop as practical infeasibility or retain only the analytic diagnostic. Do not weaken statistical validity to fit. |
| **3. Mechanism pilot — 20 min** | Does the implemented assay behave plausibly before confirmation? Stage 2 | At most 40 development datasets spanning primary null/alternative, M128/M512, late lag, K23 and P3; inspect counts, signs, p-values and cost | Do not use a noisy small pilot to decide the 80% research target. Gross conflict with exact C4 laws triggers targeted repair. If only sampling uncertainty remains and budget fits, continue to locked R. No tuning to favorable pilot outcomes. |
| **4. Locked comparison — 130 min** | Does current software meet the fixed useful-power target and distinguish the controls within scope? Stages 1–3 and frozen manifest | Full/reduced matrix; all five methods; results/failure records; exact calibration; sequential bridge; no hidden exclusions | Order: primary cell including all null replicates; late-lag M2048 limit; sequential bridge; K23/P3 mechanism controls; remaining core/breadth. Follow budget checkpoints. A detected defect invalidates affected runs; fresh replacement is allowed only within the fixed repair caps. Otherwise preserve and report incomplete/invalid evidence. |
| **5. Diagnosis and report — 20 min** | What conclusion is justified? All reached-stage artifacts | Regenerated tables/plots; oracle-vs-empirical comparison; null alarms, uncertainty, failure accounting, scope and status | Deliver one of §11's outcomes with reproducible evidence. No extra benchmark merely because a result is disappointing. |
| **Shared repair reserve — 30 min** | A concrete diagnosed discrepancy, not a search for a positive finding | Root cause, minimal changed code/config, preserved failed artifacts, affected check rerun | At most two repair episodes; fresh confirmation if output logic changed. Exhaustion ends repair, preserving an honest invalid/inconclusive outcome. |
| **Total — 240 min** | | | |

Freeze the protocol after calibration/pilot, before any confirmation data are inspected. The staging table's freeze requirement is a transition into Stage 4; small engineering fixes made during Stage 3 are still development changes and must be captured in the final frozen code.

The exact C4 power result means a repeatedly failing witness at M512 is more likely a bug or design mismatch than a discovery that the theorem has no finite-sample value. Conversely, the long-lag upper bound means weak performance there is expected even for a sound strong baseline. A positive result on a deliberately nonstationary or identity-corrupted input, if examined during debugging, is not automatically a false positive for time asymmetry: its data law may actually violate the reversible null. Such inputs cannot validate the stationary reset mechanism.

## 11. Repository, commands and required outcomes

### 11.1 Minimal repository to create

All names and commands below are **interfaces you must implement**, not commands claimed to have been executed during planning. Keep the project ordinary and local.

```text
time_arrow_challenge/
  README.md
  pyproject.toml
  requirements.in
  requirements.lock.txt
  config.json
  protocol_frozen.json
  inputs/                         # unchanged user sources, if received
  src/time_arrow_challenge/
    __init__.py
    cli.py                        # subcommands, ledger, status/checkpoints
    models.py                     # graph fixtures, validators, update actions
    simulate.py                   # ages, interval transitions, event reference
    exact.py                      # tiny Q/pi/P, moments, lumpability, TVs, KL
    tests_of_reversal.py           # overlap, Bowker, two-bit, count, calibration
    external.py                    # official API smoke and verified JSD adapter
    experiment.py                  # fixed matrices, seeds, paired data access
    report.py                      # tables, figures, outcome classification
  tests/
    test_exact.py
    test_simulation.py
    test_inference.py
    test_external_adapter.py
  results/
    environment.json
    budget_ledger.jsonl
    protocol_versions/
    correctness/
    raw/
    metrics.jsonl
    failures.jsonl
    tables/
    figures/
    REPORT.md
```

No database or workflow service. JSON configuration is sufficient. Use a local Git repository for code/protocol history; no external publication is required. Keep generated results out of routine source-control diffs if bulky, but include hashes and a release archive/manifest. Do not overwrite invalid or prior result versions.

Minimum module interfaces (signature notation, to implement as actual Python functions):

```text
stationary_draw(graph, rates, rng) -> state
advance_dsp(state, delta, graph, rates, rng) -> state
generate_pairs(config, seed) -> observations, metadata
generate_sequential_windows(config, seed) -> observations, metadata
exact_joint(graph, rates, tau, process) -> states, pi, Q, F
score_dataset(observations, alpha, B, seed) -> list[MethodResult]
run_matrix(config, protocol_hash, output_dir) -> run_manifest
build_report(results_dir) -> report_paths
```

Define `MethodResult` as a simple dictionary/dataclass with the fields in §8.3, including an explicit status. Separate `oracle_metadata` from `observations` in API design; a detector should be callable with no graph object. Validate graph assumptions in the generator, not by pretending they can be inferred from arbitrary observed data.

Essential default configuration values:

```json
{
  "alpha": 0.05,
  "randomization_replicates": 999,
  "randomization_chunk": 128,
  "primary_graph": "c4_unit",
  "primary_lag": "log(4/3)",
  "primary_pairs": 512,
  "power_target": 0.8,
  "core_lags": [0.02, "log(4/3)", 2.0],
  "core_pairs": [128, 512, 2048],
  "core_replicates": 200,
  "extra_primary_null_replicates": 200,
  "robustness_pairs": 2048,
  "robustness_replicates": 100,
  "sequential_pairs": [512, 2048],
  "sequential_replicates": 100,
  "sequential_tv_allowance": 0.0001,
  "compute_budget_seconds": 14400,
  "locked_stage_budget_seconds": 7800,
  "dataset_timeout_seconds": 10,
  "max_scored_datasets": 5500,
  "max_transition_draws": 10000000,
  "max_null_tables": 6000000
}
```

Resolve named analytic expressions through a fixed parser or explicit constants in code, not arbitrary `eval`. The configuration also needs the exact graph definitions, root-seed namespaces, full/reduced matrix, disk/RSS limits and method versions; populate these explicitly from this brief before freezing.

### 11.2 Intended command sequence

Assuming a suitable native Python 3.12 is present, the ordinary environment commands are:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements.in
python -m pip install -e . --no-deps
python -m pip freeze > requirements.lock.txt
```

Create `requirements.in` from §9.1; baseline version stays pinned. Record resolved wheel hashes using pip's download/report facilities or a small lock manifest. `pip freeze` records versions but by itself does not record hashes. If the preferred Python is absent, use the compatible interpreter established by the environment gate and record its actual command; do not claim the sample commands were already verified on the Mac.

Implement these project commands and make them print concise status plus artifact paths:

```bash
python -m time_arrow_challenge.cli inspect --out results/environment.json
python -m time_arrow_challenge.cli verify --config config.json --out results/correctness
python -m pytest -q
python -m time_arrow_challenge.cli smoke --config config.json --namespace development
python -m time_arrow_challenge.cli calibrate --config config.json
python -m time_arrow_challenge.cli pilot --config config.json --namespace development
python -m time_arrow_challenge.cli freeze --config config.json --protocol full
python -m time_arrow_challenge.cli run --protocol protocol_frozen.json --resume
python -m time_arrow_challenge.cli report --results results
python -m time_arrow_challenge.cli reproduce-essentials --results results
```

Use `--protocol reduced` when the calibration decision requires it. `freeze` must refuse if required correctness gates are unresolved. `run --resume` must verify hashes, preserve prior statuses and skip already completed matching datasets; it must not treat a changed protocol as a continuation under the old ID. `reproduce-essentials` should regenerate analytic constants and tables/plots from saved raw data/counts without rerunning the whole benchmark. All these operations consume the same budget ledger; verification and pytest must not accidentally repeat a costly stochastic suite twice. Mark the stochastic sampler check as a cached, hashed gate that the second entry point verifies rather than reruns unnecessarily.

Use append-only JSONL ledger entries with stage, job ID, start/end times, elapsed seconds, input sizes, generated samples/null tables, bytes written and outcome. A parent subprocess watchdog can enforce per-dataset timeouts and sample RSS at a modest interval. Do not build a generic job framework. Flush data and failure records on timeout, keyboard interruption and budget exit where possible.

### 11.3 Final artifacts for the stages actually reached

Deliver runnable code and concise usage instructions if implementation occurred; source/license/version manifest; configuration/protocol hashes and seeds; environment/backend/precision data; correctness logs; raw observations/counts and all measurement/failure records for executed experiments; essential CSV tables and static PNG/PDF plots; and `results/REPORT.md`. Include one compact archive if that is the practical way to transfer the repository and results. Do not claim an unexecuted Mac benchmark in the report or README.

Required report content:

1. **Decision:** supported contribution, informative negative, practical infeasibility, inconclusive, invalid experiment, or a clearly explained combination for separate claims.
2. **What was actually run:** target hardware, protocol version, stages reached, exact full/reduced matrix, deviations and cumulative costs; distinguish preexisting supplied results and planning calculations from new execution.
3. **Validity evidence:** independent oracle checks, sampler checks, official adapter equivalence, calibration derivation, null results/alarms, numerical tolerances and failed cases.
4. **Primary answer:** each information-preserving method's power/interval at C4 M512/tau*, and the external method's explicit status; explain whether a full-state configuration is useful at this operating point.
5. **Why negative results differ:** matched reversible explanation for counts/single channels, certified restricted two-bit blindness, exact late-lag information bound and measured sampling effects. Keep these oracle conclusions separate from detector outputs.
6. **Practical implications:** the exact configuration a maintainer can use as a regression fixture; cases it should detect; cases where a negative result is appropriate; coverage limits and deployment information requirements.
7. **Limitations and one next step:** no global novelty/superiority claim, no real-system/AI-memory/thermodynamic extrapolation, dependence and acquisition costs; identify at most one next experiment worth its cost if the evidence supports it.

Essential figures, only when empirical stages run: rejection probability versus M by C4 lag with pointwise intervals, null calibration with uncertainty, and a compact comparison of full-state/witness/two-bit/count outcomes. Mark the analytic late-lag upper bound and exact overlap-power curve distinctly from measured data. Include a small resource table with cold/warm and total costs. Do not create decorative plots or make a numerical reproduction look like a new experiment.

For an analytic stop, deliver the verifiable derivation/counterexample, a minimal reproducer if computation was used, assumptions and scoped conclusion. A benchmark, dashboard or synthetic result table is not required to fill an inapplicable section.

### 11.4 Honest outcome rules

| Outcome | Evidence required | Interpretation and action |
|---|---|---|
| **Supported contribution** | Correct challenge/oracles; valid calibrated inference; strong baseline actually evaluated; specified operating-point claim meets its CI target or a concrete validated software limitation is established | State precisely which configuration is reliable or blind on the tested family. Standard-tool success is allowed. Claim an original method only if separate novelty evidence supports it. |
| **Informative negative result** | Valid implementation and adequate evaluation place an upper CI below a claimed useful-power target, or a verified mathematical obstruction rules out the scoped aim | Name the method, observation design and budget. The long-lag bound is a scoped negative about achievable power, not a failure of the theorem. If the external method passes and overlap adds no advantage, report that straightforward existing analysis suffices. |
| **Practical infeasibility** | Measured target costs or supported scaling prevent a meaningful protocol within the caps | State which resource/step blocks the useful study; preserve a smaller diagnostic without upgrading it to the intended claim. |
| **Inconclusive evidence** | CI straddles target, insufficient valid replication, inaccessible external comparison/target hardware, or budget exhaustion before the necessary evidence | Report what is known and missing; do not label lack of improvement or a timeout as a refutation. |
| **Invalid experiment** | Generator, baseline, observations, null calibration or measurement defect invalidates the comparison | Preserve failed results, identify affected claims, use bounded repair if available; otherwise deliver the defect and no unsupported power conclusion. |

An unknown-system nonrejection is never assigned a hidden-world truth label. A theorem being true does not guarantee a software implementation is correct; a failing implementation does not falsify the theorem. A finite synthetic study cannot establish universal superiority or applicability.

## 12. Research rationale, closest prior work and coverage

### 12.1 Why this project, rather than another detector

The source has constructive stationary sampling, a closed local moment system and a tiny exact counterexample to overinterpreting simple observations. Those features make it suitable for a reliability challenge with decisive oracles. They do not establish a new broadly useful detector. The research question is deliberately a configuration-validation question; the narrow contribution can survive an existing method performing as well as or better than the witness.

| Prior work | Material implication for this study |
|---|---|
| Dandekar–Krapivsky, *Dynamic Space Packing* [R2], especially Appendix D, Eqs. D1–D5 | Generator, equal-rate local-maxima stationary representation and moment recursion are established foundations. The graph mapping is stationary, not equality with every other model having the same static law. |
| Zanin, *Assessing discrete time series irreversibility through detailed balance analysis* [R3], Sec. 2/Eqs. 1–3, official implementation [R4] | Closest direct competitor already supports collections of symbolic paths. Complete labeled states fit its input directly. Omitting it would make the comparison artificially weak. The v1.4.0 source defines the exact baseline semantics; our calibration is explicitly an adapter. |
| Luppi et al., *Information decomposition reveals hidden high-order contributions to temporal irreversibility* [R5], main derivation and Appendix | Higher-order irreversibility is established. Their worked Gaussian estimator is unsuitable as a fair default for non-Gaussian binary states because covariance can discard the target signal. A full discrete PhiID implementation would require a declared redundancy measure; this study does not make claims against it. |
| Lynn et al., *Emergence of local irreversibility in complex interacting systems* [R6], Secs. II–III and Appendix A | General complete-state transition irreversibility is relevant. Its independent/interaction decomposition requires multipartite dynamics with one component changing at a time; neighbor erasure can violate that condition. Do not transplant that decomposition or conflate its interaction order with two individual binary samples. |
| Martínez et al., *Inferring broken detailed balance in the absence of observable currents* [R7], Results Eqs. 1–7 and Methods | Waiting-time information can expose hidden dynamics despite zero currents in a coarse projection. Therefore pair-current symmetry alone does not prove observed path reversibility. C4 is special: its count process is proved a reversible Markov chain, so no waiting-time method can recover its hidden arrow from counts alone. |
| Zanin–Papo's 2025 comparison [R8], methods and discussion | Performance depends on system and sampling/coarse-graining scale. It motivates explicit operating points, not a claim that a named package fails our challenge. Its software snapshot predates the current discrete detailed-balance method. |
| Hinich–Rothman [R9], Sec. 2, and Gaspard [R12], Secs. II–IV | Cubic/bispectral methods and Markov transition reversal are prior art. A univariate bispectrum of one binary channel would receive a provably reversible projection; failing there is not a fair defeat of an equally informed competitor. |
| Cheng–Bao–Hou, 2026 preprint [R13], main Setup/Eqs. 3 and 8–13 | Recent work also organizes multi-time irreversibility and distinguishes temporal/observation information losses. This further limits broad conceptual novelty. A physical EPR hierarchy is not needed to test finite-pair detection, and no EPR superiority claim is made. |

Complete-state Bowker and JSD are enough to address the selected Markov observation design. Nonparametric whole-trajectory classifiers, neural direction classifiers, discrete PhiID, and semi-Markov waiting-time estimators would broaden the comparison without resolving a missing primary uncertainty. Their omission means there is no ranking of all irreversibility methods or all forms of hidden observation. The source's local four-bit TV remains a check that two-channel snapshot methods can be strong, even though a separate four-bit empirical baseline is not necessary.

### 12.2 Search and access record

Searches were conducted on **1 November 2026** using two search engines, prioritizing primary full papers, official PyPI/source releases and technical documentation. Topics included the supplied named papers, dynamic space packing/time reversal, local maxima and graph overwrite processes, common-shock binary Markov/reliability terminology, cubic/bispectral tests, complete-state symmetry, hidden semi-Markov inference, and current discrete irreversibility software. Backward inspection reached the dynamic-space-packing foundations and older third-order/Markov literature; forward/current inspection identified the 2026 detailed-balance method and September 2026 hierarchy preprint. The predecessor *Birds on a Wire* was checked at metadata/abstract level only; no conclusions here require a full audit of it.

Relevant source content was read, not only search snippets. The supplied 18-page note and its relevant proof/code/result material were inspected. The author-hosted/institute PDF of Zanin's 2026 paper and the official v1.4.0 package source were downloaded and read. Direct publisher/GitLab/PMC access was intermittent: some requests returned 403/CAPTCHA/empty wiki content, and accessible primary PDFs, indexed full text or the official PyPI source were used instead. The package's actual target runtime remains an early execution gate.

No exhaustive global priority search was completed. An equivalent local calculation in common-shock/reliability theory, a thesis or an unpublished note remains possible. No claim that the source's theorem is historically new is licensed by failure to locate an equivalent in this bounded search. The full Bowker 1948 article was not accessible; bibliographic/abstract information was verified, and the conditional calibration used here is derived explicitly rather than inferred from an inaccessible proof. References distinguish published work, preprints, official code and supplied material.

### 12.3 References

Critical mathematics is cited to source sections/pages above. URLs are stable landing pages or primary copies where available. Access date for live sources: 1 November 2026. An attachment reference is a filename/location, not an invented public URL.

**[R1] Supplied research note, author not identified.** *An exact test for irreversible local revision*. 1 November 2026, 18 pages. Attachment `revisable_memory_research.pdf`; duplicate in `revisable_memory_research.zip/output/pdf/`. Full editable text: archive `research/deliverable/report_blocks.json`; temporal checks: `research/learning/verify_time_arrow.py`. Theorem 1 p. 3, proof pp. 4–7; Theorem 2 pp. 7–8; three-bit result pp. 8–9; C4 p. 9; four-bit appendix pp. 16–17. Unpublished supplied material; no verified historical-priority status.

**[R2] Rahul Dandekar and P. L. Krapivsky.** *Dynamic Space Packing*. Journal of Statistical Mechanics: Theory and Experiment, 103403 (2023). DOI: [10.1088/1742-5468/ad0223](https://doi.org/10.1088/1742-5468/ad0223). [arXiv:2306.14358](https://arxiv.org/abs/2306.14358), [primary PDF](https://arxiv.org/pdf/2306.14358). Graph/local-maxima correspondence and moment recursion: Appendix D, PDF pp. 19–20, Eqs. D1–D5. Equal-rate foundation; do not attribute arbitrary-rate temporal classification to this source.

**[R3] Massimiliano Zanin.** *Assessing discrete time series irreversibility through detailed balance analysis*. Chaos, Solitons & Fractals 203, 117717 (February 2026; available online 1 December 2025). DOI: [10.1016/j.chaos.2025.117717](https://doi.org/10.1016/j.chaos.2025.117717). [Author-institute landing page](https://ifisc.uib-csic.es/en/publications/assessing-discrete-time-series-irreversibility-thr/); [primary PDF](https://ifisc.uib-csic.es/media/publications/publication/q2RRAPLkQK235d40JIlyrw.pdf). Sec. 2, Eqs. 1–3: weighted JSD and shuffled z-score; Sec. 3: conditional version. Full source read; runtime implementation details additionally checked in [R4].

**[R4] Massimiliano Zanin.** *Irreversibility Tests Library*, `irreversibility` release **1.4.0**, 15 September 2026. [Versioned PyPI page](https://pypi.org/project/irreversibility/1.4.0/); [official GitLab repository](https://gitlab.com/MZanin/irreversibilitytestslibrary). Inspected source files: `pyproject.toml`, `irreversibility/DiscreteMetrics/DetBalance.py`, `optionalNJIT.py`, and `tests/test_DetBalance.py`. Required NumPy/SciPy/statsmodels; Python >=3.11; AGPL-3.0-only. Official source-distribution checksum appears in §6.6. Source-level API verified; complete Mac installation/execution not yet verified.

**[R5] Andrea I. Luppi, Fernando E. Rosas, Gustavo Deco, Morten L. Kringelbach and Pedro A. M. Mediano.** *Information decomposition reveals hidden high-order contributions to temporal irreversibility*. arXiv:2308.05664v1, 10 August 2023. DOI: [10.48550/arXiv.2308.05664](https://doi.org/10.48550/arXiv.2308.05664). [Primary full HTML](https://arxiv.org/html/2308.05664v1), [PDF](https://arxiv.org/pdf/2308.05664). Main irreversibility decomposition and Eqs. 1–5; Appendix treatment of Gaussian information estimates/MMI. Cite as a preprint; no journal metadata was established in this review.

**[R6] Christopher W. Lynn, Caroline M. Holmes, William Bialek and David J. Schwab.** *Emergence of local irreversibility in complex interacting systems*. Physical Review E 106, 034102 (2022). DOI: [10.1103/PhysRevE.106.034102](https://doi.org/10.1103/PhysRevE.106.034102). [Primary author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC9751845/). Sec. II: joint transition probabilities and local irreversibility; Sec. III and Appendix A: multipartite restriction on the decomposition.

**[R7] Ignacio A. Martínez, Gili Bisker, Jordan M. Horowitz and Juan M. R. Parrondo.** *Inferring broken detailed balance in the absence of observable currents*. Nature Communications 10, 3542, 6 August 2019. DOI: [10.1038/s41467-019-11051-w](https://doi.org/10.1038/s41467-019-11051-w). [Primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6684597/). Results Eqs. 1–7 and Methods: semi-Markov waiting-time evidence and the Markov special case.

**[R8] Massimiliano Zanin and David Papo.** *Algorithmic Approaches for Assessing Multiscale Irreversibility in Time Series: Review and Comparison*. Entropy 27(2), 126, 25 January 2025. DOI: [10.3390/e27020126](https://doi.org/10.3390/e27020126). [Primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11854910/). Methods and Sec. 9 discussion support configuration/timescale dependence; they are not results on the present challenge.

**[R9] Melvin J. Hinich and Philip Rothman.** *Frequency-Domain Test of Time Reversibility*. Macroeconomic Dynamics 2(1), 72–88, March 1998. DOI: [10.1017/S136510059800604X](https://doi.org/10.1017/S136510059800604X). [Author-hosted primary PDF](https://www.la.utexas.edu/hinich/files/Statistics/TR-test.pdf). Sec. 2.1–2.2, Eqs. 1–4: third-order/bispectral basis and testing context.

**[R10] Belinda Phipson and Gordon K. Smyth.** *Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn*. Statistical Applications in Genetics and Molecular Biology 9(1), Article 39, 31 October 2010; corrected 9 February 2011. DOI: [10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585). [Author-hosted corrected primary paper](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf). Basis for including the observed outcome and avoiding zero Monte Carlo p-values; exchangeability for this specific pair design is derived in §6.1.

**[R11] Albert H. Bowker.** *A Test for Symmetry in Contingency Tables*. Journal of the American Statistical Association 43(244), 572–574 (1948). DOI: [10.1080/01621459.1948.10483284](https://doi.org/10.1080/01621459.1948.10483284). [Bibliographic record](https://pubmed.ncbi.nlm.nih.gov/18123073/). Historical attribution for the symmetry statistic; full article was not accessible during this review. This study uses the explicitly derived conditional binomial calibration, not an unverified sparse-table asymptotic claim.

**[R12] Pierre Gaspard.** *Time-reversed dynamical entropy and irreversibility in Markovian random processes*. Journal of Statistical Physics 117, 599–615 (2004). [Author-hosted primary PDF](https://gaspard.pierre.web.ulb.be/G.JSP.04.pdf). Secs. II–IV, especially Eqs. 17 and 27–29: Markov trajectory reversal and sampled transition kernels. Physical entropy-production interpretations require their additional physical assumptions.

**[R13] Yijia Cheng, Ruicheng Bao and Zhonghuai Hou.** *Hierarchical Reconstruction of Time-arrow from Multi-time Correlations*. arXiv:2604.25749v3, 8 September 2026. [Versioned primary HTML](https://arxiv.org/html/2604.25749v3), [abstract](https://arxiv.org/abs/2604.25749). Main Setup and Eqs. 3, 8–13: multi-time correlation hierarchy and observation losses. Recent preprint, inspected for related conceptual scope; not a reproduced comparator here.

**[R14] SciPy community.** Official API documentation, accessed 1 November 2026: [`scipy.linalg.expm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html), [`scipy.stats.binomtest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html). Use the installed version's scalar interfaces and record its exact version. Matrix exponentials are numerical oracles checked against independent rational identities, not proofs by themselves.

## 13. Review findings, stress tests and residual risks

Three distinct postdraft passes were performed by independent agents, and reconciled by the research lead: (1) mathematical/source/novelty review, (2) adversarial experimental review, and (3) execution from an empty project on the stated laptop. The mathematical reviewer also rechecked the late-lag KL value and the affected source/control corrections. These were reviews of this brief and small planning calculations, not independent executions of the PoC.

Material findings and their disposition:

| Finding | Repair or explicit consequence |
|---|---|
| “Pairwise” could overstate the theorem | Limited the certificate to isolated two-bit laws and the explicitly defined ternary-difference/binomial test; full-state and joint-feature methods remain legitimate strong competitors |
| A trivial clique null cannot expose overlap miscalibration | Added the stationary-matched reversible refresh process, with proofs of its matched observations |
| Four diagnostic labels could falsely imply black-box identifiability | Split oracle-backed annotations from detector outputs and allowed plain known-model reversibility without requiring a hidden explanation |
| C5 was initially described as escaping C4 count lumping | Corrected it: C5 also has a reversible count generator, derived/checked it, and added the corresponding null-alarm coverage |
| The current external method was closer than the original cited comparison | Selected the 2026 detailed-balance statistic, inspected its official code, and specified list-of-paths input, empty-row behavior, required adapter-equivalence checks and explicit calibration boundaries |
| A copied API snippet had a stray character | Repaired it; Python snippets and JSON were parsed during final artifact checks |
| A separate count-only randomization loop could exceed the cap | Required exact two-category reduction or projection of shared full-state null tables, with the binomial aggregation justification |
| Sample accounting and dense-record reversal could be misimplemented | Defined endpoint/gap accounting, timestamp reversal conventions, elementary-flip weights and relabeling-coupled randomization |
| A budget stop could omit the sequential bridge after unnecessary breadth | Prioritized the primary/late-lag/bridge evidence before remaining breadth; retained explicit incomplete-outcome rules |

The following stress cases were considered before delivery and are addressed operationally:

| Stress case | Required handling already built into the brief |
|---|---|
| The theorem is nonconstructive or hides a costly oracle | Explicit last-event/refresh algorithms; exponential enumeration isolated to tiny validation. If that construction fails its checks, stop rather than inherit a guarantee. |
| A defining assumption fails on realistic inputs | Restrict claims to known stationary fixed-rate simulations; reject declared invalid inputs and state assumptions on unknown records. No causal attribution from a rejection. |
| The mechanism is equivalent to a known method | Established transition/JSD/cubic prior art is credited; the contribution is a bounded diagnostic, with historical novelty unresolved. |
| Only favorable synthetic cases work | Include a matched null, low-power short-record cell, a mathematically hard late-lag cell, balanced/unbalanced controls and a sequential bridge; generalization remains limited. |
| A properly implemented strong baseline removes the gain | This is an allowed result; report that standard full-state analysis suffices and do not claim witness superiority. |
| Preprocessing or null generation dominates runtime | Measure total and isolated costs; verify the exact counter adapter; forecast with 2x margin and count all setup/repeats. |
| Finite precision changes the conclusion | Rational identities, independent small matrices, stable exponential differences, sparse exact calibration and conservative tie rules precede interpretation. |
| RAM or runtime exceeds estimates | Chunk first, choose the predeclared reduced protocol before confirmation, then stop if meaningful inference still does not fit. |
| A small pilot is inconclusive | Do not decide power from it; continue to fixed independent confirmation when correctness and resources permit. |
| A critical source, package or target machine is unavailable | Use available primary copies/explicit formulas; complete unaffected work; identify the missing comparison or hardware validation, without inventing it. |
| The main hypothesis is false | Preserve the primary 512-pair target, report a method-specific limitation or inconclusive interval, and resist moving the goalposts to another lag or sample count. |

No known fixable defect remains that presently invalidates the design. This is not an assurance of flawless implementation or historical originality. The essential remaining uncertainties are target-machine dependency compatibility, independent validation of the new interval sampler and count adapter, actual comparative power, and measured runtime. They are explicit early gates, not unstated assumptions. Execute the authorized stages autonomously within those gates and deliver the supported outcome, including an honest early stop when warranted.




