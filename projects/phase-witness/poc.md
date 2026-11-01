# POC_EXECUTION_BRIEF

## 1. Project decision and mandate

**Project:** Finite-resolution limits of phase certification, with an exact bounded-curvature diagnostic.

**Assigned executor:** An autonomous coding agent configured as **gpt-6.1-sol**, reasoning effort **xhigh**.

**Prepared:** 27 October 2026. Literature search date: 27 October 2026.

**Your task:** Implement, validate, run, and interpret the small study specified below on one MacBook with an M3 Max. Follow the evidence gates and stop rules autonomously. Produce the research result, including an informative negative result when warranted. This document is the execution specification; the proof of concept has not already been implemented or benchmarked.

### The decision this study informs

Decide whether finite observations of scalar quantum loop phases can support the proposed claim that a refined model has the wrong Euclidean phase geometry, and identify the extra assumptions needed for a sound, useful rejection.

The broad application proposal needs narrowing. On any fixed dyadic grid, its principal discontinuous example is **exactly indistinguishable from a smooth bilinear phase law with sufficiently large curvature**. More measurements on that same grid, lower noise, and exact associativity checks do not remove this ambiguity. An independently justified bound on curvature changes the problem into ordinary bounded-error, wrapped-parameter feasibility.

Accordingly, execute a **limitation and diagnostic study**, not a new quantum simulation algorithm. The smallest defensible contribution is:

1. A verifiable, proposal-specific account of the finite-grid obstruction, including an explicit smooth alias.
2. A sound finite-data diagnostic that returns either a bounded-model exclusion, a compatible parameter, or an explicit reason it cannot decide.
3. A quantitative boundary between guaranteed discrimination and unavoidable ambiguity under a declared phase-error budget.
4. A controlled comparison showing exactly when enforcing one common parameter across scales adds information beyond checking every loop separately.

The no-go calculation and interval method are elementary consequences of established structures. **Do not claim historical novelty for them.** Their value here is resolving an unsupported application claim and specifying a smaller, valid capability. The present literature assessment did not establish a distinct practical advantage over existing estimation or structure-preserving methods.

### One primary research question

> Given a finite dyadic control grid, an independently specified curvature bound, and simultaneous bounded phase errors, when do the observed loop phases exclude every regular bilinear phase model, and when do the available observations admit a smooth explanation?

The question has an analytic part and an executable part. The analytic part already rules out unrestricted finite-grid certification of continuum discontinuity. The implementation must verify the bounded alternative and quantify its scope; it must not reinterpret a numerical demonstration as proof of a general stability theorem.

### Fixed scope and exclusions

The implemented model has two real label coordinates, one real curvature parameter, exact dyadic labels, and scalar phases supplied as rational circular intervals. Test at most seven elementary loops per multiscale query set. All scientific decisions use exact rational arithmetic.

Do not implement an oscillator simulator, Hamiltonian dynamics, quantum hardware control, tomography, a general model-reduction framework, machine learning, arbitrary-dimensional mixed-integer optimization, or a new approximate-cocycle stability theory. Do not infer scalar closure, a curvature bound, continuity, or control calibration from a handful of successful algebra checks.

This revision retains the practical purpose of preventing unjustified model-consistency claims. It does not certify full quantum dynamics, Hilbert-space truncation, energy-bounded simulation, physical devices, general coordinate compression, or arbitrary dense subgroups. Those are excluded because the current theorem and available evidence do not supply the necessary quantitative bridge.

A supported result can justify an inexpensive model-specification checker. It does not yet justify investing in general experimental quantum certification.

## 2. Evidence status and supplied inputs

### 2.1 Attachments and provenance

Two files were supplied to the planning investigation.

| Input | Original location in the planning workspace | Status |
|---|---|---|
| **arithmetic_phase_research(1).pdf** | /workspace/scratch/172be03857b4/upload/arithmetic_phase_research(1).pdf | All 13 pages read, including the appendix and reference list; key equations on pp.5–6 also inspected in rendered pages. |
| **Pasted text(20261027-204158).txt** | /workspace/scratch/172be03857b4/upload/Pasted text(20261027-204158).txt | Complete application proposal and research-planning instructions read. |

Their SHA-256 hashes are:

~~~text
arithmetic_phase_research(1).pdf
d0985ae72b3ab983327b0d03ea62aa0204bf86a7a090bc18d47dada3864110d3

Pasted text(20261027-204158).txt
9b2fb0c43641248711510e73991090d3a8c9159671c92aeaf7c4b179efa83b9d
~~~

The absolute paths above identify the actual planning inputs; they are not assumed to exist on the Mac. If the files accompany your handoff, retain them under **sources/**, with the PDF optionally renamed **arithmetic_phase_research.pdf** and the text **application_request.txt**, preserving their bytes and hashes. No experimental dataset or external service is required. The definitions and proofs essential to this selected study are reproduced below, so the implementation does not depend on the PDF extraction.

The PDF refers to **verify_arithmetic_phases.py**, source files, JSON results, and an audit bundle in §B, p.13. Those files were **not supplied**. Its reported enumeration counts are source-reported observations, not checks independently rerun here. Recreate only the small relevant checks specified below. Missing companion files do not block the selected study.

### 2.2 Source audit

The paper is titled *Regularity constraints on arithmetic with real exponents*, subtitled *An autonomous mathematical investigation*, dated 27 October 2026 [S1]. It gives proofs but makes no claim of external peer review or formal proof-assistant verification.

| Claim | Location and evidence status | Consequence for this study |
|---|---|---|
| An associative normalized scalar cocycle has an alternating bicharacter as its exchange factor. | [S1], §1, eqs.(1)–(3), p.1; established background also in [R2], §3.1. | The exact generators below are legitimate scalar phase algebras. |
| For dense \(G\subseteq\mathbb R^d\), separate continuity gives an additive frequency map and a graph-span restriction; \(\dim_HG>d-1\) implies a jointly continuous bilinear extension. | [S1], Theorem 2.1 and proof, pp.2–3. Independently reviewed during planning; no material defect found in the mechanism used here. | A theorem about an entire specified group and regularity assumptions, not a test of finite samples. |
| One-dimensional exceptional groups and their integer phase index. | [S1], Theorem 3.1, p.4; cocycle realization, Proposition 3.3, p.5. | Relevant background and an independent finite-noise limitation; not an implemented branch. |
| Refinement by \(1/N\) requires \(N^2\mid k\) in that quadratic classification. | [S1], Theorem 4.1, p.5. | Exact symbolic obstruction; not automatically robust to finite noisy index inference. |
| Divisibility eliminates exceptional discontinuity. | **Already [S1], Corollary 4.3, p.6.** The application text presents it as an additional derivation, but it is present in the supplied PDF. | Cite it correctly. The proof also works for any fixed integer \(p\ge2\), although the PDF states prime \(p\). |
| Stability under approximate cocycle/continuity assumptions. | Explicitly unresolved in [S1], §6.3, p.8. | Do not inherit a quantitative guarantee the paper does not prove. |
| Hahn-field dimension/root-solvability constructions. | [S1], Appendix A, pp.9–13. Read; independent of the selected phase diagnostic. | No PoC implementation or empirical claim about these constructions. |
| Finite-grid smooth alias, bounded feasibility, and discrimination boundary below. | Derivations in this brief, with close prior structures in [R3]–[R5]. | Verify the derivations and implement their finite consequences. No general novelty claim. |

Separate continuity is not automatically joint continuity on an arbitrary dense Euclidean subgroup. Existing bihomomorphism results use additional hypotheses such as \(g\)-barrelledness or the Baire property [R1, Theorem 3.3 and Proposition 3.4]. Do not silently apply these hypotheses to the dyadic group.

### 2.3 Why the alternatives were not selected

| Alternative | Decision-relevant prior art or issue | Decision |
|---|---|---|
| Detecting an invalid linear compression of a known alternating form | The criterion is standard linear algebra. Structure-preserving quantum reductions already preserve physical realizability within their own approximation objectives [R6, Theorem 2; R7, §3.1]. | Do not construct an artificially broken reduction and call its detection a new method. |
| Detecting bad finite-Fock displacement matrices | Existing work explicitly distinguishes unitarity of truncated-generator exponentials from accurate displacement matrices and supplies enlarged-space construction and checks [R8, §§4–6]. | An oscillator benchmark would require a separate quantitative error study and a credible advantage over this work. It is not needed to resolve the current identifiability issue. |
| Robust quantum phase estimation | Established protocols estimate phases with defined observation and resource models [R9]. | They do not eliminate aliases of an unrestricted real curvature; implementing them would change the study. |
| General interval optimization | Bounded-error feasible-set estimation is established [R4]; wrapped parameter aliases are established [R5]. | Use the elementary one-parameter specialization. Do not create a new optimizer or claim algorithmic novelty. |

The full-space condition \(\Omega=P^\mathsf T\Omega_{\rm red}P\) for preserving **every** phase through a compression is not the same objective as approximating input-output dynamics while retaining a valid reduced commutation structure. The former must not be used to dismiss the latter.

## 3. Mathematical specification

### 3.1 Objects, units, and observation contract

Let

\[
G=\mathbb Z[1/2]^2,\qquad
G_L=2^{-L}\mathbb Z^2,\qquad L\in\mathbb Z_{\ge0}.
\]

The topology under discussion on \(G\) is inherited from \(\mathbb R^2\). Each fixed \(G_L\) is discrete in that topology. Continuity at every individual level therefore says nothing by itself about continuity on the union.

For labels \(g=(g_1,g_2)\) and \(h=(h_1,h_2)\), define the signed area

\[
a(g,h)=\det(g,h)=g_1h_2-g_2h_1.
\]

The regular candidate class is

\[
\mathcal R_M=\left\{
\beta_\omega(g,h)=\exp(2\pi i\omega a(g,h)):
\omega\in[-M,M]
\right\},
\qquad M\ge0.
\]

Equivalently, \(\Omega=\omega J\), with

\[
J=\begin{pmatrix}0&1\\-1&0\end{pmatrix},
\qquad g^\mathsf T\Omega h=\omega a(g,h).
\]

Phase is measured in **turns**, so one turn is \(2\pi\) radians. Curvature \(\omega\) has units of turns per unit label area. Fix the label units in each configuration. Rescaling labels without transforming \(M\) changes the problem.

The circle distance is

\[
d_{\mathbb T}(x,y)=\min_{k\in\mathbb Z}|x-y-k|\in[0,1/2].
\]

An observation is a tuple

\[
(g_i,h_i,\widehat y_i,\varepsilon_i),
\quad
\widehat y_i\in[0,1),\quad
0\le\varepsilon_i\le1/2,
\]

asserting that the true scalar loop phase in turns lies within \(\varepsilon_i\) of \(\widehat y_i\). All labels, centers, radii, and finite bounds supplied to the executable core are rational.

The error assertion must be simultaneous across the observation set. Deterministic bounds need no probabilistic independence. If future inputs use confidence regions, their joint coverage must be justified separately; per-query confidence levels cannot simply be called a simultaneous guarantee.

The physical motivation is

\[
U_gU_hU_g^{-1}U_h^{-1}=\beta(g,h)I.
\]

A coherent reference path can expose this phase. The current implementation receives the scalar phase intervals; it does not construct the \(U_g\), prove this scalar identity for unknown operators, or infer it from a single state. No hardware measurement is being performed.

### 3.2 The exact dyadic adversary

For a common denominator \(2^n\), write

\[
g=2^{-n}(m_1,m_2),\qquad h=2^{-n}(r_1,r_2).
\]

Define

\[
b(g,h)=\frac{m_1r_2-m_2r_1}{3}\pmod1,\qquad
\beta_{\rm bad}(g,h)=\exp(2\pi i b(g,h)).
\]

Also define a cocycle in turns,

\[
s_{\rm bad}(g,h)=\frac{m_1r_2}{3}\pmod1,\qquad
c_{\rm bad}(g,h)=\exp(2\pi i s_{\rm bad}(g,h)).
\]

Increasing the common denominator by a factor two multiplies each numerator by two and these products by four. Since \(4\equiv1\pmod3\), both formulas are representation independent. On any common grid the exponents are bilinear in the integer coordinates, which proves the bicharacter and cocycle identities. Antisymmetrizing \(s_{\rm bad}\) gives \(b\).

The elementary loops

\[
g_n=2^{-n}e_1,\qquad h_n=2^{-n}e_2
\]

have phase \(b(g_n,h_n)=1/3\) at every scale. Both labels tend to zero. More strongly,

\[
\beta_{\rm bad}(e_1,2^{-n}e_2)
=\exp(2\pi i2^n/3)
\]

alternates between two nontrivial values and does not tend to one. Thus separate continuity fails for the Euclidean topology.

This is an internally consistent algebra with an unsuitable regular continuum interpretation. Do not label the algebra itself inconsistent. Related refined twisted algebras are established objects; [R3, Definition 1.1] explicitly uses the **discrete** topology.

### 3.3 Finite-grid nonidentifiability: the decisive obstruction

For any fixed finite \(L\), set

\[
\omega_L=\frac{4^L}{3}.
\]

For every \(g=2^{-L}m\) and \(h=2^{-L}r\),

\[
\omega_L a(g,h)=\frac{\det(m,r)}{3},
\]

so

\[
\boxed{\beta_{\omega_L}(g,h)=\beta_{\rm bad}(g,h)
\quad\text{for all }g,h\in G_L.}
\]

The smooth cocycle

\[
c_{{\rm reg},L}(g,h)=
\exp(2\pi i\omega_L g_1h_2)
\]

also agrees exactly with \(c_{\rm bad}\) on \(G_L\times G_L\). Therefore even scalar multiplication data for finite words in those labels do not distinguish these two specified algebras.

**Operational consequence.** With a hard maximum refinement depth \(L\), a phase-only observation channel has identical data distributions under these two models whenever its noise mechanism is otherwise identical. Adaptive choices within the grid, repeated measurements, and zero measurement noise cannot distinguish them. This statement does not cover extra access to generators, control-resource bounds, state-dependent responses, or other physical observables.

Any fixed finite set of dyadic queries lies on some such grid. A terminating finite transcript always has a compatible smooth interpolant for this adversary. Do not extend this statement without proof to every possible randomized strategy with unbounded refinement depth and merely almost-sure finite stopping.

**General finite-grid version.** Put \(G_L^{(d)}=2^{-L}\mathbb Z^d\). Any alternating bicharacter on \(G_L^{(d)}\cong\mathbb Z^d\) is determined by its basis-pair phases. Choose real lifts \(\theta_{ij}\) of those phases for \(i<j\), put \(\Omega_{ij}=4^L\theta_{ij}\), \(\Omega_{ji}=-\Omega_{ij}\), and \(\Omega_{ii}=0\). The regular law \(\exp(2\pi i g^\mathsf T\Omega h)\) agrees with the bicharacter on the whole grid. This uses the free abelian lattice structure; it is not a claim about arbitrary finite samples from every dense subgroup.

These calculations are explicit forms of finite-stage interpolation and aliasing, closely related to [R3, Theorems 2.3 and 3.7] and [R5, §3]. They are a decision-relevant limitation of the proposal, not a claimed new general theorem.

### 3.4 What the source regularity theorem does, and does not, add

For completeness, the relevant proof mechanism of [S1, Corollary 4.3] is short. Separate continuity extends each character in the second variable to \(\mathbb R^d\), giving a unique additive frequency map \(a:G\to\mathbb R^d\) such that

\[
\beta(g,h)=\exp(2\pi i\langle a(g),h\rangle).
\]

Alternation implies \(q(g)=\langle a(g),g\rangle\in\mathbb Z\). If \(g/p^n\in G\) for all \(n\),

\[
q(g/p^n)=q(g)/p^{2n}\in\mathbb Z,
\]

hence \(q(g)=0\). Polarization makes the graph of \(a\) totally isotropic for the nondegenerate symmetric form

\[
B((x,u),(y,v))=\langle u,y\rangle+\langle v,x\rangle
\]

of signature \((d,d)\). Its real span has dimension at most \(d\); density of \(G\) gives dimension at least \(d\). It is consequently the graph of a skew linear map.

This supplies an exact classification under an assumed global regularity property. It supplies neither a finite test of that property nor a bound on \(\|\Omega\|\). Every finite sample has Hausdorff dimension zero. The finite-grid construction above explains why adding a finite sample cannot fill this gap.

### 3.5 Bounded finite-data feasibility

For a finite, externally supplied \(M\), define

\[
F(\widehat y,\varepsilon;M)
=[-M,M]\cap
\bigcap_i\{\omega:
d_{\mathbb T}(a_i\omega,\widehat y_i)\le\varepsilon_i\}.
\tag{1}
\]

If \(a_i>0\), its individual constraint is

\[
\bigcup_{k\in\mathbb Z}
\left[
\frac{\widehat y_i-\varepsilon_i+k}{a_i},
\frac{\widehat y_i+\varepsilon_i+k}{a_i}
\right].
\tag{2}
\]

Only finitely many intervals intersect a finite current parameter set. Thus (1) is exactly computable using rational interval unions.

Use these meanings:

| Diagnostic status | Meaning |
|---|---|
| **EXCLUDED_BOUNDED_CLASS** | \(F=\varnothing\), with a complete exact exclusion computation. Every member of the declared bounded bilinear class violates at least one supplied interval. |
| **COMPATIBLE_FINITE_DATA** | \(F\ne\varnothing\); return a rational \(\omega\in F\). This proves finite compatibility only. |
| **ASSUMPTION_MISSING** | A required prior, scalar-phase interpretation, exact-label contract, or certified error statement is absent. |
| **INCONCLUSIVE_RESOURCE** | The permitted computation cannot establish the answer within its limits. |
| **INVALID_INPUT** | Malformed rational data, inconsistent dimensions, negative radii/bounds, unsupported input type, or a broken input contract. |
| **INVALID_IMPLEMENTATION** | An independent correctness check fails. Comparative conclusions from affected results are invalid until repaired and revalidated. |

A scientifically valid exclusion can arise from wrong curvature, wrong units, incorrect controls, loss of scalar closure, or a discontinuous law. It does **not** identify which explanation is responsible. A regular law outside \([-M,M]\) may correctly be excluded.

A rejecting witness may require several loops whose constraints contradict each other. Return a finite witness set and an exclusion record; do not promise one loop when no single loop suffices.

### 3.6 Sharp finite-noise boundary

Use the prescribed multiscale query set

\[
Q_L=\{(2^{-n}e_1,2^{-n}e_2):n=0,\ldots,L\},
\qquad a_n=4^{-n}.
\]

For the bad law, every ideal center is \(y_n^{\rm bad}=1/3\). Define

\[
D(M,L)=
\min_{|\omega|\le M}
\max_{0\le n\le L}
d_{\mathbb T}(\omega4^{-n},1/3).
\tag{3}
\]

The minimum exists because the interval is compact and the objective is continuous.

There are two different thresholds:

- Exact-center observations with uncertainty radius \(t\) exclude the regular class iff \(D(M,L)>t\).
- If observations can differ from either ideal model by at most \(\varepsilon\) per coordinate, their two observation sets are disjoint iff
  \[
  \boxed{D(M,L)>2\varepsilon.}
  \tag{4}
  \]

For necessity in (4), suppose a regular vector is at circle sup-distance at most \(2\varepsilon\) from the bad vector. A coordinatewise midpoint along shortest circle arcs lies within \(\varepsilon\) of both. For sufficiency, the triangle inequality would make any common observation imply distance at most \(2\varepsilon\). Closed intervals make equality **ambiguous**, not separated.

This is a worst-case bounded-error result, not a rejection-rate claim under a fitted noise distribution. No Monte Carlo trials are needed to establish it.

Compute the exact decision in (4) by evaluating (1) at the bad centers and derived radius \(t=\min(2\varepsilon,1/2)\). A half-turn radius imposes no constraint, so this cap preserves the decision and keeps the internal call within the public radius contract. Numerical bisection is only for reporting a narrow rational bracket for \(D\), not for making an uncertain decision at a boundary.

If this capped feasibility call returns \(\omega_*\), construct

\[
r_n=
\left((a_n\omega_*-1/3+1/2)\bmod1\right)-1/2,
\qquad
\widehat y_n^{\rm mid}=(1/3+r_n/2)\bmod1.
\tag{5}
\]

Verify exactly that both ideal phase vectors are within \(\varepsilon\) of this observation. This is an explicit indistinguishability certificate.

### 3.7 Cheap baseline, exact thresholds, and the mechanism control

Give the simple baseline **the same complete observations**, bound, and error radii as the joint method. It tests each loop independently against some \(\omega_i\in[-M,M]\), and rejects only if at least one individual constraint is empty. It is permitted to use a different parameter for each query; this is precisely the removed common-parameter requirement.

Implement this baseline with the complete closed-form test

\[
\max(0,d_{\mathbb T}(\widehat y_i,0)-|a_i|M)\le\varepsilon_i.
\]

It is valid even when the allowed arc wraps: the image of the symmetric prior is centered at zero, and covers the entire circle once \(|a_i|M\ge1/2\). For \(a_i\ne0\), a parameter minimizing the single-query phase error is obtained by taking a signed representative of \(\widehat y_i\) in \([-1/2,1/2]\), clipping it to \([-|a_i|M,|a_i|M]\), and dividing by the signed \(a_i\). Return it as a compatible witness only when the individual feasibility test passes. For \(a_i=0\), check the zero-phase condition. Do not call the full interval solver separately for every baseline query. The complete individual interval unions are needed only for tiny reference fixtures such as eq.(8).

For \(Q_L\) and constant bad centers, the strongest individual query is the finest one, and its ideal separation is

\[
D_{\rm individual}(M,L)
=\max(0,1/3-M4^{-L}).
\tag{6}
\]

The sharp corresponding complex-plane chord separation is

\[
2\sin\!\left(\pi\max(0,1/3-M4^{-L})\right).
\tag{7}
\]

Do not substitute \(\sqrt3-2\sin(\pi M4^{-L})\) as an exact distance. That expression is only a weaker radial/triangle bound in the regime where it is positive.

For exact noiseless data containing the finest elementary cell, compatibility holds iff \(M\ge4^L/3\). This threshold is not valid for every arbitrary subset of labels; missing primitive determinant information can weaken it.

**Known-answer control.** For \(M=1,L=1\),

\[
D_{\rm individual}=1/12,\qquad
D=1/5,\qquad
\omega_{\rm opt}=8/15.
\]

At \(\omega=8/15\), the two signed residuals are \(+1/5\) and \(-1/5\). For \(t<1/5\), the finest constraint forces \(\omega\ge4/3-4t>8/15\), while the viable coarse branch forces \(\omega\le1/3+t<8/15\); other winding branches cannot satisfy both within \([-1,1]\).

For an actual same-input difference between the methods, use

\[
\varepsilon=1/16,\qquad
(\widehat y_0,\widehat y_1)=(1/3,13/48).
\tag{8}
\]

This observation is within \(\varepsilon\) of the bad vector. The individual feasible sets are

\[
F_0=[-35/48,-29/48]\cup[13/48,19/48],
\qquad F_1=[5/6,1].
\]

They are individually nonempty and jointly disjoint. The joint method rejects; the individual baseline does not. The exact-center vector at this same radius would already be rejected by the finest query, so it is not the correct fixture for isolating this mechanism.

This is a designed positive control with an analytically known answer. It proves the nature of the comparison, not prevalence or utility on real simulators.

### 3.8 Independent noiseless oracle for arbitrary dyadic queries

For a finite query set on \(G_L\), write \(a_i=D_i/4^L\), with integer \(D_i\). For the bad generator, matching all phases requires

\[
D_i(\omega/4^L-1/3)\in\mathbb Z.
\]

If at least one \(D_i\ne0\), let \(d=\gcd_i|D_i|\), omitting zeros. Bézout's identity gives

\[
\omega=4^L(1/3+j/d),\qquad j\in\mathbb Z.
\]

Therefore the minimum possible absolute curvature is

\[
M_{\min}=\frac{4^L}{d}\operatorname{dist}(d/3,\mathbb Z)
=
\begin{cases}
0,&3\mid d,\\
4^L/(3d),&3\nmid d.
\end{cases}
\tag{9}
\]

If all areas vanish, \(M_{\min}=0\). This is a separate elementary oracle for the zero-noise general-input tests. Include primitive, nonprimitive, zero, and negative areas. Do not extrapolate the finest-cell threshold to a nonprimitive data set.

## 4. Implementation specification

### 4.1 Stack and invariants

Use standard **Python 3.11 or newer**, preferably an available native arm64 interpreter. The required core uses only the standard library: fractions, math, itertools, json, pathlib, hashlib, argparse, time, resource, subprocess, platform, sys, unittest, and csv as needed. No GPU, CUDA, Metal, BLAS, scientific simulator, external solver, or paid service is needed.

All decision quantities use **fractions.Fraction**, constructed from integer numerator/denominator strings, never from a binary float. Preserve rational strings in inputs and outputs. Floats may appear only as labeled display approximations and in timers/resource measurements.

Normalize all phase centers modulo one. Keep closed interval endpoints exactly; retain singleton intervals and merge touching intervals. Do not round away a narrow interval, truncate winding branches, or replace an interval union by its convex hull.

In the public input format, reject JSON numeric decimals for scientific rational fields. Accept strings such as "1", "-2/3", and "13/48". Accept integer level indices. This avoids ambiguous conversion policies.

Core execution must be serial initially. There is no numerical acceleration requirement. A dependency or backend failure is not a reason to replace exact arithmetic with floating-point comparisons.

### 4.2 Minimal repository

Create the following small project. This is a specification of files to create, not a claim that they already exist.

| Path | Responsibility |
|---|---|
| README.md | Setup, commands, status meanings, limits, and how to reproduce a certificate. |
| THEORY.md | Proofs of finite-grid aliasing, exact interval completeness, noise boundary, and oracle (9). |
| phase_audit.py | Exact labels/phases, interval solver, individual baseline, minimax bracketing, and CLI. |
| reference.py | Independent tiny winding-branch enumeration and optional two-variable vertex enumeration. Do not call the main interval-update routine. |
| verify_certificate.py | Small checker of compatible parameters, ambiguous observations, and complete exclusion records. |
| experiments.py | Deterministic generators, development/pilot/main selection, cumulative budget ledger, execution, and CSV/JSONL output. |
| report.py | Regenerate research report and essential tables from saved results without rerunning experiments. |
| test_phase_audit.py | Necessary mathematical correctness and failure-behavior tests. |
| config.json | Exact fixed protocol and hard resource caps. |
| sources/ | Optional copies of attachments; source identifiers and hashes. |
| results/ | Append-only run records, certificates, environment, frozen protocol, and report outputs. |

Do not create services, dashboards, distributed workers, plug-in frameworks, a generic experiment database, or an optimizer abstraction.

### 4.3 Public interfaces

Implement functions with the following logical contracts; ordinary dataclasses or dictionaries are acceptable.

~~~python
bad_phase_turns(g, h) -> Fraction
bad_cocycle_turns(g, h) -> Fraction
regular_phase_turns(omega, g, h) -> Fraction

feasible(observations, M, budget) -> FeasibilityResult
individual_checks(observations, M, budget) -> IndividualResult

minimax_bad(L, M, tolerance, budget) -> BoundaryResult
ambiguity_witness(L, omega, epsilon) -> AmbiguityCertificate

finite_grid_alias(L) -> Fraction
noiseless_bad_minimum_curvature(queries, L) -> Fraction

verify_certificate(certificate, budget) -> VerificationResult
~~~

A FeasibilityResult contains status, an ordered disjoint union of closed intervals if completed, a rational witness when feasible, processed query IDs, interval/candidate counters, timing, and any stop reason. Limit output verbosity for large unions; the complete small certificate can be regenerated from its exact inputs.

An exclusion record must contain the bound, exact observation data, arithmetic conventions, input hash, complete processing order, and either the intermediate interval unions or sufficient deterministic reconstruction metadata for the verifier to replay every allowed branch. The verifier must not accept "the final list is empty" without recomputing the complete intersections. The original observations themselves form an adequate nonminimal witness set; do not spend the budget minimizing it.

For a compatible result, independently check every residual at the returned rational witness. For ambiguity, check both model vectors against the same observation and radius. For resource termination, no exclusion certificate may be emitted.

### 4.4 Exact interval algorithm

The current feasible set is a sorted disjoint union of closed rational intervals. Process constraints in ascending absolute nonzero area, breaking ties by stable query ID, with zero-area constraints first. This order often limits the initial number of intervals; correctness must be independent of order.

~~~text
FEASIBLE(observations, M, budget):
    Validate rational fields, labels, dimensions, and assumptions.
    If M is missing:
        return ASSUMPTION_MISSING.
    Require M >= 0.
    current := [[-M, M]]

    For each observation in the prescribed stable order:
        a := det(g, h)
        y := center mod 1
        t := supplied radius

        If t >= 1/2:
            continue
        If a == 0:
            if circle_distance(0, y) > t:
                return EXCLUDED_BOUNDED_CLASS with complete record
            continue
        If a < 0:
            a := -a
            y := (-y) mod 1

        next := empty interval list
        For each [l, u] in current:
            k_min := ceil(a*l - y - t)
            k_max := floor(a*u - y + t)
            Pre-count this branch range against the remaining budget.
            If a hard cap would be exceeded:
                return INCONCLUSIVE_RESOURCE; never drop branches

            For every integer k from k_min through k_max:
                left  := max(l, (y - t + k)/a)
                right := min(u, (y + t + k)/a)
                If left <= right:
                    append [left, right] to next
                Update counters and check time/RSS limits

        current := sort and exactly merge overlapping/touching intervals(next)
        If current is empty:
            return EXCLUDED_BOUNDED_CLASS with complete record

    omega := midpoint of first interval in current
    Independently evaluate all circle residuals at omega.
    If any fails:
        return INVALID_IMPLEMENTATION
    return COMPATIBLE_FINITE_DATA with current and omega
~~~

For rational \(x=p/q\) with \(q>0\), implement floor and ceiling by integer operations. Test negative values explicitly. Do not use floating-point floor/ceil near integer boundaries.

**Completeness argument to include in THEORY.md:** the winding range includes exactly all translates of the observed arc that intersect the image of a current interval. Each intersection is exact. The union therefore equals the old feasible set intersected with the new constraint. Induction proves both soundness and completeness on termination. A resource stop preserves neither a completed feasible set nor an exclusion conclusion.

### 4.5 Boundary computation

For each \((L,M)\), evaluate the bad-center feasible set at radius zero. If nonempty, report \(D=0\) exactly. Otherwise set lower \(=0\) (known infeasible) and upper \(=1/3\) (known feasible because \(\omega=0\) fits), and bisect.

At each rational midpoint \(t\), a nonempty feasible set moves upper to \(t\); an empty set moves lower to \(t\). Maintain a feasible witness at upper. Stop when upper minus lower is at most **1/1048576 turn**, or after 20 total main-solver calls for that boundary, whichever occurs first. An unresolved wider bracket is explicitly incomplete, never silently rounded into the desired precision. The initial width permits the requested tolerance within this call allocation for successful evaluations.

For every listed noise radius \(\varepsilon\), separately run exact feasibility at \(\min(2\varepsilon,1/2)\) to determine uniform separation or ambiguity. Public radii outside \([0,1/2]\) are invalid inputs; derived radii are capped before validation. Equality is decided by closed-interval feasibility, not by a rounded printed bracket. This direct call is essential near the boundary.

The precision target is over a thousand times smaller than the smallest nonzero study radius \(1/1024\), and much smaller than the secondary effect threshold below. It is a reporting precision, not a tolerance in the proof.

### 4.6 Independent reference and certificate validation

On tiny cases with at most three constraints and \(M\le2\), implement a structurally different exhaustive reference:

1. Derive finite winding ranges from the original full interval \([-M,M]\).
2. Stream the Cartesian product of winding assignments.
3. For each assignment, intersect all its elementary parameter intervals with \([-M,M]\).
4. Union all surviving intervals and compare exactly with the main solver.

Do not use main-solver pruning or its interval update function. Sharing rational parsing and the definition of determinant is acceptable; avoid sharing the code that could create a false exclusion.

For an additional known-answer optimization check, enumerate vertices of the rational two-variable linear inequalities in \((\omega,t)\) for tiny winding assignments, with \(0\le t\le1/2\). This can independently reproduce the minimax oracles. It is optional after the mandatory exact known answers and branch reference agree; do not build a general LP framework.

The certificate verifier can replay intersections in the reverse query order, enumerate all wraps directly from the original bound, and compare the reconstructed result. Its purpose is to detect missing branches and corrupted records; it is not a formal proof-assistant guarantee.

### 4.7 Representative input

Create this exact pilot file programmatically as **results/pilot_input.json**:

~~~json
{
  "label_group": "dyadic_2d",
  "label_units": "fixed dimensionless coordinates",
  "topology_claim": "Euclidean subspace",
  "phase_contract": "scalar loop phase intervals in turns",
  "bound": {
    "M": "1",
    "provenance": "predeclared synthetic model specification"
  },
  "error_contract": "simultaneous deterministic angular bounds",
  "observations": [
    {
      "id": "coarse",
      "g": ["1", "0"],
      "h": ["0", "1"],
      "center_turns": "1/3",
      "radius_turns": "1/16"
    },
    {
      "id": "fine",
      "g": ["1/2", "0"],
      "h": ["0", "1/2"],
      "center_turns": "13/48",
      "radius_turns": "1/16"
    }
  ]
}
~~~

A provenance string documents the assumption; it does not prove an empirical bound. Generated truth labels must live in evaluation metadata and must not be supplied to a diagnostic as hidden information. The two smooth/bad alias inputs must have identical public observations.

## 5. Data, comparisons, and locked evaluation

### 5.1 Deterministic generation and control envelope

No external dataset is required. The mechanism concerns exact arithmetic/topology and finite-error logical feasibility, so deterministic synthetic inputs are appropriate. Their limitation is central: they do not establish how often these issues occur in deployed models.

For the main elementary queries, both label norms are at most one, maximum depth is six, and the smallest nonzero coordinate is \(1/64\). These are normalized test limits, not measured hardware capabilities. For arbitrary-query correctness fixtures, keep coordinates in \([-1,1]\) and denominator depth at most six.

Use exact rational generation. Check denominator-representation and query-order invariance; bicharacter and cocycle identities under label addition; and \(b(-g,h)=-b(g,h)\pmod1\), with \(b(-g,-h)=b(g,h)\). Reversing one label conjugates the phase rather than leaving it unchanged. Model-code algebra tests have access to their known generator; data-based comparisons do not. Do not score this difference in information access as algorithmic superiority.

### 5.2 Development and protocol freeze

Development uses the explicit oracle \(L=1,M=1\), small \(L\le2,M\le2\) reference cases, boundary equality tests, and selected \(L=3\) calibration cases. These are available for debugging. Do not tune a scientific hypothesis to their runtime or results.

Before main execution, save **results/frozen_protocol.json** and its SHA-256 hash. It records all matrix values, observation/error conventions, algorithm order, caps, code revision, and the predeclared reduced matrix if selected by calibration.

The confirmatory grid is fixed below. Its deterministic mathematical answers are not a statistical held-out population; the point of the freeze is to prevent selective coverage and retrospective threshold changes. Mark any main row already used in development. Primary comparisons must still include all scheduled rows or explicit failure records.

If a material code or interpretation defect appears after freeze, label the affected run invalid, repair within the allowance, make a new protocol revision, and rerun every affected row. Do not pool incompatible revisions. No additional favorable-only grid is authorized.

### 5.3 Main matrix

For each

\[
L\in\{0,1,2,4,6\},
\]

take

\[
M=\mu4^L,\quad
\mu\in\{0,1/12,1/6,1/4,1/3-1/1024,1/3,1/2\}.
\]

This gives **35 \((L,M)\) configurations**. For each, evaluate

\[
\varepsilon\in\{0,1/1024,1/256,1/64,1/16\},
\]

giving **175 bounded-noise decision cells**. Compute \(D\) once per \((L,M)\), not once per noise radius.

The normalized choices span no curvature, a clear exclusion regime, values approaching exact alias feasibility, the exact threshold, and a class containing a smooth alias. The near-threshold offset is fixed in advance. The grid is a coverage design, not a sample of physical instruments.

Across different \(L\), holding \(\mu\) fixed changes the absolute bound \(M=\mu4^L\). Interpret such comparisons as normalized-bound coverage, not as the causal effect of finer controls at one fixed physical curvature bound. The known-answer cases with \(M=1\) provide small fixed-bound checks; no broader fixed-\(M\) performance claim is authorized.

| Family | Work | Scientific purpose |
|---|---|---|
| Finite-grid alias | For each distinct main \(L\), verify \(\omega_L=4^L/3\) against all main queries and a small deterministic collection of mixed-coordinate labels. Check the cocycle too. | Exact nonidentifiability and correct normalization. |
| Bad multiscale boundary | All 35 configurations; bracket \(D\). | Quantify the best possible common-parameter separation for this query set. |
| Bounded-noise decisions | All 175 cells; feasibility at \(\min(2\varepsilon,1/2)\). Construct and verify ambiguity witnesses whenever feasible. | Distinguish guaranteed discrimination from impossible worst-case discrimination. |
| Same-information individual baseline | All the same cells and actual observation fixtures. Use complete per-query feasible tests, not principal-angle regression. | Isolate only the common-parameter requirement. |
| Regular null controls | For every \((L,M)\), use \(\omega=M\) and \(\omega=-M/2\), with radius \(1/64\) and deterministic errors \(e_n=(-1)^n/64\) modulo one. | Verify no false rejection of known in-class models under allowed error. At \(M=0\), duplicate truths may share data but preserve coverage metadata. |
| Smooth alias attribution control | Use \(\omega_L\) with the exact same centers as the bad law at each \(L\), and the same \(M\). | Verify identical conclusions from identical public data; a rejection is about the bound, not discontinuity. |
| Joint-mechanism positive control | Equation (8), including its individual interval sets. | Detect incorrect query handling and demonstrate the intended mechanism. |
| Nonprimitive and degenerate queries | Tiny deterministic cases for eq.(9), zero/negative determinants, repeated queries, and wrap-crossing arcs. | Catch false assumptions about resolution and phase wrapping. |
| Missing-prior and contract controls | Missing \(M\); radius \(1/2\); unsupported non-scalar or complex-amplitude input; invalid label denominator; deliberate resource-cap stop. | Ensure honest refusal and failure classifications. |

No repeated seeds or significance tests are appropriate for these exact assertions. Use a fixed seed only if supplementary mixed-coordinate correctness fixtures are generated pseudorandomly, and record it. These must remain bounded correctness checks, not additional empirical evidence about practical prevalence.

### 5.4 Metrics and thresholds

Primary outputs are exact logical decisions and verifiable witness objects:

- Whether the bounded regular class is excluded.
- Whether bounded-error observation sets overlap.
- The rational bracket for \(D\), and \(D_{\rm individual}\).
- A compatible curvature or ambiguity observation when one exists.
- Correct handling of all in-class controls, aliases, equalities, and resource failures.
- Completed, failed, invalid, and unrun rows, with reasons.

**Zero known false exclusions** is mandatory on every mathematically certified feasible control. A single reproducible false exclusion is an implementation validity failure, not an acceptable small error rate. Correct finite tests do not prove the implementation universally correct; the exact algorithm proof and independent oracle provide the additional evidence.

For secondary characterization, report the increase in worst-case tolerable angular error,

\[
G(M,L)=\frac{D(M,L)-D_{\rm individual}(M,L)}{2}.
\]

A gain of at least **1/64 turn** is the predeclared "material normalized gain" threshold. It means tolerating an extra \(5.625^\circ\) phase error using the same observations. A phase shift of that size can change an ideal optimally oriented Ramsey probability by \(\sin(\pi/64)\approx0.0491\). This translates the chosen dimensionless threshold; it is not a demonstrated hardware precision or application requirement.

Propagate the rational bracket for \(D\) to a bracket for \(G\), also using the analytic bound \(D\ge D_{\rm individual}\). Declare the secondary threshold met only if the lower gain bound is at least \(1/64\), and unmet if the upper gain bound is below \(1/64\). Otherwise report **threshold unresolved at the prescribed reporting precision**. Do not classify a straddling bracket using a rounded midpoint or spend an unplanned search budget to force a label.

The study's primary no-go conclusion does not depend on achieving this gain. The designed \(L=1,M=1\) control has \(G=7/120\), so reproducing it alone does not establish general utility. Report the full predeclared matrix, including zero gains and ambiguous regimes. Do not claim improvement over the exact strong reference: it solves the same feasible-set problem.

Record total elapsed computation, preprocessing, interval candidates, peak retained intervals, maximum integer bit lengths, peak RSS, and output size. Runtime is a feasibility measurement, not a promised speedup. A method that uses a different prior or receives extra data is not a fair comparator.

Measure total case wall time from the parent immediately before worker launch through result serialization and mandatory verification; this cold-start total includes interpreter/import overhead. Also record parsing/generation, core solve, verification, and writing durations as diagnostic components without double-counting them in the total. If a warmed core time is useful, label it separately and debit its extra run; it cannot replace the cold-start total. There is no accelerator synchronization step because the specified backend is CPU-only exact arithmetic. One execution per deterministic row is sufficient for correctness; do not make a speed-superiority claim from these timings.

### 5.5 Handling failures and exclusions

All rows remain in the denominator of coverage reporting. Distinguish unsupported inputs, mathematical ambiguity, resource stops, implementation defects, and successful exclusions. A timeout is never evidence that \(F\) is empty.

If a null control fails, stop comparative interpretation, preserve raw data, and repair or classify the experiment invalid. If the simple baseline matches all useful conclusions, report no demonstrated benefit from joint processing in the tested region. If that contradicts the exact positive control, it is a bug or interpretation defect, not a scientific negative.

The no-go is analytic and remains valid even if the implementation fails. A failed implementation cannot substantiate claims about the diagnostic's reliability or measured practicality.

## 6. Environment, resources, and execution commands

### 6.1 Inspect the actual Mac first

The M3 Max designation does not specify RAM, free disk, macOS, Python architecture, or thermal/power state. Record the actual values before allocating work.

Capture, using ordinary read-only commands and Python:

~~~bash
uname -m
sw_vers
system_profiler SPHardwareDataType
sysctl -n hw.memsize
vm_stat
df -h .
python3 --version
python3 -c "import platform, sys; print(platform.platform()); print(platform.machine()); print(sys.executable)"
~~~

Record whether the process is native arm64 or translated. Do not silently switch interpreters mid-study. No acceleration or floating-point capability is assumed; the actual mathematical backend is CPython arbitrary-precision integers and Fraction.

If the target Mac is unavailable, finish the derivations, code, configuration, static review, and instructions that do not require that resource. Mark Mac execution and resource calibration blocked. Do not present runs on another host as M3 validation. The small planning probes described later were run on Linux and provide no M3 performance evidence.

### 6.2 Hard cumulative caps

These are planning ceilings, not measured performance predictions. Expected scientific computation is seconds to minutes for small cases, potentially longer for large interval unions; calibration must establish the feasible scale.

| Resource | Hard ceiling |
|---|---:|
| Experimental computation, all stages and methods combined | **60 minutes** |
| Stage-0-to-report active work | **8 hours**; planning estimate **5–8 hours** including implementation, analysis, bounded repairs, and computation |
| Charged solver, reference, baseline, and verification invocations combined | **2,000**, with the counting rule below |
| Generated interval candidates | **2,000,000 per call**, **20,000,000 cumulatively** |
| Retained intervals | **100,000 per call** |
| Wall time of one solver invocation | **60 seconds** |
| Peak combined parent-plus-worker RSS | **min(1 GiB, one eighth of measured physical RAM)** |
| Total task-created files, including failed runs and temporary outputs | **500 MiB** |
| Network/dependency setup troubleshooting | **15 minutes** within the active-work cap |
| Repairs after the first complete correctness pass | **Two bounded repair attempts**, at most **45 minutes active work total**, drawn from the 8-hour cap and stage computation allowances |
| Protected final analysis/report reserve | **90 minutes active work** and **8 minutes computation**, both within the totals above |

Do not assume that a successful top-level process stayed within memory limits. Check native process RSS. On macOS, Python's resource.ru_maxrss is in bytes; on Linux it is commonly KiB. Record the platform-specific conversion and raw value. With at most one worker, monitor combined parent-plus-worker RSS and retain each process's high-water mark; label a sampled combined peak as sampled rather than confusing it with the sum of noncontemporaneous maxima. A subprocess per measured case makes worker peak memory meaningful. Python allocation tracking alone omits non-Python/native memory and is not a sufficient RSS measurement.

The parent owns the timeout and ledger. It must save a failure record even if it terminates a worker. The per-invocation timeout covers that invocation's parsing, generation, solve, and verification work as applicable; starting a new worker does not reset a case or stage budget. The verifier follows the same complete-branch and resource rules as the main solver, even when its reversed query order creates more intervals.

Check disk before and during runs. Do not expand the artifact cap by deleting unfavorable results; preserve compact records and avoid excessive traces from the start.

Every exploratory run, reference solve, verification replay, failed attempt, timing repeat, and repair contributes to the cumulative ledger. Restarting a process, changing stages, choosing the fallback, or revising the code does not reset it.

**Invocation accounting:** charge one invocation for each executed complete interval-feasibility solve or tiny exhaustive reference solve, including solves inside boundary calculations, tests, and verifiers. Charge one invocation for each complete closed-form individual-baseline batch over one observation set, and one for a standalone direct witness-verification batch that performs no interval solve. A wrapper is not charged a second time on top of its already charged internal solve; if a verifier performs two full solves, charge both. Record every primitive residual check, generated branch, and elapsed task separately even when it does not create another invocation. All time and memory remain charged.

Use the following full-matrix planning count to prevent accidental omission of controls or verification:

| Stage-4 work | Planned maximum charged invocations |
|---|---:|
| Boundary calculations | 510: ten zero-boundary configurations need one call; twenty-five others need at most twenty |
| Exact bounded-noise decisions | 175 |
| Closed-form individual-baseline batches | 175 |
| Regular null controls | 70 |
| Smooth alias attribution controls | 35 |
| Main certificate verification | 315: 175 noise decisions, 70 nulls, 35 aliases, and 35 boundary records |
| Additional fixed main fixtures and margin | 120 |
| **Stage-4 allocation** | **1,400** |

The zero-boundary count follows from the already verified alias criterion at \(\mu\ge1/3\); a failure of those cases is a validity problem, not permission for a larger allocation. A boundary-record verification uses at most one replay of its infeasible lower endpoint, plus direct checks of the upper witness. A compatible or ambiguous result is checked directly; an exclusion is replayed completely. Verify each main result during stage 4. Build **results/essential_certificates.json** with at most 25 already verified records for the final recheck: the exact pilot/oracles, finite-grid aliases across included levels, representative exclusions and ambiguities, and the largest observed interval cases. The final command rechecks this manifest, not every certificate again.

### 6.3 Scaling and calibration policy

The main hidden cost is the number of winding intervals, not the number of scalar parameters. On a full bound, one area \(a\) may admit approximately \(2M|a|+2\) integer branches. Processing surviving intervals can fragment the union. Sorting and exact rational allocations also cost memory.

At the largest planned \(M=2048\), a coarse unit-area query has on the order of four thousand candidate wraps. The exact cost depends on the radius and prior pruning; this is a scaling observation, not a benchmark. Stream branch products in the tiny reference. Never materialize a large Cartesian product.

As a conservative memory planning estimate, allow up to roughly **1 KiB per retained interval including endpoints, objects, and temporary sorting storage**, and account for multiple simultaneous lists. Measure actual RSS; do not rely on this estimate as a guarantee.

Calibrate on representative small and medium instances, including one with many viable coarse wraps. Measure parsing/generation, solver, verification, and writing separately, and retain a total. Use the measured candidate rate and actual interval counts to project remaining work, with a **threefold safety factor**.

Before freezing the main protocol, choose the largest authorized matrix whose projected total fits both the remaining stage cap and the global caps. The only predeclared matrix reduction is to remove **all \(L=6\) main configurations**, leaving 28 bound configurations and 140 noise cells. Keep the exact boundary cases and all families. This retains the scientific question while reducing alias count.

The freeze command must save an enumerated workload manifest and a conservative count of all planned solves, baseline batches, controls, and mandatory verification, plus the measured cost projection. Refuse to freeze a schedule that cannot fit the remaining invocation, time, candidate, memory, or disk limits. Use the explicit full-matrix allocation above as a cross-check, not as permission to omit a required family.

After freezing, a budget stop yields explicitly unrun or incomplete rows; do not silently change the matrix. If even the reduced matrix is too costly, complete the exact oracle and a minimal reproducible diagnostic, and report practical infeasibility of the planned coverage. Do not replace exact arithmetic, enlarge tolerances, or discard winding branches.

Reserve the entire final-stage allocation before starting confirmatory work: at least 90 minutes of the eight-hour active-work budget and eight minutes of the compute budget. Stop launching implementation changes, repairs, or experiments when their projected cost would consume either reserve. The exact no-go derivation can still support a completed analytic outcome.

### 6.4 Intended command interface

Create and document these commands. They are **proposed interfaces to implement**, not commands verified during planning.

~~~bash
python3 -m venv .venv
.venv/bin/python experiments.py inspect --config config.json --out results
.venv/bin/python -m unittest -v test_phase_audit
.venv/bin/python experiments.py smoke --config config.json --out results
.venv/bin/python experiments.py calibrate --config config.json --out results
.venv/bin/python experiments.py pilot --config config.json --out results
.venv/bin/python experiments.py freeze --config config.json --out results
.venv/bin/python experiments.py main --protocol results/frozen_protocol.json --out results
.venv/bin/python verify_certificate.py --manifest results/essential_certificates.json --ledger results/ledger.jsonl
.venv/bin/python report.py --results results --output results/RESEARCH_REPORT.md
~~~

No pip installation is required for the core. If no suitable Python is available, use an already approved standard installation mechanism within the setup allowance. Record the exact version actually used. Do not require optional plotting packages to complete the study.

The inspect command initializes or resumes the shared ledger before any test solve. The unittest suite must use **config.json** and **results/ledger.jsonl**, debit every test/reference/verification invocation through the same budget object, and write a passed/failed gate record containing code hashes, configuration hash, counters, and timestamp. Later commands must refuse a stale pass after a material code or scientific-configuration change; repeat only the affected checks within the remaining allowance and record the renewed gate. Initialize active-work accounting at the start of stage 0; bootstrap read-only inspection before the script exists can be recorded as a first explicit ledger entry. Do not run an unmetered duplicate test suite.

Commands must refuse to overwrite a prior run's evidence. Give runs unique deterministic revision/sequence IDs and append to the shared ledger. The report command reads saved results only; it must not silently rerun a failed experiment.

Make the smoke command create and analyze the equation (8) fixture and verify a smooth alias at one level. The main command must require a frozen protocol and passed gates.

### 6.5 Fixed configuration content

Store scientific rationals as strings. Include at least the following values; additional logging paths are engineering choices.

~~~json
{
  "schema_version": 1,
  "study": "finite_dyadic_phase_certifiability",
  "phase_units": "turns",
  "arithmetic": "fractions.Fraction",
  "main_levels": [0, 1, 2, 4, 6],
  "mu_values": ["0", "1/12", "1/6", "1/4", "1021/3072", "1/3", "1/2"],
  "noise_radii": ["0", "1/1024", "1/256", "1/64", "1/16"],
  "boundary_width": "1/1048576",
  "boundary_max_calls": 20,
  "material_noise_gain": "1/64",
  "max_total_compute_seconds": 3600,
  "max_active_work_seconds": 28800,
  "reserved_report_active_seconds": 5400,
  "reserved_report_compute_seconds": 480,
  "max_total_feasibility_calls": 2000,
  "max_candidates_per_call": 2000000,
  "max_candidates_total": 20000000,
  "max_intervals_per_call": 100000,
  "max_seconds_per_call": 60,
  "max_rss_bytes": 1073741824,
  "max_output_bytes": 524288000,
  "fallback": "omit_all_level_6_rows_before_freeze"
}
~~~

Here \(1021/3072=1/3-1/1024\). Apply the additional measured-RAM fraction to the RSS setting during inspection and store the effective cap. Scientific settings stay fixed except for the declared fallback.

## 7. Evidence gates and stop rules

Do not build later-stage infrastructure before the earlier uncertainty is resolved. A justified early stop is a completed research outcome when accompanied by its derivation or diagnostic evidence.

The computation ceilings below sum to 60 minutes; invocation ceilings sum to 2,000. They include references, verifiers, failures, and repair reruns. Unused early allowances may be transferred only by recording the change before the later run, while preserving the eight-minute final reserve and every global limit.

| Stage | Question and prerequisites | Minimum work and evidence | Compute / invocation cap | Decision and next action |
|---|---|---|---:|---|
| **0. Constructive feasibility** | Is the selected problem well posed and executable? Start from this brief and available inputs. | Inspect host; validate the finite-grid proof and bounded interval construction; identify input availability; write scope and source status. | **2 min / 20** | Continue if exact finite-grid alias and bounded finite enumeration are understood. If a material derivation fails, perform one focused repair/review; otherwise stop with the defect. Missing Mac blocks execution claims, not analytic work. |
| **1. Tiny correctness** | Can exact computations implement the mathematical contract? Requires stage 0. | Core rational functions; main interval solver; independent tiny branch reference; exact oracles, zero/negative areas, wrap boundaries, singleton endpoints, deliberate cap stop; compact tests. | **10 min / 200** | Continue only with exact agreement and verified witnesses. Any false exclusion blocks interpretation. Repair ordinary bugs within the shared allowance; unresolved defects imply invalid experiment. |
| **2. Resource calibration** | Will representative interval growth fit the actual Mac and remaining budget? Requires passing tiny checks. | Measure candidate counts, RSS, bit lengths, and total costs on small/medium/wrap-rich cases. Store projections with the threefold factor. | **5 min / 100** | Select full or predeclared reduced matrix before freeze. If neither fits, stop large allocations and report measured infeasibility with the useful analytic/tiny results. |
| **3. Mechanism pilot** | Does joint consistency detect the intended effect, and do aliases/ambiguity behave correctly? Requires stages 1–2. | Equation (8); \(D(1,1)=1/5\); a regular null; identical bad/smooth alias inputs; a midpoint ambiguity case; missing-bound refusal. | **10 min / 100** | Continue if all exact mechanisms behave as proved. An uncertain numerical bracket must be resolved by exact feasibility at the decision radius, not tuned. An unexplained contradiction is a bug/invalidity gate. |
| **4. Locked comparison** | What is the finite-bound/noise boundary and added value across the prescribed coverage? Requires passed pilot and frozen protocol. | Run 35/175 or reduced 28/140 matrix, controls, baseline, witness validation, and complete failure accounting. | **25 min / 1,400** | Complete within caps. Stop launching work before the reserve is threatened. Report unrun rows. No extra favorable regimes, new noise model, or alternative algorithm without a documented exploratory revision. |
| **5. Diagnosis and report** | What conclusion is supported, including negative/inconclusive outcomes? Always reached. | Verify essential saved certificates; generate tables; reconcile sources and budgets; report scope, limitations, failed runs, and one justified next step. | **8 min / 180** | Deliver the completed research record for the stages actually reached. Never invent unperformed verification or fill missing rows with inferred measurements. |

Certificate checking performed as part of a main row counts against stage 4; stage 5 is for essential final rechecks and report generation. This distinction prevents double-budgeting.

Implementation and analysis effort are separate from compute. Plan approximately 30–45 minutes for feasibility/source reconciliation, 1.5–2.5 hours for the small implementation and independent reference, 30–60 minutes for calibration/pilot/setup of the fixed evaluation, and 1–1.5 hours for analysis/reporting, plus bounded repairs. These are estimates, not measurements. Keep total active work within eight hours.

**Noisy-pilot rule:** this core pilot is exact and has known answers; it should not be described as statistically inconclusive. If a future supplied interval is too wide or the midpoint certificate exists, that is an information limitation, not a failed stochastic test. Do not keep narrowing error bars without an independent measurement justification.

**Analytic-stop rule:** if an indispensable input bound or scalar-phase contract cannot be supplied for a proposed real use case, deliver the explicit finite-grid alias and explain why the broader claim cannot be tested. The synthetic bounded diagnostic remains authorized if the resource gates pass; do not build a physical benchmark simply to avoid an analytic conclusion.

## 8. Scientific interpretation and decision criteria

### 8.1 What would support the selected result

A supported **operational characterization** consists of:

- A correct finite-grid indistinguishability derivation and explicit alias.
- Sound and complete rational feasibility on the stated scalar, one-parameter input contract.
- Agreement with the independent tiny reference and exact oracles.
- Verified ambiguity witnesses when the bounded observation sets overlap.
- Correct refusal to attribute a bounded-class rejection to discontinuity.
- A complete, honestly scoped table of the tested finite-bound/noise regimes.
- Measured feasibility on the actual target if Mac execution occurred.

These establish a useful limitation and a conditional model-specification diagnostic. They do not establish a newly discovered mathematical theory or an improvement over standard exact set-membership estimation.

A material secondary gain can support the statement that requiring a shared parameter across the selected scales improves robustness over independent per-loop feasibility within this synthetic family. It cannot support superiority over a properly implemented joint wrapped estimator, because that estimator is the strong mechanism used here.

### 8.2 Outcome categories

| Outcome | Required evidence | Wording and consequence |
|---|---|---|
| **Supported contribution** | Valid derivations and implementation, complete relevant checks, nonvacuous certificates in the declared regime; any comparative claim satisfies the fair-input protocol. | A supported scoped certifiability boundary/diagnostic. Historical novelty remains unestablished. |
| **Informative negative result** | Exact alias proves impossibility without a bound, or valid comparison shows no added useful regime beyond a simpler method under the tested conditions. | Reject or narrow the specific proposed capability, not quantum verification generally. |
| **Practical infeasibility** | Actual measured cost or supported scaling exceeds caps while preserving the meaningful question. | Identify which coverage or use case is infeasible and preserve the smaller verified result. |
| **Inconclusive evidence** | Missing indispensable input/hardware, incomplete coverage, or unresolved access/budget issue prevents the intended conclusion. | State exactly what remains unperformed or undecided. |
| **Invalid experiment** | Incorrect implementation/reference, unsupported error contract, inconsistent units, selective reruns, or other defect invalidates comparison. | Preserve the evidence and withhold affected scientific claims. |

A compatible finite-data result is not failure of the diagnostic. It is often the mathematically required answer. A timeout, bug, or absence of comparative gain is not a scientific refutation.

### 8.3 Physical deployment remains a separate gate

For real scalar-phase observations, \(M\) must come from a fixed model specification, calibrated canonical coordinates, or another independently justified bound. It cannot be increased until the same observations fit and then called independently validated.

The phase interval must include every relevant error source. For example, if actual labels differ by vectors bounded by \(\delta_g,\delta_h\), then under the candidate class

\[
|a(g+\Delta g,h+\Delta h)-a(g,h)|
\le \|g\|\delta_h+\|h\|\delta_g+\delta_g\delta_h.
\]

Multiplying by \(M\) gives a conservative angular error in turns before reducing circle distance. Measurement, calibration, and independently proved model-approximation errors can be added by the triangle inequality. Cap the resulting circular uncertainty at \(1/2\), at which point the observation may be uninformative.

This supplies a conservative null-model tolerance under \(\mathcal R_M\); it does not bound calibration effects for an arbitrary alternative. Applying the symmetric factor-of-two discrimination boundary requires independently valid error bounds under both hypotheses. The discontinuous adversary need not obey the null-model calibration bound.

This formula is explanatory only: the current study does not measure \(\delta_g,\delta_h\), prove scalar closure, or derive approximate-cocycle stability. Small cocycle residuals on sampled triples do not establish an unseen uniform phase error.

Likewise, a noisy complex amplitude is not automatically a scalar unit phase. Normalizing an amplitude of modulus below one can conceal leakage. No conversion from arbitrary complex samples or ancilla shot counts is included. Such a converter would need an independently validated simultaneous confidence/error bound.

If \(g\mapsto U_g\) is strongly continuous and commutators are scalar, the scalar phase is jointly continuous: products of strongly continuous bounded unitaries are strongly continuous, and a fixed unit-vector expectation extracts the phase. The exceptional weakly regular examples therefore do not expose a loophole in ordinary continuous quantum controls.

For finite \(D\times D\) unitaries satisfying \(UV=zVU\), determinants imply \(z^D=1\). Irrational scalar phase identities are not exact finite-matrix identities. No finite-matrix approximation is used in this PoC, so no oscillator truncation claim follows.

A worthwhile subsequent physical study requires **one actual candidate model** with independently supported \(M\), exact control units, and justified scalar phase intervals. Without that input, additional hardware-oriented simulation would not repair the identifiability issue.

## 9. Required deliverables and reproducibility

Produce only artifacts justified by the stages reached. When implementation occurs, deliver runnable code, configuration, tests, and usage instructions. When experiments occur, deliver their raw results and witnesses. An analytic stop must include a verifiable derivation and explicit counterexample/interpolant.

The final research package must include:

1. **README.md** with actual commands, tested Python/OS/architecture, target-machine status, and a one-paragraph scope statement.
2. **THEORY.md** with the proofs used in this brief and any material corrections discovered during execution.
3. The small implementation and independent reference.
4. **results/environment.json**, recording actual hardware, RAM, free disk, interpreter, process architecture, date/time, and relevant execution conditions.
5. **results/frozen_protocol.json**, configuration hash, source hashes when inputs are available, and code commit or file hashes.
6. **results/ledger.jsonl**, append-only records of every run, failed attempt, verification, time/candidate budget debit, and reason for stopping.
7. **results/results.jsonl** and **results/summary.csv**, containing exact rational values as well as labeled display decimals.
8. **results/certificates/** for the alias, known-answer pilot, representative exclusion sets, all generated ambiguity witnesses, and required control witnesses.
9. **results/RESEARCH_REPORT.md**, regenerated from saved results, with a self-contained answer and the outcome classification.
10. An explicit list of blocked or unperformed work and material deviations.

Each run record should include configuration ID, source model identity retained only for evaluation, public-input hash, query list, bound and its provenance, error contract, method, arithmetic, code revision, all counters, timing boundaries, RSS, result status, witness path, verification status, and any exception. Record failed runs before attempting repair.

Report essential tables rather than requiring a plotting dependency:

- No-go example and its exact smooth alias.
- \(D\), \(D_{\rm individual}\), and noise-tolerance gain across the 35 or 28 configurations.
- All 175 or 140 robust separation/ambiguity decisions.
- Control results, failures, and incomplete coverage.
- Total resource use by stage and method, including preprocessing and verification.

If a plot is added, generate it from saved exact data with a standard plotting library and record its version. It is optional and must not be a new dependency gate. Do not interpolate untested regions in a way that implies a proof or observed performance.

The report must distinguish mathematical derivation, source-reported fact, planning probe, target-machine experiment, and untested hypothesis. It must say that the method is a standard one-parameter wrapped feasible-set computation. It must not call it a newly validated universal quantum verifier.

## 10. Work already performed during planning

The following activities occurred before your execution. They inform the design but do not replace your verification.

### 10.1 Document and source work

The entire supplied mathematical paper and application request were read. Key PDF formulas were visually checked. Three independent agent assignments examined mathematics, prior art, and experimental design. Their conclusions were reconciled by the research lead; this was internal reasoning review, not external peer review.

The primary source locations listed in the references were opened and inspected. The investigation covered separate/joint continuity, quadratic pseudolattices, noncommutative solenoids, phase aliasing/unwrapping, bounded-error set estimation, symplectic quantum reduction, oscillator displacement truncation, and robust phase estimation. Search was not exhaustive, and no claim of global novelty is warranted.

### 10.2 Small exact computational probes

These probes were limited design checks, not the completed PoC.

| Probe | Actually performed result | What it does not establish |
|---|---|---|
| Finite-grid alias checks | 1,400 exact Fraction comparisons across \(L=0,\ldots,6\) all agreed. Generator: Python random.Random(1729); 200 pairs per \(L\); independent denominator depths chosen uniformly from 0 through \(L\); four numerators in \([-4,4]\); common-denominator bad phase compared with \((4^L/3)\det(g,h)\) modulo one. | Universal proof, physical realism, Mac performance, or correctness of the future implementation. |
| Independent tiny minimax calculation by the research lead | Rational vertex enumeration over winding choices \(-2,\ldots,2\), for \(M=1\), returned \(D(1,0)=0\), \(D(1,1)=1/5\) at \(8/15\), and \(D(1,2)=5/17\) at \(32/51\). | Main-matrix results or performance on the target. |
| Separate pilot arithmetic check | An independent Fraction branch enumeration confirmed the interval sets in eq.(8), the singleton feasible set at radius \(1/5\), and emptiness at \(199/1000\). | A statistical evaluation or deployment certificate. |
| Source example sanity checks | Integer Pell recurrence and high-precision values were checked for the paper's shrinking quadratic example. | A reason to add this separate family to the main PoC. |

The research-lead probe ran under Python 3.12.14 on Linux x86_64; its roughly 0.07-second elapsed time is only a record of that tiny planning calculation. **It is not an M3 Max estimate.** The proof, rather than these finite checks, establishes aliasing.

No full diagnostic repository, locked matrix, target-machine calibration, physical experiment, or hardware error budget was completed during planning.

## 11. Review findings and their disposition

Three distinct review passes were conducted: mathematical/novelty review, experimental review, and execution review. The following material issues were repaired in the design.

| Review finding or stress case | Disposition |
|---|---|
| The exact source theorem was being treated as if it supplied a finite noisy verifier. | Added explicit finite-grid alias proof and narrowed the project to bounded finite-data consistency. |
| Refinement rigidity was described as newly derived after the paper. | Corrected attribution to [S1], Corollary 4.3. |
| A novel method could merely repackage standard estimation. | Explicitly identifies interval feasibility/phase unwrapping as prior mechanisms and makes limitation/characterization the contribution. |
| An oscillator benchmark could rediscover known finite-Fock errors. | Removed it from execution; recorded close prior art and required a separate deployment justification. |
| A real kernel direction from an arbitrary compression need not be an accessible group label. | Removed arbitrary compression witness construction. A dense countable group may intersect an irrational kernel only at zero. |
| A radial magnitude bound was being confused with the exact chord separation. | Replaced with eq.(7); all decisions use rational circle distance. |
| Exact-center rejection was being confused with robust separation under errors on both hypotheses. | Added the factor of two in eq.(4), closed-boundary handling, and explicit midpoint witnesses. |
| A single-loop baseline could receive fewer observations. | It receives all observations and the same bound/error budget; only the shared-parameter constraint is removed. |
| The positive control could already be caught by the finest loop. | Uses the perturbed observation in eq.(8), with exact independent feasible sets. |
| Finite compatibility could be mislabeled as continuum regularity. | Separate status names and explicit scope; smooth aliases share the same public input. |
| Rejection could falsely be attributed to discontinuity. | Added regular out-of-bound alias controls and prohibits causal attribution from these data alone. |
| Finite precision, wrap boundaries, or dropped branches could create false certificates. | Exact rational arithmetic, closed intervals, independent branch oracle, and inconclusive resource status. |
| A missing source bundle could be mistaken for a reproduced experiment. | Identified missing files and separated reported counts from actually performed probes. |
| A small pilot or a properly implemented baseline could remove the gain. | Exact pilot answers act as validity checks; zero gain elsewhere is reportable. No requirement for superiority over the strong same-problem reference. |
| Preprocessing or interval fragmentation could dominate. | Counts generation, verification, allocations, and file writing; uses representative calibration and cumulative caps. |
| RAM, backend, or runtime assumptions could be invented from the chip name. | Requires actual Mac inspection and reports hosted planning probes separately. |
| Essential sources or data could be inaccessible. | All core formulas and deterministic generators are supplied here; record unverified source provenance if the attachments are absent. No external dataset or repository is essential. |
| The main broad application hypothesis is false. | The exact alias supplies a valid negative result; the bounded diagnostic remains a scoped alternative. |
| Only favorable synthetic cases demonstrate an effect. | Includes exact aliases, nulls, no-information cases, nonprimitive queries, and full matrix reporting; forbids real-world generalization. |
| The theorem's frequency map is not computationally recoverable from arbitrary weak-regularity data. | The selected input class is explicitly parametrized; no hidden oracle for the global frequency map is assumed. |
| Runtime/memory exceed estimates. | Predeclared coverage reduction before freeze, explicit resource stops afterward, preserved analytic outcome, and reserved reporting time. |
| The literal test sequence and baseline implementation could evade the ledger or overrun the invocation cap. | Inspection initializes accounting before tests; the baseline uses its exact closed form; stage-4 calls and certificate replays are explicitly allocated; final rechecks use a bounded manifest. |
| The active-work budget could expire before the report was written. | Protected 90-minute active-work and eight-minute compute reserves apply before further implementation, repairs, or experiments. |
| Addition/sign identities, derived radii, or gain brackets could be implemented incorrectly. | Corrected conjugation/additivity rules; cap derived doubled radii at a half turn; preserve unresolved secondary gain thresholds. |
| A calibration bound valid under the null could be transferred to the discontinuous alternative. | Explicitly requires independent error validity under both hypotheses before using the symmetric discrimination bound. |

Residual risks remain: no independently calibrated physical \(M\) or scalar-phase error model was supplied; the proposed operational characterization may already be known in equivalent terminology; and actual M3 performance has not been measured. These are stated limitations, not hidden assumptions. No known fixable issue from these reviews is intentionally left to invalidate the selected study.

## 12. Supporting mathematical limits outside the implemented core

These points explain the narrowing and prevent overinterpretation. Do not turn them into additional projects.

### 12.1 Quadratic refinement divisibility has no unrestricted noise margin

On the paper's exceptional group, all measured phase values are powers of a base phase

\[
z_k=\exp(2\pi i kA\theta),
\]

with integer determinants. Extendability to \(G/N\) requires \(N^2\mid k\). Since \(A\theta\) is irrational, the extendable base phases \(\exp(2\pi iN^2jA\theta)\) are dense on the circle. Approximating the base phase sufficiently closely approximates any finite set of its integer powers. Thus unrestricted noisy phase observations cannot robustly certify that exact divisibility obstruction.

This does not make the symbolic divisibility test invalid when the code specifies \(k\) exactly. It shows why symbolic and data-inferred certificates must not be conflated. The chosen dyadic study has an even stronger finite-grid obstruction: exact smooth agreement without noise.

### 12.2 Linear compression is a different, mostly standard problem

If a bilinear phase on dense \(G\subseteq\mathbb R^d\) must factor through a linear map \(P\) in the form

\[
\Omega=P^\mathsf T\Omega_{\rm red}P,
\]

then \(\ker P\subseteq\ker\Omega\) and \(r\ge\operatorname{rank}\Omega\). Conversely, that kernel condition lets the alternating form descend to the image of \(P\), with an extension to the reduced ambient space. Density removes nonzero integer-valued bilinear differences when equality is required for all labels.

This counts continuous phase-label coordinates, not qubits or Hilbert-space dimension. Producing an exactly accessible violating label also requires attention to the group: a real vector in \(\ker P\) need not belong to a countable \(G\). These facts do not create a new model-reduction method and are not required by the selected diagnostic.

## 13. References and source access notes

References use ordinary URLs and precise locations so this file can be used without the planning chat. Titles, dates, and versions identify the inspected sources. Avoid relying on auto-generated HTML manuscript dates when they differ from stable arXiv version dates.

**[S1] Supplied research paper.** Author unspecified; subtitle *An autonomous mathematical investigation*. *Regularity constraints on arithmetic with real exponents*. 27 October 2026. Attachment **arithmetic_phase_research(1).pdf**, 13 pages. Essential locations: §1, eqs.(1)–(3), p.1; Theorem 2.1 and proof, pp.2–3; Theorem 3.1, p.4; Proposition 3.3 and Theorem 4.1, p.5; Corollary 4.3 and eq.(11), p.6; quantitative limitations §6.3, p.8; reproduction statement §B, p.13. Original workspace location and hash are in §2.1. No public URL was supplied.

**[S2] Supplied application proposal and task specification.** Author unspecified. Attachment **Pasted text(20261027-204158).txt**, supplied 27 October 2026. In particular, problem §§4–10 describe refinement rigidity, the dyadic adversary, finite-noise limitations, and the proposed oscillator direction. Original workspace location and hash are in §2.1. Claims were evaluated rather than treated as verified external evidence.

**[R1] R. Beattie and H.-P. Butzmann.** *On the continuity of separately continuous bihomomorphisms*. arXiv:0805.2730v1, 18 May 2008. [Record](https://arxiv.org/abs/0805.2730), [full text](https://arxiv.org/html/0805.2730v1). Relevant locations: Theorem 3.3, Proposition 3.4, and surrounding examples. Establishes broader separate/joint continuity results under additional group hypotheses.

**[R2] Yu. I. Manin.** *Real Multiplication and Noncommutative Geometry (ein Alterstraum)*. arXiv:math/0202109v1, 12 February 2002. [Record](https://arxiv.org/abs/math/0202109), [PDF](https://arxiv.org/pdf/math/0202109). Relevant locations: §§1.1–1.2 for pseudolattices and real multiplication; §3.1, eqs.(3.1)–(3.5), PDF pp.34–35, for cocycles, skew pairings, and central extensions.

**[R3] Frédéric Latrémolière and Judith Packer.** *Noncommutative Solenoids*. arXiv:1110.6227v1, submitted 28 October 2011; published in *New York Journal of Mathematics* 24A (2018), 155–191. [Record](https://arxiv.org/abs/1110.6227), [PDF](https://arxiv.org/pdf/1110.6227), [journal record](https://nyjm.albany.edu/j/2018/24a-6.html). Relevant locations: Definition 1.1, PDF p.2, discrete topology; Theorems 2.1 and 2.3, multiplier descriptions; Theorem 3.7, PDF p.13, inductive limits of rotation algebras. Closest structural prior art for refinement-compatible twisted algebras.

**[R4] Luc Jaulin and Eric Walter.** *Set inversion via interval analysis for nonlinear bounded-error estimation*. *Automatica* 29(4), 1053–1064, 1993. [DOI](https://doi.org/10.1016/0005-1098(93)90106-4), [author-hosted full PDF](https://webperso.ensta.fr/jaulin/paper_automatica93.pdf). Relevant locations: §1, eqs.(1)–(4), pp.1053–1054, feasible-parameter formulation; §§4–5, interval set inversion and SIVIA. The present one-dimensional wrapped interval computation is an elementary specialization of bounded-error feasibility, not an implementation of their entire nonlinear algorithm.

**[R5] Robby G. McKilliam, Barry G. Quinn, I. Vaughan L. Clarkson, Bill Moran, and Badri N. Vellambi.** *Polynomial phase estimation by phase unwrapping*. arXiv:1211.0374v1, 2 November 2012. [Record](https://arxiv.org/abs/1211.0374), [PDF](https://arxiv.org/pdf/1211.0374). Relevant locations: §3, Theorem 1 and Corollary 1, PDF pp.5–6, exact aliases and identifiable regions; §4, eqs.(4)–(6), wrapped observations and least-squares unwrapping. Its statistical objective differs from the bounded-error feasibility objective here.

**[R6] Hendra I. Nurdin.** *Structures and Transformations for Model Reduction of Linear Quantum Stochastic Systems*. arXiv:1308.6062v1, 28 August 2013. [Record](https://arxiv.org/abs/1308.6062), [full text](https://arxiv.org/html/1308.6062v1). Relevant locations: §3.1, Theorem 2, physical realizability under conjugate-pair subsystem truncation; §5, Theorem 11, quasi-balanced error bounds under its stated hypotheses. Different objective from exact full-label phase factorization.

**[R7] Alfio Borzì and Guofeng Zhang.** *Symplectic \(\mathcal H_2\) Model Reduction for High-Dimensional Linear Quantum Systems*. arXiv:2605.07152v1, submitted 8 May 2026. [Record](https://arxiv.org/abs/2605.07152), [full text](https://arxiv.org/html/2605.07152v1). Relevant locations: §3.1, eqs.(12)–(18) and Lemma 3.1; §4, Q-IRKA; §8, reproducibility statement. The text promises MATLAB scripts at an author GitHub page; a pinned public implementation was not verified. This preprint is contextual prior art, not an execution dependency.

**[R8] Jan Provazník, Radim Filip, and Petr Marek.** *Taming numerical errors in simulations of continuous variable non-Gaussian state preparation*. *Scientific Reports* (2022). [DOI](https://doi.org/10.1038/s41598-022-19506-9), [arXiv:2202.07332v2 PDF](https://arxiv.org/pdf/2202.07332), version dated 7 October 2022. Relevant locations: §§4–6, displacement errors, truncated approximate matrix exponential, and validation; Fig.2 and Algorithm 1. Establishes close prior work that an oscillator truncation diagnostic would need to improve upon.

**[R9] Shelby Kimmel, Guang Hao Low, and Theodore J. Yoder.** *Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation*. *Physical Review A* 92, 062315 (2015). [DOI](https://doi.org/10.1103/PhysRevA.92.062315), [corrected arXiv:1502.02677v3 PDF](https://arxiv.org/pdf/1502.02677v3), 21 October 2021. Relevant locations: Theorem I.1 and §V, with the explicit errata notice on p.1. The correction changes details of the original error analysis; no claim here depends on its original constants. This is context for bounded-error phase estimation, not a substitute for the prior bound needed in this study.

### Search coverage and limits

The planning research inspected primary source content rather than relying solely on search snippets. It followed the supplied solenoid and quantum-reduction references to finite-stage multiplier structure, wrapped phase estimation, bounded-error set inversion, and oscillator truncation alternatives. Search terms included variants of "separately continuous bicharacter", "dyadic noncommutative solenoid", "finite rotation algebra", "phase aliasing identifiable region", "bounded-error set membership", "symplectic quantum model reduction", and "truncated displacement operator".

No claim is made that all equivalent formulations, older terminology, or relevant literature have been found. The main study does not depend on proving novelty. The source theorem's historical originality and the operational prevalence of the adversarial models remain unresolved.

## 14. Final instruction to the coding agent

Execute the authorized stages, inspect their results, and follow the decision rules. Resolve ordinary reversible implementation choices yourself. Preserve failed runs and unfavorable findings. Do not spend the remaining budget trying to rescue a preferred positive conclusion.

Your final answer must state which of these occurred: a verified conditional diagnostic with a scoped characterization, a supported limitation of the proposed application, measured infeasibility, inconclusive evidence, or an invalid attempted experiment. More than one may apply to distinct parts of the work.

The most important result may be that a finite-data tool cannot infer the claimed continuum property without an independent bound. Make that result explicit and reproducible. If the bounded alternative works, state exactly what it certifies and what additional evidence would be needed before using it on a physical quantum model.
