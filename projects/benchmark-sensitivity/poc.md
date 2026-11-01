# POC execution brief: what exact predictive neutrality adds to bounded reporting sensitivity

**To:** an autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning.  
**Research and protocol date:** 1 November 2026.  
**Target experimental machine:** one MacBook with an M3 Max; inspect its actual configuration.  
**Deliverable from you:** an independently checked mathematical limitation result and a small reproducible diagnostic, with a research report. This document is the execution instruction, not a claim that the PoC has already been completed.

## 1. Project decision and smallest defensible contribution

### 1.1 Execute this project

Investigate the following single question:

> **When the admissible reporting uncertainty consists of all proper common priors supported on an independently specified interval, can the supplied finite-horizon neutrality theorem change a justified robustness conclusion for an individual AI preference report, beyond the sharp bounded-selection baseline? If it cannot, quantify the extra reporting uncertainty required by its stronger, simultaneous statement.**

The planning audit has already derived a decisive limitation: for this class, the exact Bayes-factor range is attained by two fixed endpoint point masses. A universal neutral prior lies within that range. Thus it cannot reveal a report-wise vulnerability that the sharp baseline misses. Independently verify this derivation, test the executable formulas against independent arithmetic, and produce the finite-horizon diagnostic below.

**The intended R&D contribution is an application-specific limitation result and a precise explanation of the theorem's remaining content.** Do not market the endpoint formula as new robust Bayesian theory. Do not assume that the notebook's quantitative laws are historically original. A useful completed result can recommend against building the proposed theorem-based report classifier under this information contract.

The theorem still distinguishes two statements:

- For each mixed report, there exists some admissible prior that neutralizes that report.
- There exists one admissible prior that neutralizes every mixed report through a fixed horizon.

This brief derives an exact support threshold for the first statement and compares it with independently audited finite bounds for the second. Their large-horizon support costs are respectively \(\log N+O_a(1)\) and \(2\log N+O_a(1)\). This is an explanatory comparison, with no claim of historical novelty.

### 1.2 Why the project has been narrowed

The original proposal sought a useful evidence-quality assessment for selecting between two AI models, potentially evaluated with a frozen Chatbot Arena subset. Preserve that purpose by first determining whether the proposed additional machinery can affect the relevant decision at all.

The analytic obstruction applies to every data-generating distribution of reports once the likelihood and admissible class below are fixed. An archive benchmark cannot overturn it. Therefore **the authorized primary study is an analytic limitation study with a deterministic computational diagnostic; an Arena benchmark, Monte Carlo error-rate study, general-purpose prior optimizer, and large quadrature benchmark are not required.** Complete the analytic stop as a finished research outcome. Do not keep expanding the project in search of a positive classifier result.

This revision no longer evaluates observed AI-model rankings, the frequency of actual selective reporting, population error rates of model replacement, or the credibility of particular reporting-bias bounds in an organization. No independent reporting audit was supplied. All numerical sensitivity factors in this protocol are experimental settings.

The conclusion is limited to the **support-only class of all proper common priors**, including point masses. Independently justified restrictions on prior shape, moments, density, actual retention information, or a different observation design can change the problem. Do not silently add such restrictions to rescue novelty. Identify one such future direction only if the final evidence supports pursuing it.

### 1.3 Decision and evidentiary standard

The primary practical decision is whether to invest in a universal-neutrality layer as an improvement to ordinary support-bounded sensitivity for a single report.

The prospective positive claim to challenge is:

> There exists a report and a fixed admissible support class for which a universal-neutrality witness reveals sensitivity missed by the sharp common-prior Bayes-factor envelope.

The derivation in §3 rules this out under the stated assumptions. A supported negative result requires a correct proof and independently checked formulas, not merely zero differences in a sampled benchmark.

A substantive counterexample would need to satisfy the same likelihood, common-prior requirement, support-only class, and decision definition. A discrepancy caused by a different uncertainty class, an omitted likelihood term, or arithmetic failure is not a counterexample.

Use an illustrative evidence threshold \(K=10\) for readable decision tables. It means a tenfold likelihood update between the two specified hypotheses. It is a declared presentation convention, **not** an empirically justified replacement policy, a posterior probability, or a frequentist error guarantee. The main limitation proof holds for every \(K>1\), so success does not depend on this choice.

## 2. Inputs, provenance, and fixed scientific contract

### 2.1 Supplied documents

Read the notebook source, its outputs, and Appendix A before accepting a source-dependent certificate.

| Input | Original location in the planning workspace | Role and provenance |
|---|---|---|
| `Predictive_Neutrality_Research(1).ipynb` | `upload/Predictive_Neutrality_Research(1).ipynb` | Unrefereed research note titled *The quantitative cost of exact predictive neutrality*, dated 1 November 2026. Fifteen notebook cells, including Appendix A; no named author or publication identifier supplied. |
| `Pasted text(20261101-120033).txt` | `upload/Pasted text(20261101-120033).txt` | Original application and execution-brief requirements. |

Planning-input SHA-256 values:

- Notebook: `4283a312099779cbde0a135be9eb9d1c14c11e50b3706ea10f2922df3c96b998`.
- Request text: `506159fb0bf141b282c1b82b9267f7e400f980addc76eb858e2a08f60a871cec`.

For a new project, place any supplied copies in `inputs/` under their original filenames. Record actual hashes; investigate a mismatch rather than asserting it is the same source. This brief contains the mathematics needed for the selected diagnostic. If the original notebook is not transferred, complete the self-contained endpoint and exact-arithmetic work, but mark source-wide reproduction unavailable and do not claim to have inspected the attachment. If a source-dependent proof cannot be reconstructed and checked from this brief and the accessible mathematical references, execute the endpoint, directional, and exact \(N\le4\) core; label the dependent larger-horizon classifications `unverified_source_dependency` rather than certified. This branch does not prevent a completed primary limitation result.

The planning review read all fifteen cells, all mathematical sections and the appendix, the numerical code and displayed outputs, and the plotted figure. Source locations below use notebook section/equation labels and zero-based cell numbers because the notebook has no stable page numbering. [S0]

### 2.2 Objects and domains

Let \(Y=1\) mean A is preferred and \(Y=0\) mean B is preferred, **conditional on a comparison being decisive**, for one fixed pair, setting, and criterion. The two precise alternatives are

\[
H_A:p=\tfrac12+\delta,\qquad
H_B:p=\tfrac12-\delta,\qquad 0<\delta<\tfrac12.
\]

Define

\[
a=\log\frac{1/2+\delta}{1/2-\delta}>0,\qquad
v=e^a=\frac{1+2\delta}{1-2\delta},\qquad
\sigma(x)=\frac1{1+e^{-x}}.
\]

Use \(v\) for the odds multiplier and \(m=\lceil N/2\rceil\) for the quadrature/moment order; this avoids overloading the notebook's symbol \(r\).

For inclusion indicator \(S\), let \(\pi_y=\Pr(S=1\mid Y=y)\), with both probabilities positive, and

\[
t=\log(\pi_1/\pi_0).
\]

At a fixed \(t\),

\[
q(p,t):=\Pr(Y=1\mid S=1,p,t)
=\frac{p e^t}{1-p+p e^t}
=\sigma(\operatorname{logit}p+t).
\tag{A1}
\]

Therefore \(q_A(t)=\sigma(t+a)\), \(q_B(t)=\sigma(t-a)\). This is the binary specialization of a weighted-selection model, not a newly proposed selection mechanism. [S1, §1, Eq. (1.1)]

For a retained record of predefined length \(n\ge1\), \(k\in\{0,\ldots,n\}\), define the ordered-record kernel

\[
f_\pm(t;n,k)=q_\pm(t)^k[1-q_\pm(t)]^{n-k}
\]

and

\[
L_\pm(n,k;\mu)=\int f_\pm(t;n,k)\,d\mu(t),\qquad
\mathrm{BF}_\mu=L_+/L_-.
\tag{A2}
\]

For unordered counts, multiply both likelihoods by \({n\choose k}\); it cancels in the BF. It does **not** disappear when calculating total probability or total variation over counts.

The uncertainty class is

\[
\mathcal M_I=\{\mu:\mu\text{ is a proper probability measure},
\ \operatorname{supp}\mu\subseteq I=[L,U]\}.
\tag{A3}
\]

Here \(L,U\) are finite and fixed independently of the tally. The **same** \(\mu\) is used under both hypotheses, and \(T\sim\mu\) is drawn once per entire record. Conditional on \(T\), retained outcomes are iid. A point mass is a legitimate proper prior. Symmetric support does not require an even prior.

The diagnostic considers \(I=[-B,B]\), \([0,B]\), and \([-B,0]\), where \(B=\log\Gamma\) and \(\Gamma\ge1\). Thus \(\Gamma\) bounds a **retention-probability ratio**. It is not a bound on the odds of retention, and it is not the usual Rosenbaum treatment-assignment odds-ratio parameter. Publication-probability sensitivity is related prior art, but its empirical calibrations do not justify an AI-reporting bound. [S2, §2.2, Eq. (2.4)]

### 2.3 Sampling and information boundary

Equation (A1) at a fixed \(t\) does not, by itself, justify integrating over the same prior after every possible selection design.

The exact record likelihood (A2) is appropriate for a prospectively fixed number of retained observations under the specified conditional iid model. It can be generated directly using \(q_\pm(T)\), or by independent rejection sampling until the prescribed retained count is reached. In the latter interpretation, the assessor's information contract must be declared in advance, and any known discarded counts or stopping information must be used. Some other designs reduce to (A2) through a valid cancellation or ancillarity argument; the presence of stopping or excluded-count information does not universally invalidate a BF. Such a reduction requires an explicit checked derivation. An unsupported design is outside this diagnostic, not automatically a refutation of the theorem.

If a completed batch size \(M\) is fixed and the number retained \(n\) is observed, \(n\) can inform both \(H\) and \(T\). Writing \(\bar\pi_H(t)=p_H\pi_1(t)+(1-p_H)\pi_0(t)\), conditioning on \(n\) generally tilts the preselection prior by

\[
\Pr(n\mid M,H,t)\propto
\bar\pi_H(t)^n[1-\bar\pi_H(t)]^{M-n}.
\]

The resulting \(\mu(t\mid n,H)\) need not be common across hypotheses. Do not omit this information and reuse (A2). Known excluded counts, known acceptance rates, outcome-dependent stopping, and selection of the best report from several runs require the appropriate full likelihood.

This diagnostic does not simulate a completed-batch thinning design. It examines the specified likelihood algebra directly, so no hidden selection metadata is manufactured or discarded.

### 2.4 Scope boundaries to preserve

- A tie, abstention, or “both bad” outcome is not a binary A/B win. An actual data analysis must retain and describe these categories. The present claim is conditional on decisiveness.
- \(H_A,H_B\) do not exhaust possible win probabilities. A tie or another advantage can make both hypotheses wrong. A BF between them is not certification of a practically important advantage against every alternative.
- A fixed real reporting bias can be epistemically uncertain; a prior over it is an assumption, not observed evidence of random dishonesty.
- Repeated evaluators, related prompts, model changes, time variation in retention, and sampling without replacement from a finite archive can violate conditional iid assumptions. They cannot inherit the theorem silently.
- No allegation about the source archive's selective reporting is made.
- \(\mathrm{BF}=1\) preserves prior odds; it does not make posterior odds equal or establish equal quality.
- Reported preference is not evidence of medical effectiveness or model safety.
- The source's “every mixed record” result excludes \(k=0,n\), and its horizon must remain fixed. Its theorem does not supply an approximate-neutrality result.

## 3. Main analytic result: exact report-wise sensitivity

### 3.1 Derive and independently verify the sharp envelope

Write \(\operatorname{sp}(x)=\log(1+e^x)\). At a fixed \(t\),

\[
g(t;n,k):=\log\frac{f_+(t;n,k)}{f_-(t;n,k)}
=2ak+n\{\operatorname{sp}(t-a)-\operatorname{sp}(t+a)\}.
\tag{B1}
\]

Differentiate:

\[
\partial_t g=n\{\sigma(t-a)-\sigma(t+a)\}<0
\quad(a>0,\ n>0).
\tag{B2}
\]

For any \(\mu\in\mathcal M_I\), all kernels are strictly positive. With

\[
d\widetilde\mu(t)=
\frac{f_-(t;n,k)\,d\mu(t)}{\int f_-(s;n,k)\,d\mu(s)},
\]

we obtain

\[
\mathrm{BF}_\mu=\int e^{g(t;n,k)}\,d\widetilde\mu(t).
\]

A weighted mean lies between the minimum and maximum of its integrand. By monotonicity, and attainment by common point masses,

\[
\boxed{
\{\mathrm{BF}_\mu:\mu\in\mathcal M_{[L,U]}\}
=[e^{g(U;n,k)},e^{g(L;n,k)}].
}
\tag{B3}
\]

Continuity of \(g\) and point masses at intermediate \(t\) establish every intermediate value. Crucially, **\(\delta_U\) minimizes and \(\delta_L\) maximizes the BF for every tally**, independently of \(k\). The baseline does not need different worst-case endpoint priors chosen after seeing different reports.

This argument optimizes the **ratio with a common prior**. Separately maximizing the numerator and minimizing the denominator using different priors would solve the wrong problem.

This is an elementary derivation in this brief, situated in established robust Bayesian selection analysis. It is not a claimed new theorem in that literature. [S1; S3, §4.1, for related degenerate-prior extrema]

### 3.2 Consequence for the proposed assessment

If an admissible prior neutralizes every mixed record through \(N\), then for each such record

\[
g(U;n,k)\le0\le g(L;n,k).
\tag{B4}
\]

The sharp baseline already recognizes the possibility of no update. Any theorem-based additional warning that declares a robust endpoint conclusion invalid would contradict (B3) or use different assumptions.

Conversely, a failure to find a universal neutral prior is **not** evidence that a particular report is robust. Single-report neutralization can be possible at much smaller \(\Gamma\).

Do not construct a second classifier by copying the baseline and claim their numerical agreement is evidence of superiority or equivalence. Prove the implication, then test its ingredients with independent oracles.

### 3.3 Report-wise output semantics

Return the nominal log BF \(a(2k-n)\), the exact-class log interval \([g(U),g(L)]\), and one threshold label:

| Label | Mathematical definition at \(K=10\) |
|---|---|
| `robust_A` | \(g(U)\ge\log K\) |
| `robust_B` | \(g(L)\le-\log K\) |
| `assumption_sensitive` | Neither robust label applies, but at least one admissible prior reaches either threshold: \(g(L)\ge\log K\) or \(g(U)\le-\log K\) |
| `insufficient_at_threshold` | Every admissible prior stays strictly between the two evidence thresholds |
| `outside_scope` | The supplied sampling/likelihood contract is contradicted or indispensable information is missing |
| `numerical_boundary` | Finite precision cannot resolve a threshold comparison under the rules in §6 |

Separately report whether zero lies in the log interval. “Insufficient” and “assumption sensitive” answer different questions. Do not call every non-robust case a reporting-bias warning.

Threshold comparisons should use exact rational likelihoods where available; otherwise follow the numerical boundary rule. No empirical error-rate interpretation is attached to these labels.

### 3.4 Independent selection-sensitivity interpretation

For a known selected-population probability \(q\),

\[
p=\frac{q}{q+e^t(1-q)}
=\sigma(\operatorname{logit}q-t),
\]

so the sharp identified range given \(t\in[L,U]\) is

\[
p\in[\sigma(\operatorname{logit}q-U),
      \sigma(\operatorname{logit}q-L)].
\tag{B5}
\]

This is another view of ordinary bounded-selection sensitivity. Do not substitute the observed proportion \(k/n\) and call this a confidence interval. Sampling uncertainty would need separate treatment. No confidence-interval comparator is required for the selected analytic claim; comparing one with a BF threshold would introduce a different decision target.

## 4. What the supplied theorem actually establishes

Use the following formal definitions. The set of priors neutral through \(N\) is

\[
\mathcal N_N(a)=\{\mu:\mu\text{ is a proper probability measure on }\mathbb R,\,
L_+(n,k;\mu)=L_-(n,k;\mu)
\ \forall\,2\le n\le N,\ 1\le k<n\}.
\]

The source's support and discrimination optima are

\[
B_N(a)=\inf\{B\ge0:\exists\mu\in\mathcal N_N(a),\,
\operatorname{supp}\mu\subseteq[-B,B]\},
\]
\[
\Delta(\mu)=\int[\sigma(t+a)-\sigma(t-a)]\,d\mu(t),\qquad
M_N(a)=\sup_{\mu\in\mathcal N_N(a)}\Delta(\mu).
\]

The \(M_N\) supremum permits unbounded-support proper priors. It is neither the TV of one selected construction nor an optimum at a specified support radius.

### 4.1 Source results and their audit status

The planning audit found no essential proof defect in the notebook's main theorem or Appendix A. This was an adversarial mathematical review with small computational checks, not external peer review or formal proof verification. You must check the portions used in the final certificate yourself.

| Source result | Exact source location | Evidence and use here |
|---|---|---|
| A proper common prior can neutralize all mixed records through any fixed finite \(N\) | [S0] §1 and §4, Eq. (11); cells 1 and 4 | Constructive mathematical claim; use the proof, not numerical residuals, for exact existence. |
| Neutrality at horizon \(N\) is equivalent to neutrality through \(N\), and to moment identities | [S0] §2, Eqs. (4)–(6); cell 2 | Audited; supplies the finite support lower certificate. |
| \(B_N=2\log N+O_a(1)\), with finite lower and upper bounds | [S0] §1, Eqs. (1),(3), proofs §§3–4 | Audited for fixed \(a>0\); finite formulas are operational, asymptotics are explanatory only. |
| \(M_N=a/(2\log N)+O_a((\log N)^{-2})\) and finite TV bounds | [S0] §1 Eq. (2), §5; cells 1 and 5 | Audited; not a target optimization in this project. Check TV directly only for the tiny exact witnesses. |
| \(B_2=0\); \(B_3=B_4=b(a)\) and a unique two-atom optimizer | [S0] §6, Eq. (16); cell 6; \(N=2\) follows directly | Core positive control and exact finite support classification. |
| No fixed proper prior works at every horizon | [S0] §2, last paragraph | Audited; prevents extrapolating a fixed witness beyond its horizon. |
| General finite-offset matching | [S0] §7; cell 7 | Known-theory quadrature corollary; outside the two-hypothesis implementation. |
| Exact \(N=3,4\) discrimination optimum and two-to-three-atom transition | [S0] Appendix A, Eqs. (1),(7),(9); cell 14 | Cubic certificates audited and analytic priors independently checked. Do not implement this separate optimization. |
| Small floating-point BF residuals and source plots | [S0] §§8–9; cells 8–12 | Reported numerical observations; they do not prove exact computed-prior neutrality. |
| Appendix A grid-LP table | [S0] Appendix A.5 | Reported observation; the LP-generating code is absent and the LP was not rerun during planning. |

### 4.2 Exact small horizons

At \(N=2\), the only mixed count is \((2,1)\); \(\mu=\delta_0\) is neutral, so \(B_2=0\).

For \(N=3,4\), put

\[
A=\cosh a,\qquad
R=\frac{A+\sqrt{A^2+8}}2,\qquad
b=\operatorname{arcosh}R.
\tag{C1}
\]

Then \(B_3=B_4=b\), attained uniquely by

\[
\mu_{34}=\tfrac12(\delta_{-b}+\delta_b).
\tag{C2}
\]

A short proof to verify: symmetrization preserves neutrality and support radius. For an even prior, the constraints through four reduce to the expectation of the following function being zero:

\[
\frac{\sinh a\,[X^2-AX-2]}{2(A+X)^3},
\qquad X=\cosh t.
\tag{C3}
\]

It is negative for \(|t|<b\), zero at \(\pm b\), and positive beyond. This proves impossibility below \(b\) and verifies equal endpoint atoms at \(b\). At equality, positivity forces support onto the endpoints; the size-two mixed constraint forces equal weights, including for an originally nonsymmetric prior. [S0, §6]

For rational \(\delta\), \(v\) and \(A=(v+v^{-1})/2\) are rational. For rational \(\Gamma\ge1\), set \(X_\Gamma=(\Gamma+\Gamma^{-1})/2\). Then the exact support test is

\[
\Gamma\ge e^b
\quad\Longleftrightarrow\quad
X_\Gamma^2-A X_\Gamma-2\ge0.
\tag{C4}
\]

Use rational arithmetic for this classification. Do not round \(e^b\) and compare rounded values.

Exact neutrality has a discontinuous degenerate limit: as \(a\downarrow0\), \(b\to\operatorname{arcosh}2\), whereas at \(a=0\) the hypotheses are identical and no bias is needed. A lower bound for **exact** neutrality is therefore not a bound on approximate neutrality or practically material evidence. Keep the fixed-\(\delta\), finite-\(N\) conclusions separate from this limit.

### 4.3 Known bias direction rules out universal neutrality

Let \(d(t)=\sigma'(t+a)-\sigma'(t-a)\), the size-two mixed kernel difference. The logistic derivative is even and strictly decreases with positive distance from zero. Hence \(d(t)<0\) for \(t>0\), \(d(t)>0\) for \(t<0\), and \(d(0)=0\).

A neutral proper prior on \([0,B]\) must therefore be \(\delta_0\). But at \(n=3,k=1\), that prior gives BF \(e^{-a}\ne1\). Thus no such prior is neutral through \(N\ge3\). Reflecting gives the same result on \([-B,0]\). This is valid even without a finite upper bound.

This brief-derived exclusion is a required negative control. It says nothing about whether an individual majority report is robust. Do not manufacture two-sided uncertainty when an audit supplies the direction.

### 4.4 Quantifier comparison: individual versus simultaneous neutrality

For a mixed report \(0<k<n\), solving \(g(t;n,k)=0\) gives a unique solution

\[
t_0(k/n)=
\log\frac{\sinh(a k/n)}{\sinh(a(1-k/n))}.
\tag{C5}
\]

Therefore a neutral prior for that report exists on \([L,U]\) exactly when \(t_0\in[L,U]\); a point mass at \(t_0\) is sufficient. The map is increasing and satisfies \(t_0(1-q)=-t_0(q)\).

Over every mixed tally with \(n\le N\), the largest absolute value occurs at \(n=N,k=N-1\) or its complement. Consequently,

\[
B_{\mathrm{ind}}(N,a)
=\log\frac{\sinh(a(1-1/N))}{\sinh(a/N)}.
\tag{C6}
\]

This threshold permits a different neutralizing prior for different tallies. It is not the source's joint optimum \(B_N\). Since

\[
e^{B_{\mathrm{ind}}}
=N\frac{\sinh a}{a}-\cosh a+O_a(N^{-1}),
\]

its log-radius grows as \(\log N+O_a(1)\), while joint neutrality costs \(2\log N+O_a(1)\). Verify the algebra and monotonicity. These are brief-derived formulas, not claims of a novel published result.

Include this concrete example in the final report when the dependent certificates pass: at \(\delta=.05,N=8,\Gamma=10\), \(e^{B_{\mathrm{ind}}}\approx7.0353<10\), but the necessary joint threshold exceeds 10.9657. Every mixed tally through eight can therefore be neutralized separately within the class, while no single prior in that class neutralizes all of them. This does not create an extra report-wise warning beyond (B3).

### 4.5 Finite support exclusion: exact arithmetic, no grid optimization

The source's moment argument supplies a particularly useful rigorous finite test. Write

\[
D_j(t)=q_A(t)^j-q_B(t)^j,\qquad
\Delta=\int D_1\,d\mu>0.
\]

Neutrality is equivalent to \(\int D_j\,d\mu=\Delta\) for \(1\le j\le N\). The occupation density

\[
\rho(p)=\mu([\operatorname{logit}p-a,\operatorname{logit}p+a])
\]

defines \(\nu(dp)=\rho(p)\,dp/\Delta\), which matches the uniform distribution's moments through degree \(N-1\). Symmetrization permits matching through degree \(2m-1\), \(m=\lceil N/2\rceil\). It is without loss for this support optimization, **not** for the report-wise envelope.

Let \(x_m\) be the largest shifted Legendre zero. The degree-\((2m-1)\) polynomial
\[
(p-x_m)\left[\frac{P_m(2p-1)}{p-x_m}\right]^2
\]
has zero uniform integral. If the occupation density's upper support endpoint is at most \(x_m\), its integral against \(\nu\) is strictly negative, because \(\nu\) is absolutely continuous. Therefore every neutral prior supported on \([-B,B]\) must satisfy

\[
\sigma(B+a)>x_m.
\tag{C7}
\]

To certify a lower bound without trusting a floating-point root finder, use these rational candidate points:

| \(m\) | Rational \(\ell_m\), specified exactly as a decimal |
|---:|---:|
| 4 | 0.930568155797 |
| 8 | 0.980144928248 |
| 16 | 0.994700467495 |
| 32 | 0.998631930924 |

Verify \(P_m(2\ell_m-1)<0\) with the exact recurrence

\[
P_0(x)=1,\quad P_1(x)=x,\quad
P_j(x)=\frac{(2j-1)xP_{j-1}(x)-(j-1)P_{j-2}(x)}{j}.
\tag{C8}
\]

Since \(P_m(1)=1\), the intermediate value theorem ensures some root exceeds \(\ell_m\), hence \(x_m>\ell_m\). No assertion that \(\ell_m\) is itself an exact root is needed.

Equation (C7) then gives the exact necessary condition

\[
\boxed{\Gamma>\Gamma_{\mathrm{cert}}(N,\delta)
:=\frac{\ell_m}{(1-\ell_m)v}.}
\tag{C9}
\]

All operands in (C9) are rational for the prescribed inputs. A \(\Gamma\) at or below this rational threshold is rigorously excluded, conditional on the reviewed moment lemma. If the exact sign check fails, report a broken certificate; do not proceed using the printed decimal as a root.

The source also supplies the elementary bound

\[
B_N\ge\max\left\{0,\,
2\log\cot\frac{\pi}{2(m+1)}-a\right\}.
\tag{C10}
\]

Use it as a numerical cross-check and explanatory bound. The rational certificate is the default finite classification method for \(N=8,16,32,64\).

### 4.6 Finite existence bound and the constructive bridge

The source proves

\[
B_N\le U_N(a):=
\operatorname{arcosh}\{C(A+1)m(m+\tfrac12)-A\},
\quad C=\cosh(a/2).
\tag{C11}
\]

This comes from an explicit positive quadrature, not a hidden optimization oracle. The construction is:

1. Put \(h=1/(A+1)\) and
   \[
   w_a(z)=[(1-(A+1)z)(1-(A-1)z)]^{-1/2},\quad 0<z<h.
   \]
2. Take the \(m\)-node Gaussian rule for \(w_a(z)\,dz\), with positive weights \(\omega_i\) and interior nodes \(z_i\).
3. Set
   \[
   t_i=\operatorname{arcosh}(1/z_i-A),\quad
   Z=\sum_i\omega_i/z_i,\quad W_i=(\omega_i/z_i)/Z.
   \]
4. Use \(\mu=\frac12\sum_i W_i(\delta_{t_i}+\delta_{-t_i})\).

For a degree-\(N\) mixed likelihood polynomial \(f\), its symmetrized difference \(G_f(z)\) is divisible by \(z\); \(G_f(z)/z\) has degree at most \(N-1\le2m-1\). Translation invariance of the integrable mixed likelihood gives its weighted integral zero. Gaussian exactness then gives neutrality. Positivity and normalization give a proper prior with at most \(2m\le N+1\) atoms. These are the essential constructive steps to audit. [S0, §4; S5–S7 for established quadrature foundations]

To check the finite support cost, rescale \(z=h x\). Apart from a constant, the weight becomes \((1-x)^{-1/2}(1-\kappa x)^{-1/2}\), where \(\kappa=(A-1)/(A+1)\). The second factor is between 1 and \(C\). The smallest Gaussian node is the minimum Rayleigh quotient \(\int xQ^2\,d\eta/\int Q^2\,d\eta\) over nonzero polynomials of degree below \(m\): Gaussian exactness proves this by a weighted average of nodes and a polynomial vanishing at the other nodes. Comparison of the two weights gives \(z_{\min}\ge h\,\xi_{\min}/C\), where \(\xi_{\min}\) is the smallest node for \((1-x)^{-1/2}\).

For that comparison weight, the polynomial
\[
J_m(x)=\frac{(1-x)^{1/2}}{m!}\frac{d^m}{dx^m}
\left[x^m(1-x)^{m-1/2}\right]
\]
is orthogonal by integration by parts. Its values \(J_m(0)=1\) and \(J_m'(0)=-m(m+\tfrac12)\) imply \(\sum_i1/\xi_i=m(m+\tfrac12)\), hence \(\xi_{\min}\ge[m(m+\tfrac12)]^{-1}\). Substitution into \(t_{\max}=\operatorname{arcosh}(1/z_{\min}-A)\) yields (C11). Verify this bridge before using (C12) as an independent certificate.

The executable source approximates the continuous measure with a 640-node Legendre discretization after removing the endpoint singularity. Its Stieltjes recurrence and eigensolve produce an approximate version of the exact rule. Discretization, recurrence conditioning, eigenvalue error, and cancellation remain numerical issues. A small residual is not an exact witness at its rounded radius.

**For this project, do not implement a general quadrature solver.** The explicit \(N=2,3,4\) witnesses and analytic existence bound are sufficient for the diagnostic, while the primary limitation is independent of this theorem. No stronger numerical construction is needed to decide the selected question.

For rational \(\Gamma,\delta\), (C11) has an exact sufficient test. With \(X_\Gamma\) as above, square the positive quantities to obtain

\[
\boxed{
(A+X_\Gamma)^2
\ge \frac{(A+1)^3}{2}\,m^2(m+\tfrac12)^2.
}
\tag{C12}
\]

If this holds, label the class `existence_guaranteed_by_bound`, not “computed exact prior.” If it fails and no exclusion applies, the status is unresolved by these bounds. Do not infer impossibility from the failure of a sufficient condition.

### 4.7 Total variation and unanimous outcomes

For a neutral prior through \(N\), the full predictive laws at each \(1\le n\le N\) have

\[
\mathrm{TV}_n=\Delta(\mu).
\]

Every mixed sequence has zero difference; the all-success difference is \(\Delta\), the all-failure difference is \(-\Delta\). For the tiny witnesses, independently calculate

\[
\mathrm{TV}_n=\frac12\sum_{k=0}^n {n\choose k}
|L_+(n,k)-L_-(n,k)|.
\tag{C13}
\]

This is only \(n+1\) terms, not \(2^n\) explicit sequences. For \(n\le4\), also enumerate ordered sequences once as a separate check.

Every all-success record strictly favors \(H_A\) and every all-failure record strictly favors \(H_B\) under any proper common prior on finite real \(t\). Their evidence can be weak; they are not neutral. The general endpoint envelope remains applicable.

The notebook's function `check_prior` only compares the all-success difference with the one-observation difference under its `TV_identity_residual` field. It does not directly compute full TV and it has no pass/fail assertions. Repair that validation gap in the tiny independent implementation.

## 5. Numerical facts established during planning

These checks were performed in a **Linux x86-64 research environment**, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, with standard-library Decimal/Fraction calculations. They are not measurements on the user's Mac. The PoC below deliberately does not require NumPy or SciPy.

At the primary separation \(\delta=1/20\),

\[
v=11/9,\quad A=101/99,\quad
a=0.200670695462151161\ldots,
\]
\[
b=1.324716243034328964\ldots,\quad
e^b=3.761117960146808102\ldots.
\]

The following table separates exact formulas, exact rational certificates displayed in decimal, and sufficient bounds. Values are rounded for display; use the formulas and fractions in computation.

| Horizon \(N\) | Individual-neutralization threshold \(e^{B_{\mathrm{ind}}}\) | Joint-neutrality necessary threshold | Joint-neutrality sufficient threshold |
|---:|---:|---:|---:|
| 2 | 1 | Exact optimum 1 | Exact optimum 1 |
| 3 | 2.004476 | Exact optimum 3.761118 | Exact optimum 3.761118 |
| 4 | 3.010076 | Exact optimum 3.761118 | Exact optimum 3.761118 |
| 8 | 7.035287 | \(\Gamma>10.965774485638\ldots\) | \(\Gamma\ge71.039179\ldots\) from (C11) |
| 16 | 15.088242 | \(\Gamma>40.389517071115\ldots\) | \(\Gamma\ge274.087549\ldots\) |
| 32 | 31.195420 | \(\Gamma>153.569364141745\ldots\) | \(\Gamma\ge1069.998998\ldots\) |
| 64 | 63.410408 | \(\Gamma>597.237744256868\ldots\) | \(\Gamma\ge4221.148563\ldots\) |

For \(N=64\), the exact rational necessary threshold checked was

\[
\Gamma>
\frac{2246921844579}{3762189959}.
\]

These are requirements on permitted support, not estimates of actual bias or statements about how much prior mass must be at extreme values. At \(t=\log\Gamma\), the B-win retention probability is at most \(1/\Gamma\), since the A-win retention probability is at most one. This does not specify the overall fraction retained.

A second deterministic illustration shows why absence of universal neutrality does not establish report-wise robustness. At \(n=100,\delta=.05,K=10\), the minimum A-win count satisfying the symmetric-support robust-A rule is:

| Declared \(\Gamma\) | 1 | 1.1 | 1.25 | 1.5 | 2 | 4 |
|---|---:|---:|---:|---:|---:|---:|
| Minimum \(k\) out of 100 | 56 | 59 | 62 | 66 | 73 | 86 |

These are arithmetic consequences of (B1), not human-evidence results.

The planning audit actually performed the following bounded probes:

- Evaluated the notebook's construction/check functions at \(a=\log(11/9)\), \(N=2,3,4,8,16,32,64,128,256\). The source's original fifteen-case experiment matrix was **not** rerun as a whole.
- Independently checked the exact \(N=3,4\) prior at 90 Decimal digits, all six mixed tallies through four, and \(n=5,k=1\). Maximum mixed log-BF residual was approximately \(10^{-89}\); the out-of-horizon BF was \(0.956949636563424639\ldots\).
- Checked Appendix A's analytic priors at \(a=.1,1,2,3,5\) and its transition; residuals were below \(2\times10^{-88}\). The reported grid LP was not rerun.
- Checked the rational Legendre signs for orders \(4,8,16,32,64\), and the exact rational certificates used above.
- Independently checked 396 rational finite-mixture containment cases: \(n=2,4,8,16,64\), every \(k\), and \(\Gamma=1,5/4,2,4\), with weights \(1/4,1/2,1/4\) on odds atoms \(1/\Gamma,1,\Gamma\). All exact containments held. A float log-formula cross-check differed by at most approximately \(6.8\times10^{-14}\).

The numerical residuals are corroboration. Exact mathematical statements rest on the derivations and exact sign checks. You must rerun the required diagnostic independently; do not copy these values into result files as if measured by your implementation.

## 6. Implementation specification

### 6.1 Stack and minimal repository

Use Python 3.11 or newer and its standard library only: `math`, `decimal`, `fractions`, `csv`, `json`, `hashlib`, `pathlib`, `time`, `subprocess`, `resource` where supported, and `unittest`. No GPU, Metal, CUDA, model inference, external solver, service, database, or paid compute is needed.

Create this small repository:

| Path | Responsibility |
|---|---|
| `README.md` | Setup, commands, supported contract, and how to reproduce the report |
| `protocol.json` | Exact input strings, matrix, gates, numerical rules, budget |
| `neutrality.py` | Float log likelihoods/envelope; Fraction oracles; Decimal witnesses; support certificates |
| `study.py` | CLI stages, environment inspection, immutable protocol lock, run ledger, CSV/JSON outputs, report generation |
| `test_neutrality.py` | Independent correctness and scope-contract checks |
| `proofs.md` | Your checked derivations of (B3), (C4), directional exclusion, (C6), and finite certificates |
| `inputs/` | Original notebook/request if available, never modified |
| `results/study/` | Manifest, attempts, raw rows, validation, certificates, locked protocol, and final report |

Use separate functions for independent arithmetic paths; do not call the float formula from the reference oracle.

Minimum interfaces to create:

- `log_kernel(t, shift, n, k) -> float`.
- `log_bf_point(a, t, n, k) -> float` and `log_bf_bounds(a, L, U, n, k) -> (lower, upper)`.
- `log_bf_mixture(a, atoms, weights, n, k) -> float`, with positivity, finiteness, and mass validation.
- `fraction_kernel(v, x, n, k, hypothesis) -> Fraction` for \(x=e^t>0\).
- `fraction_bf(v, odds_atoms, weights, n, k) -> Fraction`.
- `legendre_fraction(m, x) -> Fraction`.
- `joint_support_status(delta, gamma, direction, N) -> record` with status and certificate operands.
- `decimal_witness34(delta, precision) -> odds_atoms, weights, exact_formula_metadata`.
- `assess_record(contract, delta, interval, n, k, K) -> record`.

All rational inputs are parsed from decimal strings into `Fraction`, never through binary float. Every rational constant in a certificate, including \(1/2\), must also be exact (for example `Fraction(1, 2)`); do not introduce float arithmetic through a literal division. Store exact numerator/denominator values in certificate JSON.

The `interval` passed to `assess_record` must carry exact positive rational `odds_lower` and `odds_upper`. Derive them directly from the original \(\Gamma\): \((1/\Gamma,\Gamma)\) for two-sided, \((1,\Gamma)\) for nonnegative, and \((1/\Gamma,1)\) for nonpositive support. Derive float \(L=\log(\text{odds_lower})\), \(U=\log(\text{odds_upper})\) only for the log path and display. Never reconstruct the exact endpoints with `exp(float_log_bound)`. Known-bias controls use coincident exact odds endpoints.

### 6.2 Fixed configuration

Use these fixed scientific choices. Engineering changes that preserve them are autonomous.

```json
{
  "project": "predictive-neutrality-limit",
  "primary_delta": "0.05",
  "delta_values": ["0.05", "0.01", "0.10"],
  "horizons": [2, 3, 4, 8, 16, 32, 64],
  "maximum_horizon": 64,
  "gamma_values": ["1", "1.1", "1.25", "1.5", "2", "4", "10", "100", "1000", "10000"],
  "directions": ["two_sided", "nonnegative", "nonpositive"],
  "evidence_threshold": "10",
  "decimal_precisions": [80, 120],
  "float_tolerance_ceiling": 1e-9,
  "decimal_check_tolerance": "1e-60",
  "compute_budget_seconds": 10800,
  "report_reserve_seconds": 1800,
  "repair_budget_seconds": 1800,
  "maximum_main_attempts": 2,
  "maximum_script_runs": 50,
  "maximum_result_rows_cumulative": 60000,
  "working_disk_cap_bytes": 268435456,
  "rss_cap_bytes": 536870912,
  "sampling_design": "fixed_retained_n_conditional_iid",
  "prior_class": "all_proper_common_priors_on_declared_interval",
  "gamma_justification": "experimental_sensitivity_settings_no_external_audit",
  "contract": {
    "basis": "declared_mathematical_model",
    "fixed_retained_n": true,
    "conditional_iid": true,
    "one_bias_per_record": true,
    "common_prior": true,
    "additional_selection_information": false,
    "outcome_dependent_report_choice": false
  }
}
```

The contract object is mandatory. These exact fields and values define the declared model used by this PoC; missing, null, unknown, or contradictory fields return `outside_scope` with a reason. Never assign this contract to an external report by default. If additional design information requires a different likelihood, the present implementation may conservatively reject it; accepting a different design would require a reviewed reduction to (A2), explicitly recorded before locking, rather than ignoring the information.

The maximum horizon for simultaneous-neutrality statements is 64. The separate \(n=100\) table in §5 is a baseline-only illustration and must be labeled outside the simultaneous-neutrality horizon. It does not change \(N\), construct a new neutral prior, or extend a witness.

The \(\delta\) sensitivity values test that the finding is not peculiar to .05; they are predefined, not tuned. No random seeds are required for these deterministic calculations. Record “not applicable” instead of inventing stochastic repetitions.

### 6.3 Stable arithmetic conventions

Use natural logarithms throughout.

For float evaluation, implement

\[
\operatorname{sp}(x)=\max(x,0)+\log(1+e^{-|x|})
\]

with `log1p`. Then

\[
\log f(t;\text{shift},n,k)
=-k\,\operatorname{sp}(-t-\text{shift})
 -(n-k)\operatorname{sp}(t+\text{shift}).
\]

Use a max-shifted log-sum-exp with `math.fsum` for finite mixtures. Reject negative weights, nonfinite nodes, or nonpositive total mass. Exact inputs must sum to one; do not silently normalize an invalid supplied prior. Combining duplicate atoms is allowed and must preserve exact mass.

For the independent Fraction path, with \(x=e^t\),

\[
q_A=\frac{xv}{1+xv},\qquad q_B=\frac{x}{v+x}.
\tag{D1}
\]

Evaluate powers and finite weighted sums directly. This gives exact rational BFs for rational odds atoms and weights, independent of the log-domain implementation.

For the \(N=3,4\) Decimal witness, compute \(v,A,R\) from rational inputs using Decimal arithmetic. Set \(x_b=R+\sqrt{R^2-1}\), use odds atoms \(x_b,x_b^{-1}\), and equal weights. Evaluate (D1) and direct powers at both precisions. This avoids an unnecessary inverse-hyperbolic calculation in the reference path.

During calibration, set the float verification tolerance to

\[
\tau=\max(10^{-12},10 e_{\mathrm{cal}}),
\]

where \(e_{\mathrm{cal}}\) is the maximum observed absolute log-BF error against independent high-precision/exact checks on the fixed calibration cases. Require \(\tau\le10^{-9}\); otherwise repair the implementation. This tolerance is for verification, not a scientific equivalence margin. Lock it before the main matrix.

Use exact Fraction comparisons for all rational \(K,\Gamma,\delta\) endpoint decisions. Where a displayed float is within \(\tau\) of a boundary, compare via the exact oracle. For nonrational witness checks, use the two Decimal precisions. If ambiguity remains, retain `numerical_boundary`; never change a label by rounding toward a desired outcome.

Require the two Decimal results to agree within \(10^{-60}\) in log BF, and the exact-witness mixed residuals through four to be below \(10^{-60}\). These targets leave twenty digits of guard precision at the lower setting. They still do not turn the computed witness into a formal exact certificate.

### 6.4 Algorithm pseudocode

```text
assess_record(contract, delta, interval={odds_lower, odds_upper}, n, k, K):
    preserve exact positive rational odds_lower <= odds_upper
    derive float L and U from those odds only for the log path
    validate a > 0, finite L <= U, n >= 1, integer 0 <= k <= n, K > 1
    if any required contract field is missing, unknown, or contradictory:
        return outside_scope with explicit reason
    compute a from delta, nominal = a * (2*k - n)
    lower = g(U; n,k); upper = g(L; n,k)
    verify lower <= upper within locked tolerance
    decide threshold inequalities using exact Fraction odds when available
    otherwise apply Decimal/boundary rule
    return nominal, lower, upper, zero_in_interval, label, assumptions

joint_support_status(delta, gamma, direction, N):
    validate exact rational 0 < delta < 1/2, gamma >= 1, N in prescribed list
    if N == 2:
        return exact_witness, mu = point mass at 0
    if direction is nonnegative or nonpositive:
        return analytically_excluded, one_sided_n2_n3_argument
    set v=(1+2*delta)/(1-2*delta), A=(v+1/v)/2
    set X=(gamma+1/gamma)/2
    if N is 3 or 4:
        if X*X - A*X - 2 >= 0:
            return exact_witness_formula, symmetric atoms from (C1)-(C2)
        return analytically_excluded, exact_small_horizon_support
    set m=ceil(N/2), ell=prespecified Fraction for m
    require legendre_fraction(m, 2*ell-1) < 0
    lower_certificate = ell / ((1-ell)*v)
    excluded = gamma <= lower_certificate
    sufficient = (A+X)^2 >= ((A+1)^3/2)*m^2*(m+1/2)^2
    if excluded and sufficient:
        fail gate: contradictory certificate implementation or proof
    if excluded:
        return analytically_excluded with exact rational operands
    if sufficient:
        return existence_guaranteed_by_bound with exact rational operands
    return unresolved_from_bounds with lower/sufficient bound information

run_locked_matrix:
    require passed proof/tiny/calibration/pilot gates and locked protocol hash
    for delta, gamma, direction, N in prescribed Cartesian product:
        compute joint_support_status once
        for k = 0,...,N:
            assess_record at n=N
            store row, including unanimous status and certificate reference
            if joint status guarantees existence and 0 < k < N:
                require exact endpoint BF lower <= 1 <= upper
    generate baseline-only n=100 threshold table for primary delta and six gammas
    preserve every attempted row, failure, and timing
    stop at any gate failure or resource boundary; generate report regardless
```

The main matrix contains 630 support-class records and 12,240 tally rows. The baseline-only \(n=100\) illustration adds 606 tally rows. These are evaluations of formulas, not independent observations or scientific sample sizes.

## 7. Required checks and minimal experiment matrix

### 7.1 Independent correctness before comparison

Use distinct controls capable of detecting model, sign, arithmetic, and quantifier errors.

| Check | Exact expectation or independent reference | Failure consequence |
|---|---|---|
| Unbiased reporting | \(I=\{0\}\), \(\log\mathrm{BF}=a(2k-n)\) | Repair sign/orientation; no main run |
| Endpoint attainment | Fraction point priors at \(e^L,e^U\) agree with (B1)–(B3) | Baseline invalid until repaired |
| Common-prior mixture | Direct Fraction likelihood sums for fixed two/three-atom priors lie in exact endpoint range | Investigate shared-prior or normalization error |
| Label swap | Swap A/B, \(k\mapsto n-k\), \(t\mapsto-t\): log interval negates with endpoints exchanged | Repair orientation |
| Known \(t\) | \(L=U=t\), envelope collapses to direct point-prior BF | Repair unsupported uncertainty |
| \(N=2\) | \(\delta_0\) gives mixed BF 1 | Repair base case |
| \(N=3,4\) | All six mixed tallies under (C2) have BF 1 analytically; Decimal residual within target | Do not claim a valid witness until resolved |
| Unanimous records | Correct strict BF direction; no mixed-neutral label | Repair theorem-scope error |
| Horizon failure | Primary \(\mu_{34}\), \(n=5,k=1\): BF near .95694963656 and \(|\log\mathrm{BF}|>10^{-3}\) | Detect accidental forced neutrality or wrong horizon |
| Direct full TV | Count sum (C13), ordered enumeration for \(n\le4\), and \(\Delta\) agree | Repair missing binomial weights or source-check weakness |
| One-sided class | Size-two integrand sign and size-three failure at zero | Repair or reject directional proof |
| Rational Legendre signs | Every listed exact sign is negative; \(P_m(1)=1\) | Broken finite-support certificate |
| Exact feasibility consistency | No class both excluded and guaranteed; \(N=3,4\) agree; derived necessary thresholds do not exceed source sufficient thresholds | Investigate algebra/certificate logic |
| Outside model | Missing required contract fields, or design information requiring a likelihood not implemented here, returns outside scope. Fixed completed \(M\), known exclusions, or stopping are not silently ignored; any claim of cancellation requires a checked derivation. | Repair applicability guard; do not assert these features universally invalidate a BF |
| Degenerate separation | Reject \(\delta=0\) for the nondegenerate theorem; separate explicit identity check has BF 1 | Do not extrapolate the discontinuous exact-support limit |

For direct Fraction mixtures, use odds atoms \(e^L,1,e^U\) when zero is inside the interval, and \(e^L,\sqrt{e^{L+U}},e^U\) only if these are rational. To avoid an irrational middle atom, the default for every direction is the arithmetic mean of the endpoint **odds**, \((e^L+e^U)/2\), which is positive, rational, and inside the odds interval. Use weights \((1/4,1/2,1/4)\) and \((1/8,1/8,3/4)\), plus the two endpoint point priors. These fixed priors are chosen before the tally.

Do not implement middle-atom selection from a winning count. A point prior at the analytic \(t_0\) is a separate mathematical existence check, explicitly identified as tally-specific.

### 7.2 Development, calibration, and locked evaluation

Use the following order:

1. **Development:** \(\delta=.05\), \(n=2,3,4,5\), \(\Gamma=1,2,4\), all directions, tiny witnesses, and analytic identities. Debug freely within the repair/time caps.
2. **Calibration:** \(\delta=.01,.05,.10\), \(n=64\), \(k=0,1,32,63,64\), \(\Gamma=1,2,10000\), all directions, exact endpoint arithmetic. Add exact \(N=4\) witness checks at 80 and 120 digits and the order-32 rational Legendre sign. This spans the planned small-separation, extreme-odds, large-count, and high-precision paths.
3. **Pilot:** primary \(\delta=.05\), \(N=4,8,16\), \(\Gamma=1,2,4,10,100\), all directions. Verify that exact witness, exact exclusion, general existence, and unresolved-bound statuses are reachable. If a status is absent in this pilot grid, add only its predeclared control: \(N=64,\Gamma=1000\) for an unresolved gap and \(N=64,\Gamma=10000\) for sufficient existence.
4. **Lock:** record the code revision/hash, complete protocol, verified derivations, arithmetic tolerance, and remaining budget. The final matrix is already specified; calibration cannot alter its scientific thresholds.
5. **Main:** execute §6.4 once, plus the baseline-only table. Verify primary claims from the proof and exact certificates; the matrix checks the implementation and communicates finite relevance.

Development and main formula domains may overlap because there is no learned model or statistical holdout claim. The separation prevents selective reporting and retroactive tolerance changes. Do not call this independent empirical validation.

A post-lock implementation bug permits one corrected main attempt within budget. Preserve and mark the original attempt invalid, state the change, and use a new code/protocol version. A scientific change to the class, metric, or hypothesis is outside this confirmatory protocol; report it as a proposed later study rather than tuning toward a gain.

### 7.3 Methods, fairness, and ablation

The simple reference is unbiased reporting, \(\mu=\delta_0\). The closest strong comparator is the exact common-prior endpoint envelope, derived from established bounded-selection robustness. The proposed theorem supplies a **simultaneous-neutrality annotation**, with an exact tiny witness or a proved support-class existence result. The control restricting the bias to one direction isolates its reliance on two-sided uncertainty.

All methods receive the same \((n,k,\delta,I)\) and sampling contract. The theorem does not receive reference-population outcomes, a post-hoc enlarged support, or a revised horizon. The nominal analysis is intentionally a reference under the extra assumption \(t=0\), not a fair substitute for the strong sensitivity baseline.

The main output is not a prediction-accuracy ranking. Report:

- Whether the proof of no additional report-wise vulnerability stands.
- Exact endpoint attainment and certificate success/failure counts.
- Maximum log-formula error against the independent oracle.
- Tiny-witness residuals, positivity/mass checks, direct TV checks, and out-of-horizon behavior.
- The finite individual-versus-joint support table and unresolved intervals.
- How threshold labels change with declared \(\Gamma\), including clean \(\Gamma=1\) controls.
- Runtime and peak memory as feasibility measurements only.
- Every failed, excluded, or unperformed calculation and its reason.

No Monte Carlo repetitions, confidence intervals for deterministic rows, empirical false-selection rates, or statistical power claims are appropriate. If a future empirical error-rate claim is proposed, it requires a separate generator, sampling model, independent unit, and protocol; it is not authorized as a rescue branch here.

## 8. Hardware feasibility, budgets, and execution gates

### 8.1 Initial inspection

Before measured work, record:

- Machine identity/CPU, architecture, macOS version, physical RAM, current memory pressure/available-memory evidence, and free disk.
- Python executable path, version, native versus translated architecture, `sys.float_info`, and Decimal contexts.
- The absence/presence and hashes of input files.
- Current project disk usage and the process resource-measurement method.

On macOS, available inspection commands include `sw_vers`, `uname -m`, `sysctl -n hw.memsize`, `sysctl -n machdep.cpu.brand_string`, `vm_stat`, and `df -k .`. Treat an unavailable command as missing metadata and use another read-only method; do not infer the answer from “M3 Max.” Physical RAM is not available RAM.

The standard-library CPU path avoids backend support uncertainty. Check that Python float has the expected binary64 properties and that Decimal `ln`/`sqrt` work at the requested precision. There is no accelerator fallback to validate.

If the target Mac is unavailable, complete the proof and diagnostic on the available machine when possible, label its environment accurately, and report Mac execution as unperformed. Do not call that a measured Mac feasibility result.

### 8.2 Hard ceilings and estimates

**Planning estimate, unmeasured on M3 Max:** the required arithmetic should take seconds to minutes per stage and use far less than the caps. Large integer operations on the prescribed counts and degree-32 certificates are small. The main matrix has fewer than 13,000 rows; stream it instead of retaining redundant copies.

Hard limits:

- **180 minutes cumulative subprocess wall time** across setup commands, development trials, tests, calibration, failed attempts, repairs, main computation, and report generation.
- **60 minutes cumulative subprocess time to obtain an informative pilot**, unless the study has already stopped with a checked analytic result.
- **Eight hours of active end-to-end work** for setup, implementation, mathematical review, execution, and writing. This is a project ceiling, not a target.
- **30 minutes of the 180-minute compute budget reserved for evidence preservation and reporting.** Stop new computation when only this reserve remains.
- **30 cumulative minutes for repair computation**, included in the 180 minutes; at most two repairs for a failing gate and at most two total main attempts.
- **50 script-run attempts and 60,000 cumulative result rows**, counting failed and superseded attempts.
- **Peak process RSS:** at most 512 MiB, further reduced to one quarter of physical RAM if that is smaller; inspect actual memory pressure before launch.
- **Project working disk:** at most 256 MiB, including inputs, results, attempts, and environment files. Require enough actual free space for this cap plus a modest operating reserve; do not fill a nearly full drive.

Estimated effort is separate from experimental runtime: approximately 15–30 minutes setup if a suitable Python is present, 2–4 hours implementation/checking, and 1–2 hours review/reporting. These are planning estimates; dependency or permission problems can change them. The lean standard-library path is chosen to avoid installation and preprocessing costs.

Keep a ledger of command, stage, attempt, timestamps, elapsed wall time, return code, result-row count, and budget remaining. Count top-level command elapsed time once; do not double-count a nested test subprocess already included in its parent. Stage changes and retries never reset the ledger. Do not spend unused allowance simply because it exists.

Check the stage/total deadline and cumulative row limit before every command and calculation batch. Give each top-level command a timeout no larger than its remaining allocation and preserve partial outputs on termination. Check disk and available resource measurements between batches. A measured RSS-cap breach makes the run failed and stops further computation pending the declared fallback; post-run RSS measurement alone is not a guarantee that the cap was never exceeded. No elaborate monitoring service is needed for this bounded scalar workload.

### 8.3 Calibration and resource measurement

Measure one cold CLI invocation, then one representative warmed calculation batch. Use `time.perf_counter` around the declared mathematical workload, and separately time total command execution including imports, parsing, and writing. On the target Mac, use `/usr/bin/time -l` on at least the pilot or main command to record peak RSS; retain its raw output. If using `resource.getrusage` too, record platform-specific units and cross-check rather than assuming Linux units on macOS.

Estimate full-run time from the representative per-row paths, including exact Fraction comparisons and output. Require the estimate multiplied by three to fit the remaining experimental allocation after the report reserve. If this fails:

1. First remove redundant repeated diagnostics or output copies, preserving every prescribed scientific case.
2. If necessary, run the analytic proof, exact tiny controls, rational support certificates, and primary \(\delta=.05\) table only; mark the secondary numerical sensitivity matrix unperformed.
3. If a mandatory core check still cannot fit, stop with the justified analytic result if independently established, otherwise inconclusive/practical infeasibility as appropriate.

Do not lower precision, loosen the tolerance ceiling, omit unfavorable cases, or change the uncertainty class to fit the machine. A missed optional matrix is not a refutation.

### 8.4 Evidence gates

Stage allocations below sum to 150 minutes, with the shared 30-minute repair allowance making the 180-minute cap. They are maximum allocations, not expected runtimes.

**Fallback override:** the full matrix counts and complete G1–G4 checklist apply only to the full execution path. If the missing-source or resource fallback is activated, record the retained claims and checks before proceeding; validate every retained check, mark unavailable dependent checks and omitted stages `not_run` with reasons, then proceed directly to G5. Do not mark an omitted gate passed or require `lock`/`run` solely to produce a partial report. The `report` command must accept this state.

| Gate | Question and prerequisites | Minimum work/evidence | Max compute | Decision and next action |
|---|---|---|---:|---|
| G0: assumptions and analytic feasibility | Does the requested support-only problem actually have the claimed obstruction? Inputs and environment inspected. | Checked derivation of (A1)–(B3), sampling caveat, source audit of the support claims used, environment manifest. | 10 min | Continue if proof and model contract agree. If source support theorem fails, preserve a counterexample and continue only the source-independent envelope result. If the endpoint proof fails, stop and diagnose; do not invent a classifier. |
| G1: tiny correctness | Are implementation and independent oracles correct? G0 accepted. | Exact endpoint controls, common-prior mixtures, N2/3/4 witness, unanimity and N5 controls, direct TV, one-sided proof, Fraction certificates. | 20 min | Pass only with exact logical checks and prescribed numerical agreement. Allow bounded repair; unresolved defects invalidate numerical conclusions. |
| G2: calibration | Can the full diagnostic run within resources? G1 passed. | Measured representative costs, RSS evidence, locked tolerance candidate, projected workload with factor-three margin. | 10 min | Authorize full matrix if margin fits; otherwise use the predeclared analytic/core fallback. Do not reduce mathematical accuracy. |
| G3: mechanism pilot | Does the diagnostic distinguish quantifiers and applicability? G2 passed. | Pilot rows and reachable positive, negative, gap, and outside-scope controls; no “absence of universal prior means robust” error. | 20 min | Pass on logical/numerical correctness, not a favorable gain. A contradictory certificate or missed control triggers bounded repair. No noisy Monte Carlo pilot is involved. |
| G4: locked diagnostic | Does the final artifact preserve the analytic conclusion across the declared finite settings? G0–G3 passed and protocol locked. | 630 support cases, 12,240 tally rows, 606 baseline-only illustration rows, exact implications, complete attempt ledger. | 60 min | Run once; zero unexplained certificate contradictions required. A verified structural negative is the intended completed outcome. Preserve a failed attempt and allow one corrected rerun only within caps. |
| G5: diagnosis and report | What is established, blocked, or invalid? Available even after an earlier stop. | Reproducible code where written, proofs, raw results, limitations, outcome classification, regenerated tables, budget totals. | 30 min reserved | Finish. Do not launch archive/optimization infrastructure after the analytic question is resolved. |

## 9. Commands and autonomous execution behavior

The following are **interfaces you must implement**, not commands already executed on the Mac. First verify that a suitable Python 3.11+ executable exists and record its architecture; use its actual executable path in place of `python3` if needed. Do not create the virtual environment with an older interpreter. If no suitable interpreter is available and ordinary local setup cannot resolve this within the budget/permissions, record setup as blocked and complete unaffected mathematical review. Once the files exist, make this sequence work from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python study.py inspect --protocol protocol.json --out results/study
.venv/bin/python study.py check --out results/study
.venv/bin/python study.py calibrate --out results/study
.venv/bin/python study.py pilot --out results/study
.venv/bin/python study.py lock --out results/study
.venv/bin/python study.py run --out results/study
.venv/bin/python study.py report --out results/study
```

No pip install is needed for the core implementation. The `check` command should run `unittest` and the independent arithmetic checks, with results incorporated in the ledger. `lock` must refuse if required gates failed. `run` must refuse an altered protocol or source-code hash. `report` must work on partial or failed runs and visibly identify which claims remain supported.

Use attempt-specific files; do not overwrite failed raw results or a locked protocol. Keep the same overall budget ledger across attempts. A reproducible rerun may use a fresh output directory, but it still counts against the current task's cumulative budget.

Execute all stages justified by the gates without seeking routine confirmation. Repair ordinary coding errors within the bounds. If target hardware, input provenance, permissions, or another indispensable resource is absent, complete unaffected work and state exactly which validation is blocked. Do not imply that an unavailable source, data file, or Mac was accessed.

Do not change the scientific class, add a prior restriction, tune \(\Gamma\) from a favorable tally, or extend \(N\) as an engineering choice. A genuinely different research question belongs in the final next-step recommendation.

## 10. Outcome interpretation and required final artifacts

### 10.1 Classify the result honestly

| Outcome | What warrants it here |
|---|---|
| **Informative negative result** | The endpoint-envelope proof and its independent checks stand: universal neutrality offers no additional report-wise worst-case BF decision power for the stated support-only class. This does not refute the notebook. |
| **Supported contribution** | A precise, reproducible limitation/quantifier-cost diagnostic is established in the stated scope. Do not relabel this as superiority of a new classifier or verified historical novelty. |
| **Practical infeasibility** | Measured or supported resource limits prevent a required computation. State what remains analytically established; a timeout is not evidence against the theorem. |
| **Inconclusive evidence** | An essential proof step, assumption, input, or numerical boundary remains unresolved within budget. A gap between necessary and sufficient support bounds is specifically unresolved, not automatically an inconclusive primary result. |
| **Invalid experiment** | A wrong likelihood, baseline, arithmetic path, scope guard, or measurement invalidated a numerical comparison. Preserve it, repair only within the rules, and distinguish a valid analytic result from invalid computations. |

The expected primary classification is an informative negative result about incremental report-wise value, accompanied by a supported explanatory diagnostic. A finding of no universal prior for a given \(\Gamma,N\) must never be reported as certification of the actual winner.

If a mathematical defect is found in the source, provide the exact equation, failed assumption, or reproducible counterexample. Determine which conclusions depend on it. The endpoint-envelope limitation does not rely on the source's quadrature or asymptotics.

### 10.2 Deliver only artifacts justified by the reached stages

For a completed diagnostic, provide:

1. Runnable standard-library code and concise usage instructions.
2. `proofs.md` with self-contained derivations, source attribution, and exact rational certificate operands.
3. `protocol.lock.json` and a manifest containing environment, hashes, arithmetic precision, gate decisions, and code revision.
4. Raw CSV/JSON results for every attempted case; failed rows carry explicit status, error, and attempt identifiers.
5. A cumulative run ledger with setup, calibration, computation, repairs, reporting, disk, and measured memory information.
6. `report.md` with the primary conclusion, finite support table, baseline-only illustration, correctness evidence, unresolved bound gaps, and limitations.
7. A report-generation command that regenerates all essential tables from raw results. A plot is optional only if it makes the individual/joint support gap clearer; it must not conceal unknown exact optima.

For an earlier analytic stop, deliver the verifiable derivation or counterexample and the minimal diagnostic actually needed to check it. Do not manufacture missing benchmark outputs, statistical tests, source reproduction, or data analysis.

The report should answer directly:

- What additional decision could a universal-neutrality layer justify under the fixed information contract? If none, say so.
- What simultaneous statement remains stronger, and what finite support cost is proved?
- Which assumptions can an external reporting audit actually supply, and which were merely sensitivity settings here?
- What is exact mathematics, exact rational arithmetic, approximate calculation, or unperformed work?
- What one next step, if any, would be worth pursuing?

A defensible next step is to obtain genuine external information about the reporting process before designing a richer uncertainty class. Do not recommend a large archive benchmark merely because the current application fails to add decision power.

## 11. Research context, closest alternatives, and access limitations

### 11.1 Closest prior art and contribution boundaries

**Robust selection models.** Bayarri and Berger explicitly model observed densities as selection weights times the original density, normalized by expected selection weight. They consider uncertainty classes for the weights and ranges of posterior functionals. Their distinction between sharp classes and loose conservative relaxations directly motivates using a strong baseline here. The binary common-prior envelope in this brief is a transparent specialization/derivation, not a verbatim invocation of their continuous-model extrema. [S1, §§1–3]

**Probability-ratio sensitivity.** Mathur and VanderWeele analyze publication probabilities favoring affirmative studies, with a sensitivity ratio and inverse weighting/tipping points. This establishes a closely related reporting-sensitivity framework. Their unit is a study defined partly by statistical significance, whereas ours is one decisive preference comparison. Their assumptions, empirical calibration, and effect estimands are not transferable wholesale. [S2, §§2.2–3.1]

**Prior-sensitive model comparison.** Llorente et al. discuss evidence extrema attained by degenerate priors and problems with improper priors and model comparison. This verifies the user-linked reference and confirms that a generic warning about prior dependence is not a new contribution. It does not replace the shared-prior ratio proof given here. [S3, §§4.1, 4.3–4.4, 5.1]

**Predictive matching.** Bayarri et al. define exact predictive matching for all samples of a designated size. The notebook restricts equality to mixed records and keeps unanimous records distinguishable. That restriction matters. It is neither complete indistinguishability nor a new general definition of predictive matching. [S4, §2.4, Definition 1]

**Finite positive representation and rational quadrature.** Bayer–Teichmann's finite positive representation permits an infinite underlying measure when the finite integrands are integrable; Gautschi's denominator-modified quadrature supplies established constructive machinery. NIST documents positive Gaussian weights and the orthogonal-polynomial/eigenvalue construction. These precedents prevent claiming generic finite matching or Gaussian construction as the new contribution. [S5, Corollary 2 and Remark 2; S6, Theorem 1.1; S7, §§3.5(v)–(vi)]

The most plausible source-specific novelty remains the precise logistic support/discrimination laws and small-horizon results. The bounded search did not establish their historical originality. This project's limitation conclusion does not depend on making that novelty claim.

### 11.2 Arena was assessed and deliberately not made a dependency

The 2023 LMSYS release describes 33,000 comparisons collected April–June 2023, with model names, votes, anonymized evaluator identifiers, timestamps, and other metadata. The Arena paper documents pairwise human-preference evaluation, ties/both-bad outcomes, and limitations of its population and preferences. These support the original application's relevance, not iid validity for an arbitrary subset. [S8, §§3.1,8; S9, “Dataset 1”]

On 1 November 2026, the canonical Hugging Face dataset card was readable but required login and agreement to share account information to access its files. Its documented prompt and model-output licenses differ. Actual records, schema, pair counts, immutable data revision, and file hashes were not verified in this planning study. No gated access was bypassed. [S10, access notice, dataset description, license section]

The selected analytic project needs no human dataset. Do not download a substitute just to fill a data section. Deterministic inputs and exact arithmetic are the appropriate accessible reference. A finite archive sampled without replacement is not an iid Bernoulli population, and resampling it would create simulation rather than additional human evidence.

If a later project uses Arena, it must independently lock pair identities, dates, eligibility and reference weights before observing outcomes; preserve nondecisive categories; address evaluator/prompt dependence; use the actual selection/stopping likelihood; and justify reporting bounds externally. Those tasks are outside this completed limitation study.

### 11.3 Search coverage and what was not established

Searches and source reads were conducted on **1 November 2026**. Coverage included robust Bayesian selection models, common nuisance-prior sensitivity, degenerate-prior extrema, publication-probability ratios, predictive matching, finite positive cubature, rational quadrature, logistic moment problems/finite mixtures, exact neutrality and bounded support, and Arena documentation.

Critical source content was read rather than inferred solely from search snippets. The first two pages of the Bayarri–Berger selection paper were also visually inspected because text extraction omitted equations. The supplied notebook figure was inspected directly. The Arena arXiv HTML supplied relevant paper content; the canonical dataset card supplied the current access limitation.

Backward connections to classical predictive matching and positive quadrature, and later treatments of prior and publication sensitivity, were examined where they affected the baseline and novelty assessment. This was not an exhaustive citation census, a search of every subscription database, or proof of absence of earlier equivalent formulas. Some broader nuisance-prior papers were visible only through metadata and are not used for theorem-level claims.

Research was stopped when the sharp baseline, source mechanism, access constraints, and minimal feasible study were sufficiently resolved. Further unrelated literature would not change the selected support-only limitation.

## 12. Claim-to-evidence map and review dispositions

### 12.1 Central claims

| Claim | Provenance/status at handoff | Remaining check for the coding agent |
|---|---|---|
| Outcome-dependent retention shifts log odds by \(t\) | Established weighted-selection model; algebra (A1) | Verify algebra and information contract |
| Support-only common-prior BF range equals endpoint range | Brief-derived proof (B1)–(B3); three separate planning reviews found the same result; rational probes passed | Independently verify proof and direct-oracle tests |
| Universal neutralization adds no missed report-wise vulnerability | Logical consequence (B4) | Check same class/threshold and preserve quantifier distinction |
| One-sided support excludes neutrality through \(N\ge3\) | Brief-derived sign proof | Exact tiny/sign controls |
| Individual-neutralization radius is (C6) | Brief-derived algebra; numerical checks performed | Verify zero solution, monotonicity, horizon maximum |
| Exact \(N=3,4\) support optimum | Supplied proof audited; independent high-precision witnesses checked | Recheck sign proof and exact Fraction support predicate |
| Larger-horizon exclusion/sufficient bounds | Supplied moment/quadrature proofs audited; rational lower signs checked | Recheck dependent lemmas and both exact predicates |
| Floating quadrature matches exact theorem numerically | Some planning reproductions; not formal certification | Not required for selected PoC; do not claim new reproduction |
| Specific \(\Gamma\) values are credible in an organization | **Unsupported; no external audit supplied** | Do not make this claim |
| Better AI-selection error rates or current rankings | **Not studied** | Do not make this claim |
| M3 Max runtime/peak memory | **Unmeasured during planning** | Measure actual target, or explicitly leave unperformed |
| Historical novelty of quantitative source laws | **Unresolved** | Attribute source; no global novelty claim |

### 12.2 Three review passes and material repairs

**Mathematical and novelty review.** Independent planning reviewers identified the sharp endpoint envelope and the directional impossibility. The source proofs, including the appendix's cubic certificates, were inspected without finding an essential defect. The brief was revised from a proposed classifier-improvement benchmark to a scoped limitation diagnostic. Symmetric support was separated from prior symmetry; rational certificates replaced floating root values for exact exclusion; report-wise and simultaneous quantifiers were separated. General prior sensitivity, predictive matching, and quadrature are attributed to prior work.

**Experimental review.** A supposed positive result could arise from a weak unbiased-only comparator, enlarged post-hoc support, unequal information, or treating absence of a universal witness as robustness. The protocol fixes the sharp baseline, information contract, and thresholds. The fixed-completed-count conditioning defect was made explicit. No human-archive accuracy claim is retained; the deterministic matrix is not presented as independent human evidence. Source code's incomplete TV check is replaced by a direct full-distribution calculation.

**Execution review.** The primary result needs neither gated data nor a numerical optimization oracle. The core was reduced to standard-library arithmetic, exact small witnesses, and finite analytical bounds. Review of the actual draft led to preserving exact odds endpoints in the API, requiring complete applicability metadata, adding explicit missing-source behavior and a self-contained upper-bound bridge, and clarifying command timeouts and nonduplicated budget accounting. Numerical thresholds, a concrete CLI, source/hash handling, partial-run reporting, and analytic-stop deliverables are specified. The general quadrature and appendix optimization are not mandatory infrastructure.

Before completing your report, repeat these three passes against your actual implementation and results. Fix material defects within budget and document changes. The following stress cases already have explicit dispositions:

| Stress case | Disposition |
|---|---|
| Generic existence is nonconstructive or costly to instantiate | Source construction is explicit; core limitation does not depend on computing it. Small exact witnesses and analytic support certificates suffice. |
| A realistic assumption fails | Mark outside scope; no certification transfers to dependent, changing, or improperly conditioned reports. |
| Mechanism is equivalent to known sensitivity | Expected main limitation; deliver it instead of inventing a weak baseline. |
| Only favorable synthetic cases appear to work | No performance-gain claim; include unbiased, one-sided, unanimous, and out-of-horizon controls. |
| A strong baseline removes the gain | The sharp baseline is mandatory from the start; no retuning to restore a gain. |
| Preprocessing dominates | No archive preprocessing is required; report setup separately and enforce total caps. |
| Finite precision changes a conclusion | Exact rational predicates/oracles, independent Decimal checks, boundary status; no silent tolerance relaxation. |
| Memory/runtime exceeds estimates | Measured calibration, factor-three margin, core analytic fallback, preserved failures. |
| Pilot does not resolve a claim | Identify the exact unresolved proof or numerical step; permit only bounded repair. Do not call absence of a numerical pattern refutation. |
| Essential source or target hardware is unavailable | Complete self-contained unaffected work; label source-wide or Mac validation unperformed. |
| The proposed practical hypothesis is false | This is an authorized completed result with proof, reproducible checks, and a scoped recommendation. |

Residual uncertainties are historical novelty, real-world admissible reporting assumptions, exact \(B_N\) for \(N>4\), source-wide formal verification, and actual target-machine measurements. None is to be hidden or filled with invented evidence.

## References

**[S0] Supplied research notebook.** Author not identified. *The quantitative cost of exact predictive neutrality*. Unrefereed research note, 1 November 2026. File: `Predictive_Neutrality_Research(1).ipynb`, original planning location `upload/`; SHA-256 in §2.1. Critical locations: §§1–6, Eqs. (1)–(16), cells 1–6; numerical implementation cells 8–11; prior-art/limitations cells 12–13; Appendix A, cell 14, especially Eqs. (1),(7),(9). The original request is `Pasted text(20261101-120033).txt`, also in `upload/`. These are supplied attachments, not public publications.

**[S1] Bayarri, M. J., and Berger, J. O.** (1998). *Robust Bayesian Analysis of Selection Models*. The Annals of Statistics 26(2), 645–659. DOI: [10.1214/aos/1028144852](https://doi.org/10.1214/aos/1028144852). [Author-hosted full paper](https://www2.stat.duke.edu/homeweb/berger/papers/selection.pdf). Critical locations: §1, Eq. (1.1), pp. 645–646, and Eqs. (1.2)–(1.3); §§2–3 on conservative versus sharp robustness bounds. Read during planning.

**[S2] Mathur, M. B., and VanderWeele, T. J.** (2020). *Sensitivity analysis for publication bias in meta-analyses*. Journal of the Royal Statistical Society, Series C 69(5), 1091–1119. DOI: [10.1111/rssc.12440](https://doi.org/10.1111/rssc.12440). [Accessible published-paper copy](https://www.med.osaka-u.ac.jp/pub/biostat/wp-content/uploads/2021/10/sensitivity-analysis-for-pb-in-ma.pdf). Critical locations: §2.2, Eq. (2.4), p. 1095; §3.1, Eq. (3.1a), p. 1096. Relevant sections read; no empirical ratio calibration transferred to AI evaluations.

**[S3] Llorente, F., Martino, L., Curbelo, E., Lopez-Santiago, J., and Delgado, D.** (2022). *On the safe use of prior densities for Bayesian model selection*. WIREs Computational Statistics, DOI [10.1002/wics.1595](https://doi.org/10.1002/wics.1595); [arXiv:2206.05210v1](https://arxiv.org/abs/2206.05210v1), 10 June 2022; [full HTML](https://arxiv.org/html/2206.05210v1). Critical locations: §§4.1, 4.3–4.4, 5.1. Relevant content read; this is the application request's linked arXiv paper.

**[S4] Bayarri, M. J., Berger, J. O., Forte, A., and García-Donato, G.** (2012). *Criteria for Bayesian model choice with application to variable selection*. The Annals of Statistics 40(3), 1550–1577. DOI [10.1214/12-AOS1013](https://doi.org/10.1214/12-AOS1013). [Full electronic reprint](https://arxiv.org/pdf/1209.5240). Critical location: §2.4, Definition 1 and Criterion 5, electronic reprint pp. 5–6; the PDF notes pagination differs from the published version. Read during planning.

**[S5] Bayer, C., and Teichmann, J.** (2006). *The proof of Tchakaloff's theorem*. Proceedings of the American Mathematical Society 134(10), 3035–3040. [Author-hosted manuscript](https://people.math.ethz.ch/~jteichma/tchakaloff120405.pdf). Critical locations: Corollary 2 and Remark 2, manuscript p. 3, and their proof on p. 4. Read during planning; infinite underlying measure is allowed when the finite functions are integrable.

**[S6] Gautschi, W.** (1993). *Gauss-type quadrature rules for rational functions*. [arXiv:math/9307223](https://arxiv.org/abs/math/9307223), 20 July 1993; [full manuscript](https://arxiv.org/pdf/math/9307223). Critical location: Theorem 1.1, manuscript pp. 2–3, denominator-modified Gaussian quadrature and its proof. Read during planning. This reference is to the retrieved manuscript date.

**[S7] NIST Digital Library of Mathematical Functions.** *§3.5 Quadrature*, especially [§3.5(v)](https://dlmf.nist.gov/3.5.v) and [§3.5(vi)](https://dlmf.nist.gov/3.5.vi). Accessed 1 November 2026. Critical formulas: 3.5.18–3.5.20 and the eigenvalue/eigenvector characterization. Official mathematical reference read during planning.

**[S8] Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Li, D., Zhu, B., Zhang, H., Jordan, M., Gonzalez, J. E., and Stoica, I.** (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. Proceedings of the 41st International Conference on Machine Learning, PMLR 235, 8359–8388. [Proceedings page](https://proceedings.mlr.press/v235/chiang24b.html). [Accessible arXiv HTML](https://arxiv.org/html/2403.04132), v1 dated 7 March 2024. Critical content read in the arXiv version: §3.1 interface and §8 discussion; the proceedings metadata was verified separately.

**[S9] LMSYS Org.** (20 July 2023). *Chatbot Arena Conversation Dataset Release*. [Official release article](https://www.lmsys.org/blog/2023-07-20-dataset/). Critical location: “Dataset 1: 33K Chatbot Arena Conversation Data.” Read 1 November 2026.

**[S10] LMSYS Org.** *Chatbot Arena Conversations Dataset*, official Hugging Face dataset card, [lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations). Accessed 1 November 2026. Relevant sections: access notice, dataset description, license. Card read; gated records, immutable revision, and schema were not independently inspected.

---

**Completion instruction:** carry the authorized study through the applicable gates, preserve unfavorable and failed results, and finish with the strongest conclusion the evidence actually supports. A checked reason not to build the proposed classifier is a successful completion of this research task.

