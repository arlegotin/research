# Evidence-aware release optimization: research and proof-of-concept plan

**Research date:** 30 October 2026  
**Target execution environment:** one MacBook Pro with an M3 Max; installed memory, free storage and macOS version must be discovered.  
**Status:** researched experimental handoff. Mathematical reproductions and a small exact accounting example were executed on Linux during preparation. The proposed release optimizer, Mac experiments and customer validation have not been executed.  
**Primary direction:** jointly choose release cohorts, approved deployment paths and verification portfolios, while a separate deterministic gate enforces unchanged evidence requirements.

## 1. Executive assessment

### 1.1 Recommendation

Proceed with a **bounded optimization experiment**, initially with no authority to deploy. Investigate whether looking across several possible releases produces better decisions than a competent pipeline that already pins candidates, caches valid results, selects affected checks, batches changes and optimizes verification for each fixed candidate.

The proposed contribution is the incremental benefit of **choosing the release subject and its verification plan together across a stream of completed changes**. It is not the invention of evidence tracking, freshness-aware scheduling, independent assurance, set cover, merge queues or conservative release gates.

This narrowing is consequential. The September 2026 AAS paper already formulates joint witness selection and scheduling for a supplied action, with exact proposal binding, validity constraints, receipt reuse and a separate admission boundary. Its inspected formulation supplies the action to the scheduler; the candidate refinement here adds backlog-wide release-cohort and transition choices. That distinction remains a hypothesis about useful differentiation, not an established novelty claim. [R1]

CI batching is also substantial prior art. BatchAll/TestCaseBatching and contemporary risk-aware batching already address capacity, grouping and feedback delay. A result against an always-rerun moving-head pipeline would therefore be scientifically inadequate. [R2, R3]

**Local claim to test:** on explicitly defined workloads, under the same acceptance policy, information, candidate action space and finite resource budget, joint optimization reduces charged verification resources per fixed unit of delivered work without an unacceptable latency or deadline regression.

**Broader claims excluded from the core PoC:** production safety, lower incident rates, enterprise willingness to pay, superiority to commercial products, and a universal law of AI-assisted delivery. The core can establish a reproducible optimization effect and bounded admission correctness; each broader claim requires additional evidence.

### 1.2 What survived scrutiny

| Proposition | Assessment | Consequence |
|---|---|---|
| Evidence must apply to the actual release and assumptions | Sound design requirement; established provenance and verification practice | Implement an explicit applicability gate shared by every method |
| More launches necessarily solve moving-target verification | False in the supplied fixed-duration model | Use the model as a diagnostic, not a production risk score |
| Pinning removes the modeled branch-churn penalty | True when every relevant input is fixed and no genuine expiry/revocation occurs | Pinning is mandatory in competitive baselines |
| Joint evidence selection and scheduling is new | Already covered by close prior art | Test the additional release-cohort decision, and credit existing methods |
| Across-release portfolio choices can affect total work | Demonstrable in a constructed exact example | A useful mechanism test, insufficient evidence of practical advantage |
| Supplied semantic mathematics measures real agent independence | Not established | Preserve requirement/oracle provenance; do not invent an independence score |
| Customer release invalidation is a material bottleneck | Unmeasured for the target customer | Measure avoidable cost after competent existing controls before adoption |

### 1.3 Decision hierarchy

1. Establish a correct, conservative admission model and an independent checking harness.
2. Determine whether meaningful headroom remains after strong baselines.
3. Test whether the joint controller captures that headroom after its own overhead.
4. Test transfer from exact examples and simulation to actual locally executed version combinations.
5. Consider customer shadow operation only after these gates. Continue to use existing production admission controls.

If a simple policy matches the joint method, ship or document the simpler engineering improvement. If even the offline optimum offers little improvement over competent practice, stop this optimization thesis for that workload. Neither outcome invalidates the supplied mathematics.

## 2. Problem, scope and hypotheses

### 2.1 Beneficiary and workflow

The intended beneficiary is a platform team supporting a multi-service product with observable delay between “implementation complete” and “release evidence complete.” Changes may be produced by agents or humans. The scheduling mechanism does not require an AI author; the PoC must not confound author identity with workload intensity.

The workflow is:

1. Complete work items enter a backlog with artifact revisions, dependencies and owner-approved requirements.
2. A pipeline chooses a release cohort and a permitted deployment plan.
3. Existing evidence is checked for applicability; missing evidence is acquired under finite resources.
4. A deterministic gate evaluates the policy against that exact subject.
5. A release-ready decision is recorded. Local experiments may execute the transition in a disposable fixture; production promotion remains outside scope.

“Implementation complete” is an external event. The optimizer cannot declare an unfinished feature complete, split one work item into several credits, remove a difficult requirement, or erase delayed work from the denominator.

### 2.2 Formal objects

A work item \(i\) has release time \(a_i\), immutable identity, weight \(w_i>0\), required artifact deltas, dependency predecessors, an atomicity group, and optionally an externally supplied deadline \(d_i\). Use \(w_i=1\) in the primary benchmark; use owner-supplied weights only in a separately registered sensitivity study.

A release subject is

\[
x=(A,L,C,S,R,P,E_0),
\]

where \(A\) is the content-addressed artifact vector; \(L\) dependency locks and runtime/toolchain identity; \(C\) configuration and feature flags; \(S\) schema/data-contract identity; \(R\) requirement/policy versions; \(P\) the deployment-plan identifier; and \(E_0\) the relevant initial environment state and currently deployed version vector.

The initial deployed state is part of a **transition** subject. Identical final artifacts reached from different schemas or rollout sequences need not have interchangeable transition evidence.

A release cohort is a set of complete atomic work items. It must be closed under dependencies, respect per-component revision order, and preserve all previously delivered work unless an explicit rollback/revert item is authorized. A later revision cannot silently count as delivery of an earlier feature that it removed.

Let \(U(x)\) be the obligations derived by a frozen resolver from the subject and the approved policy. Obligation predicates can differ across deployment plans, but the resolver and underlying requirements are fixed. Choosing a different approved plan may legitimately change the evidence needed; editing the resolver to obtain a better score is prohibited.

Each evidence receipt contains:

\[
e=(\text{id, claim, method, input projection, assumptions, verifier,
result, provenance, observation/completion times}).
\]

Applicability is a three-valued judgment:

\[
\operatorname{Applicable}(e,x,t)\in\{\mathrm{yes},\mathrm{no},\mathrm{unknown}\}.
\]

A missing dependency, unknown environment identity or unverified equivalence yields unknown. Unknown never satisfies a mandatory clause. A receipt may be valid evidence of a failure; applicability and a passing result are separate fields.

For each obligation \(u\), policy \(P_u\) specifies allowed combinations of evidence. Then

\[
\operatorname{Ready}(x,t)=
\bigwedge_{u\in U(x)}
P_u\!\left(\{e:\operatorname{Applicable}(e,x,t)=\mathrm{yes}\}\right).
\]

Ready means that the approved evidence policy is satisfied. It does not assert that all relevant requirements were correctly specified, every oracle is truthful, or the software is bug-free.

### 2.3 Optimization target and constraints

For an episode with fixed offered work, let \(R_i\) be its first gate-ready time for a subsequently completed release, and let \(T_i\) be that release's actual local transition-completion time. Let \(C\) include verification, setup, cancellation, retries, materialization, hashing and controller work. A planning objective is

\[
J_\lambda=C+\lambda\sum_i w_i(T_i-a_i),
\]

with explicit deadline, worker, memory, state-lock and admission constraints. \(\lambda\) converts work-item waiting seconds to resource seconds. It is selected on development data or supplied by the owner, never chosen after viewing confirmation results.

The **primary empirical metric is charged resource seconds per fixed offered work item in a finite episode that drains all deliverable work**. Latency, tail latency, deadline misses and all-work completion are guardrails. This prevents the scalar objective from hiding a worse practical tradeoff. A separate fixed-budget experiment directly tests completed work by a fixed horizon.

No infeasible instance becomes a successful run through relaxed requirements. Distinguish “a feasible plan was proved not to exist in this bounded model,” “the heuristic did not find one,” and “a tool or resource limit prevented a decision.”

### 2.4 Core scope

The required PoC uses:

- Immutable native local artifacts and deterministic identifiers.
- A bounded set of explicitly specified verification actions, including narrow and combined alternatives.
- Fixed, owner-like policy files and a separately implemented oracle for the benchmark.
- A discrete-event simulator and a native API–worker–database fixture.
- Finite CPU workers; no GPU, model training, paid API, remote machine or production integration.
- One active deployment transition, with development and verification allowed to proceed separately within the same resource budget.

Human approvals are simulated only in designated sensitivity experiments. Measured human coordination savings require a later observational study. Production diagnostics that mutate customer systems are excluded.

### 2.5 Hypotheses and falsifiers

| ID | Testable proposition | Alternative explanation | Discriminating evidence / falsifier |
|---|---|---|---|
| H1 | Joint release/verification planning lowers total charged resources at acceptable delivery latency | Better metadata, pinning, caching, batching, extra compute or easier work | Same metadata/gate/resources for all methods; compare against the development-selected strong baseline and simple amortized-reuse heuristic; failure to beat them weakens H1 |
| H2 | Across-release reuse and transition constraints explain incremental gains | Ordinary one-release set cover or queue tuning | Crossed ablations, fixed-candidate replay and mandatory-transition controls; disappearance of the joint interaction limits attribution |
| H3 | The applicability kernel rejects wrong-subject and unknown evidence within the bounded model | Shared bug between optimizer and evaluator | Independent full-subject checker and fault catalogue; any false admission is a blocking defect |
| H4 | Effects survive actual local execution and overhead | Simulator timing assumptions or free planning | Randomized paired physical runs, measured setup/controller cost; erased gain means no demonstrated local practical advantage |
| H5 | A customer has enough avoidable work to repay integration | Queueing, flaky tests, reviewers or unclear requirements are the true bottleneck | Shadow measurement and break-even analysis; no removable verification cost means stop for that customer |

H1 and H4 are confirmatory efficiency claims. H3 is a correctness gate, not a statistical proof of rare-event safety. H2 is attribution; H5 is a later commercial prerequisite.

### 2.6 Assumptions and boundaries

Core assumptions are explicit: the action catalogue faithfully states what each check establishes; accepted reuse footprints are complete within the fixture; requirement files cannot be changed by the scheduling policy; artifact identities are reliable; all methods receive identical online information; and resource accounting observes actual work.

These assumptions are themselves challenged by adversarial tests. A passed test on a known artifact is not a proof that a dependency declaration is complete. The system must degrade to full-subject binding or report blocked assurance when that declaration cannot be justified.

## 3. Research findings and novelty assessment

### 3.1 What the supplied material contains

The source bundle contains a complete mathematical note, intermediate derivations, internal audit notes, four numerical checking programs, recorded outputs, a rendering pipeline and a figure. The separately attached HTML is byte-identical to the HTML inside the ZIP; it is a duplicate presentation of the same research, not additional independent evidence.

The full LaTeX source, all four mathematical notes, supplied audit notes, and numerical scripts were inspected across the research team. Supplied audit filenames are internal research artifacts, not external endorsement.

| Input | SHA-256 |
|---|---|
| Pasted assignment text | 3e7e840c140ea7471e7847a08e5580763399c3e4dfa45b4e0385fd761b85e8f5 |
| Research ZIP | cb321668aa6ba6e91731b37b0ff89eb7ca0348a602ccbb30f889afa36ae8078f |
| Research HTML, both copies | ca6f49c6386bc1635196ccbda16ba028031b6fb8b881b342ce29e6be72e6e064 |

This plan restates the definitions and tests needed for implementation; possession of those attachments is not required to execute the core PoC.

### 3.2 Freshness mathematics: retained result and corrected application

The supplied model fixes finite obligation sets \(U\), coverage sets \(V_j\) whose union is U, and invalidation footprints \(D\). Every obligation has strictly positive total invalidation hazard. Independent Poisson changes occur at rates \(s\lambda_D\); independent verifier launches occur at \(\nu_j s^{\alpha_j}\). Runtime marks are independent, their distributions remain fixed as \(s\) grows, and jobs may overlap without a concurrency limit. A job must survive relevant changes during execution; evidence then remains current only until affected inputs change.

If \(F_j(t)=\Theta(t^{\beta_j})\) near zero, the supplied proof gives

\[
p(s)=\Theta(s^{-\kappa}),\qquad
\kappa=\min_{\cup_{j\in B}V_j=U}
\sum_{j\in B}(1+\beta_j-\alpha_j)_+.
\]

Here \(\beta_j=0\) requires an atom at zero. A continuous density positive at zero gives \(\beta_j=1\). A real test with a positive startup floor does not satisfy that near-zero power-law assumption.

For fixed durations \(\delta_j>0\), the corresponding logarithmic rate is

\[
\lim_{s\to\infty}-s^{-1}\log p(s)=
\min_{\cup_{j\in B}V_j=U}
\sum_D\lambda_D
\max\bigl(\{\delta_j:j\in B,\ D\cap V_j\ne\varnothing\}\cup\{0\}\bigr).
\]

Shared invalidations produce a maximum-duration contribution, not independent per-test penalties: several jobs can benefit from the same quiet interval.

For one obligation the exact moving-head expression is

\[
p=e^{-rd}\frac{v}{r+v}.
\]

It follows by integrating \(1-e^{-v(a-d)_+}\) against the exponential age of the last invalidating change. With \(r=8/\mathrm{hour}\), \(d=0.25\) hour, \(v=12\) or 24/hour, values are 0.0812012 and 0.1015015; the infinite-launch limit is 0.1353353. These are model calculations. Average concurrent jobs would be \(vd=3\) or 6, so even these launches imply nontrivial capacity.

For a candidate pinned at time zero with all relevant inputs fixed, no initial receipt, and launches beginning after pinning,

\[
\Pr(\text{ready by }t)=1-e^{-v(t-d)_+}\to1.
\]

Unrelated branch churn disappears from this expression. In an ordinary deterministic run, the receipt simply becomes available at completion. This is the crucial negative control.

No substantive error was found in the conditional proof arguments reviewed. Nevertheless, finite workers, queues, batching, cancellation, correlated runtimes, load-dependent service times, growing obligation sets and chosen release boundaries require a different model. The PoC therefore uses empirical durations and explicit events, not a fitted asymptotic exponent.

### 3.3 Semantic mathematics: useful limitation, no production score

The supplied semantic model starts with false retained constraints excluding the true behavior in a finite field. New unconstrained coordinates arrive at rate \(\lambda\); independent truthful linear audits arrive at rate \(\mu\). Exposure means that no ideal repair can satisfy the retained suite, not that a particular deployed implementation first fails a test.

The note derives a killed birth–death process, an established rejected-service queue interpretation, and a transform of the vector-matroid characteristic polynomial. For a finite wrong-alternative set \(S\), define \(r(A)=\dim\operatorname{span}(A)\) and \((a;Q)_k=\prod_{j=0}^{k-1}(1-aQ^j)\), with an empty product equal to 1. For \(Q=1/q\), \(\rho=\mu/\lambda<1\),

\[
\Pr(T=\infty)=
\sum_{\varnothing\ne A\subseteq S}
(-1)^{|A|+1}(\rho;Q)_{r(A)}.
\]

If \(\lambda\le\mu\), exposure is almost sure. Classical matroid structure determines the small-\(\rho\) exposure exponent. These are conditional mathematical results; historical originality of the exact transform is unresolved. The queue and matroid ingredients are known. [R17, R18]

The practical safeguard is to separate approved requirements from generated implementations and tests, record evidence origin, and test oracle errors. There is no validated mapping from lines of code to unconstrained dimension, model-provider count to independent audits, or actual programs to this characteristic polynomial. Targeted truthful checks can immediately expose a false requirement; uniform random audits are a restrictive assumption.

### 3.4 Checks actually executed during preparation

All four supplied programs were reviewed, copied and run without modifying their originals. Environment: Linux 6.18.57 x86_64, CPython 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8. No M3 Max execution occurred.

| Check | Reproduced outcome |
|---|---|
| Instantaneous CTMC versus moment recurrence | 11,881 verifier/reset family pairs, eight moments each; maximum absolute discrepancy \(5.5511\times10^{-16}\) |
| Weighted cover enumeration versus dynamic program | 600 exact rational-weight cases agree |
| Separate dependency moment checks | 360 comparisons; maximum discrepancy \(6.6613\times10^{-16}\) |
| Semantic finite-field counts and hitting systems | Exhaustive configured counts agree; hitting discrepancy \(1.8874\times10^{-15}\) |
| Truncated semantic mean-time systems | Maximum discrepancy \(5.0176\times10^{-7}\); a finite-cap approximation |
| Matroid finite-generator cases | Four cases, 450–675 states, nullity cap 45; maximum discrepancy \(2.4758\times10^{-13}\) |

Total sequential elapsed time was approximately 5.17 seconds in that environment. These checks do not validate real test selection, reuse soundness, hardware performance, release throughput or rare defects. The reflected mean-time boundary and absorbing-survival matroid boundary have different approximation biases; neither is an infinite-state proof.

### 3.5 Closest prior art

| Work / capability | Mechanism and evidence | Exact overlap / difference | Reuse and comparison treatment |
|---|---|---|---|
| AAS, He and Yu, 2026 [R1] | Joint evidence acquisition for a supplied action; constrained scheduling and repair; generated-workload evaluation | Very close assurance machinery. The inspected formulation does not optimize a backlog of release cohorts | Fixed-subject adaptation is a mandatory comparator; no located implementation is not proof none exists |
| VP-CONTROL, Zheng et al., 2026 [R29] | Fixed-proposal verification portfolios, shared failure modes and atomic commit guards | Direct overlap with evidence choice and admission integrity; single-action scope | Reuse guard/lineage lessons; local bounded adapter is not a reproduction of its HTTP experiments |
| BatchAll / TestCaseBatching, FSE 2023 [R2] | Dynamic CI batching and interaction with parallelism; paper, code and Chrome trace artifact | Already combines batch formation with capacity; build-result aggregation is not exact mixed-version correctness | Port documented policies into common resource engine; verify policy decisions against source |
| Lightweight dynamic batching, 2025 [R30] | Online batch-size adaptation with feedback-delay concerns; paper and small replication archive | A strong inexpensive way to capture grouping gains | Separate named B3 configuration; archive license and exact port behavior are gates |
| Risk-aware performance batching, 2026 [R3] | Risk-adaptive grouping, aging, build/backfill costs and finite-worker simulation | Strong contemporary batching alternative; performance culprit detection differs from release policy satisfaction | Implement specified risk/age policy with identical locally available predictions; broader published results remain context |
| GitHub merge queue [R4] | Required checks on target branch plus preceding queued changes | Combined changes and exact merge-group identities already addressed | Functional behavior comparator; do not claim a local emulator benchmarks GitHub |
| Zuul speculative gating [R5] | Dependency-aware speculative combinations and shared queues | Strong existing answer to concurrent integration | Reference behavior for local queue baseline; full distributed deployment is optional |
| Trunk merge queue [R6] | Parallel queues, batching and build-system awareness | Commercial overlap substantial | Documentation is capability evidence, not a measured head-to-head result |
| Bazel and Build Systems à la Carte [R7, R8] | Dependency tracking, content-based reuse and incremental build abstractions | Reuse is established; a complete cache key remains an assumption | Share equivalent valid caching with every method |
| Safe regression test selection / Ekstazi [R9, R10] | Select tests from change/dependency information under stated assumptions | Avoiding unaffected tests is established; observed coverage alone is not general semantic noninterference | Use explicit conservative footprints; learned selection cannot waive mandatory clauses |
| Predictive test selection [R11, R12] | Rank likely failures using historical information | Useful scheduling signal; changing retained checks may change assurance | Ranking may order mandatory checks; omission requires explicit policy equivalence |
| SLSA VSA and in-toto [R13, R14] | Artifact-, verifier- and policy-bound records | Provenance/evidence envelopes are existing infrastructure; scope is not general functional correctness | Reuse envelope concepts; custom functional claims need their own documented predicate |
| Classical freshness/repair/AoI [R15, R16, R19, R40] | Grouped refresh, coverage and scheduling under resource constraints | Freshness plus set cover and correlated-update scheduling are known | Mathematical context; objectives differ from all-obligation release readiness |
| Continuous/incremental certification [R20] | Re-evaluate changed artifacts and dependencies | Evidence maintenance under evolution is established | Supports conservative delta verification; no “first continuous assurance” claim |
| Independent agent assurance [R21] | Specification-first and implementation-aware testing study with seeded faults | Supports examining oracle provenance, not a general independence guarantee | Optional source of small external test artifacts; no new model generation required |
| Microsoft release-readiness patent disclosure, 2026 [R38] | Component-oriented scheduling and release-readiness determination | Additional technical prior art against broad scheduling claims | Description and selected claims inspected; no legal novelty or status conclusion |

Scientific novelty is **unestablished**. The likely initial contribution is an engineering method and an empirical characterization of when cross-release optimization helps. A genuinely new algorithmic claim requires a formal distinction from [R1–R3], a proof or discriminating experiment, and a repeated prior-art search for that exact claim.

### 3.6 Current market and motivation

GitHub’s October 2026 report says detectable agent involvement initiated more than 40% of September pull requests, up from 17% in March. This is platform telemetry, not a causal productivity or defect estimate. [R22] CloudBees’ survey of 213 enterprise technology leaders reports increased downstream costs; it is vendor-sponsored, self-reported evidence. [R23] Atlassian reports a governance gap, while DORA emphasizes that AI amplifies the surrounding engineering system. [R24, R25]

These observations justify looking for a bottleneck. They do not establish that release evidence invalidation is the dominant one. CloudBees Unify and Harness already market cross-tool governance, readiness and deployment verification. [R26, R27] Stronger formal verification also exists under explicitly scoped machine and specification assumptions; it changes the verifier catalogue, not the need to bind a result to its subject. [R28]

The product recommendation is an optimizer integrated with existing delivery systems, conditional on measured incremental benefit. A theorem alone is not a moat.

### 3.7 Headroom before sophistication

Measure baseline verification resources \(C_B\), optimistically removable work \(W\), and incremental optimizer/integration cost \(O\). Necessarily,

\[
\text{net saving}\le W-O.
\]

Estimate \(W\) from duplicate applicable jobs, avoidable invalidations, alternative approved portfolios and exact tiny-instance comparisons. Do not classify all retries or waiting as removable: a flaky-test retry, unavailable reviewer and actual incompatibility require different remedies.

For a customer, break-even episode count is

\[
N_{\mathrm{BE}}=
\frac{\text{one-time integration cost}}
{\text{saved resource cost + valued delay reduction - ongoing cost}},
\]

when the denominator is positive. Keep engineering hours and machine seconds separate until an owner supplies conversion rates. The original 25–30% commercial aspiration is not a scientific success threshold.

### 3.8 Direction selection and strongest objections

| Direction | Value and limitation | Decision |
|---|---|---|
| Generic evidence-aware assurance control plane | Useful integration, but substantial product and research overlap; an integration win would not establish scheduling novelty | Preserve its evidence model as infrastructure, not the research contribution |
| Better pinning, caching and fixed-subject portfolios | Direct, inexpensive remedies with established methods; may capture most available savings | Mandatory shared controls and a valid simpler outcome |
| Across-release cohort/transition planning | Can prefer reusable evidence or a better migration sequence; benefit depends on residual coupling and search overhead | Selected bounded hypothesis, tested against B5/B6 and speculative controls |
| Learned semantic equivalence or a universal trust score | Would require a justified program-analysis/oracle model absent from the supplied algebra | Excluded from the minimum PoC; no hidden assumption of reliable learned independence |
| Larger-scale distributed optimization | Could matter when search/capacity structure grows, but cannot be validated by a small local timing result alone | Conditional escalation after a structural prediction and measured local headroom |

The strongest case for the selected direction is a repeated, unavoidable sequence of releases in which a locally cheap verification choice destroys future reuse or forces an expensive transition. The strongest case against it is that competent pinning, caching, batching and a simple reuse-aware rule already capture that value, while the remaining delay comes from necessary tests, humans or genuine incompatibility. The exact example tests the former mechanism; B5/B6, speculation, overhead accounting and customer headroom measurement test the latter explanation. Neither is presumed to win.

## 4. Selected mechanism and implementation specification

### 4.1 Architecture

Use three separable components:

1. **Evidence store and applicability kernel:** deterministic input binding, policy resolution and receipts.
2. **Candidate and verification planner:** may make inaccurate cost/runtime predictions, but cannot edit policy or forge receipts.
3. **Runner and independent evaluator:** execute checks, account for resources, and independently validate every admission against the full benchmark state.

The small gate controls all admissions. The optimizer only proposes legal actions. This architecture is established assurance practice and is not claimed as new.

~~~mermaid
flowchart TD
  B["Completed work and deployed state"] --> C["Legal release candidates"]
  C --> P["Joint planner"]
  E["Evidence store"] --> P
  P --> R["Verification runner"]
  R --> E
  P --> G["Deterministic admission gate"]
  E --> G
  G --> D["Release-ready decision"]
  D --> A["Independent evaluator"]
~~~

### 4.2 Policy representation

Represent each obligation as a disjunction of approved clauses. Each clause is a conjunction of typed evidence requirements, for example:

- (retry-state-machine test AND rolling-compatibility test), OR
- (a designated combined transition checker).

An approval requirement may additionally demand a particular principal or role and bind the exact subject/policy digest. A test and an approval are different evidence types. Never replace conjunctive requirements with “one vote per checker.”

Store clauses directly; avoid exponential conversion from arbitrary logical formulas. Core instances have a bounded explicit catalogue. A clause records allowed method IDs/revisions, claim, input scope, accepted outcome, optional expiry semantics, and any explicit distinct-principal constraint. Copies of one receipt do not create additional independent witnesses.

Before timing, validate every approved alternative against the independent bounded behavioral oracle and registered fault catalogue. Each alternative must establish the same stated bounded obligation; matching claim names is insufficient. A combined action may save shared setup but may not omit required cases. If alternatives have deliberately different assurance standards, separate that study from the fixed-quality comparison.

Tests on immutable inputs have no TTL by default. Time limits apply only when the policy specifies a changing assumption, such as an external lease or live observation. Unknown or revoked assumptions invalidate admission even when artifacts are pinned.

### 4.3 Applicability and cache keys

For evidence method \(j\), define an approved projection \(F_j(x)\) containing every relevant input. The cache key is

\[
k_j(x)=H(\text{schema version},j,\text{method revision},
F_j(x),\text{oracle revision},\text{policy scope}).
\]

Use SHA-256 of canonical UTF-8 JSON: sorted keys, fixed separators, integers for time/units, ordered arrays where order matters, no NaN, no floats in identity fields, no absent-versus-null equivalence. Hash file bytes, not modification times. Use an explicit normalized path manifest and reject duplicate keys/path collisions.

A projection can contain the whole subject. Narrower reuse is permitted only under a documented complete scope for the bounded fixture. Dynamic access tracing can reveal missing dependencies, but failure to observe an access cannot certify independence. External repositories start with whole-subject binding.

~~~text
applicable(receipt, subject, now, state):
    if receipt bytes or provenance fail integrity checks: return NO
    if required metadata or scope completeness is unknown: return UNKNOWN
    if method/oracle/policy revision is not currently accepted: return NO
    if recorded input projection != approved_projection(subject): return NO
    if any assumption is revoked or contradicted: return NO
    if an assumption's identity or validity cannot be established: return UNKNOWN
    if policy has a validity window and the required use interval is outside it:
        return NO
    return YES
~~~

Freshness begins at the observation time required by the method, not automatically at job completion. For transition evidence requiring validity during a rollout, check the full protected interval. No time window implies persistence under unchanged inputs, not perpetual trust in a revocable verifier.

Store failed and indeterminate outcomes. They can avoid pointless repeated work or block a contradicted claim, but do not satisfy passing clauses. For deterministic method/input identities, a recorded failure cannot be overwritten by a later pass without an explicit flaky-check or remediation policy. Core failures require a new subject or method revision.

The policy defines an authoritative contradiction relation: an applicable FAIL establishing the negation of a mandatory claim vetoes admission even if a selected alternative returned PASS. Enforce that rule inside both gates, not only the planner. INDETERMINATE cannot satisfy a clause and is not automatically a semantic failure. A superseding requirement, remediation or exception must be an external recorded policy event, never an optimizer choice.

Bind actual fixture-data identities, input generators/seeds, transition enumeration bounds, test selection, runtime binary and environmental assumptions in the method/projection key. A result for one bounded input suite does not certify another merely because the source artifacts match.

**Conditional reuse argument:** suppose a qualified deterministic verifier's bounded claim depends only on the complete projection F_j, its recorded oracle and method, and the explicitly listed assumptions. Equal canonical projections and unchanged accepted method/oracle identities then reproduce the same qualified claim while those assumptions remain valid. Reusing that receipt preserves the evidence-policy predicate. This is a conditional noninterference argument, not a proof that a practical footprint is complete. The independent full-state checker and omitted-dependency faults test that premise within the fixture. Nondeterministic checks need an explicit statistical evidence policy and are outside this deterministic argument.

### 4.4 Candidate generation

Maintain per-component revision chains and an item dependency DAG. Every proposed candidate:

1. Starts from the currently deployed state.
2. Includes whole atomic groups and all required predecessors.
3. Respects one selected frontier per component; incompatible branches are not silently merged.
4. Uses a predeclared approved deployment path.
5. Preserves outstanding work identities and their ages.

For tiny cases enumerate all legal downward-closed subsets. For larger cases generate a common candidate pool from earliest-deadline/oldest-item prefixes, dependency closures, component frontiers and maximal currently feasible cohorts. Give this same pool to all methods that choose candidates. Include each strong baseline's proposed candidate even if a pool cap would otherwise discard it.

Sort candidates deterministically by oldest included deadline, arrival sequence and digest. Record all pruned candidates and why. Changing the cap changes the search approximation and must be exposed in sensitivity results. Do not claim a global optimum over a restricted pool.

No new arrival mutates an active candidate. Superseding it creates another subject; jobs already running continue on their captured inputs unless cancelled under the common cancellation policy. Their completed evidence can still be stored and reused where applicable.

### 4.5 Deployment transition model

The core fixture has two API instances, two worker instances and a shared schema. A rollout plan specifies allowed steps and a partial order, such as expand schema, replace API instances, replace workers, drain legacy work, then contract schema.

Enumerate the reachable bounded version-vector states and operation interleavings permitted by the plan. Check obligations in every enumerated state and across transition edges; final-state tests alone are insufficient. Include in-flight messages written by the old API and consumed by a new worker, old workers against expanded schema, and rollback points.

Do not assume all mixed-version combinations are reachable. Conversely, do not omit a reachable combination merely because the normal rollout script visits one preferred ordering. The exact bound on message histories and interleavings is part of the method identity. Bounded enumeration establishes only that bounded property.

A candidate's transition evidence is keyed by start state, end artifacts, path/partial order, configuration, schema and verifier bound. If another release changes the start state before promotion, the transition witness must be re-evaluated for applicability. Record this as a state conflict, not an ordinary branch update.

### 4.6 Shared fixed-subject solver

Implement a shared function:

~~~text
plans_for_subject(observation, subject, planning_limits)
    -> list[Plan(resource_cost, completion_time, receipts_expected,
                 jobs, assumptions, search_status)]
~~~

Represent cached receipts as zero-incremental-verification-cost alternatives retaining their original input identities and validity intervals. Remove an obligation from future scheduling only when those receipts remain applicable throughout its proposed use interval. Enumerate approved clauses and qualified action portfolios; account once for any action covering several clauses; respect prerequisites, resources and genuine validity windows; and retain nondominated cost/completion plans.

Return more than the cheapest portfolio when useful: a slightly more expensive portfolio may leave evidence reusable by a subsequent candidate. The joint outer planner and all lookahead controls receive the same portfolio frontier. A dominated plan may be discarded only when it is no better in cost, completion, resource feasibility **and future receipt applicability**. Dominance based on cost alone can remove the indispensable mechanism.

The initial executable pruning rule is conservative: compare only plans with identical continuation fingerprints of retained receipt projections/claims, validity intervals, assumption versions, running jobs, deployed/intermediate state and residual resources. Within that equivalence class discard a plan only when cost and completion are no better. Otherwise retain both. A later finite observed-candidate applicability signature may strengthen pruning, but unknown comparisons remain incomparable; never promise dominance over unseen future states. Report whether the frontier is complete or search-limited.

For exact tiny cases use integer ticks and enumerate all selected-job start assignments up to a finite horizon, including intentional idle time. Validate every assignment against resource occupancy, prerequisites, arrival times, evidence intervals and the full policy. If a bounded instance has \(m\) jobs and horizon \(H\), the naive temporal search can grow as \((H+1)^m\) before portfolio/candidate choices. Start with three items, four actions and eight ticks; increase only after measured node counts justify it.

An event-state search, branch-and-bound or optional solver may replace enumeration only after agreeing with the independent finite-grid oracle. A feasible list schedule is not automatically an exact optimum under precedence/resources. “Exact” always means exact for the explicitly bounded, discretized planning model with its assumed durations; it does not mean exact for continuous stochastic execution.

At larger scales, use the same deterministic bounded search or heuristic inner solver for competitors and the proposed method. Return statuses FOUND, PROVED_INFEASIBLE_WITHIN_MODEL, SEARCH_LIMIT or ERROR. SEARCH_LIMIT must not be reported as infeasibility. Cache solver results by full planning-state identity, and charge cache lookup/creation.

### 4.7 Proposed outer controller

This is an application of established receding-horizon/model-predictive control. A generic controller over identical states, actions and objective is the same method, not a distinct algorithm that can be claimed beaten by renaming it.

The minimum method is **bounded lookahead over observed pending work**, with no learned future-arrival predictor. At each event:

1. Generate the shared legal candidate/transition pool.
2. Obtain the shared nondominated verification plans for each candidate.
3. Evaluate sequences of at most two candidate admissions, carrying forward precisely applicable receipts and changed deployed start states.
4. After the lookahead boundary, complete all remaining observed work with B_ref, the development-selected policy from B2/B3 with B1 as a feasible fallback. Freeze B_ref before fitting lookahead methods. It cannot invoke J or B5 recursively. This terminal rollout prevents uncounted residual backlog from making a partial plan look cheap.
5. Reject a rollout that violates a known hard service constraint or is already contradicted by evidence.
6. Choose the lowest \(J_\lambda\) feasible rollout; resolve ties by less total work, earlier oldest-item delivery, then stable IDs.
7. Execute only the next legal action set; replan after arrivals, completions, revocations or state conflicts.

The observed-backlog terminal rollout is a prediction, not a guarantee about unseen work. An optional arrival forecast is a later ablation and must use only development data and revealed history. Do not make model training indispensable.

Useful actions are FORM, LAUNCH, CANCEL_REQUEST, WAIT_UNTIL and REQUEST_ADMISSION. Jobs capture immutable inputs on launch. Waiting must name the next completion, arrival notification, expiry, deadline-derived timer or finite batching timer; infinite deferral is invalid. A policy-independent oldest-item/deadline watchdog invokes a legal fallback or records infeasibility, without changing requirements.

Core jobs are nonpreemptive, with cancellation disabled. There is at most one active candidate and one executing transition. Once the first job for a selected subject launches, lock that subject until admission and rollout completion, or explicit terminal failure/retirement. Arrivals update the backlog; they do not launch checks for a different candidate. Replanning an active subject may change its remaining approved verification portfolio but cannot silently replace its artifacts. Section 4.12 adds the matched speculative comparison required for a broader competitiveness claim.

After a terminal failure or recorded retirement, drain all noncancelable jobs for that subject before returning to candidate selection. Preserve their actual expenditure and applicable receipts. Never launch a second copy of a running method/input/attempt identity. Operational timeouts are execution failures, not a cost-free scheduling cancellation.

Persist the selected portfolio and its continuation when forming a candidate. The active-candidate branch executes that plan, or repairs it with the subject fixed under that method's original continuation objective and horizon. It must not replace J's deliberate reusable portfolio with a cheapest-current-subject plan after an arrival or partial completion. Include a test where an unrelated arrival occurs between FORM and the first check, and another between the two narrow checks in Section 4.8; both must preserve the registered decision semantics.

~~~text
propose(observation):
    if transition_is_executing(observation):
        return observe_or_advance_authorized_transition()
    if ready_active_candidate(observation):
        return REQUEST_ADMISSION(active_candidate_id)
    if active_candidate_exists(observation):
        return next_missing_actions_for_same_pinned_subject_or_block()
    pool = common_candidate_pool(observation)
    proposals = []
    for x in pool:
        for p in shared_fixed_subject_frontier(observation, x):
            for continuation in legal_second_candidates_or_terminal(x):
                rollout = evaluate(p, continuation, B_ref_terminal_policy)
                if rollout.respects_planning_constraints:
                    proposals.append(rollout)
    if no proposal was found:
        return common_fallback_or_explicit_blocked_reason()
    best = stable_argmin(proposals, registered_objective)
    return first_legal_action_set(best)
~~~

The controller budget is calibrated from measured planning throughput and removable verification cost. Use deterministic node-expansion limits in simulation to avoid machine-speed-dependent decisions; physical runs additionally enforce a watchdog. Log fallback count, search nodes, forecast error and controller wall/CPU time.

### 4.8 An exact mechanism example, not a claimed benchmark win

Two required sequential subjects are \(x_1=(a_1,b_1)\) and \(x_2=(a_1,b_2)\). Assume the release sequence is externally required, for example by a staged migration. Each needs evidence for obligations A and B. Approved options:

| Action | Covers | Scope | Resource cost |
|---|---|---|---:|
| AB | A and B | a and b | 5 |
| A-only | A | a | 4 |
| B-only | B | b | 2 |

Whole-run AB evidence cannot be salvaged after b changes. A-only evidence can. Exact enumeration performed during preparation gives:

| First portfolio | Second portfolio | Incremental costs | Total |
|---|---|---:|---:|
| AB | AB | 5, 5 | 10 |
| AB | separate | 5, 6 | 11 |
| separate | AB | 6, 5 | 11 |
| separate | separate | 6, 2 | 8 |

Thus per-candidate cost minimization can lose to cross-candidate planning, even with correct caching. The first admission costs one extra resource unit; a tighter deadline or planner overhead can erase the advantage. If the two subjects may instead be combined into one release, ordinary batching may win. A simple amortized-reuse rule can also attain 8. Include those counterexamples.

This is an exact four-case constructed calculation on Linux/Python 3.12.14, not measured software performance, a new theorem, or validation of the proposed online controller.

B5 can also attain 8 through its receipt-aware terminal rollout. This example is not expected to separate J from B5; that requires cases in which the second candidate decision matters. If none arise in an expressive workload, additional horizon has no demonstrated benefit.

To test the full proposed mechanism, extend the fixture with a required bridge release plus an unrelated third work item that may join either stage. Include multiple approved narrow/combined verification portfolios and finite workers. Enumerate all legal cohorts and verification plans. The test is informative only if alternative boundaries are genuinely legal, preserving the same original work and obligations.

### 4.9 Simulator and causal observation contract

Use integer microseconds for normal simulation; use integer abstract ticks for tiny exhaustive cases. Round measured positive durations upward and record the scale. Positive jobs occupy at least one tick. Do not round away setup costs.

Process equal-time event groups atomically in this order:

1. Assumption/policy revocations and expirations.
2. External arrivals and authorized deployed-state changes.
3. Job completions and cancellation acknowledgments.
4. Receipt validation, resource-ledger updates and gate recomputation.
5. Admission requests and new planner actions.

Expiry intervals are half-open: evidence whose expiry equals use time is expired. Completions at that time do not bypass an earlier revocation. A new arrival only changes the backlog for an already pinned subject.

Methods receive serialized observations containing current time, deployed subject, pending work, shared candidate pool, visible receipts, running jobs, completed history and resources. They never receive the future event queue, hidden fault labels, evaluator oracle state or future realized durations.

For paired simulated comparisons derive randomness from a stable hash of (episode seed, method-independent action ID, input projection hash, attempt number). Do not consume a single RNG stream in policy-dependent order. For shared congestion, add an exogenous time-indexed environmental process; action-keyed independent draws alone cannot express correlated slowdowns.

Counterfactual outcomes are defined by a frozen structural fixture/oracle, not by “a batch fails if any individual commit failed.” Passing individual checks cannot establish that a new combination passes. Every physical candidate claimed as validated must actually be executed or covered by valid explicit evidence.

Planning is not instantaneous. Model one controller lane and W verification lanes within the frozen machine budget. Calibrate deterministic planner wall/CPU charges from operation counts, lookup counts and search-node counts on development measurements; use the same accounting model for every policy. Advance virtual time through planning, processing external arrivals and job completions normally. At planning completion validate the proposal against the now-current state; stale reservations or subjects trigger revalidation/replan and retain spent cost. In physical runs use measured planner elapsed/CPU time. Report sensitivity to the calibrated overhead bounds. Decisions never execute retroactively at the observation timestamp.

Allow at most one planning invocation in flight. Events received during that interval update authoritative state and coalesce into one pending replanning notification; they do not create overlapping controller jobs. Store the observed snapshot digest and the resulting plan separately.

Represent not-yet-launched job times as nonnegative offsets from decision completion. At actual PLAN_COMPLETE, anchor those offsets to the current time, preserve job order, and recheck prerequisites, resources and receipt-use intervals. Already running jobs retain their real timestamps. Forecasts include the development-calibrated planning delay; forecast error is allowed and measured. A stale assumption requires repair, but a harmless shift caused solely by the planner's own elapsed time must not trigger an endless restart. Test progress in a no-arrival episode with nonzero planning time, as well as revocation during planning.

### 4.10 Persistent records and interfaces

The following is a proposed project schema to implement, not an existing package API.

| Record | Required fields |
|---|---|
| WorkItem | item_id, arrived_us, component deltas, predecessors, atomic_group, weight, deadline_us or null, acceptance mapping |
| Subject | subject_id, artifacts, lock/runtime IDs, config, schema, requirement/policy IDs, base state, deployment plan |
| Method | method_id/revision, claim/coverage, complete scope or unknown, prerequisites, resource demand, cancellation rule, duration model |
| Receipt | receipt_id, job_id, method/oracle revisions, claim, subject projection, result, assumption IDs, observation/completion times, validity interval or null, provenance |
| Job | job_id, immutable input IDs, attempt, submitted/start/finish times, reservations, consumed resource, outcome, cancellation state |
| Decision | decision_id, observed-state digest, candidate/plan considered, chosen action, reasons, forecast, search status, elapsed planner cost |
| Admission | candidate ID, base-state version, policy digest, exact receipt IDs, gate verdict/reasons, independent evaluator verdict |

Use enums rather than overloading null: result is PASS, FAIL or INDETERMINATE; job state includes PENDING, RUNNING, SUCCEEDED, FAILED, CANCEL_REQUESTED, CANCELLED and INTERRUPTED. Receipt applicability is separately YES, NO or UNKNOWN with a reason code.

Every external job, child process and RPC has a frozen method-specific wall timeout and bounded cleanup grace, calibrated on development data. A timeout yields FAILED or INDETERMINATE, preserves resource use and triggers owned-process cleanup plus state reconciliation. Core “no cancellation” means no discretionary policy cancellation; it does not allow a hung worker to hold the experiment indefinitely. Record timeout/cleanup settings in the method catalogue and run manifest, and use identical rules for all methods.

Validate schemas on ingestion. A local hash chain detects accidental changes; it does not authenticate a malicious verifier. The local PoC's trust boundary is a controlled benchmark process. A production deployment would require authenticated principals, authorized policy changes and an actual atomic promotion guard.

### 4.11 Admission atomicity

Before recording local release readiness, re-evaluate the exact policy and relevant state versions. Atomically record receipt references, admission intent, budget accounting and a reservation against the base-state version. If the version guard changed, reject that attempt and replan.

Use explicit VERIFYING, READY, EXECUTING, DEPLOYED and FAILED/RECOVERING states. The admission transaction does not enact a distributed rollout. Execute the approved path step by step, persisting actual intermediate states and step IDs. Advance the completed deployed-base identity and credit work only when the path finishes successfully. An interruption or partial rollout preserves the actually reached state and requires recovery/revalidation from that state; it cannot roll back physical effects merely by reverting a controller record.

Represent rollout and recovery steps as metered jobs with kind ROLLOUT or RECOVERY in the same resource ledger. They occupy declared slots, time and state locks; their cost appears once in the job sum. Controller/admission CPU remains the separate controller term. Hypothetical continuations include each step's predicted occupancy and duration, and reach the next deployed base only after predicted transition completion. A verification-only plan cannot receive free deployment or recovery.

For physical transition execution, a local compare-and-swap/transaction covers the benchmark's full declared precondition. A preflight observation cannot replace atomic enforcement of mutable conditions. This issue is already studied in portfolio/commit-gate work. [R29]

Idempotent decision and job IDs prevent duplicate work or duplicate delivery credit after an interrupted response. Do not rerun a completed transition merely because its acknowledgment was lost.

### 4.12 Matched speculative extension

The one-candidate core is the smallest mechanism experiment. A claim beyond competent speculative queues requires this additional local phase before final broad confirmation.

Permit up to two simultaneously verifying immutable candidate subjects, one controller lane and the same total W workers; only one deployment may execute at a time. Give J and the queue/batching controls identical launch, retirement and cancellation capabilities. A verified successor may depend on a prospective predecessor base; it cannot be admitted until that predecessor is actually deployed with the exact expected state.

If a predecessor fails or a different candidate deploys, re-evaluate each successor's base-bound transition evidence. Keep unrelated steady-state receipts, preserve actual job costs, and rebuild only the required evidence. A late completion retains its old subject identity. Use the same deterministic cancellation policy for all methods; alternatively keep cancellation disabled for all and measure its cost.

Implement B4's real speculative prefixes and supported TestCaseBatching behavior here, not a silently serialized substitute. J's action search may allocate a free worker to either active subject; its oracle/control counterparts receive the identical choice. Calibrate and freeze this phase separately. A core-only result is expressly limited to non-speculative pipelines; a negative speculative comparison blocks a broader superiority claim.

## 5. Baselines, workloads and confirmatory evaluation

### 5.1 Comparison contract

Every competitive method receives the same:

- Completed work, arrival events, deadlines, atomic groups and permitted candidate/transition alternatives.
- Policy resolver, applicability kernel, provenance validation and independently checked acceptance requirements.
- Initial cache, valid reuse rules, job catalogue, resource limits and measured runtime estimates.
- Setup costs, failure handling, retry limits and common generic implementation optimizations.
- Opportunity to tune on development data and validation with recorded cost.

The proposed controller cannot gain extra semantic information, exact future runtimes or free optimization compute. A stronger method may deliberately spend more controller CPU, but that cost must be charged.

### 5.2 Mandatory local baseline families

| ID | Implemented behavior | Purpose / reproduction standard |
|---|---|---|
| B0 | Moving-head restart/rerun diagnostic | Validate source-model pathology; excluded from superiority claims |
| B1 | Oldest eligible atomic closure, immediately pinned, cache/reuse, shared fixed-subject solver | Strong simple no-batching reference |
| B2 | Tuned size/window/deadline batching, pinned, shared solver | Competent conventional pipeline |
| B3 | BatchAll-style adaptive cohorts and lightweight outcome-adaptive batching; TestCaseBatching in Section 4.12 | Published batching competitors [R2, R30]; policy adapters in common engine |
| B4 | Dependency-aware speculative queue prefixes in Section 4.12 | Required strong-comparator phase for claims beyond non-speculative pipelines; Zuul-inspired, not a product benchmark [R5] |
| B5 | Myopic enumeration of the same next subjects/portfolios, with the shared solver and B_ref terminal rollout | Strong one-decision control; AAS-inspired restricted inner adaptation [R1] |
| B6 | Simple amortized-reuse heuristic with deadline guard | Tests whether a cheap rule obtains the same across-release benefit |
| J | Proposed bounded joint release/verification planning | Experimental method |
| O | Exhaustive bounded offline oracle with future knowledge | Headroom/regret diagnostic only |

B5 enumerates the same currently legal next cohorts, approved paths and portfolio frontier as J, but explicitly searches only one admission before the B_ref terminal rollout. It uses the same objective and service constraints. J explicitly searches two admissions before that identical terminal policy. This stronger control prevents crediting J merely for considering more next-candidate choices. Also retain the simpler decoupled “batch first, optimize fixed subject second” policy among B2/B3 configurations.

B1–B4 select the cheapest currently feasible member of the shared portfolio frontier, breaking ties by earlier completion and stable IDs; a cost/latency-weighted alternative may be tuned as a separately identified configuration. Do not mix named published policies into an undocumented hybrid.

B6 follows one realizable B_ref continuation through observed pending work. For each new action j, count \(n_j\ge1\) admissions along that continuation for which its receipt remains applicable, including the current admission. Mutually exclusive alternative candidates do not each earn reuse credit. Rank portfolio P by \(\sum_{j\in P}c_j/n_j\) plus the common registered delay penalty; deduplicate shared actions and give existing receipts zero incremental execution cost. Enforce the same policy/deadlines and stable ties. This rule can match the 8-unit constructed example.

Risk-aware batching [R3] is included as a B3 extension if usable risk scores exist. Train a small, interpretable predictor only on training history, or use prior observed failure rates; give those same scores to J. Do not claim a faithful reproduction of the paper's model-training results when using a different predictor. If none is available, retain the risk-independent batching baseline and state the missing comparison.

The inspected public baseline implementation associated with this research line [R42] uses a timing model that appears to divide accumulated test work by CPU count; this need not equal discrete-worker makespan. Preserve its policy decisions while using the common finite-worker engine. Differential-check one-worker and homogeneous cases, document every adaptation, and do not compare the original simulator's virtual times directly with physical Mac timings. A possible batch-cap edge condition in the inspected code also requires a boundary test.

Full Zuul, commercial products, AAS implementation, or broader published results are reference comparators until actually run faithfully. Lack of a local reproduction limits the claim; it does not justify calling an approximation the state of the art.

### 5.3 Workload ladder

**L0 — Exact bounded cases.** Use hand-derived and exhaustive episodes to establish semantics, policy equivalence, correct accounting and oracle behavior. Include impossible policies, deliberate waiting, shared coverage, resource contention, no-reuse/all-reuse, unrelated churn, staged migration and the four-case example. This is a harness gate.

**L1 — Generated event episodes.** Generate structure as well as event noise: component dependency graphs, atomic groups, coverage alternatives, service times, arrival processes and approved transition plans. Use independent episode identities. Include every predeclared regime in the aggregate, not only favorable high-churn cases.

Required regimes are no bottleneck; high branch churn with pinned candidates; low/high finite-worker load; positive runtime floors; heterogeneous checks; overlapping versus disjoint coverage; correlated/bursty arrivals; shared setup costs; staged migrations; narrow versus conservative full-subject reuse; and misspecified runtime predictions.

Set workload scales from dimensionless quantities measured in pilot: offered verification work per worker capacity, maximum/minimum check-duration ratio, reusable fraction under B5, and atomic-group size. Use a development-frozen grid spanning below, near and above measured saturation. Do not assert that synthetic parameter values describe industry.

**L2 — Actual native service execution.** Build a small API, worker and schema fixture with two replicas of each executable service, SQLite, localhost communication and immutable candidate directories. It is an executable constructed workload, not an industrial benchmark.

The frozen behavioral specification includes:

1. Tenant isolation for requests, status reads and effects.
2. Repeated (tenant, idempotency-key) requests create at most one logical job.
3. Duplicate delivery and worker retry produce at most one committed external-effect record.
4. Each successfully acknowledged job eventually reaches a terminal status under the bounded fair schedule.
5. Expand/migrate/contract steps preserve old-reader/new-writer compatibility where the plan permits them.
6. Crashes before/after transactional boundaries and rollout interruptions preserve the specified recovery behavior.

Use a separately coded reference state machine over request/effect/job event logs. Expected results must not call production functions. Deterministic barriers force check-then-insert races, commit-before-ack crashes and old-message/new-consumer combinations. Enumerate all schedules within the registered bound for core transition checks, then keep additional schedule/fault families for independent evaluation.

Construct versioned changes for API behavior, worker retries, schema indexes, schema format evolution and unrelated components. Correct throughput episodes contain only work verified independently as deliverable under at least one allowed path. Fault episodes contain intentional defects and test rejection, not throughput; repair arrivals, if studied, are exogenous and identical across methods.

Verification actions must perform real assertions, I/O, process restart and migration work. Do not add sleep to manufacture “compute savings.” Vary meaningful request counts or schedule bounds only before freezing the benchmark. Simulated duration sweeps are labeled separately.

**L3 — External transfer, conditional.** Prefer a permissioned real repository with exact subjects and actual check runs. If unavailable, use an inspected public sample such as Online Boutique only after native-image, license and local-cluster resource gates. Its documented local setup has material VM/storage requirements; it is optional. [R37]

An external seeded-service corpus can challenge oracle assumptions, but does not supply release histories. The MIT-licensed service artifacts in assure-harness can be reused without generating new model output; inspect the pinned service files and execute only the required tests. Its model-evaluation workflow is not a dependency of this PoC. [R21]

### 5.4 Public traces, licenses and counterfactual limits

GHALogs and the older continuous-integration CIBench are useful optional sources of arrivals and observed runtimes. Their accessible records do not supply complete release obligations, sound dependency scopes or outcomes for new combinations. Call resulting experiments **trace-calibrated simulation**, not production replay validation. [R35, R36]

The inspected GHALogs record lists roughly 1.1 GB of run metadata and 69.2 MB of repository metadata; its full raw logs are approximately 142.3 GB. CIBench lists an approximately 888.9 MB processed archive versus 38.3 GB of raw logs. Do not download the large raw archives for the core. The accessible record rendering did not expose a usable license value; verify an explicit license before acquisition/use and preserve it.

For any external artifact:

1. Resolve the official source and exact revision/version.
2. Record license text or the absence of permission, redistribution conditions, compressed size and checksums.
3. Inspect schema and a bounded sample before ingestion.
4. Stream records; do not assume archive size equals peak RAM or expansion size.
5. Freeze inclusion/exclusion rules before examining comparative outcomes.
6. Reject unavailable or unlicensed assets; continue the core synthetic/executable study with a correspondingly narrower claim.

For repository sampling, publish the source population and deterministic selection rule. Chronological splits separate training, validation and final test; keep repositories or independent release episodes intact where possible. The same event history under several seeds is not several independent repositories.

### 5.5 Metrics and accounting

Primary per-episode cost:

\[
C_k=
\sum_{\text{jobs }j}(\text{reserved core slots}_j)
(\text{occupied seconds}_j)
+C_{\mathrm{controller,CPU}}
+C_{\mathrm{setup,CPU}}.
\]

This is a hybrid **charged core-seconds proxy**, not a measurement of energy or an exact cloud bill. Setup CPU outside a reserved job is charged once; setup inside a job's occupancy is not added again. Record actual user/system CPU separately from reserved worker-seconds. Persistent fixture processes belong to their reserved verification group. Controller processes run within the total local capacity plan; planner elapsed time delays decisions under Section 4.9.

Report \(Y_k=C_k/N_k\), where \(N_k\) is the original fixed count of offered deliverable work items. No deployment-count denominator. Also report absolute \(C_k\), all work delivered, mean/p95/maximum flow time, last-drain time, deadline misses, backlog area, wasted speculative/cancelled work, reuse hit cost, planning overhead, fallback frequency, peak RSS and bytes stored.

Readiness time R is the first instant the exact candidate passes the gate, including controller/admission processing. Delivery time T occurs only after its local transition completes successfully. Report implementation-to-readiness waiting R-a, readiness-to-completion T-R and total flow T-a separately; use total flow for anti-starvation/service guardrails and the planning objective. Credit each original work item once at T. Failed or superseded READY attempts earn no delivery credit. Include transition execution resources and time; do not mix endpoints across methods.

Safety/quality outputs are separate counts: wrong-subject acceptance, unknown accepted as yes, missing policy atom, forbidden transition, expired/revoked witness, policy-compliant behavioral fault, false reuse and unnecessary invalidation. Record fault families and triggering schedules. A full audit is itself a bounded empirical oracle, not omniscience.

Offline evaluation-only audits are identical for all methods and excluded from operational-policy cost, with their total experimental cost reported separately. If a method needs such an audit to make its admission safe, it becomes an operational check charged to that method. Do not use its hidden result to help the controller.

### 5.6 Service envelope and full drain

Use owner-provided absolute SLOs/deadlines when available. Otherwise the default construction is: run B1 on the development population; use its stratum-weighted mean and p95 total flow as the primary benchmark service envelope; select the lowest-cost B2/B3 configuration satisfying that envelope as B_ref, falling back to B1 if none qualifies. Freeze this rule before viewing J outcomes. Record maximum observed age for the watchdog and choose any tighter/looser sensitivity envelopes prospectively. These are experimental conditions, not customer commitments. Without owner deadlines, do not manufacture a contractual deadline; retain the watchdog, full-drain cap and mean/p95 guardrails.

All arrivals stop at registered horizon \(T\); then drain all deliverable items. A fixed drain cap is set using baseline pilot and tiny-instance feasibility. An algorithmic timeout, deadlock or missing delivery remains a failed/censored episode, with expenditure and backlog retained. Never drop it or count its partial cost as a saving.

For an unconditional efficiency-win claim all required episodes must meet the registered feasibility policy. If failures occur, lead with failure/feasibility results; paired costs on jointly completed episodes are descriptive unless that population was independently defined beforehand.

A resource reduction with increased latency is a tradeoff unless it satisfies the predeclared service envelope and the exact claim's guardrails. “No significant latency difference” does not establish equivalence.

### 5.7 Tuning, splits and fixed-sample confirmation

1. Create separate development, validation and protected confirmation episode manifests. All template/topology and fault-family splits are explicit. Freeze seeds and data hashes.
2. Apply Section 5.6 in order: B1 calibration, baseline-derived service envelope, B2/B3 tuning, then B_ref selection/freeze. Tune remaining baselines and J within that envelope; do not use J outcomes to set its own service target. Share generally useful optimizations. Start with logarithmic size/window grids and refine around validation optima, recording effort. Expand a baseline's range when its best configuration lies on a boundary.
3. Select B* on validation as the lowest-cost feasible competitor under the primary service envelope. In the one-candidate core, eligible families are B1, B2, non-speculative B3 configurations, B5 and B6. In the matched speculative phase, all of those plus B4 and TestCaseBatching are eligible with the identical Section 4.12 capabilities. Select and freeze a separate B* for each phase and workload population; B0 and clairvoyant O are not competitive selection candidates. Freeze its implementation, inner solver, parameters and training metadata before confirmation. Retain all baseline curves.
4. Freeze J, hypotheses, metrics, service envelope, guardrail margins, exclusions, sample size and analysis code. Store a signed-by-hash protocol manifest. Hashing is a reproducibility record, not an external preregistration service.
5. Run the complete fixed confirmation set without comparative interim selection. Operational monitoring may detect crashes/resource pressure; it must not stop for a favorable effect.

The independent unit is a generated episode sampled from the declared structural population, or an independent repository/release block in transfer work. Repeated timings of the same episode are nested measurement replicates. Seeds varying only runtime noise within one fixed topology support conditional claims about that topology, not broad generalization.

Use the same pilot-selected work-item count N0 in every generated confirmation episode. For paired completed episodes define \(D_k=Y_{J,k}-Y_{B^*,k}\), in charged core-seconds per original work item. The primary estimand is the predeclared weighted arithmetic mean of D across workload strata. Also report the ratio of correspondingly weighted arithmetic mean costs. For external episodes with unequal work counts, use a predeclared ratio of weighted total cost to weighted total offered work, and bootstrap that complete ratio.

Primary report: arithmetic resource difference, arithmetic cost ratio and 95% confidence intervals. Geometric ratios are secondary: they can improve while total compute grows, so they cannot establish the customer promise. Aggregate repeated physical timings within episode first. Use a cluster bootstrap at the independent episode/repository level, with paired methods kept together; for stratified generated families, resample within each frozen stratum and retain fixed population weights. Use 10,000 bootstrap resamples and a fixed analysis seed. Report percentile intervals plus a paired t-interval sensitivity where its assumptions are reasonable; material disagreement triggers distribution inspection and cautious interpretation.

Define latency tails from the weighted pooled distribution of original item flow times, not an average of episode p95 values. In generated stratum s with n_s episodes of N0 items and frozen weight p_s, each item has weight p_s/(n_s N0). The p95 is the smallest value whose cumulative weight reaches 0.95. For repeated physical timings, each item's repetitions divide its fixed item weight equally. Compute the mean from that same distribution. Resample whole paired episodes and recompute the pooled mean and quantile in every bootstrap replicate. External unequal-sized episodes use their separately registered item/population weights; do not silently change the estimand.

Calibrate sample count from pilot dispersion, before protected outcomes:

\[
n \gtrsim
\left[
\frac{(z_{0.975}+z_{0.90})s_D}{\delta_{\mathrm{abs}}}
\right]^2.
\]

Here \(\delta_{\mathrm{abs}}\) is a scientifically meaningful absolute saving per work item to resolve, supplied by break-even economics or a declared precision objective. It is not an arbitrary “must improve 30%” hurdle. Use a conservative upper estimate of pilot variability and validate power/coverage with pilot-resampled synthetic effects and the actual stratified estimator. If no practical effect threshold exists, select a target confidence-interval half-width \(h\), use \(n\gtrsim(z_{0.975}s_D/h)^2\), and report achievable precision rather than claiming powered superiority.

For illustration only, \(s_D=2\) charged core-seconds/item and an absolute effect of 0.5 imply approximately 169 independent episodes at 90% nominal power before multiplicity adjustment. These values are not measured for this project. Recompute after pilot. If the required sample cannot run locally, reduce timing variance or narrow the population before freezing; otherwise label the study underpowered. Do not substitute thousands of correlated seeds.

A confirmed primary improvement requires an arithmetic cost-difference upper 95% bound below zero (with the arithmetic ratio also reported), no admission/behavioral violation within the registered bounded population, complete delivery, and the frozen latency/deadline guardrails. For a “no slower” claim use one-sided simultaneous upper bounds for mean and p95 latency difference, adjusted across those guardrails. Margins must come from the owner's SLO or predeclared benchmark preference; if no nonzero margin is justified, use zero and accept possible inconclusiveness.

Keep J-versus-B* as the primary comparison. A claim that additional across-release lookahead contributes also requires the registered J-versus-B5 comparison; if B*=B5 these coincide. When both are distinct confirmatory superiority claims, use Holm's procedure over the two tests and conservative Bonferroni simultaneous intervals. Plan sample size using the corresponding adjusted quantile. Correct further confirmatory pairwise claims or label them exploratory. Never choose the winning workload subset after results. After a methodological change, use an amendment and fresh independent confirmation.

### 5.8 Fixed-budget customer-promise experiment

As a secondary registered test, give every method the same resource budget \(B\), arrivals and horizon \(H\). Count original completed work items at H. Before launching any charged operation, reserve its finite maximum resource charge, including setup, planner/admission work, rollout steps and cleanup. Development calibration sets a common catalogue of execution timeouts and cleanup grace limits; a job exceeding its timeout is terminated under the same rule for all methods, yields no passing receipt, and retains all incurred cost. Unbounded sampled durations are right-censored by this explicit execution policy, never treated as known finite worst cases.

At every launch require spent charge plus outstanding reservations plus the new reservation to be at most B. On completion convert its reservation to actual charge and release the remainder. Use slot-count times the timeout including cleanup for reserved jobs, and the analogous CPU upper charge for a one-lane controller operation. Rollout admission reserves the complete remaining transition and recovery allowance. Prevent further launches when a reservation cannot fit. If the operating system exceeds the enforced bound, retain the overrun and mark the budget-compliance test failed; do not conceal it by truncating accounting.

At H stop new policy actions and record completed items. Let already reserved operations finish or hit their registered timeout, charging their full tails to B; completions after H earn no H credit. Any subsequent full drain is a separately labeled supplemental run/expenditure with a new budget, not part of the fixed-B result. Report expenditure through H, reserved tail, final charged total, completed count, backlog area, oldest incomplete age and overdue count. Stratify by item class to expose easy-work preference. This directly tests “more completed work with the same budget,” while the full-drain primary protects against neglecting hard work.

### 5.9 Physical timing protocol

Perform one unscored smoke/warm-up pass, then randomize paired method order within each episode block. Reset fixture state and use identical declared cold/warm cache conditions. Do not run competing methods simultaneously on the laptop. Fix local worker concurrency and document plugged-in power mode, OS version and competing-load observations.

Use monotonic high-resolution wall timing, process CPU counters and process-tree RSS sampling. Include subprocess spawn, artifact materialization, hashing and result ingestion inside end-to-end boundaries. macOS process-spawn behavior differs from Linux; measure it locally. [R31]

Choose repeated timing count from pilot relative variance; repetitions reduce timing noise, not increase independent sample size. Thermal drift, swap or external load can invalidate a timing block under frozen mechanical rules; retain the block, reason and rerun. Never discard a slow block because it favors the baseline.

## 6. M3 Max feasibility and hardware-adaptive execution

### 6.1 Stack decision

Use native CPython, its standard-library event heap, subprocess facilities, JSON, hashing and SQLite. Add pytest for implementation checks and optionally NumPy/SciPy for analysis or reproducing the supplied numerical scripts. No model assets or acceleration backend are required.

The official Python macOS documentation describes native Apple Silicon universal2 binaries. That supports this stack choice; it is not a target-machine measurement. [R31] Prefer an already installed native supported CPython 3.12 or 3.13 environment if the selected packages support it. Otherwise install a current supported native CPython from its official distribution and record the reason/version. Do not silently run under Rosetta.

Use OR-Tools only if exhaustive search becomes the measured bottleneck on informative instances. Its official Mac installation documentation includes arm64 support, but the exact selected wheel and operations must be smoke-tested on the target. [R32] If unavailable, retain the small exact enumerator. Replacing an exact solver with an unchecked heuristic is not a faithful fallback for an exactness claim.

The controller database uses one writer and default rollback journaling. SQLite's current documentation reports a WAL-reset defect fixed in 3.51.3 and specified backports. Do not enable WAL without checking the bundled runtime against a documented fixed version. [R33] SQLite is a local database here; no distributed database semantics are inferred.

Dependencies are selected once at bootstrap, pinned with transitive versions and downloaded-wheel hashes, and retained in a wheelhouse where licenses allow. No version number or wheel filename is prescribed without local discovery. Preserve upstream license notices. The original standard-library core remains runnable without optional scientific packages.

### 6.2 Existing discovery commands

The following are existing shell/Python commands, not project commands. Run them in a new project directory on the target Mac. If a command is absent, record that failure and use the stated discovery fallback rather than fabricate output.

~~~bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.ncpu
sysctl -in sysctl.proc_translated
df -h .
python3 --version
python3 - <<'PY'
import json, platform, sqlite3, sys
print(json.dumps({
    "python_executable": sys.executable,
    "python_version": sys.version,
    "machine": platform.machine(),
    "platform": platform.platform(),
    "sqlite_version": sqlite3.sqlite_version
}, indent=2))
PY
~~~

The Rosetta indicator may be unavailable on a native process; that absence alone is not a failure. Combine architecture and executable-build evidence. Discover memory pressure with Activity Monitor or the available macOS command-line facilities, recording only aggregate resource measurements needed for the experiment.

Create a virtual environment using the selected interpreter's absolute path, then record its package inventory. Use that interpreter consistently in every command. Do not modify Apple's system Python. Smoke-test process spawning with a main guard, localhost requests, database commit/rollback, canonical hashing, event ordering, resource sampling and clean shutdown before any tuning or large download.

### 6.3 Memory model and configuration rule

Estimate and then measure

\[
M_{\mathrm{peak}}\approx M_{\mathrm{controller}}
+M_{\mathrm{events}}+M_{\mathrm{receipts}}+M_{\mathrm{candidate\ manifests}}
+W(M_{\mathrm{check}}+M_{\mathrm{services}}+M_{\mathrm{DB}})
+M_{\mathrm{solver}}+M_{\mathrm{reserve}}.
\]

Do not materialize a complete source tree for every hypothetical candidate. Share immutable source blobs and compact manifests; create working directories only for executed candidates. Stream logs and keep bounded in-memory indexes.

Initial design targets, **not measurements**, are below 1 GiB for controller/simulator and roughly 0.1–0.5 GiB for a small fixture/check group. Python-object and database-cache overhead can exceed these estimates. For example, 100,000 event objects at an assumed 1 KiB each consume about 98 MiB before auxiliary indexes; calibrate bytes/event with actual process RSS and allocation measurements.

Begin with one verification group. Choose a project memory budget no greater than half physical RAM and no greater than currently available memory minus an 8 GiB operating reserve. Treat “available” as a documented conservative estimate, not an exact OS guarantee. If less than 2 GiB is usable, run tiny simulation first and defer the physical multi-process stage until it fits; never kill unrelated processes.

After W=1 and W=2 calibration, select the largest tested W whose measured peak plus 30% contingency fits the project budget without sustained swap growth or memory pressure. Use a small initial ceiling of four physical verification groups until profiling demonstrates headroom; this is an implementation starting point, not a scientific scale limit.

Also enforce a CPU admission cap: one controller slot plus the sum of declared slots for simultaneously active verification/rollout jobs must not exceed the discovered logical CPU count. Calibrate each job group's demand including child processes and native-library threads; W counts verification lanes, not permission for unbounded child parallelism. Record both reserved-slot charges and measured process-tree CPU. This is a conservative scheduler cap, not a claim of exclusive physical-core affinity or equal performance across the M3 Max's core types. Pilot throughput and contention determine the final shared concurrency setting.

On pressure, reduce simultaneous episodes/groups first, then reduce candidate-search breadth and exact-oracle bounds while preserving exactness of the remaining bounded cases. Stream more episodes sequentially. Do not omit obligations, alter quality policy, weaken scope precision or delete adverse cases to fit RAM.

### 6.4 Storage, runtime and scale estimates

| Stage | Initial estimate / governing model | Calibration and fallback |
|---|---|---|
| L0 semantics and oracle | Small source/data footprint; exponential search time, not GPU memory, is the risk | Record states/sec and largest completed exact grid; reduce dimensions, retain explicit bound |
| L1 event simulation | Memory proportional to active events/receipts, not total episode count | Stream output; estimate seconds/episode and bytes/episode on several regimes |
| L2 native fixture | Target 2–5 GB project footprint before optional data; multiple source snapshots and raw logs dominate | Measure peak process tree and per-episode logs; deduplicate immutable source blobs |
| Optional numerical reproduction | Small dense systems in supplied checks | Use selected native scientific-package wheels; these are not core scheduler dependencies |
| Optional public metadata | More than 1 GB compressed may be involved | License gate, size preflight, streaming parse; no raw-log archive requirement |
| Optional Online Boutique | Official local Docker instructions specify material VM resources and about 32 GB disk [R37] | Inspect actual image architectures and smoke-test; skip if it cannot fit faithfully |

For every substantial run, estimate

\[
\widehat T_{\mathrm{total}}
=\sum_{\text{methods, strata}}n_{ms}r_{ms}\widehat t_{ms}
+T_{\mathrm{tuning}}+T_{\mathrm{analysis}}+T_{\mathrm{audit}},
\]

using pilot means and upper quantiles, including setup and failures. Here r is nested timing repetition. Compare required sample size to measured throughput; a runtime budget follows from scientific sensitivity, not an arbitrary short deadline.

Reserve at least twice the pilot-projected output footprint in addition to the operating-system free-space reserve. Stop gracefully before the reserve is crossed. An archive with uncertain expansion size is not admitted until a sample or format inspection establishes a defensible bound.

Virtual worker counts in simulation can exceed the laptop's physical workers; such results remain simulation. Actual local concurrency claims apply only to measured Mac configurations. If an effect requires larger dependency graphs, L1 can test its structural prediction while L2 validates a smaller executable mechanism. State exactly which scaling claim remains unverified.

### 6.5 Measurement and compatibility gates

Record CPU time and wall time separately. On macOS, verify the units of resource counters instead of assuming Linux ru_maxrss units. Report sampled concurrent process-tree RSS rather than summing unrelated lifetime maxima and calling it an observed simultaneous peak. Sample at a fixed interval, record it, and disclose missed short spikes.

Use top-level picklable process targets and a main guard for spawn. Avoid inherited mutable global state as a correctness dependency. Fix thread counts for optional native numerical libraries; oversubscription is an experimental variable only when explicitly studied.

No Metal, CUDA, quantization or reduced precision is necessary. Integer identities and event ticks remain exact. Numerical optimization tolerances, if introduced, must be validated by the independent integer schedule checker.

## 7. Ordered work packages and decision gates

### 7.1 Repository and command contract

Create the following project structure. These are paths and commands the coding agent must implement; no such package is asserted to exist already.

| Path | Responsibility |
|---|---|
| README.md, pyproject.toml, requirements.lock, wheelhouse/ | Bootstrap, exact dependencies and reproduction instructions |
| protocol/plan.md, freeze.json, amendments.jsonl, decisions.jsonl, progress.json | Durable scientific and execution state |
| schemas/ | Work, subject, receipt, job, event, admission and run-manifest schemas |
| src/readiness/subjects.py, policy.py, evidence.py | Canonical identities, obligation resolution and applicability |
| src/readiness/sim.py, ledger.py, executor.py | Event semantics, resources and physical jobs |
| src/readiness/methods/ | B0–B6, J and common inner solver |
| src/readiness/oracle/ | Independent finite-grid scheduler and behavioral reference |
| fixtures/job_service/ | Native API, workers, schema variants, patch catalogue |
| tests/ | Identity, policy, event, oracle, adapter, concurrency and recovery checks |
| configs/ | Hardware, development, validation, confirmation and robustness configs |
| data/sources/, derived/, splits/ | Exact source/license manifests and transformations |
| runs/<run_id>/ | Immutable configuration, events, decisions, resources, results and failures |
| reports/ | Generated tables, figures, interpretation and reproduction report |

Implement a small command-line entry point, for example:

~~~bash
python -m readiness bootstrap --output runs/bootstrap
python -m readiness validate --suite core
python -m readiness calibrate --config configs/development.json
python -m readiness tune --config configs/development.json
python -m readiness freeze --config configs/confirmation.json
python -m readiness run --freeze protocol/freeze.json
python -m readiness resume --run-id RUN_ID
python -m readiness report --freeze protocol/freeze.json
~~~

These are **proposed interfaces**, not commands to run before implementing the modules. Each command validates its inputs, records its source/config hash and emits a nonzero exit code on an invalid scientific or execution precondition.

Create src/readiness/__init__.py and __main__.py before these invocations. In an isolated development shell, set PYTHONPATH to the absolute project src directory, or install the completed package into the chosen virtual environment. The Section 6 discovery snippet runs independently before that package exists. Bootstrap does not depend on its own output manifest.

A configuration must explicitly specify schema version, phase, data split, method/config IDs, policy/catalogue hashes, resource limits, event unit/order, candidate-pool rule, oracle bound, cache condition, service envelope, cost accounting, seeds, sample counts, timing repetitions, retry/cancellation behavior and output destination. Fail on missing required values. Optional unknown fields may be retained only when their semantics cannot affect a frozen run.

### WP0 — Freeze the research contract and source decisions

**Objective/risk:** avoid building a generic assurance product or claiming an occupied invention.

**Inputs/entry:** this plan; access to public sources. No large downloads.

**Actions:** create protocol/plan.md, a source ledger and decision log. Record the selected outer-planning hypothesis, B5/B* comparisons, fixed work denominator and quality scope. Add the exact source dates/versions and known implementation gaps. Confirm whether newer AAS/queue code is available; inspect only changes that could alter the comparator. Review the patent disclosure as technical prior art, without legal conclusions.

**Outputs/tests:** contribution-to-prior-art-to-test table; no broad “first evidence scheduler” claim; named current implementation and adapter versions or an explicit not-located status.

**Resources:** reading and small source downloads only.

**Gate:** if an equivalent outer controller exists, replicate it as the proposed method or baseline and frame the work as a replication/application study. Continue if a measurable practical question remains. Stop an algorithmic novelty claim when no substantive difference survives.

### WP1 — Discover hardware and establish the native runtime

**Objective/risk:** unverified laptop memory, interpreter, database and process support.

**Dependencies:** WP0 contract.

**Actions:** execute Section 6 discovery, create isolated environment, select/pin dependencies, smoke-test all critical operations and resource samplers. Implement bootstrap and machine manifest.

**Outputs/tests:** hardware.json, dependency lock and hashes, smoke-results.json, process-cleanup check, measured startup/RSS baseline.

**Resources:** W=1; no dataset/model download.

**Gate:** proceed only with native execution and faithful critical operations. Missing optional solver/scientific packages route to standard-library enumeration. If actual multi-service execution cannot fit, retain L0/L1 and mark H4 blocked; do not claim local feasibility by simulation alone.

### WP2 — Implement subjects, policy and the independent admission checker

**Objective/risk:** wrong-version evidence, unknown treated as true, policy weakening.

**Dependencies:** WP1.

**Actions:** implement schemas, canonicalization, input projections, receipt status, policy clauses, revocations and transactional admission. Write a separately coded slow checker that reconstructs full expected subjects and does not call the production applicability function.

**Outputs/tests:** all Section 8 identity/policy cases; canonical digest fixtures; explicit UNKNOWN reason records; proof sketch of the conditional reuse invariant.

**Resources:** bounded small cases, no optimization.

**Gate:** zero disagreements/false admissions on the exhaustive bounded suite. A failing case fixes the gate/harness before performance work. If a scope cannot be justified, replace it with full-subject scope for all methods, recording lost headroom.

### WP3 — Build the event engine, resource ledger and exact small oracle

**Objective/risk:** fictitious savings from timing, free cancellation, hidden future information or an incorrect oracle.

**Dependencies:** WP2.

**Actions:** implement equal-time event ordering, immutable job capture, keyed randomness, core resource occupancy and full-drain accounting. Implement independent finite-grid enumeration including idle schedules. Add a read-only observation boundary. Reproduce the analytic moving/pinned controls and exact 10/11/11/8 portfolio table.

**Outputs/tests:** known-optimum fixtures, identical-policy placebo, intentional worse-policy sensitivity, conservation of resources, leakage canary, impossible-instance classification, checkpoint determinism.

**Resources:** start three items/four actions/eight ticks; measure search expansion before increasing.

**Gate:** independent schedule checker agrees with all returned oracle plans, equivalent policies agree logically, and deliberate differences are detected. A grid limit yields unknown optimality, not a proof. No benchmark until this gate passes.

### WP4 — Implement and validate strong comparators

**Objective/risk:** gains caused by weak batching or a weaker inner solver.

**Dependencies:** WP3.

**Actions:** implement B1–B6 and shared fixed-subject frontier; inspect and pin published source where licensed. Differential-check batch/queue order on small known cases, including capacity boundaries and failure isolation. On the small development fixtures, run B1 first, construct the preliminary envelope by Section 5.6, then tune B2/B3 and freeze a provisional B_ref for WP6. Implement it without recursive calls. These development settings can be recalibrated in WP7 before protected confirmation; every dependent B5/B6/J configuration must then be retuned against the final B_ref.

**Outputs/tests:** adapter-provenance report, behavioral comparison tables, shared-information/cache audit, provisional B_ref selection and configuration hashes.

**Resources:** small policies first; no full PostgreSQL restore required.

**Gate:** published-policy adapters match documented decision rules in supported cases. If source cannot run, retain a disclosed adaptation and narrow claims. A vague or unverified comparator cannot support a superiority claim.

### WP5 — Build the executable service and fault oracle

**Objective/risk:** a simulator-only advantage or circular behavioral oracle.

**Dependencies:** WP2; can proceed in parallel with WP3–WP4 after schema freeze.

**Actions:** build the native service, versioned change catalogue, deterministic barriers and independent state machine. Define approved rollout paths and enumerate their reachable bounded states. Measure every action's scope, setup, duration and memory. Produce correct-change episodes and separately sealed fault families.

**Outputs/tests:** reference passes, each fault activates under a documented schedule, exact manifest/reproduction commands, action timing distributions, resource peaks and full-subject audit outcomes.

**Resources:** begin one fixture group and minimal request histories, then use Section 6 scaling.

**Gate:** all benchmark properties can be checked independently, and required cross-version interactions actually occur. If not, fix the proxy rather than adding optimizer complexity. A distributed-system claim remains excluded from SQLite-only evidence.

### WP6 — Implement J and measure residual headroom

**Objective/risk:** extra search provides no value after competent practice.

**Dependencies:** WP3–WP5; baseline gate passed.

**Actions:** implement Section 4 bounded lookahead, receipt-aware plan frontier, terminal rollout, deterministic search limits and safe fallback. Compare tiny cases to O; compare J with B5/B6 at equal search budgets and with full overhead.

**Outputs/tests:** regret/headroom plots, forecast-error logs, decision explanations, time/RSS per decision, nonrecursive terminal-policy tests.

**Resources:** calibrated node limits; sequential episodes.

**Gate:** if even O shows no residual benefit on expressive cases, document the boundary and do not expand expensive search. If only J fails while O has meaningful headroom, diagnose search/forecast quality. If benefit exists only against B0 or through unequal metadata, reject the claimed contribution.

### WP7 — Pilot, tune and freeze the confirmatory protocol

**Objective/risk:** insufficient sensitivity, baseline under-tuning or protected-data leakage.

**Dependencies:** WP4–WP6 correctness and local calibration.

**Actions:** apply the B1 → envelope → B2/B3 → B_ref ordering in Section 5.6, then tune remaining baselines and J. Estimate variance, within-episode timing noise, runtime and storage; calculate sample size; freeze B*, B5, J, workload weights, seeds, fault families, margins and analysis. Pilot the matched speculative extension before freezing any claim beyond the one-candidate core.

**Outputs/tests:** protocol/freeze.json with all hashes; power/precision calculation; projected time/disk range; blinded analysis fixtures with known effect and null effect.

**Gate:** proceed only when the fixed sample is locally feasible and sufficiently sensitive for the registered claim. Otherwise narrow the claim/precision prospectively or declare that confirmation cannot currently be afforded. A larger future effect is not a substitute for a sensitive test now.

### WP8 — Execute confirmation and physical replication

**Objective/risk:** favorable-only reporting and simulation-to-execution failure.

**Dependencies:** immutable WP7 freeze.

**Actions:** run all fixed L1 episodes and paired L2 blocks, preserving failures and adverse regimes. Monitor resource health without comparative peeking. Recheck admissions with the independent evaluator and generate primary/guardrail statistics from raw records.

**Outputs/tests:** complete run ledger, raw logs, paired estimates/intervals, feasibility table, all-work counts and cost decomposition.

**Gate:** classify using Section 10. A protocol bug invalidates affected results; retain them, amend, and obtain fresh confirmation. A valid negative result is reported, not repaired by retuning on the test set.

### WP9 — Attribution, robustness and clean reproduction

**Objective/risk:** mistaken mechanism attribution, fragile effects, inconsistent resume.

**Dependencies:** WP7 freeze; registered attribution runs may execute alongside WP8.

**Actions:** run Section 8 ablations and adversarial cases; conduct interruption/recovery at all transaction boundaries; reproduce selected episodes from a clean environment using saved assets and locks.

**Outputs/tests:** crossed-factor table, fault-family outcomes, difference between simulation and actual cost, reproducibility and resume report.

**Gate:** retain only claims supported after generic optimizations and strong controls. Failure confined to one mechanism narrows attribution; gate failure blocks release optimization use until corrected.

### WP10 — Optional transfer and adoption decision

**Objective/risk:** constructed workloads fail to represent customer economics.

**Dependencies:** WP8–WP9 useful bounded result plus explicit data access/license/resource gates.

**Actions:** instrument a permissioned repository in observation-only mode, or run a qualified public multi-service sample. Measure actual evidence reuse, runtime, integration effort and causes of readiness waiting. Counterfactual candidates require actual runs or separately justified applicable evidence.

**Outputs/tests:** transfer population definition, observed bottleneck decomposition, net-value interval and break-even range.

**Gate:** transfer failure is a boundary of the result. Stop a product claim when competent existing tooling captures the savings. Production write/deployment authority requires a separate authorized integration decision; it is not implied by successful local experiments.

## 8. Ablation, robustness and adversarial program

### 8.1 Attribution experiments

Run the following on registered paired episodes with identical work and policy. Do not label a weakened control a competitive baseline.

| Intervention | What it isolates | Interpretation |
|---|---|---|
| J with one admission of explicit lookahead = B5 | Additional future candidate choice | Equality means extra horizon has no demonstrated value |
| Fix candidate sequence, vary portfolio planning | Across-release verification choice | A gain is portfolio/reuse scheduling, not cohort selection |
| Fix portfolio rule, vary cohort/path choice | Release boundary value | A gain is candidate choice, not a stronger inner solver |
| Common full-subject binding | Dependence on narrow justified reuse | A gain disappearing here identifies metadata requirements |
| Give all methods finer per-claim receipts where the runner can emit them | Whether an easy instrumentation change solves the problem | If it erases J's gain, prefer that simpler improvement |
| Match solver node/time budget | More search versus better decisions | Report quality/cost frontier, not free-compute improvement |
| No forecast versus shared fitted forecast | Prediction value | Credit forecasting separately; a misforecast must not bypass the gate |
| Measured planner overhead versus zero-overhead diagnostic | End-to-end value | A simulation-only gain erased by overhead is not practical success |
| Matched speculative extension | Competence against concurrent integration | A core gain alone does not answer this comparison |
| Full policy versus an explicitly simplified diagnostic policy | Which constraints create coupling | Never combine the simplified arm with fixed-quality claims |

For a crossed cohort/portfolio design, use four fully specified policies and report the interaction

\[
I=(Y_{11}-Y_{10})-(Y_{01}-Y_{00}).
\]

Negative I on a cost metric indicates additional benefit beyond the two main effects for that population. It does not establish a universal theorem. Preserve meaningful independent control: portfolio-only runs use an exogenously fixed candidate sequence; cohort-only runs use the common immediate portfolio rule. If a factor cannot be independently varied without changing policy, report that limitation rather than invent a factorial interpretation.

Negative controls remove particular mechanisms, not every possible optimization: a single fixed candidate removes cohort choice; no shared applicable receipts removes cross-release reuse; pinned unrelated churn removes moving-head invalidation. Unlimited workers do not necessarily remove batching resource savings.

### 8.2 Required adversarial cases and expected results

| Case | Required result |
|---|---|
| Same commit, different flag/config, dependency lock, schema, runtime or input generator | Different binding or UNKNOWN; never reuse by commit alone |
| Check/oracle implementation changes | Reject old receipt unless explicit policy accepts that revision |
| Missing dependency declaration | UNKNOWN or conservative full-subject check; omission fault must expose a shared-map error |
| Broad run covers A/B; only B changes | Reuse A only if the runner actually emitted a separately scoped A receipt |
| Authoritative FAIL plus selected PASS for the same claim | Gate veto, independent of planner selection |
| Cached receipt expires at 5; missing check finishes at 7 | Acquire fresh evidence or return bounded infeasibility |
| Expiry/revocation at exactly admission time | Reject under the half-open event convention |
| Candidate has a 10-tick job; unrelated arrival at tick 1 | Core subject stays pinned; arrival remains backlog |
| Receipt for A+B offered as evidence for B after A fails | Reject unless complete approved projection is genuinely unchanged |
| Planner starts at 0, takes 2 ticks; revocation at 1 | Events continue; stale plan cannot admit or launch retroactively |
| A ready at 5, rollout completes at 9 | No final deployed-base update or work credit before 9 |
| Crash after schema expansion but before worker replacement | Resume from actual intermediate state, preserving old/new compatibility |
| Lost transition acknowledgment | Idempotent reconcile; no duplicate effect or delivery credit |
| Wrong tenant/idempotency scope, commit-before-ack retry | Independent behavioral oracle catches duplicate/cross-tenant effects |
| Policy requirement changed by a code-producing participant | Unauthorized policy hash rejected; cannot improve score |
| Multiple complete changes jointly incompatible | Check actual candidate interaction; individual pass results do not imply combined pass |
| Revert masks a fault or removes an already counted feature | No false continued-delivery credit; explicit substitution/revert semantics required |
| Flaky result repeatedly rerun until PASS | Disallowed; fixed rerun/quarantine rule and all attempt costs retained |
| Running check cancelled just before completion | Charge consumed occupancy and cancellation tail; reservation released only after acknowledgment |
| Unknown external state changes without notifications | No confident reuse claim; block or require a new authorized observation |
| Search limit reached | SEARCH_LIMIT, never proved infeasible or optimal |
| A 5-unit combined plan versus a 6-unit reusable plan | Preserve both in frontier; no cost-only dominance |
| Arithmetic/geometric sign reversal in paired cost fixtures | Analysis reports total-resource regression despite a favorable geometric ratio |
| Repeated copies of one receipt or two model-provider names | No automatic independent-witness credit |

### 8.3 Workload robustness

Vary correlated runtime slowdowns, heavy tails with finite mean, positive setup floors, burst structure, failure isolation costs, cache cold/warm state, scope coarseness, policy changes and shared dependencies. Keep zero/low-headroom conditions in the results.

Withheld tests/fault families must differ structurally from tuning defects, not only in random seed. An independent evaluator should have extra transition schedules and input combinations beyond the optimizer-visible operational tests. A discrepancy is evidence about policy/oracle limits, not permission to quietly reclassify the episode.

For optional learned runtime or risk models, evaluate temporal drift and input corruption. Predictions influence efficiency only. Replace optimistic durations with measured upper-quantile models as a sensitivity; neither quantiles nor small-sample zero failures provide a hard guarantee.

### 8.4 Boundaries of safety evidence

Finite exhaustive checks establish exact properties only within their finite domains and trusted model. Fault injection establishes detection of those injected classes. Zero observed false admissions is not a proof of unchanged production defect probability.

For an honestly independent Bernoulli sample, the “roughly 3/n” zero-event upper-bound intuition may be reported only with its assumptions; correlated templates and rare catastrophic faults do not satisfy those assumptions automatically. Do not substitute such a calculation for real safety evidence.

## 9. Reproducibility, recovery and reporting

### 9.1 Experiment manifest

Each run records:

- Run ID, schema version, parent/resume ID, phase and declared hypothesis.
- Exact source revision, dirty-diff hash, protocol/config hashes and all method IDs.
- OS, architecture, interpreter path/version, SQLite version, dependency lock, package/wheel hashes, memory/storage budget and worker/controller capacities.
- Data source URLs/revisions/checksums, licenses, preprocessing code and split manifest.
- Episode, topology/template and fault-family IDs; all randomness keys, attempt counts and generator versions.
- Requirements, policy, catalogue, oracle, input-generator and deployment-plan digests.
- Cache condition, timing boundaries, cost units, service envelope, margins and stopping rules.
- Start/end wall timestamps with timezone, monotonic timing epoch, interruption markers, result status and failure reasons.

Use UTC in machine logs and retain local timezone metadata for human reports. Timestamps do not substitute for sequence numbers. A record is identified by its content and schema, not a friendly name alone.

A run directory includes manifest.json, events.jsonl, decisions.jsonl, resource_samples.jsonl, receipts/, admissions.jsonl, metrics.json and failures.jsonl. Large output may be chunked/compressed with per-chunk hashes. Logs are append-only; corrections are new records with predecessor IDs.

The reporting pipeline must regenerate tables from raw event/ledger data. It must not read a hand-edited summary as truth. Include unavailable/censored/failed runs in the run census and disclose missing data.

### 9.2 Authority and methodological amendments

The coding agent may choose routine non-destructive implementation details: data structures, parser organization, local paths, logging mechanics and equivalent low-level optimizations. It may repair implementation bugs while preserving failed runs.

A change to hypothesis, evaluation population, primary estimand, baseline capabilities, policy, service envelope, scope completeness, fault model or interpretation requires a methodological amendment before further confirmation. Record old/new hashes, reason, affected runs, whether protected outcomes were seen and the independent data reserved for reconfirmation.

Do not ask for human approval for every reversible coding decision. Escalate a genuinely unresolved requirement-owner decision or external access/production action. Lack of such authority does not block the local constructed benchmark.

### 9.3 Checkpoints and restart semantics

Checkpoint atomically at episode boundaries and durable controller transitions. For simulation include:

1. Event heap and next sequence number.
2. Pending/credited work IDs and candidate/transition states.
3. Receipt store, applicability assumptions/revocations and policy versions.
4. Running jobs, attempt counters, immutable input IDs and resource reservations/consumption.
5. RNG/keyed-stream state, planning snapshot/continuation and calibrated overhead model.
6. Source/config/schema hashes.

On resume, reject incompatible state. A different policy, source revision or dependency build starts a new run with a recorded relationship; it cannot silently resume confirmation.

For physical execution, reconcile durable job/transition completion before relaunch. An unfinished process is INTERRUPTED, not passed. Identify owned processes by launch nonce plus PID/start identity; do not kill an unrelated process because its PID matches an old checkpoint. Preserve actual intermediate database/service state and use idempotent step IDs.

An interrupted physical timing block is invalid for uninterrupted latency inference. Retain its cost/outcomes for the recovery study, then rerun the full paired block under the frozen rule with a new attempt ID. Do not splice monotonic intervals from different epochs or discard the interruption's work from recovery cost.

### 9.4 Recovery acceptance tests

Inject termination before job launch, after launch before acknowledgment, mid-check, after result persistence before receipt commit, after receipt commit before admission, after admission before first rollout step, mid-rollout, and after rollout completion before metrics.

For simulation, resumed canonical logical decisions and results must match uninterrupted execution, apart from explicitly modeled interruption events/cost. For physical execution, require no false readiness, no duplicate effects/credit, conserved accounting and correct reached state; byte-identical timings are not expected.

### 9.5 Reporting requirements

The final experimental report must contain:

- One claim table distinguishing source facts, deductions, estimates, proposed tests and actual measurements.
- Baseline versions/configurations and reproduction/adaptation limits.
- Full workload population, inclusion/exclusion census and split provenance.
- Arithmetic primary effects with uncertainty, absolute resource values, latency/deadline guards, failures and unfavorable regimes.
- Cost decomposition showing planning/setup, valid reused work, redundant work and cancellation/failure costs.
- A separate L1 simulation result and L2 physical result, each with its own population, sample-size/precision record and inference scope. Do not pool simulated and physical episodes as independent samples of one real-world effect.
- Attribution/robustness results and all deviations.
- Peak-resource measurements and actual M3 hardware configuration.
- Reproduction and recovery status, remaining limitations and the next evidence needed.

If any cost stratum is exactly zero, the arithmetic primary remains defined; report ratios only where their denominators are positive and never add a favorable pseudocount. Aggregate nested repeated timings using the frozen arithmetic mean of costs, preserving individual repetitions for variability plots.

### 9.6 Minimal report artifact set

The coding agent should produce a standalone results report, machine-readable metrics, frozen manifests, dependency/source provenance, raw logs and a one-command reproduction entry point after implementation. Keep unfavorable results. Store progress.json after each work package with last completed gate, blocking issue, next executable action and authoritative file hashes.

The current handoff is this plan only. It does not claim that those future artifacts, executable commands or benchmark outcomes already exist.

## 10. Outcome interpretation and evidence-triggered pivots

| Observed outcome | Supported conclusion | Next action |
|---|---|---|
| Gate/oracle disagreement or wrong-subject admission | Implementation/evidence model invalid | Fix before interpreting speed; preserve invalid runs and reconfirm independently |
| Simulator cannot distinguish a known better schedule or reports a placebo gain | Measurement harness invalid | Repair accounting/randomness/oracle path; rerun harness gates |
| Tiny oracle has no headroom on a case lacking coupling | Proxy may be uninformative | Add only a justified missing mechanism, then retest; do not generalize impossibility |
| Tiny oracle has no headroom on representative expressive cases | Strong controls may already solve the problem | Stop added optimizer complexity for that population |
| Oracle headroom exists; J fails to capture it | Search/forecast design limitation | Diagnose with regret and decision traces; one preregistered revision may be tested on fresh data |
| J wins only against B0/B1 | Competent caching/batching explains benefit | Report engineering baseline improvement, not joint optimization advance |
| B5 or B6 matches J | Extra horizon/complexity unneeded | Prefer simpler controller; narrow contribution accordingly |
| Arithmetic cost improves, latency worsens beyond accepted envelope | An uncompensated tradeoff | Report frontier; no unchanged-service claim |
| Small stable saving survives all guards but integration cost dominates | Scientific effect without demonstrated business value | Publish bounded result; target only environments with favorable break-even |
| CI includes zero and meaningful savings/regressions | Inconclusive at achieved precision | Report interval; expand only under a prospective sample plan |
| Confirmed negative effect with sufficient sensitivity | Evidence against value in that population | Stop or pivot to the diagnosed bottleneck |
| Synthetic gain disappears in actual execution | Model/overhead/transfer failure | Identify missing physical costs; no practical-win claim |
| Core gain disappears under matched speculation | Existing queue capability captures benefit | Restrict claim to non-speculative environments or use existing queue |
| Local physical and speculative gains survive; customer transfer unknown | Credible local optimization result | Pursue permissioned shadow study; do not claim enterprise safety or market validation |
| Customer bottleneck is flaky tests, reviewers or requirements | Wrong target for freshness optimization | Redirect to that measured bottleneck without retroactively claiming H1 succeeded |
| Exact equivalent prior art is found | Novelty claim invalidated | Replicate, compare or contribute an engineering improvement; do not relabel it |

A scale-dependent mechanism deserves escalation only if a measured structural trend and exact small-instance analysis support the necessary direction, and resource projections justify the next scale. “It might work at scale” cannot rescue repeated fair negative results indefinitely.

The preferred initial product outcome, if supported, is a recommendation record: exact candidate, unmet obligations, reusable receipts, proposed next checks/path, expected cost/latency range, and an auditable reason. Avoid an opaque assurance score.

## 11. Audit record, unresolved risks and limitations

### 11.1 Audits actually conducted

Three parallel agent reviews were conducted during preparation: scientific/prior-art, experimental validity, and implementation/feasibility. They are distinct internal reviews within this research session, not external peer review or proof-assistant validation.

After the first revisions, a second challenge pass examined the completed downstream design. Targeted final rechecks verified the resulting corrections and found no further substantive correction to the audited scientific, experimental or implementation contract. A paper-title copy edit was also corrected. These are textual/design reviews, not execution of the proposed benchmark or a guarantee that implementation will be defect-free.

The source mathematics and numerical scripts were reviewed and the reported checks in Section 3.4 rerun. No proposed controller, external CI implementation, production release or Mac benchmark was executed. Public sources were opened and inspected to the extent recorded in Section 12.

### 11.2 Material defects found and revisions made

| Defect identified | Revision in this plan | Residual condition |
|---|---|---|
| Moving-head theorem treated as direct optimizer justification | Pinned-candidate negative control; finite-worker model; no exponent-based safety score | Customer invalidation headroom remains unmeasured |
| Broad scheduling novelty contradicted by AAS and batching work | Narrow outer release/planning hypothesis; shared fixed-subject solver; strong B5/B6 | Exact novelty and proprietary internals remain unresolved |
| Geometric mean improvement could conceal total resource growth | Arithmetic cost per original work item is primary; matching inference/power rules | Population weights must be frozen |
| Missing addition signs in resource formula | Dimensionally valid charged-core-seconds sum | Proxy differs from billing/energy |
| Planner could get free time or pause the world | Controller lane, calibrated virtual overhead, intervening events, stale-plan recheck | Physical calibration still required |
| Active-candidate loop implicitly enabled forbidden speculation | Core candidate lock and explicit matched speculative phase | Broad claim requires that extra comparison |
| Readiness transaction pretended rollout was atomic | Separate READY/EXECUTING/DEPLOYED states and stepwise recovery | Real distributed deployment remains outside scope |
| Cached evidence removed before checking future expiry | Original receipt intervals retained as scheduling constraints | External assumptions require reliable observation/revocation |
| Cost-only frontier pruning lost reusable portfolios | Conservative continuation fingerprints; retain incomparable plans | Larger frontiers can be expensive |
| Equally named evidence alternatives could have unequal strength | Independent bounded semantic/fault validation before timing | Unmodeled faults remain possible |
| B_ref/envelope tuning could be circular | Explicit B1 → envelope → B2/B3 → B_ref → B5/J order | Confirmation must retain the frozen rule |
| Reuse count could include mutually exclusive releases | Count along one realizable continuation | Heuristic forecast may still be inaccurate |
| A shared incomplete scope could fool both gate and optimizer | Full-state independent evaluator and omitted-dependency faults | Fixture scope completeness is not general program analysis |
| Mathematical restatement lacked assumptions/notation | Positive hazard/coverage, initially empty pin, rank and product definitions | Historical mathematical novelty unresolved |
| Pinned-candidate repair could discard the chosen reusable portfolio | Persist the plan or repair under the method's original horizon objective | Implementation must pass the intervening-arrival test |
| Every computed schedule could already be in the past | Completion-relative dispatch times and a no-arrival liveness test | Runtime revalidation remains mandatory |
| Rollout costs were required but lacked an execution interface | Metered ROLLOUT/RECOVERY jobs and future-state timing | Measured physical step costs are still needed |
| “No cancellation” could leave hung fault tests running forever | Frozen job/RPC timeouts and owned-process cleanup | Timeout/cleanup cost and reached state must be retained |
| p95 could mean incompatible aggregations | One weighted pooled item-flow quantile and paired cluster resampling | Tail precision may require more independent episodes |
| Strict-budget test assumed a finite worst case for unbounded durations | Enforced caps/reservations, overrun failure, separately funded post-horizon drain | Operating-system overruns can disqualify strict-budget compliance |

### 11.3 Remaining blocking gates

Actual native Mac compatibility and resource calibration are WP1/WP5 gates. Sound evidence scopes and verifier alternatives are WP2/WP5 gates. Baseline adapters and exact oracle behavior are WP3/WP4 gates. Useful residual headroom is WP6. Sensitive independent confirmation is WP7/WP8. External licenses/access and realistic transfer are WP10.

Unavailable AAS/VP-CONTROL implementations were not proven nonexistent. If their published code becomes available, evaluate whether it changes a relevant baseline. Public CI metadata does not fill the counterfactual-outcome gap. Publicly downloadable archives do not resolve missing licenses.

The patent disclosure in [R38] is an additional reason to avoid broad invention claims. No exhaustive patent search, legal patentability assessment, infringement analysis or freedom-to-operate opinion was performed.

### 11.4 Limits of this handoff

This plan can support a careful local test; it cannot predetermine a positive result. A constructed executable workload can isolate real mechanisms while still failing to represent enterprise maintenance, human review, deployment infrastructure or incident severity.

The remaining scientific uncertainty is concentrated in a falsifiable question: after competent pinning, reuse, batching, fixed-subject assurance and speculation, is there enough beneficial release/verification coupling to justify the optimizer's complexity and operating cost?

## 12. References and search coverage

### 12.1 Source register

Sources were researched through **30 October 2026**. Live documentation was inspected on that date unless another date is given. A stable DOI or versioned paper identifies the cited publication; a repository or “latest” documentation URL does not freeze implementation behavior. WP0 must record the exact revision used in an experiment. No third-party scheduler, commercial product, public benchmark or Mac binary was executed during this research.

**[R1]** Jun He and Deying Yu. *Before Agents Act: Assurance-Aware Semantic Scheduling for Evidence Acquisition in Distributed Systems*. arXiv:2609.34376v1, 28 September 2026. [Versioned primary text](https://arxiv.org/html/2609.34376v1), [DOI](https://doi.org/10.48550/arXiv.2609.34376). Inspected full text, especially the supplied-action formulation, temporal solver, receipt repair, gate and generated experiments. Closest assurance-scheduling prior art. No runnable implementation was located; absence is not established. Initial unversioned retrieval failed, but versioned full text was subsequently available.

**[R2]** Emad Fallahzadeh, Amir Hossein Bavand and Peter C. Rigby. *Accelerating Continuous Integration with Parallel Batch Testing*. ESEC/FSE 2023. [DOI 10.1145/3611643.3616255](https://doi.org/10.1145/3611643.3616255), [author manuscript](https://arxiv.org/pdf/2308.13129). Inspected full paper, parallel batching policies and evaluation assumptions. Supports strong batch/capacity controls; retrospective build outcomes do not establish unseen mixed-version functional correctness. Related runnable assets are recorded in [R42].

**[R3]** Ali Sayedsalehi, Peter C. Rigby and Gregory Mierzwinski. *Risk-Aware Batch Testing for Performance Regression Detection*. arXiv:2604.00222v1, submitted 31 March 2026. [Primary text](https://arxiv.org/html/2604.00222v1). Inspected batching, risk/aging, build/backfill costs and resource assumptions. Uses performance-regression histories; its culprit-finding objective differs from release admission. Neither the original learning pipeline nor its published results were reproduced.

**[R4]** GitHub. *Managing a merge queue*. Current official documentation. [Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue). Inspected merge-group checks and combined target behavior. Establishes existing capability, not comparative performance on this workload.

**[R5]** Zuul project. *Project gating* and pipeline/job configuration documentation. [Gating](https://zuul-ci.org/docs/zuul/latest/gating.html), [pipelines](https://zuul-ci.org/docs/zuul/latest/config/pipeline.html), [jobs](https://zuul-ci.org/docs/zuul/latest/config/job.html). Inspected speculative dependencies, queue repair, cancellation and job-deduplication conditions. Informs B4. The proposed common-engine adapter is explicitly not a full Zuul deployment; pin the selected version before adaptation.

**[R6]** Trunk. *Merge Queue* and optimization documentation. [Product description](https://www.trunk.io/merge-queue), [official documentation repository](https://github.com/trunk-io/docs/tree/main/merge-queue/optimizations). Inspected parallel queues, batching and build-system integration claims. Strong commercial overlap; proprietary internal optimization and measured superiority remain unverified.

**[R7]** Bazel project. *Remote caching*. [Official documentation](https://bazel.build/remote/caching). Inspected action/content caching and warnings about incomplete tool/input tracking. Establishes reusable build machinery and the need for complete keys; a build cache is not itself a functional correctness certificate.

**[R8]** Andrey Mokhov, Neil Mitchell and Simon Peyton Jones. *Build systems à la carte: theory and practice*. Journal of Functional Programming 30, 2020; follows the 2018 work. [DOI 10.1017/S0956796820000088](https://doi.org/10.1017/S0956796820000088), [full manuscript](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/04/build-systems-jfp.pdf), [reference implementation tag](https://github.com/snowleopard/build/releases/tag/jfp-preprint). Inspected dependency/rebuilding abstractions and artifact availability. Supports sharing generic incremental computation across competitors.

**[R9]** Gregg Rothermel and Mary Jean Harrold. *A safe, efficient regression test selection technique*. ACM Transactions on Software Engineering and Methodology 6(2), 173–210, April 1997. [DOI 10.1145/248233.248262](https://doi.org/10.1145/248233.248262). Historical terminology and publication metadata checked; publisher full text was blocked during retrieval. Detailed local scope rules rely on the accessible sources and explicit bounded model, not an uninspected theorem from this paper.

**[R10]** Milos Gligoric, Lamyaa Eloussi and Darko Marinov. *Practical Regression Test Selection with Dynamic File Dependencies*. ISSTA 2015, 211–222. [DOI 10.1145/2771783.2771784](https://doi.org/10.1145/2771783.2771784), [author PDF](https://users.ece.utexas.edu/~gligoric/papers/GligoricETAL15Ekstazi.pdf), [Ekstazi repository](https://github.com/gliga/ekstazi). Inspected full paper and repository compatibility information. Supports selective rerun under dependency assumptions; the Java implementation is not a required dependency of the native Python fixture.

**[R11]** Mateusz Machalica, Alex Samylkin, Meredith Porth and Satish Chandra. *Predictive Test Selection*. ICSE-SEIP 2019. [Primary preprint record](https://arxiv.org/abs/1810.05286), [conference record](https://2019.icse-conferences.org/details/icse-2019-Software-Engineering-in-Practice/22/Predictive-Test-Selection). Inspected primary abstract and publication information. Supports the distinction between historical failure prediction and unchanged mandatory evidence. Its industrial results are context, not a reproduced comparator.

**[R12]** CloudBees. *How CloudBees Smart Tests select tests*. [Official documentation](https://docs.cloudbees.com/docs/cloudbees-smart-tests/latest/features/predictive-test-selection/how-we-select-tests). Inspected the semantic-relevance ranking, configurable subset target and additional historical signals. Confirms a deployed product category; does not establish semantic equivalence of omitted tests.

**[R13]** SLSA project. *Verification Summary Attestation*, specification v1.2. [Versioned specification](https://slsa.dev/spec/v1.2/verification_summary). Inspected subject, verifier and policy binding. Reusable envelope concepts concern the attested supply-chain properties; they do not imply general application correctness.

**[R14]** in-toto project. Project overview and linked specification material. [Primary project site](https://in-toto.io/). Inspected artifact/supply-chain provenance scope. Provides established provenance vocabulary; no full deployment or cryptographic threat-model validation was performed here.

**[R15]** Mohan Yang, Haixun Wang, Lipyeow Lim and Min Wang. *Optimizing Content Freshness of Relations Extracted From the Web Using Keyword Search*. SIGMOD 2010. [Author-hosted primary PDF](https://lipyeow.github.io/home/pub/sigmod10-contentfreshness.pdf). Inspected model and grouped-refresh mechanism. Adjacent freshness/coverage optimization predates this proposal; web-data synchronization is not release-policy satisfaction.

**[R16]** Abhishek Mukherji and Elke A. Rundensteiner. *Achieving High Freshness and Optimal Throughput in Resource-Limited Execution of Multi-Join Continuous Queries*. WPI-CS-TR-11-01, February 2011. [Institutional technical report](https://web.cs.wpi.edu/~mukherab/i/JAQPOT-TR.pdf). Inspected resource-limited freshness/throughput formulation. Establishes an adjacent coupled scheduling problem with different query semantics and objectives.

**[R17]** James B. Martin. *Stationary distributions of the multi-type ASEP*. Electronic Journal of Probability 25, article 43, 2020. [DOI 10.1214/20-EJP421](https://doi.org/10.1214/20-EJP421), [primary preprint](https://arxiv.org/abs/1810.10650). Inspected queueing construction relevant to the supplied q-geometric discussion. Supports a historical connection, not priority of the new supplied theorem or a practical software-assurance interpretation.

**[R18]** Keisuke Shiromoto. *On Critical Exponents of Matroids and Linear Codes*. RIMS Kôkyûroku 1889, 7–12, 2014. [Institutional primary PDF](https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1889-02.pdf). Inspected characteristic-polynomial/critical-exponent material. Informs the semantic-audit mathematics and distinguishes established algebra from unverified program-analysis applicability.

**[R19]** Vishrant Tripathi and Eytan Modiano. *Optimizing Age of Information with Correlated Sources*. MobiHoc 2022, 41–50. [DOI 10.1145/3492866.3549719](https://doi.org/10.1145/3492866.3549719), [NSF-hosted manuscript](https://par.nsf.gov/servlets/purl/10381114). Primary manuscript inspected in the mathematical audit; later retrieval attempts encountered timeouts/access errors. Correlated-source freshness is relevant background, with a different objective from simultaneous release readiness.

**[R20]** European Commission CORDIS / AssureMOSS consortium. *Assurance and certification in secure Multi-party Open Software and Services*, grant 952647, 2020–2023. [Official results register](https://cordis.europa.eu/project/id/952647/results). Inspected project results and incremental/continuous certification deliverable descriptions. Full deliverable text was not obtained; no detailed algorithm claim depends on those unavailable documents.

**[R21]** Sion Israel Sion and Alain April. *Who Verifies the Builder? Toward Independent Assurance for Agent-Produced Software*. arXiv:2610.28667v1, 16 October 2026. [Primary text](https://arxiv.org/html/2610.28667v1), [assure-harness artifact](https://github.com/besionisrael/assure-harness). Inspected full paper and artifact README. Three stateful services and 46 seeded faults support a bounded oracle-independence investigation; original builder programs were correct in that study. MIT-licensed service artifacts are optional reusable fixtures. The model-evaluation environment and documented runner limitations need separate checks if reused.

**[R22]** GitHub. *Octoverse: Commits and pushes reach new highs in 2026 as developers build with agents*. 28 October 2026. [Primary platform report](https://github.blog/news-insights/octoverse/octoverse-commits-pushes-reach-new-highs-in-2026-as-developers-build-with-agents/). Inspected the reported agent-contribution measure and dates. Motivation from platform telemetry; it does not identify causal delivery gains or the target customer's bottleneck.

**[R23]** CloudBees. *Enterprise Technology Leaders Report Production Failures from AI-Generated Code*. 19 May 2026. [Survey release](https://www.cloudbees.com/newsroom/enterprise-technology-leaders-report-production-failures-from-ai-generated-code). Inspected the 213-leader survey and cost responses. Vendor-sponsored self-report; neither representative population sampling nor a causal estimate is assumed.

**[R24]** Atlassian. Team '26 Europe AI SDLC announcement. 7 October 2026. [Primary company report](https://www.atlassian.com/blog/company-news/team26-europe-ai-sdlc). Inspected adoption/governance survey claims. Organizational motivation only; no inference that the proposed scheduler resolves the reported gap.

**[R25]** DORA. *2025 DORA report*. [Official research page](https://dora.dev/research/2025/dora-report/). Inspected the landing-page findings about AI and the surrounding engineering system, not the complete underlying study. Used as a counterweight to a universal downstream-harm narrative, not as a causal effect estimate.

**[R26]** CloudBees. *CloudBees Unify*. [Official product description](https://www.cloudbees.com/unify). Inspected cross-tool delivery governance/readiness claims. Establishes category overlap; product internals, pricing, integration effort and head-to-head effectiveness were not evaluated.

**[R27]** Harness. *Continuous Delivery*. [Official product description](https://www.harness.io/products/continuous-delivery). Inspected orchestration, verification and rollback capabilities. Established product overlap; no local product benchmark or general quality guarantee is inferred.

**[R28]** Anthropic and Theorem. *LLMs are making formal software verification feasible at a larger scale*. 22 October 2026. [Primary research report](https://www.anthropic.com/research/formal-verification). Inspected the stated verification setup and machine/specification restrictions. Demonstrates the importance of scoped formal claims; its benchmark success is not a guarantee for arbitrary services or deployment transitions.

**[R29]** Zihao Zheng, Baichuan Li, Junyi Yao and Jiayu Long. *Engineering Reliable Commit Gates for Agentic AI: Cost-Aware Verification Portfolios under Common-Mode Data Failures*. arXiv:2609.10969v1, 10 September 2026. [Primary text](https://arxiv.org/html/2609.10969v1). Inspected fixed-proposal portfolios, shared failure lineage, atomic guards and HTTP/SQLite race experiments. VP-CONTROL is the system name. Close admission prior art; no independent reproduction or located public implementation is claimed.

**[R30]** Divya M. Kamath, Bram Adams and Ahmed E. Hassan. *Lightweight Dynamic Build Batching Algorithms for Continuous Integration*. Empirical Software Engineering 30, article 51, published 8 January 2025. [DOI 10.1007/s10664-024-10555-4](https://doi.org/10.1007/s10664-024-10555-4), [author PDF](https://mcislab.github.io/publications/2025/emse_divya_light.pdf), [replication archive v1](https://zenodo.org/records/13760638), [repository](https://github.com/divyamadhav/dynamic_batching). Inspected full paper and archive listing. The 13 September 2024 archive is 39.9 MB; retrieved metadata did not settle all license conditions. Code/data execution remains a planned gate.

**[R31]** Python Software Foundation. Official Python documentation. [Using Python on a Mac](https://docs.python.org/3.12/using/mac.html), [multiprocessing](https://docs.python.org/3.12/library/multiprocessing.html), [time](https://docs.python.org/3.12/library/time.html), [resource](https://docs.python.org/3/library/resource.html). Inspected native Apple Silicon packaging, process behavior and measurement interfaces. Supports stack selection, while exact installed interpreter, spawn behavior and counter semantics require the proposed Mac smoke tests.

**[R32]** Google OR-Tools project. *Installing OR-Tools for Python on macOS*. [Official installation documentation](https://developers.google.com/optimization/install/python/pkg_mac), [package metadata](https://pypi.org/project/ortools/), [source/license](https://github.com/google/or-tools). Inspected arm64 packaging guidance and Apache-2.0 licensing. Optional solver only; an exact wheel/version and required operations were not executed on the target hardware.

**[R33]** SQLite project. *Write-Ahead Logging*. [Official documentation](https://www.sqlite.org/wal.html), updated 25 August 2026. Inspected journaling semantics and the WAL-reset issue. The page records fixes in 3.51.3 and backports 3.44.6/3.50.7. This motivates default rollback journaling and an exact-runtime check before optional WAL; it is not evidence that the planned single-writer setup is corrupt.

**[R34]** Dependency licensing records. Python Software Foundation, pytest maintainers and SQLite project. [Python license](https://docs.python.org/3/license.html), [pytest package metadata](https://pypi.org/project/pytest/), [SQLite copyright](https://www.sqlite.org/copyright.html). Inspected licenses for the proposed minimal stack. Pin and retain actual distributions/notices; a transitive dependency or optional asset needs its own inventory. The core does not depend on downloading model weights.

**[R35]** Moriconi, Durieux, Falleri, Troncy and Francillon. *GHALogs* dataset, v1.0.0, 4 December 2024. [Versioned Zenodo record / DOI 10.5281/zenodo.10154920](https://zenodo.org/records/10154920). Inspected collection description, file sizes and metadata. The record lists 1.1 GB structured data, 69.2 MB metadata and 142.3 GB raw logs. Use only qualified smaller files after license/access checks; these histories lack unobserved candidate outcomes.

**[R36]** Xianhao Jin and Francisco Servant. *CIBench*, v5, 8 January 2021. [Versioned Zenodo record / DOI 10.5281/zenodo.4682056](https://zenodo.org/records/4682056). Inspected artifact description and sizes: 888.9 MB processed data and 38.3 GB raw data. Licensing remained unresolved in retrieved metadata. Useful workload context, not counterfactual release correctness ground truth.

**[R37]** Google Cloud Platform. *Online Boutique / microservices-demo*. [Official repository](https://github.com/GoogleCloudPlatform/microservices-demo), [development guide](https://github.com/GoogleCloudPlatform/microservices-demo/blob/main/docs/development-guide.md), [releases](https://github.com/GoogleCloudPlatform/microservices-demo/releases). Inspected local resource requirements, Apache-2.0 license and arm64 release support, including v0.10.3 notes. Optional transfer workload; Docker/local-cluster disk and memory costs and exact image architectures must be checked before download.

**[R38]** Microsoft Technology Licensing, LLC. *Component-based scheduling and release readiness determination*, US20260252471A1; filed 26 February 2025, published 27 August 2026. [Published patent text reproduced by Justia](https://patents.justia.com/patent/20260252471). Inspected description and claims 1, 10 and 19, plus publication metadata. Technical prior-art lead only. Primary disclosure is reproduced by a third-party host; official legal status, claim construction and exhaustive patent coverage were not verified.

**[R39]** Kenneth S. Brown and Persi Diaconis. *Random Walks and Hyperplane Arrangements*. Annals of Probability 26(4), 1813–1854, 1998. [Author-hosted primary PDF](https://pi.math.cornell.edu/~kbrown/papers/hyperplane_walk.pdf). Inspected the arrangement-walk/stationary-law connection used when challenging the supplied finite-state mathematics. Historical resemblance is not a proof that the supplied exact statement is either new or previously published.

**[R40]** Michael N. Katehakis and Cyrus Derman. *Optimal Repair Allocation in a Series System*. Mathematics of Operations Research 9(4), 615–623, 1984. [DOI 10.1287/moor.9.4.615](https://doi.org/10.1287/moor.9.4.615). Publisher abstract/metadata inspected during the mathematical audit; full-text access was limited. Adjacent repair-allocation terminology only; no detailed theorem or implemented comparator is attributed to an uninspected full paper.

**[R41]** Umut A. Acar. *Self-Adjusting Computation*. Carnegie Mellon University dissertation, CMU-CS-05-129, 2005. [Institutional primary PDF](https://csd.cs.cmu.edu/sites/default/files/phd-thesis/CMU-CS-05-129.pdf). Inspected front matter, abstract and technical chapter structure, not the full dissertation. An older dependency/memoization/change-propagation trail; a recent crawl timestamp must not be mistaken for its publication date.

**[R42]** Emad Fallahzadeh, Peter C. Rigby and Bram Adams. *Contrasting Test Selection, Prioritization, and Batch Testing at Scale*. Empirical Software Engineering 30(1), article 33, 2025. [Author PDF](https://mcislab.github.io/publications/2025/emse_scale.pdf), [ContrastTestOptimizations repository](https://github.com/emadfallahzadeh/ContrastTestOptimizations). Inspected publication, README, license and BatchAll implementation details. The repository exposes MIT-licensed code with Python/PostgreSQL requirements. Port policy behavior into the shared finite-worker engine; document the timing-model and boundary adaptations instead of comparing incompatible simulator numbers.

### 12.2 Search coverage and stopping rationale

Two search engines were used, with primary-source follow-up for consequential technical claims. Backward references and later work around assurance scheduling and CI batching were traced. Representative vocabulary below describes the search coverage; it is not an exhaustive literal query log.

| Search family | Terminology and source trail | Decision affected |
|---|---|---|
| Direct evidence scheduling | assurance-aware scheduling, evidence acquisition, expiring witnesses, receipt repair, commit gates | AAS and VP-CONTROL invalidate broad scheduling novelty |
| Narrowed outer decision | release-cohort optimization, candidate formation, deployment transition verification, receding horizon / model-predictive CI scheduling | Selected application remains plausible; no new generic MPC algorithm is claimed |
| CI grouping and feedback | parallel batch testing, BatchAll, TestCaseBatching, dynamic build batching, risk-aware performance batching | Stronger B3 controls and finite-worker accounting |
| Speculative integration | merge queue, speculative gating, dependency queues, bors-style batching, cancellation and deduplication | Matched speculative phase is required for a broad competitive claim |
| Incrementality and provenance | build systems, action/content caches, regression test selection, self-adjusting computation, SLSA, in-toto | Correct reuse and evidence envelopes are established tools |
| Continuous assurance | incremental certification, continuous certification, AssureMOSS, independent assurance of agent-produced software | Separate policy/oracle authority; limited claim about seeded faults |
| Mathematical neighbors | correlated-source age of information, content freshness, resource-limited query freshness, repair allocation, arrangement walks, matroid critical exponents, ASEP queues | Supplied results remain conditional; historical novelty is unresolved |
| Product and adoption | official merge-queue, test-selection, release governance and deployment-verification descriptions; platform telemetry and surveys | Integrate with existing systems; first measure actual customer headroom |
| Patents | component scheduling, release readiness and related publication terminology | Found a directly relevant disclosure; avoid broad invention claims |
| Local feasibility/assets | official Python/macOS, OR-Tools arm64, SQLite behavior, dataset archives, sample-service development guides | Native standard-library-first core; exact compatibility, licensing and size gates |

Later targeted searches reinforced these method families and added implementation constraints rather than another clearly different indispensable comparator. That is the practical stopping rationale for this handoff, not a claim that search coverage is complete. A source found at WP0 that changes the exact outer-controller comparison must amend the protocol before confirmation.

Important blind spots remain: proprietary scheduler internals; unlocated or subsequently released AAS/VP-CONTROL code; inaccessible certification deliverables and selected older full papers; incomplete dataset license metadata; an exhaustive patent/legal search; customer-specific release histories; and actual M3 Max measurements. Each affects a stated claim or blocking gate rather than being silently assumed favorable.

### 12.3 Reuse and contribution ledger

| Reuse rather than invent | Project-specific implementation | Claim it could support |
|---|---|---|
| Content identities, conservative incremental checks, artifact-bound receipts | Complete bounded subject/projection catalogue and independent checker | Bounded applicability correctness and reproducible fault cases |
| Published batching and queue policies | Disclosed common-engine adapters with differential policy checks | Fair local comparison, not commercial product superiority |
| Fixed-subject assurance portfolios and temporal constraints | Shared bounded solver plus complete-frontier status | Controlled inner machinery for all methods |
| Receding-horizon planning | Release-cohort/approved-transition search carrying future receipt reuse | An incremental optimization effect if J beats B5, B6 and B* after overhead |
| Standard statistical and performance methods | Frozen episode population, arithmetic resource endpoint and full-drain protocol | Reproducible efficiency evidence within that population |

The strongest defensible initial publication or engineering artifact may therefore be a negative result, a small simple optimizer, a well-controlled empirical boundary map, or a benchmark demonstrating a real cross-release coupling effect. The experiments determine which description is justified.



