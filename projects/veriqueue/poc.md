# Research and proof-of-concept plan: verification-aware software delivery

**Research cutoff:** 30 October 2026, Europe/Amsterdam.  
**Target execution environment:** one MacBook Pro with an M3 Max; actual memory, storage and macOS must be discovered.  
**Status:** researched execution plan, with a targeted reproduction of the supplied mathematics. The proposed delivery controller, comparative experiments, customer measurements and M3 execution have **not** been performed.  
**Primary direction:** test the incremental value of version-specific verification state and scarce-reviewer replacement costs in upstream release scheduling, using established rollout and queueing techniques.

## Reading and execution guide

This document is the complete handoff. Sections 1–4 establish the evidence, scope and claim; Sections 5–8 specify the mechanism, comparisons, evaluation and hardware path; Section 9 gives ordered work packages; Sections 10–12 specify robustness, reproducibility and outcome decisions. Sections 13–14 record audits and sources. The mathematical reference in Appendix A can be implemented without the original attachments.

All commands beginning `python -m vaware` are **proposed project commands that the coding agent must implement**. Existing operating-system commands, supplied archive commands and documented upstream SubmitQueue commands are identified separately. No proposed command is represented as already available.

Evidence labels used throughout:

- **Inspected:** supported by the attached materials or a primary source actually examined during this investigation.
- **Reproduced:** a calculation or check actually executed in the research environment, described in Section 2.
- **Deduction:** an argument from stated assumptions, rather than an observed customer outcome.
- **Proposed:** an experiment, implementation or acceptance rule for subsequent execution.
- **Unresolved:** a prerequisite or uncertainty that remains open.

## 1. Executive assessment and selected direction

### 1.1 Verdict

**Pursue a bounded comparative study and an optional scheduling extension; do not begin by building a new review platform or claiming a new general scheduling principle.** The supplied mathematics is a credible, reproducible benchmark for completion when previous work can become stale. The proposed business mechanism is plausible, but its prevalence and economic importance in the intended customers remain unmeasured.

The general idea is substantially anticipated. Modern merge systems already track dependent changes, reuse applicable results, preserve running work, adapt speculation and batching, and avoid some verification waste. Uber SubmitQueue is a closer comparator than the original proposal recognizes. Cost-sensitive protection of expensive work also appears in adaptive transaction control, while task rework and design freezes have a long operations-research history. Generic rollout for stochastic rework scheduling is established. [R08–R16, R22–R25]

The strongest surviving question is:

> For releases containing expensive, heterogeneous verification obligations, does explicitly representing which completed evidence still applies, its **future replacement effort**, and specialist availability improve useful delivery beyond strong scoped isolation, selective reuse, tuned batching and speculative queues?

This is a potential **new empirical finding and engineering optimization**, with a useful benchmark as another possible contribution. Methodological novelty is not established. An equivalent implementation already using the same state, actions and optimization criterion would further narrow the contribution to replication or integration. A different name for generic rollout would add no novelty.

### 1.2 What the local PoC should establish

The core deliverable should demonstrate, or fail to demonstrate, a conditional scheduling advantage in controlled workloads whose dependencies, artifacts, obligations and costs are explicit. It should also exercise real local Git changes and tests to verify evidence binding. Simulated reviewer minutes must remain labeled simulated. A laptop experiment cannot establish market prevalence, actual human attention saved, long-term defect safety, or organization-wide productivity.

Use three evidence levels:

| Level | What is tested | Permissible conclusion |
|---|---|---|
| Analytical and exact small cases | The supplied chain, timing conventions, immutable-candidate lower bounds, tiny scheduling states | Correctness and existence or absence of a mechanism under specified assumptions |
| Controlled scheduling plus a local repository | Common obligations and resources; real version binding; simulated human service; actual controller overhead | Conditional algorithmic and implementation results, not measured human productivity |
| Prospective customer telemetry and controlled deployment | Active human effort, actual applicability, complete release outcomes, interference | Practical prevalence and causal delivery benefit within the studied teams |

The first two levels constitute the local PoC. The third is an explicitly separate continuation, not a dependency that silently makes the laptop program impossible.

### 1.3 Selected implementation

Implement a small event-driven evaluation environment and a common evidence ledger. Establish strong simple policies first. Then add **ordinary bounded rollout with evidence-aware state**, comparing immediate admission, short scoped stabilization and legally independent release candidates. The controller can also choose verification order under specialist calendars. It must leave evidence requirements and the release gate unchanged.

Use the current public SubmitQueue implementation as a faithful local comparator where its pinned scheduler can be driven correctly. Its documented Go code, local fake provider and bare-Git provider make this plausible; execution and an exact source revision remain prerequisites. The public code is not the private Java system evaluated in the 2025 paper. If a faithful adapter is unavailable, retain a clearly labeled documented-behavior adaptation and narrow the competitive claim. [R08–R10]

No LLM, model training, GPU, paid API or remote execution is necessary for the core mechanism. Unsupported language-model assertions about independence must never authorize evidence reuse.

### 1.4 Early reasons to stop or narrow

Stop the **product expansion** if measurement shows little avoidable invalidation after normal release isolation, or if overhead and delay consume the available savings. Stop the **new-controller claim** if tuned scoped batching or current queue behavior matches the proposed controller within sufficiently narrow uncertainty. Retain a useful negative result or configuration recommendation.

Do not stop merely because a small example cannot express shared dependencies, heterogeneous costs or multiple releases. Conversely, a hypothetical benefit at larger scale is not a reason to ignore negative results in an expressive, adequately sensitive workload.

The illustrative 20% target in the supplied idea is **not** adopted as a scientific success threshold. Estimate effects with uncertainty. Derive any practical threshold from measured customer costs and acceptable delivery delay; if those are unavailable, report a sensitivity frontier and leave commercial materiality unresolved.

## 2. Audit of the attached research and reproduction package

### 2.1 Inventory and provenance

The investigation inspected the pasted assignment, both copies of the research PDF, the complete LaTeX source, both Python programs, saved JSON results, README and figure. The external PDF and the PDF inside the ZIP each contain 11 pages. Their byte hashes differ, but their extracted page text is identical. Representative mathematical and figure pages were rendered and visually inspected using PyMuPDF after the available Poppler binary failed to load a shared library.

| Input | SHA-256 |
|---|---|
| `Pasted text(20261030-221902).txt` | `af34df226caa984cf3ebcc0c6128113b7fcae7698513f15ac4e1a2faeef0d7d6` |
| `perishable_review_reproducibility(1).zip` | `da2e69c7c5862d1f482d00c6e4f81e0955d2d74f646495f4bfd037acf1240973` |
| External `perishable_review_research(1).pdf` | `c6c60603012679149bc7367e633f8f80004dd42d999d15a219d4d9c3188d41d5` |
| PDF inside archive | `867b625bc70117e77141661360c77b00e80c2a42e919a7439f8aae16a41d0243` |

The note's motivating Hacker News discussion was not separately attached or independently audited here. It is anecdotal motivation, not a source of model parameters or customer prevalence. No license was supplied for the user's archive; retain attribution and do not assume permission to redistribute it publicly. The core specification below does not require redistributing the archive.

### 2.2 What the theorem actually says

There are $n$ homogeneous required reviews, initially stale. A single reviewer completes one review per round. Each previously current review independently survives a round with probability $q\in(0,1]$. The chosen stale item becomes current **at the end** of the round. Expiration is perfectly observed and costs nothing to detect.

Always reviewing a stale item is optimal in stochastic order in this model. Before hitting $n$, the number of current items follows

$$
K_{t+1}=1+\operatorname{Bin}(K_t,q),\qquad
T_n=\inf\{t:K_t=n\}.
$$

Define

$$
F_q(k)=\sum_{r=1}^{k}\binom{k}{r}
                 \prod_{j=1}^{r-1}(q^{-j}-1),\quad F_q(0)=0.
$$

The exact remaining mean is $\mathbb E_k T_n=F_q(n)-F_q(k)$. Its positive-term recurrence gives an $O(n)$-arithmetic evaluation; this is not a linear bit-complexity bound for exact rational arithmetic. For $0<s\le1$, the generating function is

$$
\mathbb E_k[s^{T_n}]=\frac{H_{1/s}(k)}{H_{1/s}(n)},\quad
H_z(k)=\sum_{r=0}^{k}\binom{k}{r}\prod_{j=0}^{r-1}(zq^{-j}-1).
$$

For the **specific family** $q_n=e^{-a/n}$, with fixed $a>0$, the note establishes:

| Regime | Mean completion behavior |
|---|---|
| $0<a<1$ | $\mathbb E T_n/n\to-\log(1-a)/a$ |
| $a=1$ | $\mathbb E T_n=n[\tfrac12\log n+\log2+\gamma/2+o(1)]$ |
| $a>1$ | $n^{-1}\log\mathbb E T_n\to I(a)>0$ |

Here

$$
I(a)=\max_{0\le x\le1}\int_0^x
\log\left[\frac{(1-y)(e^{ay}-1)}{y}\right]dy.
$$

The positive maximizer solves $-\log(1-x)=ax$. At criticality,

$$
T_n/n-\tfrac12\log n\Rightarrow\log(\sqrt2/|Z|),\quad
Z\sim N(0,1),\quad F(x)=\operatorname{erfc}(e^{-x}).
$$

For $a_n=1+\beta/\sqrt n$, bounded $\beta$, the mean's constant becomes

$$
\Psi(\beta)=\log2+\gamma/2+
\int_0^\infty e^{-t^2/4}\frac{e^{\beta t}-1}{t}\,dt.
$$

The model's Poisson interpretation gives $q=e^{-\lambda\tau}$ and $a=n\lambda\tau$. This is not a universal operational threshold. For fixed finite $n$, the mean is finite and smooth for every $q>0$; for fixed $q<1$, the note instead obtains growth with a quadratic exponent in $n$. The finite-count zero-drift condition $n(1-q)=1$ also differs from $-n\log q=1$.

### 2.3 Proof and numerical findings

**Targeted proof audit:** no substantive defect was found in the homogeneous mean, generating function, three regimes, critical window, critical distribution or timing-matched random comparison. The binomial-moment cancellation and finite-mean stopping argument are valid under the stated assumptions. This is not formal verification or an exhaustive specialist review.

**Actually reproduced:** the supplied `run_exact_checks()` was imported and executed without invoking its file-overwriting main routine. The transition systems are assembled separately from the closed formulas, but within the same supplied program; this is reproduction of its checks, not an independent implementation of every theorem.

| Exact checks passed | Systems or identities | Starting-state values |
|---|---:|---:|
| Controlled means | 72 | 324 |
| Controlled generating functions | 216 | 972 |
| Careless-collector means | 36 | 234 |
| Stationary distributions | 36 | — |
| Initial boundary identities | 90 | — |
| Timing-matched random-refresh means | 24 | 108 |
| Matched binomial-moment identities | 720 | — |
| Additional boundary identities | 13 | — |

The functional check took approximately 0.241 seconds on **Linux x86_64, Python 3.12.14**. This single timing is not an M3 measurement or a performance benchmark. Scaling spot checks reproduced the archived centered critical means at $n=10^3,10^4,10^5,10^6$: 0.9788285354, 0.9807893253, 0.9814455342 and 0.9816567346, approaching $\log2+\gamma/2\approx0.9817550130$. These are evaluations of formulas, not observed review behavior.

**Separately implemented during this investigation:** Appendix A enumerates survival bit patterns and builds exact transition systems without importing the archive. Its 24 mean systems (84 starting-state values), 48 PGF systems (168 values) and 24 floating log-mean comparisons all passed. Maximum observed absolute log error was approximately $3.55\times10^{-15}$. The recorded functional run took about 0.123 seconds excluding imports on the same Linux/Python environment; this also is not a benchmark. This independent finite-state check covers the controlled homogeneous chain, not every extension or asymptotic proof.

### 2.4 Attribution and application limits

All-working completion and repair priorities are old problems: Smith (1978), Katehakis–Derman (1984), and especially Frostig's fixed-repair-time model (1999) are material antecedents. Their accessible abstracts were inspected; their full proofs were not all accessible. The exact discrete formulas' historical priority remains unresolved. Lachaud (2005) already gives the transformed-normal distribution in an Ornstein–Uhlenbeck hitting problem. Long (2026) already establishes the fixed-loss careless-collector result explicitly acknowledged in the attachment. [R01–R07]

Do not import a heterogeneous optimal priority from an abstract. The two-item fixture in Section 10 shows why timing and ordering conventions matter. The original count $K_t$ generally ceases to be Markov with heterogeneous costs, dependencies, correlated losses, specialty resources or controlled admissions.

The most consequential software-model correction is that a real review or test observes a **start snapshot**. Completion does not make its evidence applicable to changes made during its execution. The theorem-matching mode must be isolated from the software mode. Under independent one-item Poisson changes, a chosen job's probability of still applying can equal $q$; the note's optional Bernoulli-success extension can represent that special case. Shared changes require the explicit dependency model.

Use the mathematics as an oracle and stress-test source. Do not use it as evidence that a customer is overloaded, that all company reviews must be current together, or that the optimal number of agents is known.

## 3. Research findings, strongest prior art and novelty assessment

### 3.1 Application evidence and counterevidence

The supplied October 2026 sources resolve and support their quoted headline figures. Their interpretation needs restraint.

| Inspected evidence | Supported inference | Unsupported inference |
|---|---|---|
| GitHub Octoverse, 28 October: over 40% of September PRs began with detected agent drafts, versus 17% in March; agent-involved two-week merge rate also increased [R26] | Agent-driven workflows are substantial and changing | Agent work necessarily creates a review crisis, or staleness causes the delays |
| Anthropic, 13 October, updated 15 October: daily merges grew from about 2,500 to over 30,000; workload explicitly unusual [R27] | Infrastructure can become a constraint at an extreme scale | This is representative of a 50–500-engineer company |
| DX, 14 October: larger PRs and longer first-review waits in high-AI-use organizations [R28] | Observational evidence of downstream capacity pressure | Causal attribution to AI, or a measurement of invalidation-driven effort |
| He et al., version 2 revised 27 October: 802 developers and 196,212 PRs; flat review rounds alongside review latency; changing human/automated review composition [R29] | Queue delay and review composition must be separated from repeated review | The proprietary data provide the needed public benchmark, or coarse revert stability proves unchanged safety |
| GitHub Copilot approval release, 28 October [R30] | Automated approvals are part of the relevant verification landscape | Another AI reviewer is a new category, or reviewer quality can vary between scheduler arms |
| METR's 2026 follow-up identifies selection and concurrent-agent time-measurement problems [R31] | Active effort, abandonment and assignment selection require explicit measurement | One historical productivity estimate transfers unchanged to current workflows |

These studies use different units, organizations, tools, definitions and causal designs. Rising merge volume can coexist with local review queues. Stable review rounds can coexist with scarce-reviewer delay. Neither resolves how much work becomes stale before release.

Anthropic's reported move toward post-merge/pre-deploy testing is also a relevant architectural alternative: stabilize the **deployment candidate** while allowing integration to continue. The PoC should include that boundary placement where its modeled release requirements permit it; it need not reproduce Anthropic's source-control system. [R27]

### 3.2 Prior-art comparison

All methods below assume some mechanism for identifying dependencies or evaluating correctness; none makes arbitrary unknown dependencies safe. Source-based capability descriptions are not controlled performance comparisons.

| Alternative | Problem and mechanism | Assumptions and evidence | Local availability / cost | Exact overlap and implication |
|---|---|---|---|---|
| Immutable scoped candidates plus selective evidence reuse | Finish a stable release and defer overlapping work | Pinnable artifacts/environment; correct scopes; no special research claim | Simple CPU implementation; mandatory baseline | May remove all relevant rework; test it first |
| Current SubmitQueue OSS [R08–R10] | Speculative paths, outcome scoring, preserved running/finished work, bounded allocation | Dependencies and build verdicts; public source inspected, not executed here | Go/Bazel unit path; Docker-backed local Git stack; pin and smoke first | Protecting work and cost-weighted allocation are already contemplated; specialist obligation state is the narrower question |
| SubmitQueue 2025 paper [R11] | Predict completion order and probability a speculative build will be needed; limit speculation | Private monorepos and before/after deployment evidence | Private Java service/data; published numbers not locally comparable | Direct prior art for reducing verification waste while accelerating delivery |
| Mergify [R12] | Scoped dependency queues, per-scope capacities, dynamic batches, safe direct reuse | Scope quality; SaaS documentation, not independent benchmark | Local rule adaptation possible; actual hosted service not needed | Scope freezes, adaptive batching and unaffected-check reuse are not novel |
| Zuul [R13] | Dependent multi-project queues and adaptive speculation window | Integration dependencies; documented AIMD-style window | Public software; full stack optional; declared local adaptation cheap | Limiting admitted speculative work to avoid discarded tests is established |
| GitHub/GitLab [R14] | Exact proposed merge-state checks, stale-approval handling, merge groups/trains | Repository rules and CI; hosted/tier-specific execution | Documented behavior can be modeled; no paid core dependency | Operational floor; insufficient as the only strong comparator |
| Build traces, incremental computation and Pact [R15–R17] | Rebuild only affected tasks; reuse results; version-specific contract compatibility | Complete relevant inputs and faithful fingerprints | Reusable concepts and open implementations | Evidence bookkeeping is established; share it across arms |
| FreshCtx and ATR [R18–R19] | Dependency freshness, selective predicate revalidation, validation-to-effect binding | Explicit dependencies, sound validators, faithful tokens, target coordination | Python artifacts; isolate versions if reproduced | Applicable evidence and safe commit boundaries are prior art, not the scheduling contribution |
| ATCC and related agent transaction systems [R20–R21] | Balance blocking against abort cost; protect expensive transactions | ATCC has a large-server evaluation; CoAgent/Cordon were inspected at design level only | Full ATCC setup outside core; simple reservation control is an adaptation | Expensive-progress protection and adaptive scoped locking are known |
| Design rework and freeze scheduling [R22–R23] | Changes reduce task maturity; schedule or freeze connected tasks | Product-design models and case studies | Local analogous heuristics; not software-review reproduction | The general problem and grouped stabilization are established |
| Stochastic project rollout and random-rework disclosure [R24–R25] | Evaluate candidate decisions through sampled continuation schedules | Modeled transitions/objective; published and disclosed methods | Generic rollout is straightforward locally | The chosen search technique itself is not a new method |
| CI batch-testing research [R32–R33] | Combine changes and prioritize tests using risk, size or age | Test outcomes and batching assumptions; simulation/production-specific evidence | Published algorithms/artifacts; scope-dependent adaptations | Strong alternative explanation: ordinary batching or priority selection |

Current SubmitQueue's documented allocator charges pending, running and cancelling jobs until they terminate and skips terminal results. Its design explicitly allows weighted budgets based on build size or historical cost; the enabled cost policy must be checked in the pinned implementation. Its extension chooses scheduling actions while a separate controller retains merge authority. The plan must preserve implemented capabilities; a baseline that cancels everything or forgets completed results would be artificially weak. [R09–R10]

Mergify's isolated mode deliberately does not test interactions between independent-looking batches. It is not a safety-equivalent comparator for coupled work. GitHub's merge limits and build concurrency must also not be misimplemented as one test-batch-size parameter. [R12, R14]

### 3.3 What can be reused

Reuse upstream dependency manifests, content hashes, check records, contract compatibility and queue schedulers where their semantics match. Keep a small common applicability API instead of rebuilding a general build system. Use a separate policy module around a queue rather than granting it authority to weaken the gate. The first repository fixture needs only standard-library Python tests and Git; it does not need a distributed service platform.

FreshCtx can supply a conformance comparison for dependency tracking; ATR supplies a precise warning about assumptions and the validation/effect gap. Neither is a delivery scheduler, so comparing its raw throughput to the proposed scheduler would be a category error. [R18–R19]

### 3.4 Precise contribution and novelty falsifiers

| Candidate contribution | Current classification | Discriminating test | What would invalidate or narrow it |
|---|---|---|---|
| Supplied finite formulas and critical window | Mathematical candidates; priority unresolved | Exact independent transition systems; comparison with older fixed-repair results | A prior equivalent formula/theorem; a proof error |
| Versioned specialist-evidence scheduling state | Potential application/representation contribution | Same dependencies, gate, capacity and forecasts; add/remove the relevant state | Existing tool already exposes and uses equivalent state for the same decisions |
| Evidence-informed rollout controller | Application of known optimization | Compare strong policies, direct loss-cost greedy and equivalent-compute ablations | Generic full-state rollout is identical; classify as one method, not two inventions |
| Useful delivery gains in affected workloads | Unproven empirical hypothesis | Protected paired comparison plus cross-release accounting | Gain vanishes with tuned isolation, fair compute, partial reuse or realistic dependency uncertainty |
| Customer value from reduced human rework | Unproven operational hypothesis | Prospective measured attention and controlled deployment | Little attributable avoidable effort; overhead/delay exceeds benefit |
| Reproducible evidence-rework benchmark | Possible research artifact | Independent reconstruction and policy conformance | Workload is circularly favorable or lacks the claimed mechanism |

The refinement search covered specialist-cost admission, cost-aware speculation, hybrid locking, rework scheduling, design freeze, incremental computation and generic rollout. It found direct conceptual predecessors, including a translated random-rework scheduling patent disclosure. That disclosure is evidence about described mechanisms only; this document gives no patentability, infringement or freedom-to-operate opinion. [R25]

### 3.5 Why a difference might still matter

A CPU-minute and a security specialist's next available 30-minute session are different constraints. Two equally sized changes can invalidate different obligations; a bundle of checks may share setup costs; a review can retain partial reasoning despite losing formal applicability. Existing build-path priority may not capture those facts. If they change the best action often enough, exposing them to a standard optimizer can matter.

The technically strongest case against the project is equally clear: correct immutable release boundaries may eliminate most invalidation; human re-review may be cheap; existing queues may already capture enough state; estimating extra dependencies and costs may cost more than it saves. These are experiments to run early, not objections to dismiss.

## 4. Problem specification, assumptions and hypotheses

### 4.1 Scope and beneficiaries

The proposed initial customer is an established software organization with interacting components, substantial concurrent human/agent changes and scarce consequential approvals. The suggested 50–500-engineer range is a customer hypothesis, not evidence. The unit of analysis is a minimal legal release and its dependency closure, not all reviews in a company.

Keep authoring outside the first intervention: changes arrive from an immutable workload tape. Agents are represented by the changes they propose, not by a new model generating code. This preserves the scheduling question while avoiding a confounded contest between authoring models. Later behavioral adaptation is explicitly outside the core claim.

### 4.2 Entities and time

Use integer microseconds internally and seconds/minutes in reports. Wall-clock telemetry uses UTC timestamps and records collection time separately. Simulated time, controller CPU time, runner service time and active human attention are distinct quantities.

| Entity | Required meaning |
|---|---|
| Capability $j$ | A fixed user-visible outcome with externally set weight $w_j>0$, arrival $a_j$, optional deadline, change prerequisites and an independent satisfaction predicate |
| Change $\Delta$ | Immutable patch or version transformation, declared preconditions, affected scopes, required predecessor changes, resulting capability effects |
| Candidate $X$ | Immutable component-version vector plus policy and relevant environment snapshot; admission creates a successor, never mutates existing bytes |
| Obligation $r$ | Versioned requirement for a particular review/test/property and authorized verifier role |
| Evidence $e$ | A result for a specific obligation and observed input projection, with outcome, provenance, validity interval and policy/tool identity |
| Verification job $g$ | Work that can produce one or several evidence items, with resource requirements, setup/shared cost, duration and cancellation/resumption rules |
| Resource | Named reviewer or role capacity/calendar, CI worker pool, or exclusive test environment |
| Release | An atomic update of the modeled deployed component vector after the common gate accepts exactly the relevant artifacts and environment tokens |

Reject change sequences whose preconditions cannot be composed; do not let a last-write-wins component update erase a capability while crediting all change IDs as delivered. Use explicit supersession/equivalence mappings where intended. Core workloads use capability-preserving changes; adversarial workloads test removal and regression. Count capability value once and verify its presence in the deployed state at the measurement horizon.

### 4.3 Evidence applicability

Define the required obligation set from a versioned, scheduler-independent rule:

$$
\mathcal R(X,P)=\operatorname{requirements}(X,P).
$$

For evidence $e$ concerning obligation $r$, store its original candidate digest for audit, but key reuse by

$$
f_r=H(\text{obligation definition},\ \text{relevant component projection},
\ \text{relevant environment tokens},\ \text{verifier/tool version},
\ \text{applicable policy},\ \text{declared dependency schema}).
$$

Do **not** put the whole candidate ID or the whole current requirement set into every reuse key. That would manufacture invalidation when unrelated work arrives. Conversely, an incomplete dependency projection cannot justify reuse. A result completed for an older candidate may support a newer one if this authorized projection is equal and the result has not expired or been revoked.

The applicability API returns `APPLICABLE`, `MISSING`, `INVALIDATED`, or `UNKNOWN`, with machine-readable reasons. `PASS/FAIL/ERROR/CANCELLED` is a separate outcome field. A passing but stale result is not passing evidence for the current release. A current test result is not proof that the software is correct.

Known uncertainty expands the dependency scope conservatively or requires a resolving review. An **undetected missing dependency** is different: only the independent workload auditor can expose it. Do not secretly supply hidden true dependencies to the runtime gate or repair one policy's decision using the oracle.

Applicability is a relation between evidence, a candidate and the current relevant policy/environment. It is not a mutable global flag on an evidence object. A successor may be incompatible with a certificate that still supports a retained predecessor. Expiry and revocation are separate events. Neither the ledger nor the destruction score may treat a certificate as globally destroyed merely because one successor cannot use it.

### 4.4 Controllable versus external invalidation

Tag every version change as one of:

1. Candidate adoption controlled by the scheduler.
2. A change to an independently versioned deployment that the scheduler can coordinate.
3. An exogenous environment/policy/revocation change it cannot prevent.

Holding a candidate stable blocks only relevant adoption or authorized coordinated deployment. It does not stop time, external security revocation, all developers or unrelated branches. A local branch change alone cannot invalidate evidence for immutable candidate bytes. Environment effects must be explicit.

### 4.5 Objectives and hard constraints

The operational objective is useful capability delivery within fixed scarce-attention and compute budgets, with unchanged requirements and bounded deferral. The core primary metric is $Q(H)$, defined in Section 7. It measures delivery, not rework by itself.

Hard constraints include required evidence, legal component combinations, resource calendars, maximum admission hold, urgent-change rules, and a common gate. Every deferred item stays visible from $a_j$. Never remove requirements, abandon difficult work, increase staff, use extra private forecasts or count PR splitting as value to improve a result.

For the initial safety-equivalent comparison, the release gate is the same implementation for every arm. Unknown critical applicability blocks release or triggers broader verification. Fairness limits constrain **scheduler-imposed deferral**; if workload demand exceeds physical capacity, report deadline/age failures instead of pretending a policy can guarantee feasible delivery.

### 4.6 Hypotheses and alternatives

| ID | Testable hypothesis | Expected observation | Principal alternative / weakening observation |
|---|---|---|---|
| H0 | Harness and evidence semantics are correct | Exact fixtures and equivalent policies agree; all required races are rejected | Apparent gains disappear after fixing event order or reuse keys |
| H1 | Explicit evidence/cost state improves delivery in the declared affected-workload population | Positive protected $\Delta Q(H)$ versus a strong frozen comparator selector | Cheap-job prioritization, extra forecasts or weak batching explain gains |
| H2 | Prevented invalidation contributes to H1 | Matched state ablations alter admissions; fixed-cohort drain effort/repetition changes as predicted | Same total effort is relabeled across releases; ordinary scheduling alone explains H1 |
| H3 | The mechanism survives credible errors and costs | Benefits remain within declared partial-reuse, cost-error and unknown-scope conditions after overhead | Conservative scopes, small residual review costs or overhead erase gains |
| H4 | Target customers have enough avoidable human effort to matter | Attributable active-time measurements and net-benefit bounds support a pilot | Repetition is mostly flaky CI, actual defects, policy changes or unavoidable new work |

H4 cannot be confirmed by synthetic success. An H1 gain without H2 is a delivery-scheduling result, not proof of the proposed rework mechanism. No effect with broad uncertainty is inconclusive, not equivalence.

### 4.7 Headroom and a no-headroom deduction

For measured human verification effort $C_H$, let $R_H^{\rm avoidable}$ be the cost of repeats adjudicated avoidable by feasible scheduling. A best possible gross attention saving is bounded above by $R_H^{\rm avoidable}\le C_H$. Incremental achievable savings are further limited by what the best current baseline already avoids. Use intervals when attribution is uncertain.

$$
\text{net benefit}=\text{captured repeat-effort saving}
+\text{delivery value change}-\text{integration/operation cost}-\text{delay cost}.
$$

Report labor, compute and delivery separately before monetizing them. Do not assume reviewer time has a universally known dollar value.

**No-headroom fixture:** with a fixed final change set, pinned environment, identical deterministic task DAG and common resources, an immutable-candidate policy that executes each distinct required task once attains the verification-service lower bound. A freshness controller cannot reduce that sum. A makespan improvement in this fixture concerns ordinary task scheduling, not avoided invalidation. This deduction is conditional, not an impossibility theorem for ongoing arrivals.

## 5. Proposed mechanism and implementation specification

### 5.1 Architecture and authority

Keep four modules separate: workload/evaluator truth, runtime evidence ledger, scheduler and release gate. Policies receive only an observation projection; they cannot import the truth object or future tape. Requirements, capability weights, dependency-certification rules and gate authority are outside the authoring-agent interface.

~~~mermaid
flowchart TD
  W["Versioned workload and events"] --> E["Common evidence ledger"]
  E --> O["Observable scheduling state"]
  O --> P["Queue or rollout policy"]
  P --> A["Admit, verify, hold or release"]
  A --> E
  A --> G["Common artifact and token gate"]
  E --> G
  G --> D["Modeled deployed versions"]
  W --> U["Independent outcome auditor"]
  D --> U
~~~

This diagram is an authority separation, not a distributed-system requirement. The initial implementation is one sequential process with one local database.

### 5.2 Minimum records and interfaces

Use dataclasses or equivalent explicit records, validated when loading JSON. Do not require a large schema framework.

| Record | Essential fields |
|---|---|
| Capability | id, arrived_us, weight, deadline_us?, required_change_ids, predicate_id, supersedes_ids |
| Change | id, content_digest, arrived_us, preconditions, effects, declared_scopes, predecessor_ids, urgent, legal_release_group |
| Candidate | id, parent_ids, component_versions, admitted_change_ids, write_set, dependency_read_set, policy_digest, status, first_sealed_us?, hold_deadline_us? |
| Obligation | id, definition_digest, scope_selector, environment_selector, verifier_role, bundle_id, predecessor_obligations, validity_rule |
| Evidence | id, execution_id, obligation_id, fingerprint, original_candidate_id, observed_tokens, outcome, completed_us, expiry_us?, revocation_epoch, verifier_identity |
| Execution | id, task_kind, fingerprint, start_snapshot, start_us, resource_ids, status, attempt, cancellable, resumable, consumed_service_us |
| Resource | id, role, capacity, calendar_intervals, cumulative_budget_us, spent_us, reservations |
| Decision | id, observation_digest, policy_digest, actions, estimated_outcomes, reason_codes, measured_cpu_ns, measured_wall_ns |
| Event | id, occurred_us, phase, kind, entity_id, payload, source, cause_code |

Actual finish times or sampled future service demands may exist in the simulator, but must not appear in a policy observation before they are revealed. Predictions are distributions or estimates, separately marked.

~~~python
class Policy:
    def decide(self, observation, predictor, decision_budget):
        """Return ordered proposed actions; no mutation of shared state."""

class Applicability:
    def evaluate(self, evidence, obligation, candidate, known_environment, policy):
        """Return status, reason codes, and compared tokens."""

class Gate:
    def release(self, candidate_id, expected_tokens, expected_deployed_versions):
        """Validate and atomically update modeled deployment, or reject."""

class Auditor:
    def evaluate_release(self, release_record, hidden_truth):
        """Score true applicability/capabilities without changing runtime behavior."""
~~~

These are project interface contracts to implement, not advertised APIs of an existing package. Native queue adapters translate the same observation/action schema without changing common eligibility, applicability or accounting.

### 5.3 Job semantics and costs

One execution may satisfy several obligations. Charge it once. Store an incidence map from executions to evidence and obligations; never sum the cost of each evidence item if one shared test run produces them all. Include setup, specialist switching, integration, deployment and cancellation costs when the workload specifies them.

Represent future replacement effort as a resource vector, for example:

$$
\mathbf c_g=(\text{general-review seconds},\ \text{security-review seconds},\
\text{runner seconds},\ \text{setup seconds}).
$$

A changed approval can require only a fraction of its original human effort to redo. Model this residual cost explicitly. Formal evidence remains invalid until the new required review completes; human memory reuse does not silently keep the old approval valid. Distinguish first-review cost, incremental re-review cost, familiarity and switching overhead.

The initial persistent-cost model assigns one latent full-review demand C to each (component, verifier role, obligation-definition) lineage. Sample C once from the declared role distribution; its realized value is hidden until the first review completes. Thereafter its observed active effort is public to every policy, and an invalidation-triggered human re-review costs f*C plus the declared fixed re-authorization overhead. Use the original anchor C, not repeatedly multiplied f powers. Before completion, forecasts use the shared prior conditional on elapsed service/noncompletion. New obligation definitions create new lineages. CI jobs retain their separately specified full bundled execution costs.

This persistence is an explicit favorable modeling assumption, not measured human predictability. A required diagnostic redraws replacement demands independently from the same role/job-kind distribution; past job-specific effort should then provide no information beyond declared features. Further sensitivities add noisy persistence and reviewer-specific familiarity. The actual demand is never exposed early to make R informed.

Persistent code defects are keyed to content and semantic state. Rechecking the same defect cannot randomly make it correct. Flaky outcomes and infrastructure failures are separate processes, with separate retries and budgets. The primary mechanism workload initially uses correct changes and successful applicable verification; defect/repair workloads add explicit repair arrivals, authoring delay and resource consumption.

Human jobs are noncancellable by default but **resumable across calendar closures**: accumulated active service advances only while the assigned reviewer is available, preserving the original input snapshot. This rule applies to all policies. CI calendars are continuous unless a modeled outage intervenes. Truly nonresumable calendar-bound work requires a public maximum duration or a declared charged-overtime rule. Never decide start eligibility using a hidden sampled finish time. For a running unfinished job, forecasts sample residual duration conditional on elapsed service and noncompletion, rather than restarting its duration or reading evaluator truth.

Charge active service against the current period's resource budget as it accrues. Schedule calendar and budget-boundary events; pause resumable work when the assigned resource closes or its budget reaches zero, retaining the snapshot and consumed service. Resume only after capacity/budget renewal. A nonresumable job requires a public duration reservation/bound or an explicitly charged overrun contract shared by every arm. Hidden actual duration never decides admission or permits a budget overrun.

### 5.4 Available actions

Every policy has the same legal action set:

1. **Admit(change, candidate):** check prerequisites; create an immutable successor; retain historical artifacts and evidence. Any abandoned verification still consumes its actual work.
2. **Seal(candidate, hold_until):** close its membership temporarily. Bounds are tied to the original pending work; resealing cannot reset deferral age.
3. **Isolate(change_group, legal_partition):** create an independent candidate only across a predeclared legal release cut, charging extra work and sharing all global resources.
4. **Start(task, candidate, resources):** bind the complete declared projection at start and reserve eligible capacity.
5. **Cancel(execution):** allowed only by the job contract. Consumed effort remains charged; capacity is released only after acknowledged termination.
6. **Release(candidate, expected_tokens):** an explicit action; no automatic deployment before the policy can observe the current event batch.
7. **Wait(until_event):** no-op until the next event or a bounded policy timer.

An admission does not have to retire a releasable predecessor. Permit the same bounded number of active candidates and the same predecessor-release choices for all methods. Charge retention/extra build costs equally. A shared atomic schema migration cannot be split merely to make a policy look independent.

Urgent changes follow a common externally set rule: override relevant holds when required, retain all evidence obligations, and record sacrificed verification and other work delayed. No urgent bypass waives the gate. Chronic overload is reported, not repaired by dropping normal work.

### 5.5 Event ordering and release transaction

Use a stable order (time_us, phase, event_id). At each timestamp:

1. Apply all offered arrivals and external version/policy/revocation changes.
2. Complete or acknowledge cancellation of executions; release half-open resource intervals [start, finish); charge service exactly once.
3. Update applicability projections and construct a policy observation.
4. Obtain an ordered action list, validate it against the current state, and apply actions sequentially. Gate validation occurs at an explicit release action, after any earlier admission in that list.
5. Continue immediate decisions only while the state changes under a finite action measure; otherwise advance to the next event.

Round positive service durations upward to at least one microsecond. Zero **human** cost can still accompany a positive-duration automated certificate operation. Reject events in the past and repeated no-op actions. Publish a tie-order sensitivity run; do not rely on a zero-time window that only one policy receives.

For the local gate, keep every authoritative relevant token and deployed pointer in **one SQLite database** or under the same event-engine lock. A transaction must:

~~~text
BEGIN IMMEDIATE
  read candidate, current environment and policy tokens
  compare expected candidate/environment/policy and relevant read/write deployment versions
  proposed_deployed = current_deployed with only candidate.write_set replaced
  derive required obligations for that proposed combination and legal release closure
  for each required obligation:
      require authorized, passing, unrevoked, unexpired evidence applicable to proposed_deployed
  check proposed_deployed compatibility and dependency read/write tokens
  reject unintended regression of already delivered capabilities
  insert uniquely identified release record
  update all affected deployed-component pointers
COMMIT
~~~

On mismatch, roll back and return a reason. Never accept a partial multi-component update. Canonical content digests alone do not cover an unversioned external environment.

Independent candidates may share an old base vector: releasing A and then disjoint B must retain both changes. Update only the declared write set and compare the relevant read/write tokens against the current deployed vector. Derive the obligation set from the actual proposed combination, not an older candidate's cached requirements; if that combination requires new evidence, return for verification. Reject stale overwrites; rollback or supersession must be an explicit permitted event. The runtime uses declared compatibility/acceptance rules; hidden semantic defects remain an evaluator finding, not an omniscient gate repair.

Evaluate fingerprints against proposed_deployed itself. If that combined state needs additional verification, materialize a new immutable candidate/snapshot; never mutate an older candidate or rebind historical evidence to a different observation.

This establishes an atomic action only in the **modeled local registry**, given its assumptions. A production deployment needs target-side transactions, compare-and-swap, fencing or equivalent coordination covering every critical effect. A preflight freshness check alone does not close the validation-to-effect race. [R19]

Use explicit SQLite transaction control, foreign keys and parameterized SQL. The sequential core uses rollback journaling and the backup API; WAL concurrency is unnecessary. Record the loaded SQLite version and do not mix incompatible implicit/Python-autocommit conventions. [R41]

### 5.6 Destruction cost: useful diagnostic, insufficient objective

The proposal's

$$
D(\Delta\mid S)=\sum_{e\in\mathcal V(S)}c_e
 \Pr(\Delta\text{ invalidates }e\mid S)
$$

is an expected additive loss score; linearity of expectation does not require independent invalidations. Its simplifications are elsewhere: shared job bundles, partial re-review, obsolete requirements, resource calendars and future releases.

For implementation, compute a bundle-aware **replacement-cost vector** over obligations that remain necessary after the action. Include only future work the action makes necessary, not historic effort as recoverable value. Never preserve an expensive certificate solely because it was expensive if it will not be used again. Record human seconds by role and runner seconds separately.

The exact marginal cost of an action is a **difference between complete feasible continuations**. A review postponed to the next release still costs something. Therefore the rollout objective must count actual future executions, and must not add a separate “saved evidence” reward on top of their avoided cost. Use the simple D only for explanations, candidate ordering and an explicit greedy comparator.

### 5.7 Selected controller: standard rollout with explicit evidence state

Call the implementation **rollout**, not a newly invented algorithm. It applies a conventional one-decision improvement over a strong continuation policy.

At a decision, form feasible action bundles from:

- The action of the development-selected baseline.
- Earliest-deadline/age and value-per-remaining-cost priorities.
- Immediate admission of each eligible front-of-lane change, subject to the common WIP bound.
- Sealing the affected candidate until its next completion or hold deadline.
- Each permitted independent release cut.
- Releasing any ready candidate and waiting to the next relevant event.
- Alternate verification orders of eligible tasks, using the same resource dispatcher.

Deduplicate equivalent bundles. A production-sized action space can be shortlisted, but the shortlist budget is fixed from development and includes the baseline action. For tiny correctness cases enumerate every feasible action. Do not prune the comparator's action or urgent work.

For each candidate action, simulate the continuation policy to a fixed planning horizon using the same **predicted** futures across actions. Realized future workload values are inaccessible. Forecast arrivals use development/past observations; a no-arrival-forecast variant is mandatory. Model external changes separately from endogenous admissions.

Compare feasible continuations lexicographically:

1. Satisfy hard resource, gate and maximum-hold rules.
2. Maximize predicted useful value deployed by the planning horizon.
3. Minimize the integral of unreleased offered value.
4. Minimize future human effort, then runner effort.
5. Prefer the baseline action, then stable action ID on exact ties.

For M common forecast draws, aggregate the vector as the arithmetic mean of (delivered value, negative restricted-backlog integral, negative future human seconds, negative future runner seconds), each over the same rolling horizon h from the decision. Use the fixed baseline selector on the initial observable class for each continuation; do not select a hindsight winner per draw. Structural legality is checked for every simulated action by the common harness. A future deadline miss is an outcome penalty/guardrail, not a reason to silently drop that forecast. Finite resource budgets remain enforced in all forecasts.

The evaluation cutoffs H and $H_D$ are not passed as policy features unless they coincide with a real, predeclared business deadline equally known to all methods. Planning uses its development-frozen rolling horizon. Evaluate all shortlisted actions on the same number of draws; if the deterministic transition-evaluation budget cannot complete that comparison, use the baseline action. Wall-time limits are a failsafe and their triggers are reported. No partial-evaluation winner receives preferential treatment.

The planning horizon and rollout sample count are development parameters. Include horizons long enough to reach at least the next legal release in diagnostic cases. If every continuation hits a computational cap before a useful outcome, or every fully evaluated action predicts **zero newly delivered pending capability** within the planning horizon, fall back to the baseline and record the reason. This includes an ordinary job longer than the horizon: do not let lower truncated effort make waiting forever win. Use an explicit terminal backlog penalty only in a separately specified sensitivity variant, shared by all rollout variants.

For the complete-cohort H2 population, apply a common liveness guard outside every policy: after one full recurring human-calendar period with pending work and no new capability release, commit to the frozen live baseline until at least one pending capability releases. Preserve running work, deadlines, urgency, gate and budgets. Reset the no-progress clock only on actual new capability delivery, not resealing or an empty release. The period is declared in the workload (24 simulated hours initially). Verify that the selected baseline eventually completes the finite correct pinnable cohort; this guard is a common operational rule, not an evaluator rescue available only to R.

~~~text
decide(observation O):
    b = frozen_baseline_selector(O.observable_class)
    A = legal_candidate_bundles(O, include=b.next_action(O))
    futures = predictor.draw_forecasts(O.past, rollout_seed, fixed_sample_count)
    scores = {}
    for a in A in deterministic_order:
        for f in futures:
            S = clone_observable_state_with_forecast_model(O, f)
            apply_checked_action(S, a)
            simulate_using_continuation(S, b, planning_horizon)
            record delivered_value, restricted_delay, all_future_resource_costs
        scores[a] = aggregate_without_hidden_actual_future()
    choose lexicographic best fully evaluated feasible a
    if timeout, invalid forecast, or all actions predict zero new delivery: return baseline action
    return a with observation digest and reason codes
~~~

An evidence-conditioned rollout and a generic full-state rollout with the same action set, transition model, objective, base policy and samples are **the same method**. Use one implementation. Study the value of evidence inputs through ablations, not a fictitious competition between names.

Forecast estimates initially use exact declared **distribution parameters**, equally provided to every method; sampled future outcomes remain hidden. Then replace these parameters with past-only medians/quantiles and empirical transition frequencies. Begin with simple estimators and calibrated sensitivity perturbations, not learned language models or RL. Estimation uncertainty and production inference costs are part of H3.

### 5.8 Review-order decomposition

Separate the new admission choice from job dispatch. Use a common dispatcher initially: eligible jobs sorted by deadline, then remaining critical-path length or value/remaining bottleneck effort, with a stable ID tie-break. Tune among these conventional rules on development data.

Then run a 2×2 decomposition: common/new admission crossed with common/rollout-chosen verification order. Fixed-admission replay and fixed-review-order experiments are diagnostic controls. The complete policies remain the head-to-head competitors.

If the resulting advantage comes entirely from ordinary shortest-processing-time or critical-path ordering, report that result and recommend the simpler method.

## 6. Baselines and comparison protocol

### 6.1 Shared foundation

Every competitive arm receives identical legal release partitions, applicability rules, evidence reuse, task bundles, dependency observations, cost estimates, specialist calendars, urgent-change policy, workload history and release gate. Generic improvements such as faster hashing, cached dependency closure and streamed logging apply to every arm.

Keep two axes separate: **method-level experiments** with one common ledger, and **end-to-end integration** with a real queue. The common-ledger comparison estimates scheduling value conditional on that ledger. It does not establish that the ledger can be deployed cheaply in an existing organization.

### 6.2 Required comparison roster

| ID | Policy and exact implementation requirement | Status / role |
|---|---|---|
| B0 | Immutable final candidate, each distinct obligation once; optimal tiny DAG ordering or tuned conventional ordering | No-headroom diagnostic, not a continuous-arrival competitor |
| B1 | FIFO per dependency lane; immutable active candidate; selective reuse; bounded WIP; release as soon as eligible unless batch rule holds | Honest operational floor |
| B2 | Scoped fixed or time-window batches, pinned during verification; independent scopes proceed; fixed WIP and deadline/age flush | Mandatory strongest simple competitor |
| B3 | B2 plus tuned adaptive batching and AIMD speculative window; correct downstream invalidation on predecessor failure | Documented-behavior adaptation of Mergify/Zuul ideas, not measured vendor performance |
| B4 | Current pinned SubmitQueue scheduler with its real candidate generator, scoring, sticky allocation and gate semantics; common verifier adapter | Strongest identified runnable implementation, conditional on fidelity checks |
| B5 | Standard deadline/age and value/remaining-bottleneck-work priorities with the same isolation and admission choices | Rules out generic task-ordering gains |
| B6 | Immediate bundle-aware loss-cost threshold: admit only if predicted delivery benefit exceeds future replacement/hold cost under a fixed conversion; tune thresholds | Particularly important simple explanation |
| B7 | Cost-aware reservation: protect an active scope when estimated avoided replacement cost exceeds estimated blocking cost; release at bounded expiry; common deadlock-free scope order | ATCC-inspired adaptation, explicitly not an ATCC reproduction |
| B8 | Free integration, but immutable deployment-candidate batches and common deployment gate; include integration conflicts/repair work | Tests stabilization boundary placement; does not reproduce Anthropic |
| R | Standard rollout using the full defined observation and continuation policy | Proposed application under test |
| O | Exact small complete-workload optimum with all future arrivals known; stochastic subset-state oracle where appropriate | Upper bound / harness diagnostic; foresight excludes competitive claims |

Do not require B7/B8 as full software platforms: they are explicit local policies. Include them in the primary roster only when their modeled semantics fit the declared population; record a source-grounded exclusion otherwise. B1–B6 and the applicable B4 implementation status must be resolved before a claim against strong existing methods.

### 6.3 Baseline configuration and tuning

Use shared development workloads, a separate validation set, and retained tuning results. Candidate parameter sets begin with:

- Batch size: 1, 2, 4, 8, plus a size inferred from observed backlog if the policy supports it.
- Flush delay and maximum admission hold: zero and fractions/multiples of the development median remaining verification time, clipped by the common release contract.
- WIP/speculation: from one up to the actual useful resource/active-candidate bound.
- AIMD: include the documented add-one/halve behavior and tuned alternatives; state integer rounding explicitly.
- Priorities: FIFO, earliest deadline, age, shortest remaining bottleneck work and value/remaining work.
- Cost thresholds and forecast/rollout limits: choose development ranges from observed costs and feasible delay, preserving a zero-protection and immediate-release endpoint.

These are initial search ranges, not a claim that a fixed grid is optimal. Expand a range if the development optimum is on its boundary and the expansion is feasible. Freeze it before confirmation. Match tuning access and total evaluator calls, and report wall/CPU cost; allow simple policies enough search to saturate their small spaces. Do not spend the entire tuning budget on R and compare default baselines.

Choose a **frozen feasible baseline selector** on validation data using observable workload descriptors, such as declared graph structure, ordinary duration statistics and capacity. The selector may use per-class policies if those classes are observable at decision time. It cannot use hidden true edges, realized future outcomes or the best test policy in hindsight. Compare R primarily against this selector and also report each named strong baseline.

### 6.4 SubmitQueue fidelity and fallback

The source inspected on 30 October was mutable main; an exact commit SHA could not be retrieved in this research environment. Never invent one. At execution:

1. Clone the public repository; record its exact revision, dirty state, license, Go/Bazel manifests and source hashes.
2. Inspect the pinned Speculator, path entities, outcome scorer, allocator, controller verdict and time hooks; map them to the common schema.
3. Run documented unit tests and small recorded action sequences. Test sticky work retention, finished-result reuse, failure propagation and cancellation capacity.
4. Prefer an in-process Go adapter around unchanged upstream scheduling code for virtual-time experiments. Any test clock or transport seam must be isolated and checked against unmodified code on identical scenarios.
5. Run the documented local bare-Git path for actual artifact/check integration after CPU and memory calibration.

Existing upstream commands observed in the README/development guide are:

~~~bash
git clone https://github.com/uber/submitqueue.git
cd submitqueue
git rev-parse HEAD
make test
PROVIDER=git make local-submitqueue-start
make demo-requests
make local-submitqueue-stop
~~~

These commands are proposed for the target laptop, not executed here. Inspect the pinned README before use because upstream interfaces may change. Go 1.25+ and Docker are documented for development/service work; Bazel supplies its toolchain and Compose. Use local fake or Git providers, never production credentials. [R08–R10]

Report one of three statuses: **native upstream execution**, **semantics-preserving adapter validated against upstream**, or **inspired/documented adaptation**. An adaptation is not a faithful reproduction. If an unavailable dependency or architecture-specific container prevents the faithful path, continue the CPU core, retain B4 as a contextual comparator, and state that superiority to actual SubmitQueue is unestablished. Do not quietly remove it and claim best-in-class performance.

Opaque specialist verification can be supplied through a common verifier service to both the queue and R. This is a controlled extension to both arms, not a native feature of the public queue. Preserve upstream behavior and distinguish queue-build safety from the final common release gate.

For native integration, the precise causal contrast is **tuned neutral admission plus pinned SubmitQueue** versus **rollout admission plus that same pinned SubmitQueue**, both with the same applicability/invalidation/resubmission adapter and common specialist dispatcher. This isolates upstream admission. The broader simulator may separately test review ordering. Do not give only R the authority to hold, isolate or resubmit work. If exact distribution parameters are provided, configure the upstream comparator with them where supported, including success probability one in the known-correct workload; do not leave a knowingly wrong default failure prior.

### 6.5 Broader reference results

Published Uber deployment gains, ATCC server results, risk-aware Mozilla batching and other CI studies inform mechanisms and configurations. They are not laptop head-to-head scores. The Mozilla work's performance-risk objective and its test-suite coverage choices differ from unchanged required release evidence; use only matched full-coverage policies as adaptations. Do not import an ML risk predictor without its data/splits or give R private labels. [R11, R20, R32–R33]

Random review and “rerun everything after every commit” are **diagnostic controls only**. Beating them does not establish practical competitiveness.

## 7. Data, metrics and confirmatory evaluation

### 7.1 Workload tiers and provenance

| Tier | Content and purpose | Provenance, rights and limits |
|---|---|---|
| T0 | Analytical chain and tiny scheduling fixtures | Equations in this document; supplied archive optional for exact reproduction; no customer inference |
| T1 | Deterministic local repository with services, shared library, configuration and migration | Create original fixture code and record its chosen license; real Git and tests, scripted human-review surrogates |
| T2 | Independently generated dependency/release episodes with known truth | Versioned generator, explicit distributions and seed namespaces; synthetic conclusions only |
| T3 | Public trace-informed duration/arrival sensitivity | Optional licensed metadata; historical CI is not human-approval truth |
| T4 | Prospective customer observation and controlled scheduling | Requires separately authorized access, suitable participation and actual attention measurement; not executed or required locally |

GHALogs v1.0.1, DOI 10.5281/zenodo.14796970, is a concrete optional source: approximately 513,000 runs and 2.3 million steps, with a 1.1 GB compressed run-metadata file. Its 142.3 GB log archive is unnecessary here. The primary reader did not expose a license identifier; verify primary rights before downloading/redistributing. Repository source-code licensing is not a license for all collected content. [R34]

GH Archive and public GitHub timelines/checks can inform coarse events, but they lack complete active human effort and semantic approval scope. The He et al. dataset is explicitly proprietary. Repeated-build studies also include unchanged-code rechecks due to flakiness, which must not be relabeled candidate invalidation. [R29, R35–R36]

### 7.2 The local repository fixture

Create fixtures/repo/ with four small Python packages: auth, orders, shared_contracts and schema, plus a versioned configuration file. Use standard-library tests; do not download a large application.

Prespecify these changes and their real acceptance assertions:

- An isolated orders capability preserving auth and contract fingerprints.
- A shared-contract change affecting two service checks.
- An authorization change requiring a scripted security obligation.
- A schema transition with an explicit old/new compatibility matrix and an inseparable co-release case.
- A configuration-only change invalidating a relevant obligation without changing source bytes.
- A semantically unrelated formatting/documentation change.
- A conflicting pair whose naive component replacement would erase an earlier capability.

Generate immutable Git commits and candidate manifests. Tests execute against explicit worktrees or exported trees, never mutable working-directory HEAD. Scripted specialist reviews return deterministic certificates after simulated service; these are not human evaluation. Inject an epoch change between validation and release and verify rejection. Confirm the same source and relevant projection yield the same reuse decision across policies.

### 7.3 Synthetic generator: explicit initial experimental population

Use a transparent designed population, not an asserted enterprise distribution. The following is the **initial development design**; confirmatory parameters are frozen after pilot checks for expressiveness, saturation and runtime.

| Dimension | Initial specification and reason |
|---|---|
| Topology | Equal representation of independent components, dependency chain, shared hub, modular graph with one shared dependency, and co-release migration graph |
| Scale | Start with 16 components and 64 capabilities per episode; sensitivity at 4/16/64 components and 16/64/256 capabilities without changing the mechanism |
| Arrivals | Fixed-count baseline: sort 64 independent uniform arrival times in the arrival window, equivalent to a homogeneous Poisson process conditional on its count; separate burst/on-off strata |
| Base service | General review median 10 simulated minutes; runner task median 1 simulated minute; lognormal heterogeneity with log-SD 0 or log(2); these are synthetic choices |
| Specialist cost | Security/general median cost ratios 1, 4 and 16; renormalize no-invalidation offered load when isolating heterogeneity |
| Resources | Two general reviewers, one specialist and two CI slots initially; calendars explicit; all release lanes share them |
| Offered load | Nominal maximum specialty utilization 0.5, 0.9 and 1.2 before rework; calibrate using unique required work, not observed completed work from one policy |
| Replacement fraction | 0, 0.25 and 1 of original review effort, with any fixed re-authorization overhead explicit; shared CI bundles charged once |
| Environment | Fully pinnable/no-external-change stratum plus a declared external-change stratum; external rates normalized to mean service and held common |
| Control costs | Zero and positive split/setup/switching/deployment cost; noncancellable human job default, cancellable CI with charged termination |
| Value | Fixed capability IDs and equal weights in the primary mechanistic design; prespecified heterogeneous weights/deadlines as sensitivity |
| Dependencies | Perfect declared graph in the main conditional study; conservative unknowns and undetected false-negative edges in separate robustness strata |

For a hub graph, select one shared node and connect relevant service obligations to it. For modular graphs, sample fixed-size modules, internal edges and a specified number of cross-module edges; record adjacency and random draw rules. Condense required co-release cycles into atomic groups. Reject infeasible synthetic transformations before assigning them to any policy; retain rejection counts and reasons.

**Operational generator defaults.** Components have stable IDs. An edge from A to B means A's declared relevant behavior consumes B. Independent graphs have no edges; chains use edges i to i−1; hub graphs use every nonhub component to component 0; modular graphs use groups of four with a local chain, plus each group's first component consuming component 0 (omit self-edges). Migration graphs start from the modular graph and choose two nonhub components as an atomic co-release pair. Smaller sizes truncate this recipe deterministically. Use transitive provider closure for the declared behavior fingerprint.

For each offered capability, choose a target component uniformly, except the hub topology assigns probability 1/4 to the hub and distributes the rest uniformly over nonhubs. A change adds its capability to the target's feature set; patches are compositional unless the explicit conflict test says otherwise. At graph creation, independently mark each **component** security-sensitive with probability 1/4. That flag persists, is observable and is independent of future capability arrivals.

**Default requirement rule:** a component's general review protects its behavior under the full declared transitive provider projection. First expand the proposed write set to satisfy any legal atomic-release group; then its impact set contains those written components plus their declared transitive consumer closure. Each impacted component needs a current general-review certificate, a security-review certificate if flagged, and its CI bundle covering unit and relevant contract checks. Deduplicate identical job-kind/relevant-fingerprint pairs. Thus a provider-only release may legitimately require a consumer's review even when the consumer's source bytes did not change. Unaffected components are excluded. This is an explicit conservative designed policy, not a claim that all teams require such broad human review.

Add a separate **compositional-contract policy stratum** in which a precisely specified contract certificate discharges designated consumer-review obligations. Declare those substitutions before execution and use them for every policy in that stratum. Compare scheduling within each unchanged gate policy; never give only R narrower requirements. This tests whether ordinary contract isolation eliminates the expensive human rework assumed by the default model.

Default human calendars repeat [00:00,08:00) every 24 simulated hours; the primary shift begins at 00:00. CI's two slots are continuously available, with their budgets declared separately. Jobs consume one unit of their required role unless a fixture explicitly specifies joint resources. Human tasks are noncancellable but pause/resume across closures without changing their captured input snapshot; positive-service demands are rounded up to microseconds. Calendar-gap sensitivities replace the calendar for every arm and recalculate expected offered load before freezing.

Draw raw task demands as in the table. Fix one scaling factor per graph/security-label/cost/load cell from **expected** reference per-change work, excluding invalidation repeats and batching savings. Compute expected general and security job counts J_g and J_s by summing each target probability times the size of its impact set and its flagged subset, respectively, over N offered capabilities. Include the deterministic target overrides for migration pairs. With raw mean demands mu_g and mu_s and reference calendar capacities C_g and C_s through $H_A$, set c=rho/max(J_g*mu_g/C_g, J_s*mu_s/C_s), then multiply all human service demands by c. A zero expected security count contributes zero load. A lognormal with median m and log-SD sigma has mean m*exp(sigma^2/2). Freeze this analytic rule and its per-graph factor before drawing that graph's confirmation episode arrivals/demands. Do not normalize using realized future demands or reveal their total to a policy. Record expected and realized loads, c and post-scaling medians. Keep CI demand separately recorded; a CI-saturated sensitivity scales that pool explicitly. Thus reference medians and offered load are not falsely claimed to be independent knobs.

In the co-release topology, capabilities 0 and 1 target the designated two components and are an inseparable change group with both preconditions checked. Their independently drawn arrivals remain visible individually, but the pair is eligible only after both arrive. Other capabilities follow the normal target rule. This adds dependency waiting without inventing an unavailable change or discarding inconvenient episodes. Use the expected reference per-change impact jobs for the load factor; the reference deliberately excludes savings from combining the pair. Additional migration verification, if enabled, is a separate declared job included in the expectation before freezing.

For the external-change stratum, each declared environment scope has a Poisson rate with $\lambda$ times the post-scaling mean human service equal to 0.1; sensitivity includes 0 and 0.5. A shared environment event affects its whole declared incidence set. The burst alternative draws each arrival from a four-component mixture centered at fractions 0.1, 0.35, 0.6 and 0.85 of $H_A$, with uniform half-width 0.05 of $H_A$, clipped to the window; draw component selection uniformly. This is an explicit burst generator, not a Hawkes model.

Set positive split/deployment overhead to one post-scaling median general review for human setup plus one median CI task per extra candidate; zero-overhead is a separate control. Default scheduler-imposed maximum deferral is one post-scaling median general review, with sensitivity at zero and four medians. The hold cannot override dependencies or unavailable capacity; distinguish forced waiting from voluntary deferral. These are designed parameters, all replaced only through a recorded preconfirmation amendment.

Do not run an unmanageable full Cartesian product. Use a versioned balanced development design covering every boundary and interaction believed essential. For confirmation, freeze an explicit sampling law over graph family, load, cost regime and control availability, stored as a JSON manifest. The default primary estimand gives graph families equal weight and samples the chosen load/cost settings with recorded probabilities. Report sensitivity to mixture weights; no mixture is a population estimate without customer evidence.

Preserve the numerical distributions and all generator code in the report. A favorable slice discovered during development can motivate a separately named confirmatory subpopulation, but must not be presented as a universal result. Include no-headroom and adverse strata even if the primary hypothesis concerns affected workloads.

### 7.4 Three horizons and complete accounting

Define separately:

- $H_A$: end of offered capability arrivals.
- $H$: primary delivery horizon.
- $H_D$: diagnostic drain limit.

An initial simulated operating shift is eight hours, arrivals through six hours, with subsequent daily calendars continuing into drain. This is a benchmark convention, not measured company behavior. Use development data to ensure the chosen primary horizon is not saturated at zero or one for every competent method. Freeze it before confirmation and include horizon/arrival-cutoff sensitivity.

At $H_A$, stop new offered changes for **all** arms. Through $H_D$, retain the same gate, capacity calendars, budgets per operating period and declared external process. Do not turn off external invalidation or grant free reviewers only to complete drain. If perpetual external changes make completion impossible, record censoring/blocked states. A separately frozen-environment drain is a lower-bound diagnostic only.

Report all pending values, ages and deadline misses at H. Later completion cannot erase an earlier missed deadline. Drain work detects cost moved into later releases but does not repair the primary metric.

For the **confirmatory H2 total-effort attribution population**, predeclare a finite offered cohort, fully pinnable environment, no exogenous expiry/revocation, correct compositional changes, finite positive work and repeating nonempty resource calendars. Every compared policy must eventually admit all dependency-ready work and release legal groups. Test this liveness condition before confirmation. Drain that population to completion; if a technical resource limit interrupts it, resume or classify its effort result as incomplete, never as a saving. For other populations, set a common $H_D$ before confirmation from the development upper completion quantile plus a disclosed margin, retain all censored cases, and use only bounded-horizon effort claims. Record the actual cutoff in simulated seconds and operating days. Never select completed episodes after observing which policy finishes.

### 7.5 Primary estimand and metric

For workload cluster g, let $W_g=\sum_{j:a_j\le H_A}w_j>0$. Let $I_{gj}(H)$ indicate that capability j satisfies its functional acceptance predicate in the modeled deployed state at H, following a release accepted by the fixed gate and audited for applicability at that release action. Evidence expiring after a legitimate release does not erase delivered value unless an explicit continuing-certification policy says so. Rollback or functional regression does remove horizon credit.

$$
Q_g(H)=\frac{\sum_j w_j I_{gj}(H)}{W_g},\qquad
D_g=Q_{R,g}(H)-Q_{B^*,g}(H),\qquad
\theta=\mathbb E_{\mathcal G}[D_g].
$$

$B^*$ is the development-frozen feasible baseline selector. $\mathcal G$ is the **declared synthetic workload law**, not all software teams. Primary aggregation is the mean of cluster-normalized paired differences. Report percentage-point differences and raw weighted capabilities as well. Relative improvement is secondary and undefined when the comparator is zero.

This primary theta refers to the **ideal virtual-time scheduler comparison** in Section 8.5, where controller delay is zero for all methods and verification/setup resources remain charged. The operational overhead-charged comparison has a separately frozen estimand/protocol. It uses independent measured overhead distributions conditioned on declared workload features, or randomized paired timing blocks whose entire block is the analysis cluster. Do not mix thermally correlated wall-time rows into the primary i.i.d. episode analysis or call an idealized win an established net operational benefit.

The primary claim is **more prespecified useful capability delivered under the same offered attention/compute budgets**. It is not automatically fewer consumed human minutes. A policy that uses previously idle capacity can improve Q without reducing rework. H2 requires the attribution tests below.

Empty workloads test conformance but are excluded by design from the primary population; do not define their Q arbitrarily. If a capability is rolled back or erased before H, remove its horizon credit. Record first-delivery time and availability duration separately so temporary delivery is visible.

### 7.6 Secondary metrics and guardrails

| Metric | Unit and aggregation | Why it is needed |
|---|---|---|
| Total active human service, by role | Person-seconds per fixed offered cohort through H and through drain | Counts actual simulated/observed attention, including cancellation and repeat work |
| Attributable repeat effort | Person-seconds and runner-seconds by cause, not just count | Separates evidence invalidation from defects, flakiness and new obligations |
| Restricted weighted delay | $\sum_j w_j[\min(T_j,H)-a_j]/W_g$, with $T_j=\infty$ if unfinished | Counts waiting before admission and unfinished work; units seconds |
| Pending value and age | Weight, number, age quantiles at H and $H_D$ | Exposes hidden upstream queues and deferred large work |
| Deadline and urgency outcomes | Weighted misses, lateness, urgent response time | Prevents throughput gains obtained through unacceptable deferral |
| Drain completion and cost | Completion indicator/time, total attention, runner service | Exposes cross-release displacement; incomplete drains stay censored |
| Evidence applicability violations | Releases with truly inapplicable required evidence | Hard observed conformance failure, not an efficiency tradeoff |
| Defect/acceptance failures | Independent injected-defect/capability outcomes | Distinct from freshness; small samples cannot establish real safety |
| Controller and integration overhead | Wall/CPU seconds, peak RSS, hashing/annotation cost | Reveals gains purchased with uncounted computation or attention |
| Resource utilization | Offered, reserved and consumed capacity by resource | Distinguishes budget, concurrency and actual effort |

Use the first valid delivery $T_j$ for delay, plus rollback/availability metrics when capabilities can later disappear. Do not report completed-only p95 as a guardrail: it can improve by leaving slow work unfinished. Compute full-cohort p95 only after a complete defined drain; otherwise report censoring and lower bounds.

Require zero observed true stale releases in the claimed valid regime. This is a finite conformance result, not proof of zero risk. For independent sampled clusters with no violating cluster in n observations, the one-sided 95% binomial upper rate is $1-0.05^{1/n}$; correlated releases within a cluster are not n independent trials.

Predefine acceptable delay/deadline/resource regressions from the release contract. If there is no defensible positive noninferiority margin, use zero and allow an inconclusive or tradeoff conclusion. Do not widen a margin after inspecting results.

### 7.7 Independence, randomization and leakage

The independent synthetic unit is a complete graph/workload cluster. A seed is an independent replication only when it generates a new unit from the declared law. Repeated timings, policy variants, multiple arrivals on one graph and several simulator seeds on one fixed trace are not independent organizations.

Reset all fitted/online policy state at each independent cluster using the frozen development artifacts. Within-cluster learning is allowed only if declared and equally available. Carrying learning across the confirmatory sequence would violate the i.i.d. cluster design and requires a different registered analysis.

Keep every policy paired on the same exogenous tape. Use stable semantic RNG keys containing workload ID, relevant task-scope fingerprint, artifact defect identity, attempt and purpose. Do not use whole candidate IDs for unrelated tasks or Python's process-randomized hash. Persistent defects, flaky attempts, correlated outages and policy-internal randomness have separate namespaces.

Two worlds identical in their observable history but different in hidden future must produce the same deterministic decision under the same policy seed. Add a no-op event and reorder independent jobs: future arrivals and matched task samples must remain unchanged. These are required leakage tests.

For real trace-informed cases, observed completion timestamps cannot be replayed unchanged under a new scheduler. Reconstruct service distributions and explicitly state assumptions about schedule-invariant arrivals, human behavior and test outcomes. Such replay is a model, not a causal field experiment.

### 7.8 Development, pilot, lock and confirmation

Use four namespaces: development, validation, pilot and confirmation. Do not pool pilot outcomes into the final confirmatory estimate.

1. Develop and validate the harness and baseline fidelity on deterministic fixtures.
2. Tune every method on development workloads; use validation to choose configurations and the baseline selector.
3. Use an independent pilot to estimate variance, runtime, state size, horizon sensitivity and feasible precision.
4. Write and hash protocol/confirmatory.json containing hypotheses, generator law, exact source/lock revisions, configurations, primary metric/contrast, all secondary claims, guardrails, sample size, analysis, exclusions and failure handling.
5. Generate fresh confirmatory workload identities only after that lock. Run the fixed sample once without inspecting accumulating treatment effects.
6. Preserve all outcomes. Any methodological change triggered by confirmation becomes an amendment and requires independent new confirmation for the changed claim.

Runtime/failure monitoring is allowed; stopping to fix a reproducibility or safety bug is not optional stopping for significance. Retain the failed runs and rerun all affected arms after a recorded common correction.

### 7.9 Precision, sample size and inference

No numerical sample size can be justified before the local pilot. Choose a desired half-width epsilon in Q units. When customer economics exist, make it small enough to distinguish net benefit from integration/delay costs. Without those economics, select and report an interpretable **simulator resolution**, such as a fraction of one prespecified capability in the reference cohort, and show the corresponding runtime curve. This is a precision choice, not a required percentage win or a claim of practical importance.

For a fixed sample of independent identically generated cluster differences $D_g\in[-1,1]$, use the following predeclared conservative primary interval, obtained by rescaling the empirical Bernstein bound:

$$
r_n=\sqrt{\frac{2s_D^2\log(4/\alpha)}{n}}
       +\frac{14\log(4/\alpha)}{3(n-1)},\qquad n\ge2.
$$

Report $[\bar D-r_n,\bar D+r_n]$ clipped to $[-1,1]$, with $\alpha=0.05$. This finite-sample bound can be conservative; pair it with a prespecified paired-cluster bootstrap interval as a sensitivity/precision description, not a post-hoc replacement when one is more favorable. The rescaling and assumptions are documented in [R42]. For K simultaneous contrasts, use $\log(4K/\alpha)$; freeze K.

Here $s_D^2=(n-1)^{-1}\sum_g(D_g-\bar D)^2$ is the ordinary unbiased sample variance across independent cluster differences.

Use a conservative pilot variance estimate to solve for a planned n whose projected radius is below epsilon; include the second term and the simultaneous critical value when relevant. Check the achieved final width. Zero pilot variance does not justify zero sample size or equivalence. If runtime cannot support the required precision, report the attainable interval and classify the study as locally limited or inconclusive.

If a stratified design replaces i.i.d. mixture sampling, calculate per-stratum bounded intervals with a prespecified error allocation and combine their weighted endpoints; do not apply the i.i.d. formula blindly to dependent rows. If one graph contributes several conditions, average them with fixed weights before inference or resample the complete graph cluster. Report effects by family and mixture sensitivity.

Calibrate the analysis on equivalent-policy nulls and known-difference fixtures. Bootstrap resampling repeats can increase until numerical endpoint error is small relative to epsilon; that is numerical convergence, not additional workload sampling until a positive result appears. Describe all estimation methods, interval widths and failed runs. [R43]

### 7.10 Mechanism attribution and practical value

Prespecify the admission/order factorial, cost-state ablations, common caching and no-headroom controls in Section 10. Permute cost labels **within the same resource role** while preserving total cost and eligibility; otherwise the experiment changes specialist capacity rather than information.

In the input ablation, permute the planner's **cost estimates** while keeping realized task costs and the entire workload identical across arms. A separate intervention may permute actual workload costs, but then both arms receive the same transformed workload. Never change actual costs only for the proposed controller.

**Operational H2 criterion:** prespecify the replacement-cost-information ablation in Section 10.2: retain the exact ledger, gate, invalidation transitions, actions, forecast budget and tie rules, but substitute role/job-kind replacement-duration distributions for job-specific future replacement costs. A claim that this cost information causes delivery gains requires a protected positive paired Q contrast against that ablation and a reduction in invalidation-attributed repeat effort on the same complete offered cohort, with total human effort also disclosed. This tests a particular use of verification state, not dependency tracking in general. Allocate the family error budget to these attribution contrasts before confirmation. For unbounded effort use paired cluster intervals with stated moment assumptions; alternatively normalize by a fixed finite resource budget through $H_D$ and use bounded intervals, retaining incomplete-work penalties. A censored drain cannot establish total-effort savings: supply justified remaining-work bounds or classify H2 as inconclusive. H1 can succeed while H2 remains unproven.

Record whether replaced evidence had already supported a valid earlier release, and include later-release work in the cohort. A claim of **saved human attention** additionally requires a decrease in complete-cohort total human effort with its prespecified uncertainty assessment. Merely reducing the pre-release-repeat label while moving the same work after an earlier release is reclassification, not an effort saving. Cost-information value for delivery and total-attention saving are separate conclusions.

Show a decision-level explanation for at least one positive and one harmful case: current candidate, applicable evidence, prospective repeated jobs, delay of waiting changes and actual downstream outcome. Use complete episodes to quantify whether saved rework is real or only reclassified.

For customer observations, record active review sessions or task timers, not browser-open time or approval latency as effort. Use blinded double adjudication of a sample of repeat events, record disagreement and bound unknown attribution between zero and its full repeat cost. Keep causes separate: candidate adoption, environment/policy change, actual defect repair, flaky rerun, infrastructure retry, voluntary review depth and unknown.

A future causal pilot randomizes conflict-connected release/resource blocks, or prespecified switchbacks with carryover handling when one shared component connects everything. Individual-PR randomization violates interference assumptions. Keep gate policy fixed, analyze assigned treatment even when bypassed, and preserve noncompleted work. That protocol requires customer data and consent; it is not claimed to have run here.

## 8. M3 Max feasibility and adaptive execution

### 8.1 What is feasible, and what is still unverified

The mechanism is discrete-event scheduling and small numerical linear algebra. It needs CPU, memory, local storage and Git, not a neural model. The Apple specification includes M3 Max configurations with 36 GB unified memory; it is incorrect to assume a 128 GB machine. The actual laptop may also have substantial memory committed to other applications. [R37]

**Feasibility assessment:** the standard-library simulator and small exact checks have a credible local path. Full native SubmitQueue integration, experiment throughput, peak memory and thermal stability remain **unmeasured on the target Mac**. A large stochastic confirmation may take hours or days; the pilot must determine this before committing the run. No result measured on the research Linux host establishes Apple compatibility.

Run these existing discovery commands and store their output:

~~~bash
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n hw.logicalcpu
df -h .
python3 -c 'import platform, sqlite3, sys; print(platform.platform()); print(platform.machine()); print(sys.version); print(sqlite3.sqlite_version)'
git --version
~~~

Require an ARM64 Python process on the target; investigate an x86_64 result before benchmarking. Record free storage and observed memory pressure, not just installed capacity. Verify the permitted work directory, power state and whether other substantial workloads are active.

### 8.2 Stack and dependency selection

| Layer | Default and reason | Verification / fallback |
|---|---|---|
| Simulation and ledger | Native CPython 3.12, dataclasses, heapq, hashlib, json, fractions and sqlite3 | Run exact fixtures first; standard library avoids a mandatory compiled-package path |
| Numerical analysis | Optional NumPy/SciPy for arrays, quadrature and larger systems; standard-library exact small checks remain authoritative | Resolve native wheels and run the actual required operations; lock versions and wheel hashes |
| Visualization | Optional Matplotlib for exported statistical figures; CSV/JSON tables always produced | No plotting dependency may block raw result preservation |
| Optimization oracle | Exhaustive deterministic state search and Fraction linear systems first | Optional OR-Tools only if a specified deterministic scheduling formulation needs it |
| Real artifacts | Git worktrees/exported trees and small standard-library tests | All checks reference a manifest, not mutable HEAD |
| Native comparator | Pinned public SubmitQueue, its documented Go/Bazel build and local providers | Separate build/service feasibility gate; keep source and adaptation provenance |

The inspected SciPy 1.18.1 release provides Python 3.12+ and ARM64 macOS wheels; OR-Tools 9.15.6755 provides a CPython 3.12 macOS ARM64 wheel. These package-index facts support a candidate stack, not proof that a particular local interpreter/OS installation works. NumPy's documented macOS Accelerate support can help ordinary CPU linear algebra but is not a requirement for this program. [R38–R39]

Resolve the smallest necessary dependency set in a clean virtual environment. Record exact versions, wheel platform tags, SHA-256 hashes, licenses and a reproducible lock. Install binary wheels first where appropriate; an unexpected source build is a calibration event, not a reason to silently change the algorithm. If optional scientific packages fail, retain the exact small oracle and standard-library metrics. If a required numerical operation has no faithful local implementation, mark that experiment blocked; do not replace it with a different precision or model and preserve the original claim.

If a native Python 3.12 is absent, the inspected Homebrew formula provides the existing command brew install python@3.12 and Apple Silicon packages for its listed macOS versions. Check the laptop OS against that current list. If Homebrew is unavailable or unsupported on that OS, select a native supported Python distribution from the official macOS release page, rerun the standard-library/required-wheel conformance checks, and record the version deviation before any protected run. Do not assume the newest interpreter has every optional wheel. [R48]

After selecting the interpreter, these are existing setup commands; the project files must first be created:

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m vaware env --out artifacts/environment.json
~~~

Only the last command is the proposed project API. The editable installation is for development; confirmation uses the frozen source and resolved lock. Replace python3.12 with the explicitly recorded alternative interpreter if the selection rule required one.

FreshCtx conformance and the ATR artifact are optional and belong in separate environments. The inspected FreshCtx 0.17.0 metadata supports Python 3.10–3.13 and an Apache-2.0 license; the ATR artifact documents Python 3.11. Neither is needed to execute the core. SubmitQueue is Apache-2.0; optional OR-Tools is also Apache-2.0. Preserve notices for any reused code. Public availability alone does not authorize redistribution of arbitrary trace data. [R08, R18–R19, R39]

For Docker-based local services, verify every image's architecture and pin its digest. Colima is an optional open-source local runtime with Apple Silicon support; use a compatible already-installed runtime if available. Container emulation can be an explicitly reported integration fallback, but its timings are not native ARM64 queue performance. No remote worker or paid model service may become a hidden requirement. [R47]

### 8.3 Memory and storage model

Estimate peak memory before increasing concurrency:

$$
M_{\rm peak}\approx M_{\rm interpreter}
+N_{\rm live\,events}b_E+N_{\rm live\,records}b_R
+N_{\rm edges}b_L+M_{\rm versions}+M_{\rm database}
+M_{\rm one\,forecast}+M_{\rm analysis}+M_{\rm adapters}.
$$

Measure the coefficients with representative states rather than quoting a universal Python-object size. Version vectors, full-history copies and forecast cloning are likely larger risks than arithmetic. Keep sparse projections and immutable shared structures; simulate forecasts sequentially; stream completed event history to SQLite/JSONL. Do not retain every time point or every rollout clone in memory.

An initial engineering target is below 512 MB per small simulator worker and below 1 GB for a small analysis process; these are **targets, not measurements**. If 100,000 retained Python records average an assumed 0.5–2 KB each, their storage alone would be roughly 50–200 MB; measure actual amplification before using this illustration for capacity planning.

Start with one worker. Measure whole-process peak resident memory including children and the native comparator. Reserve headroom for macOS and active applications. An initial total process budget is the smaller of 40% of physical memory and the observed safely available memory after reserving operating headroom. Choose worker count using

$$
w\le\min(\text{useful CPU limit},
\lfloor M_{\rm experiment\,budget}/(1.3M_{\rm measured\,worker\,peak})\rfloor).
$$

Reduce to one worker if the computed limit is below two; if even one worker exceeds the budget, stream/prune obsolete caches while preserving semantics. Changing graph coupling, obligation scope or the offered workload to fit memory is a new experimental population and requires a recorded amendment. Run multiple tiny episodes instead of keeping a giant batch of independent episodes resident. Never count that engineering adjustment as evidence about an untested large connected system.

Keep at least twice the projected final-run footprint plus the largest checkpoint and an explicit free-space reserve. Forecast log volume from pilot bytes per episode/decision. Preserve full event traces for fixtures, failures and a preselected sample; all episodes retain sufficient action, resource and result records to reconstruct primary metrics. If compact logs are used, publish the exact retention rule before confirmation.

### 8.4 Runtime model and calibration

Measure each stage separately: graph generation, applicability hashing, event simulation, scheduling, native queue calls, SQLite writes, metric aggregation and analysis. For policies p and independent episodes g, a first runtime projection is

$$
T_{\rm run}\approx N\sum_p t_p/w_{\rm effective}
+T_{\rm build}+T_{\rm tuning}+T_{\rm analysis},
$$

with contention and thermal margins estimated locally. Effective workers need not scale linearly with cores. Report medians, upper quantiles and failure cases, not just the fastest episode.

| Stage | Initial scale and resource procedure | Evidence before escalation |
|---|---|---|
| Exact mathematics | n up to 6–8 with Fraction arithmetic; stable formula at larger n | Correct identities; record bit-size/time growth before enlarging exact state systems |
| Tiny scheduling oracle | At most four changes and four obligations initially; explicit state cap | Agreement with enumerated examples; an interrupted search reports bounds/inconclusive, never “optimal” |
| Event harness | 16 components, 64 capabilities, one worker | Measured memory/event rates, complete replay and fixture invariants |
| Controller pilot | Shortlists and forecast counts from 1, 4, 16; expand only where outcome/budget sensitivity warrants | Full comparisons fit a deterministic transition budget; fallback rate and horizon sensitivity reported |
| Native queue | Unit tests, then one bare-Git demonstration | Actual ARM64 build and service footprint; no production credentials |
| Confirmation | Pilot-derived independent N; frozen policies/population | Projected confidence width and wall/storage budget are feasible |
| Scale study | 4/16/64 components and 16/64/256 capabilities, then an evidence-justified larger connected case | No correctness/capacity failure; meaningful interaction preserved; predicted memory and runtime fit |

The proposed conservative interval has material sample demands. At alpha=.05 and precision epsilon=1/64, its second term alone requires at least **1,310 independent episodes**; using an assumed paired SD of .05 gives approximately 1,701 and SD .2 gives approximately 3,575. For epsilon=1/128 the corresponding planning counts are about 2,619, 3,784 and 10,315. These are calculations from the bound, not measured sample requirements; multiple protected contrasts increase them.

For illustration, 2,000 episodes at an aggregate one second across all compared policies require about 33 minutes before overhead on one worker; 30 seconds require about 16.7 hours. Replace these illustrative rates with the pilot. If the required run is too costly, improve a common implementation, use a cheaper development-selected controller, or report attainable precision. Do not silently replace the protected interval with a favorable bootstrap result or stop early after a positive estimate.

The fixed interval can be conservative. A less conservative primary analysis is permitted only through a preconfirmation amendment supported by its assumptions and a null-calibration study; it cannot be chosen after observing confirmation outcomes.

### 8.5 Timing and overhead conventions

Separate two comparisons:

1. **Ideal scheduler study:** virtual-time operations have zero controller delay, while all modeled verification/setup work is charged. This isolates scheduling choices.
2. **Operational study:** measure actual decision time on the Mac and charge it as dispatch delay or execution occupancy under a declared deployment architecture. If events occur during a decision, revalidate the action's observation digest before applying it. Already running work continues; do not freeze the entire system merely because one controller computes.

Charge common dependency indexing/hash checks to both arms; charge additional estimator computation, annotation and maintenance only where actually incurred. The production estimator's human maintenance cost remains unknown unless measured.

Warm the interpreter/imports and native service before steady-state measurements; separately report cold startup. Randomize policy timing order, use isolated output databases, and state cold/warm filesystem and evidence-cache conditions. Fix BLAS thread counts when used and record multiprocessing configuration; macOS processes use explicit safe startup and no inherited database connections. Python's process and transaction conventions must be checked against the pinned interpreter. [R40]

Record wall and process CPU time, whole-process peak RSS, memory pressure, power/thermal conditions and other active workloads. macOS and Linux report some resource fields in different units; normalize to bytes and seconds. A longer repeat reduces measurement noise, not workload sampling uncertainty. The core does not use a GPU, so GPU synchronization is irrelevant.

### 8.6 Checkpointing and interruption

Use one authoritative SQLite writer and explicit transactions. The Python backup API can produce consistent checkpoints; copying a live database file without its journaling state is not sufficient. Prefer rollback journaling in the sequential core. If WAL is introduced, verify the loaded SQLite revision against current upstream fixes; the official WAL documentation records a race fixed in 3.51.3, reinforcing the need to record the actual embedded library. [R40–R41]

Checkpoint after a complete event batch and after each episode, with the state digest, event cursor, random-stream identity, accumulated metrics and learned policy state. On interruption, roll back an incomplete transaction and resume from the last committed state. An out-of-memory or disk-full failure must preserve the failure reason and last consistent manifest. Resume only with matching code/config/data/lock hashes; a changed methodology creates a new run lineage. Run the small crash/resume fixture in WP03 and repeat it before the WP07 freeze; WP10 performs the post-experiment reconstruction. Section 11 gives the required recovery test.

## 9. Ordered work packages and decision gates

The coding agent may choose routine names, internal data structures and non-destructive optimizations consistent with this specification. It may not silently change the scientific question, workload law, protected analysis, gate or comparator access. Record those changes as methodological amendments.

The dependency path is WP00 → WP01/WP03 → WP04/WP05 → WP06 → WP07 → WP08/WP09 → WP10/WP11. WP02 can run alongside the local work; its absence limits customer claims. WP12 is a separate conditional extension. Baseline-native integration in WP04 must be resolved before any protected claim requiring that comparator.

### WP00 — Discover hardware, dependencies, rights and scope

- **Objective/risk:** establish a truthful local execution path and resolve access/licensing before investment.
- **Entry/inputs:** this document, a fresh M3 Max work directory and ordinary network access for permitted public code. The original archive is optional.
- **Actions/files:** create the repository layout in Section 11; save hardware.json using Section 8 commands; create pyproject.toml and minimal lock; implement the proposed command python -m vaware env --out artifacts/environment.json. Inspect exact upstream licenses and revisions; create provenance/sources.json and licenses/. Mark unavailable optional datasets explicitly.
- **Checks/output:** native architecture and interpreter, SQLite transaction/backup smoke check, import/operation checks for every selected compiled dependency, writable storage, recorded source revision or unresolved status. Output all sizes in bytes and times in seconds.
- **Resources:** one process; no dataset larger than metadata needed for its license decision; measure baseline RSS and disk reserve.
- **Gate/branches:** pass to WP01/WP03 if the standard-library core works. Repair ordinary installation faults. If optional wheels/native queue fail, use the faithful fallback with a narrower claim. If the core cannot run faithfully, stop local execution and report the exact blocker; do not substitute remote compute.

### WP01 — Reproduce mathematics and expose timing assumptions

- **Objective/risk:** verify the benchmark and prevent a theorem-to-simulator mismatch.
- **Entry/inputs:** WP00; formulas and executable reference in Appendix A.
- **Actions/files:** implement src/vaware/oracle/chain.py and tests/test_chain.py; implement math-check --out artifacts/math_checks.json. Use Fraction transition systems independently of the closed formula. If the archive is available, reproduce its routines in an isolated output directory and record its supplied hashes separately.
- **Checks/output:** exact means and PGFs for small n/q/s; q=1 gives n−k; n=1 gives one round from stale; large-n log-domain evaluation; critical centered means; explicit distinction between fresh-at-end theorem mode and start-snapshot production mode.
- **Resources:** seconds-scale checks are plausible from the research reproduction but not promised for the Mac; measure local time/bit growth. Avoid huge exact rational systems.
- **Gate/branches:** algebraic disagreement blocks later theorem validation until diagnosed. Numerical overflow with a correct exact oracle triggers stable arithmetic, not a theory rejection. Passing checks supports the benchmark, not scheduling novelty. Missing old full texts leaves historical priority unresolved without blocking empirical work.

### WP02 — Establish practical headroom or mark it unknown

- **Objective/risk:** determine whether avoidable verification rework deserves product investment.
- **Entry/inputs:** WP00; optional explicitly authorized telemetry or interviews. No private system access is assumed.
- **Actions/files:** create protocols/observation.md and schemas/repeat_event.json. Define original/repeat obligation, candidate fingerprint, cause, active-time measurement, shared execution, partial reuse and later-release accounting. Build ingest validation using synthetic rows first. For permitted observations, sample conflict-connected episodes and double-adjudicate repeat causes.
- **Checks/output:** denominators include all offered work and all measured review effort; unknown cause is retained with lower/upper attribution bounds. Report avoidable human effort, best-baseline residual headroom, collection cost and missingness. Do not estimate active attention from merge latency.
- **Resources:** tiny metadata ingest; optional public traces begin with a licensed subset and measured expansion factor, not full log archives.
- **Gate/branches:** absent data → continue the conditional local PoC with H4 unresolved. Clearly negligible residual headroom → recommend release discipline and stop product expansion. Material but uncertain headroom → retain a measurement agenda. Poor measurement → revise instrumentation, not declare the mechanism present or absent.

### WP03 — Implement the common ledger, gate and event harness

- **Objective/risk:** eliminate invalid releases, cost undercounting, leaked futures and artificial invalidation.
- **Entry/inputs:** WP00; WP01 timing conventions; Sections 4–5 and fixture specifications in Section 10.
- **Actions/files:** implement model.py, events.py, resources.py, applicability.py, gate.py, metrics.py and JSON schemas. Add a truth/observation boundary, deterministic event IDs, replay, checkpoint transactions and exact cost charging. Create the small Git fixture and artifact-bound test runner. Implement fixture --suite core --out artifacts/fixtures.json.
- **Checks/output:** all deterministic conformance cases pass, including disjoint merge preservation, new requirement derivation, stale in-flight completion, duplicate callbacks, shared bundles, deadline visibility, policy/environment races and crash replay. Output event JSONL, gate verdicts, cost ledgers and final digests.
- **Resources:** one worker, small graphs; record per-state/event memory and service-calendar edge behavior.
- **Gate/branches:** any invariant failure is an implementation defect and blocks comparative claims. Excess memory prompts streaming or cache changes applied to all arms. A required external atomic effect unsupported locally narrows the gate claim to the modeled registry; it does not justify pretending production atomicity.

### WP04 — Implement and validate the strongest feasible baselines

- **Objective/risk:** prevent a positive result caused by weak release discipline or an inaccurate queue reproduction.
- **Entry/inputs:** WP03; baseline roster and pinned current source. No proposed controller is required.
- **Actions/files:** implement policies/fifo.py, batch.py, priorities.py, reservation.py and baseline_selector.py; write baseline configuration grids; add adapters/submitqueue/ with revision/license and conformance mapping. Execute documented upstream tests before any adapted clock/transport. Implement baseline-check --config configs/baselines.json --out artifacts/baseline_checks.json.
- **Checks/output:** equal legal actions and information; exact batch flush and AIMD behavior; stable active jobs/results; correctly charged cancellation; fair probability scorer; same resubmission/final gate. Compare native and adapter action traces on tiny workloads. Record the native/adapted/inspired status explicitly.
- **Resources:** measure build and service RSS separately. Begin with unit tests; start containers only for integration that needs them.
- **Gate/branches:** native fidelity passes → include B4 in its matched upstream-admission contrast. Fidelity uncertain/blocked → preserve adapted results but prohibit a claim against actual SubmitQueue. Another baseline already implements the entire proposed state/action rule → mark replication and continue only if the benchmark or empirical comparison has value. No missing native comparator is silently omitted from the report.

### WP05 — Determine expressive headroom with exact small cases

- **Objective/risk:** show that the harness can express both benefit and no benefit before expensive search or tuning.
- **Entry/inputs:** WP01, WP03 and simple policies from WP04.
- **Actions/files:** implement oracle/schedule.py for finite deterministic workloads using complete-state enumeration with explicit state caps; create fixtures/headroom/*.json. Start with up to four changes/obligations, a horizon at most 20 ticks, positive integer service demands, a finite hold/action set and one-tick waits. Enumerate the entire declared discrete model before calling its solution exact. An event-only restriction needs a completeness argument; otherwise it supplies only a restricted-action feasible result. Separate an omniscient offline oracle from every causal policy.
- **Checks/output:** immutable service lower bound; cross-release trap; heterogeneous two-review values; common-bundle accounting; binding resource calendars; known-difference and identical-policy nulls. A capped search reports its best feasible value (a lower bound on the offline maximum) and only separately justified admissible upper bounds. Report which action differences cause them.
- **Resources:** log explored states, peak memory and termination reason. Exhaustion is not optimality.
- **Gate/branches:** a proxy with no interacting evidence cannot falsify the mechanism; revise to the prespecified expressive case. An expressive case with no extra headroom beyond strong baselines supports simplification and limits the controller agenda. Existing causal headroom or unresolved expressive uncertainty permits the minimum rollout implementation. No requirement to observe a positive R result before implementing R creates a circular gate.

### WP06 — Implement the minimum complete controller

- **Objective/risk:** test the incremental scheduling value of explicit state using known rollout.
- **Entry/inputs:** WP03–WP05 passed; selected base policy, finite action generator and observation schema.
- **Actions/files:** implement policies/rollout.py and forecasts.py exactly as Section 5.7; fixed common forecast draws, conditional remaining service, deterministic transition budget, common gate/resource transitions and baseline fallback. Implement explanation rows listing future marginal jobs without a second “saved evidence” reward.
- **Checks/output:** include baseline action; identical full-state implementations agree; no future tape or actual finish time in observations; no-op events do not alter exogenous randomness; all actions fully scored with the same M; bounded holds cannot reset. Check ablations change only specified planning inputs.
- **Resources:** one worker; profile planning by action/draw count and episode size. Cache common immutable projections identically across policies.
- **Gate/branches:** invalid action/future leakage → fix before outcomes. Frequent computational fallback → calibrate a smaller declared controller or retain simple policy. Failure to beat a baseline on development is evidence for tuning/expressiveness checks, not permission to search hidden confirmation results.

### WP07 — Pilot, tune, calibrate and freeze

- **Objective/risk:** establish baseline strength, precision, local feasibility and a complete protected protocol.
- **Entry/inputs:** WP04–WP06; versioned development/validation/pilot namespaces.
- **Actions/files:** run pilot --config configs/pilot.json; tune on development, select on validation, and estimate variance/resources on the independent pilot. Include candidate horizon and arrival-cutoff sensitivity, mixture weights, max-hold, terminal behavior and controller overhead. Freeze explicit generator probabilities, expected-load factors, policies, shortlist sizes, forecast budgets, sample size, contrasts, error allocation, guardrails, drain handling, raw-log sampling and exclusions. Implement freeze --protocol protocol/confirmatory.json.
- **Checks/output:** meaningful nonzero/nonunit delivery variation; equivalent-policy null calibration; enough headroom and local precision to interpret the test; actual workload/class observability for the baseline selector; a recorded content-hash manifest and protocol audit table.
- **Resources:** use measured rates and memory in Section 8, including all tuning. Report required N and projected duration; neither a tiny arbitrary N nor an arbitrary maximum runtime becomes a hidden success criterion.
- **Gate/branches:** inadequate sensitivity → revise population/horizon transparently before confirmation or declare the feasible precision. Resource limits → optimize shared implementation or choose a frozen cheaper policy; do not remove difficult test cases. Baseline matches R precisely → a negative comparative study is legitimate. Proceed only when the protected protocol is fully specified, not only when pilot effects look favorable.

### WP08 — Execute the protected comparison

- **Objective/risk:** estimate H1 without selection or optional-stopping bias.
- **Entry/inputs:** immutable WP07 protocol, fresh independent confirmation identities, passed WP03/WP07 crash-resume smoke fixture and space/runtime budget.
- **Actions/files:** run --protocol protocol/confirmatory.json --out runs/<run_id>; run the same offered workload for every arm with paired exogenous draws and isolated policy state. Monitor failures/resources without inspecting accumulating treatment effects. Implement analyze --run runs/<run_id> --protocol protocol/confirmatory.json.
- **Checks/output:** primary Q and paired interval; every named comparator; pending/backlog/deadlines, resource effort, applicability violations and measured controller overhead. Include failures/censored cases and prespecified protected contrasts. Preserve full common episode manifests.
- **Resources:** pilot forecast with periodic resource-only checks; resume rather than restart completed correct episodes.
- **Gate/branches:** confirmed H1 → proceed to the already prespecified attribution checks, not a breakthrough claim. Narrow nonpositive interval → evidence against H1 in this population. Broad interval → inconclusive. Harness defect → retain failed artifacts, correct all affected arms and issue an amendment; changed claims need new independent confirmation. Applicability violations disqualify safety-equivalent superiority in that regime.

### WP09 — Attribute gains and test robustness

- **Objective/risk:** distinguish evidence preservation from batching, priority, compute, forecasting or shifted cost.
- **Entry/inputs:** WP07-frozen Section 10 program; it may run alongside WP08. Post-hoc questions use new namespaces.
- **Actions/files:** run --protocol protocol/attribution.json for protected H2 contrasts and full-cohort drain; run separately labeled robustness manifests. Use matched admission/order factorials, planner-state masking, partial review reuse, declared unknowns, hidden missing edges, resource calendars, setup costs and noisy forecasts.
- **Checks/output:** H2 paired delivery and repeat-effort results, total complete-cohort effort, cost/latency frontiers, adverse examples, maximum-hold compliance and no-headroom controls. Analyze correlated conditions at the cluster level.
- **Resources:** fixed planned ablations; screen wide parameter ranges in development, confirm only named hypotheses with allocated error.
- **Gate/branches:** gain disappears after sharing caches/compute/tuning → reject the claimed mechanism. H1 remains without H2 → classify ordinary delivery improvement. Robust only under perfect dependencies/high full-repeat costs → narrow the supported regime. Unsafe hidden-edge results → document the dependency-certification limitation; do not “fix” the test with oracle information.

### WP10 — Independently reconstruct and recover

- **Objective/risk:** ensure the result survives a fresh environment and interruption rather than merely re-running a live process.
- **Entry/inputs:** locked code, protocol, raw manifests and a second clean local virtual environment; no second physical machine required.
- **Actions/files:** use the documented setup from scratch; rebuild a preselected subset from manifests, including an adverse episode; replay raw events into metrics without policy execution; kill and resume at each defined transaction boundary; implement resume --run runs/<run_id> and reproducibility report.
- **Checks/output:** exact event/final-state hashes where deterministic, identical primary metrics, declared tolerance only for floating predictions, no duplicated effort/releases, rejected incompatible resumes, confirmed raw-to-summary reconstruction. Record cold versus warm integration costs.
- **Resources:** small representative subset and failure fixtures; broaden only for a concrete discrepancy.
- **Gate/branches:** mismatch → locate nondeterminism/version drift/transaction error and correct affected results. Native comparator nondeterminism → control/log its random source and compare distributions appropriately, without inventing bit identity. Passing supports reproducibility on the local configuration, not universal platform validation.

### WP11 — Synthesize, report and choose the next research action

- **Objective/risk:** preserve negative results and match each conclusion to its evidence.
- **Entry/inputs:** completed or explicitly limited WP08–WP10; WP02 status.
- **Actions/files:** implement report --run runs/<run_id> --out reports/final.md. Produce the exact report table in Section 11, protocol/amendment index, effect intervals, baseline fidelity status, resource report, audit record and reproducible commands. Compare actual results with each hypothesis; do not change their definitions.
- **Checks/output:** raw-to-summary traceability; all offered capabilities and failures accounted for; causal and simulated language correct; no practical or methodological novelty inferred from significance alone.
- **Resources:** streamed aggregation and static figures; no new model/data requirement.
- **Gate/branches:** use Section 12 to continue, simplify, narrow, publish a negative benchmark, or stop product expansion. A scale extrapolation must identify a measured trend and an untested prediction rather than promise future gains.

### WP12 — Conditional prospective operational study

- **Objective/risk:** test H4 and actual human attention savings when local results justify it.
- **Entry/inputs:** authorized participating team, adequate baseline measurement, compatible release control, unchanged gate and accepted research protocol. This is outside the required local PoC.
- **Actions/files:** finalize observation definitions, block/switchback assignment, carryover, business weights, active-effort recording and incident adjudication. Begin with shadow recommendations, then a separately authorized controlled intervention. Record bypasses and analyze assigned condition.
- **Checks/output:** complete delivery cohorts, actual attention and operation cost, uncertainty, important regressions and incident context. Review latency alone remains insufficient.
- **Resources:** estimate from measured instrumentation and support effort; no unsupported revenue or staffing forecast.
- **Gate/branches:** no practical headroom or no net benefit → recommend simpler queue/isolation configuration. Positive credible causal benefit → consider integration/product work. Insufficient observations or rare-event power → report uncertainty, not safety or market proof.

## 10. Conformance, attribution and failure-analysis program

### 10.1 Deterministic fixtures with exact expectations

These tests are necessary scientific checks for the harness, not a requirement to write low-value tests for every implementation detail.

| Fixture | Construction | Required result / interpretation |
|---|---|---|
| Immutable lower bound | Fixed final candidate, pinned environment, deterministic required task DAG, equal resources | Each unique required execution once achieves the service lower bound. Any remaining makespan gain is ordinary scheduling |
| Cross-release effort trap | A arrives at t=0, B at t=4; B changes a shared approval input; noncancellable approval duration 5 | Mutable execution at 0–5 is stale, then 5–10: both release at 10, effort 10. Freeze: A releases at 5, B at 10, effort still 10. A prespecified batch tick at 4 releases both at 9, effort 5 |
| Heterogeneous two-review chain | Fresh-at-end model, unit service, q_H=.1 and q_L=.9 | Starting by H costs 281/91≈3.087912 expected rounds; starting by L costs 201/91≈2.208791. Count-only state is insufficient |
| Candidate-relative evidence | Certificate supports X; successor X′ changes a relevant projection while X is retained | Certificate remains usable for X and unusable for X′; no global “invalid” flag |
| Disjoint stale-base candidates | A and B fork the same base but write disjoint components | Release A then B retains both. A relevant read-token conflict rejects B pending rebasing/reverification |
| Requirements after combination | Combining an old candidate with current deployed versions adds a declared contract obligation | Gate derives the new requirement and rejects missing evidence; no cached-old-requirements bypass |
| In-flight start snapshot | Start a job for X; create a successor before completion | New result can serve the successor only if its authorized relevant fingerprint still matches; original candidate ID alone is not a reuse key |
| Shared bundle and duplicate callback | One execution emits three obligations; callback repeats | Charge work once, store three certificates, apply completion/release idempotently |
| Hidden future | Two tapes identical up to t but different afterwards | Policies have identical observations and decision distributions at t; no realized task end time leaks |
| Random-stream stability | Insert an unrelated no-op event or reorder independent logging | Exogenous arrivals, durations, failures and environment events retain their draws |
| Equal-time race | External relevant token changes at exactly verification completion/release time | Common event ordering applies the mutation first; invalid release is rejected |
| Token transaction | Change a token between preflight and commit; interrupt a multi-component update | Compare-and-swap/transaction rejects or rolls back all; no partial deployment |
| Outcome versus applicability | Current FAIL, stale PASS, expired PASS, revoked PASS, unknown applicability | Only authorized current PASS is eligible; each rejection has a distinct reason |
| Capability accounting | Split one capability into ten PRs; later overwrite a previous capability | No extra value from splitting; missing acceptance predicate removes horizon credit |
| Bound on repeated seals | Repeatedly seal/unseal the same waiting work | First voluntary-deferral timestamp persists; no reset of maximum hold |
| Long job beyond planning horizon | One necessary job takes longer than h; there is no nearer release | Zero predicted new delivery triggers the baseline action; repeated short horizons cannot cause permanent idling |
| Calendar and urgency | Specialist unavailable; an urgent change conflicts with a hold | No work outside calendar; urgent policy overrides hold but never the gate; impossible deadline is reported |
| Unknown versus hidden edge | Explicitly unknown scope; separately, an omitted edge known only to evaluator | Conservative runtime handling for the first; evaluator detects the second without feeding oracle information back |
| Recovery | Interrupt just before/after event and release commits | Resume has the same final state, effort and release digest as uninterrupted execution; mismatched configuration refuses resume |

The effort-trap batching time is a **predeclared periodic tick**, or else an explicitly clairvoyant oracle. It cannot be treated as a causal policy that happened to know B would arrive at 4. Freeze's earlier A release can have genuine value, but relabeling a stale execution as a new release's verification does not save attention.

For the two-review fixture, if V_H denotes the remaining mean when only H is current, then V_H=1+(1−q_H)V_L and V_L=1+(1−q_L)V_H. Add the first round to obtain the stated totals. This fixture must not be generalized to arbitrary specialties or used to claim that an older repair theorem is wrong; conventions and objectives must first match.

### 10.2 Required attribution contrasts

| Contrast | What changes | What stays fixed | What it diagnoses |
|---|---|---|---|
| Admission × ordering | Conventional/rollout admission and conventional/rollout verification order | Workload, gate, scope, capacity and queue engine | Admission value versus ordinary task-order improvements |
| Job-specific human replacement cost | Actual planner estimates versus role-level replacement-cost estimates | Actual job costs, dependencies, ledger, legal actions and computational allowance | Value of knowing which human evidence is expensive to replace |
| Forecast cost-label permutation | Within-role estimated repeat costs shuffled | Real workload and realized resource demands | Whether decision-relevant cost alignment matters |
| Initial evidence abstraction | Forecast starts with no reusable historical certificates, while live eligibility remains exact | Runtime gate, actual ledger and costs | Diagnostic effect of a deliberately conservative planning abstraction; not a competitive baseline |
| Simple loss-cost rule | Greedy D threshold versus full rollout | Same D inputs and actions | Whether continuation search adds anything beyond a transparent local rule |
| Future arrivals | No arrival forecast versus shared calibrated forecast | Past observation and actual future tape | Dependence on prediction quality |
| Full versus partial re-review | Replacement fraction 1, .25, 0 plus measured/declared fixed overhead | Original task structure and resource capacity | Whether alleged savings vanish when humans retain useful reasoning |
| Shared optimizations | Common caching, dependency closure, incremental hashing and job bundling enabled for all | Controller choice | Gain from generic implementation rather than scheduling |
| Compute allowance | Matched forecast/evaluator budget, then actual operational overhead | Workload and resource accounting | Extra planning computation versus useful delivery |
| Stable release discipline | Immutable scoped batching and legal split/setup choices | Obligations and capability goals | Whether ordinary isolation eliminates the headroom |

The protected cost-information ablation must be implemented concretely. For **not-yet-started human replacement jobs only**, substitute a development-frozen role/job-kind replacement-duration distribution independent of the particular evidence's original effort. Retain exact current applicability, required jobs, relevant fingerprints, CI cost forecasts and future invalidation rules. Preserve elapsed-time conditioning for already-running work. The planning simulator still uses the common gate on its simulated state. Actual work and costs remain identical across policies. Do not let the ablated code reconstruct the masked cost from the old execution duration as a side channel.

Required null: with log-SD zero, equal replacement fractions/overheads within role/job kind and common forecast draws, the full-cost and role-surrogate models produce identical forecasts and decisions. Any difference indicates an implementation, tie-breaking or random-stream confound. The independently redrawn replacement-cost regime is a second, nontrivial information-value control.

This contrast tests the value of **job-specific replacement cost**, not the invention of dependency tracking. A broader removal of evidence state is a separate diagnostic abstraction, not a substitute for the strong existing baselines. Never disable forecast invalidation and let impossible stale evidence pass a fictitiously weaker gate.

Use the same code for R and these variants, with typed observation/model transformations and a diff assertion listing changed fields. Preserve common forecast random identities; altered distribution parameters may change forecasts but must not alter realized workload draws.

### 10.3 Boundary conditions and adverse regimes

Prespecify these conditions, including unfavorable ones:

- Zero external invalidation, no new arrivals and an immutable fixed final candidate.
- Independent components where shared queues should already avoid interference.
- High overlap with costly specialist re-review, and the same overlap with nearly free partial re-review.
- Correlated shared environment changes that no admission hold can prevent.
- Heavy load above capacity, specialist vacations/gaps, urgent arrivals and long jobs that cannot be cancelled.
- Expensive split/deployment setup, so tiny releases can lose despite lower invalidation.
- Cheap high-value tasks versus expensive indispensable prerequisites; ensure value prioritization cannot abandon the latter invisibly.
- Incorrect cost estimates, stale duration estimates, heavy tails, bursty arrivals and changed cost/arrival correlation.
- Conservative unknown dependencies that expand scopes, then hidden false-negative dependencies audited independently.
- Passing-but-incomplete tests and injected defects, with defect escape separated from evidence applicability.
- Changes that alter requirements or policy, revoke a result, become obsolete, supersede another capability or require an atomic migration.
- Planner timeouts, native queue failure, service restart, out-of-order callbacks and disk exhaustion.

Do not assert an exponential completion law for these heterogeneous cases without a separate derivation. Scaling can demonstrate a measured growth trend and sensitivity; a finite range of fitted curves does not prove asymptotic exponential behavior.

### 10.4 Resource and numerical stress rules

Increase graph size and capability count separately to distinguish **coupling size** from independent throughput. Run hub and chain cases, not only many disconnected copies. Preserve the ratio of external-change rate to service time and the declared arrival/load model when testing a particular dimensionless scaling.

For lognormal/heavy-tail sensitivities, state the distribution and finite moment assumptions. Rare very long jobs cannot be discarded as outliers after viewing results. Record censored duration, blocked resource and cumulative service separately. Exact-identity tests use rational arithmetic; probabilistic scheduling uses ordinary double precision with recorded comparison tolerances and stable tie IDs. Do not use reduced precision or quantization to make one method appear faster.

Safety-equivalent success requires zero observed applicability violations in the tested regime, zero deliberate gate weakening and no hidden extra resources. Zero observed failures is not a proof of a tiny real-world failure rate, especially under incomplete dependency discovery. Report applicable-evidence failures, functional defects and policy/deadline regressions separately.

### 10.5 Escalation criteria

Move to larger connected cases only after correctness holds and the current case can express the hypothesized interaction. Escalation is justified by measured unresolved headroom, a stable effect whose mechanism depends on coupling, or a concrete boundary question. It is not justified solely by a negative result and a hope that scale will reverse it.

If the smallest case lacks sufficient overlap or calendar contention, increase those declared dimensions before concluding against the mechanism. If an expressive, adequately precise comparison shows no advantage over tuned baselines, prefer the simpler policy and record the supported null/negative finding.

## 11. Repository, experiment records and reporting

### 11.1 Minimal repository layout

| Path | Required content |
|---|---|
| README.md | Setup, evidence status, minimal run commands, native-comparator status and resume instructions |
| pyproject.toml; requirements.lock; licenses/ | Interpreter/dependency constraints, resolved versions/hashes and notices |
| src/vaware/model.py; events.py; resources.py | Typed state, event queue, service/calendar/budget accounting |
| src/vaware/applicability.py; gate.py | Fingerprint projection, requirements/impact closure, atomic release checks |
| src/vaware/workloads.py; forecasts.py | Separate workload truth and past-only forecast generators |
| src/vaware/policies/ | Shared policy protocol, baselines, selector, rollout and precise ablations |
| src/vaware/oracle/ | Exact chain checks and bounded tiny scheduling search |
| src/vaware/adapters/ | Real Git/check runner and pinned native-queue bridge |
| src/vaware/metrics.py; analysis.py; cli.py; __main__.py | Metric reconstruction, protected inference and proposed CLI/module entry point |
| tests/; fixtures/ | Deterministic conformance cases, generated original repository and their expected outputs |
| schemas/; configs/; protocol/ | Record schemas, development configurations, protected protocol and amendment history |
| provenance/; data/raw/; data/derived/ | Source/license records and immutable permitted input data with transformation hashes |
| runs/<run_id>/ | Run manifest, episode manifests, events, decisions, costs, database, checkpoints and failures |
| reports/ | Tables/figures, environment report, audit, interpretation and exact reproduction instructions |
| decisions.jsonl; progress.md | Durable progress, blocked prerequisites, decisions and next steps |

Use Git for the project. Record exact source revision and a dirty-tree patch hash, or require a clean commit for confirmation. Never push code, publish data or contact teams merely because the local plan mentions them; those are separate actions.

### 11.2 Configuration conventions

Validate configuration before generating workload data. Unknown keys are errors. Use schema versions, units in field names, explicit null for unresolved prerequisites and no hidden defaults in a protected run. A configuration is development-only until all confirmatory fields are resolved and hashed.

This is an initial **development configuration**, not a ready-made frozen study:

~~~json
{
  "schema_version": 1,
  "phase": "development",
  "time_unit": "microsecond",
  "components": 16,
  "capabilities": 64,
  "arrival_model": "uniform_conditional_count",
  "arrival_end_s": 21600,
  "primary_horizon_s": 28800,
  "human_calendar_period_s": 86400,
  "human_available_interval_s": [0, 28800],
  "general_reviewers": 2,
  "security_reviewers": 1,
  "ci_slots": 2,
  "raw_general_median_s": 600,
  "raw_ci_median_s": 60,
  "security_median_ratio": 4,
  "duration_log_sd": 0.6931471805599453,
  "component_security_probability": 0.25,
  "nominal_max_human_load": 0.9,
  "replacement_fraction": 0.25,
  "requirement_rule": "transitive_consumer_impact",
  "external_changes": false,
  "scheduler_hold_limit_general_medians": 1,
  "worker_count": 1,
  "pilot_initial_episode_count": 16,
  "confirmatory_sample_size": null,
  "confirmatory_allowed": false
}
~~~

The generator manifest additionally specifies the chosen topology/mixture, target probabilities, policy variant, setup overhead, resource period budgets, expiry/revocation rules, fixed cost components, common WIP/candidate cap, urgency/deadline distribution and RNG namespaces. Use the defaults in Section 7 for the first fixtures; an absent applicable field blocks freeze. The initial pilot count is a throughput/variance probe, not a justified final sample size. Expand the independent pilot only to resolve a named calibration uncertainty and retain all pilot results.

The human role period budgets initially equal their available active calendar time; CI budgets equal the sum of its continuously available slot time in each 24-hour period. Additional tighter budget sensitivities keep calendars unchanged and trigger the pause rule in Section 5.3. No prediction can spend tomorrow's human budget today.

The frozen protocol must include the exact scalar conversions and tie rules for B6/B7, not merely “tuned thresholds.” Represent their admission score in a common unit such as weighted-capability seconds; fit a conversion from role effort to that unit only on development data, publish the coefficients and include zero-protection endpoints. Unconverted addition of a dollar, a CPU second and a human second is invalid.

### 11.3 Randomness and information separation

Use distinct namespaces for development, validation, pilot, confirmation, bootstrap, workload truth and planner forecasts. Store each independent episode's master seed and a stable seed-derivation algorithm. Python's randomized built-in hash is unsuitable for reproducible stream keys.

Prefer counter-keyed draws based on a recorded cryptographic digest of (episode seed, process kind, semantic entity key, occurrence). An optional standard-library uniform construction uses the top 52 bits k of a SHA-256 block and u=(k+0.5)/2^52, which lies strictly between zero and one in binary64. Specify the distribution transformation and its version. This use is deterministic random-stream organization, not a security claim.

Key potential execution randomness to stable job semantics—job kind, relevant fingerprint and retry occurrence—not a policy's global event count. An action that creates a different job creates a different semantic key; the same job under two policies receives the same draw. Predetermine persistent defects by content. Exogenous environmental events and arrival tapes must not change when one policy emits extra log events or decisions.

Policies must not receive the truth master seed, future tape, sampled completion time, hidden edges or oracle objective. Planner seeds are separate, with common scenarios across actions in the same comparison. Native adapters receive only the supported observation projection. Add a test where the future changes but the complete observable past does not.

Reset online estimator and policy state at each independent cluster, except state explicitly learned on development and frozen. If adaptation across episodes is a research objective, make the entire adaptive trajectory the independent unit and redesign the analysis before confirmation.

### 11.4 Required run and event records

The run manifest contains:

- Run ID; phase; parent/amendment ID; start/end UTC; planned and completed episode counts.
- Hypothesis, protocol/config/data/source/lock hashes; exact native queue revision and fidelity status.
- Machine/OS/interpreter/library versions; actual memory and storage; worker/thread settings.
- All seed namespaces; generator mixture; time/calendar/budget conventions; cost and value units.
- Baseline selector/training provenance, tuning budget, chosen settings and rejected alternatives.
- Primary and secondary estimands, intervals/error allocation, failure treatment and log-retention rules.
- Complete termination status, resource totals, warnings and paths/hashes of raw outputs.

Every event record contains episode ID, policy ID, event ID, simulated time, phase, kind, entity IDs, pre/post state digests, consumed resource increments and reason codes. Every decision contains the observable-state hash, available actions, selected action, forecast settings, outcome estimates, fallback reason, measured wall/CPU time and any stale-observation rejection.

Keep raw cost execution IDs, original/repeat flags and adjudicated cause separate from aggregated costs. Do not infer causes only from a final count. Store every rejected/expired/obsolete certificate with its provenance; pruning in-memory state does not erase audit history.

At minimum, each episode-policy result row contains:

| Field group | Required values and units |
|---|---|
| Identification | episode/cluster/policy IDs, graph family, protocol/hash, status |
| Delivery | offered weight, delivered weight at H, Q(H), capability IDs and functional-regression count |
| Attention/compute | original and repeated active seconds by role, runner seconds, setup/switch/cancel seconds; through H and drain separately |
| Delay | restricted weighted delay in seconds, pending weight/age, deadline misses including unfinished work |
| Validity | gate rejection counts, evaluator applicability violations, distinct functional defects and escaped defects |
| Drain | complete/censored/blocked status, completion time if known, remaining work/bounds and total cohort effort |
| Controller | wall/CPU seconds, decision count, peak RSS bytes, fallback count, evaluation transitions, index/storage bytes |
| Recovery | checkpoint IDs, restarts, interrupted transaction count and final state/event digest |

### 11.5 Failure handling and reproducibility

A declared timeout with the specified fallback is a method outcome and remains in the analysis. A policy crash is not silently dropped: record its last valid deployed state, pending work and consumed resources, and apply the predeclared recovery/unavailability rule while the clock continues. If the error instead invalidates the common simulator or measurement, retain the affected runs, repair the common defect and rerun all affected arms on the same workload under a recorded correction. Do not discard hard seeds.

Checkpoint state includes the next event ID/phase, exogenous cursor, in-flight jobs and captured inputs, consumed service, available budgets/calendars, evidence/revocation state, deployed vector, current decisions/forecasts, policy estimator state and metric accumulators. Use deterministic idempotency keys for completion and release. Recover at a whole event-batch boundary or replay the uncommitted batch from its start.

Required recovery test: run a small episode uninterrupted, then interrupt before and after each of (a) applying a version event, (b) completing/charging a task, and (c) committing a release. Resume and compare exact canonical digests and all costs/metrics. A deliberately changed config, source or lock must be rejected by default. Never overwrite an incompatible completed run.

A reproduction subset is selected before viewing results and includes a favorable, neutral and adversarial **fixture**, plus randomly selected confirmation episode IDs; do not select only the best observed episodes. Preserve raw events so the final metrics can also be recomputed without rerunning the scheduler.

### 11.6 Final result report

The reporting command must produce:

1. **Claim table:** H0–H4, evidence level, tested population, result, uncertainty and residual limitation.
2. **Primary paired comparison:** absolute Q differences, intervals, raw capability units, all named baselines, selector rule and native fidelity.
3. **Tradeoffs:** attention/CI/setup/controller costs, backlog/deadlines, complete-drain costs, applicability failures and functional outcomes.
4. **Attribution:** protected cost-state contrast, admission/order decomposition, partial reuse, ordinary-isolation controls and counterexamples.
5. **Resources:** actual laptop configuration, stage wall/CPU/RSS/storage, tuning cost, fallback frequency and attainable precision.
6. **Reproducibility:** source/lock/protocol/data hashes, exact commands, raw/derived mapping, failed runs, amendments and recovery results.
7. **Interpretation:** supported claim, unsupported larger claim, practical headroom status, and the next evidence needed.

For the secondary unbounded complete-cohort effort differences, choose and freeze a paired-cluster bootstrap interval (percentile by default) and its simultaneous error allocation before confirmation. Calibrate coverage on **nondegenerate known-zero-mean and known-shift paired distributions** with the intended skew/clustering, including rare large positive and negative differences. Identical policies with D=0 are a harness parity check, not a nondegenerate coverage test. Report binomial Monte Carlo uncertainty of estimated coverage. State the finite-moment and large-sample approximation assumptions. This interval does not inherit the finite-sample guarantee of the bounded Q interval. If calibration or width is inadequate, H2's effort conclusion remains unresolved. Report raw effort means/quantiles and tradeoffs regardless.

Use effect-versus-workload plots and cost/delay frontiers with uncertainty where informative. Plot all declared strata, including adverse ones. A plot of only winning settings or a ratio with a near-zero denominator is not an adequate result.

## 12. Outcome interpretation and evidence-triggered pivots

| Finding | Supported conclusion | Next action |
|---|---|---|
| Exact oracle/harness disagrees or resume changes costs | Measurement is invalid | Repair and revalidate; do not interpret comparative outcomes |
| Current tool already implements the same state/action/optimization rule | Methodological novelty is absent | Reclassify as replication, integration or benchmark; retain only a useful empirical question |
| H1 positive, H2 supported, declared guardrails satisfied | Conditional evidence-informed scheduling advantage in the tested law | Independent confirmation and prospective headroom measurement |
| H1 positive, H2 absent/inconclusive | Delivery improvement with another or unproven mechanism | Analyze batching/priority/forecasting; narrow the causal claim |
| Q is saturated but complete-cohort effort improves | A secondary efficiency finding | Report it; design a fresh study with effort as primary if worthwhile |
| Delivery gain accompanies material urgent-delay, attention or operating-cost regression | A tradeoff | Present the frontier; seek an externally justified operating point |
| Interval includes zero and a worthwhile positive effect | Inconclusive at achieved precision | Report local limits; choose any further study before collecting new independent data |
| Precise interval excludes an externally justified worthwhile benefit | Evidence against practical value in this regime | Prefer the strong simpler policy or narrow the target |
| Advantage disappears with fair baseline tuning/shared reuse | No demonstrated incremental benefit | Recommend the existing configuration; preserve the negative comparison |
| Gain depends on full human restarts or broad conservative scopes | Restricted model result | Test measured partial reuse/compositional contracts; do not generalize to teams |
| Hidden dependency omissions cause invalid releases | Applicability assumptions fail in that regime | Improve scope certification/unknown handling and reconfirm; efficiency cannot excuse the violation |
| Native strongest comparator cannot be run faithfully | Local adapted-policy comparison only | State competitive superiority to that tool remains unestablished |
| Synthetic success fails on real Git/token binding | Implementation/application mismatch | Fix the representation or narrow the computational claim |
| Local simulation succeeds but H4 remains unknown | No established market or actual attention benefit | Measure prospective headroom before product expansion |

A **small statistically detectable gain** is not automatically worth deploying. Show the expected delivered value and attention/maintenance/delay costs. If business weights or acceptable regressions are unknown, leave the adoption decision open and report the measurable frontier.

For deadline and resource noninferiority, derive acceptable margins from the release contract before confirmation. If no justified positive margin exists, use zero and accept that the result may remain a tradeoff or inconclusive. Do not widen margins after seeing outcomes. Hard resource caps, statistical noninferiority and absence of a detected difference are different claims.

Possible useful outcomes include a simple scoped-batching recommendation, a queue extension that consumes existing evidence records, a reusable benchmark with negative results, or a mathematical replication with clarified attribution. A standalone product requires additional evidence of sustained customer pain, acceptable integration cost and an advantage that existing platforms cannot cheaply reproduce.

No universal agent-count formula, universal A<1 rule, guarantee of human-review replacement, historical novelty theorem, or production safety certification follows from this program.

## 13. Audit record, unresolved risks and limits

Three distinct assistant-agent review tracks challenged scientific/prior-art claims, experimental validity, and implementation/feasibility. The draft was revised and subjected to targeted follow-up challenges. These are documented design reviews within this investigation, **not external peer review**, customer validation or executed production tests.

| Material weakness found | Revision in this handoff | Acceptance evidence / remaining gate |
|---|---|---|
| The opportunity looked more novel when compared only with basic queues | Added current SubmitQueue internals, ATCC, design rework/freeze and established rollout; narrowed contribution | Prior-art matrix and explicit novelty falsifiers; exact historical formula priority unresolved |
| Homogeneous fresh-at-end math could be confused with real checks | Separate theorem oracle from start-snapshot evidence binding | Supplied checks reproduced; independent Appendix A finite-state checks passed; production-mode fixtures still to implement |
| A stable release can shift work to a later release without saving it | Added the exact cross-release effort trap and complete offered-cohort drain | WP03/WP05 fixture and H2 full-cohort accounting required |
| Candidate IDs or global invalid flags could create false invalidation | Relevant projection keys and candidate-relative applicability | Retained-predecessor and unrelated-change fixtures |
| Releasing an old branch could erase another release or use obsolete requirements | Write/read masks, actual proposed-state requirement derivation and atomic gate | Disjoint-branch, newly composed requirement and token-race fixtures |
| “Necessary dependencies” and security obligations were underspecified | Explicit consumer-impact closure, persistent component labels and separate compositional-contract policy | Generator/schema validation and common gate within each stratum |
| Shared evidence costs and previous expenditure could be counted repeatedly | Charge execution bundles once and only future marginal replacement work | Cost conservation, duplicate callback and obsolete-certificate fixtures |
| Hidden finish times or future-demand normalization could leak information | Conditional residual forecasts, service/budget pause rules and expected per-cell normalization | Same-past/different-future fixture and observation-field diff |
| The proposed broad state-masking ablation was not operationally coherent | Precise substitution of future job-specific replacement costs; all applicability rules retained | Matched typed-input transform; H2 interpretation narrowed |
| Job-specific replacement costs could be unknowable or leaked | Explicit latent-cost lineage, observed completed effort, persistent/independent cost regimes and homogeneous null | Forecast information boundary and null parity; realistic persistence remains unmeasured |
| A short rollout horizon could prefer permanent idling to a long necessary review | Zero-new-delivery fallback and a common finite-cohort progress guard | Long-job fixture and baseline liveness check before confirmation |
| An event-only search was called exact without a completeness argument | Finite tick/action/horizon oracle, explicit restricted-search status and valid bounds | Full enumeration required before any exact-optimum statement |
| Repeated graph rows, adaptive learning or censored completions could distort inference | Cluster analysis, reset learned state, fixed protocol, complete H2 population and retained censoring | Null/known-difference analysis calibration; no complete-case selection |
| A conservative confidence bound could make the promised study far larger than expected | Explicit sample arithmetic and measured runtime/precision gate | Pilot-derived N and feasible local runtime, or an honest inconclusive result |
| Ideal and overhead-charged outcomes could be pooled under incompatible independence assumptions | Primary idealized estimand separated from operational timing-block confirmation | Frozen protocol and nondegenerate secondary-interval calibration |
| Native queue and proposed controller could have unequal admission authority or priors | Same queue/admission interface, common refresh adapter and calibrated success scorer | Native conformance and exact revision remain execution prerequisites |
| Maximum laptop RAM and optional packages were treated as assumptions | Actual hardware discovery, native-operation checks, streamed CPU core and faithful fallbacks | WP00 and measured pilot; no M3 results claimed |
| Recovery was requested only after the protected run | Preconfirmation crash smoke plus later independent reconstruction | Matching deterministic hashes and costs after interruption |

### Residual scientific and practical uncertainty

1. **Prevalence and value:** target-team avoidable human rework, partial reasoning reuse, annotation cost and willingness to adopt are unknown.
2. **Historical originality:** important older repair and design-freeze papers were accessible only in abstract or limited publisher form; no definitive novelty claim is justified.
3. **Comparator execution:** current public source was inspected, but an exact pinned native SubmitQueue experiment has not run. The public and published-private implementations must remain distinct.
4. **Dependency completeness:** a local declared graph can establish conditional applicability, not automatically discover every semantic dependency in a real system.
5. **Human behavior:** scripted attention, authoring tapes and fixed value weights omit adaptation, negotiation, fatigue, learning and shifting business priorities.
6. **Hardware and scale:** local throughput, RAM, long-run thermal effects and compiled integrations remain unmeasured. Large connected-system behavior cannot be inferred from disconnected replicas.
7. **Statistical scope:** the primary estimand concerns the declared synthetic law. Secondary unbounded effort inference has additional assumptions. No finite study establishes zero rare production risk.
8. **Production atomicity:** the SQLite gate proves only the modeled local registry transition. Real deployments require coordination covering actual target effects.

After the recorded corrections, the targeted follow-up reviews identified no further material design change within their reviewed scope. Remaining uncertainty is assigned to explicit work-package gates rather than converted into assumed success. The coding agent should preserve this audit record, append discovered defects and amendments, and stop optional rechecking once a concrete acceptance condition has been met; substantive new evidence warrants reopening the affected claim.



## 14. References and search coverage

All mutable pages were accessed during the investigation ending **30 October 2026**. “Full relevant text” means the cited sections were inspected, not every proof or implementation path. Source/documentation inspection is distinct from execution. Abstract-only and blocked-source limits are stated below. Several grouped IDs contain multiple primary URLs; they are not independent performance replications.

### 14.1 Mathematical and repair antecedents

**[R01]** Donald R. Smith. *Optimal Repair of a Series System*. Operations Research **26**(4), 653–662, **1978**. [DOI](https://doi.org/10.1287/opre.26.4.653). **Inspection:** publisher abstract; full proof unavailable in the session. Supports longstanding all-working repair/completion formulation, not a transferred software-scheduling theorem.

**[R02]** Michael N. Katehakis and Cyrus Derman. *Optimal Repair Allocation in a Series System*. Mathematics of Operations Research **9**(4), 615–623, **1984**. [DOI](https://doi.org/10.1287/moor.9.4.615). **Inspection:** publisher abstract, including availability-optimal allocation under exponential failure/repair and costless reassignment; full proof not inspected.

**[R03]** Esther Frostig. *Optimal Repair of a Series System with Fixed Repair Times*. Probability in Engineering and Informational Sciences **13**(4), 477–487, **1999**. [DOI](https://doi.org/10.1017/S0269964899134041). **Inspection:** publisher abstract; fixed nonpreemptive repair model is close, but priority-order interpretation/full proofs unresolved. Do not assert exact equivalence with the attached discrete timing convention.

**[R04]** B. Lachaud. *Cut-off and hitting times of a sample of Ornstein–Uhlenbeck processes and its average*. Journal of Applied Probability **42**(4), 1069–1080, **2005**. [DOI](https://doi.org/10.1239/jap/1134587817). Primary PDF: [PDF](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/CD293CFF472367F2527441B37FE67961/S002190020000111Xa.pdf/cutoff_and_hitting_times_of_a_sample_of_ornsteinuhlenbeck_processes_and_its_average.pdf). **Inspection:** full primary PDF, specifically Proposition 3 and the transformed-normal density. The primary PDF was reopened during the final source review.

**[R05]** C. Lancia, G. Guadagni, S. Ndreca and B. Scoppola. *Asymptotics for the late arrivals problem*. Mathematical Methods of Operations Research **88**, 475–493, **2018**. [DOI](https://doi.org/10.1007/s00186-018-0643-3). **Inspection:** full open text, thinning recursion and stationary q-products. Does not itself settle priority of the precise controlled first-passage formula.

**[R06]** E. Cruciani and A. Dudeja. *The Careless Coupon Collector's Problem*. FUN 2026, LIPIcs **366**, 14:1–14:21, **15 May 2026**. [DOI](https://doi.org/10.4230/LIPIcs.FUN.2026.14); [source](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FUN.2026.14). **Inspection:** primary PDF model and Section 4.1, exact finite-state solve and phase analysis.

**[R07]** Christopher D. Long. *Clumsy and Careless: Stationary-Entry Flux in Non-monotone Coupon Collectors*. arXiv:2605.14511v1, **14 May 2026**. [source](https://arxiv.org/html/2605.14511v1). **Inspection:** full relevant HTML theorem statements, entry-flux argument and Section 5 conventions. Preprint; no reviewed publication inferred.

### 14.2 Systems, scheduling and rework

**[R08]** Uber Technologies and contributors. *SubmitQueue* public repository and development/testing guides. **Mutable main, accessed 2026-10-30; exact revision unresolved.** [code](https://github.com/uber/submitqueue); [code](https://github.com/uber/submitqueue/blob/main/doc/howto/DEVELOPMENT.md); [code](https://github.com/uber/submitqueue/blob/main/doc/howto/TESTING.md); [code](https://github.com/uber/submitqueue/blob/main/go.mod). **Inspection:** README, guides and Go manifest. Apache-2.0 and Go 1.25.0 observed; local fake/Git paths documented, not run. No claim that current Go implementation reproduces the private Java deployment.

**[R09]** Uber Technologies and contributors. SubmitQueue *Speculation* RFC and speculation entity source. **Mutable main, accessed 2026-10-30.** [code](https://github.com/uber/submitqueue/blob/main/doc/rfc/submitqueue/speculation.md); [code](https://github.com/uber/submitqueue/blob/main/submitqueue/entity/speculation.go). **Inspection:** source/docs for path identities, assumptions, retained status, build/cancel actions and controller-owned verdict. Dependency relaxation marked unimplemented in the inspected RFC. No complete source audit.

**[R10]** Uber Technologies and contributors. SubmitQueue *Outcome Scorer* RFC and speculation allocator README. **Mutable main, accessed 2026-10-30.** [code](https://github.com/uber/submitqueue/blob/main/doc/rfc/submitqueue/outcome-scorer.md); [code](https://github.com/uber/submitqueue/blob/main/submitqueue/extension/speculation/allocator/README.md). **Inspection:** source/docs for configured scoring, sticky retention, terminal-result handling, cancellation capacity and optional weighted costs. Supported extension possibility is not evidence that a particular weighted allocator is enabled by default.

**[R11]** Dhruva Juloori, Zhongpeng Lin, Matthew Williams, Eddy Shin and Sonal Mahajan. *CI at Scale: Lean, Green, and Fast*. ICSE **2025**; arXiv:2501.03440, v1 **7 January 2025**, v2 **19 May 2025**. [source](https://arxiv.org/abs/2501.03440); [source](https://arxiv.org/html/2501.03440v2). **Inspection:** full relevant architecture, prediction/threshold, evaluation and limitations. Private Java service, pre/post deployment and modeled counterfactuals, not a local randomized comparison.

**[R12]** Mergify. *Queue modes*, *Batches*, *Scopes*, and *Direct merge*. **Live official docs, accessed 2026-10-30.** [source](https://docs.mergify.com/merge-queue/queue-modes/); [source](https://docs.mergify.com/merge-queue/batches/); [source](https://docs.mergify.com/merge-queue/scopes/); [source](https://docs.mergify.com/merge-queue/direct-merge/). **Inspection:** relevant full documentation for dependency lanes, scoped capacities, batching/failure splitting and conditional CI reuse. Hosted implementation not run; local version is an adaptation.

**[R13]** Zuul. *Project Gating* and pipeline configuration documentation. **Live official docs, accessed 2026-10-30.** [source](https://zuul-ci.org/docs/zuul/latest/gating.html); [source](https://zuul-ci.org/docs/zuul/latest/config/pipeline.html). **Inspection:** dependent/shared queues, speculative invalidation and window parameters. Documented initial window 20, add-one/halve default; exact pinned source/event rounding not audited.

**[R14]** GitHub. *Managing a merge queue*; GitLab. *Merge trains*; GitHub approval/ruleset documentation. **Live official docs, accessed 2026-10-30.** [documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue); [source](https://docs.gitlab.com/ci/pipelines/merge_trains/); [documentation](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/approving-a-pull-request-with-required-reviews); [documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets). **Inspection:** full relevant merge-queue/train docs. Approval-policy pages were discovery-only in the reference review; exact current dismissal rules require checking the selected repository policy; this plan uses only the general stale-approval concept.

**[R15]** Andrey Mokhov, Neil Mitchell and Simon Peyton Jones. *Build Systems à la Carte*. ICFP **2018**. [source](https://www.microsoft.com/en-us/research/publication/build-systems-la-carte/); [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/build-systems-final.pdf). **Inspection:** primary paper, especially Sections 4.2–4.4 and 5.3, verifying/constructive traces and scheduler/rebuilder separation. Do not conflate the inspected 2018 paper with its expanded journal version.

**[R16]** Matthew A. Hammer, Khoo Yit Phang, Michael Hicks and Jeffrey S. Foster. *Adapton: Composable, Demand-Driven Incremental Computation*. PLDI **2014**. [DOI](https://doi.org/10.1145/2594291.2594324); [PDF](https://matthewhammer.org/adapton/adapton-pldi2014.pdf); author project page [source](https://matthewhammer.org/adapton/). **Inspection:** relevant primary-paper sections and author project metadata. Supports incremental-computation provenance, not software specialist cost claims.

**[R17]** Pact contributors. *can-i-deploy*. **Live official docs, accessed 2026-10-30.** [source](https://docs.pact.io/pact_broker/can_i_deploy). **Inspection:** version matrix and compatibility/deployment workflow. Does not by itself establish atomic validation-to-deployment or evidence for noncontract properties.

**[R18]** Hyperwise LLC and contributors. *FreshCtx*, version **0.17.0**, released **28 September 2026**. [package](https://pypi.org/project/freshctx/); [code](https://github.com/Hyperwise-LLC/freshctx). **Inspection:** package metadata/files/license and repository README/public API, during this investigation. Pure Python, stated supported Python 3.10–3.13, Apache-2.0; not executed. Check-before-call is not target-side atomicity.

**[R19]** Yongjian Lyu, Yang Ren, Ruofei Lai and Wenting Liu. *From Version Conflicts to Decision Conflicts: Selective Revalidation for Long-Running AI Agents*. arXiv:2609.08015v1, **7 September 2026**. [source](https://arxiv.org/html/2609.08015v1); author artifact [code](https://github.com/ezreal13/atr-decision-validation). **Inspection:** full relevant model/propositions/limitations and artifact instructions during this investigation. Paper identifies artifact commit `89c32fa`, Python 3.11 + NumPy; artifact not executed in this research. Guarantees depend on complete dependencies, sound validators and enforced effect binding.

**[R20]** Weixing Zhou, Zhiyou Wang, Zeshun Peng, Hetian Chen, Yanfeng Zhang and Ge Yu. *ATCC: Adaptive Concurrency Control for Unforeseen Agentic Transactions*. arXiv:2603.13906v1, **14 March 2026**. [source](https://arxiv.org/html/2603.13906v1). **Inspection:** Sections 4–6, cost/priority, mode switching, locking, commit and evaluation. Reported large-server openGauss-MOT setup not reproduced; B7 is explicitly an adaptation.

**[R21]** Hongtao Lyu, Dingyan Zhang, Mingyu Wu, Xingda Wei and Haibo Chen. *CoAgent: Concurrency Control for Multi-Agent Systems*. arXiv:2606.15376v1, **13 June 2026**, [source](https://arxiv.org/html/2606.15376v1). Also Zheng Chen, Hanqing Liu, Duling Xu, Dong Dong, Jialin Li, Bangzheng Pu and Jidong Zhai. *Cordon: Semantic Transactions for Tool-Using LLM Agents*. arXiv:2606.17573v1, **16 June 2026**, [source](https://arxiv.org/html/2606.17573v1). **Inspection:** abstracts, introductions/design outlines and headings, not full evaluation/implementation audit. No venue acceptance inferred from manuscript headers.

**[R22]** Jakob F. Maier, David C. Wynn, Wieland Biedermann, Udo Lindemann and P. John Clarkson. *Simulating progressive iteration, rework and change propagation to prioritise design tasks*. Research in Engineering Design **25**, 283–307, **2014**; publisher online date **10 June 2014**. [DOI](https://doi.org/10.1007/s00163-014-0174-8); [source](https://link.springer.com/article/10.1007/s00163-014-0174-8). **Inspection:** full open article available; abstract, model/context and empirical scope scanned, not exhaustive model validation.

**[R23]** Jongwook Lim, Sora Kang, Jun Lee and Yoo Suk Hong. *An effective design freeze strategy for establishing design rules in modular product development*. Journal of Engineering Design **35**(4), 483–503, **2024**; published online **27 February 2024**. [DOI](https://doi.org/10.1080/09544828.2024.2321228). **Inspection:** primary publisher abstract and bibliographic fields; full opening blocked. Supports existence of dependency-aware grouped freeze alternatives, not independently verified performance.

**[R24]** Haitao Li and Norman K. Womer. *Solving stochastic resource-constrained project scheduling problems by closed-loop approximate dynamic programming*. European Journal of Operational Research **246**(1), 20–33, **2015**. [DOI](https://doi.org/10.1016/j.ejor.2015.04.015); [source](https://www.sciencedirect.com/science/article/abs/pii/S037722171500288X). **Inspection:** publisher abstract and metadata; full opening blocked. Establishes prior use of rollout/closed-loop ADP; its own guarantees cannot be transferred to capped sampled planning.

**[R25]** Guangdong University of Technology. *Random rework multi-task scheduling method, device, equipment and medium*. Patent publication **CN117933655B**, publication **10 October 2025**; listed priority **26 January 2024**, application publication **26 April 2024**. [source](https://patents.google.com/patent/CN117933655B/en). **Inspection:** translated abstract and detailed scheduling method; some equations absent from extracted translation. Publication identifier and described mechanism only; no legal status, claim-scope or patentability opinion. Inventor spelling omitted rather than guessing a transliteration.

### 14.3 Application evidence, batching and data

**[R26]** GitHub Staff. *Octoverse: Commits, pushes, reach new highs in 2026 as developers build with agents*. **28 October 2026**. [source](https://github.blog/news-insights/octoverse/octoverse-commits-pushes-reach-new-highs-in-2026-as-developers-build-with-agents/). **Inspection:** primary main report, workflow figures and comparisons during this investigation. Observational platform/detection metrics; no causal human invalidation effort.

**[R27]** Anthropic Engineering. *Redesigning source control at Anthropic*. **13 October 2026**, updated **15 October 2026**. [source](https://www.anthropic.com/research/redesigning-source-control). **Inspection:** primary growth, architecture, implications and limitations during this investigation. The extreme-workload caveat and post-merge/pre-deploy architecture support the draft's bounded claims.

**[R28]** DX. *New DX data finds 94% of AI token usage is re-reading existing context*. **14 October 2026**. [source](https://getdx.com/news/new-dx-data-finds-94-of-ai-token-usage-is-re-reading-existing-context/); associated report landing page [source](https://getdx.com/report/state-of-ai-impact-in-engineering-q3-report/). **Inspection:** vendor announcement and landing page during this investigation; no underlying complete methods/data audit. Avoid calling this a causal or independently verified estimate.

**[R29]** Hao He, Shyam Agarwal, Yegor Denisov-Blanch, Pavel Azaletskiy, Sanmi Koyejo and Bogdan Vasilescu. *AI Writes Faster Than Humans Can Review: A Longitudinal Study of an Enterprise 2× Mandate*. arXiv:2607.01904v2, submitted **2 July 2026**, revised **27 October 2026**. [source](https://arxiv.org/html/2607.01904v2); [DOI](https://doi.org/10.48550/arXiv.2607.01904). **Inspection:** abstract, full relevant methods/outcomes/supplement, threats and data availability during this investigation. Proprietary data, nonrandom adoption, changing review composition and missing file-level changes limit mechanism attribution.

**[R30]** GitHub. *Copilot code review: Merge with confidence with Max and approvals*. **28 October 2026**. [source](https://github.blog/changelog/2026-10-28-copilot-code-review-merge-with-confidence-with-max-and-approvals/). **Inspection:** official availability/capability announcement during this investigation. Product capability is not controlled efficacy evidence.

**[R31]** Joel Becker, Nate Rush, Tom Cunningham, David Rein and Khalid Mahamud. *We are Changing our Developer Productivity Experiment Design*. METR, **24 February 2026**. [source](https://metr.org/blog/2026-02-24-uplift-update/). **Inspection:** full relevant primary follow-up on selection, abandonment, measurement and design; reopened for exact metadata.

**[R32]** Emad Fallahzadeh, Amir Hossein Bavand and Peter C. Rigby. *Accelerating Continuous Integration with Parallel Batch Testing*. arXiv:2308.13129v1, **25 August 2023**. [source](https://arxiv.org/abs/2308.13129); associated DOI [DOI](https://doi.org/10.1145/3611643.3616255). **Inspection:** primary abstract and bibliographic metadata only. It explicitly names DynamicBatching and TestCaseBatching, including allowing new builds to join before full test execution. No full algorithm/artifact reproduction claimed.

**[R33]** Ali Sayedsalehi, Peter C. Rigby and Gregory Mierzwinski. *Risk-Aware Batch Testing for Performance Regression Detection*. arXiv:2604.00222v1, **31 March 2026**. [source](https://arxiv.org/html/2604.00222v1); [source](https://arxiv.org/abs/2604.00222). **Inspection:** relevant full methods/limitations and primary metadata. Performance-risk objective and selective coverage differ from fixed mandatory release evidence. Reported annual savings are a model calculation, not a laptop or customer forecast.

**[R34]** Florent Moriconi, Thomas Durieux, Jean-Rémy Falleri, Raphael Troncy and Aurélien Francillon. *GHALogs*, **version 1.0.1**. [DOI](https://doi.org/10.5281/zenodo.14796970); [source](https://zenodo.org/records/14796970); code/sample repository [code](https://github.com/D2KLab/gha-dataset). **Inspection:** primary version records, file metadata and README during this investigation. Exact publication date was not confirmed in the reference audit. Primary license identifier unresolved, full logs not downloaded, no human-review causal labels.

**[R35]** GH Archive contributors. *GH Archive*. **Live project documentation, accessed 2026-10-30.** [source](https://www.gharchive.org/). Supporting GitHub official REST documentation: [documentation](https://docs.github.com/en/rest/issues/timeline); [documentation](https://docs.github.com/en/rest/checks/runs); [documentation](https://docs.github.com/en/rest/actions/workflow-runs); [documentation](https://docs.github.com/en/rest/activity/events). **Inspection:** project description and relevant read-API fields/limits during this investigation. Historical data completeness, rights and semantic reconstruction unverified.

**[R36]** Rungroj Maipradit, Dong Wang, Patanamon Thongtanunam, Raula Gaikovina Kula, Yasutaka Kamei and Shane McIntosh. *Repeated Builds During Code Review: An Empirical Study of the OpenStack Community*. arXiv:2308.10078v1, **19 August 2023**. [source](https://arxiv.org/abs/2308.10078); [source](https://arxiv.org/html/2308.10078v1). **Inspection:** abstract and accessible full HTML during this investigation; unchanged-change-set rechecks differ from invalidation by new changes.

### 14.4 Hardware, inference and supporting background

**[R37]** Apple. *MacBook Pro (14-inch, M3 Pro or M3 Max, Nov 2023) — Technical Specifications*. [source](https://support.apple.com/en-us/117736). **Inspection:** CPU and memory specs directly opened during this investigation; a later retrieval returned HTTP 429. This is source access variation, not evidence reversal. Hardware model is November 2023; no unsupported page-update date. User's exact configuration remains unknown.

**[R38]** SciPy developers. *SciPy 1.18.1* package metadata, released **21 August 2026**, [package](https://pypi.org/project/scipy/1.18.1/); NumPy developers, *NumPy 2.0.0 Release Notes*, [source](https://numpy.org/doc/2.0/release/2.0.0-notes.html). **Inspection:** primary PyPI compatibility/wheels/license and official Accelerate release notes during this investigation. Specific environment installation untested; no event-simulation speedup inferred from matrix-kernel support.

**[R39]** Google OR-Tools contributors. *OR-Tools 9.15.6755* package metadata, released **14 January 2026**. [package](https://pypi.org/project/ortools/9.15.6755/). **Inspection:** primary Python requirements, native macOS ARM64 wheel and Apache-2.0 metadata during this investigation. Optional exact optimization, not an executed/proven optimal oracle.

**[R40]** Python Software Foundation. *Python 3.12 documentation: multiprocessing* and *sqlite3*. **Versioned docs, accessed 2026-10-30.** [source](https://docs.python.org/3.12/library/multiprocessing.html); [source](https://docs.python.org/3.12/library/sqlite3.html). **Inspection:** official documentation inspected for macOS spawn guidance, transaction-control conventions and backup API.

**[R41]** SQLite. *Online Backup API* and *Write-Ahead Logging*. **Official docs, accessed 2026-10-30.** [source](https://www.sqlite.org/backup.html); [source](https://www.sqlite.org/wal.html). **Inspection:** official documentation inspected for backup snapshot semantics and WAL limitations, including documented WAL reset correction fixed in 3.51.3. This does not establish crash safety of unwritten code.

**[R42]** Andreas Maurer and Massimiliano Pontil. *Empirical Bernstein Bounds and Sample Variance Penalization*. **2009**. [source](https://arxiv.org/abs/0907.3740); primary proceedings PDF [PDF](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf). **Inspection:** Theorem 4 and assumptions read during this investigation; bounded paired-difference interval is a stated algebraic rescaling.

**[R43]** Tim P. Morris, Ian R. White and Michael J. Crowther. *Using simulation studies to evaluate statistical methods*. Statistics in Medicine, **2019**. [DOI](https://doi.org/10.1002/sim.8086); full primary-author manuscript [source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/). **Inspection:** full relevant simulation-design/reporting and Monte Carlo uncertainty sections during this investigation.

**[R44]** M. C. Morais and A. Pacheco. *On Hitting Times for Markov Time Series of Counts with Applications to Quality Control*. REVSTAT **14**(4), 455–479, **2016**. [DOI](https://doi.org/10.57805/revstat.v14i4.198); primary PDF [PDF](https://www.ine.pt/revstat/pdf/rs160405.pdf). **Inspection:** institutional description and author-uploaded text discovered; primary PDF timed out and DOI could not be opened in final check. No full primary proof audit. Supplementary background; not relied on for the executable method.

**[R45]** Y. Ahmed, A. Ghosh, C.-C. Wang and N. B. Shroff. *Beyond Freshness and Semantics: A Coupon-Collector Framework for Effective Status Updates*. arXiv:2603.26998v1, **27 March 2026**. [source](https://arxiv.org/html/2603.26998v1). **Inspection:** primary full-text model and reward equation during this investigation. Expiration-aware average-reward communication MDP is adjacent, not a multi-item software release-completion result. Supplementary background.

**[R46]** Nimmi Rashinika Weeraddana, Mahmoud Alfadel and Shane McIntosh. *Dependency-Induced Waste in Continuous Integration: An Empirical Study of Unused Dependencies in the npm Ecosystem*. Proceedings of the ACM on Software Engineering **1**, FSE, Article 116, **July 2024**, 23 pages. [DOI](https://doi.org/10.1145/3660823); primary author PDF [PDF](https://rebels.cs.uwaterloo.ca/papers/fse2024_weeraddana.pdf). **Inspection:** relevant full text and author PDF/project metadata. Measures unused-dependency CI work, not specialist reapproval prevalence. Supplementary background.

**[R47]** Colima contributors. *Colima*. **Live repository/documentation, accessed 2026-10-30.** [code](https://github.com/abiosoft/colima). **Inspection:** official repository/runtime-support documentation; no exact release pin or installation executed. Do not infer every Compose image is native ARM64 merely because the runtime supports Apple Silicon.

**[R48]** Homebrew. *python@3.12 formula*, [source](https://formulae.brew.sh/formula/python@3.12); Python Software Foundation, *Python Releases for macOS*, [source](https://www.python.org/downloads/macos/). **Live official docs, accessed 2026-10-30. Inspection:** the investigation opened formula command/platform availability and official download index. Local OS/toolchain support remains a discovery gate, not a completed install. No patch version or unsupported OS availability invented here.


### 14.5 Search-coverage record

The search combined current primary documentation and source inspection, mathematical and software-engineering literature, and backward citation tracing. It covered these families:

| Search family | Representative terms and cross-checks | Consequence for the plan |
|---|---|---|
| Completion under decay | Perishable reviews, series-system repair, fixed repair times, thinning with immigration, expiring/careless coupons, hitting time, critical window, Ornstein–Uhlenbeck cutoff | Older repair and hitting-time antecedents limit broad mathematical novelty; exact benchmark retained |
| Merge scheduling and CI | SubmitQueue speculation/scorer/allocator, merge trains, scoped queues, adaptive batches/windows, test batching, discarded builds | Strong native/adapted comparators and shared reuse/retention required |
| Evidence and freshness | Build traces, incremental computation, contract version matrix, dependency freshness, selective revalidation, validation-to-effect binding | Ledger/gate concepts recognized as existing work |
| Protection of costly progress | Cost-aware concurrency control, abort cost, scoped locking, design maturity/rework, grouped design freeze | Simple reservation and stabilization alternatives added |
| Optimization method | Stochastic resource-constrained project scheduling, closed-loop approximate dynamic programming, rollout, random rework scheduling | Rollout classified as established, including a directly relevant patent disclosure |
| Application and contrary evidence | October 2026 agent-change volume, review latency/rounds, authoring productivity measurement, CI rechecks and human effort | Broad workflow pressure separated from unmeasured invalidation-driven attention |
| Local execution | Apple memory configurations, ARM64 package wheels, Python process/database behavior, native queue build/providers | Hardware discovery, exact dependency pins, faithful fallback and recovery gates |

Targeted follow-ups challenged apparent novelty and the strongest-baseline choice rather than only looking for supportive examples. They materially changed the recommendation: current SubmitQueue, cost-aware transactions, product-design rework and generic rollout are closer predecessors than basic stale-review protection.

The bounded patent search identified a described mechanism; it is not patent clearance or a legal assessment. This review does not establish exhaustive historical priority. Full proofs/methods for several older or publisher-restricted papers, a coherent pinned native queue checkout, proprietary operational data and the optional dataset's rights remain unresolved as indicated above. No target-Mac performance or customer causal effect was measured.

Further literature searching should be triggered by a concrete equivalence claim, newly available comparator or missing historical proof. The present evidence is sufficient to specify a fair conditional PoC without pretending the remaining gaps have been closed.

## Appendix A. Standalone exact benchmark and stable mean calculation

This reference was separately written and executed during the investigation. It needs only the Python standard library and no original attachment. Save the complete code block as appendix_math.py and run the existing command:

~~~bash
python3 appendix_math.py > appendix_results.json
~~~

The exact transition constructor enumerates all survival patterns and then adds one fresh completed review. It solves the transient mean and PGF linear systems using Fraction arithmetic and compares them with the finite formulas. The log-mean routine targets exactly q=exp(-a/n), avoids forming q, and uses O(n) arithmetic with O(1) additional storage. The matrix/bit-pattern checks are intentionally small and do not have that complexity.

Expected counts: 24 exact mean systems / 84 state values, 48 exact PGF systems / 168 state values, and 24 floating comparisons, all passing. Runtime and platform fields vary. The executed source SHA-256 is c60b5015bf1393b3c19d6f5a0768fa0e264fc280ed197b68710aec205ad7734a.

~~~python
#!/usr/bin/env python3
"""Standalone checks for fresh-at-completion, one-review-per-round model.
Standard library only. Run: python3 appendix_math.py
"""
from fractions import Fraction as F
from itertools import product
from math import comb, exp, expm1, isfinite, log, log1p
import json
import platform
import sys
import time


def solve(matrix, rhs):
    """Exact Gauss-Jordan elimination; no floating-point tolerances."""
    size = len(rhs)
    a = [[F(x) for x in row] + [F(y)] for row, y in zip(matrix, rhs)]
    for col in range(size):
        pivot = next(row for row in range(col, size) if a[row][col])
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [x / scale for x in a[col]]
        for row in range(size):
            if row != col:
                scale = a[row][col]
                a[row] = [x - scale*y for x, y in zip(a[row], a[col])]
    return [row[-1] for row in a]


def transition(n, q):
    """Enumerate survival bit patterns, independently of mean/PGF formulas."""
    rows = []
    for current in range(n):
        row = [F(0)] * (n + 1)
        for bits in product((0, 1), repeat=current):
            probability = F(1)
            for bit in bits:
                probability *= q if bit else 1 - q
            row[1 + sum(bits)] += probability  # new review is fresh at end
        assert sum(row) == 1
        rows.append(row)
    return rows


def potential(k, q):
    coefficient, total = F(1), F(0)
    for r in range(1, k + 1):
        total += comb(k, r) * coefficient
        coefficient *= q**(-r) - 1
    return total


def eigenfunction(k, q, lam):
    coefficient, total = F(1), F(1)
    for r in range(1, k + 1):
        coefficient *= lam * q**(-(r - 1)) - 1
        total += comb(k, r) * coefficient
    return total


def log_mean(n, a):
    """log E_0[T_n] for exactly q=exp(-a/n); O(n) time, O(1) storage.
    n >= 1; finite a >= 0. Avoids forming q, which can round to 1 or 0.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    a = float(a)
    if not isfinite(a) or a < 0:
        raise ValueError("a must be finite and nonnegative")
    if n == 1:
        return 0.0
    term = total = log(n)
    if a == 0:
        return total
    for r in range(1, n):
        log_x = log(a) + log(r) - log(n)
        if log_x < -36:
            log_factor = log_x  # log(expm1(x))=log(x)+O(x), below roundoff
        else:
            x = a * (r / n)
            log_factor = log(expm1(x)) if x < 1 else x + log1p(-exp(-x))
        term += log(n - r) - log(r + 1) + log_factor
        if not isfinite(term):
            raise OverflowError("log mean exceeds this floating-point range")
        hi, lo = max(total, term), min(total, term)
        total = hi + log1p(exp(lo - hi))
    return total


def main():
    if not __debug__:
        raise RuntimeError("Run without -O; assertions are required for checks")
    started = time.perf_counter()
    mean_systems = mean_values = pgf_systems = pgf_values = 0
    log_errors = []
    for n in range(1, 7):
        for q in (F(1, 3), F(1, 2), F(9, 10), F(1)):
            p = transition(n, q)
            direct = solve([[F(i == j) - p[i][j] for j in range(n)]
                            for i in range(n)], [F(1)] * n)
            expected = [potential(n, q) - potential(k, q) for k in range(n)]
            assert direct == expected, ("mean", n, q)
            mean_systems += 1
            mean_values += n
            err = abs(log_mean(n, -n * log(float(q))) - log(float(expected[0])))
            assert err <= 2e-13, ("float log mean", n, q, err)
            log_errors.append(err)
            for s in (F(1, 2), F(3, 4)):
                direct = solve([[F(i == j) - s*p[i][j] for j in range(n)]
                                for i in range(n)], [s*p[i][n] for i in range(n)])
                denominator = eigenfunction(n, q, 1 / s)
                expected = [eigenfunction(k, q, 1 / s) / denominator
                            for k in range(n)]
                assert direct == expected, ("PGF", n, q, s)
                pgf_systems += 1
                pgf_values += n
    assert all(log_mean(1, a) == 0 for a in (0, .5, 1, 1000, 1e308))
    assert log_mean(32, 0) == log(32)
    assert log_mean(32, 5e-324) == log(32)
    assert abs(log_mean(32, 1e-10) - log(32) - 31e-10/64) < 2e-14
    assert abs(log_mean(2, 2000) - 1000) < 2e-13  # mean=1+exp(a/2)
    for bad_n, bad_a in ((0, 1), (True, 1), (2, -1), (2, float("inf"))):
        try:
            log_mean(bad_n, bad_a)
        except ValueError:
            pass
        else:
            raise AssertionError(("accepted invalid parameters", bad_n, bad_a))
    monotone = [log_mean(32, a) for a in (0, 5e-324, .1, .5, 1, 2, 10, 1000)]
    assert monotone == sorted(monotone)
    critical = [{"n": n, "centered_mean": exp(log_mean(n, 1)-log(n))-.5*log(n)}
                for n in (1000, 10000, 100000)]
    print(json.dumps({
        "all_checks_passed": True, "mean_systems": mean_systems,
        "mean_state_values": mean_values, "pgf_systems": pgf_systems,
        "pgf_state_values": pgf_values, "float_comparisons": len(log_errors),
        "max_absolute_log_error": max(log_errors), "critical": critical,
        "wall_seconds": time.perf_counter()-started,
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "timing_scope": "main checks, excluding imports"
    }, indent=2))


if __name__ == "__main__":
    main()
~~~

Run without Python's optimization flag because the checks use assertions. Exact identities have no floating tolerance; the log-mean comparisons use an absolute tolerance of 2e-13 in log expected rounds. For very small x, the stable routine replaces log(expm1(x)) with log(x) when the omitted correction is below ordinary floating resolution. This does not guarantee arbitrary precision for arbitrarily long accumulations.

The routine excludes n=0 because the empty-workload mean is zero and its logarithm is not finite; handle empty workloads separately. Do not exponentiate enormous returned log means. If the logarithm itself exceeds floating range, the code raises an error.

This is an oracle for the homogeneous **fresh-at-completion** model only. It establishes finite-case agreement, not historical originality, all asymptotic proofs, real-world review dynamics, or an optimal production scheduler.
