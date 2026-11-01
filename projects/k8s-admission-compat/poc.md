# Execution brief: an applicability and added-value audit for Kubernetes mutation conflicts

**For:** an autonomous coding agent configured as `gpt-6.1-sol`, reasoning `xhigh`  
**Research and source-access date:** 1 November 2026  
**Execution target:** one locally available MacBook with an M3 Max; inspect its actual resources first  
**Deliverable from you:** the completed, reproducible diagnostic study and its report, following the gates below. This document is the plan, not evidence that the Kubernetes PoC has already run.

## 1. Project decision and smallest defensible contribution

### 1.1 Execute this project

Implement a **small theorem-to-admission applicability and added-value audit**, using three actual webhook boundaries: Istio sidecar injection, Gatekeeper resource defaulting, and Kyverno security defaulting. Examine one fixed security-policy change: moving a missing seccomp default from Pod scope to each regular container while preserving the required effective security setting. Keep the component binaries and other policies fixed.

The primary question is:

> For exactly three independently invocable mutators, can the supplied bounded reachable-defect mechanism provide additional conflict observations or use fewer mutator evaluations than a cached test of all six orders, once both retain intermediate objects and receive the same admission evidence?

**Planning has already found a structural negative answer to that narrow advantage claim.** For three operations, the theorem's checks and the complete six-permutation prefix tree contain exactly the same 15 transition positions. Verify this derivation and its implementation independently; do not organize a speedup contest around a weaker baseline. This is a limitation of the proposed application at three components, not a refutation of the mathematical theorem or a result about larger stacks.

Use a bounded real-component diagnostic to determine how much of the abstract model survives real callback semantics, and whether its warnings have a final admission consequence. That diagnostic is part of the same audit. It must not become a search for a favorable product bug.

The smallest defensible contribution is:

1. A verifiable explanation of why the proposed mechanism supplies no additional first-pass observations over a strong conventional baseline at the prescribed three-component scale.
2. A reproducible account of assumption eligibility, reachable differences, final contract outcomes, and attribution limits for one realistic, pinned stack and one declared policy change.
3. If a consequential interaction actually occurs, a complete local reproduction identifying the workload, configuration revision, applicable trace, responsible mutations, and final violated predicate. Otherwise, a useful bounded negative or inconclusive result.

The first item is a derivation made during preparation of this brief, not a novelty claim made in the supplied paper. Configuration determinacy, preservation, commutativity-based testing, and trace diagnosis have substantial prior art [R1–R3]. Do not describe this project as the invention of policy testing, a new configuration-testing foundation, or a globally novel theorem.

### 1.2 Why this revises the original proposal

The original practical objective is retained: improve the evidence available before changing an interacting admission stack. The proposed efficiency or coverage advantage is narrowed because a fair, inexpensive baseline obtains the same observations. A theorem can provide a useful interpretation of those observations; it cannot receive exclusive credit for observations that the baseline already has.

This project does **not** attempt a general faster detector, a human diagnostic-usability study, policy synthesis, safe repair, prevalence estimation, or a broad product comparison. It does not establish safety for an organization's unspecified production stack. The fixed versions below form a historical, reproducible compatibility fixture; conclusions do not automatically extend to newer releases.

### 1.3 Falsifiable quantities and decision thresholds

For each fixed initial object and environment, let `Q_A` and `Q_B` be the sets of unique deterministic transition queries required by the theorem checker and the cached all-order baseline. A transition key includes component/configuration identity, context, and the complete input object. Let `W_A` and `W_B` be their six bounded pair/context comparison records, including differences and incomplete branches, under identical equality rules. These are not an enumeration of all reachable defective states or all causal pairs.

The proposed incremental-observation claim would require `Q_A` to supply a relevant observation unavailable from `Q_B`, or fewer queries for the same evidence. The exact result to check is instead:

~~~text
prefixes(theorem branches) = prefixes(all six permutations)
depth counts = {1: 3, 2: 6, 3: 6}; total = 15
W_A = witnesses reconstructed from B's intermediate objects
~~~

This equality is the primary exact outcome. A discrepancy in code is an implementation, cache, or information-access problem until explained; it is not an empirical discovery overturning the set identity. The number 15 excludes eligibility certification, failures, reinvocation, built-ins, setup, and validation, all of which must be accounted for separately.

For the platform decision, the meaningful threshold is **one reproducible, applicable, newly introduced contract violation**. It justifies blocking or investigating this candidate for the affected fixture. No percentage of failing synthetic states is required. Conversely, passing the finite fixture census is only a scoped pass.

A positive claim about reduced human effort would require an independently designed human evaluation, which is outside this project. Witness length, number of JSON fields, or polished output is not a surrogate proof of engineering time saved.

### 1.4 Fixed scope and exclusions

Include Pod `CREATE`, one local Kubernetes version, one frozen environment per policy revision, regular containers in `spec.containers`, and the observed admission outcome. Preserve all other fields, including injected init containers, in states and traces. Init-container security and runtime functionality are outside the contract; they are not discarded from equality.

Exclude updates, deletions as experimental operations, reconciliation, rollout controllers, network-fault injection, container runtime behavior, throughput under production load, arbitrary external data, random/time-dependent policies, native-sidecar mode, additional mutator products, and `k > 3` scaling experiments. Cleanup deletes only the study's own local resources. Application containers must not be scheduled or run.

Run authorized stages autonomously. Resolve ordinary implementation issues within the repair allowance. Do not ask for permission to carry out the already specified local study. If the Mac, local virtualization, required permissions, source material, or images are unavailable, finish the unaffected analytic work and report the exact blocked portion without claiming it ran.

## 2. Evidence available now, and what remains unperformed

### 2.1 Supplied material examined

The preparation audit read the complete mathematical report, its LaTeX source, all investigation notes, the separate algebra audit, verifier programs, and recorded outputs. Both supplied PDFs are 13 pages and have **identical complete extracted text**, although their file bytes differ.

| Input | Location in the supplied material | Role/status |
|---|---|---|
| `research_report(1).pdf` | Standalone attachment; pp. 3–10 are the relevant main paper | Mathematical source [A1] |
| `research_bundle(1).zip` | Archive attachment | Source and verification record [A2] |
| `research_report.pdf`, `research_report.tex` | Inside `research_bundle/` | Same report text; inspect source for ambiguous notation |
| `verify_retractions.py` | Inside `research_bundle/` | Exact finite checks of pair repair and amplified obstruction |
| `verify_defect_reachability.py` | Inside `research_bundle/` | Exact finite witness/repair checks and sharp examples |
| `verify_binary_cycles.py` and its JSON/output | Inside `research_bundle/` | Independent appendix; not the selected mechanism |
| Algebra, binary, combinatorial investigation notes; `independent_algebra_audit.txt`; README and captured outputs | Inside `research_bundle/` | Context, superseded conjectures, limitations; not independent published validation |
| `Pasted text(20261101-135652).txt` | Task attachment | Original problem and research requirements |

Supply this brief and preferably the original PDF and ZIP to the coding agent. Store received originals unchanged in `inputs/` and record SHA-256 hashes. The mechanism, proofs needed for this project, controls, and protocol are specified here so that a missing auxiliary script need not block independent implementation. If the ZIP is missing, report that the supplied verifier rerun could not be reproduced; do not fabricate its contents or silently replace the mathematical source.

Original PDF SHA-256 values recorded during preparation:

~~~text
research_report(1).pdf:
b46eb1fcfab5b594cef0b7116b8783925b3d362c6d24934010b5de2e005167ff
research_bundle/research_report.pdf:
8492db1fff82439c6f8da6d544c0ca92885a9ea8c01ff980324bdde9927282fb
~~~

### 2.2 Work actually performed during preparation

An independent mathematical subagent reran the three supplied verifiers in the hosted preparation environment. All completed successfully:

| Check | Coverage actually checked |
|---|---|
| Pair repair | 40,207 ordered idempotent pairs, state-set sizes 1–5 |
| Amplified obstruction and explicit repair | `m = 0, 1, 2, 9, 100, 1000`; all 216 one-entry alterations of the nine-state core rejected |
| Defect reachability | 69,949 ordered idempotent triples, state-set sizes 1–4 |
| Additional reachability checks | 40,000 seeded tuples in the four settings provided by the source |
| Sharp radius constructions | `k = 2,...,8`, shortest distances `k−2` |
| Unused binary appendix | 312,830 exact contingency tables and rational odd-cycle examples through length 19 |

Two independent review tracks derived the 15-position equivalence; one also checked the prefix sets computationally. The per-pair and masking counterexamples in §3 were checked during preparation. Critical equations and the proof on PDF pp. 6–8 were inspected visually. No central proof defect was found; this is mathematical review plus finite checking, not formal verification or external peer review.

Official documentation, selected pinned source files, install manifests, and image-index metadata were inspected. Read-only OCI manifest probes found ARM64 entries for the selected main controller images. **No Kubernetes cluster, webhook PoC, Mac benchmark, full normalizer build, or production experiment was run.** The timings and machine budgets below are planning choices and estimates, not measured performance.

### 2.3 Evidence labels to use throughout execution

Use `SOURCE_RESULT`, `PREPARATION_DERIVATION`, `MEASURED`, `ASSUMPTION`, `ESTIMATE`, and `BLOCKED` where they prevent ambiguity. Distinguish a mathematical finite-table fact from a claim about the live service implementing that table. Repeated service responses support determinism; they do not prove it for all future requests.

## 3. Mathematics and the exact executable mechanism

### 3.1 Objects and assumptions

The source considers maps `f_i : X → X`, `i = 1,...,k`, on a common set. They are total, deterministic, and exactly idempotent:

~~~text
f_i(f_i(x)) = f_i(x), for every x in X.
f_i f_j(x) means f_i(f_j(x)): composition acts right to left.
B_ij = {x : f_i(f_j(x)) != f_j(f_i(x))}
B = union over i<j of B_ij
D = {x : some finite composition, including the empty one, sends x into B}
~~~

For `S = {s_1 < ... < s_r}`, let `w_S = f_s1 ... f_sr`, and `w_empty = identity`. Theorem 4.1 [A1, pp. 6–7] proves:

~~~text
D = union over i<j, and S subset of {1,...,k}\{i,j}, of w_S^(-1)(B_ij).
~~~

Thus membership in the set of states from which **some** pairwise defect is reachable has a witness involving at most `k−2` distinct other generators, in canonical order. The theorem does not enumerate every defective reachable state or guarantee that a particular chosen pair has such a short witness.

The reachability statement does not need a finite `X`. Finite `X` is needed for the source's normalized state counts and for exhaustive transition-table certification. There is no vector dimension, floating-point tolerance, convergence iteration, matrix solver, or numerical conditioning parameter in this mechanism.

### 3.2 Essential proof, sufficient to audit the implementation

Take a shortest word of length `r` sending `x` into `B`. Every shorter word sends `x` outside **all** pairwise defect sets. Adjacent letters in any word of length `r` can therefore be exchanged at `x`: the suffix before that exchange has length at most `r−2` and reaches a state where the pair commutes. A repeated letter could be moved next to its other occurrence and removed by idempotence, contradicting minimality. The shortest word is consequently square-free and can be sorted.

If generator `a` occurs in that word, reorder it so the reached state is `y = f_a(z)`, with `z` reached in `r−1` steps. At `z`, all pairs commute, and idempotence gives, for every `j`:

~~~text
f_a f_j(y) = f_a f_j f_a(z)
           = f_a f_a f_j(z)
           = f_a f_j(z)
           = f_j f_a(z) = f_j(y) = f_j f_a(y).
~~~

No failing pair at `y` contains `a`. The failing pair therefore excludes every letter in the word, leaving at most `k−2` letters. This establishes the formula, including the empty-word case. Audit composition direction carefully when translating this into chronological execution paths.

### 3.3 Specialization to three callbacks and the strong baseline

Name the three selected operations `I` (injector), `R` (resources), and `S` (security). For each unordered pair, compare both execution orders at the root and after the remaining operation. In the following table, strings describe **chronological execution**, left to right.

| Pair | Root comparison | Comparison after the third operation |
|---|---|---|
| `I,R` | `IR` versus `RI` | `SIR` versus `SRI` |
| `I,S` | `IS` versus `SI` | `RIS` versus `RSI` |
| `R,S` | `RS` versus `SR` | `IRS` versus `ISR` |

The right-hand column contains every full permutation. Every length-two branch in the middle column is a prefix of one of those permutations. Taking all prefixes gives three length-one, six length-two, and six length-three paths. A cached six-permutation traversal already supplies the complete theorem input and all intermediate witnesses. The naive uncached theorem expression count costs 30 callback evaluations at `k=3`; fair prefix reuse reduces it to the same 15 positions as the baseline. Further merging of equal full states benefits both equally.

This is an equality of query sets, not an adversarial lower bound saying every decision procedure always needs 15 queries. Early stopping and pre-known identities may reduce work for either method. Use full evaluation for the exact comparison; if an optional early-stop demonstration is retained, give both methods identical stopping information.

For arbitrary fixed `k`, the source uses `choose(k,2) × 2^(k−2)` comparisons and at most `(k+2) × choose(k,2) × 2^(k−2)` naive evaluations [A1, §4.3, p. 8]. It is exponential in `k`; “constant-query” means fixed `k` and a promised algebraic input. Do not extrapolate the three-component equivalence into a result for every `k`.

### 3.4 Other source results: what they do and do not authorize

| Source result | Exact content relevant here | Permitted use |
|---|---|---|
| Lemma 2.1 and equation (1), pp. 3–4 | Two idempotents have an explicit commuting repair changing one map only on its commutator defect set; coefficient one is sharp | Explain why a pair theorem does not supply a three-product repair |
| Theorem 3.1, pp. 5–6 | On `9+6m` states each pair has one defect, but minimum total changed table entries for a commuting idempotent triple is `m+2` | Explain failure of universal small Hamming repair for triples |
| Corollary 4.2, p. 7 | Keep each map on `X\D`, make every map identity on `D`; the repaired maps commute and are idempotent | Mathematical implication only; never deploy this repair |
| Equation (2), p. 7 | If every generator fiber has size ≤`M`, then `|D|/|X| ≤ (1+M)^(k−2) × sum_(i<j) d_H(f_i f_j,f_j f_i)` | No application estimate without a justified finite domain and fiber bound |
| §4.3, p. 8 | With uniform independent finite-state samples and exact idempotence, an epsilon-far tuple is missed with probability ≤`(1−epsilon)^s` | Do not turn fixture frequencies into production probabilities |
| Appendix A, pp. 10–13 | Separate biased Bernoulli correlation-cycle result; even-cycle optimum remains conjectural | Exclude from this project |

Here `d_H(f,g)` is the fraction of **uniformly counted arguments in the specified finite set** on which function values differ. The triple construction's sum of three Hamming distances tends to `1/6`; the average over all three tables tends to `1/18`. Neither number estimates rollout risk, revenue, repair effort, or workload prevalence. The identity repair may remove the required sidecar and all required defaults. No policy-preserving repair is implied.

### 3.5 Mandatory tiny controls

Implement these independently of service adapters. Function-table tuples list outputs at states `0,1,...`.

1. **Pair repair sanity:** `f=(0,0,2)`, `g=(0,2,2)`. Their defect set is `{1}`. Reproduce equation (1)'s single-entry repair if you retain the optional repair sanity check; do not build a repair optimizer.
2. **Context-only positive control:** use the nine-state table below with seed `y`. All root pair checks pass, but `f(y)=a` exposes a `g,h` defect; full permutations reach both `p` and `q`. The direct-pair ablation must miss the root while the theorem and full-trace baseline detect it.
3. **Masking control:** `f=(0,0,2)`, `g=(0,2,2)`, `h=(0,0,0)`, seed `1`. The pair differs, but every full permutation ends at `0`. Under contract “final state is 0,” report a warning and a passing final outcome.
4. **Global-versus-per-pair control:** `f=(0,0,2)`, `g=(0,1,1)`, `h=(2,1,2)`. All are idempotent. `B_fg={2}` and `B_fg ∪ h^(-1)(B_fg)={0,2}`, but seed `1` reaches `B_fg` via `h(f(1))=2`. The global formula catches another pair. Do not implement the false per-pair completeness claim.
5. **Non-idempotent exclusion:** a three-cycle `f=(1,2,0)` with two identity maps. Eligibility must fail even if pair checks pass.
6. **Commuting but invalid:** three identity maps and a contract false at the seed. The algebra passes; the contract fails. Absence of commutation defects is not satisfaction of requirements.
7. **Outcome handling:** test explicit rejection, unsupported patch type, invalid JSON patch, callback exception, and out-of-bound output. None becomes an identity transition.
8. **Representation:** object-key permutations compare equal; list permutations, missing versus null, differing annotations, and resource strings compare different unless the actual pinned decoder has already normalized them.
9. **Dynamic applicability:** on a tiny JSON object, let one synthetic operation set label `enabled=true` and another add a field only when that label is true. Assert that each step re-evaluates the current label. Also test an unsupported guard: it must not become a skip. This tests the harness; it does not add a fourth real product or change the real policies.

| State | `f` | `g` | `h` |
|---|---|---|---|
| a | a | d | e |
| d | d | d | q |
| e | e | p | e |
| p | p | p | p |
| q | q | q | q |
| y | a | d | Y |
| z | a | Z | e |
| Y | e | q | Y |
| Z | d | Z | p |

These are mathematical controls, not Kubernetes failures. The report must keep that distinction explicit.

### 3.6 Avoid hidden guarantees and unbounded closure

Passing repeat-application checks at raw workloads is insufficient: idempotence is required at every state in the modeled domain. Establish it exhaustively on a completed, finite closed transition table, or make the theorem conclusion explicitly conditional on an independently justified domain-wide proof. This project chooses finite-table certification and bounded failure behavior, not a speculative source proof of all three products.

Individual idempotence does not bound the number of reachable states. For example, on `(n,b)` with `n` a nonnegative integer and `b` in `{0,1}`, let `f(n,b)=(n,0)`, `g(n,0)=(n+1,1)`, and `g(n,1)=(n,1)`. Both maps are idempotent, but alternating them yields an infinite orbit. Therefore a closure cap is a real gate, not an implementation convenience. Reaching it means incomplete coverage; it does not permit truncation followed by a certificate.

## 4. Real stack, fixed policy change, and workload contract

### 4.1 Pinned historical fixture

Use the following exact application versions. They were selected because official source and manifests are accessible and the Kubernetes compatibility ranges overlap, not because they are the latest releases. Istio 1.27's published support interval includes Kubernetes 1.33; Kyverno 1.15's interval includes 1.33. Istio 1.27 is no longer a supported current release as of the research date [K5, K6]. End-to-end behavior of this exact three-product combination still needs validation.

| Component | Fixed version and boundary | Acquisition |
|---|---|---|
| Kubernetes | `v1.33.4`, one kind control-plane node | kind image below; source tag `v1.33.4` |
| kind | `v0.30.0`, native ARM64 | Official release [K1] |
| Istio | `1.27.3`, complete applicable injector callback | Official release, minimal profile, one istiod, no gateways or CNI; ordinary container sidecar mode |
| Gatekeeper | `v3.20.1`, complete mutation callback `/v1/mutate` | Official tagged deploy manifest, mutation enabled, one manager |
| Kyverno | `v1.15.2`, complete applicable resource mutation callback | Official tagged install manifest, one admission controller |

Use this node image exactly [K1]:

~~~text
kindest/node:v1.33.4@sha256:25a6018e48dfcaee478f4a59af81157a437f15e6e140bf103f85a2e7cd0cbbf2
~~~

Read-only preparation probes observed ARM64 entries at these controller image indexes. Recheck on the Mac, resolve child platform digests, and record every image actually used, including init containers. An index digest does not by itself establish that every deployment dependency works on ARM64.

~~~text
docker.io/istio/pilot:1.27.3
  sha256:0d1103581b970b5eaaa8aef977779d569a153ebf18b541a88281efdb5a70652c
docker.io/openpolicyagent/gatekeeper:v3.20.1
  sha256:6fda2898bdbdb9d446ce71b1f6bfe0efad87076025989642b5d151c17fd9a77c
reg.kyverno.io/kyverno/kyverno:v1.15.2
  sha256:16e710776afbb45ed4d40227c94a2e923c9f32209bfd5edfa75181d603932879
reg.kyverno.io/kyverno/kyvernopre:v1.15.2
  sha256:115c28b061d2668aa22b85cefb84771bbecfe72024fc81860793ebd121bb136a
~~~

Do not silently upgrade products to rescue a failed study. A missing pinned artifact permits one alternate official mirror/source retrieval, not a different release. If unavailable, complete the analytic result and classify the live portion as blocked.

### 4.2 Local environment and deployment shape

Create an isolated cluster named `admission-poc`; use an explicit kubeconfig and check the context before every mutating command. Use an existing native local Docker-compatible runtime or free local Colima. No remote cluster, remote Docker endpoint, paid compute, CUDA, or x86 emulation is allowed for the PoC. Normal source/image downloads are permitted.

Use study namespaces `poc-eval` and `poc-dev`, both with Istio injection enabled. Exclude controller namespaces from study policies. Scope R and S to these namespaces and label `compat.poc/enabled: "true"`. Disable Kyverno policy autogeneration and background mutation; use only Pod policies. Do not configure image verification, external context providers, inventory-dependent rules, generated resources, or external-data mutation. Disable optional audit/report/background/cleanup workloads if supported by the pinned manifests, while retaining any controller needed for readiness or webhook configuration. Record the actual configuration rather than assuming a disabled process is irrelevant.

No LimitRange, ResourceQuota, RuntimeClass, custom validator, or additional workload-mutating policy is added. Record all enabled built-in admission plugins and all registered matching webhooks. Keep ordinary built-in behavior visible; do not disable it to make replay appear faithful. Set the evaluation namespaces' Pod Security Admission enforcement to a declared profile compatible with the ordinary Istio init container, such as `privileged`, and record this deliberate limitation. This study checks its own regular-container contract and does not assert compliance with the entire restricted Pod Security Standard.

Prepopulate `serviceAccountName: default`, `automountServiceAccountToken: false`, `enableServiceLinks: false`, and the usual explicit not-ready/unreachable `NoExecute` tolerations with 300 seconds in fixtures, to reduce irrelevant built-in changes. Do not assume this removes every built-in effect. Give every fixture `schedulerName: poc-no-scheduler`; install no scheduler with that name. Local real-create confirmations can then inspect admission and storage without running the application containers. Verify Pods remain unscheduled.

### 4.3 Three actual boundaries

One mathematical operation is one whole externally callable component, including all its internal mutation rules and guards. The four Gatekeeper assignments below form **one R operation**. The Kyverno rules form **one S operation**. Never reorder or split internal rules into extra external generators.

Inspect registrations to identify which concrete webhook entry matches each fixture. Istio may have multiple mutually exclusive registration entries; a fixed workload must not silently call multiple injector entries and still be described as one invocation. If additional independent Pod mutators cannot be eliminated from this isolated fixture, stop the three-boundary operational claim and report the mismatch.

The selected pins' source defaults are `Never` for Istio, omitted/therefore `Never` for Gatekeeper, and `IfNeeded` for Kyverno's resource mutator. Gatekeeper's installation uses fail-open mutation behavior. Preserve and record the actual registered policies; source defaults are not a substitute for inspection [K2–K4, K7–K9]. Errors remain explicit even if the API server subsequently ignores a call error.

### 4.4 Resource policy R, unchanged in both revisions

Create four Gatekeeper `Assign` objects, with current pinned CRD schema validation. Each applies to group `""`, version `v1`, kind `Pod`, named study namespaces, and the enabled label. For every regular container selected by `name:*`, assign a missing leaf only:

| Location suffix under `spec.containers[name:*]` | Default |
|---|---|
| `.resources.requests.cpu` | `100m` |
| `.resources.requests.memory` | `128Mi` |
| `.resources.limits.cpu` | `1` |
| `.resources.limits.memory` | `512Mi` |

Use `parameters.pathTests` with `MustExist` at `spec.containers[name:*]` and `MustNotExist` at the same full leaf path as the assignment, plus `parameters.assign.value` with the listed string. Do not create new container entries or overwrite an existing value. Preserve missing-parent behavior as implemented by Gatekeeper; test that the selected schema actually creates the intended structure. These assignments are an authored platform policy fixture using a real policy engine, not a hand-coded substitute for it. Gatekeeper's own internal convergence mechanisms remain part of the callback [K7].

### 4.5 Security policy S: one predeclared change

Both revisions are a Kyverno `ClusterPolicy`, `background: false`, with annotation `pod-policies.kyverno.io/autogen-controllers: none`. Every rule matches only Pod `CREATE` in the named namespaces with the enabled label. Use strategic-merge **add-if-absent** anchors; do not force values over explicit input.

The shared rule iterates `request.object.spec.containers` and supplies missing `securityContext.allowPrivilegeEscalation: false` for each matched name. The only revision difference is where a missing seccomp profile is supplied:

| Revision | Seccomp default |
|---|---|
| `S0` before change | Add missing Pod `spec.securityContext.seccompProfile: {type: RuntimeDefault}` |
| `S1` candidate | Add missing `securityContext.seccompProfile: {type: RuntimeDefault}` to each regular container; do not add a Pod profile |

Essential patch bodies are below. Translate them into the full validated policies, preserving these semantics [K8].

~~~yaml
# S0 Pod-level rule's mutate.patchStrategicMerge:
spec:
  securityContext:
    +(seccompProfile):
      type: RuntimeDefault

# Common foreach item in both revisions:
list: request.object.spec.containers
patchStrategicMerge:
  spec:
    containers:
      - (name): "{{ element.name }}"
        securityContext:
          +(allowPrivilegeEscalation): false

# S1: add this sibling under the foreach securityContext above:
# +(seccompProfile):
#   type: RuntimeDefault
~~~

Add the entire missing `seccompProfile` object, not a missing `type` inside an existing object; this avoids silently changing an explicit `Localhost` profile. This migration is plausible when a platform moves security defaults to container scope, for example to avoid relying on Pod inheritance for special init-container cases. It is a chosen study fixture, not evidence of an observed production migration or product defect.

The possible mechanism is specific: an injector can create a new regular container after a container-scoped default has run. Whether S is then reinvoked, and whether reinjection removes third-party changes, affects the final result. Istio's pinned `stripPod` code removes previously injected containers before regenerating them to support reinjection idempotency [K4, function `stripPod`]. Do not replace I with “skip if status annotation exists.” Do not assume that I is idempotent on every intermediate object merely from that implementation comment.

### 4.6 Injector settings and the final contract

Use the standard sidecar template, ordinary non-native sidecar mode, no custom injection template, no CNI, and no TPROXY. Set the ordinary proxy resource defaults explicitly to CPU request `100m`, memory request `128Mi`, CPU limit `1`, memory limit `512Mi` using supported installation values; verify the rendered injector ConfigMap before freezing. Avoid unneeded telemetry/probe configuration changes. Record all defaults actually affecting the output.

The **same semantic contract** applies to S0 and S1 on each enabled, core fixture:

1. Admission succeeds. There are no duplicate container names. All original application containers and their images/commands remain present.
2. When injection is enabled for the fixture, there is exactly one regular container named `istio-proxy`. When explicitly disabled and no helper was supplied, no helper is added. An input `istio-proxy` customization with `image: auto` may be replaced with the pinned proxy image as documented; it must not produce a duplicate [K5].
3. Every regular container has CPU and memory requests and limits. Quantities are positive, and requests do not exceed their corresponding limits. An explicit fixture value is preserved unless a specifically documented injector override in the fixture permits replacement. Missing application values receive R's defaults. Proxy values must satisfy the declared injected/override profile, not an undocumented equality to application defaults.
4. Every regular container has `allowPrivilegeEscalation: false`.
5. Every regular container's **effective seccomp profile** is `RuntimeDefault`: use the container's profile when present, otherwise the Pod's profile. Both representations are permitted. Therefore different S0/S1 JSON layouts can satisfy the same contract.
6. Explicit compatible application security/resource fields are preserved. Fields outside the stated mutation permissions are retained in state comparisons; an unexplained change is investigated, not discarded.

For skip controls with `compat.poc/enabled: "false"`, R and S must make no changes; their required-default predicates are inapplicable. For explicit `Localhost` controls, allow that profile only on the container where it was supplied and require exact preservation; all other regular containers retain the RuntimeDefault requirement. For the explicit invalid-resource control, expected outcome is API rejection. Write these per-fixture permissions into the frozen contract rather than inferring them from observed output.

The contract is an external, independently implemented predicate. Algebraic equality is not the contract. If every output violates a predicate in the same way, the contract must still detect it. If S0 already fails on a workload, report a pre-existing incompatibility; do not call S1 a newly introduced regression there.

## 5. Implementation specification

### 5.1 Keep two levels of evidence separate

**Callback model:** actual component endpoints, fixed configuration/context, real patches, and the pinned Kubernetes patch/decode/default semantics. This supports exact statements about a completed finite transition table and conditional statements about the corresponding frozen services.

**Completed admission:** the actual local API server's response, audit annotations, and, for the live subset, stored object. This supports a contract outcome under the trace that actually occurred. Built-ins, registration matching, reinvocation history, validators, and storage-related changes belong here.

Do not treat these as interchangeable. The theorem does not model the full API-server scheduler simply because its three callbacks are real. A negative defect certificate for the callback model is not authorization to omit final validation or approve every production admission.

### 5.2 State, context, and equality

Represent a modeled state as `(pod, context_id)`. `pod` is the complete typed-serialization JSON for a v1 Pod at the declared callback boundary. Store the full object bytes and parsed tree. The context referenced by `context_id` includes the original admission name/namespace, `operation=CREATE`, resource/kind/version, fixed user info, dry-run mode, namespace labels, relevant server/configuration snapshots, selectors, and all policy identities and content hashes. Freeze the AdmissionReview request UID for replay; verify the response echoes it. Do not mistake UID echoing for a changed Pod.

Use a distinct context for `dryRun=true` and `dryRun=false`. Never merge them without a justified behavioral equivalence. Fresh real API requests necessarily have different request UIDs and server metadata; retain their raw records and compare contract predicates rather than pretending the complete objects are identical.

Retain full environment snapshots, but define the behavior fingerprint explicitly from policy `spec`, relevant ConfigMap `data`, injector template/mesh settings, registration matching/client/reinvocation/failure fields, namespace labels, and any declared external inputs. Controller-owned status, resource versions, timestamps and request-accounting annotations do not automatically constitute behavioral drift. This fingerprint rule applies to environment monitoring, not to dropping fields from modeled Pods. Re-read the relevant configuration before and after bounded batches and use a lightweight revision token within a batch. Meaningful drift invalidates that batch; do not issue full cluster reads on every transition.

Canonical equality may sort JSON object keys and ignore insignificant JSON whitespace. Preserve arrays, null versus absence, labels, annotations, status-like injection annotations, strings, and every field emitted by the typed boundary. Do not sort containers, tolerations, capabilities, or environment variables. Do not strip a field because the contract does not mention it. Any field discarded by the **actual typed decoder** is a recorded boundary transformation, not permission for the analysis to discard arbitrary data.

Use SHA-256 of canonical bytes as an index, with stored-byte equality to resolve matches and detect any collision. Reject NaN/infinity and lossy number coercion. Resource quantities use Kubernetes' exact parser and comparison routines; no Python `float` conversion. Separate the mathematical equality of normalized outputs from Kubernetes' own semantic `changed` flag, which can use different equality rules.

A complete transition record contains:

~~~text
run_id, stage, revision, fixture_id, context_hash,
component_id, component_version/config_hash, input_state_hash,
guard_result, request_body_hash, raw_response_hash,
outcome_kind, patch_type, patch, output_state_hash,
changed_before_default, elapsed_ns, cache_status,
attempt_index, source_trace_id, error_detail
~~~

Outcome kinds are `OK`, `SKIP`, `REJECT`, `CALL_ERROR`, `PROTOCOL_ERROR`, `PATCH_ERROR`, and `OUT_OF_MODEL`. `SKIP` is the only intentional identity from applicability. An allowed response without a patch is an `OK` no-change result. A failure that the API server ignores remains a failure observation, with a separate `failurePolicy` decision in the live trace.

### 5.3 Callback adapter and pinned normalization helper

Use a thin Python orchestrator, standard-library graph algorithms, JSON/JSONL files, and one small native Go helper for the server's typed patch/default semantics. Do not build a dashboard, framework, custom policy engine, distributed executor, or SMT encoding of the products.

Acquire the complete Kubernetes `v1.33.4` source tree including its vendor/staging content. Build the helper inside that tree with the declared Go toolchain, using vendored dependencies where available. The source declares Go 1.24.0 or newer; record the exact toolchain actually used. Import the pinned legacy scheme and core registrations, rather than assuming client-side API types include server defaulting. Do not build Kubernetes itself.

Build the helper once, not with `go run` per request. Relevant imports are `k8s.io/kubernetes/pkg/api/legacyscheme`, blank `k8s.io/kubernetes/pkg/apis/core/install`, `k8s.io/apimachinery/pkg/runtime/serializer/json`, `k8s.io/apimachinery/pkg/api/equality`, and `gopkg.in/evanphx/json-patch.v4`. Record relevant feature-gate/emulation settings and confirm the helper's defaults match the pinned API server. Keep the compiled helper as a subprocess with a simple JSON-lines interface, or invoke the built binary directly if calibration shows startup overhead is negligible.

Implement helper operations:

~~~text
normalize_seed(raw_v1_pod) -> normalized_v1_pod, normalization_diff
apply_response(input_v1_pod, raw_admission_review, expected_request_uid)
    -> outcome, output_v1_pod, changed_before_default, boundary_diff
quantity_compare(a, b) -> -1 | 0 | 1, or parse_error
~~~

For `apply_response`, reproduce `callAttrMutatingHook` at the pinned source [K2]: verify response protocol/UID, preserve explicit denial, handle empty/no patch without an extra defaulting pass, decode supported JSONPatch with the same library, encode the typed input, apply the patch, decode into a new typed object, compute semantic changed status **before** defaulting, then call the registered object defaulter. Use the same error distinctions. For the selected Pod-only scope, no CRD conversion or arbitrary resource support is required.

The helper specification is source-grounded but was **not built during preparation**. A compile or fidelity problem is an early bounded gate. Do not substitute hand-written Python defaults or strip unstable fields to pass it. If it cannot be completed within the repair allowance, retain raw callback observations as explicitly unnormalized diagnostics, drop theorem-to-admission boundary claims, and finish the analytic result plus any valid live admission work.

For direct callback calls, read the concrete Service path, port, CA bundle and TLS name from the installed registration. Use TLS-verified local port forwarding or an equivalent verified local route. Paths differ by product; do not hardcode all services to `/mutate`. Preserve the entire AdmissionReview envelope and raw response. Invoke the complete service callback; do not replace it with `istioctl kube-inject`, `kyverno apply`, or selected internal engine functions. Those can support unit checks but are not automatically equivalent to installed callback boundaries.

Construct a v1 `AdmissionReview` with outer `apiVersion: admission.k8s.io/v1` and `kind: AdmissionReview`; include request UID, Pod `kind`/`requestKind`, pods `resource`/`requestResource`, original name and namespace, operation `CREATE`, fixed `userInfo`, full `object`, `oldObject:null`, the cohort's `dryRun`, and matching `CreateOptions`. Verify the response envelope, UID, `allowed`, patch type and base64 patch encoding before application. Do not treat malformed or mismatched responses as no-ops.

For these fixtures, scope matching uses ordinary kind/operation/namespace/object-label rules. Re-evaluate object selectors on the current object before each model invocation. Snapshot namespace labels and recompute internal policy matching through the real callback. If an active CEL match condition or unsupported external predicate appears, implement its pinned semantics only if it fits the adapter repair allowance; otherwise mark that context ineligible. Never assume initially matching rules remain matching.

### 5.4 What audit evidence can establish

Enable a narrowly scoped `RequestResponse` audit policy for Pod CREATE in the study namespaces, with the audit log mounted to a local results directory. Correlate request and response events by audit ID, and sort webhook annotations by numeric round/index. Preserve the `mutation.webhook.admission.k8s.io/...` and `patch.webhook.admission.k8s.io/...` annotations.

Audit logs show invoked webhook identities, round/order, mutation flags, and patches when available. **They do not provide each callback input or all built-in mutation patches.** In the pinned CREATE handler, the audited request precedes built-in admission and field-management work [K3]. Label any reconstructed intermediate objects as reconstructed, and report unexplained differences. A matching final object alone does not prove every reconstructed intermediate state is correct.

Do not add a transparent proxy or patch controller registrations solely to obtain an attractive completeness claim. This small study intentionally accepts a boundary on attribution: a live final failure can be confirmed, while its exact correspondence to a replayed algebraic witness may remain unproven. If audit patches and source semantics are insufficient for a causal attribution, report the responsible callback set as unresolved rather than fabricating a complete transcript.

Kubernetes documentation does not promise fixed-point reinvocation. The pinned implementation executes at most two passes of the mutating admission chain, includes built-ins in the additional pass, and tracks previously invoked reinvocable callbacks. A webhook skipped in the first pass is not invoked for the first time in the second pass just because later labels match [K2, K3]. Preserve the observed schedule; do not use “repeat all rules until unchanged” as the admission oracle.

All six replay orders are valid algebraic experiments. They are not six observed deployment schedules. This study does not rename managed registrations to manufacture those schedules. If a replay order differs from the deployed order, classify its consequence as counterfactual unless independently reproduced under a supported, explicitly recorded deployment configuration. Such an extra configuration is not part of the mandatory experiment matrix.

### 5.5 Unambiguous algorithms

The pseudocode below specifies interfaces to create. It is not an existing executable package. `invoke` performs a real call unless a cached record is explicitly requested; `fresh=True` bypasses all callback-result caching.

~~~python
def step(component, state, context, fresh=False):
    assert current_batch_revision_token_is_valid(context)
    guard = evaluate_current_guard(component, state.pod, context)
    if guard == NO_MATCH:
        return Skip(state)  # record the exact guard and reason
    if guard != MATCH:
        return OutOfModel(reason="unsupported or erroneous applicability")
    raw = invoke_complete_callback(component, state.pod, context, fresh=fresh)
    result = pinned_go_helper.apply_response(state.pod, raw, context.request_uid)
    record_raw_request_response_and_result(result)
    if result.is_error_or_reject:
        return result  # never turn this into the input object
    if exceeds_model_bounds(result.pod):
        return OutOfModel(result)
    return Ok(State(result.pod, state.context_id))

def run_word(root, chronological_word, transition_store):
    state = root
    trace = []
    for component in chronological_word:
        result = transition_store.get_or_evaluate(component, state)
        trace.append(result)
        if not result.is_ok_or_skip:
            return Terminal(result, trace)
        state = result.output
    return Complete(state, trace)

def theorem_three(root, transition_store):
    witnesses = []
    incomplete = []
    for i, j in [(I, R), (I, S), (R, S)]:
        other = only_element({I, R, S} - {i, j})
        for prefix in [(), (other,)]:
            left = run_word(root, prefix + (i, j), transition_store)
            right = run_word(root, prefix + (j, i), transition_store)
            if not (left.complete and right.complete):
                incomplete.append((prefix, i, j, left, right))
            elif left.state != right.state:
                witnesses.append((prefix, i, j, left, right))
    return ObservedChecks(witnesses, incomplete)

def baseline_six(root, transition_store):
    # Independent traversal implementation; retain every prefix object.
    traces = [run_word(root, p, transition_store)
              for p in all_permutations((I, R, S))]
    return traces, pair_context_witnesses_from_retained_prefixes(traces)
~~~

Do not let `ObservedChecks` automatically imply a theorem certificate. Certification requires the following complete table construction and eligibility check:

~~~python
def capture_closed_table(roots, context, limits):
    queue = FIFO(roots)
    states = set(roots)
    table = {}
    while queue:
        enforce_remaining_time_memory_disk_and_query_limits()
        x = queue.pop()
        for i in (I, R, S):
            a = step(i, x, context, fresh=True)
            b = step(i, x, context, fresh=True)
            if not same_successful_object_outcome(a, b):
                return IneligibleOrIncomplete(states, table, a, b)
            y = a.output
            table[i, x] = y
            if y not in states:
                if would_exceed_state_cap(y):
                    return Incomplete(states, table, frontier=y)
                states.add(y)
                queue.push(y)
    # Every successor is present; errors/rejections never entered as identities.
    for x in states:
        for i in (I, R, S):
            if table[i, table[i, x]] != table[i, x]:
                return NonIdempotentWitness(i, x, table)
    return ClosedIdempotentTable(states, table, conditional_on_service_fidelity=True)

def independent_reachability_oracle(closed_table):
    B = {x for x in states
         if any(table[i, table[j, x]] != table[j, table[i, x]]
                for i, j in all_unordered_pairs((I, R, S)))}
    # Build predecessor lists independently from the theorem expressions.
    return reverse_BFS_from_B_over_all_three_transition_edges(B)
~~~

Implement early double-application probes on seeds and discovered successors **before** expensive closure, so an obvious assumption failure stops unnecessary work. The full table check remains necessary for a finite-table certificate. Where multiple roots share the same frozen context, merge exact states; context differences remain part of the key.

If a callback fails or rejects on an arbitrary replayed intermediate object, the complete successful-map model for that root is ineligible. Preserve the witness and continue already affordable actual admission checks. Do not invent absorbing error maps and claim the original theorem automatically proves admission safety. An API error can itself be a consequential outcome, but only if produced on an applicable completed admission and unexpected under the fixture contract.

On a closed eligible table, compare theorem membership to the independent reverse-reachability oracle for **every stored state**, not just seed workloads. This is an exact correctness check. If closure is incomplete, keep all witnessed differences, but report unobserved reachability as unknown. A no-warning result on incomplete capture is not a certificate.

### 5.6 Fair baseline and ablation

| Method | Information and role | Interpretation |
|---|---|---|
| Complete local API admission | Actual registered stack, built-ins, matching, reinvocation and validators; S0 and S1 | Practical reference for the deployed fixture's outcome |
| Cached six-permutation traces | Same actual callback adapter, roots, environment, full states, cache key, outcome handling and contract as the theorem | Principal computational baseline; can reconstruct every theorem witness |
| Theorem checker | Six pair/context comparisons; finite-table eligibility required for reachability claims | Proposed mathematical mechanism |
| Direct pair ablation | Three pair checks at the root only | Isolates the role of a third-operation context; deliberately weaker, not the main comparison |
| Independent table oracle | Reverse graph reachability on a complete tiny/real table | Correctness reference, not a scalable competing product |

All setup, normalization, closure, determinism probes, policy readiness, final validation, and failure reproduction costs are common or explicitly charged. Do not report the theorem as free because its input table was built by the baseline, or report a baseline as expensive because it was denied the same cache. Attribute shared acquisition cost to both complete methods when presenting standalone cost, and count it once in the actual machine budget.

There is no tuning allowance for the scientific mechanism. Both use the same complete outputs, selectors, and equality. Optimization is limited to ordinary memoization and efficient serialization available to both. Do not introduce learned surrogates, field projection, selective sampling based on observed defects, or hand-selected favorable orderings.

## 6. Data, experiments, and locked analysis

### 6.1 Deterministic fixture generation

Generate YAML/JSON programmatically; store every generated object and its factor manifest. Use explicit names, namespace, image references, commands, labels and fixed values. Runtime images are identifiers here, not executed workloads. Resolve any image reference that enters the contract to a fixed version/digest where available; do not depend on mutable tags for a claimed runtime property.

The **24 core confirmation fixtures** are the complete Cartesian product:

| Factor | Levels |
|---|---|
| Application container count | 1 or 2 |
| Injection situation | Enabled with helper absent; enabled with one `istio-proxy` override; explicitly disabled with helper absent |
| Application resources | All four fields missing; all four explicitly supplied |
| Application security | Missing; explicit compatible per-container fields |

Explicit application resources are CPU request `250m`, memory request `256Mi`, CPU limit `1`, memory limit `512Mi`. Explicit compatible security is `allowPrivilegeEscalation:false` plus `seccompProfile.type:RuntimeDefault`. For the helper override, use `image:auto` and a CPU request override `250m`, leaving other supported proxy fields to the documented injector/defaulting policy. Freeze the exact permitted proxy result before viewing interaction results. All core fixtures have the R/S enabled label. Use the injection label/annotation mechanism actually supported by the pinned registrations; verify that explicit disabled overrides the namespace setting.

Create **eight development fixtures** separately, covering each individual callback, no-op behavior, missing parents, explicit-field preservation, helper customization and the normalization path. Their expected local assertions come from the declared policy, not from recording whatever the implementation returns. They may share structural patterns with the core census; they are not statistically independent samples. Do not describe holdout names as an independent production distribution.

Create **eight fixed stress fixtures**, each derived from a documented core pattern:

1. Only an application CPU request is supplied; all other resource fields are missing.
2. Application limits are supplied and requests omitted. Predeclare normal Kubernetes request-from-limit defaulting as permitted; distinguish it from R supplying a missing field after the server has already defaulted it. Verify this exception against the pinned normalizer before freezing the contract.
3. One application container has an explicit `Localhost` seccomp profile with a fixed relative profile name. Permit and preserve that profile for that container; do not test whether the profile exists on a node.
4. R/S enabled label is false, with explicit application settings. Confirm actual skip behavior; default predicates are inapplicable.
5. A complete previously injected Pod snapshot generated by I alone using its intended frozen evaluation name, namespace and request context, retaining all injection annotations and override data. Prepare it before confirmation and record its provenance; do not inject a development Pod and then rewrite identity-dependent fields.
6. Injection is explicitly disabled while namespace injection is enabled, with two applications and mixed supplied settings.
7. Application CPU request `2` with limit `1`, expected to be rejected by API validation. This is an outcome-path control, not a claimed cross-product regression.
8. Two applications with heterogeneous settings: first explicit, second missing. This checks iteration coverage and preservation together.

The corpus is an authored, bounded compatibility fixture. It is neither a random production sample nor evidence of defect prevalence. No realistic failing instance is promised. Do not modify resource constants, policy scope, order, or fixture selection after seeing whether a failure appears.

### 6.2 Development, pilot, freeze, and confirmation

Use development fixtures and mathematical controls to debug. Run a small mechanism pilot on development fixtures only. Confirm that endpoint invocation, actual mutation, no-op behavior, expected validation failure, and S0/S1 representation differences are observable. The pilot is not used to select a winning method or maximize warning rate.

After activating either revision, wait for relevant readiness/observed-generation status and use a deterministic missing-seccomp sentinel through server-side dry-run to demonstrate the intended Pod-level versus container-level placement. Require two consecutive correct sentinel responses before starting a batch, with at most four sentinel attempts and a two-minute total activation deadline. Sentinels never use live CREATE; count each against the 256-request total. Batch reproductions by revision where possible to limit unnecessary switching. A hash of the API policy object alone does not prove the controller's cache has loaded it. Failure of this barrier is a readiness/configuration problem; do not infer nondeterminism or a policy regression from stale revisions.

Before the core census, write and hash `protocol-lock.json`, containing all source/image identifiers, installed manifests, policy revisions, fixture bytes, contract predicates, context snapshots, equality rules, budget/cap ledger, and analysis rules. If calibration requires the smaller predeclared allocation in §8, record that choice before examining confirmation results.

A syntax repair preserving the declared semantics is allowed before locking. A later change to policy intent, primary question, baseline, contract, or corpus makes the affected results exploratory. Do not reuse those results as fresh confirmation. If insufficient budget remains for a fresh locked run, report the limitation rather than continuing until a desired outcome appears.

### 6.3 Minimum experiment matrix

| Experiment | Inputs and repetitions | Claim or uncertainty resolved |
|---|---|---|
| Analytic query-set check | Exact symbolic prefixes, one deterministic run | Three-component observation equivalence |
| Tiny tables and controls | All ordered idempotent triples on ≤4 states if the supplied or independent verifier finishes within its cap; mandatory named controls regardless | Formula correctness, composition direction, masking and exclusions |
| Adapter smoke | 8 development fixtures; both revisions as relevant | Real callbacks, policy semantics, decoder/default fidelity |
| Calibration | 100 direct fresh calls drawn round-robin across the three components and representative development inputs; 8 full API dry-runs | Time, memory, readiness and feasible allocation |
| Mechanism pilot | Development fixtures under S0 and S1; cached traces and a bounded closure attempt | Eligibility, observable warning classes, baseline equality |
| Locked replay/model census | 24 core + 8 stress fixtures under S0 and S1; one capture per frozen context, duplicate fresh responses per transition | Closed-table eligibility where attainable; exact witness and baseline agreement |
| Locked actual admission census | Same 32 fixtures × 2 revisions × 2 fresh server-side dry-run requests = 128 completed requests | Observed final contract outcomes and repeatability on current deployment |
| Live CREATE parity/reproduction | At most 40 creates total; fixed benign controls plus every distinct candidate failure up to four witnesses | Whether dry-run evidence also applies to non-dry-run CREATE |
| Diagnosis/robustness | Only justified witness shrinking and one implementation/boundary-fidelity repair; all count toward common caps | Reproducibility and attribution, not prevalence or speedup |

The eight development fixtures can add up to 32 dry-runs across revisions/repetitions; readiness sentinels and retries are also charged. The total complete-admission request cap is 256, including dry-run and live create attempts. Cleanup deletions are logged separately and do not count as experimental CREATE outcomes.

For live parity, select four benign core fixtures deterministically by factor coverage before observing results: one missing-settings case with injection, one explicit-settings case with injection, one helper override, and one injection-disabled case. Run each in both revisions. For each selected candidate failure, require three fresh live CREATE repetitions under S0 and three under S1, with the identical original fixture and contract. Delete the prior study Pod and verify deletion between creates; keep names fixed within a fixture where possible. Reapply the revision activation barrier when switching. If more than four distinct failures occur, retain all observations and reproduce the first four under lexicographic fixture ID, reporting the reproduction cap. Do not silently count unreproduced candidates as confirmed.

Reserve the live request budget before shrinking: `4 benign × 2 revisions + 4 candidates × 2 revisions × 3 repetitions = 32` creates, leaving eight of the 40 for bounded diagnosis/repair. An S0 dry-run pass plus an S1 live failure is insufficient to establish a newly introduced live regression.

Dry-run is its own context, not automatically equivalent to live admission. If a product reacts differently to `dryRun`, preserve both results and narrow the claim. A dry-run-only failure is a confirmed dry-run incompatibility until its live applicability is demonstrated. A live create accepted but violating the contract is also a consequential result; rejection is not required.

### 6.4 Classify evidence on separate axes

Use separate fields instead of one overloaded pass/fail score:

| Axis | Values |
|---|---|
| Model eligibility | Closed/idempotent table; non-idempotent; nondeterministic evidence; terminal/error path; incomplete closure; boundary fidelity unverified |
| Algebra | Witness found; no defect in certified model; no defect observed with incomplete coverage |
| Actual admission | Contract pass; unexpected rejection; accepted contract violation; expected rejection; not run/error |
| Schedule applicability | Observed pinned deployment; supported separately reproduced configuration; abstract replay only |
| Attribution | Source/audit-supported interaction; final failure with incomplete causal link; single-component or pre-existing issue; unresolved |
| Change effect | S0 pass/S1 fail; both fail; both pass; S0 fail/S1 pass; insufficient evidence |

A **confirmed newly introduced admission incompatibility** requires S0 to satisfy the same fixture contract, S1 to violate it reproducibly on a live applicable CREATE, and evidence linking the consequence to the interacting stack rather than a malformed test or unavailable service. If the final violation is real but the interaction attribution is unresolved, report both facts separately; do not upgrade the causal claim.

Operationalize attribution using the existing matched injection-disabled fixtures and isolated S probes. If S1 fails on original application containers in the same way when injection is disabled, classify that as a single-component/policy problem unless additional evidence establishes interaction. A proxy-only omitted setting after injection can support interaction attribution when S1 demonstrably defaults an equivalent already-present regular container and the actual audit/source evidence supports the relevant ordering or reinvocation explanation. If the required intermediate input cannot be established, keep the confirmed final violation and mark its interaction attribution unresolved. Do not blame R merely because it belongs to the stack.

An algebraic witness alone is a warning. A difference removed by later mutation remains a warning. Two accepted final objects that are both allowed by the contract are not an incompatibility. A counterfactual replay failure is not a demonstrated problem in the current registration order. A rejected arbitrary replay intermediate is not an unexpected live rejection.

### 6.5 Diagnosis and fair added-value analysis

For every warning/failure, save the seed, revision pair, context and policy hashes, component pair, context prefix, full left/right traces, structural JSON difference, contract predicate results, actual response/audit IDs, and reproduction status. Report all supporting components when a short global witness does not identify the originally suspected pair. Never claim the theorem finds every causal pair.

Use the cached baseline to reconstruct the same witness record and contract checks. Compare exact contents and query-key sets. If the theorem's report contains information absent from the baseline report, determine whether it came from additional calls, a different context, or presentation code; allow the baseline the same computations before calling it an advantage.

When shrinking a witness, use a fixed greedy sequence of optional field removals with at most 20 candidate replays and at most 6 additional complete API requests per selected witness, inside the global limits. Preserve the original evidence and contract class. Do not call a greedy result globally minimal. If mutation changes applicability, rerun matching and the actual oracle before retaining the smaller example.

Finish the reserved paired reproductions before spending live requests on shrinking; all witnesses together share the eight remaining live creates. Shrink candidates remain diagnostic examples, not fresh confirmation fixtures. The global admission, callback, time and disk caps always override per-witness allowances.

### 6.6 Measurements, failures, and uncertainty

Report exact counts: fixtures planned/completed, closed tables, states, transition calls and cache hits, direct/contextual warnings, actual violations, expected/unexpected rejects, call/patch errors, incomplete captures, and baseline disagreements. Use fixture/revision as the unit of descriptive comparison. The multiple permutations, repeated calls, and repeated admissions of one fixture are not independent workload samples.

No significance test, confidence interval on production failure probability, or seed sweep is needed for the exact set equality and deterministic census. Repeated service calls check reproducibility. If outcomes vary, report the variation and disqualify the deterministic interpretation; do not average it away.

Use monotonic wall-clock timing around complete client operations, including serialization, transport, patch application and normalization. Record service startup/readiness, source compilation, corpus generation, closure capture, pure table analysis, actual admission, retries, and report generation separately. Report a cold startup/capture and subsequent cached analysis as different activities. Do not call cache hits a service speedup. There is no device synchronization issue because no GPU backend is used.

Record process peak RSS using macOS facilities such as `/usr/bin/time -l` and periodic process-tree sampling; separately monitor the VM/container allocation and resident use. A Linux cgroup reading inside kind is not the Mac's total memory. Record disk size of sources, build caches, images, object store and logs. Calibration medians and upper quantiles describe feasibility only; they are not a comparative performance result.

Include every timeout, error, excluded context and missing run in denominators and in a failure ledger. Preserve partial logs before stopping. A timeout is practical infeasibility or incomplete evidence at the applicable scale, not a falsification of the theorem. An implementation or measurement defect invalidates the affected comparison; fixing it does not erase the failed attempt from the budget.

## 7. Minimal repository and runnable handoff

Create only the modules required by the stage reached. A suitable layout is:

~~~text
README.md                         commands, prerequisites, scoped result
requirements.txt                  PyYAML==6.0.2; otherwise Python standard library
dependency-lock.json              exact Python, Go, CLI, image and source identities
config/study.json                  scientific settings and cumulative caps
config/kind.yaml                   local cluster and audit mounts
config/audit-policy.yaml           narrow Pod CREATE auditing
config/protocol-lock.json          frozen confirmation specification
inputs/                           unchanged supplied documents/source bundle
policies/resources/                four Gatekeeper Assign objects
policies/security-s0.yaml          complete before policy
policies/security-s1.yaml          complete candidate policy
fixtures/dev/                     eight development objects
fixtures/core/                    24 confirmation objects
fixtures/stress/                  eight declared stress objects
admission_poc/__main__.py          CLI entry point
admission_poc/model.py             outcomes, full states, equality, object store
admission_poc/adapters.py          real AdmissionReview calls and guard results
admission_poc/capture.py           bounded closure and fresh-response checks
admission_poc/theorem.py           six bounded comparisons
admission_poc/baseline.py          independent permutation-prefix traversal
admission_poc/reference.py         tiny tables and reverse-reachability oracle
admission_poc/contracts.py         independent final predicates
admission_poc/actual.py            API requests, auditing, revision barriers
admission_poc/report.py            deterministic tables and report assembly
helper/kube_patch/main.go          copied into the pinned source tree to build
scripts/                          setup, cleanup and environment inspection
tests/                            exact controls and meaningful adapter tests
results/<run-id>/                  append-only ledger, raw calls, objects, audit
report/REPORT.md                   research conclusion and reproducibility index
~~~

Use Python 3.12 or newer, recording the exact version. PyYAML is only for manifest handling, not mutation semantics. Standard-library `unittest`, integer arithmetic, JSON, hashing, and graph code are sufficient. Prefer the installed compatible native toolchain; install pinned CLI releases from official sources where version matters. Record download checksums and executable versions. Resolve command-line flags against each installed binary's help; do not assume current documentation matches the historical release.

These are **interfaces you must implement**, not commands verified during preparation:

~~~bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m admission_poc preflight --config config/study.json
python -m admission_poc mathcheck --config config/study.json
python -m unittest discover -s tests
python -m admission_poc setup --config config/study.json
python -m admission_poc smoke --config config/study.json
python -m admission_poc calibrate --config config/study.json
python -m admission_poc pilot --config config/study.json
python -m admission_poc lock --config config/study.json
python -m admission_poc run --protocol config/protocol-lock.json
python -m admission_poc report --results results --output report/REPORT.md
python -m admission_poc cleanup --config config/study.json
~~~

`run` must check completed gates and remaining budgets before executing; `lock` must refuse an incomplete scientific configuration. `report` must work after an early stop and without network access. Cleanup must first preserve logs and identify the exact study cluster; it must not delete unrelated clusters, namespaces, caches or user files. A reader must be able to regenerate the essential evidence tables from saved raw results without redeploying services.

Required machine-readable outputs are `budget-ledger.jsonl`, `environment.json`, `source-lock.json`, `calls.jsonl`, `outcomes.jsonl`, `coverage.json`, `failures.jsonl`, and `witnesses.json`. Store complete objects and large raw responses by content hash with references from JSONL records. SQLite is optional only if ordinary files become cumbersome; no external database or service is required.

## 8. Mac feasibility, cumulative limits, and evidence gates

### 8.1 Inspect before allocation

Record `sw_vers`, `uname -m`, `sysctl -n hw.memsize`, CPU identification, `vm_stat`, memory pressure, free disk, native runtime details, Docker context, and all tool versions. Confirm that the target is the user's M3 Max Mac and that images execute as Linux ARM64. Hosted preparation measurements must not be presented as laptop measurements.

The live stack requires at least 24 GiB installed RAM, normal memory pressure, and 45 GiB free disk under this plan. These are conservative planning requirements, not facts inferred from the chip. If they are not met, complete the analytic/tiny study and report the live allocation as unavailable under the chosen limits. Do not start swap-driven experiments or delete unrelated material to create space.

Use a native VM with 6 virtual CPUs and 8 GiB RAM. One increase to 10 GiB is allowed as a recorded implementation/resource repair, only if the total memory limit below still holds. The analysis process has a 2 GiB RSS ceiling. Compile the Go helper before starting the full stack where possible; allow up to 3 GiB for compilation. The combined measured resident/allocation envelope must stay below `min(14 GiB, 0.60 × installed RAM)`; do not add component ceilings together and ignore the combined limit.

### 8.2 Hard cumulative ceilings

| Resource | Ceiling and reason |
|---|---|
| Download/build/deployment setup | 60 minutes total, including failed setup attempts; protects against infrastructure consuming the project |
| Experimental/checking/report runtime | 180 minutes total across all stages, exploratory runs, retries, repairs and fallback work; 30 minutes reserved for preservation/analysis/report generation |
| Implementation and interpretation effort outside those running jobs | 6 hours cumulative; overall active mission ceiling 10 hours including setup and experiment budgets |
| New project disk use | 35 GiB including VM/image/source/build caches and logs; retain at least 10 GiB free |
| Serialized modeled Pod | 128 KiB; exceeding it produces `OUT_OF_MODEL`, not a truncated object |
| Reachable-state capture | 256 states per root/context, 2,048 per revision, 4,096 total across revisions including development captures |
| Fresh callback calls | 40,000 total, including duplicate checks, failed attempts, shrink attempts and actual API callback invocations where observable |
| Complete API admission requests | 256 total; at most 40 are live CREATEs |
| Model capture concurrency | One outstanding direct callback call initially; keep serial execution unless calibration justifies a documented change |
| Repairs | One substantive implementation/fidelity repair cycle, at most 20 minutes experimental runtime; consumes the same caps |

Do not reset any counter on a revision change, a rerun, or a fallback. Track controller uptime and idle/setup overhead in the total mission account even when not included in a timed experiment. The machine-runtime caps are not a claim that building a correct harness is instantaneous. Estimated implementation/interpretation is 3–6 hours, setup 30–60 minutes, and an informative pilot 30–55 minutes after prerequisites; the full experimental allowance is three hours. These estimates have not been measured on the target.

A complete 4,096-state table has at most 12,288 deterministic edges; two fresh calls per edge require up to 24,576 callback calls before additional validation and retries. Raw state bytes at the 128 KiB ceiling total 512 MiB; Python overhead, indexes, patches, responses and service memory are extra. Stream records to disk, avoid retaining multiple decoded copies, and stop before the memory cap. The 40,000-call allowance leaves bounded room for controls and real admission; it is not permission to exhaust every cap simultaneously.

The dominant likely costs are service startup, native compilation, per-call HTTP/serialization/defaulting, and closure size. They are not large dense numerical kernels. There is no reason to add GPU acceleration. Use exact reference code before any optimization.

### 8.3 Calibration and resizing rules

Before confirmation, estimate remaining runtime using measured call costs. For direct capture, use the larger of measured p95 call latency and 1.5 times the mean, multiplied by the planned uncached call count. Add measured API-request cost, expected closure bookkeeping, unresolved startup work, and the full 30-minute reporting reserve. Treat this as a conservative planning estimate, not a statistical bound.

If the forecast exceeds the remaining allocation:

1. Remove optional shrinking and additional robustness work first.
2. Use a **predeclared balanced 12-fixture core subset**: index count as `c∈{0,1}`, injection situation as `i∈{0,1,2}`, resources as `r∈{0,1}`, and security as `s∈{0,1}`; retain exactly those with `s=(c+r+i) mod 2`. Retain stress cases 1, 3, 4 and 7. This keeps every factor level while giving up the complete 24-case factorial census. Lock and disclose the reduction before results.
3. Reduce expensive closure certification to that locked subset, without enlarging state limits or discarding a failing root. Preserve ordinary admission evidence separately.

Reduced mode uses the same 16 fixtures for both the replay/model census and actual admission census, requiring `16 × 2 revisions × 2 repetitions = 64` locked dry-run requests. The experiment matrix's 32 fixtures and 128 requests describe full mode only. Select the four benign live controls from the locked core allocation before viewing its results.

If even that allocation is infeasible, stop the locked real-component comparison and report measured practical infeasibility or insufficient coverage. A smaller hand-picked successful example is not the planned study. Never weaken equality, remove status/annotations, skip troublesome components, or change security requirements to fit the budget.

### 8.4 Gates and required next actions

Stage allowances below sum to 180 experimental minutes; setup has its separate 60-minute ceiling. Unused time may flow forward within the total, except that the reporting reserve cannot be consumed by new experiments.

| Gate | Question and minimum work | Evidence/artifact | Allowance | Decision and next action |
|---|---|---|---:|---|
| G0: constructive/analytic feasibility | Inspect environment and inputs; independently derive prefix-set equality and inspect source assumptions | Environment inventory, exact derivation, source hashes | 10 min | Equality verified: lock the no-incremental-observation result and authorize only the bounded applicability diagnostic. Missing hardware does not block this result. A genuine proof discrepancy requires bounded audit before any live work. |
| G1: tiny correctness | Named controls, independent tiny table oracle, source verifier rerun if available | Exact control outputs, failing assertions with witnesses | 5 min | All essential controls pass: proceed. A failed universal property/control is an implementation defect until resolved; use repair allowance once, otherwise stop invalidated portions. |
| Setup gate | Obtain native tools/sources/images; create isolated cluster; install exact policies and helper | Tool/image/source lock, manifests, readiness log | 60 setup min | Missing artifact/architecture or failed installation after one bounded repair: stop affected real work, retain analytic outcome. No product-version substitution. |
| G2: adapter and calibration | Validate policy intent, TLS/protocol handling, pinned normalization, meaningful snapshot checks, 100 calls and 8 admissions | Calibration measurements, boundary tests, allocation decision | 20 min | A faithful callback model and feasible forecast: continue. Unresolved normalization permits only explicitly raw-callback/live diagnostics. Hidden extra callbacks or external state block the three-map guarantee. |
| G3: mechanism pilot | Development data; early idempotence checks, limited closure, A/B witness equality, actual S0/S1 outcomes | Pilot table, eligibility failures, warning/finality examples if any | 20 min | A reproducible idempotence/nondeterminism failure stops global theorem certification; preserve it and proceed only with bounded observational admission work. No warnings is not a failed experiment. Small incomplete pilot: use forecast and one bounded extension only if it resolves a named uncertainty. |
| G4: locked census | Freeze protocol; capture planned tables and actual outcomes with all failures counted | Lock hash, raw calls/admissions, coverage and outcome tables | 60 min | Exact A/B agreement and complete coverage support the analytic limitation and scoped census. Missing coverage stays inconclusive. Stop before reserve/caps; do not rescue the hypothesis by changing inputs. |
| G5: targeted diagnosis | Paired live reproduction and justified attribution/shrinking only | Reproduction package, original and reduced witnesses, causal limits | 15 min | Reproducible applicable violation: classify it precisely. Masked/counterfactual or unsupported causal claim: warning/unresolved. No failures: report the actual pass scope. |
| Shared repair reserve | One ordinary implementation, readiness or boundary-fidelity repair, with before/after evidence | Deviation and repair log | 20 min | Repair must preserve scientific intent. If not resolved, stop the affected claim. No semantic policy repair campaign. |
| G6: preservation and report | Analyze completed evidence, regenerate tables, document limits, package artifacts | Report, data/code index, checksums, rerun instructions | 30 min | Always execute, including after a fatal earlier gate. |

A reproducible counterexample to a realistic operation's idempotence invalidates the full proposed union-domain promise. Do not delete that fixture to report a theorem-backed pass. You may report predeclared successful subdomains separately, but the original family remains ineligible. If only source access or service availability is missing, that is a blocker/inconclusive applicability result, not proof that realistic operations cannot be idempotent.

The analytic conclusion remains worthwhile even if all live work stops. Do not build the later infrastructure merely to fill a checklist after a decisive gate failure.

## 9. Outcome interpretation and final artifacts

### 9.1 Required final verdicts

State separate verdicts for the analytic advantage claim, model applicability, and observed release decision. Do not compress them into a single successful/failed PoC label.

| Research category | What justifies it in this project |
|---|---|
| Supported contribution | The exact limitation is verified and documented; or a separately stated, reproducible real-stack finding is supported. Do not relabel expected A/B equality as detector superiority. |
| Informative negative result | The 15-position equivalence rules out the proposed observation economy at `k=3`; or a valid, sufficiently covered diagnostic contradicts a specified eligibility/application hypothesis. State which claim is negated. |
| Practical infeasibility | Measured setup/capture/memory costs or supported scaling prevent the declared scope under the fixed caps. Preserve the measurements and extrapolation. |
| Inconclusive evidence | Missing source/hardware, unresolved boundary fidelity, incomplete closure, inconsistent responses, insufficient reproduction, or incomplete workload coverage prevents the particular conclusion. |
| Invalid experiment | Adapter, reference, baseline, contract, protocol, data or measurement errors undermine the attempted comparison. Preserve failed runs and identify which results must not be interpreted. |

These can coexist across subclaims: for example, a verified analytic negative result and inconclusive live applicability. A quiet stack is not a universal proof of compatibility. Failure to find a bug is not evidence that cross-component problems never occur. Non-idempotence in one chosen version is not a universal impossibility result for all Kubernetes mutators.

For a platform recommendation use: `BLOCK_CANDIDATE_FOR_FIXTURE`, `INVESTIGATE`, `SCOPED_PASS`, or `NO_DECISION`, accompanied by workload IDs, versions and coverage. A scoped pass never means an unqualified production rollout recommendation.

### 9.2 Required deliverables, proportional to stages reached

- A standalone `report/REPORT.md` leading with the three verdicts, the proposed change, tested scope, and the decision it supports. Include the exact prefix equivalence derivation, relevant source assumptions, real-stack findings, failure ledger summary, measured costs, limitations and one justified next step if any.
- Runnable code and a concise README for every implemented stage; no fictional commands for work never implemented. If stopping analytically, provide the derivation, exact controls and minimal verifier, not a hollow service framework.
- All generated fixtures, full before/after policies, actual registrations/config snapshots, source/image/toolchain identifiers, and the protocol lock.
- Raw results, failed attempts, exact transition tables where completed, pending closure frontier where incomplete, actual admission/audit records, and reproducible witness bundles where found.
- Essential tables: source/assumption status; exact A/B query-set and witness agreement; per-fixture eligibility/algebra/outcome/applicability/attribution; stage costs/cap consumption. A simple state-count or timing plot is optional only if it explains a real limit. Do not add charts for appearance.
- Checksums and an offline command regenerating the essential tables from raw files. Package a source/results archive if needed to make transfer simple; do not include downloaded image layers or secrets.

Report actual hardware, architecture, backend, arithmetic choices, exact versions, seeds used by any randomized verifier, and every substantive deviation. No seed is needed for the deterministic corpus. Preserve unredacted local research records as appropriate, but do not publish kubeconfig credentials, TLS private keys, service tokens or unrelated cluster data in the portable package.

Do not claim a reviewer, search, proof check, experiment, dry/live parity result, benchmark, or source inspection that you did not perform. Distinguish the preparation checks in this brief from your own execution.

## 10. Supporting research, claim map, and completed design review

### 10.1 Closest alternatives and novelty assessment

Rehearsal already analyzes configuration determinacy and idempotence, using commutativity-based pruning; its discussion includes later writes erasing earlier nondeterminism [R1, §4.3 and §6]. Hanappi, Hummer and Dustdar formulate action success/failure, resource preservation, a preservation-based convergence theorem, and reduced transition-graph testing [R2, §§4–6]. Their preservation equation uses chronological composition, so do not copy it into this paper's opposite convention without translation. Neither work is a drop-in executable model of arbitrary Kubernetes webhook services.

Dynamic partial-order reduction defines independence using both preservation of enabledness and commuting effects [R3, §2.2, Definition 1]. It explains why observed output equality alone is insufficient to prune schedules with selectors or scheduler history. General confluence/critical-pair language is relevant, but local joinability, exact commutation, global termination and final contract satisfaction are distinct properties. Do not assume a termination-based confluence theorem applies to idempotent generators: §3.6 exhibits an infinite orbit.

Gatekeeper documents the difference between individual and collective idempotence and its own mutation-convergence design. Kyverno supports predeclared policy-result tests; Chainsaw supports Kubernetes end-to-end tests [R4, R5]. They establish a strong ordinary-testing baseline. Reimplementing an old Puppet verifier or an asynchronous reconciliation tester would add cost without matching this admission-only question; no comparison to incompatible published timings is authorized.

The broader mathematics also does not furnish a free operational algorithm: almost-commuting permutation stability concerns invertible maps [R6], while strongly sofic monoid work explicitly uses fiber control [R7, Definition 3.1]. These are related foundations, not a license to transfer guarantees to arbitrary partial, external-state-dependent webhook behavior. The supplied exact reachability proof is constructive and sufficient for the selected finite checker; no hidden optimization oracle is required. Certifying that the real operations satisfy its promise is the substantial application cost.

Searches on 1 November 2026 used both available search engines and terms spanning almost-commuting retractions, idempotent transformation stability, unary property testing, confluence, partial-order reduction, configuration preservation/determinacy, Kubernetes webhook conflicts, Gatekeeper mutation and Kyverno testing. Backward links to partial-order reduction and forward work on Puppet trace diagnosis were followed where relevant. Primary paper sections and official documentation/source were read, not just snippets. No exact historical antecedent of the supplied `k−2` formula was located, but no exhaustive novelty claim follows. Inaccessible full text, dissertations, internal tools, patents and alternate terminology remain coverage limits. A final-version Puppet trace-analysis PDF was inaccessible; its preprint was read for context only and supports no central claim here. The supplied report itself notes incomplete access to the centralizer literature.

### 10.2 Compact claim-to-evidence map

| Central claim | Source/derivation and status | Remaining uncertainty and resolving check |
|---|---|---|
| Exact bounded reachability under idempotence | [A1] Theorem 4.1; proof audited; exact finite checks performed | Implementation direction/equality: tiny oracle and all-state comparison |
| No extra three-generator transition observations | §3.3 derivation; independently checked prefix sets | Harness must not add asymmetric caches/queries: compare actual query keys |
| Small commutation defect does not imply small triple repair | [A1] Theorem 3.1 | No policy repair follows; no empirical repair experiment needed |
| Real callback maps fit the theorem | Hypothesis, not established by preparation | Fresh duplicate responses, early repeat checks, closed-table exhaustive idempotence; retain fidelity condition |
| Pod-to-container seccomp migration has meaningful interaction potential | Declared fixture plus injector/source semantics [K4, K8] | Real S0/S1 admission census and paired live reproduction; no predicted bug asserted |
| Reinvocation is not arbitrary fixed-point iteration | Official docs and pinned code [K2, K3, K9] | Inspect actual registrations/audit rounds; do not extrapolate implementation details across versions |
| All apparent disagreements are deployment failures | False in general; exact masking control | Final external contract and actual schedule required |
| Proposed implementation is M3 Max-feasible | ARM64 index availability checked; costs are estimates | Target preflight, native helper build, latency/memory/disk calibration |
| Historical novelty or productivity advantage | Unresolved/not claimed | Specialist literature or a separate human study would be needed; neither is a rescue branch |

### 10.3 Three review passes and stress-test disposition

The preparation process used separate mathematical, experimental and execution reviewers, with the final decisions reconciled in this brief. They are independent agent reviews, not external peer review. Material findings and dispositions:

| Review finding/stress scenario | Disposition |
|---|---|
| Strong baseline removes the anticipated gain | Promoted the exact 15-position equivalence to the primary result; removed the speedup benchmark |
| README could imply per-pair completeness | Restricted theorem claims to union reachability; added a verified counterexample |
| An intermediate defect can be erased, or all equal outputs can be invalid | Added both controls and an independent final contract |
| Real assumptions may fail; closure may be infinite | Added early eligibility probes, explicit finite closure caps and a stop/fallback path |
| Nonconstructive theorem or hidden oracle | Essential proof is constructive; no optimizer required; assumption verification is charged explicitly |
| Only favorable synthetic controls work | Synthetic evidence validates code only; real fixed corpus is required for practical claims; no fault-search branch |
| Extra preprocessing or implementation imbalance creates a gain | Shared capture/normalization costs and identical baseline information are mandatory |
| Numerical representation changes the result | Exact typed boundary, preserved arrays/fields and exact quantities; no tolerance adjustment |
| Applicability changes or predicate is unsupported | Current-state matching and explicit non-skip unsupported outcome; dynamic guard harness control |
| A stale policy cache fabricates a revision effect | Mandatory activation barrier and semantic configuration drift checks |
| Dry-run/live difference fabricates a regression | Paired S0/S1 live reproduction with reserved request budget |
| Audit evidence is mistaken for a complete transcript | Explicitly separate observed annotations from reconstructed intermediates and leave causal attribution unresolved when necessary |
| Runtime/memory or small-pilot uncertainty exceeds expectations | Forecast before locking, one fixed reduced allocation, then stop; preserve report reserve |
| Essential source/image/hardware is missing | One alternate official retrieval; analytic work still completed; blocked empirical claims identified |
| Main application hypothesis is false | Informative negative outcome is an authorized completion; no requirement to invent a superior detector |

Residual risks are substantive and visible: the normalizer and policies remain unexecuted; the historical stack's combined deployment may fail; service determinism is empirical beyond the recorded table; audit data may not permit precise causal reconstruction; and closure may exceed the available resources. Each has an early gate or a limited conclusion. There is no claim that the brief eliminates every possible implementation defect.

## References

All web resources below were accessed during preparation on **1 November 2026**, unless explicitly identified as context-only. URLs to pinned source tags are preferable to moving documentation for implementation facts. Record the full commit resolved from each tag during execution.

### Supplied sources

- **A1.** *Repairing almost-commuting retractions: An exact two-versus-three threshold and a sharp bound on defect reachability. With an independent sharp inequality for binary correlation cycles.* Credited in the document as “Autonomous mathematical investigation,” 1 November 2026. Supplied `research_report(1).pdf`, and identical text in `research_bundle/research_report.pdf` and `.tex`. Critical locations: Theorem 1.1 p. 3; Lemma 2.1 pp. 3–4; equation (1) p. 4; Theorem 3.1 pp. 5–6; Theorem 4.1 pp. 6–7; Corollary 4.2 and equation (2) p. 7; testing consequence §4.3 p. 8; novelty/limits §§6.2–6.3 pp. 9–10; separate appendix pp. 10–13. No public DOI or verified public URL was supplied.
- **A2.** *Research investigation bundle*, same supplied investigation/date. `research_bundle(1).zip`, directory `research_bundle/`. README, three verifier scripts, captured outputs, binary JSON results, all investigation notes and independent algebra audit. Earlier speculative notes are subordinate to the final report's proved statements; captured results are not proof of universal claims.

### Research and testing prior art

- **R1.** Rian Shambaugh, Aaron Weiss, Arjun Guha. *Rehearsal: A Configuration Verification Tool for Puppet.* PLDI 2016, pp. 416–430. DOI: [10.1145/2908080.2908083](https://doi.org/10.1145/2908080.2908083). [Author PDF](https://weiss.city/pubs/pldi16-rehearsal.pdf); [arXiv record](https://arxiv.org/abs/1509.05100). Relevant: §4.3 commutativity/pruning; §6 erased nondeterminism example. Full relevant content read.
- **R2.** Oliver Hanappi, Waldemar Hummer, Schahram Dustdar. *Asserting Reliable Convergence for Configuration Management Scripts.* OOPSLA 2016, pp. 328–343. DOI: [10.1145/2983990.2984000](https://doi.org/10.1145/2983990.2984000). [Author PDF](https://dsg.tuwien.ac.at/team/sd/papers/OOPSLA_2016_O_Hanappi.pdf). Relevant: §4.1 success/failure semantics; §5.2 Definition 7, p. 333; §5.3 Theorem 3; §§6.2–6.5 test generation. Relevant theorem/proof and testing sections read.
- **R3.** Cormac Flanagan, Patrice Godefroid. *Dynamic Partial-Order Reduction for Model Checking Software.* POPL 2005, pp. 110–121. DOI: [10.1145/1040305.1040315](https://doi.org/10.1145/1040305.1040315). [Author PDF](https://patricegodefroid.github.io/public_psfiles/popl2005.pdf). Relevant: §2.2, Definition 1, PDF p. 3, enabledness and commutation.
- **R4.** Gatekeeper project. *Background Information on Mutation.* Official versioned documentation, accessed 2026. [Document](https://open-policy-agent.github.io/gatekeeper/website/docs/mutation-background/), “Mutation Chaining” and “Mutation Recursion.” The externally linked convergence-proof document was not relied on as a verified proof in this brief.
- **R5.** Kyverno project. *kyverno test*, official command reference, and *Chainsaw*, official project. [Test reference](https://kyverno.io/docs/kyverno-cli/reference/kyverno_test/), Synopsis; [Chainsaw repository](https://github.com/kyverno/chainsaw). Accessed 2026. Used to establish existing policy-test and integration-test capabilities, not comparative performance.
- **R6.** Goulnara Arzhantseva, Liviu Păunescu. *Almost commuting permutations are near commuting permutations.* Journal of Functional Analysis 269 (2015), 745–757. DOI: [10.1016/j.jfa.2015.02.013](https://doi.org/10.1016/j.jfa.2015.02.013). [Paper](https://arxiv.org/pdf/1410.2626). Relevant: introduction/Main Theorem, PDF pp. 1–2; permutation domain differs from noninvertible retractions.
- **R7.** Tullio Ceccherini-Silberstein, Michel Coornaert, Xuan Kien Phung. *Strongly sofic monoids, sofic topological entropy, and surjunctivity.* arXiv:2410.17688v2, version dated 7 February 2025 in the arXiv header. [Versioned text](https://arxiv.org/html/2410.17688v2). Relevant: §2.7 and Definition 3.1, bounded-fiber condition. The served HTML also shows a later manuscript date; preserve the retrieved version metadata rather than inferring a new theorem date.

### Implementation and operational sources

- **K1.** Kubernetes SIGs. *kind v0.30.0 release*, 27 August 2025. [Release notes](https://github.com/kubernetes-sigs/kind/releases/tag/v0.30.0). Exact v1.33.4 node digest and AMD64/ARM64 availability listed under prebuilt images.
- **K2.** Kubernetes authors. *Mutating webhook dispatcher*, source tag `v1.33.4` (2025). [dispatcher.go](https://github.com/kubernetes/kubernetes/blob/v1.33.4/staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/mutating/dispatcher.go). `Dispatch`, especially reinvocation eligibility around lines 145–151; `callAttrMutatingHook`, response/patch/decode/default path around lines 314–391. [Webhook reinvocation context](https://github.com/kubernetes/kubernetes/blob/v1.33.4/staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/mutating/reinvocationcontext.go). Source bytes were read during preparation; names/functions are authoritative if line numbers shift.
- **K3.** Kubernetes authors. Source tag `v1.33.4`: [admission/reinvocation.go](https://github.com/kubernetes/kubernetes/blob/v1.33.4/staging/src/k8s.io/apiserver/pkg/admission/reinvocation.go), `Admit`, lines 35–48; [handlers/create.go](https://github.com/kubernetes/kubernetes/blob/v1.33.4/staging/src/k8s.io/apiserver/pkg/endpoints/handlers/create.go), audit placement and later admission; [mutating_webhook_manager.go](https://github.com/kubernetes/kubernetes/blob/v1.33.4/staging/src/k8s.io/apiserver/pkg/admission/configuration/mutating_webhook_manager.go), `getMutatingWebhookConfigurations`, lines 117–147. Also [core v1 defaults](https://github.com/kubernetes/kubernetes/blob/v1.33.4/pkg/apis/core/v1/defaults.go) and [legacy scheme](https://github.com/kubernetes/kubernetes/blob/v1.33.4/pkg/api/legacyscheme/scheme.go) for the helper.
- **K4.** Istio authors. Source tag `1.27.3` (2025): [inject.go](https://github.com/istio/istio/blob/1.27.3/pkg/kube/inject/inject.go), `injectRequired` and `stripPod` (latter around lines 530–555); [webhook.go](https://github.com/istio/istio/blob/1.27.3/pkg/kube/inject/webhook.go), `reapplyOverwrittenContainers`; [standard injection template](https://github.com/istio/istio/blob/1.27.3/manifests/charts/istio-control/istio-discovery/files/injection-template.yaml); [installation values](https://github.com/istio/istio/blob/1.27.3/manifests/charts/istio-control/istio-discovery/values.yaml), `sidecarInjectorWebhook.reinvocationPolicy`.
- **K5.** Istio authors. *Installing the Sidecar*, “Customizing injection,” and *Supported Releases*, historical compatibility table. [Injection documentation](https://istio.io/latest/docs/setup/additional-setup/sidecar-injection/); [support table](https://istio.io/latest/docs/releases/supported-releases/). Accessed 2026. Existing proxy overrides, resource interaction concern, and 1.27's Kubernetes support/EOL facts; not evidence of a current bug in the chosen fixture.
- **K6.** Kyverno authors. *Installation*, historical compatibility table. [Archived official documentation](https://release-1-17-0.kyverno.io/docs/installation/installation/), table including Kyverno 1.15 and Kubernetes 1.30–1.33. Accessed 2026; archived documentation navigation can contain stale version labels, so verify against pinned artifacts.
- **K7.** Gatekeeper authors. *Mutation*, version v3.20.x. [Official documentation](https://open-policy-agent.github.io/gatekeeper/website/docs/v3.20.x/mutation/), Assign, matching, `pathTests`, and conditional defaults. [Pinned install manifest](https://github.com/open-policy-agent/gatekeeper/blob/v3.20.1/deploy/gatekeeper.yaml). Release artifact/schema is the implementation authority.
- **K8.** Kyverno authors. *Mutate Rules*, archived 1.15 documentation. [Official text](https://release-1-15-0.kyverno.io/docs/policy-types/cluster-policy/mutate/), “Add if not present anchor” and “foreach.” [Pinned webhook controller source](https://github.com/kyverno/kyverno/blob/v1.15.2/pkg/controllers/webhook/controller.go), `buildResourceMutatingWebhookRules` around lines 1037–1066. [Pinned release installation manifest](https://github.com/kyverno/kyverno/releases/download/v1.15.2/install.yaml); the similar `config/install.yaml` path is not present in this source tag.
- **K9.** Kubernetes authors. *Dynamic Admission Control* and *Admission Webhook Good Practices*, official documentation, accessed 2026. [Dynamic admission](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/), matching, dry-run, reinvocation, failure policy and mutation auditing; [good practices](https://kubernetes.io/docs/concepts/cluster-administration/admission-webhooks-good-practices/), ordering/idempotence and validation of completed mutation. Moving documentation gives general semantics; the pinned source governs implementation-specific behavior.
- **K10.** Kubernetes authors. *Configure Multiple Schedulers*, official documentation, accessed 2026. [Document](https://kubernetes.io/docs/tasks/extend-kubernetes/configure-multiple-schedulers/), “Verifying that the pods were scheduled using the desired schedulers.” Supports the unscheduled local CREATE technique, not any claim about workload runtime correctness.

**Execution directive:** perform the authorized stages until their evidence gates resolve the study or a genuine blocker is reached. Preserve unfavorable and partial evidence, complete the report after any stop, and judge the outcome by what the data and derivations support.


