# Execution brief: test whether equality classification helps certify a bot difficulty ladder

**Addressed to:** an autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning.  
**Research and planning date:** 1 November 2026.  
**Target:** one MacBook with an M3 Max; inspect the actual machine before running the study.  
**Deliverable from you:** a small runnable diagnostic, independently checked exact certificates, and a research report. This document is the execution instruction, not a claim that the PoC has already been completed.

## 1. Project decision and contribution

### 1.1 Execute a bounded applicability and attribution study

Build one playable, finite game: ordinary 3×3 tic-tac-toe, with the exact rules and three frozen policies specified below. Permit opponents to choose one of those policies randomly at the start of each game and retain it for that game. Determine the complete feasibility frontier for an opponent roster in this family. Then determine whether the supplied **new equality classification** contributes any decision beyond the published determinant bound, its published equality conditions, and a strong elementary analysis of the actual game.

**Primary research question:** In this implemented policy family, does the supplied complete equality classification enable any correct ladder-feasibility decision that the strongest simple exact reference cannot already make?

The motivating product request is five opponents, each scoring at least 60% against **every** earlier opponent. Also test a slightly more demanding five-opponent request and the six/seven-opponent boundary. These are expected-score requirements: wins count 1, draws 1/2, losses 0.

The planning investigation found a decisive candidate limitation. For the selected policies, an exact game probe produced a role-averaged base payoff matrix of the form

\[
K_{ab}=u_a-u_b,\qquad u=(0,1,1).
\]

If you independently reproduce this matrix and its connection to the actual policies, every permitted mixture has a scalar strength coordinate, and the exact criterion is

\[
\boxed{\text{A roster exists}\iff 2\delta(N-1)\le1.}
\tag{F}
\]

Here \(N\ge2\), and the requested score is \(1/2+\delta\), with \(0<\delta\le1/2\). This criterion has rational, executable witnesses and a short impossibility proof. It is strictly stronger than the universal rank-two bound for every \(N\ge3\). Consequently, the equality classification adds **zero additional feasibility decisions in this family** if the game and reduction are confirmed.

**Do not launch an optimizer benchmark after confirming this obstruction.** Complete the independent checks, the exact decision table, the small playable implementation, resource accounting, and report. That is a valid completed research outcome. An expensive tuning race is unnecessary for a mathematical redundancy claim and would invite an unfair comparison with a needlessly weak baseline.

### 1.2 What this can contribute

The smallest defensible result is a reproducible **limitation of this proposed application**, accompanied by a useful exact answer to a concrete roster-design decision. It establishes that structural rank two alone does not make the new equality classification useful for calibration: a plausible episode-mixture architecture can admit a complete, simpler characterization that dominates it.

The telescoping proof of (F), minimax search, rational expectation, and mixture evaluation are elementary. Do not advertise any of them as newly discovered mathematics. Do not claim global historical novelty for this particular negative example. The supplied classification is a separate mathematical contribution whose priority remains provisional. Your study audits its incremental application value; it does not prove a new classification theorem.

The expected decisions, conditional on your independent verification, are:

| Design request | Exact best minimum expected score in this family | Decision |
|---|---:|---|
| Five opponents, at least 60% against every earlier opponent | \(5/8=62.5\%\) | Implementable; supply and verify a roster. |
| Five opponents, at least 65% | \(5/8\) | Impossible within the permitted family. |
| Six opponents, at least 60% | \(3/5=60\%\) | Implementable exactly at the family boundary. |
| Seven opponents, at least 60% | \(7/12\approx58.333\%\) | Impossible within the permitted family. Reduce the count or margin, or change the permitted policies. |

These values are derived planning expectations, not completed PoC findings. The implementation must earn its own verdicts.

### 1.3 Why this scope, and what it gives up

No existing production game was supplied. Tic-tac-toe provides actual legal decisions, terminal outcomes, draws, first-player effects, and an independently solvable finite game. The policy family is specified before your implementation, and its rank condition follows across the whole family. The three policies are ordinary weak, tactical, and optimal choices; the game is not a payoff table manufactured to realize the extremal polygon.

This selection is a **diagnostic proxy**, not a representative sample of tactical games. The planning probe revealed a collapse to one strength coordinate. Preserve that unfavorable finding rather than searching for another prototype to rescue the proposed classification. The revised scope no longer promises a general tuning tool, a positive new application of the equality theorem, or a measured saving of developer time. It retains the original practical decision: whether to continue tuning a precisely defined architecture or change its requirements.

Exclude human learning, enjoyment, perceived difficulty, conceptual depth, universal superiority, generalization to unseen opponents, training new policies, rule balancing, extra-health handicaps, per-move mixture policies, learned embeddings, neural networks, and broad game benchmarks. A mixed weak/strong opponent is permitted calibration; it is not evidence of new strategic concepts.

### 1.4 Evidence threshold and honest stopping rule

The primary quantity is the **number of additional correct decisions attributable specifically to the new classification**, beyond the exact family reference. The practically meaningful threshold for an incremental decision claim is at least one otherwise unresolved permitted request correctly resolved because of a new equality condition. This is a logical threshold, not a statistical effect size. When (F) is established, the reference resolves every valid request, so the additional count is zero over the whole domain, not merely in the finite test table.

An affirmative practical efficiency claim would additionally require a fair end-to-end comparison, including prerequisite discovery, base-game evaluation, rank justification, construction, and independent verification. **This protocol retains no such speedup hypothesis**, because the exact reduction already gives a complete answer. Record actual costs descriptively; do not infer time saved by comparing with a deliberately prolonged search. A valid negative application finding is preferable to an artificial performance result.

If independent execution contradicts the planning matrix or proof, follow the bounded repair rules in Section 8. Do not assume the negative result is true, hard-code outcomes, change policies to restore it, or classify a bug as scientific refutation.

## 2. Evidence status and essential source findings

Use these labels throughout your report: **source theorem**, **source numerical observation**, **planning computation**, **derivation**, **execution assumption**, and **unverified target measurement**. References are ordinary Markdown citations, defined in Section 12.

| Central claim | Source or derivation | Status at handoff | Remaining check |
|---|---|---|---|
| Positive planar minors have minimum max/min ratio \(s_d/s_1\). | Ogranovich, Theorem 2.13 [R1]; definitions below. | Published result; relevant proof and source content inspected. | Audit notation and retain attribution. |
| Extreme orbit values and orbit geometric means are fixed at equality. | Ogranovich, Theorem 3.1 [R1]. | Published necessary conditions. | Include these in attribution, not just the scalar bound. |
| Full equality family is regular, except for the specified alternating family. | Supplied note, Theorem 1, §§3.1–3.4 [A1]. | Proof examined in planning; no essential defect found. Not formally verified or externally peer reviewed. | Check the concise proof in Appendix A; report any defect. |
| The source experiments establish global uniqueness numerically. | No valid source supports this. | **False interpretation.** Their floating-point checks and local searches illustrate the proof. | Never use optimizer success or residuals as uniqueness evidence. |
| The frozen game produces the stated \(G\) and \(K\). | Three separately written small planning checks, including a bitboard retrograde check; Appendix B. | Planning computation in a research container. | Reproduce from game rules with independent implementation on the target. |
| All permitted mixture matchups equal \(t(p)-t(q)\). | Exact coefficient identity plus independent episode selection; §4. | Conditional derivation. | Verify base outcomes, episode semantics, and coefficient identity. |
| Criterion (F) is necessary and sufficient; equality machinery is redundant here. | Telescoping, rational construction, and strict dominance proof; §4. | Mathematical derivation conditional on verified model. | Check exact certificates and independent witnesses. |
| This saves substantial practical tuning time. | No measurement. | Not established and not a retained claim. | Do not claim it from operation counts or hypothetical search budgets. |
| Runtime and memory fit this particular M3 Max laptop. | Small finite-state construction; §7 estimates. | Plausible but target uninspected. | Environment inspection and bounded calibration. |
| The classification's historical completion is globally new. | Limited literature search, §11. | Provisional; no later completion located. | No absolute novelty claim or further priority campaign is required for this diagnostic. |

### 2.1 Mathematical objects and applicability

For an ordered roster \(p_0,\ldots,p_{N-1}\), let

\[
S(j,i)=\Pr(p_j\text{ wins against }p_i)+\tfrac12\Pr(\text{draw}),
\qquad B_{ji}=2S(j,i)-1.
\]

The request is \(B_{ji}\ge\varepsilon\) for every \(j>i\), where \(\varepsilon=2\delta>0\). The probabilities include the fixed role distribution and the declared independent episode lotteries. They are exact model expectations, not empirical sample proportions.

For \(N\ge3\), define

\[
\theta=\pi/N,\quad d=\lfloor N/2\rfloor,\quad s_k=\sin(k\theta),
\quad
\beta_N=\frac{s_1}{s_d}
=\begin{cases}
\sin(\pi/N),&N\text{ even},\\
2\sin(\pi/(2N)),&N\text{ odd}.
\end{cases}
\tag{1}
\]

If the expected signed roster matrix is exactly skew-symmetric, has rank at most two, and all absolute entries are at most \(M\), a positive ladder necessarily satisfies

\[
\varepsilon\le M\beta_N.
\tag{2}
\]

Use \(M=1\) here. Any smaller amplitude must be justified over the entire rejected family; a small observed maximum from a few bots is insufficient. Rank zero cannot support a positive margin. For \(N=2\), use the direct score cap, not a theorem stated for \(N\ge3\).

To see the bridge, a nonzero skew matrix of rank two admits \(B_{ji}=\det(v_i,v_j)\), with \(v_i\in\mathbb R^2\). A qualifying ordered roster makes all these ordered minors positive. If their smallest value is \(m\) and largest is \(L\), then

\[
\frac{M}{\varepsilon}\ge\frac{L}{m}\ge\frac{s_d}{s_1}.
\]

The supplied note's §4 makes this deduction from the known determinant bound [A1, equation (8)]. It does not establish game-specific attainability. The abstract sine payoff matrix mentioned there is not an implemented game certificate.

**Crucial domain distinction:** the rank condition is on the actual numerical signed expectation matrix. Parameter count, sign-rank, low-rank fits, Elo scores, or a two-dimensional visualization do not establish it. Bertrand et al.'s disc rating work uses transformed payoff models; Vadori and Savani's sign-rank result preserves ordering rather than these numerical margins [R4, R5]. Neither is a shortcut to applicability.

### 2.2 What the supplied equality result adds

Set \(c_i=(\cos(i\theta),\sin(i\theta))^\mathsf T\), and \(C=(c_0,\ldots,c_{N-1})\). For strictly positive ordered minors, the supplied classification states that every minimizing frame is

\[
X=GC\quad\text{if }N\text{ is odd or }N\equiv0\pmod4,
\]

or

\[
X=GC_q,\qquad
C_q=(q^{(-1)^i}c_i)_{i=0}^{N-1},\qquad
\sqrt{\cos\theta}\le q\le(\cos\theta)^{-1/2}
\tag{3}
\]

if \(N\equiv2\pmod4\), \(N\ge6\), with \(\det G>0\) [A1, Theorem 1, pp.2–4]. Labels matter; uniqueness is modulo a **common** orientation-preserving linear map, not Euclidean regularity of an arbitrary drawing. At the signed-payoff endpoint \(\varepsilon=M\beta_N\), the normalized ordered payoff array must be \(M\det(c_i,c_j)/s_d\), with the alternating factors in (3) where applicable.

The candidate new content is completeness for the exceptional family, its exact interval, and uniqueness for multiples of four from eight onward. Odd-size uniqueness, the four-column case, and a local alternating family were already in the earlier work [R1, §3]. The proof audit is in Appendix A.

For \(\varepsilon>M\beta_N\), the published bound already rejects the request. For \(\varepsilon<M\beta_N\), exclusion from the equality family alone says nothing about feasibility. At equality, one must still solve the intersection between this family and **realizable game policies**. A numerical fit does not solve that intersection, and a failed nonlinear search does not prove it empty. The supplied Appendix B assumes exactly equal adjacent minors; it is not a general stability or noise theorem.

The selected study avoids an unbounded realizability optimization: it derives the entire realizable payoff family exactly.

## 3. Freeze the game, policies, and permitted configurations

### 3.1 Game rules

Use a 3×3 board with cells numbered 0 through 8 in row-major order. Initially every cell is empty. X moves first, then players alternate placing their mark in an empty cell. A player wins immediately upon completing a row, column, or either main diagonal. Stop at the first win. If the ninth move fills the board without a win, the result is a draw. There are no passes, resignations, time-based losses, extra moves, or other randomness in the rules.

The eight winning triples are `(0,1,2)`, `(3,4,5)`, `(6,7,8)`, `(0,3,6)`, `(1,4,7)`, `(2,5,8)`, `(0,4,8)`, `(2,4,6)`.

Both players observe the complete public board and have the same legal actions and scoring rights. Give the two evaluated bots the two roles with probability 1/2 each, independently of their policy selections. The conclusion is for the empty board and this role average; it is not a guarantee conditional on either role or on another starting board.

### 3.2 Frozen deterministic policies

| ID | Name | Exact action rule |
|---|---|---|
| 0 | `first` | Choose the smallest-index empty cell. |
| 1 | `tactical` | Scan legal cells in ascending order for a move that wins immediately for the current player; take the first. Otherwise scan in ascending order for a cell where the opponent would win immediately if allowed to place there; block the first. Otherwise choose the first legal cell in `(4,0,2,6,8,1,3,5,7)`. |
| 2 | `optimal` | Exact full-depth minimax/negamax for win/draw/loss, followed by the smallest-index move among all optimal legal moves. No faster-win or slower-loss preference. |

For policy 2, use values \(-1,0,1\) from the player-to-move perspective. With X represented by +1 and O by −1, a terminal board with winner \(w\) has value \(p w\) for current player \(p\); a terminal draw has value zero. Otherwise,

\[
V(b,p)=\max_{a\in\mathrm{legal}(b)}[-V(b\text{ with }p\text{ at }a,-p)].
\tag{4}
\]

Memoize by complete board and side to move. Precomputing this table is permitted computation and must be counted in costs. No policy learns from earlier games or sees the opponent's private selected mode. A policy may use the player's role and public board to interpret marks, but the distribution over policy IDs is the same in both roles.

### 3.3 Episode mixtures and the exact implementation domain

Each permitted opponent is a fixed vector

\[
p=(p_0,p_1,p_2),\qquad p_a\ge0,\quad \sum_a p_a=1.
\]

At the start of each game, independently select one policy ID according to \(p\), and keep it for the entire game. Store the selected ID in the episode state. Individual action functions are deterministic and must not resample a mode.

The mathematical analysis covers the entire real simplex \(\Delta_2\). The executable interface accepts nonnegative rational weights encoded as strings, with an exactly unit sum. This is a finite-machine implementation domain inside the real relaxation. Impossibility on the real simplex excludes every permitted rational implementation; the sharp constructions below are rational, so there is no unattainable-real-number gap for any request supported by the interface.

Sample rational weights using an integer denominator, cumulative integer weights, and a uniform integer draw. Do not convert to floating probabilities. The exact expected-value evaluator sums over the independent role and mode choices; it does not treat a finite pseudorandom stream as proof of the expectation. Use reproducible, separately labeled RNG streams for interactive demonstrations, and test that each bot samples exactly once. Distinct seeds alone are not a mathematical proof of independence. Product-event enumeration verifies the expectation under the declared model; it does not by itself verify that the playable program implements separate draws.

Make episode construction accept three explicit injected inputs: the role bit and one uniform-integer mode draw for each named participant. Exhaust their Cartesian product for the small witness denominators, and check that each selected mode depends only on that participant's own draw and weights, that role assignment uses only the role bit, and that both modes persist unchanged. Test the RNG adapter's provider calls separately to catch accidental reuse of one draw or role-conditioned selection. In this additive family, some role-independent correlations between modes preserve the expected payoff, so aggregate score agreement would not expose that bug. Such correlation remains outside the frozen protocol even when its marginal score happens to agree.

The only tunable bot parameter is this probability vector. No rule, starting-state, role-weight, tactical-priority, tie-break, or computation-policy change belongs to the certified family. Changing one creates a new scope and invalidates reuse of old certificates until revalidated.

### 3.4 Scope identifiers

Create a canonical manifest containing rule version, winning triples, start state, scoring, role distribution, policy definitions and source hashes, episode-selection semantics, rational parsing conventions, and certificate version. Hash its canonical UTF-8 JSON. Include this `scope_hash` in every certificate, result, roster, and report.

A valid certificate also records the observed base-game matrix and the independent-check result. A matching hash is an integrity aid, not a substitute for proving the model. Do not silently transfer a certificate to a changed game, policy, or evaluation distribution.

## 4. Exact computational mechanism and proofs

### 4.1 Obtain the matrix from actual legal games

Let \(G_{ab}\in\{-1,0,1\}\) be the signed outcome **to X** when policy \(a\) plays X and policy \(b\) plays O from the empty board. Execute all nine games, recording every board, action, side to move, selected policy ID, and terminal reason. Never populate the matrix directly from the expected fixture.

For fixed policy \(a\) versus fixed policy \(b\), fair role assignment gives signed expected payoff to \(a\)

\[
K_{ab}=\frac{G_{ab}-G_{ba}}2.
\tag{5}
\]

The planning fixture is

\[
G=\begin{pmatrix}1&-1&-1\\1&0&0\\1&0&0\end{pmatrix},
\qquad
K=\begin{pmatrix}0&-1&-1\\1&0&0\\1&0&0\end{pmatrix}.
\tag{6}
\]

In particular, \(G_{00}=1\) is compatible with \(K_{00}=0\): the first player wins that deterministic self-match, but either named participant receives the first-player role with probability 1/2. Do not incorrectly require \(G\) itself to be skew-symmetric or to have zero diagonal.

For arbitrary mixtures, exact expectation gives

\[
\mathcal A(p,q)=\sum_{a,b}p_aq_bK_{ab}=p^\mathsf TKq,
\qquad S(p,q)=\frac{1+\mathcal A(p,q)}2.
\tag{7}
\]

Equivalently, enumerate the 18 role/mode events with probabilities \(p_aq_b/2\). This independently verifies orientation and draw scoring. The base outcomes must come from the game engine, not a proposed abstract payoff array.

### 4.2 Establish rank over the entire family

For \(P=(p_0,\ldots,p_{N-1})\in\mathbb R^{3\times N}\), the roster matrix is \(B=P^\mathsf TKP\). Every real 3×3 skew-symmetric matrix has rank zero or two, so this already proves \(\operatorname{rank}B\le2\) for every roster and every permitted mixture. This argument relies on exact role averaging and episode mixtures, not on observing a few low singular values.

The selected game has additional structure. Check every coefficient, using exact arithmetic, against

\[
K=u\mathbf1^\mathsf T-\mathbf1u^\mathsf T,\qquad u=(0,1,1)^\mathsf T.
\tag{8}
\]

Then with \(t(p)=u^\mathsf Tp=p_1+p_2=1-p_0\),

\[
\mathcal A(p,q)=t(p)-t(q),\quad 0\le t(p)\le1,
\qquad B=t\mathbf1^\mathsf T-\mathbf1t^\mathsf T.
\tag{9}
\]

This is an exact polynomial identity on the whole simplex. Random tests of mixture vectors are useful debugging but are not the proof. The two stronger base policies are payoff-equivalent against this family under this evaluation; their actions differ, and nothing here establishes equivalence against all other opponents.

### 4.3 Necessity, sufficiency, and the executable witness

If a roster qualifies, its adjacent pairs necessarily satisfy

\[
t_{i+1}-t_i\ge\varepsilon.
\]

Summing gives

\[
(N-1)\varepsilon\le t_{N-1}-t_0\le1.
\tag{10}
\]

If \((N-1)\varepsilon>1\), this is a complete impossibility certificate over the full real simplex. It is not a failed search or a grid result.

Conversely, choose

\[
p_i=\left(1-\frac{i}{N-1},\ 0,\ \frac{i}{N-1}\right),
\qquad 0\le i<N.
\tag{11}
\]

For every \(j>i\),

\[
\mathcal A(p_j,p_i)=\frac{j-i}{N-1}\ge\frac1{N-1}.
\tag{12}
\]

Thus (11) meets any permitted request satisfying (F). It can be implemented exactly as `draw an integer r uniformly from 0,...,N-2; choose optimal if r < i, otherwise first`. No real-valued oracle, continuous optimization, discretization, convergence condition, or numerical tolerance is involved.

Adjacent conditions are sufficient **because of the proved difference form**. Continue to verify every pair in each returned witness; do not generalize this shortcut to cyclic games or arbitrary payoff matrices.

The sharp attainable minimum score is \(1/2+1/[2(N-1)]\). For a fixed \(\delta\), the maximum permitted count is \(1+\lfloor1/(2\delta)\rfloor\). These are architecture conclusions, not limits on all possible tic-tac-toe bots.

### 4.4 Prove the equality classification is redundant here

For even \(N\ge4\), strict concavity of sine above its chord on \([0,\pi/2]\) gives \(\sin(\pi/N)>2/N\). For odd \(N\ge3\), it gives \(2\sin(\pi/(2N))>2/N\). Therefore

\[
\beta_N>\frac2N>\frac1{N-1},\qquad N\ge3.
\tag{13}
\]

Every achievable nontrivial roster in this family lies strictly below the universal equality boundary. The published bound is weaker than the exact scalar criterion, and the new classification cannot add a decision to a necessary-and-sufficient criterion.

There is also an attribution check using the **old** equality conditions: Ogranovich's cyclic adjacent orbit contains the \(N-1\) consecutive differences and the endpoint difference. At universal equality, all would equal the same positive minimum \(c\). In the scalar family, the endpoint difference is the sum of the consecutive differences, so \(c=(N-1)c\), impossible for \(N\ge3\). Thus even this endpoint exclusion does not need the new classification.

Do not claim that this proves equality classification is useless for every game or every computational task. It proves zero additional feasibility information for the specified family. The failure mechanism is a stronger realizability restriction, not an error in the classification theorem.

### 4.5 Required pseudocode

```text
BUILD_AND_CHECK_MODEL(protocol):
    validate frozen rules, starting state, policies and episode semantics
    build production game engine and memoized optimal policy
    for a in [0,1,2]:
        for b in [0,1,2]:
            trace[a,b] = play_actual_game(X_policy=a, O_policy=b)
            G[a,b] = terminal_signed_result_to_X(trace[a,b])
    independently replay traces and solve game with the reference implementation
    require exact agreement on legal actions, values, optimal tie breaks and G
    K[a,b] = Fraction(G[a,b] - G[b,a], 2)
    require K == -transpose(K), entries in [-1,1]
    verify every coefficient of K[a,b] == u[a] - u[b], u=(0,1,1)
    require actual G and K agree with the planning fixture
    write model, full provenance, reference checks, and scope hash
    return VERIFIED_MODEL

ASSESS(N, target_score, VERIFIED_MODEL):
    parse N as integer and target_score exactly from a rational/decimal string
    require 2 <= N <= 64 and 1/2 < target_score <= 1
    epsilon = 2*target_score - 1
    if model is missing, changed, or not independently verified:
        return UNRESOLVED(reason), never a mathematical certificate
    if (N-1)*epsilon > 1:
        return IMPOSSIBLE_CERTIFICATE(
            span=1, required_span=(N-1)*epsilon,
            contradiction=(N-1)*epsilon-1, scope_hash, proof='scalar_span_v1')
    roster[i] = (1-i/(N-1), 0, i/(N-1)), computed as rationals
    for every i<j:
        independently enumerate role and mode events from actual base-game outcomes
        require exact expected_score(roster[j],roster[i]) >= target_score
    verify each rational lottery implements its declared mixture
    return ACHIEVABLE_CERTIFICATE(roster, all_pair_scores, scope_hash)

VERIFY_CERTIFICATE(certificate, independent_reference):
    reject malformed inputs, invalid rational weights, or mismatched scope hashes
    regenerate base outcomes and check the model's coefficient identity
    for an impossible result:
        independently verify the span inequality and its positive violation
    for an achievable result:
        verify legal implementations and every exact pair score
    never trust a recorded 'passed' field or a rounded printed score
```

The executable cap of 64 opponents is an output-size and accidental-input guard, not a mathematical limit. State the theorem for every finite \(N\ge2\); do not allocate quadratic output for an unbounded user-supplied integer. Invalid input produces a clear input error, not an impossibility verdict.

### 4.6 Failure behavior and numerical conventions

Use Python integers and `fractions.Fraction` for probabilities, scores, expectations, residuals, and certificate inequalities. Parse `"3/5"` or `"0.6"` directly; never construct the requirement from a binary float. Do not add tolerance to the ≥ comparison. Python's official documentation explains the exact-rational representation and the distinction between decimal strings and floating inputs [R8].

Ordinary `math.sin` values are allowed for labeled display only. The planning calculation returned `0.49999999999999994` for a quantity mathematically equal to 1/2, illustrating why floating equality is unsuitable. No floating-point matrix rank, SVD threshold, or approximate equality can authorize a certificate.

If (8) fails after valid independent game evaluation, the specialized solver is inapplicable. Preserve the exact matrix and traces; follow the bounded fallback in Section 8. Do not silently fit a potential, round a small residual to zero, invoke the equality theorem below its endpoint, or replace the game.

## 5. References, baselines, and attribution controls

### 5.1 Strong comparisons for the claim actually retained

| Method or control | Information and work allowed | Valid output | Role in this study |
|---|---|---|---|
| Actual-game exact evaluation | Nine deterministic base games, minimax construction, cached outcomes, exact role/mode summation. | Exact score of a proposed roster. | Simple credible reference and independent witness check. |
| Scalar family analysis | The same exact \(G,K\), coefficient identity (8), telescoping and construction. | Necessary-and-sufficient feasibility for every request. | Strongest simple baseline; primary implemented decision mechanism. |
| Published rank-two bound | Same game model, exact rank argument, amplitude \(M=1\), equation (2). | Impossibility when the requested margin exceeds the universal cap; otherwise unresolved. | Prior-art information comparison. Passing a necessary bound never means achievable. |
| Published bound plus old equality conditions | Additionally Ogranovich Theorem 3.1. | Endpoint exclusions that those conditions actually prove. | Required attribution control; prevents old results being credited to the new classification. |
| Published results plus complete classification | Additionally (3), with actual policy realizability required. | Additional endpoint decisions only if a valid new argument supplies them. | Target mechanism. Here prove analytical redundancy; do not build a generic intersection solver. |

Give every computational comparison the same cached base-game information. Re-running stochastic games at every optimizer step while the certificate method receives the exact matrix would primarily compare caching and information access. That would not isolate the new theorem.

Hernandez et al. study response-graph targets with simulation-based parameter search and a black-box optimizer [R2]. Their task motivates the unresolved-search-versus-impossible distinction, but reproducing their larger system is unnecessary after the exact obstruction. Liu and Marschner study a different inverse balancing problem with permitted handicaps [R6]. Ratings, equilibrium solvers, and those handicaps are not substitutes for the all-earlier numerical-margin decision.

Do not import published runtime numbers, train weak comparison bots, add artificial computation to the game, or use a lower-quality baseline merely to obtain a positive timing result. No Optuna, SLSQP, RL, or global nonlinear optimizer is required by this selected study.

### 5.2 How to report the old bound without numerical miscertification

For the locked queries, record the exact symbolic \(\beta_N\), a decimal labeled approximate, and whether the prior bound alone can reject the request. The attribution comparison is mathematical; it is not a benchmark of two implementations of sine.

The relevant strict comparisons can be verified without a general interval-arithmetic dependency:

- For \(N\le8\) in the table, use explicit familiar trigonometric values or the lower bound \(\beta_N>2/N\) where sufficient. In particular \(\beta_5=(\sqrt5-1)/2>3/10\), and \(\beta_6=1/2\).
- At \(N=16\), \(\beta_{16}=\sin(\pi/16)<\pi/16<11/56<1/5\), using \(\pi<22/7\). Thus the prior bound certifies impossibility at 60%.
- At \(N=15\), \(\beta_{15}=2\sin(\pi/30)>2[x-x^3/6]>1/5\), with \(x=31/300\), using \(\pi>31/10\), sine monotonicity on this interval, and the alternating sine-series lower bound. Check the last inequality with fractions.
- If desired, verify the two coarse bounds on \(\pi\) from Machin's identity \(\pi=16\arctan(1/5)-4\arctan(1/239)\), bounding each arctangent by consecutive alternating partial sums. Two and three terms for \(1/5\), and one and two for \(1/239\), suffice. This is a bounded exact check, not a new numerical research task.

For arbitrary CLI requests outside this comparison table, the exact scalar solver remains authoritative. Do not emit a universal-bound certificate from an unvalidated floating comparison; return a display-only value or omit that auxiliary status. This does not weaken the complete family decision.

### 5.3 Why no optimizer timing experiment is required

Once (F) and (13) are verified, the additional-decision question is settled analytically over an infinite request domain. There is no statistical uncertainty about a computed exact inequality. Running many seeds cannot strengthen this proof. A hypothetical optimizer's eventual failure also cannot add an impossibility certificate.

Measure the cost of the actual diagnostic, including preprocessing and checking, to establish local feasibility. The report must distinguish **a certificate that permits stopping futile tuning** from **an empirically demonstrated reduction in real development effort**. Only the first is established by this protocol; its mathematics is not due to the new equality classification.

## 6. Minimal experiment and correctness protocol

### 6.1 Independent correctness before interpretation

Implement the production engine with an immutable tuple board and memoized negamax, or an equally simple representation. Implement the reference separately, preferably with two bitboards, a separately constructed reachable-state graph, and bottom-up values from X's perspective. The reference must not import production terminal detection, transitions, policy action selection, or minimax code. Sharing configuration data and parsing conventions is acceptable; sharing the mechanism being checked is not.

Enumerate every legally reachable state from the empty board, stopping expansion at wins or full boards. Compare the engines' legal actions, terminal status, winner, and optimal value/action on all applicable reachable states. The planning count was **5,478 distinct reachable boards including the empty board**. This is an expected checksum, not the sole correctness oracle. Count legal continuations and check board-count and turn invariants independently. Report the actual count; never confuse a minimax cache that did not query the empty board with the full reachable-state census.

The reference's bottom-up update is terminal X payoff at leaves, maximum child value on X's turn, minimum on O's turn. Convert perspectives explicitly: \(V_{\mathrm{production}}(b,p)=p\,V_{\mathrm{reference},X}(b)\). Compare the chosen smallest-index optimal action directly, not merely an equal final value. The empty-board minimax value should be zero. Test immediate wins, mandatory blocks, a full-board draw, a win before move nine, and rejection of moves after termination.

Reproduce all nine trajectories in Appendix B. Agreement must include legality, move sequence, termination point, and payoff orientation. These finite traces are the actual-game grounding for the base matrix.

For exact mixture verification, enumerate independent mode pairs and the two roles. Use win/draw/loss scores directly in the reference and compare with \((1+p^\mathsf TKq)/2\). Do not make the reference call the scalar closed form. Check every pair of each returned roster. For the rational sampling interface, exhaust the possible integer draws for the frontier witnesses and a mixture such as `(1/2,1/3,1/6)`; check zero and one-hot weights and exact sample-count frequencies. Exercise the injected episode-construction inputs and provider-call checks from §3.3 as well. This finite enumeration tests the mapping from uniform integers to modes and correct composition, not the statistical quality or independence of a seeded PRNG.

### 6.2 Locked decision table

Run these eleven requests after freezing the protocol and after the independent correctness gate. The final column is a planning expectation to verify, not input data for the solver.

| ID | \(N\) | Target expected score | Purpose | Expected exact family verdict |
|---|---:|---:|---|---|
| Q01 | 5 | `3/5` | Original representative request. | Achievable. |
| Q02 | 5 | `13/20` | Impossible request with the original roster count. | Impossible. |
| Q03 | 6 | `3/5` | Exact family boundary. | Achievable, equality. |
| Q04 | 7 | `3/5` | Smallest impossible count at 60%. | Impossible. |
| Q05 | 6 | `5999/10000` | Just below the exact boundary. | Achievable. |
| Q06 | 6 | `6001/10000` | Just above it; catches tolerant comparisons. | Impossible. |
| Q07 | 8 | `3/5` | Size with newly classified uniqueness, away from universal endpoint. | Impossible by family criterion; no new-classification attribution. |
| Q08 | 15 | `3/5` | Prior universal bound still does not reject. | Impossible by family criterion. |
| Q09 | 16 | `3/5` | Positive control for published-bound rejection. | Impossible by both bounds. |
| Q10 | 2 | `1` | Direct two-opponent boundary; outside the \(N\ge3\) classification. | Achievable. |
| Q11 | 6 | `3/4` | Exact universal endpoint \(\varepsilon=1/2\). | Impossible; old equality conditions already exclude the family. |

The 0.0001 score offsets in Q05/Q06 are deliberately small exact arithmetic checks, not a proposed human-perception threshold. Do not estimate them by simulation. Do not increase case counts to obtain significant-looking statistics.

For each row, report exact target, exact family frontier, verdict, witness or contradiction, all-pair minimum when applicable, theorem applicability, the old bound's status, and the source of the decisive argument. The unique evaluated game/family is the scientific unit. Eleven requests are eleven logical checks, not eleven independently sampled game designs.

### 6.3 Small controls for plausible errors

Keep these controls small and label any abstract matrices as unit fixtures, never as practical game evidence.

| Control | Required observation | Error it detects |
|---|---|---|
| `optimal` versus itself | Draw in each role; expected score 1/2. | Dropping draws or renormalizing only decisive games. |
| `first` versus itself | \(G_{00}=1\) but role-averaged signed payoff zero. | Wrong role averaging or treating the first-role table as skew. |
| Same scalar strength with different splits across policies 1 and 2 | Equal payoff against this family, despite different policy traces. | Confusing behavior labels with a strict strength ordering. |
| Reverse a known qualifying roster | At least one all-earlier condition fails. | Sign, index, or ordering errors. |
| Cyclic three-policy skew fixture | A displayed cycle fails any three-item all-earlier ordering. | Treating pairwise wins or rank two as sufficient for a ladder. |
| Four-policy skew fixture with all lower-triangular entries +1 and upper entries −1 | Exact rank four; reject use of the rank-two theorem. | Inferring numerical rank from transitivity or a two-dimensional rating. |
| Alter a policy tie break, role distribution, or starting board in a copy of the manifest | Old certificate rejected due to changed scope. | Reusing conclusions after conditions change. Do not require the new game to have a different verdict. |
| Make a known-invalid certificate mutation: a negative witness weight, a nonunit probability sum, or an impossibility claim with required span no greater than the allowed span | Independent verifier rejects it. | Trusting producer success flags or unchecked files. A different but still valid witness must not be rejected merely because it differs from the producer's preferred one. |

For the cyclic fixture, use lower/upper values corresponding to rock-paper-scissors, scaled by 1/2, and enumerate its six permutations. This only proves failure for that finite fixture, not impossibility for all its mixtures. For the rank-four fixture, the 4×4 Pfaffian is nonzero (its determinant is 1); exact elimination or the Pfaffian verifies rank four. You may implement a tiny exact rank helper for these guards, but no general game-analysis framework is needed.

An additional source-audit fixture is available in Appendix A: a rational, nonregular six-column equality array. It guards against the false claim that every equality array is regular or irrational. Use it only if implementing an equality-pattern checker; such a checker is **not required** after the analytic stop.

### 6.4 Exploration, lock, and deviations

The planning probes and expected table are disclosed development evidence. This is an exact diagnostic, not a held-out prediction exercise. Independent confirmation comes from separately implemented semantics and complete finite checks, not from pretending that known outcomes are blind test data.

During development, use three representative games and Q01/Q03/Q04 as smoke cases. Once rules, policies, parsers, certificate schema, source hashes, and resource settings pass, write a frozen protocol and commit or hash it. The locked run regenerates the full game/reference checks and all eleven queries from clean result paths.

Permit at most two documented repair cycles after the lock, for concrete implementation defects, within the cumulative repair budget. Retain failed outputs and label superseded runs. A change to the game family, question, outcome definition, baseline, or metric is a scientific protocol change, not an ordinary repair. It is outside this study's authorization to expand the project; preserve the evidence and report the resulting limitation.

No statistical hypothesis tests, confidence intervals on exact outcomes, random training seeds, or repeated searches are required. Optional interactive games are demonstrations and cannot validate expected-score inequalities. If only sampled outcomes are available because exact execution failed, the planned certificate study is not complete; do not silently replace it with Monte Carlo.

## 7. Concrete feasibility and measurement on the Mac

### 7.1 Inspect before allocating

Record OS version, actual chip identification, CPU architecture, Python executable and version, physical RAM, available disk, power mode, and whether the process is native ARM64 or translated. Useful local commands include `sw_vers`, `uname -m`, `sysctl -n hw.memsize`, and Python's `platform` and `shutil.disk_usage`. Query chip/CPU information with the available system tools; do not infer RAM, free disk, or interpreter architecture from the M3 Max name. Avoid collecting serial numbers or unrelated personal machine data.

Use a single CPU process. The core requires only Python 3.11 or later and its standard library (`fractions`, `functools`, `itertools`, `json`, `hashlib`, `time`, `resource`, `subprocess`, `unittest`). Prefer an already installed supported native Python; record its exact version. No CUDA, GPU, MPS, Metal, paid compute, remote workload, proprietary data, browser service, or game engine dependency is needed. No NumPy/SciPy dependency is required for the certificates. The source's numerical scripts have additional dependencies but are not a dependency of this diagnostic.

If the target machine is unavailable, complete source work, the mathematical derivation, and code preparation that do not require it. Do not claim target validation or launch a substitute remote benchmark. Record exactly what is blocked. The planning computations in this brief were performed elsewhere and establish no M3 Max timings.

### 7.2 Expected scale and calibration

The raw board encoding space is \(3^9=19,683\) boards; legal reachable boards are a much smaller finite subset. A memoized or retrograde solver visits each reachable board once and examines at most nine moves per nonterminal board. The matchup model needs nine deterministic games. Exact roster verification uses \(O(N^2)\) pairs and at most 18 base events per pair. No massive payoff tensor, factorial roster enumeration, or continuous search is required.

**Unmeasured planning estimate:** a correct standard-library implementation should use well below 1 GiB peak process memory and run the core exact checks in seconds to a few minutes on the target. Setup, writing, independent implementation, and analysis take longer than the computation. Treat the measured calibration, not this estimate, as authoritative.

First time one cold optimal-policy table build, one full reference state census, and three representative base games. Measure actual time, state count, and peak RSS. If either solver takes more than 60 seconds, or more than 100,000 distinct state keys appear, investigate memoization, terminal handling, or state representation before proceeding. These are diagnostic guards based on the finite state-space bound, not performance competition thresholds.

Permit one bounded repair of a calibration defect. Do not make the board smaller or remove independent correctness checks to fit the budget: that would change or invalidate the experiment. If the declared 3×3 workload remains infeasible, report practical infeasibility or an implementation failure as justified by the evidence.

### 7.3 Cumulative ceilings

All execution, exploratory calls, failures, retries, repairs, and fallback diagnostics count cumulatively. The allocation is an upper bound, not a target to consume.

| Resource | Hard study ceiling | Rationale and action |
|---|---:|---|
| Cumulative computational wall time | 60 minutes | Ample for two tiny solvers and exact checking; includes development runs and report generation. Stop new experiments at 50 minutes and preserve the final 10 for diagnosis and output. |
| Individual workload subprocess | 120 seconds | A bounded finite game should not require an unattended long run. Timeout is a recorded failure; investigate once within repairs. |
| Peak memory of a workload process | The smaller of 1 GiB and 10% of measured physical RAM | The state space is tiny. Avoid large dense intermediates; stop a process that breaches the limit. Record actual RSS, not just Python allocations. |
| Generated project/results disk | 250 MiB, excluding an existing Python installation | Traces, tables, source copies, and reports are small. Require at least 500 MiB free before starting. Do not generate videos or huge logs. |
| Full locked runs | One, plus at most two justified repair reruns | Deterministic correctness, not seed averaging. Retain every run. |
| Cold full-pipeline cost captures | At most three, including the locked run | Descriptive local feasibility only. Skip repetitions if the analytic outcome and resource gate are already clear. |
| Development/calibration invocations | At most 30 workload invocations | Enough to repair ordinary mistakes; no indefinite rescue campaign. |
| Optimizer trials / trained policies / alternative games | Zero | They cannot improve the retained analytic attribution conclusion. |
| Total implementation, setup, and analysis effort | Estimate 4–8 active hours; cap 10 active hours | Separate from compute. Stop feature expansion early enough to write a complete report. Hosted coding-agent reasoning is not laptop experimental compute. |

Maintain one study-level append-only ledger at `results/study_ledger.jsonl`, outside individual run directories, with stage, command, start/end time, elapsed time, status, exit code, timeout flag, peak RSS and units, and cumulative totals. A killed or failed run remains in the ledger. Stage changes, fresh result directories, and repaired runs do not reset budgets. Charge each top-level workload's elapsed wall time once; nested component timings explain that total and are not added again. Peak memory is a maximum, not a sum across sequential runs.

Use a small standard-library supervisor (`python -m ladder.guard`) for every development test, smoke run, verification, report generation, and locked workload. It appends to this single ledger, enforces remaining time and invocation caps, and records failures even if the child crashes. Before this supervisor exists, keep equivalent manual entries for any tiny development executions and import them once. Setup/download and agent implementation effort have their separate active-effort accounting; do not disguise experimental computation as setup. The workload child may report component timers but must not re-charge its parent invocation.

### 7.4 Timing boundaries and memory accounting

Record one cold end-to-end execution that includes imports, game/policy construction, base games, exact model/rank checks, requests, independent verification, and writing results. Separately time these components with a monotonic timer so preprocessing and checking are visible. If an optimal-policy table is reused, label the query timing warm and include its construction in total cost. Never compare a warm certificate query against a cold reference evaluation.

Use fresh subprocesses for any repeated cold capture. Report each observed value and a range or median if repeated; do not claim statistically established speedups from this small descriptive sample. Record background load and power status if readily available. There is no accelerator to synchronize.

On macOS, use `/usr/bin/time -l` and/or `resource.getrusage` to capture peak RSS, verifying the units against the local documentation. Keep the raw value, unit, and byte-normalized value; do not apply Linux's units automatically. An external lightweight runner can poll RSS and enforce the ceiling while waiting with a timeout. If a measurement facility is unavailable, state the missing measurement and avoid the corresponding resource claim. Do not substitute `tracemalloc` allocation counts for total resident memory.

Amortized costs are optional descriptive arithmetic only: if reported, show total setup plus verification cost and the assumed number of future requests. The primary conclusions do not need amortization or a break-even estimate.

## 8. Evidence gates and autonomous decisions

Proceed without requesting approval for routine reversible coding choices or the bounded checks specified here. The stages can be implemented as ordinary script subcommands. Do not build later infrastructure before its prerequisites pass.

| Stage and question | Minimum work and evidence | Compute allocation within the 60-minute total | Decision and next action |
|---|---|---:|---|
| **0. Constructive feasibility:** are inputs, target, and exact construction available? | Inspect target; read this brief and available original sources; record environment and source provenance; check the family definition and no-oracle construction. | 3 min | Continue if execution is available. If target is absent, prepare unaffected artifacts and report target validation blocked. If a source is inaccessible, use the self-contained derivations but disclose the unperformed source cross-check. |
| **1. Tiny correctness:** do two implementations agree on actual gameplay? | Production engine, three policies, independent finite-state reference, nine games and traces, draw and role controls. | 12 min | Continue only on exact agreement. One repair cycle for concrete bugs; preserve counterexamples. A valid semantic disagreement not resolved in the remaining repair allowance blocks certification. |
| **2. Resource calibration:** is the intended scale affordable? | Cold state census/table build, representative games, state/time/RSS measurements, projected total remaining cost. | 5 min | Continue if full checks fit remaining caps. Repair duplicate expansion or equivalent defects once. If the unchanged game cannot fit, report infeasibility; do not shrink it. |
| **3. Mechanism pilot:** does the exact realizable family collapse as predicted? | Derive \(K\), verify every coefficient of (8), demonstrate one rational five-level witness and the seven-level contradiction. Audit the dominance proof and old equality-condition exclusion. | 8 min | If verified, accept the analytic-stop route: build no optimizer or generic equality solver. Continue only to locked verification and report. If it fails, use the bounded diagnostic fallback below. |
| **4. Locked comparison:** are the certificates and attribution reproducible? | Freeze protocol; run the eleven requests, independent all-pair verification, certificate corruption/scope controls, and complete attribution table. | 10 min | Require zero invalid certificates and no unexplained disagreements. A concrete defect permits a repair rerun within the shared allowance; no change of family or metric. |
| **5. Diagnosis and report:** what decision follows? | Regenerate essential tables, report resource totals/failures, prove the all-\(N\) conclusion, package code and usage, document remaining limitations. | 12 min | Deliver the appropriate outcome, including an informative negative if redundancy is confirmed. No optional benchmarks after the question is answered. |
| **Shared repair reserve** | At most two post-lock repair cycles total; earlier repairs also debit this time. | 10 min | Once exhausted, preserve evidence and report invalid/inconclusive/infeasible as appropriate. Never silently retry until a desired result appears. |

The per-stage allowances sum to 60 minutes including the repair reserve. Unused allocations may be reassigned to a specific unresolved correctness issue, but never beyond the cumulative cap, invocation cap, or the final ten-minute reporting reserve.

**Bounded fallback if the predicted reduction fails:** First compare the earliest divergent move or state against Appendix B, the written policy rule, and both engines. If a correct implementation of the frozen rules genuinely gives a different matrix, preserve it and retract the planning fixture. Within the remaining reserve, test whether its exact matrix still has an additive potential by setting \(u_a=K_{a0}\) and checking \(K_{ab}=u_a-u_b\) for every pair.

If this identity holds, set \(R=\max_a u_a-\min_a u_a\), and verify \(0\le R\le1\) from the exact payoff matrix. The corrected exact criterion is \((N-1)\varepsilon\le R\). For \(R>0\), let `lo` and `hi` be the smallest-index minimizer and maximizer of \(u\), respectively, and use \(p_i=(1-i/(N-1))e_{\mathrm{lo}}+(i/(N-1))e_{\mathrm{hi}}\). Its minimum signed margin is exactly \(R/(N-1)\). If \(R=0\), every positive-margin request is impossible and no witness or division by \(R\) is allowed. The global amplitude is \(M=R\); for positive \(R\), the same strict-dominance argument applies after scaling.

This explicitly authorized correction branch replaces the planning-fixture and fixed-\(u\) assertions in §4.5 with the independently observed matrix and verified potential identity; it does not authorize a change of game or policies. Store the actual potential, span, and extremal policy IDs in the model, and make assessment and verification use them rather than hard-coded span 1 or policies 0 and 2. Correcting an expectation under independently confirmed unchanged written semantics is an evidence correction; changing those semantics remains outside this fallback. Record the reason, corrected model version, actual span, and affected expectations; create a fresh protocol lock and verification run within the remaining caps, consuming one permitted repair rerun. Do not silently keep the original (F) if \(R\ne1\).

If the matrix is nonadditive, the planned complete closed-form diagnostic has failed. Report the exact matrix, the verified rank-at-most-two scope, any verified proposed rosters, and any impossibility certificates still supplied by the published bound with established premises. Leave remaining requests and incremental classification value unresolved; do not claim a complete frontier. Do not begin a global optimizer, select another game, or invent a continuum-impossibility certificate from a grid.

If the supplied equality proof develops a material unresolved defect, the scalar game result can still be completed because it does not depend on that classification. Report the source defect separately and narrow any statement about the classification. Never discard unaffected valid work.

## 9. Repository, interfaces, and concrete execution commands

### 9.1 Minimal repository

Create a small repository, for example `ladder-feasibility-diagnostic/`, with these responsibilities. Combine tiny modules if that improves clarity; preserve the independence of the reference.

| Path | Responsibility |
|---|---|
| `README.md` | Setup, exact scope, main commands, result interpretation, playable demonstration. |
| `config/protocol.json` | Frozen rules/policies, eleven requests, caps, demonstration seeds, schema versions. |
| `ladder/__main__.py` | CLI for inspect, smoke, run, verify, report, and play. |
| `ladder/game.py` | Production board, legal moves, transition, terminal outcome, trace generation. |
| `ladder/policies.py` | The three fixed policies and exact episode lottery interface. |
| `ladder/model.py` | Actual-game \(G\), rational \(K\), exact mixture expectation, coefficient/rank proofs represented as checks. |
| `ladder/certificates.py` | Request validation, criterion (F), witnesses, impossibility records, scope binding. |
| `ladder/run.py` | Stage orchestration, timeouts, cumulative ledger, resource capture. |
| `ladder/guard.py` | Small top-level subprocess supervisor; single study ledger, caps, and crash/timeout recording. It may share simple bookkeeping with `run.py`. |
| `ladder/report.py` | Deterministic tables and Markdown report from saved results. |
| `reference/retrograde.py` | Independent game implementation, state enumeration, values/actions, mixture and certificate checking. |
| `tests/` | Meaningful correctness and corruption controls from §6, using `unittest`. |
| `inputs/` | Optional copies of supplied source files and a source manifest; never overwritten by experiments. |
| `results/<run_id>/` | Frozen configuration, environment, provenance, traces, checks, certificates, ledger, and report. |

No service, database, dashboard, plugin system, graphical game engine, dependency-heavy test framework, or distributed orchestration is needed. A terminal interface accepting cell numbers is sufficient to make the real game playable. A human can choose role and opponent level for a demonstration; those sessions are outside the certified evaluation distribution unless using its role coin, and do not alter the scientific results.

### 9.2 Core interfaces to implement

Use explicit typed functions or equivalent simple interfaces:

```python
terminal(board) -> int | None              # X winner +1, O winner -1, draw 0
legal_moves(board) -> tuple[int, ...]       # ascending
apply_move(board, player, cell) -> board
choose_action(board, player, policy_id) -> int
play_game(x_policy, o_policy) -> GameTrace
sample_mode_from_integer(weights, draw) -> int
start_episode(p, q, role_bit, p_draw, q_draw) -> Episode
evaluate_base_games() -> BaseGameModel      # includes G, K, traces and checks
expected_score(p, q, model) -> Fraction
assess_request(n, target_score, model) -> Assessment
verify_assessment(assessment, reference_model) -> VerificationResult
```

For matchup outputs \(K\), \(B\), and \(\mathcal A\), return the row participant's signed payoff consistently. Terminal game outcomes and \(G\) are explicitly from X's perspective, while \(V\) in (4) is from the player-to-move perspective; keep these interfaces distinct. Persist fractions as reduced `"numerator/denominator"` strings, or as an explicitly documented numerator/denominator object; never a rounded JSON float. Array rows and columns must be labeled by policy IDs or roster levels.

The final results must distinguish three ladder verdicts: `demonstrated_achievable`, `certified_impossible_within_scope`, and `unresolved`. Record `theorem_applicability` separately, for example `established_for_entire_family`, `not_established`, or `outside_scope`. A valid model can be in theorem scope while the theorem is not decisive. Invalid input and invalid experiments are separate error/status fields, not extra mathematical conclusions.

An impossibility certificate for Q04 should contain the exact target `3/5`, \(\varepsilon=1/5\), allowed scalar span `1`, required span `6/5`, and positive contradiction `1/5`, with proof identifier `scalar_span_v1`, scope hash, base-model hash, and independent-verification evidence. The verifier must regenerate and check these values, not merely read them.

### 9.3 Configuration contents

Set `study_id` to `ttt_episode_mixture_equality_attribution_v1`; policy IDs and cell ordering are fixed above. Include `start: "empty"`, `role_probability: "1/2"`, `mixture_selection: "once_per_game_independent"`, the eleven requests from §6.2, `max_opponents: 64`, and all caps from §7.3. Use integer seed `20261101` only for reproducible demonstrations, not as a source of statistical repetitions or exact proof.

The executable must reject unexpected policy IDs, nonunit probability sums, negative probabilities, terminal-state actions, malformed fractions, unsupported scope versions, and reused certificates with a different manifest. Never repair invalid probabilities by silently renormalizing them.

### 9.4 Intended commands

The following are **interfaces you must create and then verify**. They were not run during preparation of this brief. Choose ordinary argument parsing and document any harmless naming change.

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m ladder inspect --out results/environment.json
python -m ladder.guard --stage development -- python -m unittest discover -s tests -v
python -m ladder.guard --stage smoke -- python -m ladder smoke --config config/protocol.json --out results/smoke
python -m ladder.guard --stage locked -- python -m ladder run --config config/protocol.json --out results/locked
python -m ladder.guard --stage verify -- python -m ladder verify --run results/locked
python -m ladder.guard --stage report -- python -m ladder report --run results/locked --out results/locked/report.md
python -m ladder play --roster results/locked/rosters/Q01.json --level 4 --human-role O --seed 20261101
```

Run all commands from the repository root. The standard-library core needs no `pip install`. If no supported Python exists, installing one is setup work and must be recorded; do not claim a target dependency was available before checking. `play` must also support a fair random role for an evaluation-compatible demonstration. Documentation should explain whether levels are zero- or one-indexed; use zero-indexed levels internally and in this example.

`inspect` records its own small inventory duration once in the same ledger without requiring the guard it helps configure. The guard uses `results/study_ledger.jsonl` and `config/protocol.json` by default, forwards the command after `--`, and supervises the whole workload process. `run` must preserve existing outputs or create a fresh run ID; it records component measurements without double-charging its enclosing guard. `verify` must be capable of detecting tampered or stale certificates. `report` must consume saved evidence rather than recomputing unlogged experiments. Interactive human demonstration time is separate from experimental compute; any automated gameplay checks must go through the guard. A single command sequence in the README must regenerate the essential result table on the target.

## 10. Outcome interpretation and final artifacts

### 10.1 Separate product verdicts from research outcomes

| Research outcome | Required evidence and wording |
|---|---|
| **Supported contribution** | The exact game implementation, independent checks, and derivations support the scoped result. A positive new-classification application would need genuinely additional valid evidence beyond prior conditions and the exact baseline; this is not expected here. Do not call an elementary bound a novel theorem. |
| **Informative negative result** | The frozen game's exact reduction and complete criterion are confirmed; equality adds no decisions. State that this particular practical application of the new classification is redundant, while reporting the useful achievable/impossible roster decisions. |
| **Practical infeasibility** | Measured costs or a supported scaling argument show that the intended unchanged diagnostic cannot fit the declared resources. Give the evidence. Missing access alone does not establish computational infeasibility. |
| **Inconclusive evidence** | A material source, model, verification, or access uncertainty remains; or a corrected nonadditive matrix defeats the planned complete analysis within the budget. An unavailable target Mac means blocked target validation and inconclusive target-performance evidence. Preserve valid partial results without extending their scope. |
| **Invalid experiment** | A game, baseline, scoring, scope, verifier, or measurement defect prevents the attempted comparison from supporting its claim. Describe the defect and the retained evidence. A bug or timeout is not refutation of the mathematics. |

A report can support the family feasibility theorem while leaving M3 resource measurements incomplete; label those claims separately. Do not force one global success label to erase such differences.

If the expected result holds, the practical recommendation is: implement the verified five- or six-level roster if that is the requirement; for seven at 60%, lower the count/margin or change the policy family. Do not keep tuning mixture weights to satisfy an excluded request. Do not invest further in an equality-classification integration for this particular architecture.

The report must explicitly say that no human difficulty order, learning gain, enjoyment, or whole-game depth was validated. A bot's unseen weaknesses do not falsify its internally verified roster relation. The result covers the declared starting state, role average, and fixed mixture policies only.

If recommending one next research step, make it conditional and limited: on the designer's actual next game and a substantively different fixed policy library, establish the exact numerical rank/realizability structure before building a new tuning system. That is a new study, not an automatic rescue branch here; do not promise that more policies will satisfy the requested ladder.

### 10.2 Required artifacts for stages actually reached

Deliver runnable source and a concise README whenever implementation occurred. Preserve the exact configuration, source hashes, Python/OS/hardware records, all base-game traces, rational matrices, independent checks, query results, witnesses/certificates, failure records, and cumulative resource ledger. Include a regeneration command for the essential decision and attribution tables.

The research report must lead with the scoped product decisions and the incremental-classification conclusion. Explain the game-to-matrix connection, full-family proof, strongest reference, source attribution, actual target costs, deviations, and limitations. Include the exact frontier table for at least \(N=2,3,5,6,7,8,15,16\), the eleven request results, and the attribution table. Tables are sufficient; a plot is optional and must not replace exact values.

For the expected analytic stop, include the verifiable derivation (8)–(13), independently checked actual-game evidence, and runnable witness/certificate verification. An optimizer, training log, large benchmark, confidence interval, and performance plot are not missing deliverables when their associated claims were deliberately not made.

If stopped earlier, deliver only the artifacts actually produced and state which stages were not reached. A mathematical source audit can be delivered even if target hardware is absent. Never populate results with expected values and label them observed.

## 11. Supporting research, novelty assessment, and review findings

### 11.1 Decision-relevant literature coverage

Research on 1 November 2026 used both available web-search engines, direct primary-source retrieval, and the supplied material. Searches included the exact Ogranovich title and arXiv identifier, author plus uniqueness/equality terms, determinant and Plücker-ratio minimizers, alternating rescalings, positive Grassmannians, rank-two games, disc games, sign-rank, strategy ladders, and mathematical/autonomous balancing. The original MathOverflow discussion was inspected; its author answer points to the 2020 preprint [R7]. Forward work on ordinal potential ratings was examined because it could otherwise be mistaken for a proof of numerical rank two.

| Closest source | What was read and why it matters |
|---|---|
| Ogranovich [R1] | Introduction and relevant extremality/uniqueness results. Establishes that the optimum and several equality facts are prior work. |
| Supplied note [A1] and package [A2] | Full main proof, appendices, source scripts, recorded results, and figures. Supplies the proposed equality completion and its limitations. |
| Hernandez et al. [R2] | Target-graph and optimization formulation, Algorithm 1, and discussion. Motivates feasibility feedback, but does not certify arbitrary failed searches. Dense fixed-policy matchup counts are quadratic in policy count; do not repeat the paper's unsupported exponential characterization for that count. |
| Lantz et al. [R3] | Scope, strategy/resource model, strength, and step definitions. A calibrated internal roster is weaker than their broader strategy-ladder proposal; head-to-head results have matchup limitations. |
| Bertrand et al. [R4] | Elementary rating limitations and disc decomposition/model definitions. A fitted transformed-payoff representation is not the exact expectation required here. |
| Vadori and Savani [R5] | Proposition 1, Theorem 2, Corollary 2.1. Ordinal/sign-rank structure does not preserve numerical score margins. |
| Liu and Marschner [R6] | §§2–4. Mathematical inverse balancing and existence structure precede this proposal; their allowed handicaps and target equilibrium differ from this study. |

No later complete resolution of Ogranovich's equality question was located in this coverage. That is not exhaustive priority verification. Differently named, unindexed, unpublished, non-English, or older equivalent results may exist. The original Berman-related work cited by Ogranovich was not independently recovered; treat its history as Ogranovich's attribution, not a newly verified source. No cited implementation was installed or benchmarked for this planning task.

The selected negative result does not depend on establishing the classification's historical novelty. A report should not spend the remaining laptop budget on an open-ended literature campaign.

### 11.2 Supplied-material audit and planning work actually performed

The source PDF has eight pages. Some equations are rasterized and disappear in ordinary text extraction; the accompanying LaTeX recovered their exact content, and the theorem, proof, payoff bridge, and appendices were checked against rendered PDF pages. The standalone and archived PDFs have different byte hashes but identical extracted text. The main note's Figure 1 illustrates explicit alternating formulas; it is not an exhaustive search.

The supplied results report 98 baseline sizes (3–100), 120 allowed-parameter checks, 98 outside-parameter checks, 294 common-linear-map checks, and 323,204 cyclic determinant-identity checks. They also report fifteen local SLSQP runs at sizes 6, 8, and 12. In planning, the stored final-frame constraint residuals for all fifteen searches were recomputed and agreed exactly with the recorded residuals under the local floating calculation. The optimizations themselves were **not rerun**. These observations verify aspects of the supplied records, not the theorem or a practical game application [A2].

The separate `binary_clock_hierarchy.tex` was read as a secondary mathematical document. It concerns binary observations of independent finite clocks and prediction horizons; it has no strategic opponent-choice or all-earlier payoff mechanism needed here. Its checker was inspected but not executed. Do not merge that result into this project or treat it as evidence for difficulty calibration.

Planning computations actually performed were the small deterministic game probes, a separate bitboard/retrograde probe, exact role averaging and scalar-coefficient checks, a 5,478-state census, simple frontier calculations, and an exact check of the rational six-column equality fixture in Appendix A. They were unblinded checks in the available research environment, not Mac measurements, not external peer review, and not a completed production implementation. Their expected traces are disclosed below so discrepancies can be diagnosed.

### 11.3 Material review findings and their disposition

Three separate agent reviews were performed after drafting: mathematical/novelty fidelity, experimental attribution, and executable handoff. Material findings were repaired and the affected sections were reviewed again. These were internal reasoning and implementation-design reviews, not external peer review or formal verification. The following issues determined the protocol; none is resolved by claiming absolute perfection.

| Challenge | Disposition |
|---|---|
| The theorem could be nonconstructive at the game-policy level. | The selected family has an explicit rational witness and full exact reduction. No general equality-intersection solver is assumed. |
| The applicability assumption could hold only on sampled bots. | Rank is proved through the complete episode-mixture representation; no numerical rank fitting. |
| The new mechanism could already be known. | Separate published optimum, published equality conditions, and the supplied completion. The elementary frontier is not claimed as new mathematics. |
| Favorable synthetic inputs could manufacture utility. | Actual tic-tac-toe traces ground the primary evidence; abstract matrices are only guards. The observed collapse leads to a negative attribution conclusion. |
| A stronger baseline could remove the gain. | It already does: the complete scalar baseline is mandatory and receives the same cached game information. No optimizer race is authorized. |
| Preprocessing or verification could dominate. | Include optimal-policy construction, all base games, and the independent reference in cold totals. Make no speedup claim. |
| Finite precision could change boundary verdicts. | Rational requirements, scores, sampling maps, and inequalities; floats are display only. |
| Aggregate scores could hide an episode-selection implementation bug. | Product expectations and injected independent-input construction checks are separate. Test provider use and persistent modes; do not infer independence from matching scores. |
| Near equality could be mistaken for stability. | No near-optimal guarantee or approximate-rank extension is assumed. Appendix B of the source is explicitly limited. |
| Runtime/memory could exceed estimates. | Inspect actual hardware, calibrate, enforce cumulative caps, and stop without changing the game. |
| Separate test commands or repair directories could bypass budget accounting. | One study-level ledger and top-level supervisor cover all workload commands; nested timers are not charged twice. |
| A small pilot could be inconclusive. | Logical identity failure triggers bounded diagnosis; noisy sampled results never become certificates. |
| An essential source could be inaccessible. | Core definitions and proofs are self-contained here; disclose any absent source cross-check and withhold affected novelty claims. |
| The main utility hypothesis could be false. | That is an anticipated valid outcome. Complete the exact negative finding and stop; do not search for a replacement game. |

Residual risks are the transfer from written rules to independent implementations, unverified target environment, and limited practical representativeness. The specified gates address the first two; the third remains a scope limitation and must remain visible in the report.

## 12. References and supplied-file provenance

### Supplied sources

**[A0] User task and selected application.** `Pasted text(20261101-130606).txt`, supplied 1 November 2026. Planning workspace location: `upload/Pasted text(20261101-130606).txt`. It defines the required decision, all-earlier expected-score margin, exclusions, Mac constraint, and execution-brief deliverable. SHA-256: `9f90437b0caf109910c74c0b45c0cc2e569824e4e36b61b3a68ac389a440f0ef`.

**[A1] Research note, author not identified in the supplied title.** *Equality cases in a planar determinant minimax problem: A complete classification and a connection to finite strategic ladders*, prepared 1 November 2026. Supplied filename `equality_classification_research_note(1).pdf`, eight pages; planning workspace `upload/equality_classification_research_note(1).pdf`. Critical locations: Theorem 1 and equation (1), p.2; §§3.2–3.3 and equations (3)–(6), p.3; §3.4 and strategic application equation (8), p.4; numerical evidence §5, p.5; novelty and limits §6, pp.6–7; Appendix A, pp.7–8; Appendix B, p.8. SHA-256: `37f517a547a9ca350b67b298e214f57d66c0940080b38eac50b8ecd3405195ab`.

**[A2] Reproducibility package.** `mathematical_discovery_reproducibility(1)(1).zip`, supplied 1 November 2026; planning workspace `upload/mathematical_discovery_reproducibility(1)(1).zip`. SHA-256: `ff631d8043f71c303002f7f5f3ff03e10ea2f20a9fbf42f15e029a8b5687528a`. Its relevant internal paths are `README.txt`, `output/research_note.tex`, `output/pdf/equality_classification_research_note.pdf`, `verify_classification.py`, `classification_verification_results.json`, `classification_plot.pdf`, `classification_plot.png`, `strategic_numerics.py`, and `strategic_numerics_results.json`. `output/build_research_pdf.py` is the rendering pipeline. The separate secondary document and checker are `binary_clock_hierarchy.tex` and `verify_binary_clock_hierarchy.py`.

These are provenance paths from planning, not assumed paths on the Mac. If the supplied files are passed to you, copy them unchanged into `inputs/` and record hashes. This brief contains everything needed to implement and prove the selected diagnostic without the original code. Original attachments are useful for source fidelity; their absence must be disclosed but does not justify inventing missing contents or block the self-contained scalar analysis. No external dataset is required.

### Primary external references

**[R1] Vadim Ogranovich (2020).** *The Regular Polygon Minimizes The Ratio of Plucker Coordinates on The Positive Grassmannian*. arXiv:2005.00185v2, 12 May 2020. [Stable record](https://arxiv.org/abs/2005.00185v2), [full PDF](https://arxiv.org/pdf/2005.00185), [HTML](https://arxiv.org/html/2005.00185v2), [DOI](https://doi.org/10.48550/arXiv.2005.00185). Critical locations: introduction p.1; equation (1.1); Theorem 2.13; Theorem 3.1; Propositions 3.2, 3.5, and 3.6. Use theorem numbering as authoritative across HTML/PDF layouts.

**[R2] Daniel Hernandez, Charles Takashi Toyin Gbadamosi, James Goodman, and James Alfred Walker (2020).** *Metagame Autobalancing for Competitive Multiplayer Games*. IEEE Conference on Games, pp.275–282. [arXiv record](https://arxiv.org/abs/2006.04419), [full PDF](https://arxiv.org/pdf/2006.04419), [accessible accepted manuscript](https://eprints.whiterose.ac.uk/id/eprint/162107/1/Autobalancing_CoG_paper_3_.pdf), [DOI](https://doi.org/10.1109/CoG47356.2020.9231762). Relevant locations: §§II–III, Algorithm 1, §III.B, discussion. The accepted manuscript has an added repository cover page.

**[R3] Frank Lantz, Aaron Isaksen, Alexander Jaffe, Andy Nealen, and Julian Togelius (2017).** *Depth in Strategic Games*. AAAI workshop paper. [Author-hosted full PDF](https://www.nealen.net/papers/Lantz2017Depth.pdf). Relevant sections: “Goals and Clarifications,” “Skill Chains and Strategy Ladders,” “What is Strength?” (PDF p.4), and “What are Steps?” (PDF p.5). This source does not validate this brief's score threshold as human-perceptible difficulty.

**[R4] Quentin Bertrand, Wojciech Marian Czarnecki, and Gauthier Gidel (2023).** *On the Limitations of the Elo, Real-World Games are Transitive, not Additive*. AISTATS, PMLR 206:2905–2921. [Record](https://proceedings.mlr.press/v206/bertrand23a.html), [full PDF](https://proceedings.mlr.press/v206/bertrand23a/bertrand23a.pdf). Relevant locations: §2.2, Example 3; §§3.1–3.3; Theorems 1–2 and equation (6). Treat model transformation and actual signed payoffs separately.

**[R5] Nelson Vadori and Rahul Savani (2024).** *Ordinal Potential-based Player Rating*. AISTATS, PMLR 238:118–126. [Record](https://proceedings.mlr.press/v238/vadori24a.html), [full PDF including appendices](https://proceedings.mlr.press/v238/vadori24a/vadori24a.pdf). Relevant locations: Proposition 1, Theorem 2, Corollary 2.1 (main paper pp.4–6). Its sign-rank statement does not establish numerical rank two.

**[R6] Albert Julius Liu and Steve Marschner (2017).** *Balancing Zero-Sum Games with One Variable per Strategy*. AIIDE 13(1):57–65. [Author-hosted PDF](https://www.cs.cornell.edu/~ajul/zerosum/zerosum.pdf), [DOI](https://doi.org/10.1609/aiide.v13i1.12924). Relevant locations: §§2–4, inverse equilibrium balancing and handicap functions.

**[R7] Vadim Ogranovich (2018; answer 2020).** *Another extremal property of regular polygons*. MathOverflow question, 14 October 2018; author answer, 11 May 2020. [Original discussion](https://mathoverflow.net/questions/312858/another-extremal-property-of-regular-polygons). Historical provenance only, not independent validation of the supplied equality completion.

**[R8] Python Software Foundation, official Python documentation (accessed 1 November 2026).** [`fractions`](https://docs.python.org/3/library/fractions.html), [`random`](https://docs.python.org/3/library/random.html), [`resource`](https://docs.python.org/3/library/resource.html). Relevant entries: `Fraction` string construction and `from_float`; `randrange` and integer selection; `getrusage` and platform-dependent resource semantics. Record the installed Python version and consult its matching documentation when an API detail differs. No claim is made that the latest documentation version is installed on the Mac.

## Appendix A. Concise mathematical audit of the supplied mechanism

This appendix preserves the necessary reasoning if external source access is temporarily unavailable. It summarizes and checks the supplied proof [A1]; it does not claim these arguments as this PoC's new contribution.

Let \(X=(v_0,\ldots,v_{N-1})\in\mathbb R^{2\times N}\) have all ordered determinants positive. Put \(v_{i+N}=-v_i\), \(\delta_i^{(k)}=\det(v_i,v_{i+k})\), and

\[
D_k=\left(\prod_{i=0}^{N-1}\delta_i^{(k)}\right)^{1/N}.
\]

All logarithms below are defined because positivity is strict. The four-column determinant identity and the elementary geometric-mean sum inequality give

\[
D_k^2\ge D_1^2+D_{k-1}D_{k+1},\qquad 2\le k\le N-2,
\quad D_{N-k}=D_k.
\]

Normalize \(D_1=s_1\), and set \(h_k=\log(D_k/s_k)\). The sine identity \(s_k^2=s_1^2+s_{k-1}s_{k+1}\) and weighted AM–GM imply

\[
h_k\ge\frac{q_k}{2}(h_{k-1}+h_{k+1}),\qquad
q_k=\frac{s_{k-1}s_{k+1}}{s_k^2}\in(0,1),\quad h_1=h_{N-1}=0.
\]

A negative interior minimum would contradict this inequality, so all \(h_k\ge0\). An interior zero forces its neighbors to zero and propagates. If the smallest/largest minors are \(m,L\), then \(m\le s_1\) and \(L\ge D_d\ge s_d\). Hence \(L/m\ge s_d/s_1\). Equality forces \(m=s_1,L=s_d,D_k=s_k\), and every adjacent and distance-\(d\) cyclic minor equals its respective extremum. These are the known inputs reconstructed by supplied Appendix A, rather than additional novelty.

For \(N\ge4\), put \(c=s_1\), \(a_i=\delta_i^{(2)}\), \(b_i=\delta_i^{(3)}\). The consecutive determinant identity is

\[
a_i a_{i+1}=c^2+c b_i.
\]

At equality, averaging its logarithm and using \(D_2=s_2,D_3=s_3\) yields equality in Jensen's inequality for

\[
\phi(z)=\log(c^2+ce^z),\qquad
\phi''(z)=\frac{ce^z}{(c+e^z)^2}>0.
\]

Thus every \(b_i=s_3\), and \(a_i a_{i+1}=s_2^2\). Therefore \(a_i\) alternates; if \(N\) is odd it is constant. For even \(N\), reciprocal scaling of the two parities makes all distance-two minors equal \(s_2\) while preserving adjacent minors. The rescaled vectors satisfy

\[
w_{i+2}=2\cos(\pi/N)w_{i+1}-w_i.
\]

This is the same recurrence as the regular columns, so a common invertible orientation-preserving map identifies the whole frame. Undoing the parity scaling produces \(C_q\). When \(N\) is divisible by four, distance \(N/2\) is even and its two parities force \(q^2=q^{-2}=1\). When \(N\equiv2\pmod4\), the largest unscaled even-distance sine is \(\cos\theta\), giving exactly the interval in (3). The lower minor constraint follows from \(\cos\theta\sin(2\theta)=2\cos^2\theta\sin\theta\ge\sin\theta\) for \(N\ge6\). Handle \(N=3\) directly: all three positive minors must be equal.

The supplied Appendix B's Jensen-gap variance inequality assumes adjacent minors are already exactly equal and supplies positive lower/upper bounds for the distance-three minors. It does not control arbitrary changes in adjacency or prove closeness of every nearly optimal frame to the equality family. No stability theorem is inherited by this execution brief.

**Exact diagnostic against a mistaken rationality exclusion.** At \(N=6\), take \(q^2=\sqrt3/2\). For \(i<j\), the normalized minors are 1/2 at odd distances 1 and 5, 1 at odd distance 3, and, at even distances 2 and 4, 3/4 when \(i\) is even and 1 when \(i\) is odd. These rational numbers have minimum 1/2 and maximum 1 and satisfy all fifteen four-index Plücker identities exactly. They form a nonregular equality array. This small exact check was performed in planning. It rules out a blanket “rational payoffs cannot attain equality” argument; it is not evidence that the selected game realizes this array.

## Appendix B. Planning traces and a compact independent-check recipe

All indices below are zero-based, and the result is the signed payoff to X. These traces were generated in small planning implementations and then reproduced with a different bitboard/retrograde representation. They are provided for debugging; regenerate them from the rules.

| X policy | O policy | Legal move sequence until termination | X outcome |
|---|---|---|---:|
| `first` | `first` | `0,1,2,3,4,5,6` | +1 |
| `first` | `tactical` | `0,4,1,2,3,6` | −1 |
| `first` | `optimal` | `0,4,1,2,3,6` | −1 |
| `tactical` | `first` | `4,0,2,1,6` | +1 |
| `tactical` | `tactical` | `4,0,2,6,3,5,8,1,7` | 0 |
| `tactical` | `optimal` | `4,0,2,6,3,5,8,1,7` | 0 |
| `optimal` | `first` | `0,1,3,2,4,5,6` | +1 |
| `optimal` | `tactical` | `0,4,1,2,6,3,5,8,7` | 0 |
| `optimal` | `optimal` | `0,4,1,2,6,3,5,7,8` | 0 |

For the independent solver, represent X and O by two disjoint nine-bit masks. Generate winning masks from geometric rows, columns, and diagonals rather than importing production win detection. Breadth-first or depth-first enumerate legal states from `(X=0,O=0,turn=X)`, without expanding terminal states. Process the resulting states in descending occupied-cell count to assign terminal X payoffs and then min/max child values. Choose the smallest-index action with the state's optimal value. Check all reachable states against the production engine and independently implement the tactical priority scan.

For the five-level witness, use optimal-policy probabilities \(0,1/4,1/2,3/4,1\) and the complementary first-policy probabilities; tactical-policy probabilities are zero. Every adjacent expected score is exactly 5/8, and every more distant comparison is stronger. For six levels, use \(0,1/5,2/5,3/5,4/5,1\); every adjacent score is exactly 3/5. For seven levels at 3/5, the six required signed increments would sum to at least 6/5 inside a unit interval, which is the certificate contradiction.

Your final report must distinguish these planning expectations from independently observed target outputs, even when every value agrees.
