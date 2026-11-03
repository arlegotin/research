# PoC execution brief: does rational feedback sharing make a smaller arithmetic circuit?

**Audience:** an autonomous coding agent configured as **gpt-6.1-sol, xhigh reasoning**.  
**Prepared:** 2 November 2026.  
**Deliverable from you:** an executed, reproducible local research study, subject to the evidence gates below. This document is the plan; the PoC has not already been implemented or benchmarked.  
**Experimental machine:** one MacBook with an M3 Max. Inspect its actual configuration. Hosted planning calculations described here are not Mac measurements.

## 1. Project decision and smallest defensible contribution

Implement and evaluate **one project**: a synthesis study of paired, exact division by fixed odd constants, using a shared feedback state derived from the supplied rational-network theorem. Return an ordinary, exact rational representation at the interface, and count its entire implementation cost.

The primary question is:

> For the six prescribed coprime denominator pairs at 16 and 32 bits, does a constructive one-state rational feedback circuit reduce total mapped standard-cell area by at least 10% against the best eligible conventional implementation, while satisfying the same service budget and a predeclared timing guard?

The exact interface is

    input:  x1, x2, unsigned W-bit integers, independently arbitrary
    output: Q1, R1, Q2, R2
    for each i: xi = di*Qi + Ri,  0 <= Ri < di
    exact rational result: xi/di = Qi + Ri/di

The denominators are compile-time constants. The outputs are exact but need not be reduced fractions. There is no rounding tolerance, divisibility promise, or discarded fractional part.

**Why this is a useful narrowing.** A bounded circuit cannot store arbitrary rationals with unbounded numerator and denominator. Moreover, returning the unevaluated pair (x,d) would make division by a known constant a trivial change of representation. Requiring the mixed-number decomposition above forces a real arithmetic operation while preserving every fractional part. The experiment therefore addresses a finite arithmetic building block relevant to the user's proposed chip direction. It does not establish a general rational ALU, accelerated arbitrary-precision arithmetic, or an advantage for symbolic numerator/denominator storage.

**Smallest defensible contribution:** a verified, source-traceable bridge from the paper's algebraic sharing construction to a finite-word circuit, followed by a controlled characterization of whether the algebraic feedback saving survives complete circuit implementation. A scoped area improvement, an explanation of why the saving disappears, or a measured feasibility limit is useful. Reproducing modular inverse division by itself is not a new contribution.

The user's supplied mathematical results are the starting research contribution. Do not conduct a global priority contest or describe this study as proof of their historical novelty. Established 2-adic serial circuits, constant inverse multiplication, and shift/add synthesis are the closest implementation mechanisms [R1–R4]. The unresolved practical question is the cost of this particular sharing construction.

### 1.1 Fixed scientific choices

| Item | Fixed decision |
| --- | --- |
| Primary endpoints | Exact Q,R outputs, total mapped cell area, cycle latency, initiation interval, and zero-wire timing estimate |
| Primary widths | W = 16 and 32 |
| Denominator pairs | (3,5), (3,7), (3,11), (5,7), (5,17), (7,11) |
| Candidate | Canonical shared graph construction in §4, lifted with modulus 16, exploiting its actual contraction exponent |
| Comparators | Independent recurrence ablation; MSB digit division; LSB carry division with exact decoding; low-word inverse multiplication; reciprocal multiplication with one correction |
| Technology model | One pinned Nangate45 typical Liberty file, Yosys/ABC mapping, OpenSTA pin-load timing |
| Accuracy | Exact integer/bitvector equality; zero accepted arithmetic or protocol mismatches |
| Primary material effect | At least 10% lower total mapped area; see §8 for eligibility and timing rules |
| Tuning | Bounded development only; no search over denominator pairs, graph encodings, mapping scripts, or seeds to obtain a favorable confirmation |
| Experiment unit | A fixed width/denominator-pair/netlist configuration, not an individual arithmetic input vector |
| Hardware status | Simulation and synthesis only; no fabricated chip, FPGA board, placement, routing, or measured power |

You may choose ordinary engineering details such as internal function names and test organization. You may repair tool-command compatibility within the budgets. Do not change the primary function, baselines, matrix, effect threshold, or outcome labels without recording a protocol deviation and treating affected results as exploratory.

## 2. Inputs, recovery, and evidence status

### 2.1 Source package and what must be handed to you

Receive this brief and, preferably, the following original attachments. Preserve them read-only and extract safe copies under the repository's sources directory.

| Attachment | Relevant contents and source status |
| --- | --- |
| arithmetic_research_bundle(2).zip | Complete arithmetic_from_binary_replacement.tex; README.txt; graph_scalar_experiments.py and results; feedback_rational_experiments.py and results; connected_core_experiments.py and results |
| finite_relations_research_bundle(4).zip | finite_relations_arithmetic.pdf and .tex; RESEARCH_README.txt; tree_arithmetic_experiments.py/.json; order_rewrite_experiments.py/.json |
| finite_relations_arithmetic(2).pdf | Byte-identical to the supporting PDF inside the preceding ZIP |
| arithmetic_chip_poc(2).zip | Existing RTL, Python cycle models, HDL testbench, shell scripts, results, README, engineering note, two-page report, and architecture diagram |
| Pasted text(20261102-213139).txt | Original planning instructions; not an experimental data source |

The separately attached arithmetic_from_binary_replacement(2).pdf was unavailable during planning. Its **complete LaTeX source was recovered from arithmetic_research_bundle(2).zip and read**. The archive does not contain that primary PDF, despite its README describing one. A local attempt to typeset the source failed because the hosted TeX format was unavailable. Thus cite the recovered source's named sections, theorem label, and line ranges; do not invent primary-PDF page numbers or claim its rendered layout was inspected. Essential mathematics is available, so this is not a scientific blocker.

The supporting PDF, its source, the primary source including appendices, all research scripts and stored summaries, and the existing chip documentation/code were inspected. Supporting equations/figure and the chip report were also visually checked. The original large mathematical enumerations were **not rerun** during planning. Their numerical summaries remain source-reported evidence.

If the attachments are absent from your handoff, the complete operational definitions and algorithms below still permit implementation. Mark attachment provenance as unavailable; do not claim to have reproduced the original bundle. Request only the specific archive needed for an unresolved source discrepancy. Absence of the primary PDF alone does not block this study when its recovered source is available.

### 2.2 Source claims versus what was checked

| Central claim | Basis and status | Remaining uncertainty / resolving check |
| --- | --- | --- |
| Nonempty positive bags form an abelian group for strongly connected binary diagrams | Primary source §2.1, Lemma 2.1, lines 119–163 [A1]; mathematical result, not a hardware observation | No bag-rewrite engine is needed; do not infer efficient computation from existence of representatives |
| Minimum rational graph feedback is max(d(T_A),epsilon(A)) | Primary source §6, Theorem 6.1, label thm:main, lines 447–656 [A1]; proof inspected, including singular hidden states | The graph resource is not a register or cell-area bound; resolve the hardware question experimentally |
| Coprime diagonal denominators can share one feedback vertex | §6.5, equation label eq:primeoverlap, lines 628–652 [A1] | Pairwise coprimality is required for the selected one-state construction; generator rejects other cases |
| Universal finite tree representation requires a locally finite coefficient ring | Supporting Theorem 1, pp.2–4; limits pp.5–6 [A2] | Q is excluded; do not use this theorem as a rational-chip construction |
| Original /3,/5 recurrence terminates exactly modulo 2^W | Existing README, Why the iteration finishes exactly, and shared_feedback_div3_div5.v [A3]; independently checked in planning | New RTL, timing, and complete Q,R conversion still require validation |
| Shared feedback gives smaller chips | **Hypothesis only**; neither paper nor previous PoC establishes it | Entire mapped design versus strong conventional baselines |
| General lifted recurrence and Q,R decoder below are correct | Explicit derivations in §§4–5; exact planning probes passed | Verify code generation, RTL widths, protocol, and mapped-netlist integrity |
| Native Mac toolchain can run this study | Official tool documentation [R6–R10]; no target installation was performed | Inspect OS/architecture, install a pinned local toolchain, run capability smoke tests |

**Source-reported experiments.** The primary bundle reports 152,285 small-network output transfers, 4,096 constructive 2-by-2 rational examples, and 4,015 strengthened connected-core examples. The supporting bundle reports 1,377 finite-ring matrices and 10,226 enumerated binary graphs. These support their stated finite checks, not universal runtime or hardware claims. The primary construction script specializes its denominator-group work to 2-by-2 examples and enumerates finite residue groups; it is not a scalable arbitrary-matrix compiler.

### 2.3 Planning calculations actually performed

These are small mathematical/cycle-model checks in the hosted planning environment, not the requested PoC:

1. The canonical coefficients in Table 1 were checked with exact rational arithmetic. The lifted recurrence, modular outputs, and Q,R decoder passed all pairs of W-bit inputs for W=1 through 6 for all six pairs: **32,760 pair cases**. This includes tiny widths with denominators larger than 2^W.
2. An independent reviewer checked the original /3,/5 shared recurrence and a simplified independent reconstruction on all pairs at W=1 through 8: **87,380 pair cases**, all passed.
3. A separate reviewer checked the supplied Python cycle models and unsigned correction on **69,900 pair cases**, including exhaustive W=8 and irregular widths through 64. This was Python execution, not HDL execution.
4. A reviewer checked the LSB carry recurrence on **65,520 scalar cases**, all x for W=1 through 12 and d in {3,5,7,9,11,15,17,31}, with partial final digits. All carry bounds and modular identities passed.
5. The paper's even-denominator example for diag(1/2,1/3), at x=(1,0), would require 6z=-3 modulo 256. There is no solution. This is a direct counterexample to blindly interpreting every rational equilibrium over a power-of-two residue ring.
6. During review of the drafted brief, a reviewer independently implemented its in-place MSB and LSB schedules and reciprocal baseline. All three passed **30,690 scalar cases** covering every x at W=1 through 10, digit sizes e in {4,6,8}, and d in {3,5,7,11,17}. This checked the partial-first/partial-last digit rules actually specified here.

These sets overlap; do not sum them into a claimed independent sample size. No HDL compiler, Yosys, or Verilator was installed in the planning environment. No HDL simulation, synthesis, OpenSTA run, physical implementation, or M3 Max benchmark was performed. Do not repeat planning timings as Mac performance estimates.

## 3. What the mathematics supports, and the implementation boundary

### 3.1 Relevant objects from the primary paper

A finite open binary replacement network has internal vertices, free input terminals x_j, optional zero terminals, and designated outputs. Each internal relation is v=a(v)+b(v), permitting repeated targets and initially loops. Interpret its equations over Q. A network realizes A in Q^(m by n) when every input has an internal solution and every such solution has output Ax; internal uniqueness is not required for this definition.

Let tau(D) be the minimum number of vertices meeting every directed cycle, and define

    T_A = (Z^m + A Z^n) / Z^m, a finite abelian group
    d(T_A) = minimum number of generators of T_A
    epsilon(A) = 0 if every entry of A is a nonnegative integer; 1 otherwise

Theorem 6.1 states

    minimum tau(D), over networks realizing A
        = max(d(T_A), epsilon(A)).

It also constructs optimal networks with unique internal states; for positive optimum they can be simple, loopless, and have one recurrent strongly connected component [A1, §6.1–§6.4].

For A=diag(1/d1,1/d2), the denominator group is Z/d1 plus Z/d2. Its generator count is one for coprime d1,d2 and two when a prime divides both. More generally the diagonal lower bound is the largest number of channels whose denominators share a prime. This is the paper's arithmetic invariant; ordinary matrix rank does not substitute for it.

### 3.2 Proof audit relevant to execution

The lower-bound proof deletes k feedback vertices and eliminates the acyclic complement, obtaining H z=B x and y=C z+D x with integral matrices. It selects a nonsingular rank-sized minor H0. Totality puts Bx in the relevant column space; output uniqueness then gives A=D+C_J H0^(-1)B_I. T_A is a subquotient of the finite abelian group coker H0, so d(T_A)<=rank(H)<=k. This argument includes singular internal systems. No essential defect was found in this portion of the supplied proof.

The upper bound chooses denominator-group generators, constructs a rational factorization, makes coefficients positive by integral offsets, and expands coefficients into binary sum networks. Its existence proof is constructive in principle but does not supply efficient optimization: coefficient magnitudes, tree expansion, denominator-group enumeration, and minimum total size are uncontrolled. The strengthened simple-core construction can increase size further.

Do not implement general Smith-form search, minimum feedback-set search, addition-chain optimization, or the paper's infinite rational ladder. The chosen two-channel closed form below removes those costly subproblems entirely. Compile-time inverses use Euclid on small integers; all coefficient identities are checked exactly.

### 3.3 Guarantees that do not transfer automatically

The algebraic equations are equilibrium constraints, not update instructions. For example z=16z+c has a unique rational solution -c/15, but ordinary real iteration from zero generally diverges. A physical combinational feedback loop is not its implementation.

The supporting tree theorem gives B_T E=EA and LE=I over locally finite rings. It explicitly does not give universal finite representation over Q, Z, R, or the 2-adic integers. Its state encoding, ring arithmetic, and potentially large tree remain costs. The primary paper's countable graph realizing Q uses infinite structure; it cannot be copied into a finite chip [A1, §4; A2, §3].

For an odd fixed d, multiplication by d^(-1) modulo 2^W is a finite feedforward function. Thus **zero recurrent arithmetic words are possible** for the modular function. The paper's feedback optimum cannot lower-bound digital registers, gates, clock cycles, or energy.

Our finite implementation works in Z/(2^W), not in unrestricted Q. It uses a separately proved nilpotent error recurrence, followed by a separately proved exact representation conversion. Neither step inherits correctness merely from the graph theorem.

## 4. Candidate algorithm: a canonical shared state

### 4.1 Parameters and coefficient generation

Inputs to the generator: positive integer W and odd integers d1,d2>1 with gcd(d1,d2)=1. The main study uses W=16,32; support W=1 through 64 for tests. Reject unsupported widths, even denominators, zero denominators, and noncoprime pairs with an explicit error. Conventional baselines may independently support those denominators; the candidate's rejection is deliberate.

Use exact Python integers:

    D = d1*d2
    hi = D // di
    ai = (-inverse(hi, di)) mod di       # representative 1,...,di-1
    ci = hi*ai
    Pij = (indicator(i==j) + hi*aj) // dj
    t = (-inverse(D,16)) mod 16         # odd representative 1,...,15
    alpha = 1 + t*D
    e = v2(alpha)                      # exponent of 2 dividing alpha
    N = ceil(W/e)
    M = 2**W; mask = M-1

Check each P numerator is divisible by its denominator. Check, with Fraction,

    Pij - hi*cj/D = indicator(i==j)/di.

There is no search over alternative a_i or feedback multipliers. The fixed modulus 16 is a planning choice extending the original four-bit update; exploit any extra powers of two obtained for free. In particular, use six bits per update for (3,7), and eight for (5,17). Comparators receive that same per-update digit allowance.

**Table 1. Exact generated constants, verified during planning**

| (d1,d2) | D | h | a | c | P, by rows | t | alpha | e |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| (3,5) | 15 | (5,3) | (1,3) | (5,9) | (2,3); (1,2) | 1 | 16 | 4 |
| (3,7) | 21 | (7,3) | (2,2) | (14,6) | (5,2); (2,1) | 3 | 64 | 6 |
| (3,11) | 33 | (11,3) | (1,7) | (11,21) | (4,7); (1,2) | 15 | 496 | 4 |
| (5,7) | 35 | (7,5) | (2,4) | (14,20) | (3,4); (2,3) | 5 | 176 | 4 |
| (5,17) | 85 | (17,5) | (2,10) | (34,50) | (7,10); (2,3) | 3 | 256 | 8 |
| (7,11) | 77 | (11,7) | (5,3) | (55,21) | (8,3); (5,2) | 11 | 848 | 4 |

### 4.2 Rational witness and finite recurrence

The algebraic witness is

    z = (1+D)*z + c1*x1 + c2*x2
    ui = Pi1*x1 + Pi2*x2 + hi*z.

It has z* = -(c1*x1+c2*x2)/D and ui=xi/di over Q. It uses one feedback coordinate, attaining the theorem's optimum for these diagonal maps. This is a closed-form specialization/derivation for the PoC, not a general network compiler.

Replace the equation by the clocked update

    z0 = 0
    z_(j+1) = (alpha*z_j + t*(c1*x1+c2*x2)) mod M
    after exactly N updates:
        ui = (Pi1*x1 + Pi2*x2 + hi*z_N) mod M.

D and t are odd, so the fixed point exists uniquely modulo M. If E_j=z_j-z* in that residue ring, then

    E_(j+1)=alpha*E_j
    E_N=alpha**N*E_0=0 mod M, because e*N >= W.

Consequently di*ui=xi mod M, with ui represented in [0,M). This is exact finite-word stabilization, not a floating-point convergence test. No norm, epsilon, approximate solve, numerical conditioning, or real stability assumption is needed. Updates are bounded in advance; never stop because the state merely looks stable.

An extra update must preserve the final state. Before update N, some inputs can already be correct; that does not justify a smaller worst-case schedule. Include the known /3,/5 counterexample W=16,x=(1,0): three updates are insufficient.

### 4.3 Bit-width conventions

Use unsigned bitvectors throughout the hardware. Negative algebraic states are represented by their residues; they are not saturated. Every modular multiplication and addition is explicitly reduced to its low W bits. Python uses arbitrary-precision intermediates and masks at the specified boundaries.

All constants, inverse words, lookup tables, and masks are generated by Python and emitted as explicitly sized HDL literals. Do not use an unsized 1<<W, a 32-bit Verilog integer for 2^W, or the original wrapper's 64-bit shifting trick at W=64. For nonmodular products specify their full mathematical width before taking slices. Signedness must never depend on an implicit HDL cast.

Use one shared constant-multiplication emitter for the candidate and arithmetic comparators. The primary form is a deterministic balanced shift/add/subtract expression from canonical signed-digit representation; all modular intermediates have W-bit wraparound. For a nonmodular product, use an explicitly sufficient extended width. Permit normal synthesis sharing and optimization; do not attach keep/dont_touch attributes to force a preferred register count. Gate G3 checks an ordinary literal-multiplication encoding as a bounded control.

The mathematical network is recorded as coefficient data and an exact transfer certificate. Literal expansion of a coefficient into that many repeated token branches would add irrelevant cost and is not required in RTL. This implementation allows ordinary shifts and subtraction when synthesizing constants; it is a digital adaptation of the graph, not a demonstration of a physical gate set containing only positive bag replacement.

## 5. Exact conversion to ordinary rational outputs

Let u in [0,M) satisfy d*u=x mod M for x in [0,M), and let b=ceil(log2(d)). Compute

    product = d*u                     # full W+b-bit unsigned product
    k = product >> W                  # 0 <= k < d
    C[k] = ceil(k*M/d)                # compile-time table
    E[k] = d*C[k] - k*M               # compile-time table, 0 <= E[k] < d
    Q = u-C[k]
    R = E[k].

The proof is d*u=x+k*M, so x/d=u-k*M/d. Taking the floor gives Q=u-ceil(k*M/d), and substituting gives x=d*Q+E[k]. Q is nonnegative for valid inputs. This also shows why u alone is generally not floor(x/d).

Compute C and E using integer arithmetic in the generator. Store C in W+1 bits during construction, with a checked sufficient hardware width after reachability analysis. Give unreachable k>=d table entries deterministic zero values, not unspecified x values. Assert k<d on reachable executions. The product-based index avoids the original wrapper's long threshold chain and handles tiny widths without duplicate-threshold bugs.

Use this same generated decoder for shared feedback, independent feedback, and inverse multiplication. The LSB carry comparator already computes k and may use the same C/E tables without recomputing d*u. Do not force that comparator to discard information it naturally has.

All Q and R outputs are **registered after conversion** and retain their preceding values until completion. This repairs the original PoC's timing mismatch, where the shared unsigned wrapper converted after output registers but the direct divider registered the final quotient.

## 6. Mandatory comparators and protocol

### 6.1 Six architecture families

| ID | Implementation | Scientific role |
| --- | --- | --- |
| shared | §4 recurrence, positive reconstruction, §5 decoder | Proposed graph-derived mechanism |
| independent | Two uncoupled recurrences below, using the same alpha and N | Ablation of cross-channel feedback sharing |
| msb_serial | Conventional high-to-low long division, e bits per update | Strong low-state reference for complete Q,R outputs |
| lsb_serial | Conventional low-to-high carry division, e bits per update, then C/E lookup | Strong 2-adic comparator; avoids a redundant full-word decode product |
| inverse | u=(x*inverse(d,M)) mod M, then §5 decoder | Established zero-recurrence modular comparator |
| reciprocal | Full-product reciprocal quotient estimate plus one correction | Conventional direct Q,R comparator with no modular representation conversion |

Python divmod and Fraction are independent correctness oracles, not performance competitors for a hardware-area claim. A literal HDL x/d and R=x-d*Q is a cheap sanity reference in tiny simulation; do not use an inefficient generic divider as the only performance baseline.

**Independent recurrence.** For each channel initialize r_i=0 and update

    r_i_next = (alpha*r_i - t*hi*xi) mod M.

After N updates r_i=xi/di mod M. This follows from (1-alpha)r=-t*(D/di)*x. Decode r directly. This ablation has the same feedback multiplier and update budget as shared, but two arithmetic state words and no cross-channel reconstruction. Subtraction is allowed. Do not retain the supplied independent RTL's redundant expressions such as 2*x+5*z when the state already gives -x/3.

**MSB serial.** Each channel uses one W-bit in-place input/quotient shift register V and a b-bit remainder r. Initialize V=x,r=0. In each update consume k high input bits and perform k cascaded bitsteps:

    tmp = 2*r + next_input_bit         # b+1 bits
    output_bit = 1 if tmp >= d else 0
    r = tmp-d if output_bit else tmp
    append output_bit to quotient

Shift V left by k and insert the new quotient bits at its low end. After W consumed bits V is Q and r is R. Use k=W-(N-1)*e on the first update, then e. Thus no zero-padding register overhead is imposed when e does not divide W. The invariant after each consumed prefix is prefix=d*quotient_prefix+r, 0<=r<d.

**LSB serial.** Initialize V=x,c=0. Consume k low input bits, one bit at a time:

    qbit = input_bit XOR (c mod 2)
    c_next = (c + d*qbit - input_bit) // 2

The division is exact and is a fixed right shift in hardware. Form its numerator in b+1 unsigned bits; for reachable states it is nonnegative and c remains in [0,d). Group e bitsteps per update, except the last update uses k=W-(N-1)*e. If the emitted bits in consumption order are q_0,...,q_(k-1), set q_digit=sum(q_j*2**j) and V_next=(V>>k)|(q_digit<<(W-k)), masked to W bits. This inserts the digit at the high end while preserving low-first bit order. At completion V=u and

    d*u-x = c*M.

Therefore c is exactly the decoder's k. Return Q=u-C[c], R=E[c] without the full d*u product. An equivalent radix formulation is q=((a-c)*inverse(d,2^k)) mod 2^k and c_next=(c+d*q-a)/2^k; use it as a software cross-check, not a second primary RTL variant.

**Inverse.** Compute K=inverse(d,M) at generation time, check d*K mod M=1, and emit the low-word constant product. There is no runtime inversion. Capture x, compute/decode, and register the final outputs on the next edge. A compiled constant multiplier is valid even though the graph model uses repeated addition.

**Reciprocal.** Set m=floor(M/d) at generation time. Form the exact full product and

    q0 = (x*m) >> W
    r0 = x-d*q0
    if r0 >= d: Q=q0+1; R=r0-d
    else:       Q=q0;   R=r0.

For 0<=x<M, q0 is either floor(x/d) or one less: the real quotient-estimate error is x*(M mod d)/(d*M)<1. Thus 0<=r0<2d and one correction suffices. This proof is a simple derivation in this brief, related to established reciprocal division [R2]. Use sufficient full-product/subtraction widths. This is a correct modest comparator, not a claim to reproduce every optimized Granlund–Montgomery code sequence.

### 6.2 External transaction contract

Every architecture exposes clk, synchronous active-high rst, start, x1, x2, busy, done, Q1, R1, Q2, R2. Quotient ports have W bits; remainder ports have ceil(log2(di)) bits.

At a rising edge with rst=1, cancel any transaction, clear busy/done, and set outputs to zero. Otherwise accept start only when busy was false immediately before the edge. Capture all needed input information at that edge; inputs may change arbitrarily afterward. Ignore start while busy, including the completion edge. done is a single-cycle pulse; outputs hold until completion or reset.

For shared, independent, msb_serial, and lsb_serial:

    edge A:       accept, initialize, busy=1, done=0
    edges A+1...A+N: perform N updates
    edge A+N:     register final Q,R from the NEW state, busy=0, done=1
    edge A+N+1:   earliest next acceptance
    latency L=N cycles; initiation interval II=N+1 cycles

For inverse and reciprocal, accept at A and register final Q,R at A+1; L=1 and II=2. Do not insert idle arithmetic cycles or dummy registers to make fast baselines match the candidate. The comparison is subject to the same **maximum** service budget, not a requirement to slow down better implementations.

The serial designs may reuse input storage in place. The graph may retain its two inputs because it needs them. Equal external behavior does not justify forcing all designs to carry identical redundant internal state. Include every retained input, arithmetic state, output, control, enable, and reset gate in area.

Combinational reconstruction/decoding on the final update may create a long path. Count it in timing; there are no multicycle exemptions. If it fails the guard, record that failure. Do not add a private pipeline to rescue the candidate after seeing confirmation results.

## 7. Correctness before comparative interpretation

### 7.1 Independent oracles and exact invariants

Use three logically different checks:

1. Python divmod(x,d) determines Q,R independently of modular arithmetic. Fraction verifies Q+R/d=x/d without float conversion.
2. The modular oracle is (x*pow(d,-1,2**W)) mod 2**W. Also check d*u mod 2**W=x. The latter determines u uniquely for odd d.
3. The full-output certificate is the **ordinary integer** equality d*Q+R=x together with 0<=R<d. Zero-extend operands and evaluate the product, sum, and comparison in W+b+1 bits, where b=ceil(log2(d)); this holds all possible port values, not only correct outputs. Checking the equality modulo 2**W would admit incorrect quotients.

For shared, check the exact coefficient certificate, contraction exponent, fixed-point residual, and per-step agreement with a separately written Python integer model. For the serial baselines, check the prefix/carry invariants from §6. Check the reciprocal bound q0 in {Q,Q-1}. Verify decoder tables by enumerating all k in [0,d), and all u at tiny widths.

### 7.2 Minimum test set

All seeds and vectors are generated locally; there is no external dataset.

| Layer | Required cases | Purpose |
| --- | --- | --- |
| Coefficient algebra | Every primary pair, exact Fraction matrix identity; rejection of (3,3), (3,9), (2,3), zero and negative denominators | Source fidelity and applicability |
| Python mechanism | Every input pair for W=1 through 8 and all six pairs | Exact recurrence, decoder, tiny widths, cross-channel independence |
| Digit functions | Every reachable carry/remainder and every bit; additionally enumerate radix transitions for e in {4,6,8} | Exhaust the small state transition functions |
| RTL exhaustive | All input pairs at W=1 through 6 for all pairs; all W=8 pairs for (3,5), across all six methods | Detect width, counter, sign, final-update, and decode defects |
| RTL irregular widths | W in {7,9,15,17,31,33,63,64}, directed cases plus 256 deterministic pairs per pair/method | Partial first/last digits, literals, boundary widths |
| Primary RTL vectors | For each of 12 primary configurations, 4,096 deterministic random pairs plus all prescribed directed vectors, shared across methods | Held-out implementation checks; not statistical area samples |
| Protocol | Reset in idle and every update position; reset on completion; start held high; start while busy; earliest next start; changing inputs while busy; idle output retention | Transaction semantics and cancellation |
| Mapping integrity | Structural checks on every netlist; formal postmapping equivalence on each claimed winning candidate/comparator pair, with gate-model simulation useful as an additional diagnostic | Detect dropped outputs, wrong top/parameters, and mapping mistakes |

Use separate seed namespaces: development 20261102; confirmation 20261103; robustness 20261104, combined deterministically with W, denominators, and a stream label. Record the generator algorithm and Python version. Seeds do not influence netlist selection or area.

Directed scalar values include 0,1,d-1,d,d+1, M-1,M-2, powers of two and their immediate neighbors, alternating bit patterns, and valid values around multiples of d near both ends of the range. To test decode boundaries, take u=C[j]-1,C[j],C[j]+1 within [0,M), convert to x=d*u mod M, and test that x. Use cross-channel pairs (v,0),(0,v),(v,M-1),(M-1,v), plus mixed asymmetric values. Bound a directed suite at 1,024 pairs per configuration by deterministic deduplication and ordering; keep all decoder boundary cases.

The primary random cases use the full unsigned range, not only multiples of d or correlated x1=x2. Classify correctness failures by arithmetic, protocol, generator, or tool failure. Preserve the first counterexample and a bounded trace. Do not delete failing vectors after repair.

### 7.3 Formal checks and their limits

Use Yosys's built-in SAT capability first; an external SMT solver is optional. Build a bounded sequential miter that instantiates the **actual generated RTL**, including reset, the acceptance edge, and the architecture's prescribed L update cycles: N for recurrent/serial designs and 1 for inverse/reciprocal. Check final NEW-state output capture where applicable. For the successful-transaction property, assert reset at edge A-1, hold reset low from A through A+L, and assert start at A with arbitrary symbolic W-bit inputs. Later data inputs and busy-time start may change arbitrarily. At A+L assert the full Q,R certificate against saved accepted inputs, and check modular identities where a residue exists. Assert that method's own done/busy schedule and output retention. Reset-cancellation scenarios are tested separately. A separately reimplemented mathematical recurrence is not RTL verification. This avoids asking a solver to compare two implementations of division that may share the same bug. Derive the full-output assertion width independently.

Mandatory formal targets are:

- All six architecture families at W=8 for (3,5), plus the candidate at W=8 for every pair.
- The candidate at W=16 and W=32 for every pair.
- Before a supported area-improvement claim, the best eligible comparator and mapped candidate/comparator implementation for every claimed winning configuration.

There is a cumulative ceiling of 64 formal jobs, including retries, at 120 seconds each. Local transition exhaustions and the algebraic proofs cover baseline algorithms; RTL tests cover their integration. A proof timeout is recorded as unknown. It is neither a counterexample nor proof of incorrect arithmetic.

Deduplicate the shared W=8,(3,5) target that appears in both small-target sets. Decomposition and mapping proofs consume the same attempt slots; the cap may limit how many otherwise promising configurations can be validated as wins. Gate-level simulation alone is not a substitute for the required postmapping equivalence of a claimed positive case.

If a large miter times out, allow one bounded decomposition: prove each channel's full-output certificate separately, or prove the modular identity and shared decoder separately. Do not weaken inputs, assume divisibility, constrain one channel to zero, or remove output ranges to obtain a pass. Charge each retry to the same cap.

If remaining formal access is insufficient, retain simulation results and qualify the scope. A configuration may contribute a measured resource observation with its verification status, but an unverified candidate/comparator pair cannot support the primary positive claim. An informative negative comparison may be based on the exact algorithm derivation, exhaustive tiny RTL, broad larger RTL tests, and verified measurement flow; state explicitly if full-width RTL proof remains incomplete.

For mapping equivalence, first demonstrate the chosen library's functional models on a tiny adder/register circuit. Read actual Liberty Boolean/sequential models or use the library's official Verilog functional models; blackboxed cell outputs cannot establish equivalence. An unmatched blackbox, unexplained X, or failed equivalence makes that mapped comparison invalid.

### 7.4 Controls that should fail, and a positive measurement control

In a separate mutant build, shorten /3,/5 W=16 by one update and require the test at x=(1,0) to fail. Corrupt one decoder offset and swap one channel's input; demonstrate that the independent oracle catches the resulting wrong outputs. Separately validate the certificate itself with the bad witness x=0, Q=floor(M/d), R=M mod d: it satisfies R<d and the narrow equality modulo M, but d*Q+R=M is not x, so the wide certificate must reject it. Narrowing a certificate alone does not mutate arithmetic outputs. Mutants are test fixtures, never benchmark entrants.

For measurement, map two small observable-register controls differing by W live flip-flops, plus a small logic cone. The reported sequential area must increase consistently with the actual mapped cells. Unconnected dummy registers would be optimized away and are not a valid control. Confirm that removing an observed output from a copy changes reports and is caught by the port audit. These controls test the instrumentation, not the arithmetic hypothesis.

## 8. Experimental design, metrics, and decision thresholds

### 8.1 Development, freeze, and confirmation

Development synthesis uses W=8 and pairs (3,5),(5,7),(5,17). These expose a sparse original recurrence, a nonsparse feedback multiplier, and larger decoder constants. Arithmetic/RTL correctness tests may cover all widths before the freeze; using them to fix correctness is allowed. Do not inspect primary comparative area results while choosing the design.

At most two constant-expression encodings are considered per arithmetic family: the common CSD emitter and literal fixed multiplication. The serial bitstep architectures have only their prescribed form. For shared, independent, inverse, and reciprocal, compare both encodings on the three development pairs, with the same mapping. Select the smaller geometric-mean-area encoding for that family if it is correct and is no more than 10% slower than the other encoding on every development pair; otherwise select the faster encoding by geometric mean delay. An area tie within 1% selects CSD. This bounded engineering choice is fixed before confirmation. If timing is unavailable, retain CSD and report the encoding comparison as structural only.

Write protocol_lock.json before primary comparison. It contains the exact formulas, matrix, selected encodings, scripts, tool/library hashes, constraints, seeds, metric rules, budgets consumed, and a hash of all participating source. Final rows use this frozen code. Retain development measurements.

If a genuine implementation bug appears after lock, preserve the original lock and failed rows, make one versioned correction, rerun all affected cells for every relevant method, and issue a new lock. Their old results cannot be selected opportunistically. A hypothesis, baseline, metric, or matrix change is a new exploratory study and does not inherit confirmatory status. Budget exhaustion is a reason to stop, not to loosen the criteria.

### 8.2 Primary matrix and why it is sufficient

The locked matrix is 6 pairs by 2 widths by 6 architectures: **72 mapped designs**.

The denominator choices are mechanistic test cases, not a random sample of applications:

- (3,5) reproduces the supplied construction and has a sparse shift-only feedback multiplier.
- (3,7) tests extra free contraction bits and nonmultiple digit widths.
- (3,11) and (7,11) require nonsparse larger multipliers and expose coefficient growth.
- (5,7) supplies another small nonsparse case without a large decoder.
- (5,17) combines a shift-only multiplier with larger output-conversion constants.

Widths 16 and 32 distinguish small demonstrators from a useful word-size block without assuming 64-bit hardware is cheap. Two optional robustness activities are fixed in advance: all six methods at W=64 for (3,5), and a four-method modular-only diagnostic at W=32 for (3,5). They run only if needed to explain a material result and within the cumulative caps. The diagnostic compares shared, independent, LSB carry, and inverse with registered modular outputs; it cannot replace the primary Q,R endpoint.

Repeated-prime and even-denominator examples are mathematical/generator controls, not artificially failed performance rows. The candidate is outside its declared domain there. Do not claim their exclusion demonstrates superiority on realistic denominators.

### 8.3 Mapping and timing model

Use one exact NangateOpenCellLibrary_typical.lib snapshot from the official OpenROAD-flow-scripts Nangate45 platform [R10]. Pin the containing Git commit and the downloaded file's SHA-256. Inspect its license, area attributes, timing/capacitance units, and cell models. The planner verified the official location but could not retrieve its full bytes through the browser's size limit. Do not invent a library checksum or assume its contents.

The mapping recipe is one frozen native Yosys/ABC flow:

    read the generated RTL and check the selected top/parameters
    synth -top TOP -flatten -noabc
    dfflibmap -liberty LIB
    abc -liberty LIB -constr CONSTRAINT_FILE
    clean; check
    stat -liberty LIB
    write mapped Verilog and JSON

This is a command recipe to implement and smoke-test, not a command already run by the planner. Load the library declarations in the mode appropriate to the actual Yosys version. Use one common script, normal optimization, and no retiming or architecture-specific constraints. Require the final hierarchy to contain one flattened design top plus supported technology cells, with every reported port audited. Record all expanded commands and warnings. Do not run a second default ABC mapping inadvertently before the intended Liberty mapping.

Use BUF_X1 as the common external driving cell if that exact cell exists and is usable in the pinned library; otherwise choose the smallest ordinary noninverting buffer and record that choice before lock. Set external output load to **5 fF** on data and status outputs. Use zero external input/output delay relative to the clock and an ideal single clock. Apply the common driver to nonclock inputs, including start/reset. Keep all operational data/control paths constrained. Do not read parasitics or claim wire timing. All these are modeling assumptions, not measurements of a fabricated device.

OpenSTA reads the same library and exact mapped netlist. Verify units with report_units. Create an initial ideal clock with period 10 ns, converting that number to the verified command units; this merely initializes timing analysis before extracting each design's minimum period. Its documented no-parasitics behavior uses cell timing and pin capacitances [R9]. Use check_setup, report_checks, and report_clock_min_period with port paths included, or an explicitly verified equivalent using actual worst slack. report_wns clips positive slack and must not be used to infer a minimum clock period.

Timing smoke tests must establish that combinational reconstruction, decode, input capture, clock-to-Q, setup, output load, and control paths are covered. Report an invalid timing model if there are unexplained unconstrained endpoints, loops, missing cell models, or material slew/capacitance violations. Do not conceal them with false paths. A synchronous reset is part of the specified control path; this study needs no asynchronous-reset timing exceptions.

Report cell area in the library's verified units, plus sequential/combinational area and actual FF counts. It is a **mapped cell-area estimate**, not die area, fabricated silicon, placement utilization, or an energy estimate. Generic Yosys cell totals, RTL operator counts, and register declarations are not interchangeable with it.

### 8.4 Comparison rule

For each primary configuration j=(W,d1,d2), let N_j be the candidate's prescribed update count. Both digit-serial conventional baselines receive e_j bits per update.

From valid OpenSTA results for msb_serial and lsb_serial, set

    T_budget(j) = 1.10 * min(Tmin_msb(j), Tmin_lsb(j)).
    L_budget(j) = N_j cycles.
    II_budget(j) = N_j+1 cycles.

This is a predeclared baseline-only rule for a common clock/service envelope. It is not adjusted to accommodate candidate gains. It permits 10% clock-period slack around the faster conventional digit implementation while testing area as the main objective. Development must demonstrate that timing extraction is precise enough to apply it.

Compute T_budget only when **both** digit-baseline timing results are valid. A mandatory comparison row with missing correctness, mapping, or timing is unknown, not ineligible. A configuration-level primary win requires every mandatory comparison row to be resolved. A completed, valid timing/service constraint failure can make a row ineligible; a crash, missing measurement, or defective baseline cannot.

A baseline is eligible when it is correct, fully mapped, has valid timing with Tmin<=T_budget, L<=L_budget, and II<=II_budget. Define A_best(j) as the smallest total area among eligible independent, msb_serial, lsb_serial, inverse, and reciprocal implementations. At least the fastest valid digit baseline is eligible by construction. Show every method, including ineligible ones, in the table. A one-cycle constant multiplier may be valuable despite failing the common-clock guard; report its natural latency and Tmin rather than hiding it.

The candidate qualifies for a **configuration-level material improvement** only when it meets the same constraints and

    A_shared(j) / A_best(j) <= 0.90.

The 10% area threshold is an engineering screening margin: a smaller model-specific difference does not justify investing in physical implementation and stronger IP comparisons. It is not a confidence interval or universal economic threshold. It is selected before any comparative result, not fitted to observed gains. Record raw ratios to sufficient precision; do not round a 9.6% gain into success.

A **family-level supported result** requires material improvement at both widths for at least three of the six pairs, including at least one nonsparse-alpha pair from {(3,11),(5,7),(7,11)}. The entire matrix must be reported. This is a predeclared breadth requirement for the chosen finite set, not a population-level statistical test. A gain only for (3,5), or only at one width, is a narrower case result and does not meet the main family hypothesis.

If the complete valid matrix has no material improvements, the primary practical hypothesis is contradicted within this construction, mapping model, and matrix. If gains occur but do not meet the breadth rule, report the exact supported cases and that the family criterion was not met. This is more informative than declaring all partially favorable evidence inconclusive.

An area-only improvement with failed or missing timing does not meet the primary claim. A missing strongest baseline also prevents that claim. If one or more central rows are invalid or missing, do not construct a complete-case success average; state which conclusions remain possible and classify the primary question as inconclusive or invalid as appropriate.

### 8.5 Analysis and uncertainty

Use deterministic per-configuration tables and Pareto plots of total mapped area versus Tmin, annotated by L and II. Also show estimated transaction time L*T_budget and sustained acceptance spacing II*T_budget for eligible designs. Keep cell-clock delay and transaction latency separate.

Report the decomposition explaining the effect: sequential versus combinational area, live input/output/state/control bits, reconstruction and decode logic, coefficient sizes/CSD weights, and the critical-path location. Do not add isolated module areas and call the sum the flattened design area: flattening changes sharing. Use full-top area as authoritative; use separately synthesized kernels only as labeled diagnostics.

There is no statistical sampling error in one deterministic mapped netlist under a frozen toolchain. Do not bootstrap thousands of arithmetic inputs into confidence intervals for area, and do not repeat deterministic synthesis seeds merely to manufacture error bars. Across the twelve configurations report the range, median, geometric-mean ratio, and complete cell-level results as descriptions of this finite set, with no universal inference.

One duplicate clean synthesis of a development configuration checks determinism. If hashes/metrics differ unexpectedly, resolve tool randomness before lock or fix and record the seed; at most three repeats are allowed. Synthesis wall time is an engineering feasibility metric, not circuit execution time.

Uncertainty is primarily model and implementation dependence: zero wire RC, one library corner, chosen synthesis algorithms, fixed schedules, and a small denominator set. Any positive recommendation must acknowledge these. Physically routed timing/power and optimized vendor/FloPoCo IP are excluded; a positive result merits comparison against those before a silicon claim [R5].

### 8.6 Failures, exclusions, and measurement boundaries

Every attempted job has an immutable status: pass, arithmetic_fail, protocol_fail, proof_unknown, synthesis_error, timing_invalid, timeout, resource_stop, or blocked. Preserve logs and first counterexamples. Distinguish a primary row not attempted because a gate stopped the study from one that failed.

A timed-out synthesis supplies no area value. A formal timeout supplies no correctness proof. A timing violation is an observed design constraint failure if the timing model is valid; a missing model is missing evidence. Never impute a win, zero area, or infinite area for an uncompleted tool run.

Measure per-stage wall time with a monotonic clock around the complete subprocess tree, including startup and file I/O. Include coefficient generation, test generation, code generation, compilation, preprocessing, synthesis, and analysis in the resource ledger. First tool executions are recorded as cold; warm repeated probes, if any, are labeled separately. Do not flush system caches or compare mixed cold/warm host timings as hardware speed.

Use process-tree RSS sampling and, where available, macOS /usr/bin/time -l for independent peak-RSS reporting; verify its units. Monitor subprocess children as well as the parent. Record sampling interval and its inability to capture every instantaneous peak. GPU/device synchronization is inapplicable: this is a CPU-only integer/EDA study.

There is no runtime inverse/preprocessing amortization claim. Constants are generated once per circuit configuration, and their entire generation/synthesis time is reported separately from the hardware per-transaction cost. If discussing deployment amortization, explicitly state the assumed transaction count; do not subtract coefficient generation, preprocessing, or exploratory execution from the experimental budget. Dependency installation has the separate setup allowance defined in §9.

## 9. Mac feasibility, dependencies, and resource accounting

### 9.1 Inspect before allocating

First record the actual chip/model, arm64 versus translated process execution, macOS version, physical RAM, current memory pressure, available disk, Python path/version, and installed tool paths/versions. Useful native commands include uname, sw_vers, sysctl -n hw.memsize, df -k, vm_stat, and the hardware overview from system_profiler. Retain the chip field, not machine serial numbers or personal identifiers.

Do not infer RAM, free storage, OS, or tool compatibility from M3 Max. If the experiment is running elsewhere, prepare code, perform source analysis, and document blockers, but do not launch the PoC workload and label it an M3 result. The authorized experimental compute target is the user's one Mac.

Use a native CPU-only stack:

- Python 3.11 or later, standard-library integers, Fraction, unittest, JSON/CSV, subprocess and hashing.
- Icarus Verilog and vvp for RTL simulation.
- Yosys with its ABC binary and built-in SAT engine.
- One pinned Nangate45 typical Liberty and usable functional cell models.
- OpenSTA for the primary timing condition.
- psutil for process-tree monitoring if available; otherwise a documented native ps/time-based monitor. Matplotlib is optional for report figures, not required for the decision.

No GPU, Metal, CUDA, neural models, remote workers, paid services, FPGA board, proprietary PDK, web service, or database is part of the experiment. Keep arithmetic exact; do not substitute floats to improve speed.

### 9.2 Bounded installation and capability gate

Prefer existing working native tools. Otherwise use a dated darwin-arm64 OSS CAD Suite release from the official repository, installed in the project tool directory, recording the archive hash and all tool versions [R6,R7]. Its detailed README and the overview documentation have differed on minimum macOS versions; inspect the actual release requirements instead of assuming compatibility. Homebrew Yosys/Icarus packages are an alternative when already available to the user [R8].

OpenSTA is not assumed to be in OSS CAD Suite. Its official repository documents a native macOS build using CMake and suitable Tcl/Flex/Bison dependencies [R9]. Follow the pinned revision's README/Brewfile, use at most two build jobs, and prefer a project-local executable. Do not invent a brew install opensta command. Do not spend the study building an entire physical-design flow.

The total setup allowance is **90 minutes**, including downloads, builds, failed installs, and one repair attempt; at most **60 minutes** of this allowance may be spent on OpenSTA. Routine user-space dependency installation and source retrieval are authorized. Respect actual permissions; if an action is denied, finish unaffected work and report the precise unavailable capability rather than silently changing platforms. Dependency downloads/builds and installation retries consume setup time. Once installed, capability probes consume G0 experiment time and their respective synthesis/formal attempt slots; count each probe once.

Before the arithmetic workload, run:

1. A tiny Verilog compile/simulation, including a synchronous reset.
2. A tiny SAT proof and a deliberately false property that produces a counterexample.
3. A tiny registered-adder synthesis into the actual Liberty; verify finite nonzero cell area and known-cell coverage.
4. An OpenSTA smoke test whose critical delay grows when a real logic stage is added.
5. A functional cell-model check for the postmapping verification route.

Pin exact executables, library bytes, scripts, and dependency versions only after they pass these probes. The commands are available interfaces to verify, not tools the planner has already installed on this Mac.

### 9.3 Hard caps

| Resource | Ceiling and rule |
| --- | --- |
| Experimental wall time | 420 minutes of cumulative local stage execution, including all executed development trials, unsuccessful runs, repairs, and designated analysis/report work; the last 35 minutes are reserved for evidence preservation/reporting |
| Setup wall time | 90 minutes cumulative; separate from the 420-minute experiment cap |
| Memory | Aggregate experiment process-tree RSS cap=min(8 GiB, 25% of installed physical RAM); reduce further if existing memory pressure demands it |
| Minimum usable memory | If the chosen cap is below 2 GiB, do only analytic/reference preparation unless calibration proves the required small flow fits |
| Disk | Project-managed footprint <=12 GiB: tools/dependencies <=8 GiB, retained results <=2 GiB, disposable intermediates <=2 GiB |
| Free-space reserve | Require approximately 16 GiB free before full setup, and maintain at least 4 GiB free; use measured extracted sizes to stop earlier |
| Concurrent heavy jobs | One synthesis/formal/simulation job at a time; at most two compiler threads during dependency builds |
| Synthesis attempts | <=132 total, including smoke tests, alternate encodings, failed jobs and repairs |
| Per-synthesis limit | 120 seconds wall; OpenSTA on its result <=30 additional seconds |
| Formal attempts | <=64 total, <=120 seconds each, including decompositions and retries |
| RTL simulation | <=2,000,000 accepted test transactions across all methods/reruns, and <=50,000,000 simulated clock cycles |
| Repair allowance | <=25 minutes of experimental wall time and <=12 synthesis attempts, included in the totals, not an extra budget |
| Waveforms | Disabled on passing bulk runs; at most 20 short failing/debug traces, <=100 MiB total |

These ceilings are conservative planning choices, not measured requirements. The 420-minute allowance is a **cumulative stage-execution clock, not a continuous seven-hour deadline starting before code authoring**. Charge elapsed time for every local study command, experimental debugging rerun, and explicit analysis/report interval; avoid double-counting overlapping intervals. Hosted-agent reasoning and source-code authoring effort are estimated separately. Never pause the experiment clock while a child experiment continues, or reclassify exploratory executions as free authoring work.

A persistent ledger records consumed wall time, attempt counts, peak RSS, and retained bytes. A stage transition, tool restart, new seed, encoding change, or fallback does not reset it. Define a synthesis attempt as a mapped resource-measurement run; SAT preprocessing belongs to the formal-job ledger unless it also launches a separate mapped measurement. Terminate an over-budget subprocess tree, retain its log/status, and continue to diagnosis. Reserve time to write the report even when the main experiment cannot finish.

### 9.4 Calibration and scaling rule

Measure representative development configurations before committing to 72 primary mappings. Cover the sparse /3,/5 circuit, nonsparse /5,/7 circuit, and larger-constant /5,/17 circuit, including the reciprocal baseline. Record separately code-generation, simulation compilation/execution, synthesis, STA, and proof costs.

Circuit storage is O(W) for these fixed constants. Serial digit logic is approximately O(e*log d) per channel; full-word reconstruction and constant products can dominate graph and multiplier cost. Formal solver time is not reliably polynomial in W. Do not extrapolate it from raw arithmetic operation counts.

For a conservative initial projection, use four times the largest W=8 synthesis time for each W=16 job and sixteen times for W=32, plus measured tool startup/STA overhead. This quadratic allowance is a planning safeguard, not an asserted scaling law. Replace it with measured stage costs as they become available, without changing the scientific matrix opportunistically.

Proceed to the full locked matrix only if projected completion, verification of potential winners, and the reserved report time fit within 75% of the remaining experiment budget. This leaves 25% headroom for modeling error. Stop launching new work when any hard cap or the reporting reserve would be threatened.

If the full matrix does not fit, the **only predeclared reduced confirmation** is all six methods at W=16 and W=32 for pairs (3,5) and (5,7): 24 designs. Freeze that reduced scope before examining its comparative gains. It can establish case-specific results, not the family-level success criterion. Do not shrink to W=4 toy circuits or remove strong baselines to claim hardware viability.

If even the reduced matrix cannot fit, or W=32 is indispensable but cannot be validated, classify practical infeasibility or inconclusive evidence as appropriate and report the measured calibration. Do not try a second architecture family as a rescue campaign.

**Planning effort estimate, unmeasured:** setup roughly 0.5–1.5 hours, implementation and integration roughly 3–6 hours of agent work, active experimental compute roughly 1–3 hours if these small netlists calibrate favorably, and analysis about an hour. The full caps allow up to 8.5 hours of setup plus experimental execution. Coding/review time is additional; expect roughly a working day of end-to-end attention, potentially longer if tool setup or verification fails. No hosted estimate establishes Mac performance.

## 10. Evidence gates and autonomous actions

The following stage allocations sum to the 420-minute experimental cap. They are maximum allocations, not instructions to spend all available time. Reallocate unused minutes only within the global cap and keep the final 35-minute report reserve. Setup has its separate 90-minute cap.

| Gate | Question and prerequisites | Minimum work/evidence | Max minutes | Decision and next action |
| --- | --- | --- | ---: | --- |
| G0: access and construction | Are the source, target Mac, toolchain, and exact function available? | Environment/source manifest; dependency capability probes; canonical coefficient certificate; applicability rejections | 10 | Continue if coherent. Missing primary PDF with available TeX is acceptable. Missing Mac blocks workload. Missing timing/library invokes only the fallback below |
| G1: exact model | Does the finite algorithm implement Q,R for arbitrary inputs? | Independent Python models, exhaustive tiny cases, decoder proof checks, premature-stop/even-denominator controls | 20 | Zero mismatches required. One bounded repair allowed. An unrepaired contradiction stops RTL/performance work |
| G2: tiny RTL correctness | Does synthesizable HDL preserve widths and the transaction contract? | All six minimal tops, tiny/irregular simulations, protocol tests, required small formal checks, visible mutation failures | 70 | Continue only with zero known mismatches. Repair ordinary coding defects within reserve. A failed reference invalidates comparison, not the graph theorem |
| G3: resource calibration | Can a meaningful W16/W32 comparison fit? | Representative W8 timing/memory, synthesis and STA smoke checks, forecast and budget ledger | 20 | Apply §9.4. Choose full matrix, predeclared reduced matrix, or stop before expensive infrastructure |
| G4: mechanism pilot and lock | Can the measurements expose the predicted state-versus-logic tradeoff? | Bounded encoding comparison on development cases; verified cell/FF measurement control; family implementations frozen | 30 | Freeze even if candidate loses. An unexplained measurement anomaly blocks confirmation. A small or noisy gain is not a reason to extend tuning |
| G5: locked comparison | Is there a material eligible improvement, and where? | Complete selected matrix, raw outputs/logs, correctness statuses, required proof/mapping checks for potential winners | 180 | Apply §8 mechanically. Preserve losers and misses. Stop for a fatal correctness/model defect; one versioned repair within remaining caps |
| G6: bounded diagnosis | What caused the result; is one prescribed robustness check needed? | Area/state/path breakdown; modular-only /3,/5 diagnostic or fixed W64 robustness if decision-relevant | 30 | Run only the stated checks needed to resolve a concrete uncertainty. No new denominators, graph search, or iterative retuning |
| G7: report | What is justified, including an early stop? | Reproducible report, tables, manifest, raw failures, and exact conclusion scope | 35 | Always complete; no further experiments consume this reserve |
| Shared repair reserve | Can an ordinary implementation/tool compatibility issue be fixed once? | Error, reason, patch, affected reruns, updated ledger | 25 | Global limit includes all repairs and fallback attempts; if exhausted, report blocked/invalid/inconclusive evidence honestly |

The synthesis count is independently capped. Its planned maximum is 8 small flow/control/determinism jobs, 30 development jobs (18 default plus 12 alternate arithmetic encodings), 72 primary jobs, 4 modular diagnostic jobs, 6 W64 robustness jobs, and 12 repairs: **132**. The eight small slots cover a registered adder, its longer-path version, two observable-FF controls, one output-removal copy, one clean determinism duplicate, one arithmetic smoke, and at most one separate functional-model smoke. Reuse compatible netlists for multiple checks; a repeated smoke still consumes an attempt. Reuse calibration results from development jobs rather than launching free additional attempts.

### 10.1 Predeclared capability fallbacks

- **OpenSTA unavailable or timing smoke invalid, but valid Liberty mapping available:** complete exact models/RTL checks and, if affordable, the 24-design reduced matrix with CSD encoding. Report mapped area and cycle counts only. Do not apply the primary timing-qualified success label; the primary question is inconclusive because timing is missing. There is no replacement of delay by RTL depth in nanoseconds.
- **Liberty unavailable, but Yosys works:** complete exact checks and at most 12 generic syntheses, W=16 for pairs (3,5),(5,7), all six methods. Report generic cells/FFs as structural diagnostics, never physical or standard-cell area. This does not resolve the main area hypothesis.
- **HDL/synthesis tools unavailable:** finish exact reference models, coefficient certificates, and a concise runnable implementation package if possible; identify the unperformed HDL/resource validation. Do not promote Python cycle tests to hardware verification.
- **Essential source contradiction:** preserve a minimal exact derivation/counterexample, identify the source location and assumptions, and stop the affected branch. This can be a completed analytic result without a large benchmark.
- **Target Mac absent:** deliver prepared code/analytic work and an explicit list of blocked validations. Do not benchmark on remote compute or another host as a substitute.

Fallback work consumes the same ledger and repair limits. An installation timeout is not scientific evidence against the arithmetic mechanism.

## 11. Minimal repository and executable interfaces

Create a small repository with ordinary scripts. No service, dashboard, job queue, graph database, or distributed framework is needed.

~~~text
README.md
protocol.json
run.py
src/
  arithmetic.py         # coefficients, exact models, independent oracles, decoder tables
  rtlgen.py             # six top generators and shared sized-constant emitter
  verification.py       # vectors, protocol schedules, SAT miters, mapping checks
  measure.py            # bounded subprocess runner, RSS/time/disk ledger, parsers
  analysis.py           # eligibility, tables, figures, report generation
flows/
  synth.ys              # parameterized, emitted per job; frozen recipe
  timing.tcl            # frozen OpenSTA constraints and diagnostics
tests/
  test_arithmetic.py
  test_protocol.py
  tb_transaction.v
sources/
  manifest.json         # original names, hashes, exact internal paths, read status
  literature.md         # references and access qualifications
generated/
  coefficients.json
  rtl/
  miters/
results/
  environment.json
  attempts.jsonl
  budget.json
  protocol_lock.json
  raw/
  measurements.csv
report/
  REPORT.md
  tables/
  figures/
~~~

Keep original inputs outside generated/output paths. Never overwrite the supplied JSON summaries. The source scripts that write beside themselves may be run only on copies, and only if a specific source discrepancy requires it; rerunning all historical enumerations is unnecessary for this hardware study.

### 11.1 Key interfaces

- coefficients(W,d1,d2) returns all integers in §4, the exact transfer certificate, supported-domain checks, and a deterministic JSON representation.
- model(method,W,d1,d2,x1,x2) returns Q1,R1,Q2,R2 and, for recurrent methods, an optional short state trace.
- generate_top(method,configuration,encoding) returns synthesizable Verilog plus a manifest of ports, register intent, constants, L and II. Code generation must be deterministic.
- generate_vectors(configuration,split) returns integer/hex vectors and expected outputs from divmod, with stable IDs and provenance.
- run_attempt(kind,command,limits,configuration_id) records start/end, environment, exit status, peak memory, log path, and cumulative cost even on failure.
- assess(measurements,lock) applies exactly §8 and produces every primary row, including failures and ineligible results.

Use JSON for configuration and attempt metadata, JSONL for append-only attempts, and CSV for comparison rows. Allow a resume mode that skips a completed job only if its source/config/tool/library hashes match. Changes create new attempt IDs; no result silently replaces a failure.

### 11.2 Configuration to instantiate

The initial protocol file contains these concrete values; it may add actual paths/versions discovered by inspection:

~~~json
{
  "study": "rational_feedback_area",
  "primary_widths": [16, 32],
  "primary_pairs": [[3,5],[3,7],[3,11],[5,7],[5,17],[7,11]],
  "development_widths": [8],
  "development_pairs": [[3,5],[5,7],[5,17]],
  "methods": ["shared","independent","msb_serial","lsb_serial","inverse","reciprocal"],
  "lift_modulus": 16,
  "exploit_actual_v2": true,
  "primary_endpoint": "registered_exact_quotient_and_remainder",
  "area_ratio_threshold": 0.90,
  "timing_guard_factor": 1.10,
  "family_min_pairs_both_widths": 3,
  "seed_development": 20261102,
  "seed_confirmation": 20261103,
  "seed_robustness": 20261104,
  "setup_seconds": 5400,
  "experiment_seconds": 25200,
  "report_reserve_seconds": 2100,
  "synthesis_attempt_cap": 132,
  "formal_attempt_cap": 64,
  "synthesis_timeout_seconds": 120,
  "sta_timeout_seconds": 30,
  "formal_timeout_seconds": 120,
  "repair_seconds": 1500,
  "repair_synthesis_attempts": 12,
  "rtl_transaction_cap": 2000000,
  "rtl_clock_cycle_cap": 50000000
}
~~~

### 11.3 Commands you must implement

The following are **intended project commands**, not commands verified by the planner. Build the CLI so these work, and make each exit nonzero on failed required checks. Keep tool installation separate from run-all so retries do not repeat setup silently.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python3 run.py inspect --config protocol.json
python3 run.py sources --config protocol.json
python3 run.py math --config protocol.json
python3 -m unittest discover -s tests
python3 run.py generate --config protocol.json
python3 run.py verify --split development --config protocol.json
python3 run.py calibrate --config protocol.json
python3 run.py pilot --config protocol.json
python3 run.py lock --config protocol.json
python3 run.py confirm --lock results/protocol_lock.json
python3 run.py diagnose --lock results/protocol_lock.json
python3 run.py report --config protocol.json --lock results/protocol_lock.json
~~~

Also implement:

~~~bash
python3 run.py smoke --config protocol.json
python3 run.py all --config protocol.json --resume
~~~

The smoke command executes only the coefficient check, a small RTL transaction test, one mapped arithmetic design if available, and a report stub with honest status. It is not the final experiment. The all command follows gates automatically, invokes only justified optional work, and always reaches reporting even after an early stop. Before any lock exists, use python3 run.py report --config protocol.json; the lock argument is optional. That early-stop route reports the reached gates and actual evidence without fabricating a lock or comparison rows.

Install only the missing lightweight Python dependencies actually used, and freeze their exact versions/hashes in the environment manifest. Include the native EDA setup commands that actually worked on the Mac in README. Do not write guessed package commands as verified instructions.

The planner verified the availability of the relevant documented tool interfaces, not their exact compatibility on this laptop. Initial help/capability probes must resolve any syntax differences. If a tool version lacks an intended proof/report option, use a demonstrably equivalent supported route within the repair budget and record it; otherwise retain the corresponding blocked status.

## 12. Outcome interpretation and final artifacts

### 12.1 Required outcome categories

| Outcome | What supports it here | What it does not imply |
| --- | --- | --- |
| Supported contribution | Complete valid evidence meeting the predeclared family rule, or explicitly narrower validated case improvements; exact scope and model stated | New arithmetic, global hardware optimality, or physical silicon advantage |
| Informative negative result | Correct implementation and adequate comparison show no qualifying benefit in the stated scope; or a verified mathematical obstruction rules out a proposed interpretation | Refutation of the rational feedback theorem or all possible graph implementations |
| Practical infeasibility | Measured target-machine costs/scaling prevent the intended meaningful workload or implementation within caps | Mathematical impossibility or proof that a faster engineering implementation cannot exist |
| Inconclusive evidence | Missing timing/source/hardware, incomplete comparisons, proof uncertainty for a prospective win, or insufficient budget prevents the decision | Support for either superiority or inferiority |
| Invalid experiment | Wrong arithmetic, baseline, ports, widths, protocol, constraint model, or measurement procedure invalidates a comparison | A negative scientific result |

Use one primary classification and subordinate findings when necessary. For example, exact arithmetic may be supported while the area question is inconclusive because STA could not be installed. A partial case gain may be valid even though the family criterion is not met. A known bug must be named and cannot be buried under an overall pass.

### 12.2 Final report content

Lead REPORT.md with the research decision: whether the tested sharing construction warrants further circuit work, and why. Include:

1. Exact input/output and hardware/service scope, methods actually executed, and any gate that stopped the study.
2. The paper-to-algorithm bridge, its proof/certificate, and the distinction between graph feedback, finite-word recurrence, and actual hardware state.
3. Actual Mac/OS/toolchain/library manifest, timing/load assumptions, and verification coverage with unproved or unavailable checks identified.
4. Every primary configuration and method: total/sequential/combinational area, FFs, Tmin, L, II, eligibility, verification status, failure status, and area ratios where defined.
5. Development/tuning record, lock hash, material deviations, failures, cumulative time/memory/disk/attempt counts, and preprocessing costs.
6. Mechanism explanation supported by mapped paths and resource breakdown. Say whether any gain survives the strong serial baselines and exact output conversion.
7. Limitations: fixed small denominators, unsigned bounded inputs, unreduced rational output, one synthesis library/corner, no routing/power/device validation, omitted optimized IP, and no universal superiority claim.
8. Exactly one justified next action if further work is worthwhile. For a positive model result, the priority is comparison with optimized constant-divider IP and physical implementation of the winning case. For a clear loss, identify the measured dominant cost and stop unless one specific bounded change addresses it.

### 12.3 Deliver only artifacts justified by reached stages

For any implemented stage, deliver runnable source, concise setup/usage instructions, the actual configuration and protocol lock if one was reached, exact test vectors/generator and manifests. For completed experiments, retain raw logs and measurements, essential comparison tables/plots, certificates/proof statuses, failed-run records, and the report. Provide a single command regenerating essential tables/plots from stored measurements without rerunning hardware tools.

For an analytic stop, deliver the verifiable derivation or minimal counterexample and scope; no synthetic benchmark is needed to fill a checklist. For an access stop, provide prepared unaffected work and the exact blocked resource/validation list.

Do not include giant dependency trees or bulk passing waveforms in the deliverable. Record official dependency sources and hashes so they can be restored. Preserve relevant licenses and source attribution. Inspect the generated report against the raw records before declaring completion.

## 13. Research coverage, alternatives, and review findings

### 13.1 Research coverage

Targeted research was performed on **2 November 2026**. Search and source-reading topics were odd-constant/exact division, low-word inverses, 2-adic bit-serial circuits, multiplierless constant arithmetic, conventional constant-divider generators, and native macOS synthesis/formal/timing tools. Author papers and official implementation/documentation pages were read; key formula/figure pages of the Vuillemin and Granlund–Montgomery PDFs were inspected directly.

Backward coverage followed the exact-division literature through the GMP implementation discussion; contemporary implementation coverage checked FloPoCo's documented constant-divider families and official EDA tooling. No exhaustive forward-citation or global novelty survey was attempted, because the decision here concerns a bounded chip implementation rather than priority of the supplied mathematical theorem.

Jebelean's 1993 exact-division paper was found at its primary publisher abstract, but full publisher access was blocked; no uninspected proof is relied on. The full Nangate45 file exceeded the browsing content limit, so its actual bytes/license/units remain G0 checks. Tool installation and chip mapping were not executed during research.

### 13.2 Why this project was selected

| Alternative | Reason not selected as the primary project |
| --- | --- |
| Literal finite-tree host for rational arithmetic | Supporting theorem excludes universal Q representation; finite-ring embeddings may have large encoding/host cost and answer a different question |
| General compiler for every rational matrix | Denominator decomposition, coefficient growth, graph optimization, and extra boundary semantics would obscure the chip mechanism and exceed the smallest useful experiment |
| General rational numerator/denominator ALU | Unbounded growth, GCD normalization, variable denominators, and application workload choice would dominate; not established by the existing PoC |
| Pure modular /3,/5 benchmark | Useful diagnostic but insufficient for the stated rational-output purpose; risks comparing encodings rather than complete arithmetic |
| Exact integer division only on multiples of d | Practical in some exact-arithmetic software [R3], but would omit arbitrary fractional results and could favor the inverse route by promise |
| Full physical ASIC or FPGA implementation immediately | Unnecessary before correctness and synthesis evidence; would add routing/platform/package constraints and setup risk |

The selected study does not prove that paired constant division is the only useful application. It is the smallest circuit-level test that preserves the supplied sharing mechanism, includes complete ordinary rational semantics, and can expose a meaningful limitation without building a broad system.

### 13.3 Material review findings and dispositions

Three independent agents reviewed the supplied sources and subsequently read the complete draft in three distinct passes: **mathematical/source-fidelity/novelty review, adversarial experimental review, and empty-project execution review**. Their checks are distinguished from source-reported results in §2.3. The mathematical pass found no essential defect in the selected derivation. The other passes identified actionable issues in flattening, positive-claim proof requirements, timing initialization, early-stop reporting, and resource accounting; those issues were repaired. The following table records material findings and dispositions, not a transcript of the review:

| Stress scenario / review finding | Disposition in this brief |
| --- | --- |
| The theorem is nonconstructive or hides expensive search | Selected closed form avoids general group enumeration, graph search, and optimal addition chains; exact coefficient certificate is an early gate |
| Algebraic equilibrium is assumed to converge physically | Nilpotent modular error proof and exact N schedule; no combinational loops or real convergence claims |
| Even denominators do not invert modulo 2^W | Explicit counterexample and domain rejection; even-denominator arithmetic is outside this finite adaptation |
| The digital mechanism is already known | Direct references to 2-adic serial circuits, inverse multiplication, and shift/add synthesis; contribution is the controlled sharing-cost result |
| Only the favorable /3,/5 case works | Six fixed pairs, two widths, nonsparse-alpha breadth requirement, and case-level reporting |
| A strong baseline removes the gain | MSB and LSB small-carry implementations are mandatory, as are inverse/reciprocal comparators; comparison to old independent RTL alone is insufficient |
| Quotient alone loses the fractional result | Q,R contract and wide integer identity preserve exact rational value |
| Residue conversion dominates | Included before registered outputs in the primary metric; same decoder for relevant methods and carry reuse for LSB baseline |
| Register-count advantage hides input/control/logic costs | Whole-top mapped area and timing, no forced redundant baseline state, and no FF-only success criterion |
| Module boundaries change apparent area savings | Explicit common flattening before mapping, one design top plus technology cells, and port audit |
| A proof checks a mathematical copy while RTL is wrong | Formal miters instantiate the actual sequential generated RTL; claimed winners require functional proof and mapping equivalence |
| Missing comparator is silently removed from the minimum | Unknown results block a primary win; both digit timings are required to set the service envelope |
| Finite precision changes the mathematics | Explicit Z/(2^W) semantics, sized literals, exact wide decoding, irregular-width tests, no floating-point approximation |
| Preprocessing dominates | Constant generation, code generation, and mapping measured separately and charged cumulatively; no amortization claim |
| Memory/runtime exceeds estimates | Cheap calibration, explicit projection, fixed reduced matrix, process-tree cap, preserved infeasibility outcome |
| Attempt/time caps omit controls or conflict with coding effort | Eight explicit smoke/control slots; 132 cumulative mappings; separate setup, cumulative local-stage, and authoring clocks |
| An early stop occurs before protocol lock | Reporting accepts the initial configuration without fabricating a lock |
| Small pilot is inconclusive | Pilot validates correctness/measurement and fixes encoding; a small gain/loss does not trigger indefinite tuning or a scientific verdict |
| Essential source/tool access is missing | Main TeX recovery resolves the missing PDF; capability-specific fallbacks retain narrower evidence and identify what remains blocked |
| Main hypothesis is false | Complete unfavorable results, scoped negative conclusion, and no automatic pivot to another project |

Residual risks are synthesis-model dependence, the bounded encoding family, difficulty of full-width formal proofs, and unverified target-machine dependency setup. They are explicit gates or limits, not hidden assumptions. The absence of a discovered essential proof defect is not proof-assistant formalization or external refereeing.

## 14. References and source provenance

All web resources below were accessed during planning on 2 November 2026 unless stated otherwise. URLs are ordinary external references; the source locations in [A1–A3] refer to the supplied files. Source facts, the brief's derivations, planning estimates, and untested hardware hypotheses are distinguished in the main text.

**[A1] User-supplied research note.** *Arithmetic from Binary Replacement: Positive bags, coherent substitutions, and the exact cost of rational feedback*. 2 November 2026. Displayed manuscript byline: Research note from an autonomous mathematical investigation. Recovered file: arithmetic_research_bundle(2).zip / arithmetic_from_binary_replacement.tex. Critical locations: §2.1 Lemma 2.1 and equations eq:rule/eq:presentation, lines 100–163; §4 Proposition 4.1, lines 350–394; §6 Theorem 6.1 (thm:main), lines 447–626; §6.5 eq:primeoverlap and equilibrium warning, lines 628–656; §7 evidence, lines 658–734; reproducibility appendix, lines 884–901. The primary PDF was unavailable. Its source SHA-256 is d7d9bf17fd38d184c311c8c8e6feaf9da5e1668202b3f75ae7d1c871b5c1fa02.

**[A2] User-supplied research note.** *Finite Relations and Arithmetic: Tree representations of linear operators and the feedback cost of additive groups*. 2 November 2026. Same displayed byline as [A1]. Files: finite_relations_arithmetic(2).pdf and finite_relations_research_bundle(4).zip / finite_relations_arithmetic.pdf and .tex. Critical locations: Theorem 1, PDF p.2; construction/normalization pp.3–4; size bound and exclusions pp.5–6; Theorem 6, pp.6–7; empirical evidence pp.7–8. Both PDF copies have SHA-256 2551f49ce3b0a20cc224b8050da404dbd0022cca1baca8697c58c7d065204cc1.

**[A3] User-supplied engineering bundle.** *Arithmetic-from-relations: digital hardware proof of concept*. 2 November 2026. arithmetic_chip_poc(2).zip. Relevant locations: README.md sections Why the iteration finishes exactly, Reconstruct ordinary integer quotients, Architecture comparison; docs/engineering_note.md (same text); docs/poc_engineering_report.pdf, pp.1–2; rtl/shared_feedback_div3_div5.v, independent_feedback_div3_div5.v, shared_feedback_unsigned_div3_div5.v, and registered reference modules; sim/verify_python.py, tb_division.v, run_synthesis.sh, run_hdl.sh; results/python_verification.json and check_run.txt. The source expressly reports Python checks, uncompiled RTL, and no synthesis/physical measurements.

**[R1] Jean E. Vuillemin.** *On Circuits and Numbers*. Author preprint, 13 October 1993; published in IEEE Transactions on Computers 43(8), 1994. [Author PDF](https://www.di.ens.fr/~jv/HomePage/pdf/cirnum.pdf). Critical locations: Proposition 2, PDF p.9, odd-denominator rational/finite-circuit correspondence; §7.3, Example 7, PDF p.19, finite serial multiplication by 1/3. Establishes directly relevant 2-adic hardware prior art, not the paper's particular two-channel sharing result.

**[R2] Torbjörn Granlund and Peter L. Montgomery.** *Division by Invariant Integers using Multiplication*. PLDI, 1994. [DOI: 10.1145/178243.178249](https://doi.org/10.1145/178243.178249). [Author PDF](https://gmplib.org/~tege/divcnst-pldi94.pdf). Critical locations: §4 for unsigned reciprocal division; §9, equations (9.1)–(9.2), PDF p.9, odd modular inverse, low-word exact division, and Newton lifting. Its promised-zero-remainder case must not be confused with arbitrary-input Q,R conversion.

**[R3] GNU MP project.** *GNU MP 6.3.0 Manual, §15.2.5: Exact Division*. Living official implementation documentation, accessed 2026. [Exact Division](https://gmplib.org/manual/Exact-Division). Relevant sections describe low-digit quotient formation and division by 3 in exact arithmetic. The zero-remainder assumption is explicit.

**[R4] Yevgen Voronenko and Markus Püschel.** *Multiplierless Multiple Constant Multiplication*. ACM Transactions on Algorithms 3(2), article 11, 2007. [DOI: 10.1145/1240233.1240234](https://doi.org/10.1145/1240233.1240234). [Author preprint](https://spiral.ece.cmu.edu/pubs/synth.pdf). Relevant locations: §1.1, shift/add/subtract and CSD formulations; subsequent graph-based constant-multiplication framework. Add count is not mapped area, and a single-input multiplier block is not automatically an independent-input channel comparator.

**[R5] FloPoCo developers.** *FloPoCo operator manual, version 5.0.git*, living documentation. [Operator list](https://www.flopoco.org/operators_5.0.git.html). Relevant entry: IntConstDiv, with quotient/remainder outputs and linear, logarithmic, reciprocal multiply/add, and table/add architectures. Inspected as evidence of established optimized alternatives; not installed or benchmarked here. The bounded study does not claim to beat all such implementations.

**[R6] YosysHQ.** *Yosys documentation: Installation; Mapping to cell libraries*. Living official documentation, accessed as 0.70 documentation. [Installation](https://yosyshq.readthedocs.io/projects/yosys/en/stable/getting_started/installation.html). [Cell-library mapping](https://yosyshq.readthedocs.io/projects/yosys/en/stable/using_yosys/synthesis/cell_libs.html). Relevant portions: native supported architectures and the dfflibmap/abc Liberty flow. Record actual installed versions; the documentation version is not a target-machine measurement.

**[R7] YosysHQ.** *OSS CAD Suite Build*, official repository and installation README, living resource. [Repository](https://github.com/YosysHQ/oss-cad-suite-build). [Release downloads](https://github.com/YosysHQ/oss-cad-suite-build/releases). Relevant portions: bundled tools, darwin-arm64 distribution, activation, and current OS requirements. Use a dated release and hash it.

**[R8] Homebrew project and Stephen Williams/Icarus contributors.** *Yosys and Icarus Verilog formulae; Icarus installation documentation*. Living official resources. [Yosys formula](https://formulae.brew.sh/formula/yosys). [Icarus formula](https://formulae.brew.sh/formula/icarus-verilog). [Icarus installation](https://steveicarus.github.io/iverilog/usage/installation.html). Native installation alternatives, subject to the actual macOS and package support.

**[R9] James Cherry / Parallax Software and OpenSTA contributors.** *OpenSTA README, Examples, and Command Reference*. Living official documentation. [Official repository](https://github.com/parallaxsw/OpenSTA). [Examples](https://opensta.readthedocs.io/en/latest/Examples/). [Commands](https://opensta.readthedocs.io/en/latest/Commands/). Relevant portions: Build on Macos/Darwin; no-parasitics timing and units; check_setup; report_clock_min_period; report_worst_slack versus report_wns. These support the specified timing model, not physical accuracy.

**[R10] The OpenROAD Project / Nangate library contributors.** *OpenROAD-flow-scripts Nangate45 platform*. Living official distribution. [Platform directory](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/tree/master/flow/platforms/nangate45). [Typical Liberty location](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/master/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib). Retrieve from a resolved immutable commit, retain the applicable license, and record the actual bytes/hash at G0. Full file contents were not verified during planning.

### 14.1 Original archive hashes

| Original file | SHA-256 |
| --- | --- |
| arithmetic_research_bundle(2).zip | cb1abbb65a86d8699a973b73db3f3e771eebf99ed3584a66aaa20fd353bf9444 |
| finite_relations_research_bundle(4).zip | 68b0fbd49115d0e73293e4ea2f5c9c8ed429f68c10d7b7f60b46bfe3e32eaada |
| arithmetic_chip_poc(2).zip | b2972f42fb43a30ac20a996498e271f2b10a2dadda0a4aa3d7e70318eb40b823 |
| Pasted text(20261102-213139).txt | af7fbd82afca7da659e84145063325210cb7c559c07e1cf5b4a2aaeb5e472fb7 |

If a supplied hash differs, identify the version change and inspect relevant differences. Do not assume that a renamed attachment is a different mathematical result or that a changed source can silently reuse the same lock.

### 14.2 Reproduce the planner's canonical arithmetic check

The following compact check reproduces the 32,760-case calculation in §2.3. It is supplied as evidence for the planning derivation; it is not a substitute for the independent models, RTL, protocol checks, or benchmarks required above.

~~~python
from fractions import Fraction as F

pairs = [(3,5), (3,7), (3,11), (5,7), (5,17), (7,11)]
cases = 0
for ds in pairs:
    D = ds[0] * ds[1]
    h = [D // d for d in ds]
    a = [(-pow(b, -1, d)) % d for b, d in zip(h, ds)]
    c = [b * u for b, u in zip(h, a)]
    P = [[(int(i == j) + h[i] * a[j]) // ds[j]
          for j in range(2)] for i in range(2)]
    for i in range(2):
        for j in range(2):
            assert (int(i == j) + h[i] * a[j]) % ds[j] == 0
            assert F(P[i][j]) - F(h[i] * c[j], D) == F(int(i == j), ds[i])
    t = (-pow(D, -1, 16)) % 16
    alpha = 1 + t * D
    e = (alpha & -alpha).bit_length() - 1
    assert e >= 4
    for W in range(1, 7):
        M = 1 << W
        N = (W + e - 1) // e
        for x1 in range(M):
            for x2 in range(M):
                xs = [x1, x2]
                z = 0
                forcing = t * sum(ci * x for ci, x in zip(c, xs))
                for _ in range(N):
                    z = (alpha * z + forcing) % M
                us = [(sum(p * x for p, x in zip(row, xs)) + b * z) % M
                      for row, b in zip(P, h)]
                for d, x, u in zip(ds, xs, us):
                    k = (d * u) >> W
                    C = (k * M + d - 1) // d
                    Q, R = u - C, d * C - k * M
                    assert d * u % M == x
                    assert (Q, R) == divmod(x, d)
                cases += 1
assert cases == 32760
print("PASS:", cases, "input-pair cases")
~~~

## 15. Completion instruction to the coding agent

Execute the authorized stages autonomously, inspect their actual outputs, and follow the gates. Repair routine implementation/tool issues only within the stated bounds. Preserve failures and unfavorable outcomes. Do not request approval for ordinary reversible choices already fixed or authorized here.

Before reporting completion, verify that all conclusions can be regenerated from the retained evidence, every attempted primary row has a status, setup/experimental costs were counted cumulatively, no blocked validation is described as performed, and no source-level feedback/register count has been substituted for mapped area or physical performance.

A justified early stop with reproducible evidence is a completed research outcome. A favorable-looking graph diagram or a passing Python example is not a completed hardware study.


