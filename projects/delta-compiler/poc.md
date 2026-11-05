# Execution brief: Does exact signed-delta compilation make reusable neural-memory blocks faster?

**Addressed to:** an autonomous coding agent configured as `gpt-6.1-sol` with `xhigh` reasoning.  
**Research/design date:** 4 November 2026.  
**Deliverable you must produce:** the bounded, reproducible study specified below, including a justified early-stop report if a gate fails. This document is the execution instruction; it is not a claim that the PoC or a Mac benchmark has already been completed.

## 1. Project decision, contribution, and research question

### 1.1 Selected project

Implement and evaluate the supplied **stable signed-delta compiler with at most `2r` instructions** for a fixed block of ungated delta-memory updates. Compare it with directly applying the block's affine summary and with portable chunk-WY execution. Run experimental workloads on **one inspected MacBook with an M3 Max**, without remote experimental compute, CUDA-only dependencies, paid computation, or proprietary data.

The practical decision is whether this compiler merits subsequent integration into a neural system that can reuse a fixed memory-loading block and needs only its final state. Seek a useful positive result, but accept a well-supported limitation or negative result as successful research.

**The central issue:** the compiler first constructs an affine map `M -> M P + B`. That same map can already be applied with matrix multiplication. Reducing the number of delta instructions relative to the original stream is insufficient evidence of an acceleration. The experiment must isolate the extra value, if any, of converting that summary into signed instructions.

### 1.2 One primary, falsifiable question

> For full-rank delta blocks of length `n = 8d`, at `(d,p) = (32,64)` and `(128,128)`, does stable signed compilation reduce the total time to prepare a fresh block and apply it serially to `R = 16` distinct incoming memories by at least **20%**, against the fastest validated affine, chunk-WY, or original-replay comparator on the same execution backend, while meeting the fixed numerical contract in Section 5?

The primary population has two deterministic synthetic generators: isotropic coefficients and temporally correlated coefficients from a frozen, untrained neural projection. There are six independently generated blocks in each of the four shape-by-generator strata: **24 primary blocks**. Section 7 defines the generators completely. Additional lengths, exact low rank, and numerical stress cases are controls; they cannot rescue a failed primary claim.

The headline metric is the ratio

\[
\rho_b=\frac{C_{S,b}+W_{S,b}(16)}{\min_{j\in\mathcal B_b}\{C_{j,b}+W_{j,b}(16)\}},
\tag{1}
\]

where `S` is stable signed replay, `C` is preparation of this coefficient block, `W(16)` is the measured serial application time, and `B_b` contains the correct, completed reference methods. A 20% reduction means `rho <= 0.80`, equivalently at least `1.25x` speedup. It does **not** mean `1.20x` speedup.

The 20% threshold is a **planning/application decision**, chosen before measurements: a modest microkernel improvement would not justify a second representation, an SVD compiler, and a restricted reuse interface. `R=16` is an explicit moderate-reuse engineering scenario, not an observed property of deployed models. Report `R=1` and modeled reuse curves separately. Do not increase reuse after seeing unfavorable results to change the primary conclusion.

The primary timing assumes an initialized process with executable shapes already compiled, but **fresh coefficients**. Charge all block-dependent preparation, conversion, and materialization. Also report first-use process/kernel costs and cold totals; a warmed result does not establish a cold-start improvement.

### 1.3 Smallest defensible contribution

The minimum valuable contribution is an accuracy-controlled, reproducible **test of the practical utility of the supplied factorization**, including its preparation and reuse costs, against the affine representation that its own construction requires. A negative result can explain a precise gap between instruction count and runtime. A numerical failure can establish a scoped implementation limitation if independently diagnosed.

The supplied `2r` affine construction, bounded-target injection, and translation-rank optimum are mathematically meaningful. Historical priority of their exact combination remains unresolved. Do not claim a new neural architecture, the invention of signed delta factors, universal contraction factorization, or compact WY batching. The closest verified contraction theorem already gives `3r` generalized Householder factors [R2]; newer work obtains different counts with a richer primitive [R4].

### 1.4 Why the original objective is narrowed

“Make neural nets faster” is broader than the attachments support. Their equivalence is the **completed affine map for every incoming memory**, not all intermediate states, token outputs, gradients, or trained-model quality. A legitimate eventual use could be a support-set loading stage followed by terminal queries, or identical fixed coefficients applied to different memories. This study does not establish that such reuse is common in an existing model.

Explicit exclusions are ordinary all-token prefill, autoregressive look-ahead, training acceleration, differentiating through the compiler, changing a trained model's gates, approximate rank compression, learned sparsification, full-model quality benchmarks, and new custom Metal kernels. Scalar/channel forgetting and complex-valued KDA are also excluded from the implemented recurrence. These exclusions preserve the intended efficiency question while preventing unsupported transfer of the theorem.

If both coefficients **and incoming state** are identical across uses, the appropriate application baseline is a cached final state. Repeated timing of the same arrays is allowed for measurement precision, but is not evidence of useful application reuse.

## 2. Evidence, inputs, and source status

### 2.1 Files to receive

The following user-supplied files were read completely during planning. The notebook contains the mathematical manuscript, proofs, appendix, code, stored results, and figure; no additional paper is needed to understand the selected mechanism.

| ID | Attachment and location within it | Role |
|---|---|---|
| A0 | `Pasted text(20261104-115534).txt` | Original research/handoff requirements. This brief incorporates the project-specific execution requirements. |
| A1 | `delta_memory_research(1).ipynb`, zero-based markdown cells 0–4 | Mathematical source: §1 definitions; §2 Theorem A; §3 Theorem B; §4 Theorem C; §5 limitations; §6 novelty discussion; Appendix A. |
| A2 | `delta_compiler(1).py` | Float64 reference, especially `delta_step`, `affine_summary`, `unrestricted_orthogonal_factors`, `signed_contraction_factors`, `compress_affine_delta_stream`, and `compile_affine_map`. |
| A1-code | Notebook cell 5 | Same source as A2 except the final `if __name__ == "__main__": main()` invocation. Cells 6–8 contain checks and a factor-count figure. |

Planning workspace copies were under `/workspace/scratch/e268eb0a6fee/upload/`. That path describes the planning environment, **not a path expected on the Mac**. Place received copies under `inputs/` in the new repository; preserve the originals and record any adaptation separately.

SHA-256 identifiers:

```text
delta_memory_research(1).ipynb
493ba059aac5da4c3267ea4f2351b320d6d948fff102d8b0124a45bea7068f3e
delta_compiler(1).py
ba221bcc89886ed6e8ab2b08f42910bfddc05d153d8654cc129f70e9e4dc1e5c
Pasted text(20261104-115534).txt
07cd5943e9329803ad7f394ae488a39c3affd0a3097a2a4007032e3bf62cc7ff
```

The definitions and construction are reproduced below so this brief stands alone. If A1/A2 are missing, implement the stated algorithm and complete the independent checks, while labeling source-code reproduction unavailable. Ask for the missing attachment only if an unresolved discrepancy with its claimed behavior becomes essential. Do not invent its content or another paper.

### 2.2 What was actually checked during planning

Three independent agents assisted with mathematics, literature, and experimental design. Their findings were reconciled by the research lead. The source figure was inspected: it plots instruction counts, **not measured speed**. The supplied numerical checks were rerun, and small finite-precision probes were performed on a **hosted Linux x86_64 CPU with NumPy 2.3.5**. These are planning checks, not the PoC, target-device validation, or evidence about M3 performance.

| Planning check actually executed | Observed result |
|---|---|
| A2 `random_stream_checks()` | 160 cases; maximum linear, bias, and state errors approximately `4.57e-15`, `7.96e-15`, `1.40e-14`. |
| A2 `arbitrary_affine_contraction_checks()` | 80 cases; maximum linear and bias errors `6.00e-15`, `6.79e-15`. |
| A2 `gate_cap_random_checks()` | 1,000 inequalities; no positive violation observed. This is not a proof. |
| Translation maps at ranks 1, 2, 3, 5, 8 | `2r` factors and zero observed map error in these cases. |
| Rotating projections, `n=1024` | Two-factor map error `3.58e-14`; second gate approximately `1.99785875`. |
| Appendix A scaled rotation | Displayed algebra agrees numerically to approximately `1.8e-16`. |
| Legal one-dimensional gate `1e-18` | The supplied short stream compiler returns a zero synthetic gate, infinite target, and NaN replay. Stable mode returns the rounded affine map without that division. |
| Nine small float32 near-identity probes | Input-plus-bias-scaled state error at most `2.14e-7`, but error relative to a tiny bias reached about `2677`. Small normwise error does not guarantee preservation of tiny writes. |
| Rotated rank-4 keys in dimension 16, rounded to float32 | Extra singular values approximately `1.48e-8` to `7.70e-8`; float64 numerical rank becomes 16. Truncating back to 4 changes the actual stored-input map. |
| Independent exact scalar WY sign check | Two half-gate writes with values 1 then 0 give exactly `1/4`; the positive-sign triangular formula below agrees. |

The original manuscript's ordinary-stream results, general affine checks, and these probes do not measure a trained neural network. **All Mac timings, backend checks, the locked experiment, and its interpretation remain work for you.**

### 2.3 Compact claim/evidence map

| Central claim | Evidence status and source | Remaining uncertainty / resolving check |
|---|---|---|
| A contractive affine map has a replay of at most `2r` signed delta instructions with bounded target energy | Exact-real constructive proof in A1 §2; independently audited algebra | Float precision, implementation order, and construction cost: Sections 5 and 8. |
| Strict contractions admit `2r-1` factors | Exact-real proof in A1 §2 | Small write denominator can round to zero; diagnostic only. |
| Pure translation requires `2 rank(B)` factors | Exact-real rank argument and construction in A1 §3 | Not an optimality proof for all strict contractions. Tiny translation checks suffice here. |
| Positive gates can require arbitrarily many exact factors even in dimension two | A1 §4; established operator geometry [R5,R6], older related factorization work [R7] | Useful instruction-count control, not runtime evidence. |
| Signed transitions and generalized Householder products are prior art | Verified relevant sections of [R1–R4] | Historical priority of the entire supplied affine statement is unresolved and not claimed. |
| Fewer factors will improve total runtime | **Hypothesis awaiting tests** | Primary comparison, including the no-SVD affine ablation. |
| A final-state result transfers to whole-model inference | **Unsupported** | Explicitly outside this PoC; would need a compatible real model workload. |
| The full study fits this particular Mac | **Unmeasured planning assumption** | Inspect hardware and calibrate before allocation. |

## 3. Mathematics and the executable algorithm

### 3.1 Objects, conventions, and admissible inputs

Use real row-memory convention throughout:

- `M` is a `p x d` memory matrix; `p` is value dimension and `d` key dimension.
- A stream has length `n`, key rows `keys[n,d]`, value rows `values[n,p]`, and gates `betas[n]`.
- A normalized key `k` has Euclidean norm one; a value `v` is in `R^p`.
- The primary original gates are in `[0.05,0.95]` before float32 storage. Synthetic compiler gates may be in `[0,2]`.
- All stream coefficients are fixed independently of the incoming memory and of its evolving state. They may be computed from an observed input sequence before this component is invoked.
- `E` denotes a `d x r` orthonormal basis for an **exactly supported** active key subspace. For primary blocks take `r=d`, `E=I`; do not use a numerical rank heuristic.
- `N` is compiled instruction count; `R` is request reuse count; `c` is WY chunk size. They are different quantities.

One update is

\[
D_{k,v,\beta}(M)=M+\beta(v-Mk)k^T
                 =M(I-\beta kk^T)+\beta vk^T. \tag{2}
\]

For a unit key, the linear factor has eigenvalues `1` and `1-beta`. It is nonexpansive for `0<=beta<=2`; `beta=2` is a Householder reflection. Gates above one permit a negative eigenvalue, not a negative gate. This primitive and its neural interpretation are established [R1,R2].

Composition yields `M_out = M_in P + B`, with factors in **execution order on the right**. Initialize `P=I_d`, `B=0[p,d]`. For each update, use old values on the right:

\[
P\leftarrow P-\beta(Pk)k^T,\qquad
B\leftarrow B+\beta(v-Bk)k^T. \tag{3}
\]

This costs `O(n(d^2+pd))`. A2 instead forms a dense elementary matrix and multiplies full matrices every token, which unnecessarily costs `O(n(d^3+pd^2))`. Retain that implementation only as a tiny independent oracle; do not benchmark it as the production summarizer.

An empty stream, or one containing only zero gates, is the identity. Remove zero-gate instructions without accessing a meaningless normalized zero key. Reject nonfinite input, incompatible shapes, and nonzero-gate zero keys. Never silently truncate mismatched arrays through Python `zip`.

### 3.2 Stable `2r` construction: primary method

In active coordinates let `Y -> Y P + B`, where `P[r,r]`, `B[p,r]`, and `||P||_2<=1`. Compute a float64 SVD

\[
P=U\Sigma V^T,\quad Q=-UV^T,\quad S=-V\Sigma V^T,\quad P=QS. \tag{4}
\]

Factor `Q` into at most `r` reflections. Execute them with zero targets, giving `Y -> YQ`. Then, for each column `e_j` of `V`, append

\[
k_j=e_j,\quad \beta_j=1+\sigma_j,\quad
v_j=\frac{Be_j}{1+\sigma_j}. \tag{5}
\]

The directions are orthonormal. Their linear factors multiply to `S`, and every later factor preserves earlier injected bias in a different direction. Thus the result is exactly `YP+B` in real arithmetic. There are at most `2r` instructions, including singular `P`, orthogonal `P`, and translations. No inverse of `I-P` or fixed point is needed [A1 §2].

The same construction establishes

\[
\sum_j\|v_j\|_2^2\le\|B\|_F^2,
\qquad \|Y_j\|_F\le\|Y_0\|_F+\|B\|_F. \tag{6}
\]

These are exact-real bounds. They do not imply uniformly small targets independent of the original stream, a finite-precision stability theorem, or preservation of updates smaller than rounding noise.

**Pseudocode to implement; not an already installed interface:**

```text
summarize_rank1(keys, values, gates):
    validate lengths, dimensions, finite values, and admissibility
    P = identity(r, float64); B = zeros(p, r, float64)
    for k, v, beta in execution order:
        pk = P @ k                  # computed before mutation
        residual = v - B @ k        # computed before mutation
        P = P - beta * outer(pk, k)
        B = B + beta * outer(residual, k)
    return P, B

compile_stable(P, B):
    validate square P, compatible B, finite entries
    if r == 0: return empty instruction sequence
    U, sigma, Vt = svd(P, full_matrices=True) in float64 on CPU
    apply the explicit contraction/roundoff policy in Section 5
    Q = -U @ Vt
    residual = Q.copy(); reflections = []
    for j in 0 .. r-1:
        x = residual[j:, j]
        w = x - first_coordinate_vector(length=r-j)
        if norm(w) <= tau_H: continue
        h = zeros(r); h[j:] = w / norm(w)
        residual = residual - 2 * outer(h, h @ residual)
        append (h, zero_value_vector(p), 2) to reflections
    require norm(residual - I) <= the float64 construction tolerance
    instructions = reflections in the order generated
    for j in 0 .. r-1:
        e = Vt[j, :]
        append (e, (B @ e)/(1+sigma[j]), 1+sigma[j])
    require count <= 2*r, finite parameters, and successful internal residual checks
    return instructions and construction diagnostics

offline_validation(instructions, independent_reference):
    reconstruct Q, P_comp, B_comp independently and apply Section 5 checks
    record validation time in the experiment ledger, outside method latency

apply_signed(M, representation):
    if full rank: Y = copy(M)
    else: Yin = M @ E; Y = copy(Yin)
    for k, v, beta in instructions:
        Y = Y + beta * outer(v - Y @ k, k)
    if full rank: return Y
    return M + (Y - Yin) @ E.T
```

The independent map reconstruction above is an experimental check, separate from timed preparation. Cheap input validation, finite/denominator checks, and the residual already produced by Householder elimination remain part of the actual compiler and its measured cost.

The left elimination satisfies `H_last ... H_first Q = I`, so `Q = H_first ... H_last`; the stored order above is correct for row memories. A reversed-order test must fail on a noncommuting example. Use `tau_H = 32 * eps64 * max(1,r)` initially; skipped steps remain subject to the independent residual check. Do not accept a count bound in place of correctness.

### 3.3 Active coordinates and what “rank” means

For ordinary positive-gate streams, the map is identity on the orthogonal complement of the key span, and its bias has rows in that span. On the active span the product is strictly contractive: norm preservation through all positive factors would require a vector orthogonal to every spanning key [A1 §2].

If `k_i=E a_i`, the ambient map is

\[
M_{out}=M+\big[(ME)(P_r-I_r)+B_r\big]E^T. \tag{7}
\]

Use full coordinates for every primary block. Exact low-rank controls use inactive coordinates stored as literal zeros, with a permutation of coordinate indices; discover them by a scan of the input and charge that scan to all reduced methods. Never give only the compiler the generator's hidden basis. A general SVD rank threshold is **not** part of this exact study.

For the rotated near-low-rank stress case, retain full dimension. Record its singular spectrum and the discrepancy that truncation would introduce, but do not introduce a timed approximate competitor. Scalar forgetting would change the complement to a scalar multiple of identity; Equation (7) must not be applied unchanged to Gated DeltaNet [R9].

### 3.4 Short construction: diagnostic, not a candidate selected after results

A1 chooses signs so `Q=U D V^T` has determinant `(-1)^(r-1)`, hence a fixed vector, reducing reflections to at most `r-1`. Writes have gates `1-d_j sigma_j`. Strict contraction makes them positive mathematically, but subtraction can lose a tiny denominator.

Reproduce the following source defect without treating it as a theorem counterexample:

```python
keys = np.ones((1, 1))
values = np.ones((1, 1))
betas = np.array([1e-18])
# A2 compress_affine_delta_stream uses short mode internally.
# P rounds to 1, so the synthetic gate becomes 0 and the value diverges.
```

The stable method's two gate-2 writes represent the **rounded** summary `P=1,B=1e-18`. They cannot recover the exact missing `1e-18` decrement of `P` after it has rounded away. If implementing short mode at all, check every write denominator before division and return `numerical_failure` if it is nonpositive or below `256*eps64`; record the condition instead of silently falling back or omitting the case. Do not put short mode in the main performance matrix.

### 3.5 Lower bounds and scope checks that matter

For pure translation `M -> M+B`, A1 §3 proves the exact minimum `2 rank(B)` even with arbitrary nonzero real gates. With key columns `K`, value columns `W`, and invertible gate diagonal `G`, its compact form is

\[
T=(G^{-1}+\operatorname{triu}(K^TK,1))^{-1},\quad
P=I-KTK^T,\quad B=WTK^T.
\]

If `P=I`, Sylvester's inequality gives `0 >= 2 rank(K)-N`, while `rank(B)<=rank(K)`. Two reflections along each orthonormal row-space direction attain the bound. This is a worst-case general-affine result, not a proof that `2r-1` is always optimal for positive-stream summaries.

The rotating-projection witness in A1 §4 uses `u_j=(cos(j theta/n),sin(j theta/n))`, `theta=pi/3`. Column-order projections satisfy `P_n u_0=cos(theta/n)^n u_n`. Positive semidefinite contractions obey the norm-angle bound `||x_m||/||x_0|| <= cos(theta/m)^m`, so this witness needs `n` positive factors. For an explicit two-factor control, write `P_n=s*u_n*u_1.T`, where `s=cos(theta/n)^(n-1)`, set `x=u_1`, `y=s*u_n`, `h=(x-y)/||x-y||`, and `gamma=||x-y||^2/(1-x.T@y)`. Then `P_n=(I-gamma*h*h.T)*x*x.T`, with `0<=gamma<=2`. In row-memory convention, execute a gate-one projection onto the line of `x` (its delta key is a unit perpendicular to `x`), followed by the delta factor `(h, zero_value, gamma)`. This yields `P_n.T`, the row-memory version of the original projection stream. The generic stable compiler may use more than these two factors; do not equate its count with the separately derived control.

For explicitly **nonnegative** gates `0<=beta_i<=b`, with `0<b<2` and positive total squared-norm loss, its dissipation specialization is

\[
m\ge\frac{2-b}{b}\frac{\|x_m-x_0\|^2}{\|x_0\|^2-\|x_m\|^2}. \tag{8}
\]

Zero gates are discarded. For zero total squared-norm loss under `b<2`, every effective displacement is zero; handle that identity-on-the-tested-vector case without dividing by zero. Negative gates are not covered by this inequality. The geometry is established prior work [R5,R6]; the witness is a correctness/positive control for instruction count, not the primary timing distribution.

Finally, implement an exact endpoint-versus-prefix counterexample: in one dimension, two gate-one writes with values 1 then 2 send any incoming state to 2, but read 1 after the first write. A shorter final-map implementation need not read 1 at its first step. A state-dependent write such as `v=M^2`, `beta=k=1`, gives a nonlinear update. The interface accepts materialized coefficient arrays and rejects callbacks or online coefficient providers. Independence from the incoming memory is a caller/data-generation precondition: it cannot generally be inferred from numeric arrays whose provenance is unknown. Use this example as an analytic scope counterexample, not as an impossible automatic provenance detector. These checks prevent an invalid whole-model interpretation.

## 4. Reference methods and computational costs

### 4.1 The four methods

| Method | Preparation | Application | Purpose |
|---|---|---|---|
| `original` | Validate/cast/materialize original arrays | Equation (2) in original order | Simple reference and count-control comparator. |
| `affine` | The optimized shared summary; optional exact coordinate reduction | `M @ P + B`, or Equation (7) | Mandatory strong baseline and no-SVD ablation. |
| `wy` | Chunk triangular preparations described below | Two matrix products per chunk | Closest established batching mechanism, implemented portably. |
| `signed` | Same summary as affine, plus SVD, reflections, values, cast/materialization | At most `2r` updates | The sole proposed timed method. |

All operate on the same stored inputs and return the same kind of materialized final-state array. All receive the same precision contract. They may reuse their own block representation across the 16 requests. They cannot cache a particular state's output as the implementation.

For reduced affine execution, allow sensible algebraic preparation: precompute `L=(P_r-I)E.T`, `B_full=B_r E.T`, and apply `M + (M E)L + B_full` if faster than Equation (7). For coordinate support, gather/scatter can replace multiplication by `E`. Give the same coordinate gather/scatter option to signed and original reduced execution. Charge representation construction and output copying consistently.

### 4.2 Correct compact-WY construction

For one chunk, use **key columns** `K[r,c]`, value columns `V[p,c]`, and `G=diag(beta[c])`. Define

\[
H=\operatorname{triu}(K^TK,1),\quad
T=\operatorname{solve\_triangular}(I+GH,G),\quad
F=TK^T. \tag{9}
\]

Then

\[
Y_{out}=Y+(V-YK)F. \tag{10}
\]

This division-free formulation handles zero gates without `G^{-1}`. Use `scipy.linalg.solve_triangular(..., lower=False, unit_diagonal=True)` on CPU float64, not an explicit inverse [R14]. You may precompute `C=V F` and apply `Y-(YK)F+C`. The Gram term has a **plus** sign in the solve. The current published DeltaNet paper's Eq. (10) and Appendix B.2 are consistent with this after transposing conventions [R1].

Mandatory exact guard: `d=p=1`, `k1=k2=1`, gates `1/2`, values `1,0`, initial state 0. The result is `1/4`; a wrong sign gives `3/4`. Also test nonorthogonal two-dimensional keys to detect orientation mistakes.

Only choose `c` from `{16,32,64}` on development data. Use contiguous chunks, preserve order, and handle a short final chunk. Never allocate an `n x n` triangular array for the whole long sequence. The portable implementation follows the established mathematics but is **not** a reproduction of the optimized FLA GPU kernel. Current FLA documentation lists several accelerator backends, but does not establish a Metal/MPS path for this Mac [R10]. Do not call FLA universally “CUDA-only,” and do not compare against published NVIDIA timings.

You may use the same prepared chunks to construct a summary with BLAS operations:

```text
P = I; B = 0
for K, F, C in chunk order:
    P = P - (P @ K) @ F
    B = B - (B @ K) @ F + C
```

Compare this summary builder with Equation (3) during development and freeze the faster correct choice per shape/length. Both affine and signed must use that same choice. Charge independent standalone costs even if a common intermediate is physically shared to economize the benchmark.

### 4.3 Costs that make the hypothesis uncertain

For full rank, leading arithmetic counts per request are approximately:

| Route | Leading arithmetic, excluding lower-order terms |
|---|---:|
| Original replay | `4 n p d` |
| Dense affine | `2 p d^2` |
| Signed replay with `N` factors | `4 N p d`; approximately `8 p d^2` at `N=2d` |
| Rank-one summary preparation | `O(n(d^2+pd))` |
| SVD and reflection generation | `O(d^3)` |

For `n=1024,d=p=128`, these application estimates are about 67.1 million operations for original replay, 4.19 million for dense affine, and 16.8 million for a 256-factor signed replay. These are **planning calculations, not M3 timings**. GEMM utilization, dispatch, memory traffic, and fusion can change runtime relationships. They do not establish a universal impossibility theorem, but they remove any presumption that a short delta program is the fastest representation.

Record storage size of every prepared representation as well as peak memory. Near `N=2d`, signed keys and values are not automatically smaller than `P,B`. Do not claim memory compression merely because `N<n`.

## 5. Numerical conventions, independent correctness, and failures

### 5.1 Canonical stored inputs

Generate the intended normalized streams in float64, then store the canonical coefficients and initial states in float32. Define the target map as Equation (2) applied to the **literal stored numbers interpreted in float64**. All methods receive these exact arrays. Hash them; never regenerate a different stream for a method.

If the compiler needs exactly normalized real keys, preserve this target by the algebraic conversion

\[
a=\|\widetilde k\|_2,\qquad
k=\widetilde k/a,\qquad
\beta=\widetilde\beta a^2,\qquad
v=\widetilde v/a. \tag{11}
\]

This transformation leaves the update unchanged in real arithmetic. Merely normalizing the key while keeping gate and value fixed would change it. Record conversion cost and residual. The primary gate margin avoids crossing the ordinary gate-one boundary through harmless key rounding. Tiny projection controls may instead start directly from exact coordinate keys.

Keep the original float32 arrays alongside their promoted/normalized forms. Original float32 replay executes the stored arrays directly, not a recast normalized copy. Summary and WY preparation may use the literal float64 promotion; normalization by Equation (11) is needed only by an interface that requires unit keys. All representations are checked against the same literal-input reference.

CPU float64 is used for SVD, triangular preparation, and reference maps. Execution performance uses float32, with prepared representations cast once. Record both the float64 representation discrepancy and the additional discrepancy after casting. No float16/bfloat16, hidden mixed precision, or approximate rank truncation is permitted in the primary study.

### 5.2 Fixed acceptance criteria

These thresholds are engineering accuracy requirements for this component, not guarantees of a trained model's task accuracy. Do not loosen them in response to a candidate failure.

1. **Float64 construction:** `||P_comp-P_ref||_2 <= 1e-10` and `||B_comp-B_ref||_F <= 1e-10*(1+||B_ref||_F)`.
2. **Represented map after float32 packing**, reconstructed independently in float64: linear error at most `1e-5`, bias error at most `1e-5*(1+||B_ref||_F)`.
3. **Actual float32 execution**, for every tested incoming state:
   - `||Y-Y_ref||_F / max(1, ||M_in||_F+||B_ref||_F) <= 1e-4`;
   - `||Y-Y_ref||_F / max(1, ||Y_ref||_F) <= 1e-4`;
   - `max_abs(Y-Y_ref) / max(1,max_abs(Y_ref)) <= 1e-4`.
4. Finite outputs, gates in the declared range up to recorded casting error, reflection reconstruction, factor count, and shape checks must all pass. Record each packed key's norm drift and effective scalar gate `beta*||k||^2`; a stored gate of 2 with a rounded key longer than one can be slightly expansive. Do not silently renormalize packed keys without the corresponding gate/value transformation. Existing map and execution thresholds govern this representation drift.

For an exactly reduced representation reconstruct the ambient map as `I+E(P_comp-I)E.T`, `B_comp E.T`. Check `P` and `B` separately; zero-state testing alone checks only `B`. Also report relative linear error using `max(||P_ref||_2,1e-30)` as denominator, even when that is large for a nearly zero map.

The real-map discrepancy obeys

\[
\|\Delta M\|_F\le\|M\|_F\|\Delta P\|_2+\|\Delta B\|_F. \tag{12}
\]

This bounds representation error, not all rounding inside the executor. Samples of float32 execution do not prove equality for every possible memory. Report three fixed unit-query readout errors as diagnostics, without interpreting them as downstream model accuracy.

For computed singular values slightly above one, set `tau_contract=100*eps64*max(1,n)` for stream summaries. If `sigma_max>1+tau_contract`, fail construction and diagnose the input/summary. Values within this roundoff allowance may be clipped to one **only with** the perturbation norm logged and the original, unclipped reference retained for all correctness tests. Never alter the baseline's target to force agreement. Use `n=1` for arbitrary-map diagnostics. Record SVD nonconvergence; allow one documented alternate LAPACK driver within the shared repair allowance, not unlimited retries.

### 5.3 Minimum tiny checks before any comparative timing

Use the supplied checks as regression evidence, but add independent oracles. Bound the entire tiny suite to **1,500 fixture execution attempts**, including the source's 160+80+1,000 cases, additional controls, backend checks, and retries. One fixture execution includes its method/oracle assertions; rerunning it counts again. Assign `pytest` to at most 80 independent fixtures/mutation guards, and `run.py check` to the 1,240 source cases plus at most 100 additional/backend fixtures. The remaining allowance covers affected retries. Do not run the full source suite from both commands. Reuse recorded passes only when source, configuration, backend, and input hashes are unchanged; all actual work still enters the time ledger.

- Exact rational scalar examples using `fractions.Fraction`: identity, a half-gate, the two-write WY example, and a reflection pair implementing translation.
- Tiny dense multiplication of homogeneous affine maps, or the original A2 dense summary, against the rank-one summarizer. Do not have both the candidate and oracle call the same summary helper.
- Explicit `M=0`, nonzero coordinate states, and a deterministic noncommuting two-dimensional example. For an affine implementation, images of zero and a complete row-coordinate basis identify the linear and bias maps.
- Pure translations of ranks 1, 2, 3; arbitrary singular and orthogonal contractions; `r=1`; empty stream; all-zero gates; beta exactly one; invalid shapes/nonfinite inputs.
- Rotating projections at `n=32` and `n=1024`: preserve the endpoint map and verify the explicit two-factor control. A measured speed comparison on these favorable examples is diagnostic only.
- The tiny-gate short-mode failure and stable-mode handling; near-identity maps with tiny bias; rounded low-rank stress.
- Mutation guards: reverse factor order on a noncommuting case, change the WY sign, and omit reflections. Each should be detected by the correctness tests. These guards test the oracle, not a new method.
- Endpoint-versus-prefix and state-dependent-coefficient counterexamples from Section 3.5.

One source check's success is not enough to bless a new optimized backend. Run the meaningful independent checks against each backend that will appear in comparisons.

### 5.4 Forgetting control

Record `||P||_2`, `||P||_F`, and `||M P||_F/max(||B||_F,1e-30)` for every block. Long strong-gate streams may largely erase their incoming state. On those blocks evaluate `Y=B` as an **untimed accuracy control** and report its real-map error bound `||M||_F ||P||_2`.

If this control meets the actual output tolerance for all 16 inputs, label the block “incoming-state contribution negligible at the study tolerance.” The exact algorithm is still meaningful algebraically, but that cell cannot establish that signed instructions improve meaningful retained-state processing. In the primary cohort, if this occurs for any block, retain it, disclose it, and require the same positive conclusion on the remaining non-negligible primary blocks. Apply the same equal-stratum analysis to that disclosed subset, resampling its actual number of blocks within each stratum. Preserve the full-cohort statistic as primary. Fewer than four non-negligible blocks in any stratum makes the retained-state interpretation inconclusive; do not treat a smaller subset as adequate evidence. Do not silently remove cases or change the primary pooled statistic. Weak-gate diagnostics in Section 7 help explain this effect without changing the confirmatory population.

### 5.5 Failures and exclusions

Every scheduled method-block evaluation gets a status: `ok`, `accuracy_failure`, `invalid_input`, `construction_failure`, `backend_unsupported`, `timeout`, `memory_limit`, or `measurement_invalid`. Log elapsed time, configuration, exception text, and any partial evidence. Timeouts are censored lower bounds, not missing-at-random observations. Do not impute favorable runtimes or compute a success-only average as the primary result.

An implementation bug yields an invalid comparison until repaired. A valid, independently checked method that cannot meet the float32 contract is a numerical limitation, not proof that the exact-real theorem is false. If the candidate fails any primary block, a general positive primary conclusion is unavailable; report the failure rate and completed-case timings separately. A failed or missing strong baseline also prevents a positive claim until repaired. An expensive original-loop timeout does not invalidate a completed signed-versus-affine/WY comparison, but its runtime ratio must be reported only as a bound.

## 6. Target environment and simple implementation stack

### 6.1 Inspect before choosing a backend or allocating a workload

Record the actual chip/model, native architecture, CPU/core information, physical and available RAM, free disk, macOS version, Python architecture/version, package versions, BLAS/LAPACK provider and thread behavior, and power state. Useful read-only commands include `sw_vers`, `uname -m`, `sysctl -n hw.memsize`, `sysctl -n hw.ncpu`, and `df -h .`. Save `numpy.show_config()` and `numpy.show_runtime()` when available [R13]. Do not infer memory capacity or OS/backend support from “M3 Max.”

If you do not have the target Mac, complete the code, static checks, mathematical derivations, and a clear blocked-work list. Do not substitute remote benchmark computation or claim M3 measurements. Distinguish the planning-only Linux checks in Section 2 from your future target-device validation.

Use a native arm64 Python virtual environment, ordinary scripts, JSON configuration, JSONL results, and NumPy/SciPy. Add only `pytest`, `psutil`, and `matplotlib` for checks, resource records, and final plots. Add MLX for the Metal execution path when the actual environment supports it. No PyTorch, model training, downloaded checkpoint, dataset framework, service, or dashboard is needed.

The retrieved MLX 0.32.4 documentation requires native Python at least 3.10 and macOS at least 14 for its Apple-silicon package, and says GPU float64 is unsupported [R11]. These facts do not verify the installed Mac environment. Install a compatible available wheel, record exact resolved versions before measurements, and use the installed API's documentation if it differs. Dependency installation on the target remains unverified.

### 6.2 Backend rule

Use **one primary performance backend**: MLX Metal float32 if the environment and tiny tests pass; otherwise CPU NumPy float32. CPU float64 remains the reference and preparation backend. A CPU fallback supports a CPU component conclusion only. Do not mix CPU and GPU timings in the primary ratio.

The MLX executor needs matrix products, vector products, reductions, broadcasts, and addition; SVD and triangular solves stay on CPU. Launch with `MLX_ENABLE_TF32=0`, record it, and validate actual matrix arithmetic against the float64 reference. Current MLX documents that nominal float32 matrix operations may otherwise use reduced precision [R12]. Unexpected CPU fallback or unsupported operations must be visible, not hidden in an apparent GPU timing.

Use the same small compiled update-block function for original and signed replay if using `mx.compile`; its arrays are runtime arguments, not captured outputs. Fix an unrolled block of 32 delta-update slots to bound graph growth. Pad a short final microblock with gate-zero identity instructions and valid unit keys, for both original and signed execution, so coefficient-dependent factor counts do not require new executable shapes. Record both nontrivial factors and executed slots; padding overhead is timed. Materialize between these blocks if needed to keep graphs bounded, using the same policy for both methods. Allow affine and WY the same compilation facilities. Record tracing/compilation and first-use costs separately. Do not build custom kernels to rescue the result.

If MLX cannot pass after one bounded repair, freeze CPU as the fallback before confirmation. All attempts consume the same experiment and repair caps. Do not run the entire confirmatory matrix on both backends. If the Metal primary result is positive, spend at most eight additional method-block cells on CPU affine and signed execution: confirmation replicate 0 in each of the four primary strata, two methods per block. These IDs are fixed before confirmation. Omit this optional check if the primary backend is already CPU or if there is no positive primary result. Report CPU timings separately, including any faster affine alternative; resident CPU versus resident GPU timings do not include switching a real model's states between backends and cannot by themselves establish that such switching is beneficial.

### 6.3 Fixed resource ceilings

These are planning ceilings, not measured forecasts:

| Resource | Ceiling and enforcement |
|---|---|
| Experimental work | 240 cumulative minutes, including checks, development trials, failed attempts, timing, repairs, diagnostics, and report regeneration. Section 9 allocates it. |
| Process memory | `min(8 GiB, 25% of physical RAM, 50% of available RAM at inspection)`; require at least 2 GiB for the full study. |
| Disk | At most 2 GiB new project data/results; require at least 4 GiB free initially. No checkpoint downloads. |
| Timed method-block cells | At most 500 across development, confirmation, diagnostics, optional backend checks, failed cells, and retries. Inner timing repetitions are additionally bounded below. |
| Main coefficient instances | 72 maximum; 24 protected primary instances. Development uses 12 separate instances. Numerical/special timed diagnostics use at most 12 instances. |
| Tiny mathematical cases | At most 1,500 including supplied checks. |
| One timed method-block cell | 20 seconds of measurements; separately at most 10 seconds to prepare its block/representation, and at most 30 seconds for a new executable shape's compilation. |
| Inner repetitions | At most 20 repeated `R=16` groups per timing round; at most five rounds per cell. |

Keep one benchmark process active. Do not parallelize timings across agents or devices. Monitor process RSS and MLX allocator peak separately; unified memory means those counters must not simply be added as if disjoint. Record peak RSS, MLX peak, array/representation sizes, cache policy, and observed system pressure. If either tracked peak exceeds the cap, or swap growth exceeds 256 MiB during a cell, stop that cell and record the condition. Do not keep allocating until an operating-system failure.

At the largest planned shape, stored keys, values, and gates occupy about `4,210,688` bytes in float32; one `128 x 128` state is 65,536 bytes. Float64 copies, 16 states, graphs, library workspaces, and caches add overhead. Stream one coefficient block at a time and save seeds/hashes plus selected diagnostic arrays; do not retain all intermediate token states.

Expected setup is 15–45 minutes; implementation and inspection are estimated at 4–6 hours; experiments and analysis have a hard four-hour ceiling. Expect roughly 8–11 hours end to end if the full study is reached, potentially much less after an early stop. These are estimates, separate from measured experimental runtime. Do not launch an open-ended optimization effort after the six-hour implementation allowance; preserve runnable partial work and explain the blocker.

## 7. Data, development separation, and experiment matrix

### 7.1 Deterministic generation; no external dataset dependency

Use `numpy.random.Generator(PCG64(SeedSequence(...)))`. Define the seed tuple as

```text
[20261104, phase_id, family_id, shape_id, length_id, replicate_id, stream_id]
phase_id: 0 development; 1 confirmation; 2 diagnostics
family_id: 0 isotropic; 1 correlated
shape_id: 0 -> d=32,p=64; 1 -> d=128,p=128
length_id: 0 -> n=d; 1 -> n=8*d; 2 -> n=32*d
stream_id: 0 coefficients; 1 initial memories; 2 diagnostic queries
```

Use explicit numeric identifiers for every added diagnostic and store them. Development uses replicate 0 of phase 0. Confirmation uses replicates 0–5 of phase 1, never used for debugging or tuning. Generate all intended confirmation IDs and freeze their configuration before examining outputs. A fixed seed is for reproducibility, not evidence of representativeness.

**Isotropic generator:** sample each key from `N(0,I_d)` and normalize in float64; values have independent entries `N(0,1)/sqrt(p)`; gates are uniform on `[0.05,0.95]`. This tests ordinary full-rank unstructured streams without a deliberately short factorization.

**Correlated neural-projection generator:** set input width `h=2d`; draw `x_0~N(0,I_h)` and `x_t=0.9*x_(t-1)+sqrt(1-0.9^2)*epsilon_t`. Draw fixed matrices `W_k[h,d]`, `W_v[h,p]` and vector `w_beta[h]` with independent `N(0,1/h)` entries. Set `k_t=normalize(x_t W_k)`, `v_t=tanh(x_t W_v)/sqrt(p)`, and `beta_t=0.05+0.90*sigmoid(x_t w_beta)`. Use a numerically stable sigmoid. Compute the whole block before execution, hold its weights and coefficients fixed across all 16 incoming memories, and store float32 coefficients.

This second generator provides temporal and key/value/gate correlation but is **untrained**. It does not establish the rank, gate distribution, or behavior of any trained DeltaNet. Both generators have full rank with probability one in exact arithmetic when `n>=d`; record the actual spectrum but keep full dimension even when conditioning is unfavorable.

Here the second argument of `N(0,variance)` is a **variance**. NumPy takes a standard deviation: use `scale=1/sqrt(h)` for every projection-weight entry, `scale=1/sqrt(p)` for isotropic value entries, and `scale=1/sqrt(d)` for incoming-memory entries.

Initial memories have independent `N(0,1/d)` entries, shape `[16,p,d]`, sampled with the separate stream ID. They are distinct, nonzero, and identical across methods. Generate three unit query vectors from the diagnostic stream for readout checks. Dataset generation and frozen projection time are measured once and reported separately as shared costs; the component headline begins with coefficients available. Adding a nonnegative common frontend cost can only reduce a fractional component advantage.

### 7.2 Matrix and purposes

| Set | Instances | What it tests |
|---|---:|---|
| Development: both shapes, both families, all three lengths | 12 | Debugging, WY chunk choice, summary builder, timing scale, backend feasibility. Not confirmatory evidence. |
| Primary: both shapes, both families, `n=8d`, six fresh replicates | 24 | Equation (1), accuracy, and the practical 20% threshold. |
| Short-block control: same shapes/families, `n=d`, six replicates | 24 | No general factor-count advantage when `n<=2d`; detect unfair dispatch/preparation comparisons. |
| Longer-block control: same shapes/families, `n=32d`, six replicates | 24 | Scaling, preparation amortization, and incoming-state forgetting. |
| Exact coordinate-rank controls: `r=d/4`, `n=8d`, isotropic local keys, two fresh replicates per shape | 4 | Favorable low rank with exact zero support; give identical discovered support to all methods. |
| Weak-gate controls: both shapes and families, `n=32d`, one fresh replicate, gates divided by 32 | 4 | Long blocks whose incoming-state effect is less rapidly erased. |
| Optional cross-backend check after a positive Metal primary result | At most 8 method-block cells; replicate 0 of each primary stratum | CPU affine/signed comparison on fixed instances; report placement limitations separately. |

The eight timed diagnostic instances fit within the allowance of 12; the remainder is available only to resolve a preidentified numerical issue, not to search for favorable workloads. Arbitrary near-identity, translation, rotated-rank, and rotating-projection examples belong to the tiny mathematical suite unless a specifically justified diagnostic timing fits the remaining cap.

The exact coordinate-rank controls choose `r` coordinate indices without replacement and store all other key components as exact zeros. Values and gates follow the isotropic generator. The rank is discovered from the arrays by both methods. Do not claim general low-rank behavior from this favorable structural case.

### 7.3 Freeze before confirmation

Development may choose one WY chunk size from `{16,32,64}` per shape, the faster correct rank-one versus chunk-built summary per shape/length, and whether ordinary `mx.compile` helps. Use at most 96 timed development method-block cells in total. Fix thread settings globally per backend; at most one alternative BLAS thread setting may be compared on development, within those same 96 cells.

Also choose a named baseline for each primary stratum on development data: the valid comparator with lowest median `C+W(16)` there. Save this selector before confirmation. It is used to corroborate negative conclusions without choosing a noisy winner on the evaluation blocks.

Freeze: git commit, dependency versions, backend/precision settings, formulas, generators, seed IDs, accepted dimensions, tolerances, primary population, preparation accounting, method implementations, the named baseline selector, timing policy, and analysis code. Save `protocol.json` with its SHA-256 and a UTC timestamp. The 20% threshold and 16-request scenario are fixed by this brief and cannot be tuned.

A material change to algorithm, comparator, primary metric, or generator after opening confirmation results is exploratory. Preserve the original protocol and results. Fresh confirmation is permitted only if it fits the already bounded reserve; otherwise report the revised result as inconclusive/exploratory. Do not restart the budget or keep drawing seeds until a gain appears.

## 8. Benchmark protocol and analysis

### 8.1 Timing boundaries

Use `time.perf_counter_ns()`. Keep input generation, independent oracle/map reconstruction, correctness calculations, plotting, and file I/O outside **both preparation and application latency**, but record their costs in the project ledger. Implement an explicit `prepare_core` / `validate_prepared` separation. Preparation latency includes the algorithm's own validation, normalization, finite/denominator checks, and internal Householder residual checks; it excludes test-only reconstruction of the full candidate affine map, diagnostic SVDs, and comparison with an oracle. Apply this separation to every method. Report validation overhead separately so it is not hidden from the study's resource accounting. The primary application returns a final state; optional readout checks are not part of its latency.

For each block and method, record separately:

1. Validation/normalization/support discovery and affine/chunk preparation.
2. SVD/reflection/value generation, where applicable.
3. Casts, allocation of the reusable representation, and materialization on the execution backend.
4. First-use executable compilation and device initialization (`F_m`), when needed.
5. Warm serial application group `W_m(16)` and a directly measured single-use time.

Measure the common summary component three times from fresh arrays after library warmup and take its median; measure each additional preparation component three times. If a component hits its preparation cap, record failure instead of silently using a cached result. Physically sharing a measured common summary across methods is acceptable for benchmark economy, but charge its full standalone cost to both affine and signed. WY and original do not pay for an unused SVD or summary.

Also measure one direct complete `prepare + 16 applications` path per method in each primary stratum. Its result must agree with the component sum within 10% or the measured repeatability envelope, whichever is larger. Investigate a discrepancy before treating reconstructed total costs as valid. This guards against double counting and omitted transfers.

On MLX, materialize inputs before hot timing; then create a **new result by calling the function** on each request. Evaluate the result and synchronize the actual stream before recording completion and before starting the next request. Re-evaluating an already evaluated result is not computation [R15,R16]. Use runtime array arguments so graph compilation cannot replace a fixed output with a constant.

```text
synchronize()
start = clock()
for state in the 16 incoming memories:
    output = apply_method(state, representation)
    evaluate(output)
    synchronize()       # serial request, output ready before next request
stop = clock()
```

Do not evaluate all 16 outputs only at the end: that is a queued-throughput experiment. Do not merge the 16 states into a batch in the primary study. If a future throughput study batches them, the affine baseline must also receive a flattened/batched GEMM; that is outside this protocol.

Returning a copied output versus an alias must be consistent across methods. Include allocations/copies required by that return contract, or use matched preallocated output buffers for all methods. Do not time checking hashes or converting GPU results to NumPy unless that conversion is part of the chosen common output contract. Primary outputs are ready on the chosen execution backend; CPU preparation and all required coefficient transfers are included in `C`.

### 8.2 Repetition, order, and noise

Default to five paired timing rounds, with method order permuted deterministically each round. Within a round, repeat a complete 16-request group only if needed for a 50–200 ms measurement window; use a development-calibrated multiplier capped at 20. Store the multiplier and normalize the reported group time. If the cap prevents a long enough window, report measurement uncertainty rather than replacing actual work with empty loops.

A global choice of three rounds instead of five is allowed **before freeze** if the full forecast would exceed the budget. Do not reduce only slow/unfavorable methods. Stop a cell when it reaches 20 measurement seconds, recording the rounds completed and timeout status. Protect candidate-versus-affine/WY evidence from an extremely slow original loop.

Run on external power if available, record power mode and major competing load, and avoid concurrent downloads during timing. Every six blocks run the same small affine sentinel. A greater-than-10% drift from its calibrated median triggers one short pause and one repeat of the affected round, retaining both measurements and the reason. Timing repairs, pauses, and repeats consume the common reserve. Do not repeatedly discard slow rounds without an external diagnosis.

### 8.3 Primary analysis and uncertainty

For each method-block, take the median valid group time and the median preparation components. Compute Equation (1) against the minimum completed **accuracy-valid** comparator cost. This empirical baseline envelope is deliberately conservative; report which comparator wins and its pairwise times. It is not a newly learned deployment selector.

Use equal weight for each of the four primary strata. Summarize their log ratios with an overall geometric mean. Estimate a 95% interval by 10,000 stratified bootstrap resamples of the **six independent coefficient blocks within each stratum**, using a separate fixed analysis seed. Timing rounds and the 16 states are not extra independent blocks. Present all 24 block ratios and each stratum's mean, range, and failure count so the small sample is visible.

A **supported positive primary result** requires all of the following:

- Every primary signed run and at least affine plus WY pass correctness and measurement checks, with no unreported exclusions.
- The overall geometric-mean ratio's 95% upper bound is at most `0.80`.
- Each of the four stratum point estimates is at most `0.80`; a favorable small-shape or input family cannot mask the rest.
- The comparison includes preparation and the result survives the no-SVD affine ablation.
- The forgetting diagnostic does not eliminate the retained-state interpretation under Section 5.4.
- The candidate-unfavorable timing-range sensitivity check below cannot reverse the 0.80 decision.

For a negative claim, corroborate the envelope result against the **named baseline selector frozen on development**. Bootstrap those paired ratios using the same strata. Require its 95% lower bound to exceed `0.80`; an envelope interval alone cannot establish a negative near the threshold because taking the minimum of noisy baseline estimates can favor that baseline. Also perform the timing-noise sensitivity check below. A smaller but repeatable improvement may be reported descriptively. An interval entirely above one against that fixed selector supports slowdown within scope.

For a negative sensitivity check, recompute each block's ratio using observed minimum signed method-specific preparation/group times and maximum named-baseline times across valid repeats. If the geometric mean of these candidate-favorable ratios is at most `0.80`, classify a threshold-level negative as inconclusive because measurement noise can explain it. For a positive sensitivity check, use maximum signed times and minimum comparator times, including the baseline envelope; require the geometric mean to remain at most `0.80`. A component physically shared between methods, especially summary preparation, must take the **same observed value** in both totals: evaluate its recorded minimum/median/maximum jointly and take the adverse result, rather than pretending its noise is independent. These observed-range checks are conservative sensitivity analyses, not additional confidence intervals or guarantees outside the observed range. Do not draw extra confirmation blocks to repair a borderline result.

If an interval crosses `0.80`, or the pooled positive interval passes but a stratum or forgetting guard fails, the full primary success claim is inconclusive/mixed, not supported. Report any narrower pooled observation separately, with its failed guard. Six blocks per stratum are a deliberately small engineering study, not a broad population survey.

If candidate or essential baseline cells fail, report the incidence over all scheduled primary blocks; do not bootstrap only surviving cases and present that as the primary conclusion. A deterministic analytic obstruction may settle a narrower claim without statistical testing; do not apply tests to exact identities.

### 8.4 Reuse and cold costs

Report measured preparation and per-use application costs, direct `R=1` and `R=16` timings, and **modeled** totals at `R in {1,4,16,64}`. Label modeled curves. For a comparison with `C_S>=C_B` and `a_B>a_S`, the smallest integer reuse for a strict win is

\[
R_{strict}=\left\lfloor\frac{C_S-C_B}{a_B-a_S}\right\rfloor+1. \tag{13}
\]

Handle equality and other sign cases explicitly; evaluate the inequality rather than blindly applying the formula. If `C_S>=C_B` and `a_S>=a_B`, there is no positive break-even in this linear cost model. A higher reuse count cannot fix measured domination. Propagate timing ranges into break-even uncertainty; do not print an exact integer when the denominator is indistinguishable from zero.

For first-use cost, show `F_m+C_m+W_m(R)`. If executable setup is shared across `J` fresh blocks, state the assumption and show `F_m+sum_b(C_mb+W_mb(R))`; do not omit `F_m` without explaining that reuse. Shared coefficient-projection time is outside the component headline but must appear in a separate total. Neither a modeled amortization curve nor a synthetic frontend establishes model-level speedup.

## 9. Evidence gates and cumulative budgets

Maintain an append-only ledger with stage, instance, method, attempt, elapsed seconds, peak memory, outcome, and cumulative totals. Changing backend, tuning parameters, retries, or stopping early does not reset any cap. Experimental budget is separate from hosted-agent reasoning cost, bounded code-writing effort, and dependency-download setup, but it includes every local experimental trial and failed attempt. Stage allocations below are for experimental execution/inspection after the minimum needed code exists; they do not require implementing the compiler in 15 minutes. Build later-stage modules only when preceding evidence justifies them, within the separate implementation allowance.

| Stage | Question and minimum work | Maximum experimental allocation | Evidence and decision rule |
|---|---|---:|---|
| G0: constructive feasibility | Inspect the actual Mac; verify fixed-coefficient/final-state interface, received sources, dependencies, precision, and resource caps. | 10 min | `environment.json`, source hashes, scope contract. Continue if required operations and at least a meaningful CPU path exist. If target access is absent, finish unaffected code/analytic work and report blocked Mac execution. |
| G1: tiny correctness | Implement independent recurrence/affine oracles, stable compiler, WY, input validation, and essential counterexamples. | 15 min | `correctness.json` plus exact derivations. Continue only after valid representations agree. One bounded repair is allowed from the shared reserve. A verified mathematical obstruction triggers an analytic report, not a large benchmark. |
| G2: resource calibration | Use `d=16,p=16,n=128` for smoke only, then representative `d=32,p=64,n=256` and `d=p=128,n=1024`; measure preparation, application, memory, and timing resolution. | 15 min | Calibrated multipliers and a `1.5x` safety-factor forecast. No scientific success may be claimed on the smoke size. |
| G3: mechanism pilot and freeze | Run development cells; choose bounded WY/summary/compilation options; check count reduction, affine ablation, backend noise, and forgetting. | 20 min | Locked protocol and implementation commit. Continue with full confirmation if informative and affordable; choose the smaller negative branch below if justified. One precision/noise repair may use the reserve. |
| G4: locked comparison and diagnostics | Run protected primary cells first, then affordable predeclared controls. | 120 min | Raw results, failures, costs, map errors. Stop launching new work when the forecast would consume reporting reserve. Do not broaden the search in response to the results. |
| G5: diagnosis and report | Analyze all scheduled outcomes, regenerate essential plots/tables, audit claims and limitations, and preserve evidence. | 30 min | Final report and reproducible outputs for the stages actually reached. |
| Shared repairs/retries/tuning overrun | At most two repair episodes, at most 15 min each, across all stages. | 30 min | Record each change and rerun only affected meaningful checks; no indefinite rescue campaign. |
| **Total** | All executed branches combined | **240 min** | Reserve G5's 30 minutes throughout. |

The intended informative pilot is completed within about one hour, excluding dependency installation and code-writing time. These stage amounts are ceilings. Ordinary shorter work does not authorize expanding the project; unused time is available only for already declared checks or preserving evidence.

### 9.1 Forecast and resizing rules

Forecast wall time from measured preparation plus actual planned calls, not only FLOPs. Include three preparation repetitions, the selected timing-round multiplier, source checks, compilation for new shapes, failures already incurred, and a safety factor of 1.5. Estimate memory from actual peak measurements and dominant live arrays/graphs.

If the 72-instance matrix will not fit G4, drop the 48 secondary-length controls **as a whole** before freeze, retaining all 24 primary instances. Next, use three instead of five rounds globally if timing precision remains sufficient. If the protected primary population still cannot fit, report practical infeasibility or an inconclusive pilot. Do not shrink the 128-dimensional primary shape into an irrelevant toy, loosen error targets, or remove difficult primary blocks.

If performance is unexpectedly worse during confirmation, preserve timeouts and missing scheduled cells. A partial primary cohort cannot establish the prescribed positive result. Save completed evidence and analyze it within the reporting reserve.

### 9.2 Early negative branch

If, on all four primary development strata, signed preparation is no cheaper than affine, its median per-use latency is at least `1.10x` affine, every paired timing round shows the same direction, and the timing procedure is stable, select the **24-instance protected primary cohort only** for fresh confirmation. This is an efficiency decision, not a conclusion from one pilot seed.

After fresh confirmation, if signed is again dominated and the interval excludes the practical threshold, complete an informative negative report. Do not run the 48 secondary-length cases merely to fill a benchmark table. State exactly which strata were not evaluated. If the pilot is noisy or its uncertainty could include `0.80`, use one bounded measurement repair; do not treat an unfavorable median as a refutation.

### 9.3 Repairs and stopping

Ordinary repairs include correcting orientation, unsupported API usage, allocation policy, or a verified measurement defect. They do not include changing the research question to a favorable low-rank distribution, excluding the affine comparator, switching accuracy contracts, or inventing a new learned method. Once a scientifically adequate outcome is reached, stop optional testing.

If a baseline is stronger than expected and removes the gain, that is a result. If only rotating projections or coordinate-low-rank streams work, report the limited mechanism control and the failed primary application. If preparation dominates, report measured costs and the honest break-even model. If finite precision fails, identify the responsible step with the smallest diagnostic rather than rerunning a large matrix.

## 10. Repository, interfaces, and intended commands

Create a small repository with these responsibilities; engineering layout may change if interfaces and evidence remain equally clear:

```text
README.md                       setup, scope, commands, actual results
POC_EXECUTION_BRIEF.md           this instruction, copied unchanged
requirements.txt                minimal compatible dependencies
requirements-lock.txt           exact installed versions after setup
config.json                     fixed workload, precision, caps, seed schema
run.py                          inspect/check/smoke/pilot/freeze/confirm/report/all
src/math_core.py                 validation, canonicalization, reference, summary, compiler
src/wy.py                       chunk preparation and application
src/backends.py                  NumPy and optional MLX execution/synchronization
src/workloads.py                 deterministic generators and IDs
src/benchmark.py                 timings, resource watchdog, budget ledger, statuses
src/analyze.py                   predeclared analysis and essential plots/tables
tests/test_math.py               exact and independent correctness controls
inputs/                         unmodified received source files
results/environment.json
results/run_manifest.json
results/protocol.json
results/ledger.jsonl
results/measurements.jsonl
results/correctness.json
results/summary.csv
results/figures/
REPORT.md                        scientific outcome and limitations
```

Use plain dictionaries or small dataclasses. Required conceptual interfaces are:

```python
canonicalize(keys32, values32, gates32) -> coefficients64, metadata
summarize(coefficients64, summary_policy) -> P64, B64, timings
prepare(method, stream, backend, config) -> representation, metadata
apply(representation, initial_state) -> final_state
check_representation(representation, P_ref, B_ref, config) -> diagnostics
measure_block(block_id, methods, config, budget_ledger) -> result_records
```

Here `stream` retains the original float32 arrays and any explicitly derived float64 forms; it is not a regenerated dataset. Do not overload `prepare` to regenerate data or hide hardware transfers. `apply` must not mutate canonical inputs. Have `run.py all` execute the authorized gates, inspect their results, follow branch/stop rules, and always attempt evidence preservation and a report. Before G0 create `results/run_manifest.json` with configuration hash, stage state, and `protocol_status="unfrozen"`. The report command must accept `--config config.json` when no frozen protocol exists, read whatever manifest/ledger/evidence is available, and identify the stage reached. Never fabricate a frozen protocol on an early stop. It must not blindly run G4 after failed correctness or calibration.

The following are **interfaces and commands you must create and verify**, not commands already verified during planning:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pytest psutil matplotlib
# Only on the inspected compatible Mac, for the proposed primary Metal path:
python -m pip install mlx
python -m pip freeze > requirements-lock.txt

python run.py inspect --config config.json
python -m pytest -q tests/test_math.py
python run.py check --config config.json
MLX_ENABLE_TF32=0 python run.py smoke --config config.json
MLX_ENABLE_TF32=0 python run.py pilot --config config.json
python run.py freeze --config config.json
MLX_ENABLE_TF32=0 python run.py confirm --protocol results/protocol.json
python run.py report --protocol results/protocol.json
# Early-stop reporting when no frozen protocol exists:
python run.py report --config config.json
# End-to-end alternative after setup; it must honor every gate and cap:
MLX_ENABLE_TF32=0 python run.py all --config config.json --budget-minutes 240
```

Use either the staged commands or `all`; do not repeat completed stages just because both examples are listed. Direct pytest/check invocations must write elapsed time and case counts into the same ledger so they cannot bypass the cap. On a CPU fallback, the MLX environment variable is harmless but output metadata must say CPU. Do not install MLX when its prerequisites are unmet.

Include a compact initial configuration with the following resolved scientific fields:

```json
{
  "root_seed": 20261104,
  "shapes": [[32, 64], [128, 128]],
  "length_multipliers": [1, 8, 32],
  "primary_length_multiplier": 8,
  "families": ["isotropic", "correlated_projection"],
  "confirmation_replicates": 6,
  "serial_reuse": 16,
  "execution_dtype": "float32",
  "reference_dtype": "float64",
  "compiler": "stable_2r",
  "wy_chunk_candidates": [16, 32, 64],
  "rounds_default": 5,
  "rounds_budget_fallback": 3,
  "max_group_repeats": 20,
  "primary_ratio_threshold": 0.8,
  "state_tolerance": 0.0001,
  "packed_map_tolerance": 0.00001,
  "float64_map_tolerance": 1e-10,
  "experiment_minutes": 240,
  "max_timed_cells": 500,
  "max_tiny_cases": 1500,
  "max_disk_gib": 2
}
```

Add inspected environment values and development choices to the frozen protocol; do not leave unresolved placeholders for the operator. Dependency versions must come from the actual installation rather than invented compatibility pins.

## 11. Required records, interpretation, and deliverables

### 11.1 Raw records sufficient to reproduce the decision

Save source/data hashes, generator IDs, seed tuples, git commit, installed versions, actual backend/device/precision, normalization transformations, executable/cache policy, BLAS settings, dimensions, original/compiled counts, chosen support, key singular spectrum, `P` spectrum, norm/accuracy diagnostics, target magnitudes, representation bytes, all setup/application timings, first-use costs, individual timing rounds, memory peaks, failures, and cumulative budget.

Preserve unexpected and unfavorable results. Avoid storing every state for every repetition. Save a small failing case's arrays and the fixed primary input identifiers; deterministic regeneration must reproduce the inputs. Provide a command that regenerates the main tables and plots from raw result files without rerunning benchmarks.

### 11.2 Outcome categories

| Category | Evidence required and permitted interpretation |
|---|---|
| **Supported contribution** | The predeclared primary criteria pass against strong valid comparators. Claim only the tested terminal-state component, backend, precision, distributions, and reuse contract. An application would still need compatible coefficients and output requirements. |
| **Informative negative result** | Correctness and adequate precision exclude the specified useful improvement, or a verified analytic obstruction defeats a particular proposed use. Explain whether setup, arithmetic, sequential execution, forgetting, or another measured factor accounts for it. Do not infer universal inferiority. |
| **Practical infeasibility** | Measured target-device costs or a supported scaling estimate prevent the intended study/application within caps. Give completed evidence and the exact blocking resource. A smoke test is not a substitute. |
| **Inconclusive evidence** | Timing uncertainty, insufficient independent blocks, missing target access, an essential inaccessible source, or the available budget prevents a justified decision. State what remains unresolved. |
| **Invalid experiment** | Candidate, baseline, data, or timing defects invalidate the comparison and could not be repaired within bounds. A bug or timeout is not a scientific refutation. |

The categories may differ by claim: for example, the theorem can remain mathematically supported while the float32 implementation has an informative numerical limitation and M3 performance is blocked. Identify one primary outcome and qualify other claims individually.

### 11.3 Final artifacts for the stages actually reached

Produce runnable code and concise usage instructions wherever implementation occurred; exact dependency/environment records; the frozen protocol and deviation log; raw measurements and failed-run records wherever experiments occurred; and `REPORT.md` with evidence-backed conclusions.

If G4 ran, essential outputs are: a table of total and warm application costs by primary stratum; correctness/failure counts including the denominator of scheduled runs; preparation-versus-application breakdown; a reuse plot clearly separating measured points from modeled curves; and an accuracy/conditioning summary. Add secondary plots only when they answer a stated uncertainty. Report both geometric mean and individual block data, not a single favorable speedup.

The report must distinguish the source's proofs, source-reported observations, planning probes, your executed Mac measurements, and new hypotheses. Explain the scope revision from neural-network acceleration, the closest prior art, deviations from this protocol, and residual limitations. If the result is negative, do not rename a raw-loop speedup as success. If only code was completed, say which validation was not performed.

For an analytic stop, provide the verifiable derivation/counterexample and its minimal reproduction if numerical code was needed. Do not fabricate benchmark tables or code modules solely to satisfy a later-stage deliverable. End with one specific evidence-based next step, if warranted; “no further integration justified under the tested contract” is an acceptable decision.

## 12. Supporting research and review record

### 12.1 Literature assessment and coverage

Targeted searches were conducted on **4 November 2026**, following the notebook's references backward to WY/operator geometry and forward through signed recurrent models, DeltaProduct, CKDA, and current implementation documentation. Search terms included affine delta replay/compression, generalized Householder contraction factorization, signed polar factorization, projection products, compact WY, and the `2r`/`2r-1` bounds. This is a decision-focused search, not an exhaustive global novelty determination.

| Closest work | Verified content and effect on this study |
|---|---|
| Yang et al., DeltaNet [R1], §3.2, Eqs. (7)–(11), Appendices B.1/B.2 and D | Delta recurrence, Householder/WY reformulation, and chunk computation already exist. Use its mathematics as a baseline, with an independently checked row convention. |
| Grazzi et al. [R2], §4.3 Proposition 1(2), Appendix C.2 | Every real contraction admits at most `3r` generalized Householder factors. Signed contraction factorization itself is not new. |
| Siems et al., DeltaProduct [R3], §4 and §5.1 | Learned multiple updates per token and signed transitions improve expressivity; this is distinct from exact post-hoc endpoint replay. |
| Siems et al., CKDA [R4], Appendix A.5 Proposition 10 | A richer Householder-plus-signed-diagonal primitive gives at most `max(1,2r-2)` contraction factors. It does not establish the same plain-delta affine theorem, but narrows novelty language. |
| Combettes–Yamada [R5], Propositions 2.1 and 2.5; Ryu–Hannah–Yin [R6], SRG composition framework and Appendix C Fact 17 | Dissipation and norm-angle composition are established foundations of the gate-cap obstruction. |
| Kuo–Wu [R7] | Older partial-isometry versus projection-factor-count distinction; accessible indexed abstract supports relevance, but the full paper was not recovered. No unverified theorem detail is needed by the implementation. |
| Schreiber–Van Loan [R8] | Classical WY provenance; author-repository abstract was inspected, full report access was incomplete. The executable equations here are independently derived and supported by [R1] and A1. |
| Gated DeltaNet [R9], §3 and Appendix A | Decay changes the recurrence and its active-subspace behavior; excluded rather than silently treated as the attached rule. |

Recent SparseGDN [R17] was checked as a possible alternative. It uses approximate sparse recurrent events and reconstructs outputs for every token, which addresses a different accuracy/output contract. Its reported GPU results are not compatible benchmark numbers for this exact terminal-state Mac study. Do not silently substitute that project for the supplied mathematics.

A forward-search result, *Semidirect Fourier Delta Attention* [R18], describes a complex phase-controlled WY formulation in its accessible abstract, but full source content was not recovered. It is a recorded coverage limitation, not implementation evidence. CKDA's manuscript was initially recovered from a readable copy after direct arXiv access failed; a later retry also recovered the official arXiv v1 HTML, where Appendix A.5 and its proof were verified. None of these access limits prevents this diagnostic, because it does not depend on a global priority claim or an unverified external algorithm.

### 12.2 Three review passes and material dispositions

**Mathematical and novelty review.** The construction, multiplication order, direct bias injection, translation rank argument, and positive-factor obstruction were challenged against the attachments and primary prior art. The stable construction is constructive and needs no oracle, but forming the summary and SVD has real cost. The short-mode zero denominator and rounded-rank problem were reproduced. Disposition: stable `2r` primary; literal stored-input target; exact coordinate support only; no priority claim. A preliminary concern about a triangular-solve sign was reconciled against the current published PDF and its Appendix B.2: the correct plus sign is supplied, with an exact implementation guard; no allegation of a defect in that published formula is retained.

**Experimental review.** Potential false gains include comparing only with a Python loop, giving the compiler an oracle basis, ignoring preparation, reusing cached lazy outputs, timing queued work as serial requests, counting timing rounds as independent samples, and benefiting from erased incoming state. Disposition: mandatory affine and WY baselines; common inputs and preparation accounting; separate algorithm cost from test-only oracle work; synchronization per request; block-level inference; fixed primary population; a development-selected baseline and noise sensitivity check for negative claims; forgetting and weak-gate controls. Favorable-only low-rank or rotating examples cannot establish the primary claim.

**Execution review.** The handoff was traced from an empty Mac project: sources and definitions are identified; datasets are generated locally; CPU SVD/triangular preparation avoids assuming unsupported GPU precision; backend installation is gated; memory, time, cases, repairs, and confirmation branches are bounded; intended interfaces are distinguished from verified commands; the report finalizer also works before a protocol has been frozen. Unknown RAM/OS/backend behavior remains an explicit G0/G2 gate, not a fabricated hardware specification.

Stress cases are addressed as follows:

| Stress scenario | Disposition |
|---|---|
| Nonconstructive theorem or hidden oracle | Not present in the selected proof; expose and charge summary, SVD, basis discovery, and reflection work. Stop if an indispensable new oracle appears. |
| Assumptions fail on a real model | No whole-model claim; reject state-dependent coefficients and all-token substitution. |
| Mechanism is already known | Attribute it; contribute a bounded utility/limitation result instead of rebranding. |
| Only favorable synthetic cases work | Primary full-rank result remains negative/inconclusive; controls cannot replace it. |
| Strong baseline removes gain | Valid negative outcome; affine is the decisive no-SVD ablation. |
| Preprocessing dominates | Report complete costs and break-even, with no rescue by arbitrary reuse. |
| Finite precision changes the map | Independent packed-map and executor checks; preserve failures and tiny-update diagnostics. |
| Memory/runtime exceeds forecast | Drop predeclared secondary work before freeze; preserve meaningful primary dimensions or report infeasibility. |
| Small pilot is inconclusive | One bounded precision repair, then fresh protected confirmation if affordable; no optional stopping for a favorable result. |
| Source/data inaccessible | No dataset dependency; core equations and the explicit projection control are included. Identify any remaining source-specific verification as unavailable. |
| Main hypothesis is false | Complete the scoped negative report, artifacts, and decision without a larger rescue project. |

Residual risks remain: no trained-model reuse frequency has been established; portable implementations are not optimized vendor kernels; six independent blocks per stratum give limited distributional coverage; float32 normwise accuracy is not a downstream-task guarantee; and historical novelty of the combined affine statement is unresolved. These limits must remain visible in the final report.

## References

Attachment locations use zero-based notebook cell indices and the source section names. User-provided mathematics is cited as A1/A2; its authorship is not identified in the files and must not be invented. Web sources below were accessed during planning on 4 November 2026 unless an access limitation is stated.

**[R1]** Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim. **Parallelizing Linear Transformers with the Delta Rule over Sequence Length.** NeurIPS 2024. [Published paper][R1]. Critical locations: §3.2, Eqs. (7)–(11), printed pp.4–5; Appendix B.1/B.2, pp.22–23; Appendix D, Listing 1. Relevant content was read; the published PDF was also checked to resolve equation-version ambiguity.

**[R2]** Riccardo Grazzi, Julien Siems, Arber Zela, Jörg K. H. Franke, Frank Hutter, Massimiliano Pontil. **Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.** ICLR 2025; arXiv v5, 18 March 2025. [Version read][R2]. Critical locations: §4.3, Proposition 1(2); Appendix C.2. DOI: [10.48550/arXiv.2411.12537](https://doi.org/10.48550/arXiv.2411.12537).

**[R3]** Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, Riccardo Grazzi. **DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products.** NeurIPS 2025; arXiv v7, 22 October 2025. [Version read][R3]. Critical locations: §4, §5.1, Appendix A. DOI: [10.48550/arXiv.2502.10297](https://doi.org/10.48550/arXiv.2502.10297).

**[R4]** Julien Siems, Riccardo Grazzi, Korbinian Pöppel, Jaisidh Singh, Arber Zela, Timur Carstensen, Jenia Jitsev, Frank Hutter, Volkan Cevher, Antonio Orvieto, Aaron Klein. **Complex KDA: Understanding and Enhancing the Expressivity of Kimi Delta Attention.** September 2026 preprint. [Stable arXiv record][R4]. Critical location: Appendix A.5, Proposition 10, manuscript p.24. Following earlier access failures, Appendix A.5 and its proof were read in the [official arXiv v1 HTML](https://arxiv.org/html/2609.24797v1), as well as an initially recovered [readable copy](https://www.researchgate.net/publication/414564250_Complex_KDA_Understanding_and_Enhancing_the_Expressivity_of_Kimi_Delta_Attention); identity was corroborated by the [official repository](https://github.com/OpenEuroLLM/ComplexKDA).

**[R5]** Patrick L. Combettes, Isao Yamada. **Compositions and Convex Combinations of Averaged Nonexpansive Operators.** *Journal of Mathematical Analysis and Applications* 425(1), 55–70, 2015. [Author manuscript][R5]. Critical locations: Proposition 2.1(iii), manuscript p.2; Proposition 2.5, p.4. These were read directly.

**[R6]** Ernest K. Ryu, Robert Hannah, Wotao Yin. **Scaled Relative Graphs: Nonexpansive Operators via 2D Euclidean Geometry.** *Mathematical Programming* 194, 569–619, 2022; arXiv v4, 16 June 2021. [Manuscript][R6]. Relevant locations: operator-composition geometry and Appendix C, Fact 17, manuscript p.42. DOI: [10.1007/s10107-021-01639-w](https://doi.org/10.1007/s10107-021-01639-w).

**[R7]** Kung-Hwang Kuo, Pei Yuan Wu. **Factorization of Matrices into Partial Isometries.** *Proceedings of the American Mathematical Society* 105(2), 263–272, February 1989. [DOI][R7]. The indexed institutional abstract was retrieved; the [institutional PDF route](https://ir.lib.nycu.edu.tw/bitstream/11536/4410/1/A1989T910900001.pdf) returned an unreadable landing response and the AMS PDF returned 403. No full-paper verification is claimed.

**[R8]** Robert Schreiber, Charles Van Loan. **A Storage-Efficient WY Representation for Products of Householder Transformations.** *SIAM Journal on Scientific and Statistical Computing* 10(1), 53–57, 1989. [DOI][R8]; [author technical-report record](https://hdl.handle.net/1813/6704). Abstract/record checked by the literature reviewer; full report access incomplete. Used for historical attribution, not an unverified implementation dependency.

**[R9]** Songlin Yang, Jan Kautz, Ali Hatamizadeh. **Gated Delta Networks: Improving Mamba2 with Delta Rule.** ICLR 2025; arXiv v3, 6 March 2025. [Version read][R9]. Relevant locations: §3.1, §3.3, Appendix A. DOI: [10.48550/arXiv.2412.06464](https://doi.org/10.48550/arXiv.2412.06464).

**[R10]** FLA contributors. **Flash Linear Attention — Installation Guide.** Living official implementation documentation, retrieved 4 November 2026. [Backend table][R10]; [repository](https://github.com/fla-org/flash-linear-attention). The listed backend support is not a verified Apple Metal path; inspect current documentation before any future attempt to use that implementation.

**[R11]** MLX Contributors / Apple. **Build and Install; Data Types.** Retrieved documentation version 0.32.4, 4 November 2026. [Python installation requirements][R11]; [float64 CPU-only note](https://ml-explore.github.io/mlx/build/html/python/data_types.html). These are documentation facts, not target-Mac inspection results.

**[R12]** MLX Contributors / Apple. **Numerical Precision.** Living official documentation, retrieved 4 November 2026. [Full-float32 setting][R12]. Relevant location: the `MLX_ENABLE_TF32=0` launch instruction and its hardware-dependent precision explanation.

**[R13]** NumPy Developers. **`numpy.linalg.svd`; `numpy.show_runtime`.** Living official reference documentation, retrieved 4 November 2026. [SVD and LAPACK notes][R13]; [runtime inspection](https://numpy.org/doc/stable/reference/generated/numpy.show_runtime.html).

**[R14]** SciPy Developers. **`scipy.linalg.solve_triangular`.** Living official reference documentation, retrieved 4 November 2026. [API reference][R14]. Relevant parameters: `lower`, `unit_diagonal`, `check_finite`; failure behavior documented under `Raises`.

**[R15]** MLX Contributors / Apple. **Lazy Evaluation.** Living official documentation, retrieved 4 November 2026. [When to evaluate][R15]. An already evaluated result is not a fresh execution.

**[R16]** MLX Contributors / Apple. **`mlx.core.synchronize`; memory management.** Living official documentation, retrieved 4 November 2026. [Synchronization][R16]; [peak-memory API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html); [memory management](https://ml-explore.github.io/mlx/build/html/python/memory_management.html).

**[R17]** Jiedong Zhuang, Peng Zhao, Jialin Tang, Haoji Hu. **SparseGDN: Chunk Sparse in Linear Attention.** arXiv:2610.41420v1, submitted 30 October 2026; retrieved manuscript dated 2 November 2026. [Paper][R17]. Relevant locations: abstract, §2, §4 “Conclusion and Limitations.” Read as a different approximate all-token mechanism; its published performance numbers are not used as comparators.

**[R18]** Tiantian Zhang. **Semidirect Fourier Delta Attention: Phase-Controlled Delta Memory with Constructive Chunk-WY Kernels.** 2026 preprint, arXiv:2607.11897. [Stable record][R18]. Abstract-level coverage only in this investigation; full-text retrieval failed. No theorem or performance claim from this source is required by the study.

[R1]: https://papers.neurips.cc/paper_files/paper/2024/file/d13a3eae72366e61dfdc7eea82eeb685-Paper-Conference.pdf
[R2]: https://arxiv.org/html/2411.12537v5
[R3]: https://arxiv.org/html/2502.10297v7
[R4]: https://arxiv.org/abs/2609.24797
[R5]: https://pcombet.math.ncsu.edu/jmaa3.pdf
[R6]: https://arxiv.org/pdf/1902.09788
[R7]: https://doi.org/10.1090/S0002-9939-1989-0977922-1
[R8]: https://doi.org/10.1137/0910005
[R9]: https://arxiv.org/html/2412.06464v3
[R10]: https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md
[R11]: https://ml-explore.github.io/mlx/build/html/install.html
[R12]: https://ml-explore.github.io/mlx/build/html/usage/precision.html
[R13]: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
[R14]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html
[R15]: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
[R16]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html
[R17]: https://arxiv.org/pdf/2610.41420
[R18]: https://arxiv.org/abs/2607.11897
