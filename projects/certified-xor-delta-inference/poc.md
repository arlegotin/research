# XOR-enabled exact neural reuse: research and local proof-of-concept plan

**Research date:** 29 October 2026.  
**Target:** one MacBook Pro with an M3 Max; actual memory, storage and macOS version must be discovered.  
**Deliverable type:** standalone research and implementation handoff, not a report of a successful neural-network benchmark.  
**Recommended primary direction:** exact reuse of binary threshold-layer outputs, certified by the net XOR difference from a cached anchor and a mathematically valid threshold margin.

## 1. Executive assessment

The central ambition is sound: design representations and execution rules that expose algebraic structure so that a network does less work. However, most of the six proposed mechanisms already have substantial prior art. Binary dot products, learned Hadamard layers, delta inference, temporal-sparsity training, Boolean circuit optimization, hash-based attention filtering and reversible coupling are established research areas. The specific XOR-indexed weight matrix in the original proposal is explicitly described in an earlier dyadic-matrix neural-processor patent. A new implementation or combination does not establish scientific novelty. [XNOR16] [DYADIC20] [EV22] [TEMP21] [LOGIC22] [RABIT26] [REV17]

**Select a narrow, falsifiable refinement:** a binary threshold block keeps an anchor input and its exact integer preactivations. It compares the current input with that anchor using XOR. If a bound on the resulting preactivation change is too small to cross any relevant threshold, it returns the cached binary outputs without updating the accumulators. Changes that reverse before the anchor is replaced cancel in the XOR difference. A failed certificate triggers exact computation; there is no approximate reuse threshold.

The prospective difference is **using the net anchor-relative change to certify the actual binary module and avoid even eager accumulator updates**. Neither the bit-flip bound nor certified caching in general is new. DATE 2021 derives the relevant binary-margin bound; LipCache in 2026 uses anchor-dependent certified reuse, but its stated guarantee concerns a proxy classifier, not exact outputs of its main model. No directly identical panel-anchor implementation was located in the targeted search. That supports a provisional research question, not a novelty declaration. [MARGIN21] [LIPCACHE26]

The immediate contribution, if successful, would be an **exact inference method, a reproducible Apple Silicon implementation, and an empirical map of when it beats strong existing execution choices**. Methodological novelty remains conditional on the final equivalence review. A negative outcome can still establish a useful boundary: certified stability may be too expensive or conservative to exploit after a good packed dense or eager-delta implementation.

### 1.1 The local experiment

Use real, chronological six-channel inertial recordings from UCI's *Smartphone-Based Recognition of Human Activities and Postural Transitions*, dataset 341. Construct explicitly causal, fixed-coordinate features and train a small binary classifier. Preserve the official subject-level test partition. Use identical exported weights and threshold semantics for every exact execution method. [HAR341]

The **primary intervention is the second binary hidden layer**. The first layer receives thermometer-encoded features, which have a stronger simple implementation through prefix-sum lookup tables. Give that optimization to every method. This prevents an avoidable encoding cost from becoming the apparent XOR contribution. The primary question is whether certified reuse beats the fastest validated, development-selected exact execution baseline for the learned hidden layer. A separate whole-stream measurement includes preprocessing, packing, dispatch, the first layer, the floating-point readout, resets and cache maintenance.

The program makes three distinct claims possible:

1. **Correctness:** outputs equal the canonical exported binary model for every valid input sequence, supported by proof plus exhaustive and adversarial implementation tests.
2. **Component efficiency:** a specific learned binary layer runs faster on held-out real streams than the strongest executable exact alternatives, with uncertainty and absolute times reported.
3. **Application value:** the resulting whole classifier lies on a useful quality/latency/memory frontier against compact real-valued, quantized and ordinary cached causal models.

Claim 1 does not imply 2, and 2 does not imply 3. None of these establishes general LLM acceleration, improved intelligence, faster training, superiority on arbitrary video, or a new universal neural architecture.

### 1.2 What is known now and what is not

| Evidence class | Status at delivery |
|---|---|
| Inspected external evidence | Papers, official repositories, dataset metadata and hardware/software documentation listed in Section 12. Important access limitations are recorded there. |
| Analytical deductions | Exact update identities, certificate proof, cost and memory models, and the experimental design below. |
| Checks actually executed | Small Python/NumPy algebra and cache-state checks on Linux x86-64: 37,448 update cases, 476,136 certificate conditions, 9,000 randomized panel steps, and small Walsh identities; all assertions passed. See Section 11.3. |
| Not executed | M3 kernels, package installation on the target, dataset training, local baseline reproduction, latency/energy experiments, or confirmatory evaluation. |
| Remaining central uncertainty | Whether useful trained representations have enough *certifiable* stability to outweigh checking, copying and refreshing costs. |

## 2. Problem, scope, assumptions and hypotheses

### 2.1 Beneficiary and computational problem

The first beneficiary is a developer running continuous classification locally, especially at batch size one. Today, such a developer can use a compact dense/quantized network, a packed binary network, ordinary causal state reuse, or delta inference. The possible additional bottleneck is repeated evaluation of binary hidden units whose outputs remain unchanged despite small input changes. Whether that bottleneck is material on this laptop is a measurement question.

At time t, a frozen binary layer receives x_t in {-1,+1}^n and computes, for row j,

    z_j(t) = sum_i w_ji x_i(t),    w_ji in {-1,+1},
    y_j(t) = 1 if z_j(t) >= tau_j, otherwise 0.

The integer threshold tau_j includes all valid frozen affine/normalization terms. The public layer output is the packed bit vector y_t. Internal preactivations are private state; a baseline that only needs output bits may terminate its dot products early.

Optimize measured wall-clock time and memory traffic while preserving these outputs exactly. The full model has a common deterministic FP32 readout. Its logits must match when supplied the same exact hidden bits and the same readout implementation. Exactness is relative to this exported model, not to an unrelated full-precision network or the stochastic training graph.

### 2.2 Algebraic distinctions

**XOR-only networks.** A composition of fixed XOR gates and fixed bit flips is affine over GF(2), and cannot express AND. GF(2) matrix rank and real matrix rank are different. A parity map is not a signed binary dot product.

**Packed binary arithmetic.** With bit 1 representing +1 and bit 0 representing -1,

    z = n - 2 popcount((x_bits XOR w_bits) AND valid_mask).

The result is an ordinary integer sum. Popcount and accumulation remain necessary. One XOR instruction is not a complete dot product, and packed storage savings are not end-to-end speedups. [BNN16] [XNOR16]

**Walsh structure.** For an unnormalized Sylvester matrix H of order n=2^k,

    H H = n I,
    W_ij = kernel[i XOR j],
    W x = H ((H kernel) elementwise_mul (H x)) / n.

This is real-valued addition/subtraction plus elementwise multiplication. The indexing algebra is XOR, but the transform is not a packed XOR computation. If Q=H/sqrt(n), the family Q diag(lambda) Q is symmetric, has constant diagonal, commutes with itself, and consecutive linear members collapse. Nonlinearities, different factors or corrections change the family. These are deductions, not evidence of task performance.

**Reversibility.** XOR coupling reconstructs discrete states, but does not supply useful gradients, remove optimizer state, or eliminate recomputation within its branch functions. Reversible networks and checkpointing are obligatory alternatives to any later memory claim. [REV17]

### 2.3 Assumptions and hypotheses

| ID | Hypothesis or prerequisite | Supporting rationale | Strong alternative explanation | Observation that weakens it |
|---|---|---|---|---|
| H0 | The exported threshold graph is deterministic and equivalent across runtimes. | Integer sums and fixed thresholds admit an exact specification. | Export or tie rules changed the model. | Any output-bit discrepancy, invalid tail bits, or nonmonotone folded affine rule. |
| H1 | An anchor-relative bound can avoid computing some current preactivations exactly. | The 2K flip bound and threshold slack prove this under the stated conditions. [MARGIN21] | A faulty certificate appears efficient by returning stale wrong results. | One counterexample invalidates the implementation; it does not require a statistical test. |
| H2 | Net XOR cancellation provides value beyond eager deltas, cumulative-change bounds and exact-input memoization. | Net distance can be smaller than total changes since an anchor. | Exact repeats, cheap changed-word updates, or a two-entry cache explain the benefit. | Gains disappear against those controls; positive-distance certified hits are negligible. |
| H3 | Useful trained hidden representations have a favorable certificate/cost distribution. | Real streams contain temporal dependence; prior event methods exploit it. [EV22] | Stable units are constant, duplicated, uninformative or trained on leaked adjacent samples. | No measured headroom after simplification, or only weak classifiers admit reuse. |
| H4 | A layer gain survives whole-stream costs. | Possible only if that layer occupies a material share of baseline time. | Preprocessing, head computation, runtime dispatch or idle power dominate. | Whole-stream confidence interval excludes a useful saving or the model is dominated on quality/cost. |
| H5, optional | Training for margins or temporal stability improves the efficiency frontier. | Both ideas have prior art. [MARGIN21] [TEMP21] | Regularization merely suppresses responsiveness or adds training compute. | Transition quality falls, smaller dense models win, or benefit disappears with equal optimization effort. |

H5 is not required for the core PoC. Do not add new training machinery before H2/H3 have measurable headroom.

### 2.4 Best case, strongest objections and headroom

The best case is a useful binary representation with modest positive margins and changes that cancel or remain confined to a small Hamming neighborhood. One comparison per panel can then avoid many weight reads and accumulator writes while preserving every output bit.

The strongest objection is that binary dense kernels are already very cheap. A certificate may scan many anchor vectors, miss because one neuron is near a threshold, then perform a full sum to recover a new margin. A dense output-only implementation may have stopped that sum early. Eager changed-word updates can exploit sparsity without panel anchors. Furthermore, a 50 Hz sensor classifier may already take only microseconds: a large ratio can have little practical value.

For a layer with m rows, S=ceil(n/64) words per input, panel size g and P=ceil(m/g), a first-order model is

    T_cert ~= P*S*c_scan + (1-h)*m*S*c_refresh
              + T_anchor_writes + T_dispatch + T_output,

where h is the measured fraction of certified panels, not the fraction whose outputs happen to be unchanged. The model is approximate because early certificate failure, vectorization, cache locality and refresh choices alter the constants. Fit constants from local measurements; validate the prediction against complete executions.

With measured baseline layer fraction f and layer speedup s, an overhead-free whole-model upper bound is

    speedup <= 1 / ((1-f) + f/s).

For illustration only, f=0.2 and s=2 permit at most about 1.11x. All new overhead reduces that bound. No 1.5x or other arbitrary percentage target is imposed.

For independent random weights and approximately Gaussian preactivations, input flip fraction p gives sign disagreement approximately acos(1-2p)/pi, which is about 2*sqrt(p)/pi for small p. This diagnostic explains potential amplification of changes; it is not a prediction for trained networks. Local panels, biased thresholds and learned correlations can behave differently.

## 3. Research findings and novelty assessment

### 3.1 Comparison across the original directions

| Direction | Closest established mechanisms | Potential value and principal limitation | Decision |
|---|---|---|---|
| Packed XOR/XNOR neural arithmetic | BinaryNet, XNOR-Net, XOR-Net, Larq; ternary BitNet is related but different. [BNN16] [XNOR16] [XOR20] [LARQ] [BITNET] | Mature storage/arithmetic savings; quality, kernels and hardware determine practical wins. | Reuse as infrastructure and baseline; do not claim this primitive is new. |
| Walsh/XOR-circulant projections | Adaptive Fastfood, ACDC, dyadic-matrix disclosure, block WHT, Monarch and BTT. [FASTFOOD15] [ACDC16] [DYADIC20] [WHT22] [MONARCH22] [BTT24] | Native MLX support makes local work feasible. Restricted expressivity and strong alternatives weaken novelty of a simple mixer plus correction. | Evidence-triggered alternative, not primary. |
| Exact temporal reuse | Sigma-Delta networks, CBinfer, DeltaCNN, EvNets and temporal-sparsity training. [SD16] [CBINFER] [DELTACNN22] [EV22] [TEMP21] | Reduces repeated work; sparse updates, checking and state costs may dominate. | Primary refinement: certified lazy reuse of the actual binary module. |
| GF(2) basis compression | Rank factorization and XOR-circuit synthesis/common-subexpression optimization. [GF2CIRCUIT18] | Valid for parity maps; rank r stores r(m+n) bits, but not a replacement for real-valued binary dot products. | Do not select without evidence that a useful task admits compact parity structure. |
| Hamming routing | Reformer and contemporary rotated binary sparse-attention proxies such as RaBitQCache. [REFORMER20] [RABIT26] | Candidate filtering can save large attention work, but recall, indexing and irregular data movement matter. | Crowded and scale-dependent; no local LLM project is required to test the foundational idea. |
| XOR-reversible memory | Classical reversible coupling and RevNets. [REV17] | Exact state reconstruction is plausible; training memory also includes branch intermediates, shadows and optimizer states. | Defer unless memory is the demonstrated bottleneck. |
| Boolean neuron/circuit learning | Differentiable logic gate networks, convolutional logic gates and Light DLGN. [LOGIC22] [CONVLOGIC24] [LIGHT26] | Real nonlinear gates avoid XOR-only affine collapse; training/discretization and wiring are substantial existing research. | Relevant alternative paradigm; not silently excluded in favor of Transformers. |

This selection changes the original suggested first Walsh experiment deliberately: prior-art overlap is more direct than the initial memo implies, whereas exact cancellation-aware reuse has a sharply isolatable local question. It preserves the purpose of eliminating work through algebraic structure.

### 3.2 Closest comparison matrix for the selected refinement

| Work | Problem and mechanism | Assumptions/evidence and limits | Implementation/resources | Exact overlap and use here |
|---|---|---|---|---|
| Sigma-Delta Quantized Networks, 2016/17 [SD16] | Transmit quantized activation changes. | Temporal redundancy and a precision/activity tradeoff; does not establish this certificate. | Historical software, no target-laptop timing used here. | Prior art for changing computational effort with signal change. |
| CBinfer, 2017/18 [CBINFER] | Cache feature maps and update changed regions. | Video locality; change thresholds can alter outputs. | Original CUDA implementation; Apache-2.0 repository. | Region caching and update granularity already exist. |
| DeltaCNN, CVPR 2022 [DELTACNN22] | End-to-end sparse frame differences and persistent state. | Real GPU execution; repository allows a zero threshold. Published results are not M3 measurements. | CUDA; archived repository; CC BY-NC 4.0 code. | Required algorithmic comparator; faithful local integer specialization, not an asserted CUDA port. |
| Event Neural Networks, ECCV 2022 [EV22] | Cached accumulators, activations and untransmitted residuals. | Supports exact linear update algebra; main operating points trade quality for fewer operations. CPU demonstration is hardware/configuration specific. | MIT PyTorch reference; custom C++ kernels in the paper's CPU experiment. | Eager exact update semantics are already established. Reuse state/reset concepts and verify a dense integer specialization. |
| Temporal-sparsity training, 2021 [TEMP21] | Penalize activation changes. | Video experiments; no general guarantee of useful stability. | Full original training environment not reproduced here. | Invalidates broad novelty for learning temporally stable representations. |
| Binary-margin tolerance, DATE 2021 [MARGIN21] | Bound sum change by 2 times the number of bit flips. | Formal theorem; experiments address bit-error robustness. | Mathematical result reusable without specialized equipment. | Supplies the known certificate bound; runtime anchor use is the refinement. |
| Fuzzy neuron memoization, 2022 [FUZZY22] | Predict whether a recurrent neuron output can be reused. | Approximate predictor and error-control policy. | Hardware-oriented evidence; no faithful native Mac runtime established. | Close reuse precedent, but not exact certification of this same module. |
| LipCache, August 2026 [LIPCACHE26] | Per-anchor margin/norm reuse radii with fallback. | Explicitly certifies GuardNet-side consistency; does not guarantee MainNet equivalence. | Edge classification study; no published number is imported as a local comparison. | Broad certified caching is known. This plan certifies actual hidden bits with integer bounds. |
| DeepCache, MobiCom 2018 [DEEPCACHE18] | Reuse mobile-video regions under similarity tests. | Approximate matching and system overhead matter. | Old Android/ncnn/RenderScript code; not an optimized Mac baseline. | Closer precedent than diffusion DeepCache for this proposal. |
| Locality-aware BNN skipping and early termination [LOCAL24] [EARLY26] | Skip predictable outputs or stop accumulations once a threshold is determined. | Exact and approximate mechanisms must be separated. Some recent full texts were inaccessible. | Often hardware-specific. | Give exact output-only early termination to the dense baseline; seek further equivalents before novelty claims. |
| Binary weight-row Hamming reuse [WEIGHTPAT25] | Reuse dot products across similar weight rows. | Patent technical disclosure, not evidence of measured M3 benefit. | No verified local reference implementation. | Another cancellation/redundancy mechanism; not temporal anchor reuse and not a legal conclusion. |

### 3.3 Six explicit prior-art answers

1. **What exists?** The underlying binary arithmetic, temporal deltas, cached state, margin bounds and general certified caching all exist. The unconstrained proposal is not a new scientific idea.
2. **What can be reused?** Larq's native packed operators and converter concepts; EvNet's state semantics; Arm popcount intrinsics; standard BNN training; the public UCI raw recordings; native MLX Hadamard if the alternative is triggered. Reuse only code whose pinned license permits it, and retain notices.
3. **What might differ?** Panel-specific input anchors, exact same-module certification, net XOR cancellation, and an implementation that skips accumulator maintenance as well as downstream propagation. The combination must survive an equivalence review.
4. **Why might it matter?** Eager updates do work even when the thresholded output is provably unchanged. One shared certificate can potentially avoid that work for several neurons.
5. **What isolates it?** Same frozen binary model; optimized dense/eager methods; cumulative-flip certificate; exact-input memoization; positive-distance certified-hit analysis; matched-anchor diagnostic traces; and native measurements including all state costs.
6. **What invalidates novelty?** An earlier method with the same acceptance condition and state semantics, or a simple equivalent cache/interval-bound implementation. Relabel the work as replication or engineering in that event. A novel implementation can still be useful without a new method.

### 3.4 Reconciling apparently favorable external results

XNOR-Net's operation/memory claims are not universal end-to-end laptop speedups. Walsh papers often measure a particular layer or report quality tradeoffs. BTT explicitly discusses cases where fewer FLOPs lose at small hardware scales. DeltaCNN's GPU result and EvNet's operation counts answer different performance questions. LipCache's certificate is narrower than exact equivalence of the expensive main network. These are compatible findings once metric, model, hardware and guarantee are kept separate. [XNOR16] [WHT22] [BTT24] [DELTACNN22] [EV22] [LIPCACHE26]

The memo's WHTMix claim was located in primary indexed arXiv metadata: arXiv:2607.25234 reports 2.65x single-image latency improvement for a specific synthetic stereo setting. Full-text retrieval failed during this research. Treat the detailed conditions as unverified and do not use that number to predict M3 performance. [WHTMIX26]

## 4. Mechanism and implementation specification

### 4.1 Canonical numerical conventions

Use a C++20 CPU runtime with a slow independent NumPy/int64 oracle. Bit i lives in word i//64 at offset i%64; serialized words are little-endian uint64. Bit 1 is +1. Use the actual fan-in n, not padded storage width. Clear unused tail bits and mask the final word in every popcount. For n divisible by 64, use an all-ones final mask; never shift by 64 in C++.

Weights and binary inputs are packed. Thresholds and full sums use int32 in the initial scale range; perform update/slack intermediate arithmetic in int64 and check ranges before narrowing. Reject a model whose declared dimensions cannot satisfy these bounds. Zero active fan-in is a constant output computed from z=0. NaNs, infinite encoder values, malformed tensors and model-ID mismatches are explicit input errors, not silent zeros.

Vector popcounts still require reduction. Do not accumulate byte counts across enough NEON vectors to overflow uint8; widen or reduce at a proved safe interval. Validate vector and scalar kernels on adversarial all-one/all-zero words and maximum fan-in. Arm's `vcntq_u8` and XOR intrinsics are relevant primitives, not evidence of a complete fast kernel. [ARM]

### 4.2 Exact eager-update baselines

For changed coordinates C between old input a and current input x,

    z_new[j] = z_old[j] - 2 * sum_{i in C} w[j,i] * a[i].

Also implement the stronger packed changed-word identity. Let D=a XOR x and Q_j=a XOR W_j. Then

    z_new[j] = z_old[j]
               + 4 * sum_words popcount(D_word AND Q_j_word)
               - 2 * popcount(D).

Only nonzero words of D need weight reads; share popcount(D) across rows. This is exact integer arithmetic derived from the standard binary dot product. Changed-bit and changed-word sparsity differ: one flip in every word is bit-sparse but not word-sparse. Neither identity is claimed as new.

Dense fallback in an eager method must establish current sums and current previous-input state together. Output-only dense inference may early-terminate, but then those sums are unavailable: switching to eager execution requires a charged full-sum initialization. Never invent valid accumulator state from a thresholded output.

### 4.3 Panel certificate

For panel p, store a single anchor input a_p, exact anchor sums z_j^a, anchor bits y_j^a, and integer slacks

    slack_j = z_j^a - tau_j                  if y_j^a = 1,
              tau_j - 1 - z_j^a              if y_j^a = 0.

Define K_p=popcount(a_p XOR x), with tails masked, and s_p=min_j slack_j. The whole panel can reuse its anchor bits if

    2*K_p <= s_p.

**Proof.** Each changed signed input changes each dot-product term by either -2 or +2, hence the current sum lies in [z_j^a-2K_p, z_j^a+2K_p]. For an anchor output of one, the lower endpoint is at least tau_j. For an anchor output of zero, the upper endpoint is at most tau_j-1. Therefore all bits are unchanged. Applying the same argument layer by layer to current exact input bits proves exact network output bits. The core 2K bound is established prior art. [MARGIN21]

Do not replace this slack with absolute distance to the threshold under the same inclusive comparison: the negative side's tie rule differs. A certificate miss means only that the bound cannot establish stability; it does not mean an output changed.

On success, leave the anchor and sums unchanged. They are **anchor sums**, not current sums. Never expose them as current logits or use them in an eager update based on the immediately preceding input. On a miss, compute the full current panel sums, produce current bits, copy the current input into the panel anchor, and rebuild slacks atomically.

### 4.4 Minimal reference pseudocode

```python
def full_sum(weight_words, input_words, n, tail_mask):
    mismatches = 0
    for k in range(len(input_words)):
        mask = tail_mask if k == len(input_words) - 1 else UINT64_MAX
        mismatches += popcount((weight_words[k] ^ input_words[k]) & mask)
    return n - 2 * mismatches

def refresh(panel, x, model):
    new_sums = [full_sum(model.W[j], x, model.n, model.tail_mask)
                for j in panel.rows]
    new_bits = [z >= model.threshold[j]
                for j, z in zip(panel.rows, new_sums)]
    new_slacks = [z - model.threshold[j] if bit
                  else model.threshold[j] - 1 - z
                  for j, z, bit in zip(panel.rows, new_sums, new_bits)]
    # Commit these fields together; x must be copied, never aliased.
    panel.anchor_input = copy_words(x)
    panel.anchor_sums = new_sums
    panel.anchor_bits = new_bits
    panel.min_slack = min(new_slacks)
    panel.valid = True
    return new_bits

def panel_step(panel, x, model):
    if not panel.valid:
        return refresh(panel, x, model)
    k = 0
    for word in range(model.words):
        k += popcount((panel.anchor_input[word] ^ x[word]) & model.mask(word))
        if 2 * k > panel.min_slack:
            # A partial count can prove failure, never success.
            return refresh(panel, x, model)
    return panel.anchor_bits
```

An optimized refresh may use the exact anchor-relative changed-word identity instead of a full dot product, provided it completes the necessary difference scan and beats the dense refresh in calibration. Charge the extra scan. The minimum implementation uses full refresh and is already sufficient to test the mechanism.

For each layer, assemble current output bits from all panels before invoking the next layer. If the complete output is unchanged, every method may skip the next layer and reuse the readout. Immutable model data may be shared across streams; mutable anchors cannot.

### 4.5 Required alternative execution modes

**Dense output-only early stopping.** After processing a valid-bit prefix with h mismatches and r bits remaining, the full sum lies between n-2(h+r) and n-2h. Return one if the lower bound is at least tau; return zero if the upper bound is below tau. Measure whether checks help. Do not force this baseline to produce complete sums merely because the proposed method needs them.

**Cumulative-flip certificate.** Maintain a layer-level uint64 cumulative flip counter C_t and store each panel's counter value at refresh. The bound is 2*min(n,C_t-C_anchor). This method needs no per-panel anchor bit copy when refresh uses dense computation. Allow that cheaper storage and scanning strategy. Handle counter overflow by an explicit all-panel refresh/reset. It is a strong simpler baseline, not a deliberately slow imitation of XOR.

**Exact-input memoization.** Implement at least a two-entry full-layer cache for synthetic repeated-input tests and a memory-matched bounded cache for real traces. Hashes may locate entries, but full masked key equality is required before returning outputs. Charge key lookup, storage and copying. Select cache capacity on development data under the candidate's state-memory allowance; include the two-entry alternative even if it uses less memory.

**Matched-anchor diagnostic.** Autonomous policies refresh at different times. To isolate the tighter net-distance bound, replay both net and cumulative eligibility against the same externally specified, periodic anchor schedule chosen on development data. This diagnostic uses no test labels or future input to choose anchors and is not the deployable performance result. Report both autonomous runtimes separately.

### 4.6 Graph simplification, panel layout and scope

Compile mathematically constant threshold units: tau<=-n always produces one; tau>n always produces zero. Propagate their constant contributions into downstream thresholds. Verify the simplified graph against the unsimplified graph. Detect exact duplicate rows/functions, and share computations where valid. Do not call a unit constant merely because it stayed constant on training data. If merging repeated inputs produces multi-bit weights, retain that as a valid exact integer baseline and report its different representation.

Start with g in {1,2,4,8,16,32,64} for the selected hidden layer, plus the no-certificate method. These values cover row-wise checking through amortized panels; they are an initial calibration grid, not a scientific scale ceiling. Measure the relationship between cost and panel minimum slack. A fixed row permutation based on development-only margin summaries is permissible; apply the same permutation and corresponding readout-column permutation to every comparator. Include original row order as an ablation.

The primary frozen candidate selects one g and, if justified by development cost, a fixed subset of panels to certify. Other panels use optimized dense output-only computation. This avoids making dynamic mode switching a prerequisite. If a later online dispatcher switches between eager and anchor states, define separate validity flags and charge full initialization on a transition. An old anchor sum cannot become a current eager sum by changing a mode flag.

### 4.7 Public interfaces and state invariants

```cpp
struct StepContext {
    uint64_t stream_id;
    uint64_t sample_index;
    bool reset;                 // recording boundary, not class transition
};
struct StepOutput {
    Span<const uint64_t> bits;   // current exact hidden output
    Span<const float> logits;   // populated only by model wrapper
};
class ExactMethod {
public:
    virtual void prepare(const CanonicalModel&, const MethodConfig&) = 0;
    virtual void reset(uint64_t stream_id) = 0;
    virtual StepOutput step(Span<const uint64_t>, StepContext) = 0;
    virtual StateBytes checkpoint() const = 0;
    virtual void restore(StateBytes, const ModelDigest&) = 0;
    virtual MethodStats stats() const = 0; // untimed diagnostics
};
```

These are **proposed project interfaces**, not package APIs. All methods expose outputs, not full current sums. Store dimensions, tail mask, bit convention, thresholds, row permutation, schema version and model digest in `CanonicalModel`. Cache types explicitly distinguish `AnchorSums` from `CurrentSums`. Restoring a state with a different model, preprocessing digest, layout or quantile version must fail. Each mutable stream has its own feature buffers, cursors, caches and counter state.

## 5. Data, models, baselines and evaluation protocol

### 5.1 Data source and protected split

Use UCI dataset 341, DOI 10.24432/C54G7M, donated in 2015. Its official metadata documents 30 participants, six inertial channels at 50 Hz, six basic activities plus six postural transitions, raw recording files, an official 70/30 subject partition, and CC BY 4.0. The archive is listed as 75.9 MB. Its 10,929 listed instances refer to windowed records, not the number of raw ticks. Do not substitute the older window-only UCI HAR dataset 240. [HAR341]

Download the archive through the official dataset page. Record the resolved URL, retrieval date, byte size, SHA-256, archive manifest, README and license. Do not redistribute unrelated pretrained weights. No pretrained asset is necessary for the core path.

Required files include `RawData/acc_expXX_userYY.txt`, `RawData/gyro_expXX_userYY.txt`, `RawData/labels.txt`, `Train/subject_id_train.txt` and `Test/subject_id_test.txt`. Confirm paths from the archive manifest rather than assuming its top-level directory name. Join accelerometer and gyro by experiment/user ID; require equal lengths, finite values, valid interval bounds and monotone sample order. Preserve the documented sensor units; do not guess conversions.

Load `activity_labels.txt` and save a literal source-ID/name-to-class-index mapping, with the twelve ordered source IDs mapped to 0..11. Validate the source's expected label range rather than silently subtracting one from arbitrary values. Unknown labels, unresolved endpoint cases and conflicting overlapping annotations are ambiguous for supervised scoring/loss, while their raw samples remain in inference. Any conflicting overlap first triggers a parser/indexing check; retain its count and resolution in the manifest.

Extract the unique official training/test participant IDs. Require disjoint sets and the documented 21/9 counts; a discrepancy is a data-version gate. Sort the 21 training IDs, permute with NumPy PCG64 seed 20261029, assign the first six to development and the remaining fifteen to training. Save the resulting literal IDs in `splits.json` before model fitting. Keep all recordings of a participant together.

The nine test participants are protected from model/panel/backend selection and exploratory trace analysis. Initial integrity checks may inspect schema, byte counts and interval validity without computing performance summaries or choosing exclusions from test outcomes. Do not concatenate overlapping packaged windows or invent a stream by shuffling them.

**Label-index gate.** The public metadata describes sample-interval endpoints but did not settle their indexing convention in the inspected material. Read the downloaded README and, if necessary, author-supplied preprocessing. Record the supported zero/one-based and inclusive/exclusive interpretation. If this remains unresolved, use the intersection of labels under the plausible conventions for supervised scoring, mark boundary samples ambiguous, retain every tick for inference, and report both convention sensitivities. Freeze this rule before accessing protected outcomes. Never choose the convention that gives the best accuracy.

### 5.2 Causal fixed-coordinate features

Order raw channels as acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z. For each, form seven features in this exact channel-major order: latest, mean16, std16, mean64, std64, mean128, std128. This gives 42 features with at most 2.56 seconds of history. Every trailing window includes the current sample. No future sample, video annotation, activity boundary or subject-specific test calibration is available to the model.

Parse the six raw numeric text fields into IEEE FP64 once, save canonical FP64 raw arrays, and give identical bytes to every same-model method. Preserve FP64 feature values and cutpoints through bin assignment; do not fit quantiles on a rounded FP32 feature cache and then encode FP64 values at inference. Cast standardized features to FP32 only at the input boundary of a real-valued model. Use one shared feature implementation for all methods:

1. Maintain the last 128 raw observations in a ring buffer and FP64 sums/sums of squares for each window.
2. Define variance as `max(0, sumsq/L - (sum/L)**2)` with population convention `ddof=0`; compute square root in FP64.
3. After inserting the current sample, recompute sums when `(sample_index + 1) % 1024 == 0`, in oldest-to-newest order separately for each window. This deterministic schedule controls drift and is shared and timed for every method. Compare against direct FP64 windows in validation. Material errors or many negative unclamped variances require a numerically stable update revision before freeze; a tolerant arithmetic check never substitutes for identical encoded bits across same-model paths.
4. During the first 127 ticks, update feature state but emit `prediction_available=false`. Time these ticks. The first prediction is at sample index 127 under zero-based indexing.
5. Feed unlabeled intervals through every stateful component. Exclude only unlabeled/ambiguous ticks and the documented warm-up from supervised loss and quality metrics. Never reset at activity changes.

For each feature, fit empirical training-only quantiles at j/16 for j=1,...,15 using an explicitly pinned quantile method (`method="linear"` in the chosen NumPy release). Remove duplicate cutpoints within that feature. Encode one bit per cutpoint using `feature >= cutpoint`; this is equivalent to right-sided insertion in sorted cutpoints. Logical input width is therefore at most 630, with padding only for storage. Record cutpoints as exact FP64 bytes. Real-valued baselines receive training-standardized original features; also test bin indices and the identical thermometer inputs to distinguish representation effects.

Quantiles and normalization fit only training participants and only available, causal feature values. Document whether unlabeled training ticks are used to fit unsupervised quantiles; default is yes, because they are available without labels. They never enter supervised loss without a valid activity label.

### 5.3 Mandatory thermometer optimization

Let feature f have L_f cutpoints and current bin b_f in {0,...,L_f}. Its first-layer contribution can be tabulated exactly:

    T[j,f,b] = 2 * sum_{r < b} w[j,f,r] - sum_{r < L_f} w[j,f,r].
    z_j = sum_f T[j,f,b_f].

An eager first-layer update adds `T[j,f,new_bin]-T[j,f,old_bin]` only for changed features. Compare this against packed dense/changed-word evaluation and use the fastest development-selected implementation as the **shared first layer**. With 512 outputs, 42 features and 16 bins, int16 tables occupy 688,128 bytes before metadata; accumulate in int32. This is an estimate from the declared layout, not a measured allocation.

The Hamming distance between two thermometer inputs also equals `sum_f abs(bin_f - anchor_bin_f)`. Thus first-layer XOR certification has an equivalent bin-distance formulation. Primary novelty attribution must concern learned hidden binary inputs; it cannot rest on pretending thermometer bits are independent.

### 5.4 Reference network and training

Initial model: thermometer input -> binary hidden width 512 -> binary hidden width 512 -> FP32 linear 12-class readout. Each binary hidden block uses signed weights, a frozen-at-export affine normalization, and the threshold convention in Section 4. No dropout, input-dependent scale, residual FP activations, online BatchNorm or changing parameters are permitted inside a certified block. A separate compact causal model supplies the broader practical comparison.

Train continuous shadow weights using a documented straight-through estimator for **both weight quantization and hidden activation signs**: forward returns exactly +1 for v>=0 and -1 otherwise; backward returns upstream_gradient times 1[abs(v)<=1]. Implement a custom autograd function or a verified equivalent, not an ordinary `torch.sign` activation with zero gradient. Clip shadow weights to [-1,1] after optimizer steps. Initialize shadows uniformly on [-1,1] with the fixed seed, BN scale/bias at 1/0, and the FP32 readout weights with zero-mean normal standard deviation 1/sqrt(hidden_width), bias zero. Binary linear layers have no separate bias. Normalize their sums by sqrt(fan-in) before BatchNorm. The STE is a surrogate optimization rule, not the derivative of the discrete deployment graph. [BNN16]

Use PyTorch FP32 with MPS when the full graph and backward smoke test succeed; otherwise CPU with identical mathematical rules, without claiming bitwise-identical training across backends. Start with AdamW, learning rates {1e-4,3e-4,1e-3}, weight decays {0,1e-4}, and candidate widths {128,256,512}. Apply weight decay to shadow/readout weights, not BN parameters or biases. Calibrate batch size from memory, initially 256 if it fits. Shuffle eligible training examples using the recorded per-epoch RNG. Use the same declared sampling procedure and effective batch for architecture comparisons. Record BN epsilon/momentum, AdamW betas/epsilon, schedule and class weights explicitly in YAML; initial defaults are BN epsilon 1e-5/momentum 0.1, AdamW betas (0.9,0.999)/epsilon 1e-8, no class weighting, and constant LR for the pilot.

The supervised objective is mean cross-entropy over the twelve logits and valid class indices in the minibatch, with no class weighting by default. Empty-label minibatches are skipped and logged, not assigned a synthetic class. Any class-weight or sampling-balance change is chosen on training/development data and applied consistently to the corresponding comparison family.

Training examples are causal feature/label pairs from training participants. Initially select labeled ticks with zero-based raw index `t >= 127 and (t - 127) % 5 == 0` to reduce near-duplicate optimization work; inference and all evaluation remain at 50 Hz. Compare one development pilot with all eligible training ticks. If the quality/convergence or flip statistics materially change, use full ticks for every final method and freeze the choice. This is a disclosed optimization choice, not downsampling the evaluation workload.

Do not prescribe a blind fixed long run. Pilot learning curves at increasing epoch endpoints; choose a common final training horizon based on development convergence and remaining improvement uncertainty, then train all final configurations to that frozen horizon. Extend the LR grid or width to 1024 only if the pilot indicates underfitting or a boundary optimum; record the reason and give comparable tuning opportunity to real-valued baselines. Training loss divergence is an implementation/optimization diagnosis, not evidence against XOR reuse.

Use the initial seed 20261029 to select the training recipe by highest development participant-macro balanced accuracy; break exact ties by lower measured full dense inference time, then smaller parameter count. This selects architecture, optimizer settings, data stride and a common final training horizon, not one favorable final checkpoint. Freeze that recipe before training three final seeds (20261029, 20261030, 20261031). Export each seed's checkpoint at the frozen final optimizer step; use all three, never the fastest or most favorable seed or a test-selected epoch. Additional seeds require a development-only precision rationale and a frozen count before test evaluation. Seeds quantify training variability, not independent users.

The width grid is a practical starting point. A tiny model that already dominates ends the application case even if a larger model offers a component speedup. Conversely, synthetic larger-layer tests remain valid for identifying hardware crossover without pretending the larger shape is required by this task.

### 5.5 Export and equality

Freeze normalization statistics and all model parameters. For each neuron, enumerate its attainable integer sums {-n,-n+2,...,n} through the frozen reference affine/sign computation F(s), including the /sqrt(n) normalization, using its specified FP32 operation order. Verify monotonicity. If the scale is negative, set w'=-w and s'=-s, and fit the increasing function F(-s'); do not apply the unchanged negative-scale affine to s'. Validate `canonical(-s)==F(s)` for every original attainable s. If the result is constant, export a constant. Otherwise choose the smallest attainable sum producing one as the integer threshold. Check every attainable sum against the exported rule; do not rely on a fragile `ceil` around floating-point ties.

If the reference affine implementation cannot be made a monotone threshold with the documented operation order, the export fails. Investigate numerical folding; do not accept approximate equality. Once accepted, the canonical integer graph is the inference reference. Its quality is evaluated explicitly; any difference from training-mode or framework inference is reported.

Larq uses different documented bit/threshold conventions and NHWC/OHWI layouts. A dense layer can be expressed as a 1x1 VALID convolution with input shape [1,1,1,n], but the exported graph must actually use the native binary convolution kernel. Logical channels stay n even when physical storage is padded. Translate signs and threshold inequalities explicitly and verify every attainable scalar sum, random tensors and complete streams. For the packed-input C1 boundary, use a native single-operator packed-tensor harness where supported; a public float-input converter wrapper may add unpack/repack work and cannot be called evidence of an inferior packed kernel. Document and charge unavoidable conversion in the corresponding end-to-end comparison. Extract exact hidden bits and run the shared native FP32 head for bitwise-logit checks; alternate framework heads use separately declared floating tolerances. Matching outputs from a Python reference interpreter does not establish native optimized execution. [LARQOPS]

### 5.6 Baselines and comparison scope

| ID | Method | Role and required configuration |
|---|---|---|
| O | Independent NumPy/int64 direct sums | Correctness oracle only; never a speed baseline. |
| D | Native packed dense output-only layer | Scalar then NEON; row blocking; measured thread setting; tail masking; with/without beneficial exact early termination. |
| E-bit | Exact changed-coordinate signed updates | Current int32 sums, changed-bit enumeration; layout and memory charged. |
| E-word | Exact packed changed-word updates | Section 4.2 identity; nonzero-word list; identical packing and vector primitives. |
| E-hybrid | Best development-tuned exact dense/eager policy | Include the cost of producing valid sums at mode changes. Static per-layer choice is acceptable if online dispatch loses. |
| C-count | Cumulative-flip certificate | Counter implementation with its genuinely smaller state, not forced anchor scans. |
| M | Exact-input memoization | Full key verification; two-entry and memory-matched development-selected capacity. |
| LCE | Native Larq Compute Engine | Same graph if exact export succeeds; separately record native build, operator selection and conversion/runtime versions. |
| P | Proposed anchor-relative certificate | Development-selected panel size/layout, dense or validated exact refresh; all state costs included. |
| FP | Compact FP32/FP16 feature MLPs | Original 42 inputs, bin-index inputs, and matched thermometer input; widths starting {32,64,128,256,512}. Tune quality and measure Accelerate/framework/native deployment. |
| Q8 | Applicable INT8 classifier | Calibrate on training only; verify actual quantized execution and report quality. Core ML compressed weights alone do not imply INT8 arithmetic. |
| SIMPLE | Multinomial logistic regression and a small boosted-tree classifier | Strong low-cost feature baselines; tune regularization/depth on development subjects. No artificial 630-input requirement. |
| CAUSAL | Compact raw-signal causal convolutional model | Same 128-sample maximum history and 50 Hz predictions; ring-buffer reuse at every temporal convolution. |
| HOLD | Predict only every k ticks and hold outputs | Approximate practical alternative with k chosen on development data; report transition lag/quality. It is not an exact same-model baseline. |

For CAUSAL, use a 1x1 six-to-C channel stem, six stride-one residual blocks with **one** causal convolution of kernel size 3 per block and dilations {1,2,4,8,16,32}, then a pointwise 12-way readout. Each block computes `ReLU(h + conv(h))`; use no temporal/global normalization or pooling. The receptive field is 1+2*(1+2+4+8+16+32)=127 samples, within the 128-sample limit. A conventional two-convolution-per-block implementation would exceed it and is not this baseline. Tune C in {16,32,64}. Zero-pad the left boundary and reset every recording. Each block caches the last 2*d values of its own input; all six caches require 126*C floats before metadata. Compare streaming with a stateless reference using a documented floating-point tolerance and matching predictions; exact integer BNN comparisons retain zero tolerance. Do not compare against repeated computation of the entire sliding window.

Fit optional fixed per-channel standardization on training raw samples only and time it at inference. Training stride selects target timestamps, not sensor samples: each selected target t still receives the consecutive raw 50 Hz window t-127 through t, and the loss uses only the final timestamp's logits. Never take every fifth raw sample, which would change the history and task. Retain all causal intermediate observations during streaming, including ticks without a supervised target.

The fastest eligible exact baseline B* is selected using development measurements, with its configuration frozen. Include D, the best eager implementation, C-count and M regardless of which wins. LCE is mandatory to attempt because Apple ARM evidence exists. If no faithful optimized local LCE build/export is possible after diagnosing concrete incompatibilities, record the blocker and narrow the claim to the validated local baselines; do not call the result superior to the strongest binary engine. A slow port is not a faithful reproduction of published runtime performance.

Practical baselines also require native timing paths. Export logistic-regression coefficients/intercepts to an affine/argmax implementation with normalization included. A tree model needs a supported native predictor/export whose predictions are verified, or it is a quality-only reference with native latency explicitly unresolved. Do not time `sklearn.predict` once per tick against a fused C++ candidate. The causal model uses native ring buffers or a compiled framework path with comparable host-call boundaries. Match C3's scored predictions from tick 127 onward for all methods, while separately reporting each architecture's earliest valid prediction and startup cost.

DeltaCNN's original CUDA runtime, published large-model BTT results, ReActNet/ImageNet, BitNet language models and custom hardware processors are **reference results**, not head-to-head entries for this sensor experiment. Their task, architecture or hardware differs. Excluding a language model from sensor classification does not weaken the sensor comparison. A published CUDA result must not be combined numerically with local M3 timings. [DELTACNN22] [BTT24] [BITNET]

All generally useful optimizations are shared: input-equality tests, downstream/readout reuse when inputs are known unchanged, fixed graph simplification, shared causal preprocessing, weight preparation outside steady state, and identical host-call boundaries. Method-specific state and its construction remain charged to that method.

### 5.7 Primary metric, populations and practical guardrails

**Primary claim C1:** for the frozen model family and unseen UCI341 participants, P reduces native execution time of the selected second binary layer under a one-native-CPU-thread budget relative to a development-selected B*, while producing identical output bits. One-thread resource matching is the primary mechanistic comparison; separately report each method's best development-tuned thread configuration so this is not mistaken for the fastest possible laptop deployment.

The primary workload consists of every hidden input generated after the 127-tick warm-up of every eligible recording, in original order, including inputs from unlabeled intervals and repeated identical inputs. Materialize these inputs with the frozen common first layer in a correctness/data-generation pass. The layer benchmark starts with a packed input available in memory and ends with its packed output consumed by a common checksum sink. It includes change detection, dispatch, cache lookup, certificates, refreshes, output assembly and all state updates. It excludes the separately measured upstream feature/first-layer computation. It is a component experiment, not whole-model latency.

Choose B* by minimizing mean log layer service time over all six development participants and all three final checkpoints under the one-thread budget. Choose P by the same aggregate. Freeze one baseline method ID and one candidate panel size/layout rule across seeds. Model-specific weights, permutations and static certified-panel masks may differ only through the recorded development-only procedure; no policy depends on the test participant. The primary test is against that frozen B*. Report all other frozen baselines too: if another baseline wins on test, the result cannot be described as best among all baselines.

**Whole-stream claim C2:** use the same frozen one-thread comparison in an integrated native pipeline for the formal hierarchical test. The timer starts when each raw six-channel sample is available in memory and ends when logits or `prediction_available=false` are returned. Include feature updates, quantiles, packing, the shared optimized first layer, the target layer, common readout, all resets and state costs. Include every raw tick, including startup and unlabeled samples. Exclude file parsing/downloads and intentional waiting for the next 20 ms acquisition interval. Report load/parse/startup separately. Also report the complete development-selected best-thread deployment frontier; a single-thread advantage does not establish fastest local deployment. A packed-trace microbenchmark cannot establish C2.

**Application claim C3:** compare practical classifiers at identical prediction cadence and no more than the permitted history. Show quality–time–memory tradeoffs, with uncertainty, rather than imposing a post-hoc accuracy tolerance. Different representations and model widths are allowed here because they are realistic alternatives; the controlled same-model comparison is C1/C2. C3 is descriptive unless a formal comparison is preregistered. Formal strict dominance requires simultaneous uncertainty support for no worse recognition quality and no greater computation cost, at least one strict improvement, and disclosed memory/transition regressions. A justified acceptable quality-loss margin must be specified before confirmation; failure to detect a loss is not noninferiority.

| Metric | Definition and units | Interpretation |
|---|---|---|
| Primary layer service time | Total native replay seconds / consumed hidden inputs; participant-equal ratio P/B* | Component acceleration, including repeated-input handling. |
| Whole-stream service time | Replay seconds / all raw ticks, reported in microseconds/tick | Actual computational duty of this pipeline; not acquisition/display latency. |
| Absolute saving | Mean across participants of baseline minus candidate microseconds per tick/input | Prevents a large ratio on trivial work from sounding transformative. |
| Exactness | Number of differing output bits; logits bitwise equal under the common head | Must be zero. Floating tolerance is not a substitute for exact binary state. |
| Recognition quality | Participant-macro balanced accuracy; per-class recall and macro-F1 | Compute each participant's balanced accuracy over classes present, report the coverage; never count windows as independent participants. |
| Transition response | Recall by transition class; interval miss rate; time to first three consecutive correct transition predictions | Offline scoring only. Treat no detection before interval end as censored/missed, not zero delay. |
| Tail and startup | Per-tick p50/p95/p99 in a separate equally instrumented run; first-prediction/reset cost | Mean speedup can hide bursts or refresh penalties. |
| Memory | Serialized weights, auxiliary tables, per-stream state, process peak RSS and accelerator allocation | Charge duplicate layouts, anchors, tables and conversion artifacts. |
| Mechanism counters | Positive-K hit fraction, K=0 hits, actual unchanged panels, changed words, refreshes, bytes read/written estimates | Diagnostic evidence, gathered separately from clean timing. |
| Training and setup | Wall time, peak memory, optimizer steps, data examples, export/packing/table construction time | Amortize against expected inference count only with explicit assumptions. |
| Energy, optional | Measured counter-domain joules / work, with idle estimate and uncertainty | Not inferred from FLOPs; see Section 6.5. |

At 50 Hz, a service time of t seconds implies a computational duty fraction approximately 50*t for a serial replay model. This is a calculation, not a power measurement. Run a separate paced 50 Hz experiment for deadline misses and scheduler behavior; do not conflate its sleep-dominated wall time with service-time acceleration.

### 5.8 Fixed confirmatory design and statistics

1. Complete all exploratory work on training/development participants. Freeze code, dependencies, data manifests, candidate, B*, full comparator list, model seeds, metrics, exclusions, timing schedule and analysis script before reading protected performance results.
2. Use all nine official test participants. For each fixed checkpoint and each method, replay all their recordings with resets at recording boundaries. Three model seeds and repeated timings are crossed robustness/measurement factors, not extra participants.
3. Run paired timing blocks sequentially, with balanced randomized method order. Warm executable code, allocator behavior and immutable weights on development data outside timing; preallocate buffers consistently. Inside each timed replay, call reset at each recording boundary, then process the recording chronologically. Charge cache invalidation and per-recording initialization. Persistent model preparation, packing and first-use compilation are separate setup measurements. Do not preload test state into caches.
4. Use two separately initiated hardware sessions with the same locked setup. They check measurement reproducibility, not independent data generalization. Freeze the number of blocks per session from the pilot.
5. Retain failed/interrupted blocks and reasons. Invalidate a complete paired block only for predeclared objective events such as an interrupted process, known competing job, missing input or corrupted state. A slow time alone is not an exclusion rule.

For method m, participant s and fixed model seed k, define tau_m,s,k as the mean over the two sessions and R blocks of `(sum_recordings elapsed_seconds)/(sum_recordings relevant_input_count)`. Sum each participant's recording times and counts before division; do not average recording-level ratios equally. Reset costs belong in elapsed time. If clock amortization repeats complete recordings, preserve their proportions and normalize by actual complete replay count. Form one participant effect

    ell_s = mean_k log(tau_P,s,k / tau_B*,s,k),    s=1,...,9.
    latency_ratio = exp(mean_s ell_s).

The primary aggregate gives each participant equal weight and each fixed model equal weight. Also report each checkpoint separately, all nine participant effects, the absolute microsecond difference, and the tick-weighted total as secondary descriptions. Do not average instantaneous speedup ratios over frames.

Use a paired Student interval on the nine log effects:

    mean(ell) +/- t_(0.975,8) * sd(ell)/sqrt(9),

then exponentiate. State the approximate normality/independent-participant assumptions and the limited ability to assess them with nine participants. Include leave-one-participant-out estimates and a sign-based directional sensitivity check; those answer different questions and must not replace the primary analysis after seeing results. A participant bootstrap can describe sensitivity for task metrics, with its small-cluster limitations stated. Do not bootstrap individual frames.

The interval is conditional on the chosen model checkpoints and laptop configuration. It does not represent the variation over all possible trained networks, devices or real-world users. The dataset's laboratory setting, volunteer population and waist placement restrict external validity.

Report each session's aggregate ratio as well as the pooled interval. A favorable pooled interval establishes an average effect over the measured sessions. Call it a reproducible improvement only if both session aggregates favor P and no unresolved A/A offset, order effect or session instability is comparable to the saving. A directional reversal requires an explicit reproducibility qualification; pooling must not hide it. Neither session needs to be separately significant, and sessions do not multiply the independent participant count.

**Pilot-driven precision.** On development data, estimate within-participant timing variance and between-participant log-effect variance separately. Choose block count R so estimated measurement variance contributes at most a predeclared small fraction, initially 10%, of between-participant variance. If the latter is nearly zero, choose a finite-data precision target using the A/A systematic-error floor. Record the resulting R and total work before test access. The 10% criterion controls measurement quality; it is not a required improvement size.

With nine participants, a rough 80%-power detectable log effect is `(2.306+0.842)*sigma/3`, about 1.05 times the between-participant log-effect standard deviation. Use a range reflecting pilot uncertainty, not a promised effect size. Timing repeats cannot remove participant heterogeneity. If small effects are not resolvable, report the attainable resolution and an inconclusive outcome rather than repeatedly sampling until significance.

One predeclared P-versus-B* contrast is primary. Claims against additional individual baselines or regularizers use a predeclared Holm family of tests, or simultaneous Bonferroni intervals, with the family written in the freeze file. A claim to beat the entire baseline set requires support against every relevant member and no omitted faster valid method. Exploratory slices and settings remain labeled exploratory.

C1 can succeed without C2. A formal C2 superiority claim uses the same locked estimator and a predeclared hierarchical test after C1, or is reported descriptively if C1 fails. Practical C3 superiority additionally requires a favorable quality/cost frontier; exactness of the BNN does not establish noninferiority to a different model.

If protected outcomes trigger a method, population, baseline or analysis change, record an amendment and label subsequent use of those outcomes exploratory. Another random split of these same 30 participants, or the overlapping original UCI HAR dataset, is not independent confirmation. A second hardware session also does not create new users. A new confirmatory generalization claim requires genuinely unexposed subjects/data with compatible labels and licensing.

### 5.9 Harness validation

Before protected timing, pass all of the following:

- **A/A:** the same method in separate state buffers under randomized method labels does not show a systematic advantage beyond its measured noise floor.
- **Known difference:** add one intentionally redundant computation whose result reaches a sink; the harness detects its cost. Confirm the compiler did not eliminate it.
- **Clock amortization:** measure whole chronological replays with `std::chrono::steady_clock`; increase work per timed block until timestamp overhead is negligible relative to the selected precision target. Divide by actual work performed.
- **No hidden asynchronous work:** CPU completion must occur inside the timer. MPS/MLX/Core ML comparisons require the appropriate completion/materialization boundary.
- **Equivalent work:** identical input files, logical outputs, cold-reset counts and valid checksums. Do not compare native fused loops against per-layer Python dispatch.
- **Instrumentation separation:** detailed panel/bit traces come from an untimed diagnostic pass; a minimal-counter or counter-free build supplies latency. Per-tick timestamping for tails is a separate run and applied equally.
- **Replay integrity:** baseline/candidate output equality is checked on every tick in a correctness pass; benchmark sinks retain output dependence without timing the slow oracle.

## 6. M3 Max feasibility and resource adaptation

### 6.1 Discovery before installation or large work

Run these existing operating-system commands individually and save their output in `environment/`. They do not install anything:

```text
uname -m
sw_vers
sysctl -n hw.memsize
sysctl -n machdep.cpu.brand_string
sysctl -n hw.physicalcpu
df -k .
vm_stat
sysctl vm.swapusage
xcode-select -p
xcrun --find clang++
clang++ --version
python3 -c "import platform; print(platform.machine(), platform.python_version())"
```

Require native arm64 for target measurements. Record actual CPU/GPU variant, available memory/storage, power mode, power connection, compiler/SDK and backend versions. Do not collect serial numbers or unrelated identifiers. M3 Max variants have different resources; this plan does not assume the maximum memory configuration. [APPLEHW]

If Command Line Tools or a package manager is absent, use the normal supported installation path and document it. Do not silently upgrade macOS or replace the user's global Python. Project environments and build directories are isolated. CPU correctness and native inference do not require a remote GPU or service.

### 6.2 Selected software stack and compatibility gates

| Purpose | Selected path | Verified basis and mandatory local check |
|---|---|---|
| Runtime | C++20, Apple Clang, scalar reference then NEON | Native Arm XOR/popcount primitives are documented. Compile, inspect hot loops and test full kernels locally. [ARM] |
| Training | PyTorch FP32/MPS; CPU faithful fallback | MPS availability is only a first check. Test the full forward/backward/optimizer graph with silent CPU fallback disabled. FP64 reference stays on CPU. [TORCHMPS] |
| Data/oracle/statistics | Python, NumPy, SciPy, scikit-learn | Pin compatible releases after a small end-to-end environment test; no guessed latest version is prescribed. |
| Packed baseline | Larq Compute Engine native arm64 | Historical official M1 results make it credible; current M3 build, converter/runtime match and actual binary operator selection remain gates. [LARQ] [LARQOPS] |
| Dense/quantized deployment | Accelerate BLAS, native framework/XNNPACK, Core ML where conversion succeeds | XNNPACK documents macOS arm64; Core ML backend/quantization behavior must be measured. [ACCELERATE] [XNNPACK] [COREML] |
| Conditional Walsh path | MLX native `mlx.core.hadamard_transform` | Documented default orthonormal scaling and shape restrictions; verify forward, gradients and actual device execution first. [MLXH] |

Select the newest non-prerelease package/tag available at execution time that supports the discovered OS/Python and relevant converter constraints; record the selection date. If it fails a documented compatibility gate, try the newest supported prior release, recording the actual failure rather than selecting an older slower baseline for convenience. Build a tiny representative graph, run forward/backward/export/native inference, and then freeze exact versions and hashes. If training and Larq conversion need incompatible dependencies, use separate environments and a versioned neutral model export. Pin repository commits and submodules; a moving `main` branch is not a reproducibility specification. A newly released implementation that changes the closest baseline or novelty assessment triggers a targeted evidence refresh before freeze.

Larq code is Apache-2.0 and EvNet's inspected PyTorch reference is MIT; DeltaCNN's code is CC BY-NC 4.0 and CUDA-specific. Avoid copying its code into a differently licensed runtime without respecting those terms. MLX is MIT; model weights have their own terms even when an inference library is open-source. No paid model API, remote compute or specialized sensor hardware is required. [LARQ] [EV22] [DELTACNN22] [MLXH]

**Precision and backend caveats.** PyTorch MPS lacks GPU FP64 support. Silent unsupported-operation fallback can invalidate timing assumptions. MLX is lazy, so output evaluation and synchronization are necessary; its documented FP32 matmul policy can use reduced internal precision. For a controlled optional Walsh test, record `MLX_ENABLE_TF32=0` and any convolution policy, then separately report production defaults. Native MLX Hadamard documentation currently lists sizes n=m*2^k with m in {1,12,20,28}, and limits on the power-of-two factor of 8192 for FP32 or 16384 for FP16/BF16. Start with pure powers of two; do not transfer Sylvester self-inverse assumptions to every generalized family. [TORCHMPS] [MLXH] [MLXPREC]

Core ML documentation warns that quantization can incur runtime decompression costs on some compute units and highlights newer hardware for activation-quantization throughput. Therefore an M3 INT8 speedup is a hypothesis. Benchmark FP32/FP16 and valid quantized variants, report selected compute units, and separate weight compression from integer execution. [COREML]

### 6.3 Memory accounting

For one binary layer with P=ceil(m/g) panels and S=ceil(n/64):

    row weights             = 8*m*S bytes
    optional column layout  = 8*n*ceil(m/64) bytes
    panel anchor inputs     = 8*P*S bytes
    anchor sums             = 4*m bytes
    thresholds              = 4*m bytes
    cached outputs          = 8*ceil(m/64) bytes, plus alignment
    panel slacks/flags       = implementation-dependent, explicitly measured
    changed indices         <= 4*n bytes if int32 indices are used

Add temporary vectors, first-layer LUTs, FP32 readout, input/output buffers, allocation overhead and executable/runtime memory. Multiply mutable state by simultaneous streams. Shared weights and immutable tables need not be duplicated. Count both packed and decoded copies if a baseline retains them.

For a 512-by-512 target layer with g=16, row weights are 32 KiB and anchor inputs 2 KiB; sums and thresholds add 2 KiB each. For a 4096-by-4096 diagnostic layer with g=32, row weights are 2 MiB and anchors 64 KiB. These are layout calculations, not process-memory measurements. Such small matrices may fit in cache, reducing the potential benefit of avoiding weight reads.

The initial 630->512->512->12 model has approximately 591,000 weight parameters before normalization terms. FP32 Adam-family training needs roughly 16 bytes per parameter for parameter/gradient/two moments, plus about 4 bytes per parameter for a materialized signed copy where used. Thus parameter-state storage is roughly 12 MB, while activations, workspaces and framework allocation can dominate. Packed inference memory is not a training-memory estimate.

For N raw ticks, canonical six-channel FP64 raw arrays cost 48N bytes, 42 FP64 features cost 336N bytes, and a packed 630-bit input costs about 80N bytes before metadata. FP32 copies used by real-valued models add their own storage; they do not replace the canonical pre-quantizer values. Do not materialize all inputs as N-by-630 FP32 tensors by default. Use per-recording files/memmaps and expand only training batches. Save traces only when needed; aggregate histograms or deterministic samples for heavy per-neuron diagnostics.

### 6.4 Calibration and hardware-dependent rules

Let M_available be an OS-reported available-memory estimate, corroborated by pressure/swap observations. Begin with a working allocation budget of 60% of that amount, preserving headroom for the OS and measurement tools. This is a conservative operational reserve, not a scientific maximum. Measure actual peak memory at two small batch sizes, fit fixed overhead plus per-example cost, and verify the proposed batch with complete training steps. Increase only if pressure remains normal and swap does not grow.

| Stage | Resource estimate or calibration procedure | Adaptation without changing the claim |
|---|---|---|
| Algebra/correctness | Exhaustive fan-in <=5; randomized widths through 4096 as needed; negligible data | CPU only; reduce test batch memory, not the set of boundary cases. |
| Data preparation | Listed archive 75.9 MB; extraction and feature sizes measured from manifest/N | Stream files; use memmaps; retain raw archive/checksum. Do not discard difficult intervals for space. |
| Kernel calibration | Cost proportional to shapes, trace length and repetitions | Start widths 64..4096 powers of two and non-word tails; extend only where observed crossover or task shape justifies it. |
| Training | `steps_per_epoch * measured_step_time * frozen_epochs`, plus validation/checkpoint | Lower batch before width; if BN behavior changes, restart that configuration and record the change. |
| Baseline builds | Disk/toolchain cost measured in isolated build trees; TensorFlow/LCE may dominate installation | Separate converter environment; stop unrelated builds once no faithful support route remains. |
| Confirmatory runs | Sum over fixed methods, nine participants, three checkpoints, blocks and sessions | Pilot fixes work count. If noise resolution is infeasible, report the limit rather than add test runs adaptively. |
| Optional energy | Long equal-work blocks or paced traces; local counter sensitivity determines duration | Energy remains unmeasured if counters/permissions/sensitivity are inadequate. |

Do not estimate hours from peak M3 FLOPs. Measure representative warm and sustained steps and extrapolate with an interval spanning the observed range. Record time separately for training, validation, export, first-use compilation, and steady inference. Before a longer run, save the forecast and the evidence that additional computation can resolve a specific uncertainty.

If memory is substantially constrained, the faithful core remains a CPU binary runtime and the small classifier. MPS absence may slow training but does not change the exact mechanism. If the minimum task-valid model cannot be trained locally, record a local-resource limitation; synthetic layer tests remain component diagnostics, not a substituted application validation.

If an optimized critical baseline cannot execute faithfully, continue unaffected correctness and cost work but narrow comparative claims. Do not use an emulated slow kernel to imply dominance over a published optimized method. If native Hadamard is unavailable in the conditional path, a Python butterfly is a correctness reference, not evidence for efficient Mac execution.

### 6.5 Timing and energy details

Use optimized release builds, initially `-O3` without unsafe floating-point reassociation; use `-ffp-contract=off` for the common feature extractor/readout whose operation order must match. Generate canonical training/evaluation features through that same implementation; a Python reference must use explicitly ordered FP64 operations rather than an unspecified parallel reduction. Practical alternate floating models may use their normal optimized arithmetic, with quality evaluated separately. Record thread count; compare one-thread mechanism performance and each method's best development-tuned local thread setting separately. Do not claim pinned performance cores unless an actually supported affinity mechanism is used and verified. Avoid concurrent benchmark methods.

Record charging state, power mode, background load, temperature/thermal state where available, and swap changes. Interleave methods to reduce systematic drift. Report cold loading/packing/compilation separately and include per-recording reset/initialization in the stream workload. Measure enough complete replays to overcome clock resolution; do not repeatedly time an already evaluated lazy tensor.

For MPS, synchronize submitted work before/after timing as appropriate. For MLX, create fresh computation, evaluate outputs, and synchronize the relevant stream; merely timing graph construction is invalid. Conversion from Python/NumPy, host/device copies and output materialization are included for an end-to-end deployment comparison. Device-resident kernel timings must be labeled separately. [TORCHMPS] [MLXH]

Energy is secondary. Discover supported `powermetrics` counters and permissions from the local OS documentation. If usable, integrate reported power over equal-work blocks, report the covered component domains, gross and idle-adjusted estimates, sampling interval and uncertainty. These are not automatically whole-laptop measurements. Compare accelerated replay and paced 50 Hz use separately; most energy at low duty may be unrelated idle/system cost. If privileged counters are unavailable or the difference is below noise, say energy was not established and continue the latency program. Never derive joules from operation counts or treat Activity Monitor's Energy Impact as joules. [ENERGY]

### 6.6 Checkpoints and interruption recovery

Save training checkpoints atomically after complete optimizer steps: weights, optimizer/scheduler, BN state, random generators, sampler position, config/source/data digests and completed-step count. An OOM retry must not silently produce a new effective training procedure. Reducing a microbatch changes BatchNorm statistics even if gradient accumulation preserves the nominal batch; record an amendment/restart when required.

For timed evaluation, prefer restarting the affected recording and paired block after an interruption. To resume mid-recording for correctness, serialize exact bytes of the raw ring, cursor, rolling FP64 sums/sumsquares, next periodic-recompute position, quantiles/version, current bits, panel anchor inputs/sums/bits/slacks, eager states/counters, model/layout hashes and next sample index. Rebuilding sums from the last window can change rounding and a quantile bit. Require resumed and uninterrupted outputs and future route decisions to match exactly. Do not combine timings across changed binaries, dependencies or configs.

## 7. Risk-ordered work packages and decision gates

### 7.1 Repository and command conventions

Use a local Git repository. All names below are proposed project files/commands the coding agent must implement; they are not pre-existing software interfaces.

```text
xor_reuse/
  README.md
  pyproject.toml
  CMakeLists.txt
  locks/                         # training, converter, analysis environments
  configs/                       # immutable YAML snapshots for each run
  environment/                   # hardware, OS, compiler, package manifests
  src/xor_reuse/
    cli.py
    data_uci341.py
    features.py
    train.py
    export.py
    oracle.py
    protocol.py
    analyze.py
  cpp/
    model_format.hpp
    dense.cpp
    delta.cpp
    certificate.cpp
    memo.cpp
    feature_runtime.cpp
    benchmark.cpp
    lce_adapter.cpp
  tests/
    test_algebra.py
    test_export.py
    test_features.py
    test_stream_state.py
    native_correctness.cpp
  data/{raw,derived,manifests}/
  artifacts/{checkpoints,exports,traces}/
  runs/<run_id>/{config,raw,summary}/
  protocol/{confirmatory.json,amendments.jsonl}/
  decisions/{journal.jsonl,status.md}/
  reports/{figures,final.md}/
```

Data directories and large artifacts are excluded from normal Git object storage but have checksummed manifests. Keep a reproducible asset retrieval command and local resumable copies. Commit code/config/protocol changes; preserve failed-run records. Each command accepts a config path and explicit output directory and refuses to overwrite an existing completed run.

Proposed command family:

```text
python -m xor_reuse.cli discover --out environment
python -m xor_reuse.cli prepare-data --config configs/data.yaml
python -m xor_reuse.cli check --suite algebra
python -m xor_reuse.cli train --config configs/train.yaml
python -m xor_reuse.cli export --config configs/export.yaml
python -m xor_reuse.cli calibrate --config configs/pilot.yaml
python -m xor_reuse.cli freeze --config configs/confirm.yaml
python -m xor_reuse.cli confirm --protocol protocol/confirmatory.json
python -m xor_reuse.cli analyze --protocol protocol/confirmatory.json
```

Implement each CLI only as its work package requires. Do not build a general experiment platform before testing the indispensable mechanism.

### WP00 — Evidence, hardware and asset gates

- **Objective/risk:** establish actual local capability and lawful, reproducible data access; avoid unsupported baseline assumptions.
- **Dependencies/entry:** this plan and a fresh target laptop; no model training required.
- **Actions/files:** implement `discover`; save OS/architecture/memory/compiler records; retrieve the UCI manifest/README/license; resolve label indexing; create isolated package environments; smoke-test C++ scalar/NEON, FP32 training/backward and native LCE conversion/inference. Inspect current release docs and lock successful versions. Record any incompatible operator/tag in `environment/compatibility.json`.
- **Tests/outputs:** native arm64 report; data checksum; license record; tiny forward/backward success; LCE logical-channel/tie adapter test; literal subject split in `splits.json` without opening test outcomes.
- **Resources:** small graph and official 75.9 MB data; measure installation disk needs before downloading optional stacks. No large pretrained model.
- **Acceptance:** data is usable and core C++/CPU training path works. A failed accelerator path may use faithful CPU training. Missing faithful LCE support narrows claims and is recorded before benchmarking.
- **Decision:** unsupported data/license blocks that application path; seek a documented equivalent raw stream before any confirmatory change. Missing core native correctness blocks downstream work. Proceed to WP01 when these prerequisites are established.

### WP01 — Algebra, invariants and export specification

- **Objective/risk:** disprove a faulty certificate or state model before optimization.
- **Dependencies/entry:** core language environment from WP00; no training/data outcomes needed.
- **Actions/files:** implement `oracle.py`, scalar dense/delta/certificate functions and `test_algebra.py`; enumerate all weights, anchors, current inputs for n=1..5 and all thresholds from -n-1 through n+1; check both eager identities and the asymmetric slack. Test affine-XOR inability to implement AND and optional small Walsh identities to preserve algebraic distinctions.
- **Tests/outputs:** zero disagreements; explicit counts; tests for zero/full changes, repeated flip/reflip, odd fan-in, tails, thresholds, aliasing and int overflow; `algebra_results.json`.
- **Resources:** tiny CPU-only exhaustive cases and seeded random larger tensors.
- **Acceptance:** every claimed exact output equals direct integer evaluation. Include a deliberately wrong absolute-margin implementation and verify the tests catch it.
- **Decision:** any counterexample stops all timing claims until corrected. If the theorem is invalid under intended model operations, remove those operations or abandon this mechanism; do not replace exactness with tolerance.

### WP02 — Correct native kernels and sensitive harness

- **Objective/risk:** establish strong local kernels and trustworthy measurement before a trained-network result.
- **Dependencies/entry:** WP01 passed; compatible compiler.
- **Actions/files:** implement `dense.cpp`, `delta.cpp`, `memo.cpp`, `benchmark.cpp`; add NEON after scalar tests; dense early stopping; changed-bit/word modes; two-entry and bounded exact-key memoization; C-count. Compile optimized and diagnostic builds. Measure shapes through actual expected model sizes and a justified larger crossover range.
- **Tests/outputs:** scalar/NEON equality; no byte-popcount overflow; A/A and known-difference controls; cost curves in ns/input and bytes/state; `kernel_calibration.json` with build/hash/thread/dtype/layout fields. Check optimized assembly or profiling to confirm vectorization/hot path.
- **Resources:** synthetic packed inputs, no trained model required; choose repetitions from clock/noise calibration.
- **Acceptance:** equivalent computations agree; harness resolves known added work; dense and sparse baselines each use their best justified local layout, with preparation/memory recorded.
- **Decision:** a measurement failure is not evidence against the hypothesis. Fix it. If certificate scan cost alone exceeds even eliminating all target work in every reachable shape, record no headroom and do only the cheaper real-trace diagnostic before further engineering.

### WP03 — Causal data pipeline and practical baseline training

- **Objective/risk:** obtain a useful task representation without leakage or artificial temporal redundancy.
- **Dependencies/entry:** WP00 data gate and WP01 oracle; no protected outcomes.
- **Actions/files:** implement streaming feature code in Python and C++; unique quantile encoding; first-layer LUT reference; model training and small practical baselines from Section 5.6. Verify the official split; record class/interval coverage on train/development only. Fit features/cutpoints only on training subjects. Train pilot width/LR settings, diagnose convergence and choose the final training schedule.
- **Tests/outputs:** causal-prefix invariance; training-only preprocessing hashes; Python/C++ feature parity under fixed operation order; label-index report; causal CNN cached/stateless equality; development learning curves, quality, memory/time forecasts and `model_selection.json`.
- **Resources:** raw/feature size from N; pilot steps calibrate training time. Memmap or batch expansion if memory constrained.
- **Acceptance:** finite, stable training; classifiers improve over trivial majority/hold controls on development; no subject/recording overlap or boundary resets. Report if the BNN is practically dominated.
- **Decision:** unusable BNN quality blocks an application claim, not the algebraic component question. Repair demonstrated training/export issues first. Do not enlarge a model merely to manufacture a timing target. Continue to WP04 with a frozen, interpretable model choice or document a narrowed diagnostic-only outcome.

### WP04 — Export, simplification and faithful baseline integration

- **Objective/risk:** prevent output changes, padding errors and weak baseline implementations from becoming claimed gains.
- **Dependencies/entry:** WP03 trained checkpoint; WP02 validated kernels.
- **Actions/files:** implement `export.py` and `model_format.hpp`; enumerate attainable sums; compile constants; identify exact duplicates; canonical digest/version. Implement and benchmark first-layer table/bin updates and packed alternatives. Integrate native LCE 1x1 conv adapter, inspect actual exported operators, and validate signs/thresholds/tails.
- **Tests/outputs:** all-sum export truth tables; full train/development stream equality; pre/post-simplification parity; `export_manifest.json`, canonical binary model, native baseline report, table memory figures.
- **Resources:** model is small; conversion environment may be larger than weights. Charge tables/duplicate layouts.
- **Acceptance:** zero bit mismatches and common-head equality; baseline methods perform the same mathematical function. Native LCE status is known and explained.
- **Decision:** model mismatch invalidates timing. A baseline build failure allows narrower local comparisons only, not a superiority claim against that engine. Proceed to WP05 when the executable reference is stable.

### WP05 — Minimum certificate and real-trace headroom pilot

- **Objective/risk:** determine whether certifiable stability pays for its own cost.
- **Dependencies/entry:** WP04 exact export and baseline harness; development data only.
- **Actions/files:** implement `certificate.cpp` exactly as Section 4; target the second hidden layer. Sweep initial panel sizes/layouts; gather net/cumulative distances, slacks, actual unchanged output fraction, positive-K hits, repeated-key frequency and state traffic. Compare native P, D, E-hybrid, C-count, M and eligible LCE. Measure first-layer/head/shared feature costs and Amdahl headroom. Add selective fixed panels or exact sparse refresh only when measured costs identify the reason to do so.
- **Tests/outputs:** untimed per-tick equality; component and integrated development timings; crossover plots; `headroom.json`; audit that generic readout reuse and thermometer optimization are shared.
- **Resources:** development trace size measured; counters sampled/aggregated for large widths; no protected analysis.
- **Acceptance:** a reproducible favorable regime exists in real learned traces or a specific unresolved measurement issue justifies a targeted revision. Report the uncertainty of small differences.
- **Decision:** if eager/table/memoization explains all gains, report that result. If high actual stability but poor certified stability dominates, revise grouping/bounds only with a concrete proposed improvement and repeat the novelty check. If all certifiable workload is trivial/static-removable, stop the usefulness claim. Do not run expensive training regularizers yet without a measured opportunity.

### WP06 — Freeze models, configurations and confirmation sensitivity

- **Objective/risk:** prevent adaptive selection and underpowered conclusions.
- **Dependencies/entry:** WP05 interpretable pilot; harness and all required baseline checks passed.
- **Actions/files:** lock model family/hyperparameters and final seed count; train final seeds; choose P and B* with development-only rules; set repetition counts/sessions using measured noise; implement analysis and mock it on pilot outputs; freeze `protocol/confirmatory.json` with all hashes, metrics, comparators, multiplicity rules, exclusions and effect-resolution forecasts. Review the freeze before opening protected traces.
- **Tests/outputs:** protocol schema check; A/A analysis reports no fabricated improvement; known-difference analysis detects known work; expected timing/memory budget; literal frozen configurations and status record.
- **Resources:** remaining final model training plus pilot-determined confirmation forecast. No fixed arbitrary performance target.
- **Acceptance:** a competent agent can run confirmation without choosing hypotheses or baselines afterward. Precision limits are explicit.
- **Decision:** if sensitivity cannot distinguish the effects of interest, revise using development data or declare the planned result capable only of a bounded/inconclusive answer. Never compensate for too few participants by counting frames as samples.

### WP07 — Protected controlled comparison

- **Objective/risk:** measure the declared effect on unseen subjects.
- **Dependencies/entry:** immutable WP06 protocol; clean environment matching the lock.
- **Actions/files:** `confirm` verifies all digests; run full correctness pass, native component and integrated replay blocks in the frozen order/schedule, practical quality comparisons and secondary paced check. Execute both predeclared hardware sessions before any confirmatory superiority decision. Session-one outputs may be checked for exactness/corruption, but cannot alter methods, schedule, baselines or analysis. Include every raw test tick under the stated boundaries. Keep failed runs.
- **Tests/outputs:** zero equality failures; raw per-recording/block timings and checksums; memory measurements; participant/seed outcomes; environmental observations. Separate cold start, steady replay and paced behavior.
- **Resources:** exactly the predeclared measured schedule, not repeated until favorable.
- **Acceptance:** complete auditable raw outputs or an explicit interrupted/blocked status. A positive result requires the statistical/guardrail rules in Section 10.
- **Decision:** mismatch -> implementation failure and invalid run; planned interval straddles one -> inconclusive at that sensitivity; interval favors baseline -> evidence against local acceleration; favorable component -> proceed to attribution, without automatically asserting application value.

### WP08 — Attribution, robustness and reproducibility

- **Objective/risk:** explain an apparent effect and expose hidden regressions.
- **Dependencies/entry:** WP07 or a documented diagnostic-only branch; ablations predeclared before test use.
- **Actions/files:** execute Section 8 tests, fixed-anchor eligibility analysis, per-subject/activity breakdowns, memory-matched memoization and panel/ordering sensitivity; analyze the second-session replication already collected in WP07. Any additional timing session is an explicitly exploratory/amended study. Rebuild from the locked environment on the same laptop in a clean directory. Perform interrupted/uninterrupted state comparison. Any newly conceived post-test test is labeled exploratory.
- **Tests/outputs:** ablation table with exactness, latency, quality, memory and claim classification; clean rebuild equality; resume identity; all nine subject effects and seed-specific results.
- **Resources:** predeclared robustness schedule; additional work only to resolve a concrete remaining interpretation risk.
- **Acceptance:** no stronger simpler explanation remains unreported; regressions are explicit. A second session is reproducibility, not a new dataset.
- **Decision:** if gains vanish after fair optimization, retract the superiority interpretation. If only a subset benefits, narrow the claim rather than redefine the primary population retroactively.

### WP09 — Synthesis, archive and next decision

- **Objective/risk:** produce an interpretable result even if negative, blocked or inconclusive.
- **Dependencies/entry:** all completed stages have raw evidence and decisions; blocked branches are documented.
- **Actions/files:** run `analyze`; generate `reports/final.md`, figures and machine-readable summaries; preserve exact configs, locks, code revisions, data/model hashes, failed runs, amendments and progress journal. Update novelty wording against the implemented algorithm, including any new targeted prior art found during execution.
- **Tests/outputs:** every reported number links to raw run IDs and an analysis version; no test-set configuration selected after results; reproduction commands work or name a concrete blocker.
- **Resources:** small analysis/reporting workload; do not introduce a new large experiment solely to seek a positive outcome.
- **Acceptance:** state separately whether correctness, component speed, whole-stream benefit and practical frontier improved; include all important unfavorable results and remaining uncertainty.
- **Decision:** continue, refine, narrow or stop according to Section 10. The coding agent may make routine non-destructive implementation choices autonomously; a methodological change requires a recorded amendment and the corresponding evidence boundary.

## 8. Attribution, robustness and failure analysis

### 8.1 Required discriminating tests

| Test | Precise comparison | What it can establish; important limitation |
|---|---|---|
| Dense versus eager versus certificate | Identical model, same native primitives and readout reuse | Whether P adds anything beyond established incremental inference. |
| Net versus cumulative | Native P and C-count; additionally identical periodic-anchor diagnostic | Whether tighter cancellation-aware eligibility pays for extra anchor storage/scans. Native hit-count differences alone are not causal attribution. |
| Equality versus positive-radius reuse | Fixed-anchor diagnostic reuses only at K=0 in one arm and at 2K<=slack in the other; anchor schedule identical | Separates exact recurrence from nonzero-distance certification. Include the deployable exact-key cache M too. |
| Thermometer versus bin table | Same first-layer function through packed bits and prefix LUT/bin deltas | Detects an encoding penalty masquerading as XOR value. Shared optimized first layer is the main result. |
| Certificate disabled | Same state layout/check code but forced exact refresh, plus plain dense | Measures certificate machinery cost; deliberately slow diagnostic is not a competitive baseline. |
| Original versus margin-grouped rows | Same weight function and readout permutation in all methods | Isolates grouping benefit without changing learned capacity. |
| Panel granularity | g grid, per-panel min slack and observed cost | Shows the checking-amortization/conservatism tradeoff; choose on development only. |
| Static simplification | Before/after constants and exact duplicate work removal, shared by all methods | Prevents constant or redundant neurons from creating an artificial temporal win. |
| Repeats and nonrepeating trajectories | Two-state toggles, two-entry cache, bounded-Hamming walks with unique full vectors | Flip/reflip validates cancellation but alone is solved by ordinary memoization. |
| Matching K, different margins | Same input difference paired with near-threshold versus far-threshold rows | Tests the certificate's causal dependence on margin, not merely sparsity. |
| Change distribution | Cluster flips in few words versus spread the same number across words | Tests whether eager changed-word cost, not total flip count, explains outcomes. |
| Order destruction | Shuffle already generated hidden inputs while preserving the input multiset | Negative control for temporal locality in the component benchmark; not a new valid physical sensor task. |
| Dense/noisy regimes | Independent random inputs, all-bit flips and bursts | Establishes overhead and failure regime; no claim that every input should accelerate. |
| Subject/activity shift | All protected subjects; static, dynamic, transition and unlabeled slices | Reveals heterogeneity; slices never replace the frozen primary population. |
| Thread/cache pressure | One thread and development-selected deployment threads; modest simultaneous-stream stress | Tests how cache-sized models and per-stream anchors scale; no unmeasured multi-device extrapolation. |
| Tail and startup | Refresh bursts, first tick, sequence cuts, model reloads | Detects a mean gain hiding tail latency or initialization regressions. |
| State restoration | Uninterrupted versus exact checkpoint/restore and replay | Ensures a resumed result is the same computation, not a new trajectory. |

Use fixed seeds and predeclared sizes for confirmatory synthetic controls. Generate at least one family with known reusable outputs, one with no useful margin, and one whose low flip count still spans most packed words. Verify the truth with direct sums before using its timing. Synthetic cases diagnose mechanism; only the real-stream evaluation supports the scoped workload claim.

### 8.2 WP05-L — Optional learning experiment after a measured opportunity

**Entry:** WP05 shows that useful outputs remain stable but the trained margins/change rates leave a plausible, quantified certificate opportunity; standard models and execution baselines are already credible. If this occurs before the protected evaluation, amend the development program and freeze afterward. If it occurs after test inspection, results on the same test subjects are exploratory.

Create four equal-budget fine-tuning arms from the same checkpoint: no added regularizer, temporal regularizer only, margin regularizer only, both. Use adjacent training ticks, including actual activity transitions; do not filter pairs by future labels. Keep preprocessing, model width and fine-tuning data/steps equal. Freeze BN statistics and its affine scale/bias for this stage in every arm, leaving the equivalent raw-sum thresholds fixed; optimize the shadow weights and readout. Constant units are handled explicitly, not by dividing through a zero affine scale. This prevents the regularizer from improving its apparent margin solely through normalization changes.

For a normalized pre-threshold variable u and temperature T fixed in the config, let p=sigmoid(u/T). A differentiable temporal proxy is the average of `p_t*(1-p_prev)+(1-p_t)*p_prev` over hidden units. Evaluate its relationship to actual hard bit changes; a lower surrogate is not success by itself. For a margin proxy, use the raw-sum distance to the equivalent frozen threshold, divided by sqrt(fan-in), with a hinge target derived from the development-measured radius at which checking becomes potentially profitable. Do not use an arbitrarily rescalable post-BN magnitude as the certificate margin. Treat the deployment threshold rounding as fixed during each training update and recompute/export/equality-check the final graph.

Choose regularization strengths on training/development data with zero included. Record exact strengths, temperature, target radius, optimization schedule and normalization. Share equivalent temporal regularization with a real-valued model if making a general learning-efficiency claim. Measure task quality, per-neuron entropy, constant/duplicate units, transition recall/lag, actual certificate hits, total training cost and native inference. The purpose is to test whether representation learning creates useful reusable computation; temporal sparsity and margin learning themselves are known. [TEMP21] [MARGIN21]

**Acceptance:** improved measured quality/cost frontier survives constant folding and stronger simple models. **Failure:** greater margins obtained through lost responsiveness, saturation, extra capacity, extra training or information are tradeoffs, not a breakthrough. Do not keep tuning against protected participants.

### 8.3 Failure categories

Every decision record assigns one of these categories, with evidence:

1. **Implementation/measurement invalid:** wrong bits, bad export, silent fallback, invalid timer, eliminated work, leakage or inconsistent state. Repair before scientific interpretation.
2. **Unsuitable proxy:** synthetic stability, repeated-key toy data, weak task representation or a scale that cannot expose the mechanism. Narrow the conclusion; use a better proxy only with a justified amendment.
3. **Resource limitation:** faithful operation/model/baseline cannot execute within actual local capacity. Report the strongest completed component claim and the missing dependency.
4. **Evidence against the local hypothesis:** correct sufficiently sensitive comparison favors the baseline in the declared real regime.
5. **No incremental advantage:** the mechanism is correct, but an existing eager/table/cache method explains or exceeds the benefit.
6. **Inconclusive:** uncertainty or incomplete baseline support prevents a conclusion. Do not recast this as either success or broad impossibility.

## 9. Reproducibility, logging and reporting

### 9.1 Configuration and provenance

Configuration files are immutable inputs to completed runs. Record schema version, task/model definition, bit/tie rules, thresholds, feature windows and numerical order, quantile method and hashes, split IDs, seeds, optimizer/schedule, training stride, backend/precision policy, method parameters, thread count and timing boundaries. Do not store only a config filename whose content can later change.

Minimal proposed configuration example (values are design defaults, not completed results):

```yaml
schema_version: 1
experiment_role: exploratory
dataset: uci_341_raw
split_manifest: data/manifests/splits.json
features:
  windows: [16, 64, 128]
  include_latest: true
  ddof: 0
  warmup_ticks: 127
  accumulator_dtype: float64
  recompute_period_ticks: 1024
encoder:
  quantile_fractions: sixteenths_1_through_15
  quantile_method: linear
  deduplicate_cutpoints: true
  equality_is_positive: true
model:
  hidden_widths: [512, 512]
  sign_at_zero: 1
  output_classes: 12
  canonical_threshold_rule: sum_ge_threshold
method:
  target_layer: hidden_2
  name: anchor_certificate
  panel_size: 16
  refresh: packed_full_sum
  share_unchanged_readout: true
measurement:
  unit: participant
  boundary: packed_input_to_consumed_output
  threads: 1
  repetitions: pilot_selected_before_confirmation
```

Before confirmation, replace symbolic choices such as repetition count with concrete values and digests. The freeze command rejects unresolved placeholders, missing comparators, changed data/config hashes or absent correctness reports. A full source revision plus a dirty-tree diff hash is required; prefer a clean committed tree for confirmation.

### 9.2 Experiment records

Write append-only JSONL records with these required fields:

| Group | Required fields |
|---|---|
| Identity | `schema_version`, `run_id`, UTC timestamp, stage/WP, exploratory/confirmatory label, protocol digest |
| Provenance | source commit, dirty diff digest, config digest, dependency lock digest, compiler/flags, executable hash, hardware/OS record |
| Assets | archive/data manifest digest, split digest, model/export digest, preprocessing/quantile digest, model seed |
| Workload | participant ID, recording ID, split role, raw ticks, consumed hidden inputs, labeled/ambiguous/warm-up counts, chronological-order policy |
| Measurement | method/config, timing boundary, session/block/order, thread count, elapsed nanoseconds, number of complete replays, resets, output checksum |
| Resources | process peak RSS bytes, accelerator peak bytes when applicable, auxiliary state/table bytes, swap/pressure observations, setup/training time |
| Correctness | mismatch count, first failing sample if any, comparison oracle/export version, state-restore test status |
| Status | completed/failed/interrupted/excluded, exception or exit status, paired-block exclusion reason, amendment ID if relevant |

Use integer nanoseconds and bytes in raw records; derived presentation units belong in the analysis. Keep complete raw timing arrays, subject summaries, quality confusion matrices, diagnostic trace manifests and negative outcomes. A failed run is not deleted when a retry succeeds.

Store progress in `decisions/status.md` and append decisions to `decisions/journal.jsonl`: issue, evidence/run IDs, chosen action, rejected alternatives, scope impact, next gate and unresolved prerequisite. This makes execution resumable without conversation history.

### 9.3 Final results format

The result report must contain:

- A short claim ledger marking C1/C2/C3 and H0–H5 as supported, contradicted, inconclusive or untested.
- Absolute times and ratio intervals; every subject effect; seed and hardware-session variation; timing boundaries and workload denominators.
- A quality/latency/memory frontier showing compact practical baselines, including any that dominate the proposed BNN.
- A breakdown of certificate cost, refresh cost, positive-K reuse, repeated-input reuse, state traffic and ordinary eager savings.
- Tail/startup/transition regressions, unsuccessful configurations, unavailable baselines, protocol deviations and access limitations.
- Commands to retrieve assets, recreate environments, reproduce canonical exports and rerun the locked analysis.

Recommended figures are ordinary exported plots: participant effect intervals; latency versus changed-word fraction and panel slack; certified versus actually unchanged panel fraction; quality versus absolute time; and memory versus simultaneous streams. Keep units, sample counts, error-bar definitions and measured-versus-estimated labels explicit. No decorative generated imagery is needed.

## 10. Outcome interpretation and evidence-triggered continuation

| Outcome | Defensible conclusion | Next action |
|---|---|---|
| Exactness fails | No valid performance result | Repair arithmetic/export/state; rerun relevant correctness and affected measurements. |
| Mathematical reuse works only synthetically | Mechanism demonstration | Report it; gather real-trace headroom before claiming usefulness or scaling. |
| P beats dense but loses to E-word/E-hybrid | Existing incremental execution explains the opportunity | Report no added certificate advantage; consider contributing the best exact baseline implementation. |
| P beats C-count but loses to exact-input memoization | Recurrence may explain the gain | Examine positive-K hits and unique trajectories; narrow or reject the special-certificate claim. |
| C-count wins despite fewer hits | Tightness is outweighed by XOR anchor overhead | Prefer the simpler counter implementation; no novelty by renaming it. |
| C1 interval favors P, both session directions agree and audits pass, C2 does not | Reproducible component finding under the stated thread budget | Publish scoped results/crossover; optimize unrelated costs only if a measured application case warrants it. |
| C2 improves but practical classifiers dominate | Faster execution of this BNN, no application superiority | Retain the component result; do not market it as a better activity recognizer. |
| Practical frontier improves with no hidden regression | Evidence of value for this workload and laptop | Seek independently sourced real streams and broader hardware replication before general claims. |
| Interval crosses one | Inconclusive at stated sensitivity | Report interval/resolution; more participants, not merely more frames, may be needed. |
| Stable tiny saving | Genuine narrow difference if above measurement error | Report absolute CPU duty and costs; do not call it a breakthrough. |
| Precise unfavorable result across relevant regimes | Evidence against this implementation/mechanism in that scope | Stop this branch unless a specific new mechanism or reachable scale has evidence-backed headroom. |
| Prior identical method located | Novelty claim fails | Reclassify as replication, application or engineering; evaluate whether the implementation still adds value. |

### 10.1 Scale and external validity

The local experiment can test exactness, representation flip/margin distributions, cost crossover and latency on real streams. It cannot establish usefulness for large language models, event-camera hardware, arbitrary video, training efficiency or whole-device energy without the corresponding evidence.

A small-layer loss does not automatically disprove a scale-dependent mechanism. Escalate synthetic widths only if measured checking/refresh costs predict a crossover within reachable memory and a plausible useful layer can occupy that regime. Validate the prediction at more than one size. If learned traces remain uncertifiable as width increases, invoking future scale is not a sufficient justification. Keep hypothetical larger-task predictions separate from measured points.

### 10.2 Conditional Walsh diagnostic

Switching to Walsh structure is justified only if primary findings suggest a more relevant dense-mixing bottleneck or if a separate compressibility diagnostic shows task-trained projections with useful structure. It is not an automatic consolation project after a negative result.

For a square d-by-d frozen dense map, define Q as the normalized Sylvester transform and

    W_hat = Q diag(lambda) Q + U V^T.

The Frobenius-optimal uncorrected lambda is `diag(Q W Q)`. Report captured weight energy and held-out activation error. For isotropic random W, the ratio of expected captured energy to expected total energy is 1/d; use that as a negative control. A projection followed by rank-r SVD of its residual is an initialization, not necessarily the jointly optimal solution. Fit the data-weighted output error on training/calibration inputs, and distinguish frozen-weight compression from training coadaptation.

Before a new application study, run synthetic teachers: exact Walsh, Walsh plus known low rank, pure low rank, arbitrary dense, and permuted/noncommuting structure. Compare dense, smaller dense, low rank, Adaptive Fastfood, Monarch/BTT, pure Walsh and Walsh plus correction. Use an explicit dense materialization of the identical proposed map as an execution control. Do not initialize both low-rank factors at zero. Use shape-specific initialization/LR tuning, not dense defaults imposed on BTT. [FASTFOOD15] [ACDC16] [MONARCH22] [BTT24]

Native MLX Hadamard makes these component diagnostics local. A later end-to-end study needs its own fixed task, splits, quality metric, compute budget and protocol before protected data use; the current sensor protocol does not silently authorize a new claim. A cheap correction that dominates the entire learned function, or a residual approaching dense cost, weakens the proposed structured explanation. Simple Walsh-plus-correction is not automatically novel given the existing literature and dyadic disclosure. [MLXH] [DYADIC20]

Other pivots require their own evidence: parity structure for GF(2) bases, genuinely long-context candidate-recall/cost headroom for Hamming routing, or measured activation-storage dominance for reversible training. None is a reason to skip the current negative evidence.

## 11. Audit record, unresolved gates and checks performed

### 11.1 Audits conducted while preparing this plan

Three distinct AI-assisted challenge reviews covered scientific/prior-art validity, experimental validity, and implementation/feasibility. They are not external human peer review. Their concrete findings changed the plan as follows:

| Audit | Material weakness found | Revision incorporated |
|---|---|---|
| Scientific/prior art | Basic dyadic/Walsh operator already explicitly disclosed | Removed basic-operator novelty; selected a narrower exact-reuse question. |
| Scientific/prior art | Event inference, stability training and bit-flip bounds already exist | Named closest alternatives; made new-method status conditional. |
| Scientific/prior art | Certified anchor reuse also has a current LipCache precedent | Distinguished same-module exact bits from proxy consistency; reduced novelty claim. |
| Scientific/attribution | Two-state cancellation is solved by a tiny memoization cache | Added exact-key caches, positive-K analysis and nonrepeating controls. |
| Experimental validity | Thermometer expansion hides a cheap exact first-layer algorithm | Added prefix LUT/bin delta and selected the learned second layer as the primary intervention. |
| Experimental validity | Small float classifiers and cached causal models could dominate | Added original-feature baselines and separated component/application claims. |
| Experimental validity | Seeds, frames and repeats could inflate nominal sample size | Fixed participant as the independent unit, froze repeat rules and limited inference from nine subjects. |
| Experimental validity | Recycled participant splits could be called fresh confirmation | Required genuinely unexposed data after a changed post-test hypothesis. |
| Experimental validity | Head caching or stale timing could create an artificial win | Shared downstream reuse; native equal boundaries; A/A and known-difference gates. |
| Implementation | Absolute-margin tie handling is wrong on one side | Specified asymmetric integer slack and exhaustive threshold tests. |
| Implementation | Output-only dense baseline might be forced to do extra work | Public interface returns bits; dense early termination remains eligible. |
| Implementation | Cumulative control could be burdened with unnecessary anchors | Gave it its natural counter-only implementation and separate native costs. |
| Implementation | Larq has different packing/sign/threshold conventions | Added explicit adapter and all-attainable-sum validation. |
| Implementation, final pass | Activation STE, negative-scale export and pre-quantizer dtype were ambiguous | Specified STE on both signs, F(-s') reparameterization, and one FP64 raw/feature path. |
| Experimental validity, final pass | Seed selection, reset costs and two-session schedule could diverge | Fixed three-export selection, one-thread primary policy, resets inside timing and both sessions in WP07. |
| Experimental validity, final pass | A favorable pooled estimate could conceal session reversal or a quality loss | Added session-direction qualifications and explicit descriptive/formal frontier rules. |
| Bibliography, final pass | Two author entries conflated names/related datasets | Corrected Franyell Silfa and the official UCI341 creator list against primary metadata. |
| Feasibility | M3 INT8, MPS and Hadamard support were being assumed | Added exact-operation smoke tests and released-version locking; no claimed target benchmark. |
| Recovery | Rebuilt floating rolling sums can alter a resumed binary trace | Require exact preprocessing state bytes or full recording replay. |

The final consistency review checked downstream dependencies after these revisions, especially primary metric definitions, model selection, common first-layer behavior and state invariants. Targeted rereads of the revised experimental and implementation sections found no additional material defect within their stated scope. Remaining uncertainty is not converted into a promise of success. Reference-definition/code-fence and embedded-checker integrity are also validated before delivery; this document review does not constitute a native implementation test or a successful fetch of every linked page.

### 11.2 Explicit unresolved prerequisites and limitations

- Actual laptop memory/storage/macOS, current package compatibility and native LCE execution remain WP00 gates.
- The raw archive was not downloaded/parsed during research; label endpoint convention and actual raw tick counts remain data-validation gates. The metadata supports selection but does not substitute for parsing.
- No trained-network flip/margin distribution or M3 speed/energy measurement exists for this project yet.
- Full WHTMix conditions and several recent hardware-skipping papers could not be inspected; they are not decisive evidence for a positive claim.
- Broader literature, unpublished code, non-English work, paywalled hardware details and comprehensive patent coverage remain blind spots. This search is not proof of absent prior art and is not a patentability or freedom-to-operate opinion.
- A conservative certificate may never outperform the best existing methods on a useful local workload. That is an allowed result.
- Nine held-out laboratory participants support limited population inference; longer recordings do not fix that limitation.
- A correct computational PoC does not validate clinical, safety-critical or general real-world deployment of an activity classifier.

### 11.3 Small checks actually executed for this handoff

Environment: Linux 6.18.55 x86-64, CPython 3.12.14, NumPy 2.3.5. Seed: 20261029. These checks were executed during planning, not on the M3 Max.

| Check | Actual result | Scope |
|---|---|---|
| Exhaustive signed binary changed-word identity, n=1..5 | 37,448 weight/anchor/current triples; all passed | Integer algebra only. |
| Threshold certificate over those triples | 476,136 conditions checked; 160,894 certified cases; no false certificate | Includes thresholds outside the sum range and both inequality sides. |
| Randomized panel state simulation | 9,000 steps across n={1,7,64,65,127,517} and g={1,4,16}; all matched full sums | Python state semantics, not NEON or trained representations. |
| State continuation | Complete deep-copied state at step 250 produced the same subsequent outputs | Not a disk-checkpoint or floating feature-state test. |
| Flip/reflip illustration | Net bound accepted all 80 constructed steps; cumulative bound without reanchoring accepted 32 | Deliberately synthetic algebra illustration; no latency/novelty evidence. |
| Walsh/XOR convolution, d=16 | Maximum absolute difference 8.881784197001252e-16 | Float64 numerical sanity check, not MLX/M3 support. |
| Walsh family and XOR-only limit | Symmetry, constant diagonal, commutation/composition checks passed; no affine two-input XOR map represented AND | Small algebra checks, not a representational benchmark. |

The exact small checker source is included in Appendix A so the handoff does not depend on an inaccessible intermediate script. Its hit counts are not estimates of real-model reuse. Native correctness, feature arithmetic, export and complete checkpoint tests remain required work.

## 12. References and search-coverage record

### 12.1 Evidence register

All URLs below were identified during research dated 29 October 2026. “Inspected” describes the accessible evidence actually consulted, including delegated research. A repository license statement applies to the inspected repository, not automatically to every linked weight or dataset. No external benchmark number is treated as a measurement from this project.

#### Binary arithmetic, logic and reversible computation

- **[BNN16]** Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv and Yoshua Bengio. *Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1*. 2016, arXiv:1602.02830. Primary abstract/metadata inspected. Establishes binary training/inference lineage; historical benchmark conditions do not establish current competitiveness.
- **[XNOR16]** Mohammad Rastegari, Vicente Ordonez, Joseph Redmon and Ali Farhadi. *XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks*. ECCV 2016, arXiv:1603.05279. Primary abstract/metadata inspected. Supports packed arithmetic and contextual memory/operation claims, not universal whole-model speedups.
- **[XOR20]** *XOR-Net: An Efficient Computation Pipeline for Binary Neural Network Inference on Edge Devices*. ICPADS 2020, DOI 10.1109/ICPADS51040.2020.00026. Publisher-indexed abstract inspected. Direct pipeline precedent for replacing XNOR by XOR and reorganizing scale computation; detailed target reproduction not performed.
- **[LARQ]** Larq authors. *Larq Compute Engine*, official documentation and repository, accessed 2026-10-29. Getting-started/performance/build pages and repository licensing inspected. Documents native optimized binary operators and historical M1 measurements; current M3 build is unverified. Repository: <https://github.com/larq/compute-engine>; build: <https://docs.larq.dev/compute-engine/build/>.
- **[LARQOPS]** Larq authors. *TFLite Operators* and *Model Optimization Guide*, accessed 2026-10-29. Operator layout, bit conventions, threshold and channel-alignment guidance inspected. These govern adapter validation, not a claim that every graph converts. Additional guide: <https://docs.larq.dev/compute-engine/model_optimization_guide/>.
- **[BITNET]** Microsoft Research. *BitNet/bitnet.cpp* official repository and linked *BitNet b1.58 2B4T Technical Report*, 2024–2026 repository history; report arXiv:2504.12285, 2025. README/model descriptions inspected. Ternary-weight inference is a relevant frontier, not an XOR-only network or sensor baseline. Report: <https://arxiv.org/abs/2504.12285>.
- **[LOGIC22]** Felix Petersen, Christian Borgelt, Hilde Kuehne and Oliver Deussen. *Deep Differentiable Logic Gate Networks*. NeurIPS 2022. Primary proceedings abstract inspected. Establishes learning mixed nonlinear Boolean gates through relaxation; reported MNIST throughput is configuration-specific.
- **[CONVLOGIC24]** Felix Petersen, Hilde Kuehne, Christian Borgelt, Julian Welzel and Stefano Ermon. *Convolutional Differentiable Logic Gate Networks*. NeurIPS 2024, arXiv:2411.04732. Primary abstract and paper opening inspected. Logical tree convolutions/OR pooling extend the neuron/circuit alternative; no local reproduction.
- **[LIGHT26]** Lukas Rüttgers, Till Aczel, Andreas Plesner and Roger Wattenhofer. *Light Differentiable Logic Gate Networks*. ICLR 2026; arXiv:2510.03250. Primary proceedings abstract inspected. Identifies parametrization, gradient and training-cost issues; evidence against treating logic-gate training as solved in general.
- **[GF2CIRCUIT18]** Andrea Visconti, Chiara V. Schiavo and Rene Peralta. *Improved upper bounds for the expected circuit complexity of dense systems of linear equations over GF(2)*. Information Processing Letters 137, 2018, DOI 10.1016/j.ipl.2018.04.010. NIST primary record and author manuscript passages inspected. Establishes cancellation/common-path circuit optimization beyond naive rank factorization; not an ML accuracy result.
- **[REV17]** Aidan N. Gomez, Mengye Ren, Raquel Urtasun and Roger B. Grosse. *The Reversible Residual Network: Backpropagation Without Storing Activations*. NeurIPS 2017, arXiv:1707.04585. Primary abstract inspected. Activation reconstruction is established; binary gradient and optimizer-memory benefits remain separate questions.

#### Structured transforms and routing

- **[FASTFOOD15]** Zichao Yang, Marcin Moczulski, Misha Denil, Nando de Freitas, Alex Smola, Le Song and Ziyu Wang. *Deep Fried Convnets*. 2015, arXiv:1412.7149. Primary abstract and ACDC's account of Adaptive Fastfood inspected. Trainable Hadamard/diagonal/permutation layers are existing techniques.
- **[ACDC16]** Marcin Moczulski, Misha Denil, Jeremy Appleyard and Nando de Freitas. *ACDC: A Structured Efficient Linear Layer*. ICLR 2016, arXiv:1511.05946. Parsed paper inspected, including parameterization, implementation and theory qualifications. Practical real DCT constructions must not inherit unrelated universality guarantees without checking assumptions.
- **[DYADIC20]** Junseok Park and Inyup Kang; Samsung Electronics assignee. *Neural network processor using dyadic weight matrix and operation method thereof*. US20200167637A1, published 2020-05-28; US11562046B2, published 2023-01-24; listed priority 2018-11-26. Public description/equations/claims inspected. Explicit XOR-indexed matrix and WHT evaluation are technical prior art; no legal-status conclusion is made.
- **[WHT22]** Hongyi Pan and coauthors. *Block Walsh-Hadamard Transform Based Binary Layers in Deep Neural Networks*. ACM TECS, 2022; arXiv:2201.02711; DOI 10.1145/3510026. Abstract and parsed paper inspected, along with the related CVPRW 2021 WHT/smooth-thresholding paper. Layer-specific Jetson and quality tradeoffs cannot become end-to-end M3 predictions. Related source: <https://openaccess.thecvf.com/content/CVPR2021W/BiVision/html/Pan_Fast_Walsh-Hadamard_Transform_and_Smooth-Thresholding_Based_Binary_Layers_in_Deep_CVPRW_2021_paper.html>.
- **[MONARCH22]** Tri Dao and coauthors. *Monarch: Expressive Structured Matrices for Efficient and Accurate Training*. ICML 2022, PMLR 162. Parsed paper and appendices inspected. Permuted block-diagonal factors are a direct structured alternative. Paper-linked code: <https://github.com/HazyResearch/fly>; current local compatibility/license must be checked before copying.
- **[BTT24]** Shikai Qiu, Andres Potapczynski, Marc Finzi, Micah Goldblum and Andrew Gordon Wilson. *Compute Better Spent: Replacing Dense Layers with Structured Matrices*. ICML 2024, arXiv:2406.06248. Method, optimization, runtime and limitation sections inspected. BTT and structure-specific optimization are necessary comparisons for a Walsh extension. Apache-2.0 reference repository <https://github.com/shikaiqiu/compute-better-spent> has a CUDA-oriented environment; ported formulas are not reproduced original kernel timings.
- **[WHTMIX26]** Sajith, Hamdan and Çetin. *WHTMix*, arXiv:2607.25234, first submitted 2026-07-28. Primary indexed abstract inspected; direct full text unavailable during research. The reported stereo latency result is contextual only; detailed conditions remain unverified.
- **[REFORMER20]** Nikita Kitaev, Łukasz Kaiser and Anselm Levskaya. *Reformer: The Efficient Transformer*. ICLR 2020, arXiv:2001.04451. Primary abstract/metadata inspected; foundational LSH-attention reference, not a direct local comparator for the selected sensor task.
- **[RABIT26]** Wenhao Li, Jinhao Dong, Hailin Zhang, Wenhang Shi, Wei Lu and Xiaoyong Du. *RaBitQCache: Rotated Binary Quantization for KVCache in Long Context LLM Inference*. ICML 2026, PMLR 306. Primary proceedings abstract inspected. Binary proxy estimation, bounds and adaptive retrieval are current prior art for the routing proposal. Code linked by authors: <https://github.com/Sakuraaa0/RaBitQCache>.

#### Temporal reuse, margins and certificates

- **[SD16]** Peter O'Connor and Max Welling. *Sigma Delta Quantized Networks*. 2016/ICLR 2017, arXiv:1611.02024. Primary abstract and EvNet's direct comparison inspected. Establishes change-driven computation with quantized transmissions.
- **[CBINFER]** Lukas Cavigelli and coauthors. *CBinfer: Change-Based Inference for Convolutional Neural Networks on Video Data*. 2017 and extended work. Official repository/architecture/license and institutional abstract inspected. CUDA, Apache-2.0; not directly executable on M3 without a port.
- **[DELTACNN22]** Mathias Parger and coauthors. *DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos*. CVPR 2022. Paper opening, official project description and repository README inspected, including zero-threshold and backend guidance. Code: <https://github.com/facebookresearch/DeltaCNN>, archived 2023, CUDA and CC BY-NC 4.0.
- **[EV22]** Matthew Dutson, Yin Li and Mohit Gupta. *Event Neural Networks*. ECCV 2022, arXiv:2112.00891. Full paper inspected, especially state equations, initialization, transmission policies and CPU experiment. MIT PyTorch reference: <https://github.com/wision-lab/event-nn-pytorch>. Eager state semantics are reused conceptually, with a separately validated integer specialization.
- **[TEMP21]** Amirreza Yousefzadeh and Manolis Sifalakis. *Training for temporal sparsity in deep neural networks, application in video processing*. 2021, arXiv:2107.07305. Primary abstract/metadata and indexed equations inspected; direct HTML full-text route unavailable. Supports prior-art status of temporal regularization, not this exact certificate's performance.
- **[MARGIN21]** Sebastian Buschjäger, Jian-Jia Chen, Kuan-Hsun Chen, Mario Günzel, Christian Hakert, Katharina Morik, Rodion Novkin, Lukas Pfahler and Mikail Yayla. *Margin-Maximization in Binarized Neural Networks for Optimizing Bit Error Tolerance*. DATE 2021, DOI 10.23919/DATE51398.2021.9473918. Full author PDF, Theorem 1 and corollaries inspected. Source of the established 2q change bound and related robustness reasoning.
- **[FUZZY22]** Franyell Silfa, José-María Arnau and Antonio González. *Saving RNN Computations with a Neuron-Level Fuzzy Memoization Scheme*. 2022, arXiv:2202.06563. Primary abstract and indexed manuscript passages inspected; one institutional PDF fetch returned 403. Approximate neuron reuse is close conceptual prior art.
- **[LIPCACHE26]** Zhengzhe Xiang, Yinlin Chen, Fuli Ying, Binbin Zhou, Hailiang Zhao and Schahram Dustdar. *LipCache: A Local Inference Proxy with Certified Caching for Edge Image Classification Service*. arXiv:2608.13144v1, 2026-08-13. Full HTML inspected, especially Section V and Remark V.2. Its own text limits certification to GuardNet consistency; it is not proof of exact MainNet behavior.
- **[DEEPCACHE18]** Mengwei Xu and coauthors. *DeepCache: Principled Cache for Mobile Deep Vision*. MobiCom 2018, DOI 10.1145/3241539.3241563. Official Microsoft Research abstract and implementation README inspected. Author PDF <https://xumengwei.github.io/files/MobiCom18-DeepCache.pdf>; code <https://github.com/xumengwei/DeepCache>. Mobile temporal cache precedent, distinct from diffusion DeepCache.
- **[LOCAL24]** Lee and Kim. *Latency and accuracy optimization for binary neural network inference with locality-aware operation skipping*. Electronics Letters, 2024, DOI 10.1049/ell2.13090. Primary article mechanism/related-work text inspected. Training for locality and operation skipping is prior art; approximate and exact variants must be distinguished.
- **[EARLY26]** *Threshold-Based Early Stopping of Accumulations in Neural Networks with Binary Activation*. arXiv:2608.06177, August 2026. Primary index/abstract evidence located; direct full text unavailable. A related 2026 publisher abstract, DOI 10.1016/j.aeue.2026.156548, concerns activation-deterministic early termination. Treat detailed current implementations as an unresolved comparison check; the simple exact remaining-sum bound is specified independently in this plan.
- **[WEIGHTPAT25]** Choong Seon Hong, Lok Won Kim, Quang Hieu Vo and Hyeon Su Kim; Kyung Hee University. *Computation method for binary neural network and computing device for executing the same*, US20250124254A1, published 2025-04-17. Hamming-distance/MST reuse between binary weight rows was inspected. Adjacent non-temporal redundancy mechanism; not evidence of target-device speed or a legal opinion.

#### Data and execution documentation

- **[HAR341]** Jorge Reyes-Ortiz, Davide Anguita, Luca Oneto and Xavier Parra / UCI Machine Learning Repository. *Smartphone-Based Recognition of Human Activities and Postural Transitions*. 2015, DOI 10.24432/C54G7M. Official dataset description, raw-file schema, split/license/download metadata inspected; archive itself not parsed during planning. CC BY 4.0.
- **[APPLEHW]** Apple. *MacBook Pro (14-inch, Nov 2023) — Technical Specifications*. Official specifications inspected during delegated research; a later root fetch was rate-limited. Supports configuration variability only; discover the actual machine locally.
- **[ARM]** Arm. *Arm Neon Intrinsics Reference*, accessed 2026-10-29. XOR, byte population count and reductions inspected. Instruction availability is not a measured complete-kernel speedup.
- **[TORCHMPS]** PyTorch. *MPS backend*, MPS environment variables and synchronization documentation, accessed 2026-10-29. Availability/precision/fallback guidance inspected. Main/stable documentation can differ from installed releases; use the selected release's requirements. Additional pages: <https://docs.pytorch.org/docs/stable/mps_environment_variables.html> and <https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html>.
- **[MLXH]** Apple MLX. *mlx.core.hadamard_transform*, installation, lazy evaluation and memory API documentation, accessed 2026-10-29 (retrieved docs labeled 0.32.4). Native operation/scaling/size rules inspected; actual target execution untested. Repository <https://github.com/ml-explore/mlx>; synchronization <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html>.
- **[MLXPREC]** Apple MLX. *Numerical Precision* and data-type documentation, accessed 2026-10-29. FP32 internal policy and CPU-only float64 guidance inspected. Lock actual precision behavior in any structured experiment.
- **[ACCELERATE]** Apple. *BLAS Library*, official developer documentation, accessed 2026-10-29. Optimized dense CPU API reference; exact local shape/thread performance remains a measurement.
- **[XNNPACK]** Google/XNNPACK contributors. Official repository and `BUILD.md`, accessed 2026-10-29. Native macOS arm64/Clang support inspected. Operator/quantization selection still requires local verification.
- **[COREML]** Apple. Core ML Tools *Quantization Performance* and *Quantization API* guides, accessed 2026-10-29. Backend-dependent compression/decompression and hardware discussion inspected. Additional API guide: <https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html>.
- **[ENERGY]** Apple. *Energy Efficiency Guide for Mac Apps: Monitor Usage Regularly*, archived official documentation, accessed 2026-10-29. Diagnostic tools and Energy Impact interpretation inspected. Current OS counters/permissions are local gates.

### 12.2 Search scope, tracing and stopping rationale

Searches used two web-search systems and primary proceedings/arXiv sources, author/institutional pages, official repositories, hardware documentation and targeted public patent disclosures. The task-specific query families included:

- binary/XOR/XNOR inference, XOR-Net, Larq, ternary BitNet and current logic-gate networks;
- Walsh/Hadamard, XOR-circulant and **dyadic** neural matrices, Fastfood/ACDC, block transforms, butterfly, Monarch/BTT and structured corrections;
- delta/event/change-based inference, temporal sparsity, cached accumulators, exact binary changes, lazy threshold evaluation and locality-aware skipping;
- anchor/Hamming/margin certificates, certified semantic caching, fuzzy neuron memoization, early termination and binary weight-row reuse;
- GF(2) rank/circuit synthesis, reversible coupling, LSH and current binary sparse-attention proxies;
- Apple/Arm/MPS/MLX/Core ML compatibility, native baseline builds, raw streaming dataset schemas and licenses.

Backward tracing from ACDC found Adaptive Fastfood; dyadic terminology revealed the almost exact Walsh disclosure. Forward tracing from Monarch found BTT and structure-specific optimization requirements. EvNet and locality-skipping references exposed eager temporal reuse and margins. Tracing LipCache exposed the older mobile DeepCache lineage and the proxy-versus-main-model guarantee distinction. Current logic-gate and RaBitQCache results tightened the alternatives beyond the original memo.

Final targeted searches for exact anchor-relative binary certification, Hamming cache radii, lazy panel updates and equivalent threshold mechanisms mostly returned these same families or adjacent verification/hardware work. They did not reveal a materially different directly executable competitor that changed the core design. That is the stopping rationale, not a paper quota and not proof that no identical work exists.

Remaining coverage gaps are listed in Section 11.2. Refresh the closest-equivalence search when the actual implemented algorithm changes and before publication. Do not repeat the entire literature search merely to postpone an informative experiment.

[BNN16]: https://arxiv.org/abs/1602.02830
[XNOR16]: https://arxiv.org/abs/1603.05279
[XOR20]: https://doi.org/10.1109/ICPADS51040.2020.00026
[LARQ]: https://docs.larq.dev/compute-engine/
[LARQOPS]: https://docs.larq.dev/compute-engine/api/operators/
[BITNET]: https://github.com/microsoft/BitNet
[LOGIC22]: https://papers.neurips.cc/paper_files/paper/2022/hash/0d3496dd0cec77a999c98d35003203ca-Abstract-Conference.html
[CONVLOGIC24]: https://arxiv.org/abs/2411.04732
[LIGHT26]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/6d8910336fe8a0eee83f2fa660347147-Abstract-Conference.html
[GF2CIRCUIT18]: https://www.nist.gov/publications/improved-upper-bounds-expected-circuit-complexity-dense-systems-linear-equations-over
[REV17]: https://arxiv.org/abs/1707.04585
[FASTFOOD15]: https://arxiv.org/abs/1412.7149
[ACDC16]: https://arxiv.org/abs/1511.05946
[DYADIC20]: https://patents.google.com/patent/US11562046B2/en
[WHT22]: https://arxiv.org/abs/2201.02711
[MONARCH22]: https://proceedings.mlr.press/v162/dao22a/dao22a.pdf
[BTT24]: https://proceedings.mlr.press/v235/qiu24f.html
[WHTMIX26]: https://arxiv.org/abs/2607.25234
[REFORMER20]: https://arxiv.org/abs/2001.04451
[RABIT26]: https://proceedings.mlr.press/v306/li26al.html
[SD16]: https://arxiv.org/abs/1611.02024
[CBINFER]: https://github.com/lukasc-ch/CBinfer
[DELTACNN22]: https://openaccess.thecvf.com/content/CVPR2022/papers/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.pdf
[EV22]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710276.pdf
[TEMP21]: https://arxiv.org/abs/2107.07305
[MARGIN21]: https://daes.cs.tu-dortmund.de/storages/daes-cs/r/publications/yayla_date21.pdf
[FUZZY22]: https://arxiv.org/abs/2202.06563
[LIPCACHE26]: https://arxiv.org/html/2608.13144v1
[DEEPCACHE18]: https://www.microsoft.com/en-us/research/publication/deepcache-principled-cache-for-mobile-deep-vision/
[LOCAL24]: https://doi.org/10.1049/ell2.13090
[EARLY26]: https://arxiv.org/abs/2608.06177
[WEIGHTPAT25]: https://patents.google.com/patent/US20250124254A1/en
[HAR341]: https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions
[APPLEHW]: https://support.apple.com/en-gb/117737
[ARM]: https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
[TORCHMPS]: https://docs.pytorch.org/docs/main/notes/mps.html
[MLXH]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.hadamard_transform.html
[MLXPREC]: https://ml-explore.github.io/mlx/build/html/usage/precision.html
[ACCELERATE]: https://developer.apple.com/documentation/accelerate/blas-library
[XNNPACK]: https://github.com/google/XNNPACK/blob/master/BUILD.md
[COREML]: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html
[ENERGY]: https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/MonitoringEnergyUsage.html

## Appendix A. Complete small algebra checker used during planning

Save the following as `validate_mechanisms.py` and execute it with Python and NumPy. It writes a small JSON result next to itself. Its platform string will describe the environment where it is rerun. It is independent of the proposed project commands and does not benchmark performance.

```python
"""Small algebra checks for the research handoff; not an M3 benchmark."""
import copy
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np


def dot(w, x, n):
    return n - 2 * ((w ^ x) & ((1 << n) - 1)).bit_count()


checks = {"integer_update_cases": 0, "certificate_cases": 0, "certified_cases": 0}
for n in range(1, 6):
    for w, a, x in itertools.product(range(1 << n), repeat=3):
        za, zx = dot(w, a, n), dot(w, x, n)
        d = a ^ x
        updated = za + 4 * (d & (a ^ w)).bit_count() - 2 * d.bit_count()
        assert updated == zx
        checks["integer_update_cases"] += 1
        for tau in range(-n - 1, n + 2):
            ya = za >= tau
            slack = za - tau if ya else tau - 1 - za
            checks["certificate_cases"] += 1
            if 2 * d.bit_count() <= slack:
                checks["certified_cases"] += 1
                assert (zx >= tau) == ya


class Panel:
    def __init__(self, weights, tau, n):
        self.weights, self.tau, self.n = weights, tau, n
        self.a = None
        self.hits = self.misses = 0

    def step(self, x):
        if self.a is not None and 2 * (self.a ^ x).bit_count() <= self.min_slack:
            self.hits += 1
            return self.y
        self.z = [dot(w, x, self.n) for w in self.weights]
        self.y = [z >= t for z, t in zip(self.z, self.tau)]
        self.min_slack = min(z-t if y else t-1-z for z,t,y in zip(self.z,self.tau,self.y))
        self.a = x
        self.misses += 1
        return self.y


rng = np.random.default_rng(20261029)
panel_steps = panel_hits = panel_misses = 0
for n in [1, 7, 64, 65, 127, 517]:
    for g in [1, 4, 16]:
        weights = [int.from_bytes(rng.bytes((n+7)//8), "little") & ((1 << n)-1) for _ in range(g)]
        tau = rng.integers(-n-1,n+2,size=g).tolist()
        panel = Panel(weights,tau,n)
        x = 0
        saved = None
        for t in range(500):
            if t % 37 == 0:
                x = int.from_bytes(rng.bytes((n+7)//8), "little") & ((1 << n)-1)
            elif t % 3:
                x ^= 1 << int(rng.integers(n))
            y = panel.step(x)
            assert y == [dot(w,x,n) >= threshold for w,threshold in zip(weights,tau)]
            if saved is not None:
                assert saved.step(x) == y
            if t == 249:
                saved = copy.deepcopy(panel)
            panel_steps += 1
        panel_hits += panel.hits
        panel_misses += panel.misses

# Flip/reflip cancellation: no output changes, but cumulative flips eventually
# force a refresh whereas the net-anchor distance stays at most one.
n = 64
anchor = (1 << n)-1
flip = anchor ^ 1
slack = dot(anchor,anchor,n)
net_passes = cumulative_passes = 0
cumulative = 0
previous = anchor
for t in range(80):
    current = flip if t % 2 == 0 else anchor
    cumulative += (previous ^ current).bit_count()
    net_passes += 2 * (anchor ^ current).bit_count() <= slack
    cumulative_passes += 2 * cumulative <= slack
    previous = current
assert net_passes == 80 and cumulative_passes == 32

# Explicit Hadamard/XOR convolution and projection-family properties.
H = np.array([[1]], dtype=np.int64)
for _ in range(4):
    H = np.block([[H,H],[H,-H]])
d = len(H)
k, x = rng.normal(size=(2,d))
W = np.array([[k[i ^ j] for j in range(d)] for i in range(d)])
error = float(np.max(np.abs(W @ x - H @ ((H @ k)*(H @ x))/d)))
assert error < 1e-12
Q = H / np.sqrt(d)
a,b = rng.normal(size=(2,d))
A,B = (Q*a)@Q,(Q*b)@Q
assert np.allclose(A,A.T) and np.allclose(A@B,B@A)
assert np.allclose(np.diag(A),a.mean())
assert np.allclose(A@B,(Q*(a*b))@Q)

# XOR-only affine Boolean gates cannot implement AND.
truth = [(u,v,u & v) for u,v in itertools.product([0,1],repeat=2)]
assert not any(all((p*u)^(q*v)^b == y for u,v,y in truth)
               for p,q,b in itertools.product([0,1],repeat=3))

result = {
    "environment": {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__},
    "seed": 20261029,
    **checks,
    "random_panel_steps": panel_steps,
    "random_panel_certificate_hits": panel_hits,
    "random_panel_refreshes": panel_misses,
    "resume_checks": "deep-copied complete state after step 250; subsequent outputs matched",
    "flip_reflip_net_passes": net_passes,
    "flip_reflip_cumulative_passes_without_reanchoring": cumulative_passes,
    "walsh_convolution_max_abs_error": error,
    "result": "all assertions passed",
    "scope": "algebra and Python state semantics only; no M3, native kernel, trained model, or performance validation",
}
Path(__file__).with_name("mechanism_check_results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
```
