# MuistoVaihde — The Memory Switchyard

**Traffic changes the tracks through which later traffic travels.** A deliberately small simulation of phase-indexed ring memories, coincidence learning, associative completion, and a third ring that records joint events. It is **not** an artificial brain, a new plasticity rule, or proof of autonomous clock-free computation.

- **Ring A:** 5 labelled cells; input cue.
- **Ring B:** 7 labelled cells; input during training, reconstructed target during recall.
- **Junction:** one bounded strength for each `(A-cell, B-cell)` pair — **35 weights**.
- **Ring C:** 35 labelled cells; stores learned *joint A/B meetings*, not the raw A or B patterns.

## Try it

Open **[`index.html`](index.html)** in any modern browser directly from disk: no install, server, API, GPU, or external packages. In order:

1. Click **Teach X ×10** (A `{1}`, B `{2,5}`).
2. Click **Teach Y ×10** (A `{3}`, B `{1,4}`). The 5×7 matrix now has four highlighted entries.
3. Click **Recall X**. Only A `{1}` is presented; B `{2,5}` is written over one 35-tick scan.
4. Click **Recall Y**. It reconstructs B `{1,4}` instead.
5. Click **Bind X → C**. The learned co-occupied meetings write two distinct cells of ring C.

Use **Load X pair**, **Step** and **Play** to inspect the actual events. Click cells on the rings to change bits. Change learning strength, time decay and gate threshold with the sliders. **Idle 5000 ticks** shows that learned connections decay even when no crossings occur.

Python alternative (standard library only):

```sh
python muistovaihde.py
python -m unittest -v
```

`legacy_rings.py` preserves Fable's original two-ring, five/seven-cell transfer toy. The new simulation uses a different, explicitly named **copy/completion** rule for associative recall, rather than falsely describing it as a conserved bit transfer.

## The rules

Let `a = (phase_A - t) mod 5` and `b = (phase_B - t) mod 7` be the labelled cells currently at the shared junction. A 35-step full cycle visits each possible `(a,b)` pair exactly once because 5 and 7 are coprime.

**Teach** (only when *both* cells hold a bit):

```text
Every tick: W[a,b] ← (1 − decay) W[a,b]     for ALL 35 pairs
If A[a] = 1 and B[b] = 1:
    W[a,b] ← W[a,b] + η (1 − W[a,b])
```

**Recall** (weights frozen except their physical-time decay):

```text
If A[a] = 1 and W[a,b] ≥ threshold:
    B[b] ← 1
```

**Bind** (weights frozen except physical-time decay):

```text
If A[a] = B[b] = 1 and W[a,b] ≥ threshold:
    C[c] ← 1
```

Here the C-ring position `c` follows the pair by the Chinese remainder correspondence:

```text
c mod 5 = a
c mod 7 = b
```

There is exactly one such `c` in `[0,35)`. Ring C rotates synchronously with its initial phase calibrated from A and B, so this address remains valid if the rings are phase-reset. **C is a record of the conjunction.** It does not create information from nothing.

Default values are `η=0.30`, `decay=0.0002/tick`, `threshold=0.55`. Every cell and weight is an *engineered classical state*. The discrete simulation uses a **shared tick**; geometry alone has not removed the need for timing.

## What was actually measured

Local run on November 12, 2026:

| Check | Result |
| --- | --- |
| Complete 5×7 meeting cycle | Exactly 35 unique pairs |
| Train with only A, or only B | No learning |
| Teach X and Y; probe A-only cues | X→`{2,5}` and Y→`{1,4}` |
| Probe untouched A cell | No B output |
| Phase-offset probes | Same labelled associations for all 5×7 resets |
| Bind X into C | Joint records `(1,2)` and `(1,5)` |
| Silent idle | Learned weights decay without crossings |
| Five distinct one-cell cues | Five stored associations recalled in this test |
| One cue linked to two separate episodes | **Fails to distinguish them**: returns their union |

**7 Python tests pass** locally; the offline browser was also clicked through in Chromium with no page errors. This is a proof of the *implemented rules*, not an independent discovery of useful AI learning: the weight matrix is explicitly designed as an associative lookup table.

### Essential limitation: pairwise correlation is not episode binding

Fable's crucial correction was to learn only on **two-sided coincidences**. A bit in A alone meets all seven B positions every 35 steps. Learning on A alone would therefore waste the 5×7 table and become just five A strengths.

But another limit remains: if A and B contain persistent bit sets throughout a full cycle, their experienced pairs are **the Cartesian product `A × B`**. The table cannot learn which specific co-occupied B bit belonged to which historical *episode*. Two stories with the same A cue and different B targets get merged. Even the five-cue capacity test uses distinct A labels, so it does not establish general temporal sequence learning or compositional intelligence. This is the next serious bottleneck.

## Where to take it next

1. Add a third contextual cue or temporal event marker and test **episode separation when A is identical**, against a matched lookup-table baseline.
2. Introduce real temporal separation, stochastic bit loss and independent timing perturbations; test recovery rather than replaying the training schedule.
3. Only then replace the hand-written junction gate with the original **Geometric Neuron**'s live `setpoint_mod` response. Its native dynamics may or may not implement this behavior.

## Lineage

The conceptual lineage includes [PerceptionLab](https://github.com/anttiluode/PerceptionLab), [GeometricNeuronOriginReview](https://github.com/anttiluode/GeometricNeuronOriginReview), [GeometricNeuronSurfaceListener (V26)](https://github.com/anttiluode/GeometricNeuronSurfaceListener), [GeometricNeuronV27](https://github.com/anttiluode/GeometricNeuronV27), [TurntableNeuron](https://github.com/anttiluode/TurntableNeuron), [BrainRailRoad](https://github.com/anttiluode/BrainRailRoad), and the user's **Slider2 Memory Lab**. `legacy_rings.py` comes from Fable's November 12 five/seven-cell ring experiment.

Related established ideas include delay-line memory, associative memories/Hebbian plasticity, residue number representations and polychronization. The combination here is an experimental construction, not a priority or novelty claim.
