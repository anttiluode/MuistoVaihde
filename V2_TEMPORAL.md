# V2: temporal feedback, source separation and a stopping event

**Status (2026-11-12):** deterministic small experiment; see `temporal_loop.py`, `test_temporal_loop.py`, and the offline [`temporal.html`](temporal.html). V1's 5×7 coincidence model in `muistovaihde.py` remains unchanged. This is a **new adjacent experiment**, not an integration with native Geometric Neuron relays.

## Why it is different from V1

V1 learned which A/B *cells co-occur*. It could not learn `P → A → B` because co-occupation across a 35-tick cycle does not encode temporal order.

V2 adds an explicit **one-event temporal trace** for learning successive *observed* events, and a **seven-slot circular delay line** that carries the self-generated successor back to the junction. One symbol/event is allowed in flight at once. The circuit is a discrete simulator with a shared clock and fixed lap delay, not an autonomous oscillator.

The learning rule is an engineered transition memory:

```
if two WORLD observations x at tick t and y at tick t+7:
    W[x,y] ← W[x,y] + 0.40 * (1-W[x,y])
```

The recurrent generation rule is:

```
on event x at junction:
    y = unique strongest successor of W[x,*] if its strength >= 0.55
    send y around a seven-tick loop; its arrival becomes the next event
    on END: send nothing
```

If two successors have exactly equal maximal strength, the junction abstains. It does **not** pick one arbitrarily and claim to have remembered context. World observations override self-generated arrivals, generate a recorded mismatch if they disagree, and **only world observations can modify W**. Merely replaying a prediction never counts as confirming it. Forgetting may be enabled per tick, but the benchmark uses decay=0, so the meaning of a terminal END is not confounded by weight loss.

## Reproduction

No dependencies beyond Python:

```sh
python temporal_loop.py
python -m unittest discover -v
```

Browser: open `temporal.html` from disk. Click **Teach P → A → B → END**, then **Replay from P**, **Cut feedback**, **Teach shuffled order**, **Teach conflicting histories**, and **World says C instead**.

## Observations, not novelty claims

| Intervention | Actual behavior |
| --- | --- |
| Untrained; single world P | Only P (no successor) |
| Teach `P → A → B → END` 8 times | P(world@0) → A(self@7) → B(self@14) → END(self@21), then silence |
| Teach `P → B → A → END` instead | Replays P → B → A → END |
| Cut feedback after P | Only P |
| Disagreeing world C at tick 7 | Predicted A logged as mismatch; world C takes precedence |
| Replay with learning gate enabled | No changes to W, because internal pulses are not observations |
| Train `P A B END` and `Q A C END` | Both replay to A, then abstain: no independent contextual memory yet |

**Interpretation:** This demonstrates a deliberately programmed finite-state sequence learner operating through a physical-in-the-simulation seven-step delay register. Its stored directed transition strengths are an ordinary associative table. It does **not** show an emergent model of the world, biological predictive coding, the activity-dependent slow gear, frequency-specific locks, apical-tuft bursts, or learning directly from the original geometric-neuron dynamics. A direct transition-table baseline is the proper comparison and should behave identically for these tasks.

## Next gate (not done)

Give the device an *independently retained contextual state* set by an earlier event. With two histories `P → A → B → END` and `Q → A → C → END`, test whether the same A now chooses B after P but C after Q **without receiving P or Q again**. First use a conventional context register as the upper-bound control. Only then test whether a slow, perturbable phase ring can supply that state with the same storage budget. Do not infer contextual memory merely by attaching a synchronized circle: in V1, the 35-cell ring C's phase is already fixed by the five- and seven-cell phases.
