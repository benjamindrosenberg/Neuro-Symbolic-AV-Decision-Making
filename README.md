# Neuro-Symbolic Decision Making for Autonomous Vehicles
### A Modal Logical Neural Network Applied to Crosswalk Engagement

> **Status:** Accepted to ICMLA 2026.

---

## Overview

This repository contains the full implementation pipeline for a preliminary
application of a Modal Logical Neural Network (MLNN) to automated decision
making in autonomous vehicles (AVs). The work investigates whether a hybrid
neuro-symbolic architecture can learn interpretable, safety-relevant logical
constraints from AV trajectory data — and whether it can do so while remaining
competitive with conventional Multi-Layer Perceptron (MLP) baselines.

The central hypothesis is that interpretability and performance are not in
tension: a network whose architecture encodes modal logical structure should
surface meaningful safety constraints as a natural consequence of training,
rather than as a post-hoc explanation.

---

## Architecture

The pipeline consists of two major components:

### Part 1 — Kinematic Simulator & Dataset Generation

A synthetic crosswalk engagement simulator generates episodic AV trajectory
data. Each episode models an ego vehicle approaching a crosswalk in the presence
of a pedestrian, with stochastic variation across ego speed, pedestrian speed
and category, crosswalk geometry, weather condition, and pedestrian behavior.

Dataset generation was implemented via a self-contained kinematic simulator
rather than a high-fidelity environment (e.g. CARLA) due to resource
constraints. The simulator replicates the predicate and branch structure of
a CARLA-based pipeline, producing an equivalent tensor representation.

**Output tensor:** `(5000 episodes × 3 branches × 30 ticks × 9 predicates)`

| Axis | Size | Contents |
|---|---|---|
| Episodes | 5,000 | One crosswalk engagement per row |
| Branches | 3 | Brake / coast / accelerate counterfactuals |
| Ticks | 30 | Branch rollout timesteps |
| Predicates | 9 | P(t) vector (see below) |

**Predicate set:**

| Index | Predicate | Type |
|---|---|---|
| 0 | DistanceToPedestrian | Continuous |
| 1 | TimeToCollision | Continuous |
| 2 | PedestrianInCrosswalk | Boolean |
| 3 | EgoInCrosswalk | Boolean |
| 4 | Collision | Boolean |
| 5 | EgoBraking | Boolean |
| 6 | EgoAccelerating | Boolean |
| 7 | EgoStopped | Boolean |
| 8 | PedestrianIntentToCross | Boolean |

### Part 2 — Modal Logical Neural Network

The MLNN encodes a clause layer over fuzzy predicate representations, followed
by modal and temporal operators (◇ eventually, □ always, ◇ possibly, □
necessarily) applied across the branch and tick dimensions. A formula head
produces predictions for two safety-critical implications:

- **F2:** □(Collision → ¬◇Accelerate)
- **F3:** □(PedestrianInCrosswalk ∧ LowTTC → ◇Stop)

Training optimizes a composite loss over masked binary cross-entropy, a
contrastive branch loss, and a sparsity penalty on clause weights.

### Part 3 — MLP Baseline Comparison

Three MLP topologies are evaluated against the MLNN on identical formula
targets, loss functions, and training hyperparameters:

| Topology | Hidden Size | Input |
|---|---|---|
| MLP-A | 8 | Episode-level 9-dim modal projection |
| MLP-B | 810 | Episode-level 9-dim modal projection |
| MLP-C | 32 | Full flattened tensor (E × B × T × P) |

---

## Key Findings

The MLNN's learned clause structures consistently emphasized interpretable
safety-relevant predicates — stopping behavior, crosswalk occupancy, collision
conditions, and pedestrian intent — across training runs. The architecture
performed competitively against all three MLP topologies on both formula
targets, supporting the hypothesis that modal logical structure aids rather
than constrains learning in safety-critical domains.

---

## Dependencies
"""
numpy
tqdm
torch
matplotlib
"""

Install via:

```bash
pip install numpy tqdm torch matplotlib
```

The notebook is designed to run in Google Colab or any local Jupyter
environment. Set the environment variable `MLNN_OUTPUT_DIR` to specify an
output directory (defaults to the current working directory; Colab users
should set to `/content`).

---

## Usage

Open `MLNN-MLP_FullPipeline_v2.ipynb` and run cells top-to-bottom. No manual
data preparation is required — the simulator generates the dataset in Part 1,
which is consumed directly by the MLNN in Part 2.

"""
[INSTALL] Install dependencies
[CONFIG] Set output directory
[SIM] Define kinematic simulator
[RUN] Generate 5,000 episodes
[TENSOR] Validate and assemble mlnn_tensor
[MLNN-UTILS] Fuzzy logic primitives and modal operators
[MLNN-MODEL] MLNN class definition
[MLNN-TRAIN] Training loop
[MLNN-RULES] Rule extraction and satisfaction report
[MLNN-RUN] Execute training and extract constraints
[MLNN-SUPPL] Predicate-clause matrix and template summary
[MLP] Baseline comparison — three topologies
"""

---

## Repository Structure
"""
/
├── MLNN-MLP_FullPipeline_v2.ipynb ← Full pipeline (simulator + MLNN + MLP)
├── README.md
└── paper/
├── [paper].pdf
└── [slides].pdf (To be added after presentation, 10/07/2026)
"""

---

## Citation

B. Rosenberg and V. Pentsos, "Interpretable Safety Constraint Inference in
Autonomous Driving using Modal Logical Neural Networks," in
*Proc. IEEE Int. Conf. Mach. Learn. Appl. (ICMLA)*, 2026,
> pp. (to appear).

---

## Notes

This work represents a **preliminary application** of the MLNN architecture
to AV decision making. The simulator, while structurally equivalent to a
CARLA-based pipeline, operates under kinematic simplifications. Results
should be interpreted in this context. Full methodological detail is
available in the accompanying paper.
