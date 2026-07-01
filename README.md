# Neuro-Symbolic Decision Making for Autonomous Vehicles
### A Modal Logical Neural Network Applied to Crosswalk Engagement

> **Status:** Submitted to ICMLA 2026. Paper and presentation materials will be
> added upon acceptance, subject to IEEE self-archiving guidelines.

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
