# ChangeVLN

**A Benchmark for Vision-Language Navigation in Changed Environments**

ChangeVLN evaluates how Vision-Language Navigation (VLN) agents behave when the environment no longer matches the conditions assumed by the original navigation instruction.

The project was developed as an undergraduate research project and was accepted to the **IEEE/RSJ IROS 2026 Workshop on Semantic and Metric Navigation (SeMaNa)**.

**Paper:** [ChangeVLN: A Benchmark for Vision-Language Navigation in Changed Environments](https://openreview.net/forum?id=HaTMBqlK1B)

---

## Motivation

Most VLN benchmarks evaluate agents in environments that remain consistent with the original instruction and reference trajectory.

Real environments, however, can change after an instruction is created. Furniture may be moved, passages may become blocked, and an agent may need to revise its navigation behavior instead of blindly following the original route.

ChangeVLN introduces controlled environmental changes and evaluates whether a VLN agent can respond appropriately to them.

---

## Benchmark Overview

The benchmark is built on a NaVILA-based navigation pipeline in Habitat.

The evaluation process includes:

1. Selecting navigation episodes from the original VLN benchmark.
2. Introducing obstacles that alter the original navigation environment.
3. Reconstructing the navigation mesh after the environmental change.
4. Comparing the original and changed navigation conditions.
5. Evaluating a baseline policy and an obstacle-aware replanning setting.
6. Measuring navigation outcomes across the selected episodes.

A total of **500 episodes** were used in the final evaluation.

- 91 episodes were reused from preliminary experiments.
- 409 additional episodes were selected and executed for the final evaluation.
- Baseline success: **99 / 500**
- Obstacle-aware setting success: **104 / 500**

---

## My Contributions

My work focused on building and validating the changed-environment evaluation pipeline.

- Designed the benchmark setup for evaluating VLN agents under environmental changes.
- Implemented obstacle insertion and environment modification in Habitat.
- Analyzed the original instruction distribution to understand how often obstacle-related situations were represented.
- Built and refined the obstacle-aware prompting / replanning procedure.
- Organized the evaluation pipeline for the 500-episode benchmark.
- Executed baseline and obstacle-aware experiments.
- Implemented scripts for episode-level statistics and final metric calculation.
- Tracked reproducibility information such as episode IDs, obstacle poses, random seeds, and navigation-mesh changes.
- Prepared the experimental results and paper for publication.

---

## Technical Stack

- **Python**
- **Habitat-Lab / Habitat-Sim**
- **NaVILA**
- Vision-Language Navigation
- Vision-Language-Action models
- Navigation evaluation and experiment automation

---

## Research Question

> How robust are VLN agents when the physical environment changes after the navigation instruction has already been generated?

ChangeVLN focuses on **out-of-distribution environmental intervention** rather than replacing the underlying model. This makes it possible to examine whether an existing navigation agent can recognize and respond to changes that are not explicitly represented in the original instruction.

---

## Evaluation Pipeline

```text
Original VLN Episode
        |
        v
Environment Modification
(Obstacle Placement)
        |
        v
Navigation-Mesh Reconstruction
        |
        v
Baseline / Obstacle-Aware Evaluation
        |
        v
Trajectory & Episode Metrics
        |
        v
Final Benchmark Analysis
```

---

## Publication

**ChangeVLN: A Benchmark for Vision-Language Navigation in Changed Environments**

- Authors: **Sohee Kang, Nuri Kim**
- Venue: **IEEE/RSJ IROS 2026 Workshop on Semantic and Metric Navigation (SeMaNa)**
- Status: **Accepted**
- Paper: [OpenReview](https://openreview.net/forum?id=HaTMBqlK1B)

---

## Repository Note

This repository is a **public portfolio overview** of the project.

The full research implementation, datasets, model checkpoints, trajectories, and experiment artifacts are not included here. This repository is intended to present the problem definition, evaluation design, technical contributions, and research outcome without publishing internal research files.
