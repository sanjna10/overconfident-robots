# overconfident-robots
Evaluating confidence calibration in OpenVLA-7B on LIBERO-Spatial — finding that the model is 3.3× more accurate when uncertain than when confident
# Confidence Calibration in Vision-Language-Action Models
### Evaluating OpenVLA-7B on LIBERO-Spatial

---

## Goal

This project investigates whether **gripper decisiveness** can serve as a reliable confidence signal for human operators of Vision-Language-Action (VLA) models.

VLA models like OpenVLA take camera images and natural language instructions as input and output robot motor commands directly. While powerful, these models give no indication of when they are uncertain — a critical gap for safe human-robot interaction.

**Research Question:** When OpenVLA is confident about a gripper action, is it actually correct? Can we trust its decisiveness as a proxy for uncertainty?

---

## Dataset

**Source:** `HuggingFaceVLA/libero` — the LIBERO robot manipulation benchmark  
**Episode used:** Episode 0 (in-distribution, LIBERO-Spatial suite)  
**Task:** *"put the white mug on the left plate and put the yellow and white mug on the right plate"*

| Property | Value |
|---|---|
| Total frames | 214 |
| Camera views | 2 (front + wrist) |
| Image resolution | 256 × 256 |
| Action space | 7-DoF: [x, y, z, roll, pitch, yaw, gripper] |
| Actions type | Delta movements (relative to current position) |
| Gripper values | -1.0 = OPEN, +1.0 = CLOSED |
| Gripper switch points | t ≈ 40, 94, 161, 207 |

The dataset contains full teleoperation demonstrations — a human expert controlled the robot arm and every frame was recorded as ground truth.

---

## Model

**Model:** `openvla/openvla-7b-finetuned-libero-spatial`  
**Base:** OpenVLA-7B (7.5B parameters, Prismatic VLM architecture)  
**Fine-tuned on:** LIBERO-Spatial task demonstrations  
**Normalization key:** `libero_spatial`

The model takes a camera image + natural language instruction and outputs 7 robot action values in a single forward pass.

---

## Techniques

### 1. Open-Loop Inference
For each of the 214 frames, we feed the ground truth camera image and task instruction to OpenVLA and collect the predicted 7-DoF action. We then compare the predicted action against the ground truth recorded from the human demonstrator.

### 2. Gripper Decisiveness (Confidence Proxy)
Since MC Dropout is ineffective on this model (all 211 dropout layers have rate p=0.0), we use **gripper decisiveness** as a proxy for model confidence:

```
decisiveness = |predicted_gripper_value|   (range: 0.0 to 1.0)

0.0 = model predicted 0.0 → completely undecided between open/closed
1.0 = model predicted ±1.0 → fully committed to one gripper state
```

Frames with decisiveness > 0.5 are classified as **decisive**; ≤ 0.5 as **undecided**.

### 3. Failure Mode Classification
Every frame is classified into one of 4 categories based on decisiveness and correctness:

| | Correct | Wrong |
|---|---|---|
| **Decisive** | `confident_right` ✅ | `overconfident` ❌ |
| **Undecided** | `lucky_right` 🟡 | `uncertain_wrong` 🔵 |

`overconfident` (decisive + wrong) is the most dangerous mode — the model signals certainty while making the wrong decision.

### 4. Switch Point Analysis
We automatically detect all gripper state transitions (OPEN→CLOSED and CLOSED→OPEN) and analyze model behavior in a 10-frame window before and after each switch point. This tests whether confidence changes anticipate or follow gripper transitions.

---

## Results

### Overall Performance — Episode 0 (214 frames)

| Metric | Value |
|---|---|
| Gripper accuracy | **14.0%** |
| Mean position error | 0.288 |
| Mean decisiveness | 0.452 |

### Failure Mode Distribution

| Mode | Frames | % |
|---|---|---|
| `uncertain_wrong` | 93 | 43.5% |
| `overconfident` | 91 | **42.5%** |
| `lucky_right` | 24 | 11.2% |
| `confident_right` | 6 | 2.8% |

### Core Finding — Negative Confidence Calibration

| Condition | Frames | Gripper Accuracy | Position Error |
|---|---|---|---|
| Decisive (dec > 0.5) | 97 | **6.2%** | 0.267 |
| Undecided (dec ≤ 0.5) | 117 | **20.5%** | 0.306 |

**The model is 3.3× more accurate when uncertain than when confident.**  
High decisiveness is an *anti*-confidence signal — it predicts failure rather than success.

### Switch Point Analysis — 10 Episodes, 45 Switch Points

| Metric | Value |
|---|---|
| Mean decisiveness BEFORE switch | 0.464 |
| Mean decisiveness AFTER switch | 0.307 |
| Mean change in decisiveness | **-0.156** (drops after) |
| Mean accuracy BEFORE switch | 39.1% |
| Mean accuracy AFTER switch | 46.2% |

The model becomes **less decisive after** gripper transitions (not before). This means the confidence signal reacts to transitions rather than anticipating them — too late for a human operator to intervene.

### Summary Finding

> OpenVLA-7B fine-tuned on LIBERO-Spatial shows **negative confidence calibration** on its training distribution. The model is overconfident 42.5% of frames, decisively wrong 3.3× more often than it is decisively correct. Confidence peaks before gripper transitions where accuracy is lowest, dropping only after the transition when intervention is too late. Raw gripper decisiveness cannot serve as a trust signal for human operators.

---

## Repository Structure

```
├── VLM2_annotated.ipynb    — fully annotated experiment notebook
├── README.md               — this file
```

---

## Dependencies

```
transformers==4.45.0
timm==0.9.16
diffusers==0.38.0
huggingface-hub==0.36.2
av==12.3.0
lerobot (from source)
torch>=2.0
```

---

## Next Steps

- Attention heatmaps — visualize where the model looks when overconfident vs correct
- Frame-to-frame prediction consistency — alternative confidence measure
- User study — test whether any confidence signal helps humans intervene correctly
- Safety-constrained fine-tuning — reduce overconfidence during RLHF fine-tuning
