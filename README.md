# Data_Collection

# GRPO Data Collection & Training Pipeline (LabOS-style)

## Overview

This guide explains how to collect data and convert it into a format
suitable for GRPO (Group Relative Policy Optimization), inspired by the
LabOS paper.

GRPO differs from supervised fine-tuning (SFT): - You do NOT store model
outputs - You store inputs + evaluation signals - The model generates
responses during training

------------------------------------------------------------------------

## 1. Data Collection Pipeline

### 1.1 Record Videos

-   Use head-mounted cameras or smart glasses
-   Capture real-world lab environments
-   Duration: 2--45 minutes

Output:

    video_001.mp4

------------------------------------------------------------------------

### 1.2 Create Gold-Standard Protocol

Example:

    {
      "protocol": [
        "Wear gloves",
        "Prepare solution",
        "Heat to 80C",
        "Mix reagent A"
      ]
    }

------------------------------------------------------------------------

### 1.3 Step Segmentation

Annotate steps with timestamps:

    {
      "steps": [
        {"text": "Wear gloves", "start": 0.0, "end": 12.5},
        {"text": "Prepare solution", "start": 12.5, "end": 60.0}
      ]
    }

------------------------------------------------------------------------

### 1.4 Error Annotation

Example:

    {
      "errors": [
        {"type": "sterile_breach", "time": 34.2},
        {"type": "step_mismatch"}
      ]
    }

------------------------------------------------------------------------

### 1.5 Entity Annotation

Example:

    {
      "entities": {
        "materials": ["reagent A", "beaker"],
        "parameters": {"temperature": "80C"}
      }
    }

------------------------------------------------------------------------

## 2. Convert to GRPO Dataset Format

Each entry should be JSONL:

Example:

    {
      "messages": [
        {"role": "user", "content": "<video>Describe the procedure. Output in <answer> tags."}
      ],
      "video": "/data/video_001.mp4",
      "protocol": ["wear gloves", "heat solution"],
      "required_safety": ["gloves"]
    }

Rules: - Last message must be user - No assistant responses - Extra
fields are passed to reward function

------------------------------------------------------------------------

## 3. Reward Function Design

### Protocol Matching

    def reward_protocol(output, protocol):
        return sum(1 for step in protocol if step in output.lower()) / len(protocol)

### Safety Compliance

    def reward_safety(output, required_safety):
        return sum(1 for s in required_safety if s in output.lower())

### Error Detection

    def reward_errors(output, errors):
        return sum(1 for e in errors if e["type"] in output.lower())

### Combined Reward

    def reward_fn(output, protocol=None, required_safety=None, errors=None):
        score = 0
        if protocol:
            score += reward_protocol(output, protocol)
        if required_safety:
            score += reward_safety(output, required_safety)
        if errors:
            score += reward_errors(output, errors)
        return score

------------------------------------------------------------------------

## 4. Training Loop (GRPO)

For each sample: 1. Generate multiple outputs (e.g., 8--32) 2. Compute
reward for each output 3. Rank outputs within the group 4. Update model
using relative advantage

------------------------------------------------------------------------

## 5. Key Differences from SFT

  SFT             GRPO
  --------------- --------------------
  Needs answers   No answers
  Static labels   Dynamic generation
  Loss-based      Reward-based

------------------------------------------------------------------------

## 6. Best Practices

-   Focus on high-quality annotations
-   Use structured reward signals
-   Avoid ambiguous ground truth
-   Include edge cases and failures

------------------------------------------------------------------------

## 7. Minimal Example

    {"messages":[{"role":"user","content":"<video>Describe procedure"}],
     "video":"/data/v1.mp4",
     "protocol":["wear gloves","heat solution"]}

    {"messages":[{"role":"user","content":"<video>Find errors"}],
     "video":"/data/v2.mp4",
     "errors":[{"type":"sterile_breach"}]}

------------------------------------------------------------------------

## Summary

GRPO training requires: - Inputs (prompt + video/image) - Structured
evaluation signals - A reward function

It does NOT require: - Ground-truth answers - Pre-generated responses
