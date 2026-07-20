## VibeThinker-3B: Explaining the Frontier of Verifiable Reasoning in Small Language Models
In this paper, Sen et al. introduce VibeThinker-3B, a compact dense model with 3B parameters developed to investigate how far verifiable reasoning can be pushed within a strictly small-model regime. In this paper, the authors are trying to answer some questions: **Instead of treating SLMs simply as compute-saving fallbacks, what is their true capability boundary? Can a strictly 3B model actually achieve frontier-level performance comparable to top-tier LLMs??**

# VibeThinker-3B Training Pipeline

VibeThinker-3B uses a **hybrid training strategy** that combines supervised learning, reinforcement learning, teacher distillation, and self-distillation. The model first learns from large, powerful teacher models and later improves itself through self-generated high-quality solutions.

The training process follows the exact sequence below.

---

## Step 1: Supervised Fine-Tuning (SFT)

### Purpose
Teach the model fundamental reasoning and problem-solving patterns.

### Process
- Train on thousands of curated examples.
- Cover domains such as:
  - Mathematics
  - Programming
  - Science
  - Logical reasoning

### Outcome
The model learns how correct solutions are structured and develops a basic understanding of problem-solving techniques.

---

## Step 2: Multi-Domain Reinforcement Learning (RL)

### Purpose
Encourage independent reasoning and exploration.

### Process
- The model attempts problems on its own.
- It generates step-by-step reasoning rather than only final answers.
- A reward is given only when the final answer is correct.

### Characteristics
- Promotes deeper reasoning abilities.
- Improves performance across multiple domains.
- Helps the model discover effective solution strategies.

### Outcome
The model becomes more capable of solving problems independently while developing stronger reasoning chains.

---

## Step 3: Strong Teacher Distillation

### Why It Is Needed

A 3-billion-parameter model has limited capacity compared to very large language models. Some advanced mathematical, scientific, and coding problems require reasoning patterns that a smaller model may not discover by itself.

Large teacher models provide these advanced reasoning paths.

### Process
- Powerful teacher models generate high-quality solutions.
- Teachers reveal hidden intermediate reasoning steps.
- The smaller model learns by imitating these solutions.

### Benefits
- Transfers advanced reasoning strategies.
- Improves performance on difficult tasks.
- Allows the small model to acquire capabilities beyond what it could independently discover.

### Outcome
The model learns superior problem-solving habits from larger, more capable systems.

---

## Step 4: Offline Self-Distillation (Final Phase)

### Purpose
Consolidate and refine learned capabilities.

### Process
1. The model generates multiple solutions on its own.
2. High-quality solutions are selected.
3. Incorrect or weak outputs are filtered out.
4. The model retrains using its best self-generated examples.

### Benefits
- Reinforces successful reasoning patterns.
- Reduces dependence on external teacher models.
- Improves consistency and generalization.
- Helps preserve learned skills without memorizing vast amounts of additional data.

### Outcome
The model effectively becomes its own teacher, strengthening and stabilizing the knowledge acquired during previous training stages.

---

## Complete Training Flow

```text
Supervised Fine-Tuning (SFT)
            ↓
Multi-Domain Reinforcement Learning (RL)
            ↓
Strong Teacher Distillation
            ↓
Offline Self-Distillation
```

---

## Summary

VibeThinker-3B follows a four-stage training pipeline:

1. **Supervised Fine-Tuning (SFT)** builds foundational knowledge.
2. **Multi-Domain Reinforcement Learning (RL)** develops independent reasoning skills.
3. **Strong Teacher Distillation** transfers advanced reasoning from powerful teacher models.
4. **Offline Self-Distillation** enables the model to refine itself using its own best solutions.

This hybrid approach allows a relatively small 3B-parameter model to achieve strong reasoning performance by combining external guidance with self-improvement.
