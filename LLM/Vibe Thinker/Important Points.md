## SFT - Supervised-Fine Tuning
## MGPO - MaxENT-Guided Policy Optimization
A smart training method ( a form of Reinforcement Learning) used to teach the AI how to solve hard math and coding Problems.
# Self-Distillation

Self-distillation is a training technique where a single AI model acts as both the **teacher** and the **student** to improve itself. Instead of relying on a separate external model for guidance, the model uses its own past outputs, deeper layers, or previous versions of itself to refine predictions and learn more effectively.

## How It Compares to Regular Knowledge Distillation

### Traditional Knowledge Distillation
In traditional distillation:

- A large, pre-trained **Teacher** model is used.
- A smaller **Student** model learns from the teacher's outputs.
- The goal is to transfer knowledge from a powerful model to a more efficient one.

### Self-Distillation
In self-distillation:

- The **same model** serves as both teacher and student.
- The model learns from its own predictions, internal representations, or earlier checkpoints.
- No additional teacher model is required.
- The goal is to improve accuracy, efficiency, and generalization while keeping the model architecture unchanged.

## Key Benefits

- Improves model performance without requiring a larger teacher model.
- Enhances generalization to unseen data.
- Reduces the need for additional training resources.
- Can improve robustness and prediction consistency.

## Summary

Self-distillation allows a model to **teach itself** by leveraging its own knowledge during training. Unlike traditional knowledge distillation, which transfers knowledge from a separate teacher model, self-distillation uses the model's own outputs and representations to achieve better performance and learning efficiency.

# Multi-Domain vs. Multimodal

**Multi-domain** and **multimodal** are two different concepts that describe how AI models work with information.

## Multimodal

A **multimodal** model can process and understand **different types of data**.

### Examples of Modalities
- Text
- Images
- Audio
- Video
- Sensor data

### Example
A multimodal AI assistant can:
- Read a text prompt
- Analyze an uploaded image
- Understand spoken audio
- Generate a text response based on all inputs

For example, a user uploads a photo of a plant and asks, *"What species is this?"* The model combines image understanding with language processing to answer the question.

## Multi-Domain

A **multi-domain** model works with the **same type of data** but across **different subject areas, industries, or styles**.

### Examples of Domains
- Medicine
- Finance
- Sports
- Law
- Education
- Cooking

### Example
A multi-domain language model can understand and generate text from:
- Medical journals
- Financial reports
- Sports articles
- Cooking recipes

Although all inputs are text, they come from different knowledge domains.

## Key Difference

| Aspect | Multimodal | Multi-Domain |
|----------|----------|----------|
| Focus | Different data types | Different subject areas |
| Input Examples | Text, images, audio, video | Medical text, legal text, sports text |
| Challenge | Combining information across modalities | Learning knowledge across domains |
| Example Task | Answer a question about an uploaded image | Answer questions from multiple fields of knowledge |

## Summary

- **Multimodal** models handle **different forms of data** such as text, images, and audio.
- **Multi-domain** models handle **different topics or knowledge areas** while often working with the same data type.
- A modern AI system can be both **multimodal** and **multi-domain**, allowing it to understand multiple data formats across a wide range of subjects.


# Cosine Annealing Learning Rate Schedule

A **cosine annealing schedule** is a dynamic learning rate strategy used during neural network training. Instead of keeping the learning rate fixed, it gradually decreases the learning rate following the shape of a **cosine curve**.

## Purpose

The learning rate controls how large a step the model takes when updating its parameters.

A good training process usually needs:

- A **high learning rate** at the beginning for rapid exploration.
- A **low learning rate** near the end for precise fine-tuning.

Cosine annealing provides a smooth transition between these two phases.

---

## How It Works

At the start of training:

- The learning rate is high.
- The model explores many possible solutions.

As training progresses:

- The learning rate gradually decreases.
- Updates become smaller and more stable.

Near the end:

- The learning rate approaches zero.
- The model fine-tunes its parameters carefully.

### Formula

```text
η(t) = η_min + 0.5 × (η_max − η_min) ×
       [1 + cos(π × t / T)]
```

Where:

| Symbol | Meaning |
|----------|----------|
| η(t) | Learning rate at training step t |
| η_max | Initial (maximum) learning rate |
| η_min | Minimum learning rate |
| t | Current training step or epoch |
| T | Total number of training steps or epochs |
| π | Mathematical constant Pi (≈ 3.14159) |

---

## Example

Suppose:

```text
η_max = 0.001
η_min = 0.000001
T = 100 epochs
```

The learning rate changes approximately as follows:

| Epoch | Learning Rate |
|---------|---------------|
| 0 | 0.001000 |
| 25 | 0.000854 |
| 50 | 0.000500 |
| 75 | 0.000147 |
| 100 | 0.000001 |

---

## Visual Shape

```text
Learning Rate
     ^
0.001|*
     | \
     |  \
     |   \
     |    \
     |      \
     |        \
0.000|----------*----------> Epochs
       0       50       100
```

The learning rate starts high and smoothly decreases toward the minimum value.

---

## Why Cosine Annealing Is Effective

### Early Training: Fast Exploration

- Large parameter updates
- Faster convergence
- Better exploration of the solution space

### Late Training: Precise Optimization

- Smaller parameter updates
- Reduced oscillation
- More stable convergence

### Smooth Transition

Unlike step-based schedules that suddenly reduce the learning rate, cosine annealing decreases it gradually, often leading to better optimization and improved final performance.

---

## Cosine Annealing with Warm Restarts

A common extension is **Cosine Annealing Warm Restarts (SGDR)**.

Instead of decreasing only once:

1. Learning rate decreases following a cosine curve.
2. It is reset to a high value.
3. The process repeats.

Benefits:

- Escapes local minima.
- Improves exploration.
- Often produces better final results.

---

## Applications

Cosine annealing is widely used in:

- Large Language Models (LLMs)
- GPT-style Transformers
- Vision Transformers (ViTs)
- Convolutional Neural Networks (CNNs)
- Reinforcement Learning
- Diffusion Models

---

## Summary

Cosine annealing is a learning rate scheduling technique that:

- Starts with a high learning rate.
- Gradually decreases it using a cosine-shaped curve.
- Ends with a very small learning rate for fine-tuning.
- Provides smoother optimization than abrupt learning rate drops.

This allows neural networks to learn quickly during early training and perform precise adjustments during later stages.
```
```


# Pass@1 and Pass@K

**Pass@1** and **Pass@K** are evaluation metrics used to measure the performance of AI models, especially on coding, mathematics, and reasoning benchmarks.

## Pass@1

**Pass@1** measures the probability that the model produces the correct answer on its **first attempt**.

### Example

The model is asked a programming question and generates one solution:

```text
Attempt 1 → Correct
```

Result:

```text
Pass@1 = Success
```

If the first answer is incorrect:

```text
Attempt 1 → Incorrect
```

Result:

```text
Pass@1 = Failure
```

### What It Measures

- First-try accuracy
- Reliability
- Precision of the model's initial response

A high Pass@1 score indicates that the model is likely to provide a correct answer immediately.

---

## Pass@K

**Pass@K** measures the probability that at least **one correct answer** appears among **K generated attempts**.

### Example (K = 5)

```text
Attempt 1 → Incorrect
Attempt 2 → Incorrect
Attempt 3 → Correct
Attempt 4 → Incorrect
Attempt 5 → Incorrect
```

Since one of the five attempts is correct:

```text
Pass@5 = Success
```

### What It Measures

- Ability to explore multiple solutions
- Diversity of reasoning
- Probability of finding a correct answer among several attempts

A high Pass@K score indicates that the model is good at generating multiple candidate solutions, even if its first answer is not always correct.

---

## Simple Analogy

Imagine taking a multiple-choice test.

### Pass@1

You get only one chance:

```text
One guess → Correct = Pass
One guess → Wrong = Fail
```

### Pass@K

You get K chances:

```text
Guess 1 → Wrong
Guess 2 → Wrong
Guess 3 → Correct
```

Because one guess is correct:

```text
Pass@K = Pass
```

---

## Comparison

| Metric | Meaning |
|----------|----------|
| Pass@1 | Correct on the first attempt |
| Pass@K | At least one correct answer within K attempts |
| Focus | Accuracy |
| Focus of Pass@K | Solution diversity and exploration |

---

## Example Scores

| Model | Pass@1 | Pass@10 |
|---------|---------|----------|
| Model A | 80% | 90% |
| Model B | 70% | 95% |

Interpretation:

- Model A is more accurate on the first try.
- Model B is better at generating multiple possible solutions and eventually finding the correct one.

---

## Why It Matters

### Pass@1 is important for:

- Chatbots
- Question answering systems
- Real-time assistants
- Production AI systems

### Pass@K is important for:

- Coding assistants
- Mathematical problem solving
- Research agents
- Systems that can generate multiple candidate solutions

---

## Summary

- **Pass@1** measures whether the AI gets the answer right on its first attempt.
- **Pass@K** measures whether the AI produces at least one correct answer within K attempts.
- High **Pass@1** indicates strong immediate accuracy.
- High **Pass@K** indicates strong exploration and solution-generation ability.
- Both metrics are commonly used to evaluate modern coding and reasoning models.
