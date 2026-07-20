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
