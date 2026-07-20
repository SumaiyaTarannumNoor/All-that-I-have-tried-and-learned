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
