PDEBench-Lang: Does Representation Affect Neural Symbolic Pruning and Reasoning?

Course: CSCI 5541 NLP S26

Team Name: Token Efforts

Team Members & Responsibilities

Member

Responsibility

Raghav Anand

Dataset generation & symbolic conversion

Krivan Semlani

Model fine-tuning & pruning evaluation

Peilin Li

Reasoning fidelity analysis & benchmarking

Yit Xiaang Ztang

Cross-dialect evaluation & benchmarking

Abstract

PDEs are the mathematical language used to describe physical phenomena like heat, waves, and fluid flow. Most neural models treat them as black-box computations—we argue they should be read as structured symbolic expressions. PDEBench-Lang tests whether the format a PDE is written in (Postfix, LaTeX, Prefix, or natural language) changes how well a language model can reason about it and narrow down its solution space.

We build on Bhatnagar et al. (2025), challenge their assumption that Postfix is the best format for neural reasoning, and introduce a metric to tell apart genuine structural understanding from lucky guessing. Our dataset spans five PDE families with controlled coefficient sampling.

Motivation and Problem Definition

Partial Differential Equations (PDEs) are the formal language used to describe physical phenomena such as diffusion, wave propagation, fluid flow, and electrostatics. A trained physicist encountering the equation:

$u_{t}=0.5~u_{xx}$

immediately recognises it as a description of diffusion—not by solving it numerically, but by reading its structural composition: the relationship between a first-order time derivative and a second-order spatial derivative. This recognition is purely symbolic.

Recent work has shown that Large Language Models (LLMs) can meaningfully assist in symbolic mathematics. Most existing approaches, however, treat equations as computational objects rather than as structured symbolic language, measuring success by numerical prediction accuracy rather than by whether the model genuinely comprehends symbolic structure.

Central Question: Does the symbolic representation format of a PDE influence how well a neural model can reason about and prune the symbolic solution space?

Background & Novelty

LLM-Guided Symbolic Pruning

Our work builds directly on Bhatnagar et al. (2025), "From Equations to Insights." Symbolic PDE solving is combinatorially explosive. A fine-tuned LLM can predict which operators are likely in the solution, allowing a symbolic solver (like FEX) to restrict its search and achieve a 4-6× speedup.

The gap in current research is the choice of representation format. Bhatnagar et al. chose Postfix (Reverse Polish Notation) on computational grounds (it removes parentheses and is easy to parse). However, they did not evaluate whether Postfix is optimal for neural semantic reasoning.

Our Hypothesis

LLMs are pretrained primarily on LaTeX-formatted mathematics and natural language descriptions—not on Postfix. Forcing Postfix may introduce representational misalignment. We hypothesize that raw LaTeX-style representation may lead to better symbolic pruning and more faithful reasoning than Postfix.

Key Contributions

Four Symbolic Dialects: We compare Postfix, Raw LaTeX, Prefix, and Natural Language.

Explicit Reasoning Chains: We require models to generate structured reasoning chains, not just operator predictions.

Reasoning Fidelity Metric: We introduce a novel metric ("Trash Score") to measure whether explanations are structurally correct.

Methodology

1. Dataset Generation

We construct a synthetic dataset spanning five canonical PDE families: Heat (Diffusion), Wave, Burgers, Laplace, and Advection. For each instance, we randomly sample equation coefficients, generate the symbolic expression, and convert it into all four target representations.

Table 1: The four symbolic dialects used to encode the same PDE $u_{t}=0.5~u_{xx}$

Dialect

Encoding

Postfix (RPN)

u t d 0.5 u x x d d * =

Raw LaTeX

u_{t}=0.5\,u_{xx}

Prefix

= d(u,t) * (0.5, d(d(u,x),x))

Natural Language

"The time derivative of u equals one-half the second spatial derivative of u."

Table 2: Complete Input-Output Data Label Example

Input (e.g. Raw LaTeX)

Equation: $u_{t}=0.5~u_{xx}$

(1) Structural Reasoning

Contains a $1^{st}$-order time derivative; contains a $2^{nd}$-order spatial derivative; structure indicates diffusive transport.

(2) Behavioural Label

Heat / Diffusion

(3) Operator Subset

{exp, polynomial}

2. Model Training

We fine-tune a sequence-to-sequence model (either T5 or BART) on the task of mapping a PDE representation to the three target outputs described in Table 2. Each of the four symbolic representations is trained independently, producing parallel models whose outputs can be compared under controlled conditions.

3. Evaluation

Evaluation proceeds along three primary dimensions. The most novel of these is Reasoning Fidelity. We define a "trash explanation" as an output that arrives at the correct PDE family label via structurally incorrect reasoning.

Table 3: Evaluation Dimensions

Dimension

Metric

Definition

Pruning Quality

Precision, Recall

Overlap of predicted vs. ground-truth operator subset; degree of search-space reduction.

Label Accuracy

Accuracy

Fraction of instances with correct PDE family classification.

Reasoning Fidelity

Trash Score

Rate of outputs with correct label but structurally incorrect reasoning chain.

Expected Contributions

A Controlled Benchmark: Introduces a benchmark for studying representation effects in symbolic PDE reasoning, treating format as an empirical variable rather than a fixed engineering choice.

Cross-Dialect Comparison: Provides the first systematic comparison of LLM-guided operator pruning across four symbolic representations.

Empirical Evidence: Offers concrete evidence on whether computationally convenient representations (Postfix) are semantically optimal, or if representations closer to pretraining distributions yield superior reasoning fidelity.

References

Bhatnagar, A. et al. (2025). From equations to insights. Preprint.

d'Ascoli, S. et al. (2024). ODEFormer: Symbolic regression of dynamical systems. ICLR 2024.

Lample, G., and Charton, F. (2020). Deep learning for symbolic mathematics. ICLR 2020.

Sun, H. et al. (2024). PROSE-PDE: Multimodal learning of governing equations. Preprint.