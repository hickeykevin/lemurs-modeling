# Modeling Strategies for Subject-Level Differences and Cold Start Resolution

This document summarizes the brainstormed approaches for modeling longitudinal user data (step counts and self-reported survey labels), with a focus on resolving user-level subjectivity and the cold start problem.

---

## The Core Problem

When predicting self-reported survey scores (e.g., a 1-5 scale) from objective time-series data (e.g., step counts), we must address two forms of individual variation:
1. **Input Variations:** Users have different step baselines (addressed by subject-wise scaling of inputs).
2. **Label Subjectivity:** Users interpret qualitative scales differently (e.g., User A's "low" is 1; User B's is 3).

To capture label subjectivity, we can introduce **User Embeddings** (a mixed-effects modeling equivalent for deep learning) to learn user-specific baseline shifts. However, this creates a **Cold Start Problem**: new/unseen users at inference/test time will not have pre-trained embedding vectors.

---

## Brainstormed Architectures & Cold-Start Solutions

### 1. Label Preprocessing: Z-Scoring Labels
* **Concept:** Convert target classes into continuous z-scores per user: `y_norm = (y - mean) / std`.
* **Pros:** Simple; pairs naturally with input standardization.
* **Cons:** Changes classification to continuous regression, losing the discrete 1-5 labels and ordinal interpretation.

### 2. Predicting the Delta (Change)
* **Concept:** Predict `Delta_Y = Y_current - Y_previous` instead of the absolute label.
* **Pros:** Inherently cancels out the personal baseline differences.
* **Cons:** Requires a very recent prior survey answer; cannot handle sparse or missing history.

---

### Strategy A: The "Average User" Fallback (Selected First Step)
* **Concept:** Introduce a standard user embedding lookup table, but reserve index `0` for the `"Unknown / Average User"`.
* **How it works:**
  - During training, randomly replace the active user ID with index `0` (e.g., 10% of the time). This trains index `0` to represent the population average.
  - At test time, any unseen user is mapped to index `0`.
* **Pros:** Extremely simple to implement; degrades gracefully.
* **Cons:** The model cannot personalize for new users until the model is retrained with their user ID in the dataset.

---

### Strategy B: Meta-Embeddings from Static Features (Metadata-driven)
* **Concept:** Replace the embedding lookup table with a generator network (MLP) that maps static user demographics or general baseline statistics to an embedding vector.
* **Pros:** Instant adaptation for new users based on demographic similarity.
* **Cons:** Requires metadata collection and formatting.

---

### Strategy C: Amortized / Contextualized Embeddings (History-driven)
* **Concept:** Use a secondary **Context Encoder** network to generate a dynamic user embedding $\mathbf{e}_u$ from their most recent historical data (e.g., last 3-5 days of step/survey pairs).
* **How it works:**
  - Divide a user's data into a **Context Set** and a **Query Set** (target to predict).
  - Pass the context sequences through the time-series encoder, combine with survey scores, and aggregate to produce the embedding $\mathbf{e}_u$.
  - Use $\mathbf{e}_u$ to predict the query sample.
* **Pros:** Purely feed-forward (no test-time backpropagation) and dynamically updates as user behavior changes.
* **Cons:** High dataset and dataloader complexity; harder to train.

---

### Strategy D: Test-Time Calibration (Few-Shot Parameter Optimization)
* **Concept:** Keep the standard user embedding lookup table, but run a tiny optimization step on the device or backend for new users.
* **How it works:**
  - Train the core model normally with embeddings on the training cohort.
  - Initialize new users to a zero vector.
  - Once a new user accumulates enough data (e.g., 5 surveys), freeze all core model layers and run a few gradient descent steps (e.g., Adam) *only* on their individual embedding vector.
* **Pros:** Highly personalized mathematically.
* **Cons:** Requires backend infrastructure to run backpropagation per user at runtime and manage user profile vectors.
