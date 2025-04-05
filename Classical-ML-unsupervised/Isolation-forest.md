**Isolation Forest: Comprehensive Notes**

---

### 1. Intuition and Mathematical Foundations

#### Intuition:
Isolation Forest (iForest) is an unsupervised anomaly detection algorithm based on the premise that anomalies are few and different, and therefore are easier to isolate compared to normal points. Instead of profiling normal data points, it isolates anomalies using random partitions.

Anomalies can be isolated with fewer random splits because they are typically sparse and lie further away from dense clusters of normal points. Hence, the path length from the root of a tree to an anomaly tends to be shorter compared to normal points.

#### Mathematical Foundations:
- The algorithm constructs an ensemble of **Isolation Trees (iTrees)**.
- Each tree is built using the following process:
  1. Randomly select a feature.
  2. Randomly select a split value between the minimum and maximum value of the selected feature.
  3. Repeat this process recursively to partition the dataset until:
     - The data point is isolated (i.e., only one instance in the node), or
     - A maximum tree height is reached (usually log2(n), where n is the subsample size).

- The **path length** of a point is the number of edges traversed from the root to reach a leaf node.
- The average path length **E(h(x))** over a forest of trees is used to compute the anomaly score:

  \[ s(x, n) = 2^{-\frac{E(h(x))}{c(n)}} \]

  where:
  - \( E(h(x)) \) is the average path length of \( x \) across all trees.
  - \( c(n) \) is the average path length of unsuccessful search in a Binary Search Tree, approximated as:

    \[ c(n) = 2H(n-1) - (2(n-1)/n) \]

    and \( H(i) \) is the harmonic number: \( H(i) \approx \ln(i) + \gamma \) (Euler-Mascheroni constant \( \gamma \approx 0.5772 \)).

- The score \( s(x, n) \) ranges between 0 and 1:
  - Values close to 1 indicate anomalies.
  - Values much smaller than 0.5 indicate normal observations.

---

### 2. Key Hyperparameters and Their Impact

1. **n_estimators (Number of Trees):**
   - More trees improve robustness and stability of scores.
   - Trade-off: More trees increase computation time.

2. **max_samples (Subsample Size):**
   - Controls the number of samples used to build each tree.
   - Lower values increase diversity among trees, which can improve anomaly detection.
   - Typical value: 256 (default).

3. **max_features:**
   - Number of features to draw from when looking for the best split.
   - Helps with dimensionality control and can prevent overfitting.

4. **contamination:**
   - Proportion of anomalies in the data.
   - Used to calibrate the threshold for classifying anomalies.
   - Affects the decision function output and final labeling.

5. **max_depth (Tree Depth):**
   - Limits the depth of each isolation tree.
   - Helps prevent overfitting and reduces computational cost.

6. **random_state:**
   - Controls reproducibility of randomness in tree construction.

---

### 3. Strengths, Weaknesses, and Use Cases

#### Strengths:
- **Efficient**: Linear time complexity in the number of samples (O(n log n)).
- **Scalable**: Suitable for large datasets due to subsampling.
- **Model-Free**: Does not assume any distribution for the input data.
- **Effective in High Dimensions**: Performs well when other methods fail due to the curse of dimensionality.

#### Weaknesses:
- **Randomness Sensitivity**: Highly dependent on random splits.
- **Not Ideal for Clustered Anomalies**: Less effective when anomalies form small, dense clusters.
- **No Probabilistic Output**: Output is a score, not a probability.
- **Static Thresholding**: Requires good choice of contamination to determine threshold.

#### Use Cases:
- Fraud detection (e.g., credit card fraud).
- Network intrusion detection.
- Industrial equipment failure prediction.
- Medical anomaly detection (e.g., identifying rare diseases).

---

### 4. Common Evaluation Metrics

Since Isolation Forest is unsupervised, labeled data is not always available. When labels are available:

1. **Precision, Recall, and F1 Score:**
   - Useful when the class distribution is highly imbalanced.
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

2. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
   - Measures the trade-off between true positive rate and false positive rate.
   - AUC closer to 1 indicates better performance.

3. **PR-AUC (Precision-Recall Curve AUC):**
   - More informative when dealing with high class imbalance.

4. **Confusion Matrix:**
   - Helps in understanding the distribution of predictions and errors.

5. **Threshold Tuning (based on contamination):**
   - Evaluate performance at different thresholds for anomaly score.

---

### Summary:
Isolation Forest is a powerful and scalable algorithm for unsupervised anomaly detection. Its core idea of isolating points via random partitioning allows for fast, distribution-free detection of outliers, especially in high-dimensional datasets. Its performance can be significantly affected by the choice of hyperparameters, particularly subsample size and contamination, making thoughtful tuning essential.