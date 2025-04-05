# K-Nearest Neighbors (KNN) Algorithm

## Introduction
K-Nearest Neighbors (KNN) is a simple, versatile, and widely-used supervised machine learning algorithm that can be applied to both classification and regression tasks. It operates on the principle of similarity, assuming that similar data points are located close to each other in the feature space. KNN is a non-parametric, lazy learning algorithm, meaning it doesn't make assumptions about the underlying data distribution and defers computation until prediction time.

---

## Intuition and Mathematical Foundations

### Intuition
The core idea of KNN is straightforward: to predict the label or value of a new data point, look at the "k" closest data points (neighbors) in the training set and use their labels/values to make a decision. For classification, it’s typically a majority vote among the neighbors, while for regression, it’s an average of their values. The "closeness" is determined using a distance metric, usually Euclidean distance.

- **Example**: Imagine you’re trying to classify a fruit as an apple or orange based on its size and color. You plot all known fruits in a 2D space (size vs. color). For a new fruit, you find the 3 nearest fruits (k=3) and see that 2 are apples and 1 is an orange. By majority vote, you classify the new fruit as an apple.

### Mathematical Foundations
KNN relies on distance metrics to measure similarity between data points. Given two points \( x = (x_1, x_2, ..., x_n) \) and \( y = (y_1, y_2, ..., y_n) \) in an n-dimensional space, common distance metrics include:

**Euclidean Distance (most common):**

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

**Manhattan Distance:**

$$
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
$$

**Minkowski Distance (generalization):**

$$
d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}
$$

When \( p = 2 \), it’s Euclidean; when \( p = 1 \), it’s Manhattan.


For classification:
- The algorithm identifies the \( k \) nearest neighbors and assigns the class with the most votes.
- Ties can be broken randomly or by weighting votes by distance.

For regression:
- The algorithm computes the mean (or weighted mean) of the \( k \) nearest neighbors’ target values.

---

## Implementation in Detail (Theoretical)

### Steps of KNN
1. **Prepare the Data**:
   - Normalize or standardize features to ensure all dimensions contribute equally to the distance calculation (e.g., height in meters vs. weight in kilograms would otherwise skew results).
   - Handle missing values if present.

2. **Choose a Value for \( k \)**:
   - Select the number of neighbors to consider (e.g., \( k = 3, 5 \)).

3. **Calculate Distances**:
   - For a test point, compute the distance to every point in the training set using the chosen metric.

4. **Identify \( k \) Nearest Neighbors**:
   - Sort the distances in ascending order and select the \( k \) points with the smallest distances.

5. **Make a Prediction**:
   - **Classification**: Perform a majority vote among the \( k \) neighbors’ labels.
   - **Regression**: Compute the average (or weighted average) of the \( k \) neighbors’ values.

6. **Evaluate and Iterate**:
   - Test the model on validation data and adjust \( k \) or the distance metric if needed.

### Theoretical Considerations
- **Storage**: KNN stores the entire training dataset, as it’s a lazy learner.
- **Complexity**:
  - Training: \( O(1) \) (no explicit training phase).
  - Prediction: \( O(n \cdot m) \), where \( n \) is the number of training samples and \( m \) is the number of features, due to distance calculations for each test point.

---

## Key Hyperparameters and Their Effects

1. **\( k \) (Number of Neighbors)**:
   - **Effect**: Controls the bias-variance trade-off.
     - Small \( k \) (e.g., 1): Low bias, high variance; model is sensitive to noise (overfitting).
     - Large \( k \): High bias, low variance; model smooths over local patterns (underfitting).
   - **Tuning**: Typically chosen via cross-validation; odd values are preferred for classification to avoid ties.

2. **Distance Metric**:
   - **Effect**: Determines how "closeness" is measured.
     - Euclidean: Works well for continuous, isotropic data.
     - Manhattan: Better for grid-like or sparse data.
     - Custom metrics: Can be tailored to domain-specific needs.
   - **Tuning**: Depends on the data’s structure and feature scales.

3. **Weighting Scheme**:
   - **Options**: Uniform (all neighbors contribute equally) vs. Distance-weighted (closer neighbors have more influence, e.g., weight = \( 1/d \)).
   - **Effect**: Distance weighting reduces the impact of distant neighbors, potentially improving accuracy.

---

## Strengths, Weaknesses, and Appropriate Use Cases

### Strengths
- **Simplicity**: Easy to understand and implement.
- **No Training Phase**: Adapts instantly to new data.
- **Versatility**: Works for both classification and regression.
- **Non-parametric**: Makes no assumptions about data distribution, effective for complex patterns.

### Weaknesses
- **Computational Cost**: Slow at prediction time, especially with large datasets, as it requires calculating distances to all training points.
- **Memory Intensive**: Stores the entire dataset.
- **Sensitive to Noise**: Outliers can heavily influence predictions, especially with small \( k \).
- **Curse of Dimensionality**: Performance degrades in high-dimensional spaces unless dimensionality reduction is applied.

### Appropriate Use Cases
- **Good For**:
  - Small to medium-sized datasets with low dimensionality.
  - Problems where local patterns are meaningful (e.g., image recognition, recommendation systems).
  - Baseline models for comparison.
- **Not Ideal For**:
  - Large datasets (due to computational inefficiency).
  - High-dimensional data without preprocessing.
  - Real-time applications requiring fast predictions.

---

## Common Evaluation Metrics

### For Classification
1. **Accuracy**: Proportion of correct predictions.
   - \( \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}} \)
2. **Precision, Recall, F1-Score**: Useful for imbalanced datasets.
   - Precision: \( \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} \)
   - Recall: \( \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \)
   - F1: \( 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \)
3. **Confusion Matrix**: Visualizes true vs. predicted labels.

### For Regression
1. **Mean Squared Error (MSE)**:
   - \( \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \)
2. **Mean Absolute Error (MAE)**:
   - \( \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \)
3. **R-squared**: Measures how well the model explains variance in the data.
   - \( R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} \)

---

## Conclusion
KNN is an intuitive and effective algorithm for tasks where similarity-based reasoning is key. Its simplicity makes it an excellent starting point, but its computational inefficiency and sensitivity to noise and dimensionality require careful preprocessing and parameter tuning. By understanding its mathematical foundations, hyperparameters, and limitations, one can leverage KNN effectively in appropriate scenarios while avoiding its pitfalls.

---