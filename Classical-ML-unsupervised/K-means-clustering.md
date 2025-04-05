# K-means Clustering Algorithm

## 1. Intuition and Mathematical Foundations

### Intuition
K-means clustering is an unsupervised learning algorithm that partitions data into K distinct, non-overlapping clusters. The core idea is to group similar data points together based on their feature similarity, specifically their Euclidean distance from cluster centers.

The algorithm works by:
1. Placing K centroids randomly in the feature space
2. Assigning each data point to the nearest centroid
3. Recalculating the centroids based on the mean of all points assigned to that cluster
4. Repeating steps 2-3 until convergence (minimal change in centroid positions)

### Mathematical Foundations

Given a set of observations (x₁, x₂, ..., xₙ), K-means aims to partition the n observations into K sets S = {S₁, S₂, ..., Sₖ} to minimize the within-cluster sum of squares (inertia):

$$\text{minimize} \sum_{i=1}^{K} \sum_{x \in S_i} \|x - \mu_i\|^2$$

Where:
- μᵢ is the mean of points in cluster Sᵢ
- ‖x - μᵢ‖² is the squared Euclidean distance between point x and centroid μᵢ

### Algorithm Implementation (Theoretical)

1. **Initialization**: 
   - Select K initial centroids (μ₁, μ₂, ..., μₖ) randomly from the data points
   - Alternative initialization: K-means++ (weighted probability selection)

2. **Assignment Step**:
   - For each data point x, assign it to the cluster with the nearest centroid:
   $$S_i^{(t)} = \{x : \|x - \mu_i^{(t)}\|^2 \leq \|x - \mu_j^{(t)}\|^2 \text{ for all } j \neq i\}$$
   - Where Sᵢ⁽ᵗ⁾ is the set of points assigned to cluster i at iteration t

3. **Update Step**:
   - Recalculate centroids as the mean of all points in each cluster:
   $$\mu_i^{(t+1)} = \frac{1}{|S_i^{(t)}|} \sum_{x \in S_i^{(t)}} x$$

4. **Convergence**:
   - Repeat steps 2-3 until either:
     - The centroids no longer change significantly
     - A maximum number of iterations is reached
     - The change in inertia falls below a threshold

## 2. Key Hyperparameters

### Number of Clusters (K)
- **Definition**: The number of clusters to form.
- **Effect**: Directly determines the granularity of the clustering solution.
- **Selection methods**:
  - Elbow method (plotting inertia vs. K)
  - Silhouette analysis
  - Gap statistic
  - Domain knowledge

### Initialization Method
- **Random initialization**: Randomly select K data points as starting centroids.
- **K-means++**: Selects initial centroids with probability proportional to their distance from previously selected centroids.
- **Effect**: Proper initialization can lead to faster convergence and help avoid poor local minima.

### Maximum Iterations
- **Definition**: The maximum number of iterations before forced termination.
- **Effect**: Prevents infinite loops but may terminate before convergence if set too low.

### Convergence Threshold (Tolerance)
- **Definition**: The minimum change in centroids required to continue iterations.
- **Effect**: Lower values ensure more precise centroid locations but may require more iterations.

### Distance Metric
- **Euclidean distance** (standard): Effective for compact, isotropic clusters.
- **Manhattan distance**: Alternative for certain applications.
- **Effect**: The choice of distance metric can significantly affect clustering results depending on the data structure.

## 3. Strengths, Weaknesses, and Use Cases

### Strengths
- **Simplicity**: Easy to understand and implement.
- **Scalability**: Linear time complexity O(n×K×d×i) where n is the number of samples, K is the number of clusters, d is the dimensionality, and i is the number of iterations.
- **Efficiency**: Works well with large datasets.
- **Adaptability**: Can be modified for different distance metrics and initialization strategies.

### Weaknesses
- **Sensitive to initialization**: Results depend on initial centroid placement.
- **Requires predefined K**: The number of clusters must be specified in advance.
- **Assumes spherical clusters**: Performs poorly with non-globular cluster shapes.
- **Sensitive to outliers**: Centroids can be heavily influenced by extreme values.
- **Struggles with varying densities**: Cannot handle clusters of different sizes and densities well.
- **Local optima**: May converge to suboptimal solutions.

### Appropriate Use Cases
- **Customer segmentation**: Grouping customers with similar purchasing behaviors.
- **Image compression**: Reducing color palette in images.
- **Document clustering**: Grouping similar documents for information retrieval.
- **Anomaly detection**: When used with other techniques.
- **Feature learning**: As a preprocessing step for other algorithms.
- **Market segmentation**: Identifying distinct market segments.

## 4. Common Evaluation Metrics

### Internal Evaluation Metrics
- **Inertia (Within-cluster Sum of Squares)**: 
  - Lower values indicate tighter, more compact clusters.
  - Formula: $\sum_{i=1}^{K} \sum_{x \in S_i} \|x - \mu_i\|^2$

- **Silhouette Coefficient**:
  - Measures how similar a point is to its own cluster compared to other clusters.
  - Range: [-1, 1] where higher values indicate better clustering.
  - Formula: $(b - a) / \max(a, b)$ where a is the mean intra-cluster distance and b is the mean nearest-cluster distance.

- **Calinski-Harabasz Index (Variance Ratio Criterion)**:
  - Ratio of between-cluster dispersion to within-cluster dispersion.
  - Higher values indicate better clustering.

- **Davies-Bouldin Index**:
  - Average similarity between each cluster and its most similar cluster.
  - Lower values indicate better clustering.

### External Evaluation Metrics (when ground truth is available)
- **Adjusted Rand Index (ARI)**:
  - Measures similarity between true labels and clustering assignments.
  - Range: [-1, 1] where 1 indicates perfect agreement.

- **Normalized Mutual Information (NMI)**:
  - Measures the mutual information between true labels and clustering assignments.
  - Range: [0, 1] where 1 indicates perfect agreement.

- **Homogeneity, Completeness, and V-measure**:
  - Homogeneity: Each cluster contains only members of a single class.
  - Completeness: All members of a given class are assigned to the same cluster.
  - V-measure: Harmonic mean of homogeneity and completeness.

### Practical Validation Techniques
- **Elbow method**: Plot inertia against K and look for the "elbow" point.
- **Silhouette analysis**: Visualize silhouette coefficients for different K values.
- **Cross-validation**: Validate stability of clusters across different data subsets.
- **Stability analysis**: Assess cluster stability by rerunning with different initializations.