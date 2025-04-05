# Principal Component Analysis (PCA)

## 1. Intuition and Mathematical Foundations

### Intuition
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. The core idea is to find a new coordinate system (principal components) where:

- The first principal component (PC) captures the direction of maximum variance in the data
- Each subsequent component captures the maximum remaining variance while being orthogonal to all previous components
- Lower-variance components (which often represent noise) can be discarded with minimal information loss

PCA essentially reveals the internal structure of data by identifying the directions (principal components) along which the data varies most.

### Mathematical Foundations

PCA is based on several key mathematical concepts:

1. **Variance and Covariance**: PCA aims to maximize variance along principal components.
2. **Eigenvectors and Eigenvalues**: Principal components are eigenvectors of the covariance matrix.
3. **Orthogonality**: Principal components are mutually orthogonal (perpendicular).
4. **Linear Transformations**: PCA performs a linear mapping from original to new feature space.

Mathematically, given a dataset X with n samples and p features, PCA finds a transformation matrix W that maps X to a new space Y:

$$Y = XW$$

Where:
- X is the original data matrix (n × p)
- W is the transformation matrix (p × k), where k ≤ p is the desired dimensionality
- Y is the transformed data (n × k)

### Algorithm Implementation (Theoretical)

1. **Data Preprocessing**:
   - Center the data by subtracting the mean of each feature:
     $$X_{centered} = X - \bar{X}$$
   - (Optional) Scale each feature to unit variance (standardization):
     $$X_{standardized} = \frac{X_{centered}}{\sigma_X}$$

2. **Compute the Covariance Matrix**:
   $$C = \frac{1}{n-1} X_{centered}^T X_{centered}$$
   
   Where C is a p × p matrix where each element $C_{ij}$ represents the covariance between features i and j.

3. **Eigendecomposition of the Covariance Matrix**:
   - Find the eigenvalues and eigenvectors of C:
     $$C\vec{v} = \lambda\vec{v}$$
   - Sort the eigenvalues in descending order and arrange the corresponding eigenvectors
   - The eigenvectors are the principal components, and eigenvalues represent the variance explained by each principal component

4. **Feature Transformation**:
   - Select the top k eigenvectors (principal components) to form the transformation matrix W
   - Project the data onto the new space:
     $$Y = X_{centered} \cdot W$$

#### Alternative Implementation: Singular Value Decomposition (SVD)

PCA can also be implemented using Singular Value Decomposition (SVD), which is often more numerically stable:

$$X_{centered} = U\Sigma V^T$$

Where:
- U is an n × n matrix (left singular vectors)
- Σ is an n × p diagonal matrix of singular values
- V is a p × p matrix (right singular vectors)

The principal components are the columns of V, and the eigenvalues of the covariance matrix are proportional to the squares of the singular values (diagonal elements of Σ).

## 2. Key Hyperparameters

### Number of Components (k)
- **Definition**: The number of principal components to retain in the dimensionality reduction.
- **Effect**: Controls the tradeoff between dimensionality reduction and information preservation.
- **Selection methods**:
  - Based on explained variance ratio (e.g., retain components that explain 95% of variance)
  - Scree plot analysis (looking for "elbow" in variance plot)
  - Based on application requirements or computational constraints

### Standardization
- **Definition**: Whether to standardize features to unit variance before applying PCA.
- **Effect**: 
  - When True: Gives equal importance to features regardless of their scale
  - When False: Features with larger variances will influence PCA more
- **Considerations**: Standardization is generally recommended when features have different units or scales

### SVD Solver (Implementation-specific)
- **Definition**: The algorithm used to compute the SVD.
- **Options**: Full SVD, randomized SVD, truncated SVD
- **Effect**: Affects computational efficiency and accuracy, especially for high-dimensional data

### Whiten
- **Definition**: Whether to rescale the principal components to have unit variance.
- **Effect**: When enabled, the principal components in the transformed space have equal variance (white noise).

## 3. Strengths, Weaknesses, and Use Cases

### Strengths
- **Dimensionality Reduction**: Effectively reduces high-dimensional data while preserving variance.
- **Noise Reduction**: By discarding lower-variance components, PCA can filter out noise.
- **Feature Decorrelation**: Principal components are uncorrelated, which can improve performance of subsequent algorithms.
- **Data Visualization**: Enables visualization of high-dimensional data in 2D or 3D.
- **Computational Efficiency**: Reduces computational complexity for subsequent analyses.
- **No Hyperparameters**: Core algorithm requires minimal tuning (primarily just selecting k).

### Weaknesses
- **Linear Transformation Only**: Cannot capture non-linear relationships in data.
- **Sensitivity to Outliers**: Outliers can significantly affect principal components.
- **Loss of Interpretability**: Principal components may not have clear physical interpretations.
- **Scale Sensitivity**: Results depend on feature scaling if not standardized.
- **Assumes Orthogonality**: May not be optimal if true data structure isn't orthogonal.
- **Global Optimization**: Optimizes variance globally, which may not preserve local structure.

### Appropriate Use Cases
- **Dimensionality Reduction**: Preprocessing step before applying machine learning algorithms.
- **Data Visualization**: Projecting high-dimensional data to 2D or 3D for visual analysis.
- **Feature Extraction**: Creating uncorrelated features from correlated inputs.
- **Noise Reduction**: Filtering out noise by discarding low-variance components.
- **Image Compression**: Representing images with fewer dimensions.
- **Anomaly Detection**: Identifying outliers in reduced dimensions.
- **Multicollinearity Handling**: Addressing collinearity in regression analyses.
- **Genomics and Bioinformatics**: Analyzing high-dimensional genetic data.

## 4. Common Evaluation Metrics

### Explained Variance Ratio
- **Definition**: Proportion of total variance explained by each principal component.
- **Formula**: $EVR_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}$, where λᵢ is the ith eigenvalue.
- **Cumulative Explained Variance**: Sum of explained variance ratios for the first k components.
- **Usage**: Select k components that explain a sufficient portion of variance (e.g., 95%).

### Reconstruction Error
- **Definition**: Measures information loss when projecting to lower dimensions and back.
- **Formula**: $\|X - X_{reconstructed}\|_F^2$, where $X_{reconstructed} = XWW^T$
- **Frobenius norm**: Measures the total squared difference between original and reconstructed data.
- **Usage**: Lower reconstruction error indicates better preservation of information.

### Kaiser Criterion
- **Definition**: Retain only components with eigenvalues greater than 1.0.
- **Rationale**: Components with eigenvalues < 1 explain less variance than a single original variable.
- **Usage**: Common rule of thumb for selecting number of components.

### Scree Plot Analysis
- **Definition**: Visual examination of eigenvalues plotted in descending order.
- **Usage**: Look for "elbow" point where the curve flattens, indicating diminishing returns from additional components.

### Proportion of Information Loss
- **Definition**: The fraction of total variance not captured by the retained components.
- **Formula**: $1 - \sum_{i=1}^{k} EVR_i$
- **Usage**: Ensure information loss is within acceptable limits for the application.

### Application-Specific Metrics
- **Downstream Task Performance**: Evaluate performance of machine learning models trained on PCA-reduced data.
- **Clustering Quality**: Measure how well clusters separate in reduced dimensions.
- **Classification Accuracy**: Compare accuracy before and after PCA.
- **Computational Efficiency Gains**: Time and memory savings from dimensionality reduction.

## 5. Advanced Considerations

### Kernel PCA
- Extends PCA to capture non-linear relationships using kernel methods.
- Applies PCA in higher-dimensional feature space implicitly defined by a kernel function.

### Incremental PCA
- Processes data in batches, enabling PCA on datasets too large to fit in memory.
- Updates eigenvectors and eigenvalues incrementally as new data arrives.

### Robust PCA
- Variants designed to be less sensitive to outliers.
- Examples include L1-norm PCA and probabilistic PCA.

### Sparse PCA
- Enforces sparsity in the principal components for better interpretability.
- Results in principal components with many zero coefficients.