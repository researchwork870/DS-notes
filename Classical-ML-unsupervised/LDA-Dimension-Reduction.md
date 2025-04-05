# Linear Discriminant Analysis (LDA)

## 1. Intuition and Mathematical Foundations

### Intuition
Linear Discriminant Analysis (LDA) is both a classification algorithm and a dimensionality reduction technique that projects high-dimensional data onto a lower-dimensional space while maximizing class separability. The core idea behind LDA is to find a linear combination of features that:

1. Maximizes the distance between the means of different classes (between-class variance)
2. Minimizes the variance within each class (within-class variance)

Unlike PCA which focuses solely on maximizing variance, LDA specifically aims to find directions that best discriminate between classes. This makes LDA a supervised technique, as it uses class label information in determining the optimal projection.

### Mathematical Foundations

LDA is based on several key statistical concepts:

1. **Between-class scatter matrix (S_B)**: Measures the distance between class means
2. **Within-class scatter matrix (S_W)**: Measures the scatter of samples around their respective class means
3. **Generalized eigenvalue problem**: Used to find the optimal projection directions

For a problem with C classes, LDA aims to find a projection matrix W that maximizes the ratio of between-class scatter to within-class scatter:

$$J(W) = \frac{W^T S_B W}{W^T S_W W}$$

The optimal projection directions are the eigenvectors corresponding to the largest eigenvalues of $S_W^{-1}S_B$.

### Algorithm Implementation (Theoretical)

#### Step 1: Compute class statistics
- For each class c:
  - Calculate mean vector μₖ of all samples in class c
  - Calculate the number of samples Nₖ in class c
- Calculate the overall mean μ of the entire dataset

#### Step 2: Compute scatter matrices
- Within-class scatter matrix:
  $$S_W = \sum_{k=1}^{C} \sum_{i \in \text{class } k} (x_i - \mu_k)(x_i - \mu_k)^T$$

- Between-class scatter matrix:
  $$S_B = \sum_{k=1}^{C} N_k (\mu_k - \mu)(\mu_k - \mu)^T$$

#### Step 3: Solve the generalized eigenvalue problem
- Find eigenvalues and eigenvectors of $S_W^{-1}S_B$:
  $$S_W^{-1}S_B w = \lambda w$$

- Sort eigenvalues in descending order and collect corresponding eigenvectors
- The maximum number of useful eigenvectors is min(p, C-1), where p is the number of features and C is the number of classes

#### Step 4: Form the projection matrix
- Select the top d eigenvectors to form the projection matrix W (d ≤ C-1)
- The projection matrix W has dimensions p × d, where p is the original feature dimensionality

#### Step 5: Project the data
- Transform the original data using the projection matrix:
  $$Y = XW$$

#### Step 6: Classification (for LDA as a classifier)
- For a new sample x, calculate the discriminant function for each class:
  $$\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(P(k))$$
  
  Where P(k) is the prior probability of class k
  
- Assign x to the class with the highest discriminant score

## 2. Key Hyperparameters

### Number of Components
- **Definition**: The dimensionality of the projection space (d).
- **Constraints**: Must be ≤ min(number of features, number of classes - 1).
- **Effect**: 
  - Larger values preserve more class-discriminative information but may include noise.
  - Smaller values create more compact representations but might lose important discriminative information.
- **Selection**: Often chosen based on cumulative explained variance ratio or cross-validation performance.

### Solver Method
- **Definition**: The algorithm used to compute the eigendecomposition.
- **Options**:
  - SVD (Singular Value Decomposition)
  - Eigenvalue decomposition
  - LSQR (Least Squares QR decomposition)
- **Effect**: Impacts numerical stability and computational efficiency, especially with high-dimensional data.

### Shrinkage Parameter
- **Definition**: Regularization parameter that shrinks the class covariance matrices toward a shared covariance matrix.
- **Range**: [0, 1], where 0 means no shrinkage and 1 means complete shrinkage to shared covariance.
- **Effect**:
  - Helps with ill-conditioned within-class scatter matrices
  - Prevents overfitting when the number of features exceeds the number of samples
  - Improves numerical stability

### Prior Probabilities
- **Definition**: The prior probabilities of the classes.
- **Options**:
  - Empirical (estimated from class frequencies in the training data)
  - Uniform (all classes have equal probability)
  - Custom (user-specified)
- **Effect**: Impacts decision boundaries and classification results, especially for imbalanced datasets.

### Tolerance
- **Definition**: Threshold for singular values to be considered non-zero in SVD solver.
- **Effect**: Controls numerical stability by ignoring components associated with very small eigenvalues.

## 3. Strengths, Weaknesses, and Use Cases

### Strengths
- **Supervised Dimensionality Reduction**: Explicitly uses class information to find discriminative projections.
- **Optimal for Normal Distributions**: Provides optimal decision boundaries when classes follow multivariate normal distributions with equal covariance matrices.
- **Computationally Efficient**: Fast to train and apply, with lower computational complexity than many nonlinear methods.
- **Data Visualization**: Projects high-dimensional data to 2D/3D while preserving class separation.
- **Handles Small Sample Sizes**: Can perform well even with relatively few samples per class (with shrinkage).
- **Feature Extraction**: Creates features that maximize class discrimination.
- **Built-in Dimension Limit**: Automatically limits outputs to C-1 dimensions, potentially avoiding overfitting.

### Weaknesses
- **Linearity Constraint**: Can only find linear boundaries between classes.
- **Gaussian Assumption**: Optimal only when classes follow multivariate normal distributions.
- **Equal Covariance Assumption**: Standard LDA assumes all classes have the same covariance structure.
- **Invertibility Requirement**: Traditional LDA requires that the within-class scatter matrix be invertible.
- **Sensitivity to Outliers**: Outliers can significantly impact class means and covariances.
- **Limited by Class Count**: Can project to at most C-1 dimensions regardless of original data dimensionality.
- **Feature Correlation**: Does not handle multicollinearity well without regularization.

### Appropriate Use Cases
- **Multi-class Classification**: Particularly effective for problems with more than two classes.
- **Text Classification**: For categorizing documents into topics after feature extraction.
- **Face Recognition**: Widely used in facial recognition systems (Fisherfaces approach).
- **Biomedical Classification**: For disease diagnosis based on biomarkers.
- **Financial Analysis**: For credit scoring and fraud detection.
- **Speech Recognition**: For speaker identification and phoneme classification.
- **Preprocessing Step**: As dimensionality reduction before applying other algorithms.
- **High-dimensional, Small Sample Size Problems**: With proper regularization/shrinkage.

## 4. Common Evaluation Metrics

### Classification Performance Metrics
- **Accuracy**: Proportion of correctly classified samples.
- **Precision, Recall, F1-Score**: More nuanced metrics especially for imbalanced classes.
- **Confusion Matrix**: Visualization of classification performance across all classes.
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve (for binary classification).
- **Log Loss**: Measures probabilistic predictions quality.

### Dimensionality Reduction Quality Metrics
- **Explained Variance Ratio**: Proportion of variance explained by each discriminant component.
- **Cumulative Explained Variance**: Sum of explained variance for the selected components.
- **Eigenvalues**: Directly related to the discriminative power of each component.

### Class Separation Metrics
- **Fisher's Criterion**: Ratio of between-class to within-class scatter along projection direction.
- **Silhouette Score**: Measures how well samples are clustered with their own class after projection.
- **Davies-Bouldin Index**: Ratio of within-class scatter to between-class separation.
- **Jeffries-Matusita Distance**: Measures separability between class distributions.

### Cross-validation Performance
- **K-fold Cross-validation**: Assessing generalization performance across different data splits.
- **Learning Curves**: Performance as a function of training set size.
- **Validation Curves**: Performance as a function of hyperparameter values.

## 5. Advanced Variants and Extensions

### Quadratic Discriminant Analysis (QDA)
- Relaxes the equal covariance assumption of LDA
- Allows each class to have its own covariance matrix
- Results in quadratic decision boundaries

### Regularized Discriminant Analysis (RDA)
- Introduces regularization to handle ill-conditioned covariance matrices
- Interpolates between LDA and QDA using shrinkage parameter

### Kernel LDA
- Extends LDA to handle nonlinear class separations
- Applies the kernel trick to implicitly map data to higher-dimensional space
- Enables finding nonlinear discriminant functions

### Flexible Discriminant Analysis (FDA)
- Generalizes LDA by replacing linear regression with nonparametric regression
- Allows for more flexible decision boundaries

### Heteroscedastic LDA
- Accounts for different class covariance structures
- Better handles cases where the equal covariance assumption is violated

### Sparse LDA
- Introduces sparsity constraints on the discriminant vectors
- Improves interpretability and reduces overfitting

### Incremental LDA
- Updates the LDA model incrementally as new samples arrive
- Useful for large datasets or online learning scenarios