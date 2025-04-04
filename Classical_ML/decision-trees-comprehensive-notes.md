# Decision Trees: Comprehensive Technical Notes

## 1. Intuition and Mathematical Foundations

### 1.1 Core Intuition
Decision trees are hierarchical structures that partition the feature space into regions, making predictions based on the region in which a new instance falls. The model creates a flowchart-like structure where:
- Internal nodes represent tests on features
- Branches represent outcomes of tests
- Leaf nodes represent class labels or regression values

Decision trees work by recursively splitting the data into increasingly homogeneous subsets with respect to the target variable.

### 1.2 Mathematical Foundations

#### Information Theory Approach
Decision trees rely heavily on information theory concepts to quantify the quality of splits:

**Entropy**: Measures the impurity or uncertainty in a set of examples:
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$
where $p_i$ is the proportion of class $i$ in set $S$, and $c$ is the number of classes.

**Information Gain**: Quantifies the reduction in entropy achieved by a split:
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$
where $S_v$ is the subset of $S$ for which attribute $A$ has value $v$.

#### Alternative Split Criteria

**Gini Impurity**: Measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the distribution of classes in the subset:
$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

**Variance Reduction**: Used in regression trees, measures the reduction in variance after a split:
$$Var_{reduction} = Var(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Var(S_v)$$

### 1.3 Tree Construction Algorithm (ID3/C4.5/CART)

1. **Base cases**: 
   - If all samples belong to the same class, create a leaf node with that class
   - If no features remain, create a leaf node with the most common class
   - If no samples remain, create a leaf node with the most common class from the parent

2. **Recursive case**:
   - Select the best feature $A$ based on the chosen split criterion
   - Create a decision node based on feature $A$
   - For each value $v$ of $A$:
     - Create a subset $S_v$ of samples where $A = v$
     - If $S_v$ is empty, add a leaf node with the most common class
     - Otherwise, recursively build a subtree for $S_v$

## 2. Implementation Details

### 2.1 Feature Selection Process

1. **Candidate Split Generation**:
   - For categorical features: Consider all possible values
   - For numerical features: Either use all unique values or a discretized subset of values
   - Binary splits: Find optimal threshold to split data into two subsets

2. **Split Evaluation**:
   - Calculate the selected criterion (Information Gain, Gini Index, etc.) for each candidate split
   - Choose the split that optimizes the criterion

3. **Multi-way vs. Binary Splits**:
   - ID3: Multi-way splits for categorical features
   - CART: Binary splits only (more efficient, handles numerical data better)
   - C4.5: Improved ID3 with ability to handle continuous attributes and missing values

### 2.2 Pruning Techniques

**Pre-pruning (Early stopping)**:
- Stop growing the tree when information gain is below a threshold
- Limit maximum tree depth
- Require minimum samples for a split
- Require minimum samples in a leaf node

**Post-pruning**:
- Reduced Error Pruning: Remove nodes that don't improve validation accuracy
- Cost-Complexity Pruning (Minimal Cost-Complexity Pruning):
  1. Grow a full tree
  2. Calculate the cost complexity measure for each node:
     $$R_\alpha(T) = R(T) + \alpha |T|$$
     where $R(T)$ is the misclassification rate, $|T|$ is the number of leaf nodes, and $\alpha$ is the complexity parameter
  3. Iteratively collapse the node with the smallest increase in error
  4. Select the subtree with the best performance on validation data

### 2.3 Handling Special Cases

**Continuous Features**:
- Sort values and consider midpoints between adjacent values as potential thresholds
- Select threshold with best split criterion value

**Missing Values**:
- C4.5 approach: Weight instances with missing values across all branches
- Surrogate splits: Use correlated features when primary feature value is missing
- Imputation: Replace missing values with mean/median/mode before training

**Multi-class Problems**:
- Direct extension of binary classification
- For each split, consider all classes when calculating impurity measures

## 3. Key Hyperparameters

### 3.1 Tree Structure Parameters

| Parameter | Description | Effect on Model |
|-----------|-------------|----------------|
| Max Depth | Maximum depth of the tree | Controls complexity; lower values reduce overfitting but may increase bias |
| Min Samples Split | Minimum samples required to split a node | Higher values prevent learning noise patterns |
| Min Samples Leaf | Minimum samples required in a leaf node | Higher values create more balanced trees |
| Max Features | Maximum number of features to consider for best split | Introduces randomness, reduces overfitting |
| Min Impurity Decrease | Minimum decrease in impurity required for a split | Prevents splits that don't significantly improve purity |

### 3.2 Split Criterion Parameters

| Parameter | Options | Trade-offs |
|-----------|---------|------------|
| Split Criterion | Gini, Entropy (classification); MSE, MAE (regression) | Entropy is more computationally intensive but may be more precise; Gini tends to isolate the most frequent class |
| Splitter | Best, Random | "Best" finds optimal splits; "Random" introduces randomness |

### 3.3 Pruning Parameters

| Parameter | Description | Effect on Model |
|-----------|-------------|----------------|
| Cost Complexity (α) | Penalizes tree complexity | Higher values lead to smaller trees |
| Pruning Method | Reduced Error Pruning, Cost-Complexity Pruning | Different trade-offs between accuracy and model size |

## 4. Strengths, Weaknesses, and Appropriate Use Cases

### 4.1 Strengths

- **Interpretability**: Provides clear decision rules that can be easily visualized and explained
- **Non-parametric**: Makes no assumptions about data distribution
- **Handles mixed data types**: Can process both categorical and numerical features
- **Minimal preprocessing**: No need for feature scaling or normalization
- **Feature importance**: Automatically provides feature importance rankings
- **Handles missing values** (with appropriate techniques)
- **Fast training and prediction**: Computationally efficient

### 4.2 Weaknesses

- **Instability**: Small variations in data can lead to completely different trees
- **Overfitting**: Prone to capturing noise, especially with deep trees
- **Bias for high-cardinality features**: Favors features with many unique values
- **Limited expressiveness**: Struggles with XOR-like problems and diagonal decision boundaries
- **Poor performance on imbalanced data**: Biased toward majority classes
- **Lack of smoothness**: Decision boundaries are axis-parallel, creating blocky decision regions
- **Difficulty capturing feature interactions**: May require very deep trees

### 4.3 Appropriate Use Cases

**Well-suited for**:
- Problems requiring interpretable models (medical diagnosis, credit scoring)
- Mixed categorical and numerical data
- Initial exploration of feature importance
- Small to medium-sized datasets
- Non-linear relationships that follow axis-parallel decision boundaries
- Base models for ensembles (Random Forests, Gradient Boosting)

**Less suitable for**:
- High-dimensional datasets with complex relationships
- Problems where smooth decision boundaries are crucial
- Highly unstable problems where small data changes shouldn't drastically affect predictions
- Applications requiring highly stable probability estimates

## 5. Computational Complexity

### 5.1 Time Complexity

**Training**:
- Best case: O(n × m × log(n)), where n is the number of samples, m is the number of features
- Worst case: O(n × m × n) ≈ O(n² × m) when the tree becomes unbalanced
- Finding the best split: O(n × m) for each node
- Sorting features (for continuous variables): O(n × log(n))

**Prediction**:
- Best case (balanced tree): O(log(n))
- Worst case (degenerate tree): O(n)
- Average case for pruned trees: O(log(n))

### 5.2 Space Complexity

- Model storage: O(nodes) ≈ O(n) in worst case
- Training memory: O(n × m) for storing the dataset

### 5.3 Optimizations

- Pre-sorting features to avoid repeated sorting during split finding
- Histogram-based approximations for continuous features
- Parallel computation of feature splits
- Memory-efficient implementations for large datasets

## 6. Common Evaluation Metrics

### 6.1 Classification Metrics

**Accuracy**: Proportion of correct predictions
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision**: Proportion of true positives among positive predictions
$$Precision = \frac{TP}{TP + FP}$$

**Recall (Sensitivity)**: Proportion of true positives correctly identified
$$Recall = \frac{TP}{TP + FN}$$

**F1-Score**: Harmonic mean of precision and recall
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**ROC AUC**: Area under the Receiver Operating Characteristic curve
- Measures discrimination ability across all possible thresholds

### 6.2 Regression Metrics

**Mean Squared Error (MSE)**:
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Root Mean Squared Error (RMSE)**:
$$RMSE = \sqrt{MSE}$$

**Mean Absolute Error (MAE)**:
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**R² (Coefficient of Determination)**:
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

### 6.3 Model-Specific Evaluation

**Tree-specific metrics**:
- Tree depth
- Number of nodes
- Number of leaves
- Average path length

**Cross-validation**: Essential for assessing stability and generalization
- k-fold cross-validation is recommended due to decision trees' variance

**Learning curves**: Plot training and validation error vs. training set size
- Helps diagnose bias-variance tradeoff

## 7. Advanced Topics and Extensions

### 7.1 Oblique Decision Trees

- Standard trees make splits parallel to feature axes
- Oblique trees allow splits along linear combinations of features:
  $$\sum_{i=1}^{m} w_i x_i > threshold$$
- Better captures diagonal decision boundaries
- Higher computational complexity but potentially more accurate

### 7.2 Fuzzy Decision Trees

- Allow partial membership in multiple leaf nodes
- Smoother decision boundaries
- Final prediction is weighted average across multiple paths

### 7.3 Incremental Learning

- Techniques for updating trees with new data without complete retraining
- Challenging due to the hierarchical nature of trees
- Usually involves node splitting/merging strategies

### 7.4 Decision Tree Ensembles

- Random Forests: Bagging of trees trained on bootstrap samples with random feature subsets
- Gradient Boosting: Sequential training of trees to correct errors of previous models
- These methods overcome many individual tree limitations at the cost of interpretability
