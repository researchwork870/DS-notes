# Logistic Regression - Comprehensive Technical Notes

## 1. Intuition and Mathematical Foundations

### Core Intuition
Logistic regression is a statistical model used for binary classification problems. Despite its name, it's a classification algorithm rather than a regression algorithm. It models the probability that a given input belongs to a certain class. The core idea is to use the logistic function (sigmoid) to constrain the output of a linear model to values between 0 and 1, which can be interpreted as probabilities.

### Mathematical Representation

#### The Logistic Function
The logistic (sigmoid) function is defined as:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

This function maps any real-valued number to the range [0,1], making it suitable for modeling probabilities.

#### Model Definition
In logistic regression, we model the probability of the positive class (y = 1) as:

$$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

Where:
- $x$ is the feature vector
- $w$ is the weight vector
- $b$ is the bias term (often absorbed into $w$ by adding a constant feature of 1 to $x$)
- $\sigma$ is the sigmoid function

The probability of the negative class is:

$$P(y=0|x) = 1 - P(y=1|x) = 1 - \sigma(w^T x + b)$$

#### Decision Boundary
For binary classification, the decision boundary is the set of points where:

$$w^T x + b = 0$$

This is a hyperplane in the feature space. The model predicts class 1 when $w^T x + b > 0$ and class 0 when $w^T x + b < 0$.

### Loss Function
Logistic regression uses the negative log-likelihood (also called cross-entropy loss) as its loss function:

$$L(w, b) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

Where:
- $n$ is the number of training examples
- $y_i$ is the true label (0 or 1)
- $p_i = P(y=1|x_i) = \sigma(w^T x_i + b)$ is the predicted probability

This loss function heavily penalizes confident incorrect predictions, and mildly penalizes less confident incorrect predictions.

### Parameter Estimation

#### Maximum Likelihood Estimation
Logistic regression parameters are typically estimated using Maximum Likelihood Estimation (MLE). The goal is to find parameters $w$ and $b$ that maximize the likelihood of observing the training data:

$$\text{maximize} \prod_{i=1}^{n} [p_i^{y_i} \cdot (1-p_i)^{1-y_i}]$$

Taking the logarithm (which maintains the same maximum but is easier to work with):

$$\text{maximize} \sum_{i=1}^{n} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

This is equivalent to minimizing the negative log-likelihood (the loss function defined earlier).

#### Gradient Descent
Unlike linear regression, logistic regression has no closed-form solution. Parameters are estimated using optimization algorithms such as gradient descent. 

The gradient of the loss function with respect to the parameters is:

$$\nabla_w L(w, b) = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) x_i$$
$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)$$

Parameters are updated iteratively:

$$w := w - \alpha \nabla_w L(w, b)$$
$$b := b - \alpha \frac{\partial L}{\partial b}$$

Where $\alpha$ is the learning rate.

## 2. Implementation in Detail

### Algorithm Steps

1. **Data Preparation:**
   - Feature scaling/normalization
   - Handling missing values
   - Feature engineering if needed
   - Splitting data into training and test sets

2. **Model Initialization:**
   - Initialize weights $w$ and bias $b$ (typically to zeros or small random values)

3. **Training Process:**
   - Compute predicted probabilities: $p_i = \sigma(w^T x_i + b)$
   - Calculate loss: $L(w, b) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$
   - Compute gradients of loss with respect to parameters
   - Update parameters using gradient descent or other optimization methods
   - Repeat until convergence or maximum iterations reached

4. **Prediction:**
   - For a new input $x$, calculate $p = \sigma(w^T x + b)$
   - Predict class 1 if $p \geq \text{threshold}$ (typically 0.5), otherwise predict class 0

### Pseudocode for Logistic Regression Implementation

```
function LogisticRegression(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    // Add column of ones to X for bias term
    X = add_column_of_ones(X)
    n_samples = X.rows
    n_features = X.columns
    
    // Initialize parameters
    w = zeros(n_features)
    
    // Gradient descent
    for i = 1 to max_iterations:
        // Calculate predicted probabilities
        z = X * w
        p = sigmoid(z)
        
        // Calculate gradient
        gradient = (1/n_samples) * X^T * (p - y)
        
        // Update parameters
        w_new = w - learning_rate * gradient
        
        // Check convergence
        if norm(w_new - w) < tolerance:
            break
            
        w = w_new
    
    return w

function sigmoid(z):
    return 1 / (1 + exp(-z))

function Predict(X, w, threshold=0.5):
    // Add column of ones to X for bias term
    X = add_column_of_ones(X)
    
    // Calculate probabilities
    z = X * w
    p = sigmoid(z)
    
    // Convert probabilities to class predictions
    y_pred = (p >= threshold)
    
    return y_pred, p
```

### Multiclass Logistic Regression

For problems with more than two classes, two common approaches are:

1. **One-vs-Rest (OvR):** Train K binary classifiers, one for each class against all others.
2. **Multinomial Logistic Regression (Softmax Regression):** Extend the binary model using the softmax function:

$$P(y=k|x) = \frac{e^{w_k^T x}}{\sum_{j=1}^{K} e^{w_j^T x}}$$

The loss function becomes categorical cross-entropy:

$$L(W) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(p_{ik})$$

Where $y_{ik}$ is 1 if example $i$ belongs to class $k$ and 0 otherwise, and $p_{ik}$ is the predicted probability that example $i$ belongs to class $k$.

## 3. Key Hyperparameters

### Regularization Parameters

1. **L2 Regularization (Ridge):**
   - Adds penalty term: $\lambda \sum_{j=1}^{p} w_j^2$
   - Modifies the loss function: $L(w) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(p_i) + (1-y_i) \log(1-p_i)] + \lambda \sum_{j=1}^{p} w_j^2$
   - Parameter $\lambda$ controls regularization strength
   - Higher $\lambda$ values lead to stronger regularization (smaller coefficients)
   - Helps prevent overfitting, especially with correlated features

2. **L1 Regularization (Lasso):**
   - Adds penalty term: $\lambda \sum_{j=1}^{p} |w_j|$
   - Encourages sparse solutions (feature selection)
   - Some coefficients become exactly zero, effectively removing features

3. **Elastic Net:**
   - Combines L1 and L2 penalties: $\lambda_1 \sum_{j=1}^{p} |w_j| + \lambda_2 \sum_{j=1}^{p} w_j^2$
   - Balances feature selection and coefficient shrinking

### Optimization Parameters

1. **Learning Rate ($\alpha$):**
   - Controls step size in gradient descent
   - Too large: may diverge or oscillate
   - Too small: slow convergence
   - Common practice: try a range of values (e.g., 0.001, 0.01, 0.1, 1)

2. **Convergence Tolerance:**
   - Threshold for stopping iterative optimization
   - Smaller values lead to more precise solutions but longer training time

3. **Maximum Iterations:**
   - Upper limit on optimization iterations
   - Prevents excessive computation if convergence is slow

4. **Classification Threshold:**
   - Cutoff probability for classifying as positive class (default: 0.5)
   - Adjusting this threshold affects precision-recall tradeoff
   - Can be optimized based on specific performance metrics (F1-score, etc.)

## 4. Strengths, Weaknesses, and Use Cases

### Strengths

1. **Interpretability:** Coefficients have clear meaning related to log-odds
2. **Efficiency:** Fast training and prediction, low computational overhead
3. **Probabilistic Output:** Naturally provides class probabilities
4. **No Distributional Assumptions:** Unlike discriminant analysis, doesn't assume normal distribution of features
5. **Feature Importance:** Coefficients indicate relative importance of features
6. **Handles Non-linearity:** Through feature transformations and interactions
7. **Well-established Statistical Foundation:** Well-understood theoretical properties

### Weaknesses

1. **Linear Decision Boundary:** Cannot capture complex non-linear relationships without transformation
2. **Feature Independence Assumption:** Performs suboptimally with highly correlated features
3. **Limited Complexity:** May underfit complex patterns in high-dimensional data
4. **Data Separation Issues:** Perfect separation can lead to unstable coefficient estimates
5. **Requires More Data:** For reliable probability estimates compared to discriminative models
6. **Sensitive to Outliers:** Extreme values can significantly impact coefficient estimates

### Appropriate Use Cases

1. **Medical Diagnosis:** Predicting disease presence/absence
2. **Credit Scoring:** Estimating default probability
3. **Spam Detection:** Classifying emails as spam or not spam
4. **Customer Churn Prediction:** Identifying customers likely to leave
5. **Sentiment Analysis:** Classifying text as positive/negative
6. **Risk Assessment:** Estimating probability of risk events
7. **Baseline Modeling:** Establishing performance benchmarks before trying complex models
8. **Feature Selection:** Using regularized approaches to identify important predictors

### When Not to Use

1. **Complex Non-linear Relationships:** When data has complex decision boundaries
2. **High-dimensional Feature Space:** May suffer from curse of dimensionality without regularization
3. **Image/Signal Classification:** Deep neural networks typically perform better
4. **Imbalanced Datasets:** Requires special handling (class weights, sampling techniques)
5. **When Probabilistic Interpretation Isn't Required:** Other models might provide better accuracy

## 5. Implementation Details and Computational Complexity

### Computational Complexity

#### Training:
- Time Complexity: $O(n \cdot d \cdot i)$
  - $n$ is the number of training examples
  - $d$ is the number of features
  - $i$ is the number of iterations
- Space Complexity: $O(n \cdot d)$

#### Prediction:
- Time Complexity: $O(d)$ per instance
- Space Complexity: $O(d)$ for model parameters

### Memory Requirements
- Must store the feature matrix $X$ (n×d)
- Model parameters require $O(d)$ space

### Optimization Techniques

1. **Batch Gradient Descent:**
   - Uses entire dataset for each gradient update
   - Stable convergence but slow for large datasets

2. **Stochastic Gradient Descent (SGD):**
   - Updates parameters using one example at a time
   - Faster but noisier convergence path
   - Time complexity per iteration: $O(d)$

3. **Mini-batch Gradient Descent:**
   - Updates parameters using small batches (e.g., 32, 64, 128 examples)
   - Balance between stability and speed
   - Time complexity per iteration: $O(b \cdot d)$ where $b$ is batch size

4. **Second-Order Methods:**
   - Newton's Method, BFGS, L-BFGS
   - Utilize second derivatives for faster convergence
   - Higher computational cost per iteration but potentially fewer iterations

### Numerical Stability
- Log-sum-exp trick for computing softmax (in multinomial case)
- Using regularization to handle collinearity
- Handling extreme probability values to avoid numerical underflow/overflow

## 6. Common Evaluation Metrics

### Binary Classification Metrics

1. **Confusion Matrix:**
   - True Positives (TP): Correctly predicted positive class
   - True Negatives (TN): Correctly predicted negative class
   - False Positives (FP): Incorrectly predicted as positive
   - False Negatives (FN): Incorrectly predicted as negative

2. **Accuracy:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
   - Overall correctness of the model
   - Misleading for imbalanced datasets

3. **Precision:**
   $$\text{Precision} = \frac{TP}{TP + FP}$$
   - Proportion of positive identifications that are correct
   - Focus: Minimizing false positives

4. **Recall (Sensitivity):**
   $$\text{Recall} = \frac{TP}{TP + FN}$$
   - Proportion of actual positives correctly identified
   - Focus: Minimizing false negatives

5. **F1 Score:**
   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
   - Harmonic mean of precision and recall
   - Balance between precision and recall

6. **Specificity:**
   $$\text{Specificity} = \frac{TN}{TN + FP}$$
   - Proportion of actual negatives correctly identified

7. **ROC Curve and AUC:**
   - ROC: Plot of True Positive Rate vs. False Positive Rate at various thresholds
   - AUC: Area Under the ROC Curve (0.5 = random, 1.0 = perfect)
   - Threshold-invariant performance metric

8. **Precision-Recall Curve:**
   - Plot of Precision vs. Recall at various thresholds
   - Better than ROC for imbalanced datasets

9. **Log Loss (Cross-Entropy):**
   $$\text{Log Loss} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$
   - Evaluates probabilistic predictions
   - Heavily penalizes confident incorrect predictions

### Multiclass Classification Metrics

1. **Confusion Matrix:** K×K matrix for K classes
2. **Accuracy:** Same formula as binary case
3. **Macro-averaging:** Average metrics across all classes with equal weight
4. **Micro-averaging:** Calculate metrics globally by counting total TP, FP, FN
5. **Weighted-averaging:** Average metrics weighted by class support

### Model Validation Techniques

1. **Train-Test Split:**
   - Randomly divide data into training and test sets
   - Typical split: 70-80% training, 20-30% testing

2. **K-fold Cross-Validation:**
   - Split data into k equal folds
   - Train on k-1 folds, validate on remaining fold
   - Repeat k times, rotating validation fold
   - Average metrics across all folds

3. **Stratified K-fold Cross-Validation:**
   - Maintains class distribution in each fold
   - Better for imbalanced datasets

4. **Leave-One-Out Cross-Validation:**
   - Special case of k-fold where k=n
   - Computationally expensive but maximizes training data

## 7. Theoretical Assumptions and Considerations

### Key Assumptions

1. **Independence:** Observations are independent of each other
2. **No Perfect Multicollinearity:** Predictor variables are not perfectly correlated
3. **Linearity of Log-odds:** The log-odds (logit) of the outcome is linearly related to the features
4. **Large Sample Size:** Reliable for moderate to large sample sizes

### Considerations for Real-world Implementation

1. **Dealing with Imbalanced Data:**
   - Class weights
   - Oversampling minority class
   - Undersampling majority class
   - Synthetic minority oversampling (SMOTE)

2. **Handling Missing Values:**
   - Imputation strategies
   - Creating missing value indicators

3. **Feature Scaling:**
   - Standardization (z-score normalization)
   - Min-max scaling
   - Important for gradient-based optimization

4. **Feature Engineering:**
   - Polynomial features
   - Interaction terms
   - Transformation of skewed features

5. **Multicollinearity:**
   - Feature selection
   - Principal Component Analysis
   - Regularization
