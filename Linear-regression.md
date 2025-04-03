# Linear Regression - Comprehensive Technical Notes

## 1. Intuition and Mathematical Foundations

### Core Intuition
Linear regression models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data. The fundamental assumption is that the relationship between variables can be approximated by a straight line (in simple linear regression) or a hyperplane (in multiple linear regression).

### Mathematical Representation

#### Simple Linear Regression
For a single feature variable, the model is expressed as:
$$y = \beta_0 + \beta_1 x + \varepsilon$$

Where:
- $y$ is the dependent variable (target)
- $x$ is the independent variable (feature)
- $\beta_0$ is the y-intercept (bias term)
- $\beta_1$ is the slope coefficient
- $\varepsilon$ is the error term (residuals)

#### Multiple Linear Regression
For multiple features, the model expands to:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \varepsilon$$

In vector form:
$$y = X\beta + \varepsilon$$

Where:
- $y$ is the vector of dependent variables
- $X$ is the matrix of independent variables (with a column of 1's for the intercept)
- $\beta$ is the vector of coefficients
- $\varepsilon$ is the vector of error terms

### Loss Function
Linear regression uses the Sum of Squared Residuals (SSR) or Mean Squared Error (MSE) as its loss function:

$$SSR = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip}))^2$$

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Parameter Estimation

#### Ordinary Least Squares (OLS)
The most common approach to estimate parameters is Ordinary Least Squares, which minimizes the sum of squared differences between observed and predicted values.

For simple linear regression, the closed-form solutions are:

$$\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

For multiple linear regression, the matrix solution is:

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

#### Gradient Descent
When datasets are large or when using regularization, parameters can be estimated iteratively using gradient descent:

**Step 1**: Initialize parameters $\beta_0, \beta_1, ..., \beta_n$ (typically to zeros or small random values)

**Step 2**: For each parameter $\beta_j$, update using the gradient of MSE:
$$\beta_j := \beta_j - \alpha \cdot \frac{\partial}{\partial \beta_j} MSE$$

Where the partial derivative of MSE with respect to $\beta_j$ is:
$$\frac{\partial}{\partial \beta_j} MSE = -\frac{2}{n} \sum_{i=1}^{n} x_{ij}(y_i - \hat{y}_i)$$

**Step 3**: Repeat Step 2 until convergence (when parameters change less than a predefined threshold)

Here, $\alpha$ is the learning rate that controls the step size of each iteration.

## 2. Implementation in Detail

### Algorithm Steps
1. **Data Preparation:**
   - Feature scaling/normalization
   - Handling missing values
   - Creating polynomial features if needed
   - Splitting data into training and test sets

2. **Model Training:**
   - Calculate the coefficients using OLS or gradient descent
   - For OLS: $\hat{\beta} = (X^TX)^{-1}X^Ty$
   - For gradient descent: iteratively update parameters

3. **Prediction:**
   - For new data points, apply the learned linear function: $\hat{y} = X\beta$

### Pseudocode for OLS Implementation
```
function LinearRegression(X, y):
    // Add column of ones to X for intercept
    X = add_column_of_ones(X)
    
    // Calculate coefficients using normal equation
    beta = inverse(X^T * X) * X^T * y
    
    return beta

function Predict(X, beta):
    // Add column of ones to X for intercept
    X = add_column_of_ones(X)
    
    // Calculate predictions
    y_pred = X * beta
    
    return y_pred
```

### Pseudocode for Gradient Descent Implementation
```
function LinearRegressionGD(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    // Add column of ones to X for intercept
    X = add_column_of_ones(X)
    n_samples = X.rows
    n_features = X.columns
    
    // Initialize parameters
    beta = zeros(n_features)
    
    // Gradient descent
    for i = 1 to max_iterations:
        // Calculate predictions
        y_pred = X * beta
        
        // Calculate error
        error = y_pred - y
        
        // Calculate gradient
        gradient = (1/n_samples) * X^T * error
        
        // Update parameters
        beta_new = beta - learning_rate * gradient
        
        // Check convergence
        if norm(beta_new - beta) < tolerance:
            break
            
        beta = beta_new
    
    return beta
```

## 3. Key Hyperparameters

### Regularization Parameters
1. **Ridge Regression (L2):**
   - Adds penalty term: $\lambda \sum_{j=1}^{p} \beta_j^2$
   - $\lambda$ controls strength of regularization
   - Higher $\lambda$ reduces overfitting but may increase bias
   - Solution: $\hat{\beta} = (X^TX + \lambda I)^{-1}X^Ty$

2. **Lasso Regression (L1):**
   - Adds penalty term: $\lambda \sum_{j=1}^{p} |\beta_j|$
   - Promotes sparsity (feature selection)
   - No closed-form solution, uses coordinate descent

3. **Elastic Net:**
   - Combines L1 and L2 penalties: $\lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2$
   - Balances feature selection and coefficient shrinking

### Optimization Parameters
1. **Learning Rate ($\alpha$):**
   - Controls step size in gradient descent
   - Too large: may diverge
   - Too small: slow convergence

2. **Polynomial Degree:**
   - When using polynomial features: $y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_d x^d$
   - Higher degree captures more complex relationships but risks overfitting

3. **Convergence Tolerance:**
   - Threshold for stopping iterative optimization
   - Smaller values lead to more precise solutions but longer training time

## 4. Strengths, Weaknesses, and Use Cases

### Strengths
1. **Interpretability:** Coefficients have clear meaning, showing the effect of each feature on the target variable
2. **Efficiency:** Fast training and prediction, low computational overhead
3. **Well-studied:** Robust theoretical foundation and well-understood statistical properties
4. **Baseline Model:** Excellent starting point for more complex analyses
5. **Feature Importance:** Naturally provides information about feature significance

### Weaknesses
1. **Linearity Assumption:** Cannot capture non-linear relationships without transformation
2. **Outlier Sensitivity:** Highly influenced by outliers due to squared error
3. **Multicollinearity Issues:** Unstable when features are highly correlated
4. **Limited Complexity:** May underfit complex patterns in data
5. **Independence Assumption:** Assumes errors are independent and normally distributed

### Appropriate Use Cases
1. **Predictive Analytics:** Forecasting quantitative outcomes in business, finance, etc.
2. **Economic Modeling:** Understanding relationships between economic variables
3. **Quality Control:** Identifying factors affecting product quality
4. **Scientific Research:** Finding correlations between variables in experimental data
5. **Baseline Modeling:** Establishing performance benchmarks before trying complex models
6. **Feature Selection:** Using regularized approaches to identify important predictors

### When Not to Use
1. **Highly Non-linear Relationships:** When data shows clear non-linear patterns
2. **Classification Problems:** Without transformation (use logistic regression instead)
3. **Time Series with Autocorrelation:** Standard linear regression assumes independent errors
4. **High-dimensional Data:** Can be unstable with many features (use regularization)

## 5. Implementation Details and Computational Complexity

### Computational Complexity

#### OLS Solution:
- Time Complexity: $O(n d^2 + d^3)$ where $n$ is the number of samples and $d$ is the number of features
  - $O(n d^2)$ for computing $X^TX$
  - $O(d^3)$ for matrix inversion
- Space Complexity: $O(n d + d^2)$

#### Gradient Descent:
- Time Complexity: $O(n d i)$ where $i$ is the number of iterations
- Space Complexity: $O(n d)$

### Memory Requirements
- Must store the design matrix $X$ (n×d)
- For OLS, must store and invert $X^TX$ (d×d)

### Numerical Stability
- OLS can be unstable when features are highly correlated (multicollinearity)
- Solutions:
  - Use pseudoinverse with SVD decomposition
  - Apply regularization
  - Feature selection/dimensionality reduction

### Scalability
- OLS becomes impractical for large datasets due to $O(d^3)$ complexity
- Alternatives for large-scale linear regression:
  - Stochastic Gradient Descent (SGD)
  - Mini-batch Gradient Descent
  - Incremental learning algorithms

## 6. Common Evaluation Metrics

### Regression Performance Metrics

1. **Mean Squared Error (MSE):**
   $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
   - Penalizes larger errors more heavily
   - Same units as target variable squared

2. **Root Mean Squared Error (RMSE):**
   $$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
   - Same units as target variable
   - More interpretable than MSE

3. **Mean Absolute Error (MAE):**
   $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   - Less sensitive to outliers than MSE/RMSE
   - Same units as target variable

4. **Coefficient of Determination (R²):**
   $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
   - Proportion of variance explained by the model
   - Range: [0, 1] for linear regression with intercept
   - Can be negative for poorly fitting models

5. **Adjusted R²:**
   $$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$
   - Adjusts for number of predictors (p)
   - Penalizes unnecessary complexity

### Model Validation Techniques

1. **Train-Test Split:**
   - Randomly divide data into training and test sets
   - Typical split: 70-80% training, 20-30% testing

2. **K-fold Cross-Validation:**
   - Split data into k equal folds
   - Train on k-1 folds, validate on remaining fold
   - Repeat k times, rotating validation fold
   - Average metrics across all folds

3. **Leave-One-Out Cross-Validation:**
   - Special case of k-fold where k=n
   - Computationally expensive but maximizes training data

### Diagnostic Plots

1. **Residual Plot:**
   - Plot residuals vs. predicted values
   - Should show random scatter around zero
   - Patterns indicate violated assumptions

2. **Q-Q Plot:**
   - Checks normality of residuals
   - Points should fall along diagonal line

3. **Scale-Location Plot:**
   - Checks homoscedasticity (constant variance)
   - Should show horizontal line with random scatter

4. **Leverage Plot:**
   - Identifies influential observations
   - High leverage points can disproportionately affect model

## 7. Theoretical Assumptions

Linear regression relies on several key assumptions:

1. **Linearity:** The relationship between features and target is linear
2. **Independence:** Observations are independent of each other
3. **Homoscedasticity:** Error variance is constant across all levels of predictors
4. **Normality:** Residuals are normally distributed
5. **No Multicollinearity:** Predictor variables are not highly correlated
6. **No Endogeneity:** Error terms are uncorrelated with predictor variables

Violations of these assumptions can lead to biased coefficients, incorrect standard errors, and unreliable inference.
