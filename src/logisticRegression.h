#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <vector>
using std::vector;
using Matrix = std::vector<std::vector<double>>;

/// LogisticRegression implements binary classification using the logistic (sigmoid) function.
/// The model is trained using gradient descent.
/// Given a design matrix X (with one row per data point and each row as a feature vector ϕ(x))
/// and binary labels y (0 or 1), the probability that y = 1 is modeled as:
///
///  p(y=1 | x, w) = σ(wᵀϕ(x)),   where σ(z) = 1/(1+exp(–z))
///
class LogisticRegression {
public:
    // Constructor. You can provide a learning rate, maximum iterations, and a convergence tolerance.
    LogisticRegression(double learning_rate = 0.01, int max_iter = 10000, double tol = 1e-6);

    // Fit the model given design matrix X and binary labels y.
    // X is an m-by-d matrix, and y is an m-dimensional vector with entries 0 or 1.
    void fit(const Matrix &X, const vector<int> &y);

    // Predict probability estimates for data in X.
    vector<double> predictProb(const Matrix &X) const;

    // Predict binary labels for data in X (using a threshold of 0.5).
    vector<int> predict(const Matrix &X) const;

    // Return the learned weight vector.
    vector<double> getWeights() const;

private:
    vector<double> weights_;  // Learned parameter vector.
    double learning_rate_;    // Learning rate for gradient descent.
    int max_iter_;            // Maximum number of iterations.
    double tol_;              // Convergence tolerance.

    // Logistic (sigmoid) function.
    double sigmoid(double z) const;
};

#endif