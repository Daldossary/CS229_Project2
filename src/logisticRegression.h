#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <vector>
using std::vector;
using Matrix = std::vector<std::vector<double>>;

/// LogisticRegression implements binary classification using the logistic (sigmoid) function.
/// The model is trained using gradient descent to minimize the cross-entropy cost function.
/// Given a design matrix X (each row is a feature vector ϕ(x)) and binary labels y, the model
/// predicts:
///
///     p(y = 1|x, w) = σ(wᵀϕ(x))  where  σ(z) = 1/(1 + exp(–z))
///
/// The cross-entropy cost is computed as:
///
///     J(w) = – (1/m) ∑[ y_i log(σ(z_i)) + (1–y_i) log(1–σ(z_i)) ]
///
class LogisticRegression {
public:
    // Constructor. The learning rate, number of maximum iterations, and tolerance for convergence can be specified.
    LogisticRegression(double learning_rate = 0.01, int max_iter = 10000, double tol = 1e-6);

    // Fit the model on design matrix X and binary labels y.
    // X is an m-by-d matrix; y is an m-dimensional vector (with 0 or 1 entries).
    void fit(const Matrix &X, const vector<int> &y);

    // Return the predicted probabilities for each data point in X.
    vector<double> predictProb(const Matrix &X) const;

    // Return binary predictions (0 or 1) for each row in X using a threshold of 0.5.
    vector<int> predict(const Matrix &X) const;

    // Return the learned weight vector.
    vector<double> getWeights() const;

private:
    vector<double> weights_;   // Weight vector.
    double learning_rate_;     // Learning rate for gradient descent.
    int max_iter_;             // Maximum number of iterations.
    double tol_;               // Tolerance for convergence.

    // Compute the logistic (sigmoid) of z.
    double sigmoid(double z) const;

    // Compute the cross-entropy cost over the dataset (used for monitoring convergence).
    double computeCost(const Matrix &X, const vector<int> &y) const;
};

#endif