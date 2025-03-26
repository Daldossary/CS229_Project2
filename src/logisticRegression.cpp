#include "logisticRegression.h"
#include <cmath>
#include <iostream>

double LogisticRegression::sigmoid(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

LogisticRegression::LogisticRegression(double learning_rate, int max_iter, double tol)
    : learning_rate_(learning_rate), max_iter_(max_iter), tol_(tol)
{ }

void LogisticRegression::fit(const Matrix &X, const vector<int> &y) {
    size_t m = X.size();
    if (m == 0) return;
    size_t d = X[0].size();
    weights_.assign(d, 0.0); // initialize weights to zeros

    // Gradient descent loop.
    for (int iter = 0; iter < max_iter_; iter++) {
        vector<double> gradients(d, 0.0);
        double max_change = 0.0;

        // Compute prediction and gradient for each data point.
        for (size_t i = 0; i < m; i++) {
            double z = 0.0;
            for (size_t j = 0; j < d; j++) {
                z += X[i][j] * weights_[j];
            }
            double p = sigmoid(z);
            double error = p - y[i];  // error term
            for (size_t j = 0; j < d; j++) {
                gradients[j] += error * X[i][j];
            }
        }
        // Update weights using the average gradient.
        for (size_t j = 0; j < d; j++) {
            double update = learning_rate_ * gradients[j] / m;
            weights_[j] -= update;
            if (std::abs(update) > max_change) {
                max_change = std::abs(update);
            }
        }
        if (max_change < tol_) {
            // Convergence achieved.
            break;
        }
    }
}

vector<double> LogisticRegression::predictProb(const Matrix &X) const {
    size_t m = X.size();
    size_t d = (m > 0) ? X[0].size() : 0;
    vector<double> probs(m, 0.0);
    for (size_t i = 0; i < m; i++) {
        double z = 0.0;
        for (size_t j = 0; j < d; j++) {
            z += X[i][j] * weights_[j];
        }
        probs[i] = sigmoid(z);
    }
    return probs;
}

vector<int> LogisticRegression::predict(const Matrix &X) const {
    vector<double> probs = predictProb(X);
    vector<int> labels(probs.size(), 0);
    for (size_t i = 0; i < probs.size(); i++) {
        labels[i] = (probs[i] >= 0.5) ? 1 : 0;
    }
    return labels;
}

vector<double> LogisticRegression::getWeights() const {
    return weights_;
}