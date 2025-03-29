#include "logisticRegression.h"
#include <cmath>
#include <iostream>

double LogisticRegression::sigmoid(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

LogisticRegression::LogisticRegression(double learning_rate, int max_iter, double tol)
    : learning_rate_(learning_rate), max_iter_(max_iter), tol_(tol) { }

double LogisticRegression::computeCost(const Matrix &X, const vector<int> &y) const {
    size_t m = X.size();
    double cost = 0.0;
    for (size_t i = 0; i < m; i++) {
        double z = 0.0;
        for (size_t j = 0; j < X[i].size(); j++) {
            z += X[i][j] * weights_[j];
        }
        double p = sigmoid(z);
        const double epsilon = 1e-10;
        cost += y[i] * std::log(p + epsilon) + (1 - y[i]) * std::log(1 - p + epsilon);
    }
    return -cost / m;
}

void LogisticRegression::fit(const Matrix &X, const vector<int> &y) {
    size_t m = X.size();
    if (m == 0) return;
    size_t d = X[0].size();
    weights_.assign(d, 0.0);
    cost_history_.clear();

    double prev_cost = 1e12;
    for (int iter = 0; iter < max_iter_; iter++) {
        vector<double> gradients(d, 0.0);
        for (size_t i = 0; i < m; i++) {
            double z = 0.0;
            for (size_t j = 0; j < d; j++) {
                z += X[i][j] * weights_[j];
            }
            double p = sigmoid(z);
            double error = p - y[i];
            for (size_t j = 0; j < d; j++) {
                gradients[j] += error * X[i][j];
            }
        }
        double max_update = 0.0;
        for (size_t j = 0; j < d; j++) {
            double update = learning_rate_ * gradients[j] / m;
            weights_[j] -= update;
            if (std::abs(update) > max_update) {
                max_update = std::abs(update);
            }
        }
        double cost = computeCost(X, y);
        cost_history_.push_back(cost);
        if (std::abs(prev_cost - cost) < tol_) {
            break;
        }
        prev_cost = cost;
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
    vector<int> predictions(probs.size(), 0);
    for (size_t i = 0; i < probs.size(); i++) {
        predictions[i] = (probs[i] >= 0.5) ? 1 : 0;
    }
    return predictions;
}

vector<double> LogisticRegression::getWeights() const {
    return weights_;
}

vector<double> LogisticRegression::getCostHistory() const {
    return cost_history_;
}
