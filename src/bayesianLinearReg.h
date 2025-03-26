#ifndef BAYESIANLINEARREGRESSION_H
#define BAYESIANLINEARREGRESSION_H

#include <vector>
using std::vector;
using Matrix = std::vector<std::vector<double>>;

// Bayesian linear regression class.
class bayesianLinearRegression {
public:
    // constructor
    // alpha = prior precision, beta = noise (likelihood) precision
    bayesianLinearRegression(double alpha, double beta);

    // fitting model given design matrix (Φ) and target vector
    void fit(const Matrix &phi, const vector<double> &t);

    // predict mean for each row in new design matrix
    vector<double> predict(const Matrix &phiNew) const;

    // calc predictive var for each row of new design matrix
    vector<double> predictiveVar(const Matrix &phiNew) const;

    // return posterior mean weight vector
    vector<double> getWeights() const;

private:
    double alpha_;   // prior precision.
    double beta_;    // noise precision (inverse variance).
    Matrix s_N_;     // posterior covariance matrix.
    vector<double> m_N_;  // posterior mean weights.
};

#endif