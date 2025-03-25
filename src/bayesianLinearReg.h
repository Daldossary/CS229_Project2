#ifndef BAYESIANLINEARREGRESSION_H
#define BAYESIANLINEARREGRESSION_H

#include <vector>
using std::vector;
using Matrix = std::vector<std::vector<double>>;

// BayesianLinearRegression implements a Bayesian linear model.
// Given a design matrix Φ (with one row per data point, where each row is the feature vector ϕ(x))
// and target values t, the model assumes:
//    p(t|x,w,β) = N(t | wᵀϕ(x), β⁻¹)
// with a Gaussian prior over the weights: p(w)=N(0,α⁻¹ I).
// The closed-form posterior is computed as:
//    S_N = (α I + β ΦᵀΦ)⁻¹
//    m_N = β S_N Φᵀ t
// and for any new input xnew the predictive distribution is:
//    mean = m_Nᵀϕ(xnew)
//    variance = 1/β + ϕ(xnew)ᵀ S_N ϕ(xnew)
class BayesianLinearRegression {
public:
    // Constructor: alpha is the prior precision and beta is the noise (likelihood) precision.
    BayesianLinearRegression(double alpha, double beta);

    // Fit the model given the design matrix (Φ) and target vector.
    void fit(const Matrix &Phi, const vector<double> &t);

    // Predict the mean for each row in a new design matrix.
    vector<double> predict(const Matrix &Phi_new) const;

    // Compute the predictive variance for each row of the new design matrix.
    vector<double> predictiveVariance(const Matrix &Phi_new) const;

    // Return the posterior mean weight vector.
    vector<double> getWeights() const;

private:
    double alpha_;   // Prior precision.
    double beta_;    // Noise precision (inverse variance).
    Matrix S_N_;     // Posterior covariance matrix.
    vector<double> m_N_;  // Posterior mean weights.
};

#endif