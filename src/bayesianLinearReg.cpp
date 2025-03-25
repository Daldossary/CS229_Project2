#include "bayesianLinearReg.h"
#include "matrixOperations.h"  // for transposeM, multMs, invertM
#include <vector>
#include <cmath>
#include <cassert>

using std::vector;
using Matrix = std::vector<std::vector<double>>;

// Helper: Multiply a matrix A (size m x n) by a vector v (size n) -> returns vector of length m.
static vector<double> multMatVec(const Matrix &A, const vector<double> &v) {
    size_t m = A.size();
    size_t n = A[0].size();
    vector<double> result(m, 0.0);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            result[i] += A[i][j] * v[j];
    return result;
}

// Helper: Create a d x d identity matrix.
static Matrix identityMatrix(size_t d) {
    Matrix I(d, vector<double>(d, 0.0));
    for (size_t i = 0; i < d; ++i)
        I[i][i] = 1.0;
    return I;
}

BayesianLinearRegression::BayesianLinearRegression(double alpha, double beta)
    : alpha_(alpha), beta_(beta) {}

void BayesianLinearRegression::fit(const Matrix &Phi, const vector<double> &t) {
    // Phi is m x d.
    // 1. Compute the transpose of Phi.
    Matrix PhiT = transposeM(const_cast<Matrix&>(Phi));
    // 2. Compute PhiᵀΦ.
    Matrix PhiTPhi = multMs(PhiT, const_cast<Matrix&>(Phi));
    size_t d = PhiTPhi.size();
    // 3. Compute A = α I + β (PhiᵀΦ).
    Matrix I = identityMatrix(d);
    Matrix A(d, vector<double>(d, 0.0));
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j)
            A[i][j] = beta_ * PhiTPhi[i][j] + (i == j ? alpha_ : 0.0);

    // 4. Compute the posterior covariance S_N = A⁻¹.
    S_N_ = invertM(A);

    // 5. Compute the posterior mean: m_N = β S_N Φᵀ t.
    // Convert vector t to a column matrix.
    Matrix tM(t.size(), vector<double>(1, 0.0));
    for (size_t i = 0; i < t.size(); ++i)
        tM[i][0] = t[i];
    Matrix PhiTt = multMs(PhiT, tM);  // d x 1 matrix.
    Matrix m_N_mat = multMs(S_N_, PhiTt);  // d x 1 matrix.
    m_N_.resize(m_N_mat.size());
    for (size_t i = 0; i < m_N_mat.size(); ++i)
        m_N_[i] = beta_ * m_N_mat[i][0];
}

vector<double> BayesianLinearRegression::predict(const Matrix &Phi_new) const {
    // For each new data point (row of Phi_new), compute: prediction = m_Nᵀ φ(x)
    vector<double> predictions;
    for (size_t i = 0; i < Phi_new.size(); ++i) {
        double pred = 0.0;
        for (size_t j = 0; j < m_N_.size(); ++j)
            pred += m_N_[j] * Phi_new[i][j];
        predictions.push_back(pred);
    }
    return predictions;
}

vector<double> BayesianLinearRegression::predictiveVariance(const Matrix &Phi_new) const {
    // For each new data point, compute: variance = 1/β + φ(x)ᵀ S_N φ(x)
    vector<double> variances;
    for (size_t i = 0; i < Phi_new.size(); ++i) {
        vector<double> phi = Phi_new[i];
        vector<double> S_phi = multMatVec(S_N_, phi);
        double dot = 0.0;
        for (size_t j = 0; j < phi.size(); ++j)
            dot += phi[j] * S_phi[j];
        variances.push_back(1.0 / beta_ + dot);
    }
    return variances;
}

vector<double> BayesianLinearRegression::getWeights() const {
    return m_N_;
}