#include "bayesianLinearReg.h"
#include "matrixOperations.h"
#include <vector>
#include <cmath>
#include <cassert>

using std::vector;
using Matrix = std::vector<std::vector<double>>;

// helper function to multiply matrix A (m x n) by vector v (n)
// returns vector of length m.
static vector<double> multMatVec(const Matrix &A, const vector<double> &v) {
    size_t m = A.size(); // number of rows
    size_t n = A[0].size(); // number of columns
    vector<double> result(m, 0.0); // result vector
    for (size_t i = 0; i < m; ++i) // for each row
        for (size_t j = 0; j < n; ++j) // for each column
            result[i] += A[i][j] * v[j]; // multiply and add
    return result; // return result
}

// helper function to make d x d identity matrix.
static Matrix idMatrix(size_t d) {
    Matrix I(d, vector<double>(d, 0.0)); // create d x d matrix
    for (size_t i = 0; i < d; ++i) // for each row
        I[i][i] = 1.0; // set diagonal to 1
    return I; // return identity matrix
}

bayesianLinearRegression::bayesianLinearRegression(double alpha, double beta) 
    : alpha_(alpha), beta_(beta) {} // constructor

void bayesianLinearRegression::fit(const Matrix &phi, const vector<double> &t) {
    // phi (m x d)
    // transpose of phi
    Matrix phiT = transposeM(const_cast<Matrix&>(phi));
    // calc phi transpose times phi
    Matrix phiTphi = multMs(phiT, const_cast<Matrix&>(phi));
    size_t d = phiTphi.size(); // number of columns
    // calc A = (alpha * identity) + (beta * (phi^Traspose * phi)
    Matrix I = idMatrix(d); // create identity matrix
    Matrix A(d, vector<double>(d, 0.0)); // create d x d matrix
    for (size_t i = 0; i < d; ++i) // for each row
        for (size_t j = 0; j < d; ++j) // for each column
            A[i][j] = beta_ * phiTphi[i][j] + (i == j ? alpha_ : 0.0); // set A

    // calc posterior covariance s_N = A inverse.
    s_N_ = invertM(A);

    // calc posterior mean: m_N = beta * s_N * phi^T * t
    Matrix tM(t.size(), vector<double>(1, 0.0)); // converting vector t to column matrix
    for (size_t i = 0; i < t.size(); ++i) // for each element in t
        tM[i][0] = t[i]; // set element in column matrix
    Matrix phiTt = multMs(phiT, tM);  // d x 1 matrix.
    Matrix m_N_mat = multMs(s_N_, phiTt);  // d x 1 matrix.
    m_N_.resize(m_N_mat.size()); // resize m_N_ to d
    for (size_t i = 0; i < m_N_mat.size(); ++i) // for each element in m_N_mat
        m_N_[i] = beta_ * m_N_mat[i][0]; // set element in m_N_
}

vector<double> bayesianLinearRegression::predict(const Matrix &phiNew) const {
    // For each new data point (row in phiNew)
    // calc prediction = m_N^T * phi(x)
    vector<double> predictions; // vector to store predictions
    for (size_t i = 0; i < phiNew.size(); ++i) { // for each row in phiNew
        double pred = 0.0; // create prediction
        for (size_t j = 0; j < m_N_.size(); ++j) // for each element in m_N_
            pred += m_N_[j] * phiNew[i][j]; // multiply and add
        predictions.push_back(pred); // add prediction to vector
    }
    return predictions; // return vector of predictions
}

vector<double> bayesianLinearRegression::predictiveVar(const Matrix &phiNew) const {
    // For each new data point
    // calc variance = 1/beta + phi(x)^Transpose * s_N * phi(x)
    vector<double> variances; // vector to store variances
    for (size_t i = 0; i < phiNew.size(); ++i) { // for each row in phiNew
        vector<double> phi = phiNew[i]; // get row
        vector<double> sPhi = multMatVec(s_N_, phi); // s_N * phi(x)
        double dot = 0.0; // dot product
        for (size_t j = 0; j < phi.size(); ++j) // for each element in phi
            dot += phi[j] * sPhi[j]; // multiply and add
        variances.push_back(1.0 / beta_ + dot); // add variance to vector
    }
    return variances; // return vector of variances
}

vector<double> bayesianLinearRegression::getWeights() const {
    return m_N_;
}