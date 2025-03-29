#include "gaussianDiscriminantAnalysis.h"
#include "matrixOperations.h"
#include <cmath>
#include <stdexcept>
#include <set>
#include <algorithm>
#include <numeric>

// helper
// calc mean vector of a set of vectors.
static vector<double> computeMean(const Matrix &X) {
    size_t m = X.size();
    if (m == 0) return vector<double>();
    size_t d = X[0].size();
    vector<double> mean(d, 0.0);
    for (size_t i = 0; i < m; i++){
        for (size_t j = 0; j < d; j++){
            mean[j] += X[i][j];
        }
    }
    for (size_t j = 0; j < d; j++){
        mean[j] /= m;
    }
    return mean;
}

// helper
// calc outer product of two vectors.
static Matrix outerProduct(const vector<double> &v1, const vector<double> &v2) {
    size_t d1 = v1.size();
    size_t d2 = v2.size();
    Matrix prod(d1, vector<double>(d2, 0.0));
    for (size_t i = 0; i < d1; i++){
        for (size_t j = 0; j < d2; j++){
            prod[i][j] = v1[i] * v2[j];
        }
    }
    return prod;
}

// helper
// add two matrices.
static Matrix addMatrices(const Matrix &A, const Matrix &B) {
    size_t m = A.size(), n = A[0].size();
    Matrix C(m, vector<double>(n, 0.0));
    for (size_t i = 0; i < m; i++){
        for (size_t j = 0; j < n; j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

// helper
// scale a matrix by a factor.
static Matrix scaleMatrix(const Matrix &A, double factor) {
    size_t m = A.size(), n = A[0].size();
    Matrix B = A;
    for (size_t i = 0; i < m; i++){
        for (size_t j = 0; j < n; j++){
            B[i][j] *= factor;
        }
    }
    return B;
}

void GaussianDiscriminantAnalysis::fit(const Matrix &X, const vector<int> &y) {
    size_t m = X.size();
    if (m == 0) return;
    size_t d = X[0].size();

    std::set<int> classSet(y.begin(), y.end());
    classes_.assign(classSet.begin(), classSet.end());
    size_t k = classes_.size();


    means_.resize(k, vector<double>(d, 0.0));
    priors_.resize(k, 0.0);
    vector<size_t> counts(k, 0);
    vector<Matrix> X_by_class(k);


    for (size_t i = 0; i < m; i++){
        int label = y[i];
        auto it = std::find(classes_.begin(), classes_.end(), label);
        int idx = std::distance(classes_.begin(), it);
        X_by_class[idx].push_back(X[i]);
        counts[idx]++;
    }

    for (size_t idx = 0; idx < k; idx++){
        means_[idx] = computeMean(X_by_class[idx]);
        priors_[idx] = static_cast<double>(counts[idx]) / m;
    }


    Matrix cov(d, vector<double>(d, 0.0));
    for (size_t i = 0; i < m; i++){
        auto it = std::find(classes_.begin(), classes_.end(), y[i]);
        int idx = std::distance(classes_.begin(), it);

        vector<double> diff(d, 0.0);
        for (size_t j = 0; j < d; j++){
            diff[j] = X[i][j] - means_[idx][j];
        }
        Matrix outer = outerProduct(diff, diff);
        cov = addMatrices(cov, outer);
    }

    covariance_ = scaleMatrix(cov, 1.0 / m);

    for (size_t i = 0; i < covariance_.size(); i++) {
        covariance_[i][i] += 1e-6; 
    }
}

vector<int> GaussianDiscriminantAnalysis::predict(const Matrix &X) const {
    size_t m = X.size();
    size_t d = X[0].size();
    vector<int> predictions(m, 0);


    Matrix covInv = invertM(const_cast<Matrix&>(covariance_));
    

    for (size_t i = 0; i < m; i++){
        double bestScore = -1e12;
        int bestClass = classes_[0];
        for (size_t idx = 0; idx < classes_.size(); idx++){
            double score = 0.0;

            for (size_t j = 0; j < d; j++){
                double sum = 0.0;
                for (size_t l = 0; l < d; l++){
                    sum += X[i][l] * covInv[l][j];
                }
                score += sum * means_[idx][j];
            }

            double quad = 0.0;
            for (size_t j = 0; j < d; j++){
                double sum = 0.0;
                for (size_t l = 0; l < d; l++){
                    sum += means_[idx][l] * covInv[l][j];
                }
                quad += sum * means_[idx][j];
            }
            score -= 0.5 * quad;

            score += std::log(priors_[idx]);
            if (score > bestScore) {
                bestScore = score;
                bestClass = classes_[idx];
            }
        }
        predictions[i] = bestClass;
    }
    return predictions;
}

vector<vector<double>> GaussianDiscriminantAnalysis::getClassMeans() const {
    return means_;
}

Matrix GaussianDiscriminantAnalysis::getCovariance() const {
    return covariance_;
}

vector<double> GaussianDiscriminantAnalysis::getClassPriors() const {
    return priors_;
}

vector<int> GaussianDiscriminantAnalysis::getClasses() const {
    return classes_;
}