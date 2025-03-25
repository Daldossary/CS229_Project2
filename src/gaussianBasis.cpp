#include "gaussianBasis.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <execution>

// -----------------------------------------------------------------------------
// Function: gaussianBasisFunction
// -----------------------------------------------------------------------------
double gaussianBasisFunction(double x, double mu, double s) {
    return std::exp(- ((x - mu) * (x - mu)) / (2 * s * s));
}

// -----------------------------------------------------------------------------
// Function: computeGaussianBasisFeatures
// -----------------------------------------------------------------------------
vector<double> computeGaussianBasisFeatures(double x, const vector<double>& centers, double s) {
    vector<double> features;
    features.reserve(centers.size());
    for (double mu : centers) {
        features.push_back(gaussianBasisFunction(x, mu, s));
    }
    return features;
}

// -----------------------------------------------------------------------------
// Function: computeGaussianBasisFeaturesMatrix
// Description: Uses C++17 parallel algorithms to compute the Gaussian basis
//              transformation for each data point in X (assumed to be univariate).
// -----------------------------------------------------------------------------
Matrix computeGaussianBasisFeaturesMatrix(const Matrix &X, const vector<double>& centers, double s) {
    size_t numData = X.size();
    size_t numCenters = centers.size();
    Matrix transformed(numData, vector<double>(numCenters, 0.0));
    
    // Create an index vector [0, 1, 2, ..., numData-1] for parallel iteration.
    vector<size_t> indices(numData);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
        if (!X[i].empty()) {
            double x = X[i][0];  // Assumes univariate data (one feature per data point)
            for (size_t j = 0; j < numCenters; ++j) {
                transformed[i][j] = gaussianBasisFunction(x, centers[j], s);
            }
        }
    });
    
    return transformed;
}

// -----------------------------------------------------------------------------
// Function: generateCenters
// -----------------------------------------------------------------------------
vector<double> generateCenters(const vector<double>& data, int numCenters) {
    if (data.empty() || numCenters <= 0)
        return {};
        
    double minVal = *std::min_element(data.begin(), data.end());
    double maxVal = *std::max_element(data.begin(), data.end());
    vector<double> centers(numCenters, 0.0);
    
    if (numCenters == 1) {
        centers[0] = (minVal + maxVal) / 2.0;
    } else {
        double interval = (maxVal - minVal) / (numCenters - 1);
        for (int i = 0; i < numCenters; ++i) {
            centers[i] = minVal + i * interval;
        }
    }
    
    return centers;
}