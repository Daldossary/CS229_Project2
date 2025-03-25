#include "sigmoidalBasis.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <execution>   // for execution policies

// -----------------------------------------------------------------------------
// Function: sigmoidalBasisFunction
// -----------------------------------------------------------------------------
double sigmoidalBasisFunction(double x, double mu, double s) {
    return 1.0 / (1.0 + std::exp( -s * (x - mu) ));
}

// -----------------------------------------------------------------------------
// Function: computeSigmoidalBasisFeatures
// -----------------------------------------------------------------------------
vector<double> computeSigmoidalBasisFeatures(double x, const vector<double>& centers, double s) {
    vector<double> features;
    features.reserve(centers.size());
    for (double mu : centers) {
        features.push_back(sigmoidalBasisFunction(x, mu, s));
    }
    return features;
}

// -----------------------------------------------------------------------------
// Function: computeSigmoidalBasisFeaturesMatrix
// Description: Using the provided matrix X (assumed univariate: each row has one feature)
//              and the given centers, computes the sigmoidal transformation for each data point.
//              To avoid TBB linking issues, we use std::execution::seq.
// -----------------------------------------------------------------------------
Matrix computeSigmoidalBasisFeaturesMatrix(const Matrix &X, const vector<double>& centers, double s) {
    size_t numData = X.size();
    size_t numCenters = centers.size();
    Matrix transformed(numData, vector<double>(numCenters, 0.0));

    // Create an index vector [0, 1, 2, ..., numData-1]
    vector<size_t> indices(numData);
    std::iota(indices.begin(), indices.end(), 0);

    // Use sequential execution (change std::execution::seq to std::execution::par if your environment
    // is linked with TBB and you wish to exploit parallelism).
    std::for_each(std::execution::seq, indices.begin(), indices.end(), [&](size_t i) {
        if (!X[i].empty()) {
            double x = X[i][0];  // Assumes univariate data: one feature per row.
            for (size_t j = 0; j < numCenters; ++j) {
                transformed[i][j] = sigmoidalBasisFunction(x, centers[j], s);
            }
        }
    });
    
    return transformed;
}

// -----------------------------------------------------------------------------
// Function: generateSigmoidCenters
// -----------------------------------------------------------------------------
vector<double> generateSigmoidCenters(const vector<double>& data, int numCenters) {
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