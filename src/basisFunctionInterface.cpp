#include "basisFunctionInterface.h"
#include <iostream>
#include <numeric>
#include <algorithm>

// Helper function to extract a specific column from X (here, univariate data assumed in column 0)
vector<double> extractColumn(const Matrix &X, size_t colIndex = 0) {
    vector<double> col;
    for (const auto &row : X) {
        if (colIndex < row.size())
            col.push_back(row[colIndex]);
    }
    return col;
}

Matrix transformFeatures(const Matrix &X, BasisFunctionType choice, double param1, double param2) {
    Matrix transformed;
    switch (choice) {
        case BasisFunctionType::POLYNOMIAL: {
            // param1 is the degree (an integer)
            int degree = static_cast<int>(param1);
            // Use the existing polynomial feature generation function from polyReg.
            // Note: genPolyFeatures expects a non-const Matrix, so we use a cast.
            transformed = genPolyFeatures(const_cast<Matrix &>(X), degree);
            break;
        }
        case BasisFunctionType::GAUSSIAN: {
            // param1: number of centers (an integer)
            int numCenters = static_cast<int>(param1);
            // param2: scale parameter for the Gaussian basis
            double s = param2;
            // Extract the univariate data column from X.
            vector<double> dataCol = extractColumn(X, 0);
            // Automatically generate centers based on the data range.
            vector<double> centers = generateCenters(dataCol, numCenters);
            // Compute the Gaussian basis transformed features.
            transformed = computeGaussianBasisFeaturesMatrix(X, centers, s);
            break;
        }
        case BasisFunctionType::SIGMOIDAL: {
            // param1: number of centers (an integer)
            int numCenters = static_cast<int>(param1);
            // param2: slope parameter for the sigmoidal basis
            double slope = param2;
            vector<double> dataCol = extractColumn(X, 0);
            // Generate centers automatically for the sigmoidal function.
            vector<double> centers = generateSigmoidCenters(dataCol, numCenters);
            // Compute the sigmoidal basis transformed features.
            transformed = computeSigmoidalBasisFeaturesMatrix(X, centers, slope);
            break;
        }
        default:
            std::cout << "Invalid basis function selection." << std::endl;
            break;
    }
    return transformed;
}