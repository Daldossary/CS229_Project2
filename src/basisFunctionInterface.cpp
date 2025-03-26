#include "basisFunctionInterface.h"
#include <iostream>
#include <numeric>
#include <algorithm>

// helper function to extract specific column from X (univariate data assumed in column 0)
vector<double> extractCol(const Matrix &X, size_t colIndex = 0) {
    vector<double> col; // extracted column
    for (const auto &row : X) { // iterate over rows
        if (colIndex < row.size()) // check if column index is valid
            col.push_back(row[colIndex]); // add the element to the column
    }
    return col; // return extracted column
}

Matrix transformFeatures(const Matrix &X, basisFunctionType choice, double p1, double p2) {
    Matrix transformed;
    switch (choice) {
        case basisFunctionType::POLYNOMIAL: {
            // p1 = degree of polynomial
            // cast degree to an integer
            int degree = static_cast<int>(p1);

            // Using existing polynomial feature generation function from polyReg.
            // const_cast is used to remove the const qualifier from X.
            transformed = genPolyFeatures(const_cast<Matrix &>(X), degree);
            break;
        }
        case basisFunctionType::GAUSSIAN: {
            // p1 = number of centers (integer)
            int numCenters = static_cast<int>(p1);
            // p2 = scale parameter for gaussian basis
            double scale = p2;
            // get univariate data column from X
            vector<double> dataCol = extractCol(X, 0);
            // generate centers based on data range
            vector<double> centers = genCenters(dataCol, numCenters);
            // calc gaussian basis transformed features.
            transformed = calcGaussBasisFeaturesM(X, centers, scale);
            break;
        }
        case basisFunctionType::SIGMOIDAL: {
            // p1 = number of centers (integer)
            int numCenters = static_cast<int>(p1);
            // p2 = slope for sigmoidal basis
            double slope = p2;
            vector<double> dataCol = extractCol(X, 0);
            // generate centers for sigmoidal function.
            vector<double> centers = genSigmoidCenters(dataCol, numCenters);
            // calc sigmoidal basis transformed features.
            transformed = calcSigmoidalBasisFeaturesM(X, centers, slope);
            break;
        }
        default:
            std::cout << "Invalid basis function selection." << std::endl;
            break;
    }
    return transformed;
}