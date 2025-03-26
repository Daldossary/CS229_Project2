#ifndef BASISFUNCTIONINTERFACE_H
#define BASISFUNCTIONINTERFACE_H

#include <vector>
#include <string>
#include "polyReg.h"       // polynomial basis functions (genPolyFeatures)
#include "gaussianBasis.h" // gaussian basis functions (genCenters, calcGaussBasisFeaturesM)
#include "sigmoidalBasis.h"// sigmoidal basis functions (genSigmoidCenters, calcSigmoidalBasisFeaturesM)

using std::vector;
using std::string;


using Matrix = vector<vector<double>>;

// enum for basis function types
enum class basisFunctionType {
    POLYNOMIAL = 1,
    GAUSSIAN   = 2,
    SIGMOIDAL  = 3
};

// helper function to extract specific column from X (univariate data assumed in column 0)
Matrix transformFeatures(const Matrix &X, basisFunctionType choice, double p1, double p2);

#endif