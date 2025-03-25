#ifndef BASISFUNCTIONINTERFACE_H
#define BASISFUNCTIONINTERFACE_H

#include <vector>
#include <string>
#include "polyReg.h"       // For polynomial basis functions (genPolyFeatures)
#include "gaussianBasis.h" // For Gaussian basis functions (generateCenters, computeGaussianBasisFeaturesMatrix)
#include "sigmoidalBasis.h"// For Sigmoidal basis functions (generateSigmoidCenters, computeSigmoidalBasisFeaturesMatrix)

using std::vector;
using std::string;

// Re-use our Matrix alias (a 2D vector of doubles)
using Matrix = vector<vector<double>>;

// Define an enumeration for our supported basis function types.
enum class BasisFunctionType {
    POLYNOMIAL = 1,
    GAUSSIAN   = 2,
    SIGMOIDAL  = 3
};

/// \brief Transforms the input feature matrix X (assumed univariate: one value per data point)
///        using the specified basis function.
/// \param X The input data matrix (each row represents one data point, one feature).
/// \param choice The basis function type, selected by the user.
/// \param param1 For POLYNOMIAL, this is the degree (an integer);
///               for GAUSSIAN and SIGMOIDAL, this is the number of centers (as a double to be cast to int).
/// \param param2 For GAUSSIAN, this is the scale parameter; for SIGMOIDAL, this is the slope parameter.
/// \return The transformed feature matrix (each row contains the basis function evaluations).
Matrix transformFeatures(const Matrix &X, BasisFunctionType choice, double param1, double param2);

#endif