#ifndef GAUSSIANBASIS_H
#define GAUSSIANBASIS_H

#include <vector>
using std::vector;

// Define Matrix as a two-dimensional vector of doubles.
using Matrix = vector<vector<double>>;

// -----------------------------------------------------------------------------
// Function: gaussianBasisFunction
// Description: Given an input x, a center mu, and a scale s,
//              computes the Gaussian basis function value:
//              φ(x) = exp( - ((x - mu)²) / (2 * s²) ).
// -----------------------------------------------------------------------------
double gaussianBasisFunction(double x, double mu, double s);

// -----------------------------------------------------------------------------
// Function: computeGaussianBasisFeatures
// Description:
//   Given a single input value x and a set of centers, returns a vector
//   whose k-th element is φₖ(x).
// -----------------------------------------------------------------------------
vector<double> computeGaussianBasisFeatures(double x, const vector<double>& centers, double s);

// -----------------------------------------------------------------------------
// Function: computeGaussianBasisFeaturesMatrix
// Description:
//   Given a matrix X (each row is a data point with one feature) and a vector
//   of centers plus a scale s, computes the Gaussian basis transformation for
//   each data point. This function uses C++17 parallel algorithms to speed up
//   the transformation. The resulting matrix has as many rows as X and as many
//   columns as there are centers.
// -----------------------------------------------------------------------------
Matrix computeGaussianBasisFeaturesMatrix(const Matrix &X, const vector<double>& centers, double s);

// -----------------------------------------------------------------------------
// Function: generateCenters
// Description:
//   Given a vector of numbers and a desired number of centers, automatically
//   generates and returns a vector of evenly spaced center values between the
//   minimum and maximum of the data.
// -----------------------------------------------------------------------------
vector<double> generateCenters(const vector<double>& data, int numCenters);

#endif