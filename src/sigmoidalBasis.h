#ifndef SIGMOIDALBASIS_H
#define SIGMOIDALBASIS_H

#include <vector>
using std::vector;

// Define Matrix as a two-dimensional vector of doubles.
using Matrix = vector<vector<double>>;

/// \brief Computes the sigmoidal basis function given an input x,
/// a center mu, and a slope s (i.e. φ(x) = 1/(1+exp(–s*(x–mu)))).
double sigmoidalBasisFunction(double x, double mu, double s);

/// \brief For a given input x, returns a vector of sigmoidal basis features
/// using the provided centers (offsets). A common slope s is assumed.
vector<double> computeSigmoidalBasisFeatures(double x, const vector<double>& centers, double s);

/// \brief Given a matrix X (each row is a univariate data point),
/// and a vector of centers with a common slope s,
/// computes the sigmoidal basis transformed feature matrix.
/// This implementation uses C++17’s execution policies (set here to sequential
/// for wider compatibility—but change to std::execution::par if you have TBB linked).
Matrix computeSigmoidalBasisFeaturesMatrix(const Matrix &X, const vector<double>& centers, double s);

/// \brief Given a vector of data and the desired number of centers,
/// returns a vector of evenly spaced centers between the minimum and maximum values.
vector<double> generateSigmoidCenters(const vector<double>& data, int numCenters);

#endif