#ifndef SIGMOIDALBASIS_H
#define SIGMOIDALBASIS_H

#include <vector>
using std::vector;


using Matrix = vector<vector<double>>;

// calc sigmoidal basis function
double sigmoidalBasisFunction(double x, double mu, double s);

// input value x, vector of centers, and slope s, output vector of sigmoidal basis functions
vector<double> calcSigmoidalBasisFeatures(double x, const vector<double>& centers, double s);

// input matrix X, vector of centers, and slope s, output matrix of sigmoidal basis functions
Matrix calcSigmoidalBasisFeaturesM(const Matrix &X, const vector<double>& centers, double s);

// input data and number of centers, output vector of centers
vector<double> genSigmoidCenters(const vector<double>& data, int numCenters);

#endif