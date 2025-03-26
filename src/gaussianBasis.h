#ifndef GAUSSIANBASIS_H
#define GAUSSIANBASIS_H

#include <vector>
using std::vector;


using Matrix = vector<vector<double>>;

// input x, center mu, and scale s,
double gaussianBasisFunction(double x, double mu, double s);

// single input value x and a set of centers, returns a vector
vector<double> calcGaussBasisFeatures(double x, const vector<double>& centers, double s);

// input matrix X, a set of centers, and a scale s, returns a matrix
Matrix calcGaussBasisFeaturesM(const Matrix &X, const vector<double>& centers, double s);

// vector of numbers, number of centers, returns a vector of centers
vector<double> genCenters(const vector<double>& data, int numCenters);

#endif