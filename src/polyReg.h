#ifndef POLYNOMIALREGRESSION_H
#define POLYNOMIALREGRESSION_H

#include <vector>
#include <string>
#include <cmath>

using std::vector;
using std::string;


using Matrix = vector<vector<double>>;


Matrix genPolyFeatures(Matrix& X, int degree);
Matrix genMultiVarPolyFeatures(Matrix& X, int degree);
vector<double> polyReg(Matrix& X, vector<double>& y, int degree, double lambda = 1e-5);
Matrix multiVarPolyReg(Matrix& X, Matrix& T, int degree, double lambda = 1e-5);

#endif