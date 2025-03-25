#ifndef MATRIXOPERATIONS_H
#define MATRIXOPERATIONS_H

#include <vector>

using std::vector;


using Matrix = vector<vector<double>>;


Matrix transposeM(Matrix& m);
Matrix multMs(Matrix& m1, Matrix& m2);
Matrix invertM(Matrix& m);

#endif