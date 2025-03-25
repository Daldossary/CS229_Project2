#ifndef CLOSEDFORMSOLUTIONS_H
#define CLOSEDFORMSOLUTIONS_H

#include <vector>
#include "matrixOperations.h"

using std::vector;


vector<double> closedFormSingleVar(Matrix& X, vector<double>& y);
Matrix closedFormMultiVar(Matrix& X, Matrix& T);

#endif