#include "polyReg.h"
#include "matrixOperations.h" 

// gen polynomial features up to given degree
Matrix genPolyFeatures(Matrix& X, int degree) {
    size_t m = X.size();    // data points
    size_t n = X[0].size(); // original features

    // polynomial feature matrix
    Matrix polyFeatures(m, vector<double>(1, 1.0)); // start with bias term (column of ones)

    // generate polynomial features
    for (int d = 1; d <= degree; ++d) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                polyFeatures[i].push_back(std::pow(X[i][j], d));
            }
        }
    }

    return polyFeatures;
}

// gen polynomial features for multivariate datasets up to the given degree
Matrix genMultiVarPolyFeatures(Matrix& X, int degree) {
    size_t m = X.size();    // data points
    size_t n = X[0].size(); // original features

    // polynomial feature matrix
    Matrix polyFeatures(m, vector<double>(1, 1.0)); // start with a bias term (column of ones)

    // Generate polynomial features for each combination of features up to the given degree
    for (int d = 1; d <= degree; ++d) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                polyFeatures[i].push_back(std::pow(X[i][j], d));
            }
        }
    }

    return polyFeatures;
}

// polynomial regression using closed-form solution
vector<double> polyReg(Matrix& X, vector<double>& y, int degree, double lambda) {
    // generate polynomial features
    Matrix polyFeatures = genPolyFeatures(X, degree);

    // calc closed-form solution
    // w = (O^T O)^-1 O^T y

    // O^T
    Matrix Xt = transposeM(polyFeatures);
    // O^T O         
    Matrix XtX = multMs(Xt, polyFeatures);

    // regularization term (lambda * I) along diagonal.
    for (size_t i = 0; i < XtX.size(); ++i) {
        XtX[i][i] += lambda;
    }

    // (O^T O)^-1        
    Matrix XtXInv = invertM(XtX);                 

    // convert y to column matrix
    vector<vector<double>> yM(y.size(), vector<double>(1, 0.0));
    for (size_t i = 0; i < y.size(); ++i) {
        yM[i][0] = y[i];
    }

    Matrix XtY = multMs(Xt, yM);                  // O^T y
    Matrix wM = multMs(XtXInv, XtY);              // (O^T O)^-1 O^T y

    // convert weight matrix to vector
    vector<double> w(wM.size(), 0.0);
    for (size_t i = 0; i < wM.size(); ++i) {
        w[i] = wM[i][0];
    }

    return w;
}

// multiVar polynomial regression using closed-form solution
Matrix multiVarPolyReg(Matrix& X, Matrix& T, int degree, double lambda) {
    // gen polynomial features
    Matrix polyFeatures = genMultiVarPolyFeatures(X, degree);

    // calc closed-form solution
    // W = (O^T O)^-1 O^T T

    // O^T
    Matrix Xt = transposeM(polyFeatures);
    // O^T O         
    Matrix XtX = multMs(Xt, polyFeatures); 

    // lambda along the diagonal to regularize.
    for (size_t i = 0; i < XtX.size(); ++i) {
        XtX[i][i] += lambda;
    }
    

    // (O^T O)^-1       
    Matrix XtXInv = invertM(XtX);  
    // O^T T               
    Matrix XtT = multMs(Xt, T); 
    // (O^T O)^-1 O^T T                  
    Matrix W = multMs(XtXInv, XtT);               

    return W;
}