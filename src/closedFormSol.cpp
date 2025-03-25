#include "closedFormSol.h"


// closed-form for single variate linear regression
vector<double> closedFormSingleVar(Matrix& X, vector<double>& y) {
    // X^T
    Matrix Xt = transposeM(X);

    // X^T * X
    Matrix XtX = multMs(Xt, X);

    // (X^T * X)^-1
    Matrix XtXInv = invertM(XtX);

    // X^T * y
    vector<vector<double>> yM(y.size(), vector<double>(1, 0.0));
    for (size_t i = 0; i < y.size(); ++i) {
        yM[i][0] = y[i];
    }
    Matrix XtY = multMs(Xt, yM);

    // w = (X^T * X)^-1 * X^T * y
    Matrix wM = multMs(XtXInv, XtY);

    // Convert wM to vector
    vector<double> w(wM.size(), 0.0);
    for (size_t i = 0; i < wM.size(); ++i) {
        w[i] = wM[i][0];
    }

    return w;
}


// closed-form for multivariate linear regression
Matrix closedFormMultiVar(Matrix& X, Matrix& T) {
    // X^T
    Matrix Xt = transposeM(X);

    // X^T * X
    Matrix XtX = multMs(Xt, X);

    // (X^T * X)^-1
    Matrix XtXInv = invertM(XtX);

    // X^T * T
    Matrix XtT = multMs(Xt, T);

    // W = (X^T * X)^-1 * X^T * T
    Matrix W = multMs(XtXInv, XtT);

    return W;
}
