#ifndef GAUSSIANDISCRIMINANTANALYSIS_H
#define GAUSSIANDISCRIMINANTANALYSIS_H

#include <vector>
using std::vector;
using Matrix = std::vector<std::vector<double>>;

class GaussianDiscriminantAnalysis {
public:

    void fit(const Matrix &X, const vector<int> &y);

    vector<int> predict(const Matrix &X) const;

    vector<vector<double>> getClassMeans() const;

    Matrix getCovariance() const;

    vector<double> getClassPriors() const;
    
    vector<int> getClasses() const;

private:
    vector<vector<double>> means_;
    Matrix covariance_;
    vector<double> priors_;
    vector<int> classes_;
};

#endif
