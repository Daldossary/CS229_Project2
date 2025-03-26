#ifndef GAUSSIANDISCRIMINANTANALYSIS_H
#define GAUSSIANDISCRIMINANTANALYSIS_H

#include <vector>
using std::vector;
using Matrix = std::vector<std::vector<double>>;

class GaussianDiscriminantAnalysis {
public:
    // Fit the GDA model on features X (m x d) and labels y.
    void fit(const Matrix &X, const vector<int> &y);

    // Given a feature matrix X, predict the class labels.
    vector<int> predict(const Matrix &X) const;

    // Return the estimated class means (one vector per class).
    vector<vector<double>> getClassMeans() const;

    // Return the shared covariance matrix.
    Matrix getCovariance() const;

    // Return the class prior probabilities.
    vector<double> getClassPriors() const;
    
    // NEW: Return the sorted unique class labels.
    vector<int> getClasses() const;

private:
    vector<vector<double>> means_; // Each row is the mean vector μₖ for class k.
    Matrix covariance_;            // Shared covariance matrix Σ.
    vector<double> priors_;        // Prior probabilities πₖ for each class.
    vector<int> classes_;          // Sorted unique class labels.
};

#endif