#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include "./src/bayesianLinearReg.h"
#include "./src/polyReg.h"
#include "./src/matrixOperations.h"

using namespace std;
using Matrix = vector<vector<double>>;

int main() {
    cout << "Testing Zero-Mean Gaussian Prior on the Bayesian Linear Regression Weights" << endl;
    
    // Create a synthetic univariate dataset.
    // Let x = 1, 2, ..., 10 and let the targets have a weak trend so that
    // with a strong prior the learned weights are pushed close to zero.
    Matrix X;
    vector<double> targets;
    for (int i = 1; i <= 10; ++i) {
        X.push_back({static_cast<double>(i)});
        // Generate target t = 0.5 * x + noise.
        // (With a very strong prior, the model will not deviate very much from zero.)
        double noise = 0.1 * ((rand() % 100) / 100.0 - 0.5);
        targets.push_back(0.5 * i + noise);
    }
    
    // For simplicity, use polynomial basis expansion (degree 2).
    // genPolyFeatures (from polyReg.cpp) will add a bias term and polynomial terms.
    int degree = 2;
    Matrix Phi = genPolyFeatures(X, degree);

    // Set hyperparameters:
    // - Use a very high alpha so that the zero-mean prior strongly pushes weights toward zero.
    // - Use a moderate beta.
    double alpha = 1000.0; // Prior precision (strong prior => variance=1/alpha is small).
    double beta = 1.0;     // Noise precision.
    
    // Create and fit the Bayesian linear regression model.
    BayesianLinearRegression blr(alpha, beta);
    blr.fit(Phi, targets);
    
    // Retrieve the learned (posterior mean) weights.
    vector<double> weights = blr.getWeights();
    
    // Output the learned weights to a CSV file in the results folder.
    string outputFile = "./results/gaussian_prior_weights.csv";
    ofstream outFile(outputFile);
    if (!outFile) {
        cout << "Error: Could not open " << outputFile << " for writing." << endl;
        return 1;
    }
    
    outFile << "Index,Weight\n";
    for (size_t i = 0; i < weights.size(); ++i) {
        outFile << i << "," << weights[i] << "\n";
    }
    outFile.close();
    
    cout << "Learned weights using the zero-mean Gaussian prior (alpha = " << alpha << ") were written to " 
         << outputFile << endl;
    
    return 0;
}