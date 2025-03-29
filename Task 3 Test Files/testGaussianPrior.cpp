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
    
    Matrix X;
    vector<double> targets;
    for (int i = 1; i <= 10; ++i) {
        X.push_back({static_cast<double>(i)});
        double noise = 0.1 * ((rand() % 100) / 100.0 - 0.5);
        targets.push_back(0.5 * i + noise);
    }
    
    int degree = 2;
    Matrix Phi = genPolyFeatures(X, degree);

    double alpha = 1000.0;
    double beta = 1.0;
    
    bayesianLinearRegression blr(alpha, beta);
    blr.fit(Phi, targets);
    
    vector<double> weights = blr.getWeights();
    
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
