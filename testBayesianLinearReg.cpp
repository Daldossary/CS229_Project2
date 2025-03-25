#include <iostream>
#include <fstream>
#include <vector>
#include "./src/bayesianLinearReg.h"
#include "./src/basisFunctionInterface.h"  // Provides transformFeatures() and BasisFunctionType
#include "./src/polyReg.h"  // For genPolyFeatures (polynomial basis functions)
#include "./src/matrixOperations.h"  // For matrix utilities
using std::cout;
using std::cin;
using std::endl;
using std::vector;
using Matrix = std::vector<std::vector<double>>;
using std::string;

int main() {
    cout << "Testing Bayesian Linear Regression with Basis Function Transformation" << endl;
    
    // Create a simple synthetic univariate dataset.
    // Let x = 1, 2, …, 10 and target t = 2*x + 1.
    Matrix X;
    vector<double> targets;
    for (int i = 1; i <= 10; i++) {
        X.push_back({static_cast<double>(i)});
        targets.push_back(2.0 * i + 1.0);
    }
    
    // In this test, we choose to use polynomial basis functions (degree 2).
    // (You could also use Gaussian or Sigmoidal by changing the interface parameters.)
    int degree = 2;
    Matrix Phi = genPolyFeatures(X, degree);
    
    // Write the design matrix (after transformation) to a CSV file.
    string designFilename = "./results/design_matrix.csv";
    std::ofstream designFile(designFilename);
    if (!designFile) {
        cout << "Error: Could not open " << designFilename << " for writing." << endl;
        return 1;
    }
    for (size_t i = 0; i < Phi.size(); i++) {
        for (size_t j = 0; j < Phi[i].size(); j++) {
            designFile << Phi[i][j];
            if (j != Phi[i].size() - 1)
                designFile << ",";
        }
        designFile << "\n";
    }
    designFile.close();
    cout << "Design matrix written to " << designFilename << endl;
    
    // Set the hyperparameters for the Bayesian linear regression model.
    double alpha = 1e-3;  // Prior precision.
    double beta = 1e+3;   // Likelihood (noise) precision.
    
    BayesianLinearRegression blr(alpha, beta);
    blr.fit(Phi, targets);
    
    // Predict on the training data.
    vector<double> predictions = blr.predict(Phi);
    vector<double> predVars = blr.predictiveVariance(Phi);
    vector<double> weights = blr.getWeights();
    
    // Write predictions and predictive variances to a CSV file.
    string predFilename = "./results/blr_predictions.csv";
    std::ofstream predFile(predFilename);
    if (!predFile) {
        cout << "Error: Could not open " << predFilename << " for writing." << endl;
        return 1;
    }
    predFile << "x,target,prediction,variance\n";
    for (size_t i = 0; i < X.size(); i++) {
        predFile << X[i][0] << "," << targets[i] << "," << predictions[i] << "," << predVars[i] << "\n";
    }
    predFile.close();
    cout << "Predictions and variances written to " << predFilename << endl;
    
    // Write the final learned weight vector to a CSV file.
    string weightFilename = "./results/blr_weights.csv";
    std::ofstream weightFile(weightFilename);
    if (!weightFile) {
        cout << "Error: Could not open " << weightFilename << " for writing." << endl;
        return 1;
    }
    weightFile << "WeightIndex,WeightValue\n";
    for (size_t j = 0; j < weights.size(); j++) {
        weightFile << j << "," << weights[j] << "\n";
    }
    weightFile.close();
    cout << "Learned weights written to " << weightFilename << endl;
    
    return 0;
}