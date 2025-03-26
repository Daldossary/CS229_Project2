#include <iostream>
#include <fstream>
#include <vector>
#include "./src/bayesianLinearReg.h"
#include "./src/basisFunctionInterface.h"  // transformFeatures() and basisFunctionType
#include "./src/polyReg.h"  // genPolyFeatures (polynomial basis functions)
#include "./src/matrixOperations.h"  // matrix utilities
using std::cout;
using std::cin;
using std::endl;
using std::vector;
using Matrix = std::vector<std::vector<double>>;
using std::string;

int main() {
    cout << "Testing Bayesian Linear Regression with Basis Function Transformation" << endl;
    
    // creating simple synthetic univariate dataset.
    //  x = 1, 2, …, 10 and target t = 2*x + 1.
    Matrix X; // feature matrix
    vector<double> targets; // target vector
    for (int i = 1; i <= 10; i++) { // x = 1, 2, …, 10
        X.push_back({static_cast<double>(i)}); // add x to feature matrix
        targets.push_back(2.0 * i + 1.0); // add target to target vector
    }
    
    // using polynomial basis functions (degree 2).
    // can also use other basis functions like Gaussian, sigmoid, etc.
    int degree = 2;
    Matrix phi = genPolyFeatures(X, degree);
    
    // writing design matrix (after transformation) to a CSV file.
    string designFilename = "./results/design_matrix.csv";
    std::ofstream designFile(designFilename);
    if (!designFile) {
        cout << "Error: Could not open " << designFilename << " for writing." << endl;
        return 1;
    }
    for (size_t i = 0; i < phi.size(); i++) { 
        for (size_t j = 0; j < phi[i].size(); j++) { 
            designFile << phi[i][j]; 
            if (j != phi[i].size() - 1)
                designFile << ",";
        }
        designFile << "\n";
    }
    designFile.close();
    cout << "Design matrix written to " << designFilename << endl;
    
    // setting hyperparameters for bayesian linear regression 
    double alpha = 1e-3;  // prior precision.
    double beta = 1e+3;   // likelihood (noise) precision.
    
    bayesianLinearRegression blr(alpha, beta); // creating bayesian linear regression object
    blr.fit(phi, targets); // fitting model
    
    // predict on training data
    vector<double> predictions = blr.predict(phi); 
    vector<double> predVars = blr.predictiveVar(phi);
    vector<double> weights = blr.getWeights();
    
    // writing predictions and predictive variances to a CSV file.
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