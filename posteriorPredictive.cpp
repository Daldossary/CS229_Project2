// testPosteriorPredictive.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include "./src/bayesianLinearReg.h"  // Contains the fit(), predict(), and predictiveVariance() methods.
#include "./src/polyReg.h"                   // Contains genPolyFeatures() for polynomial basis expansion.
#include "./src/matrixOperations.h"          // For matrix utilities (re-used by the above modules).

using std::cout;
using std::endl;
using std::vector;
using Matrix = std::vector<std::vector<double>>;
using std::string;
using namespace std;

int main() {
    cout << "=== Testing Posterior and Predictive Distribution ===" << endl;

    // ----- Step 1. Create a simple synthetic training dataset -----
    // Let the univariate input x take the values 1,2,...,10,
    // and let the target be generated as t = 2 * x + 1 (without added noise for clarity).
    Matrix X;         // raw input data
    vector<double> t; // targets
    for (int i = 1; i <= 10; ++i) {
        X.push_back({ static_cast<double>(i) });
        t.push_back(2.0 * i + 1.0);
    }

    // ----- Step 2. Feature transformation using a polynomial basis -----
    // We use degree 2 polynomial features. The function genPolyFeatures() (in polyReg.cpp)
    // pre-pends a bias term (column of ones) and then appends x^1 and x^2.
    const int degree = 2;
    Matrix Phi = genPolyFeatures(X, degree);

    // ----- Step 3. Set hyperparameters and fit the Bayesian linear regression model -----
    // Hyperparameters: 
    //    α (prior precision) and β (likelihood precision). 
    // These hyperparameters enter the closed‐form posterior:
    //    Sₙ = (α I + β ΦᵀΦ)⁻¹  and  mₙ = β Sₙ Φᵀ t.
    const double alpha = 1e-3;
    const double beta  = 1e+3;

    BayesianLinearRegression blr(alpha, beta);
    blr.fit(Phi, t);

    // Write out the learned posterior weight parameters (the posterior mean mₙ).
    vector<double> weights = blr.getWeights();
    {
        std::ofstream weightsFile("./results/posterior_weights.csv");
        if (!weightsFile) {
            cout << "Error: Could not open file for writing posterior_weights.csv" << endl;
            return 1;
        }
        weightsFile << "Index,Weight\n";
        for (size_t j = 0; j < weights.size(); ++j) {
            weightsFile << j << "," << weights[j] << "\n";
        }
        weightsFile.close();
        cout << "Posterior weight vector written to ./results/posterior_weights.csv" << endl;
    }

    // ----- Step 4. Compute predictions on the training data -----
    vector<double> predictions = blr.predict(Phi);
    vector<double> predVars    = blr.predictiveVariance(Phi);
    {
        std::ofstream predFile("./results/posterior_predictive_training.csv");
        if (!predFile) {
            cout << "Error: Could not open file for writing posterior_predictive_training.csv" << endl;
            return 1;
        }
        predFile << "x,t,prediction,variance\n";
        for (size_t i = 0; i < X.size(); ++i) {
            predFile << X[i][0] << "," << t[i] << "," << predictions[i] << "," << predVars[i] << "\n";
        }
        predFile.close();
        cout << "Predictive distribution on training data written to ./results/posterior_predictive_training.csv" << endl;
    }

    // ----- Step 5. Compute predictive distribution for new inputs -----
    // For example, let new input x range from 11 to 15.
    Matrix X_new;
    for (int i = 11; i <= 15; ++i) {
        X_new.push_back({ static_cast<double>(i) });
    }
    // Transform new inputs using the same polynomial basis.
    Matrix Phi_new = genPolyFeatures(X_new, degree);

    vector<double> predictionsNew = blr.predict(Phi_new);
    vector<double> predVarsNew    = blr.predictiveVariance(Phi_new);
    {
        std::ofstream newPredFile("./results/posterior_predictive_new.csv");
        if (!newPredFile) {
            cout << "Error: Could not open file for writing posterior_predictive_new.csv" << endl;
            return 1;
        }
        newPredFile << "x,prediction,variance\n";
        for (size_t i = 0; i < X_new.size(); ++i) {
            newPredFile << X_new[i][0] << "," << predictionsNew[i] << "," << predVarsNew[i] << "\n";
        }
        newPredFile.close();
        cout << "Predictive distribution for new inputs written to ./results/posterior_predictive_new.csv" << endl;
    }

    cout << "Posterior and predictive distributions have been successfully computed." << endl;
    return 0;
}