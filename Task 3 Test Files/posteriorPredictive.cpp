#include <iostream>
#include <fstream>
#include <vector>
#include "./src/bayesianLinearReg.h"
#include "./src/polyReg.h"
#include "./src/matrixOperations.h"

using std::cout;
using std::endl;
using std::vector;
using Matrix = std::vector<std::vector<double>>;
using std::string;
using namespace std;

int main() {
    cout << "=== Testing Posterior and Predictive Distribution ===" << endl;

    Matrix X;
    vector<double> t;
    for (int i = 1; i <= 10; ++i) {
        X.push_back({ static_cast<double>(i) });
        t.push_back(2.0 * i + 1.0);
    }

    const int degree = 2;
    Matrix Phi = genPolyFeatures(X, degree);

    const double alpha = 1e-3;
    const double beta  = 1e+3;

    bayesianLinearRegression blr(alpha, beta);
    blr.fit(Phi, t);

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

    vector<double> predictions = blr.predict(Phi);
    vector<double> predVars    = blr.predictiveVar(Phi);
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

    Matrix X_new;
    for (int i = 11; i <= 15; ++i) {
        X_new.push_back({ static_cast<double>(i) });
    }
    Matrix Phi_new = genPolyFeatures(X_new, degree);

    vector<double> predictionsNew = blr.predict(Phi_new);
    vector<double> predVarsNew    = blr.predictiveVar(Phi_new);
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
