#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "./src/logisticRegression.h"
#include "./src/polyReg.h"  // For genPolyFeatures (for basis expansion)
using std::cout;
using std::endl;
using std::vector;
using Matrix = std::vector<std::vector<double>>;

// Helper function: compute classification accuracy (percentage)
double computeAccuracy(const vector<int>& y_true, const vector<int>& y_pred) {
    if (y_true.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); i++) {
        if (y_true[i] == y_pred[i])
            correct++;
    }
    return 100.0 * correct / y_true.size();
}

int main() {
    cout << "Testing Logistic Regression for Binary Classification" << endl;
    
    // Generate a synthetic univariate dataset:
    // Create 100 data points uniformly in [0, 10]. The true rule is:
    // if x > 5 then label = 1, else 0.
    const int N = 100;
    vector<vector<double>> X;
    vector<int> y;
    std::default_random_engine generator(42);
    std::uniform_real_distribution<double> uniformDist(0.0, 10.0);
    
    for (int i = 0; i < N; i++) {
        double x = uniformDist(generator);
        X.push_back({x});
        int label = (x > 5.0) ? 1 : 0;
        y.push_back(label);
    }
    
    // Optionally, apply a basis expansion (here we use a polynomial basis of degree 2)
    int degree = 2;
    Matrix Phi = genPolyFeatures(X, degree);
    
    // Create and train the logistic regression model.
    double learning_rate = 0.001;
    int max_iter = 1000;
    double tol = 1e-6;
    LogisticRegression logReg(learning_rate, max_iter, tol);
    logReg.fit(Phi, y);
    
    // Predict probabilities and binary labels on the training data.
    vector<double> probs = logReg.predictProb(Phi);
    vector<int> preds = logReg.predict(Phi);
    
    // Compute and display accuracy.
    double accuracy = computeAccuracy(y, preds);
    cout << "Training Accuracy: " << accuracy << "%" << endl;
    
    // Write predictions along with true labels and input values to a CSV file.
    std::ofstream predFile("./results/logistic_regression_predictions.csv");
    if (!predFile) {
        cout << "Error: Unable to open logistic_regression_predictions.csv for writing." << endl;
        return 1;
    }
    predFile << "x,probability,true_label,predicted_label\n";
    for (size_t i = 0; i < X.size(); i++) {
        predFile << X[i][0] << "," << probs[i] << "," << y[i] << "," << preds[i] << "\n";
    }
    predFile.close();
    
    // Write the learned weight vector to a CSV file.
    vector<double> weights = logReg.getWeights();
    std::ofstream weightFile("./results/logistic_regression_weights.csv");
    if (!weightFile) {
        cout << "Error: Unable to open logistic_regression_weights.csv for writing." << endl;
        return 1;
    }
    weightFile << "Index,Weight\n";
    for (size_t j = 0; j < weights.size(); j++) {
        weightFile << j << "," << weights[j] << "\n";
    }
    weightFile.close();
    
    cout << "Logistic Regression experiment completed. Check the results folder for CSV outputs." << endl;
    return 0;
}