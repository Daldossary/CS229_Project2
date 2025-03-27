// runAllExperiments.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>

#include "./src/logisticRegression.h"
#include "./src/polyReg.h"           // provides genPolyFeatures()
#include "./src/sigmoidalBasis.h"    // provides genSigmoidCenters() and calcSigmoidalBasisFeaturesM()
#include "./src/gaussianBasis.h"     // provides genCenters() and calcGaussBasisFeaturesM()
#include "./src/gaussianDiscriminantAnalysis.h"

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;
using Matrix = std::vector<std::vector<double>>;

// Utility: Compute classification accuracy.
double computeAccuracy(const vector<int>& actual, const vector<int>& predicted) {
    int correct = 0;
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] == predicted[i])
            correct++;
    }
    return 100.0 * correct / actual.size();
}

// Utility: Compute binary confusion matrix.
vector<vector<int>> confusionMatrix(const vector<int>& actual, const vector<int>& predicted) {
    vector<vector<int>> cm(2, vector<int>(2, 0));
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] == 0) {
            if (predicted[i] == 0) cm[0][0]++;
            else cm[0][1]++;
        } else {
            if (predicted[i] == 1) cm[1][1]++;
            else cm[1][0]++;
        }
    }
    return cm;
}

// Utility: Write a vector to CSV.
void writeVectorToCSV(const string &filename, const vector<double> &vec, const string &header) {
    std::ofstream file(filename);
    file << header << "\n";
    for (size_t i = 0; i < vec.size(); i++) {
        file << i << "," << vec[i] << "\n";
    }
    file.close();
}

// Utility: Write a matrix (e.g., confusion matrix) to CSV.
void writeMatrixToCSV(const string &filename, const vector<vector<int>> &mat) {
    std::ofstream file(filename);
    for (const auto &row : mat) {
        for (size_t j = 0; j < row.size(); j++) {
            file << row[j] << ((j < row.size()-1) ? "," : "\n");
        }
    }
    file.close();
}

////////////////////////////////////////////////////////////////////////////////
// Synthetic Experiment: Generate a 2D dataset and run logistic regression
////////////////////////////////////////////////////////////////////////////////
void runSyntheticExperiment() {
    cout << "=== Running Synthetic Experiment ===" << endl;
    
    const int N = 200;
    Matrix X;
    vector<int> y;
    std::default_random_engine generator(42);
    std::uniform_real_distribution<double> uniformDist(0.0, 10.0);
    
    // Generate 2D points and set label = 1 if (x + y) > 10, 0 otherwise.
    for (int i = 0; i < N; i++) {
        double x1 = uniformDist(generator);
        double x2 = uniformDist(generator);
        X.push_back({x1, x2});
        y.push_back((x1 + x2 > 10.0) ? 1 : 0);
    }

    // Define parameters for basis expansions.
    const int polyDegree = 2;
    // For sigmoidal and gaussian basis, obtain centers from the data.
    vector<double> data1D; // collect all values from, say, x1 dimension for simplicity.
    for (const auto &row : X)
        data1D.push_back(row[0]);
    int numCenters = 5;
    double s_value = 1.0;
    vector<double> sigmoidCenters = genSigmoidCenters(data1D, numCenters);

    vector<double> gaussCenters = genCenters(data1D, numCenters);
    double gauss_s = 1.0;

    // Helper lambda: run LR experiment given feature design matrix and basis name.
    auto runLRandOutput = [&](const string &basisName, const Matrix &Phi) {
        LogisticRegression lr(0.001, 1000, 1e-6);
        lr.fit(Phi, y);
        vector<int> predictions = lr.predict(Phi);
        double acc = computeAccuracy(y, predictions);
        cout << basisName << " basis accuracy: " << acc << "%" << endl;
        
        // Write predictions CSV.
        {
            std::ofstream file("./results/synthetic_" + basisName + "_predictions.csv");
            file << "x1,x2,true_label,predicted_label\n";
            for (size_t i = 0; i < X.size(); i++) {
                file << X[i][0] << "," << X[i][1] << "," << y[i] << "," << predictions[i] << "\n";
            }
        }
        // Write cost history CSV.
        writeVectorToCSV("./results/synthetic_" + basisName + "_cost_history.csv",
                           lr.getCostHistory(), "Iteration,Cost");
        // Write confusion matrix CSV.
        vector<vector<int>> cm = confusionMatrix(y, predictions);
        writeMatrixToCSV("./results/synthetic_" + basisName + "_confusion_matrix.csv", cm);
    };

    // 1. Polynomial basis.
    Matrix Phi_poly = genPolyFeatures(X, polyDegree);
    runLRandOutput("poly", Phi_poly);

    // 2. Sigmoidal basis.
    Matrix Phi_sigmoid = calcSigmoidalBasisFeaturesM(X, sigmoidCenters, s_value);
    // Add bias term.
    for (auto &row : Phi_sigmoid)
        row.insert(row.begin(), 1.0);
    runLRandOutput("sigmoid", Phi_sigmoid);

    // 3. Gaussian basis.
    Matrix Phi_gauss = calcGaussBasisFeaturesM(X, gaussCenters, gauss_s);
    // Add bias term.
    for (auto &row : Phi_gauss)
        row.insert(row.begin(), 1.0);
    runLRandOutput("gaussian", Phi_gauss);

    // Optionally, generate a grid of points and use the learned LR from each basis to create
    // a CSV file for plotting decision boundaries with Python.
    // (Omitted here for brevity; you can loop over grid points, compute predictProb, and output CSV.)
}

////////////////////////////////////////////////////////////////////////////////
// Iris Experiment: Load binary Iris dataset and run logistic regression with basis expansion.
////////////////////////////////////////////////////////////////////////////////
void runIrisExperiment() {
    cout << "=== Running Iris Experiment ===" << endl;
    
    // Open the Iris CSV.
    std::ifstream file("./datasets/Iris_Classification/Iris_Classification.csv");
    if (!file.is_open()) {
        cerr << "Error: Unable to open Iris_Classification.csv" << endl;
        return;
    }
    string line;
    std::getline(file, line); // skip header
    
    Matrix X;
    vector<int> y;
    // Use two features: PetalLengthCm and PetalWidthCm.
    // We choose only the first 100 rows (assuming rows 1-50 are "Iris-setosa" and 51-100 are "Iris-versicolor")
    // (Alternatively, filter by species.)
    int count = 0;
    while (std::getline(file, line) && count < 100) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        vector<string> tokens;
        string token;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() < 6) continue;
        // tokens[3] = PetalLengthCm, tokens[4] = PetalWidthCm, tokens[5] = Species
        double petalLen = std::stod(tokens[3]);
        double petalWid = std::stod(tokens[4]);
        // Map species: use Iris-setosa as label 0, Iris-versicolor as label 1.
        int label = (tokens[5] == "Iris-setosa") ? 0 : 1;
        X.push_back({petalLen, petalWid});
        y.push_back(label);
        count++;
    }
    file.close();

    // Define basis parameters as before.
    const int polyDegree = 2;
    vector<double> data1D;
    for (const auto &row : X)
        data1D.push_back(row[0]);
    int numCenters = 5;
    double s_value = 1.0;
    vector<double> sigmoidCenters = genSigmoidCenters(data1D, numCenters);
    vector<double> gaussCenters = genCenters(data1D, numCenters);
    double gauss_s = 1.0;

    auto runLRandOutput = [&](const string &basisName, const Matrix &Phi) {
        LogisticRegression lr(0.001, 1000, 1e-6);
        lr.fit(Phi, y);
        vector<int> predictions = lr.predict(Phi);
        double acc = computeAccuracy(y, predictions);
        cout << "Iris " << basisName << " basis accuracy: " << acc << "%" << endl;
        
        // Write predictions.
        {
            std::ofstream file("./results/iris_" + basisName + "_predictions.csv");
            file << "PetalLength,PetalWidth,true_label,predicted_label\n";
            for (size_t i = 0; i < X.size(); i++) {
                file << X[i][0] << "," << X[i][1] << "," << y[i] << "," << predictions[i] << "\n";
            }
        }
        // Cost history.
        writeVectorToCSV("./results/iris_" + basisName + "_cost_history.csv", lr.getCostHistory(),
                           "Iteration,Cost");
        // Confusion matrix.
        vector<vector<int>> cm = confusionMatrix(y, predictions);
        writeMatrixToCSV("./results/iris_" + basisName + "_confusion_matrix.csv", cm);
    };

    // 1. Polynomial basis.
    Matrix Phi_poly = genPolyFeatures(X, polyDegree);
    runLRandOutput("poly", Phi_poly);

    // 2. Sigmoidal basis.
    Matrix Phi_sigmoid = calcSigmoidalBasisFeaturesM(X, sigmoidCenters, s_value);
    for (auto &row : Phi_sigmoid)
        row.insert(row.begin(), 1.0);
    runLRandOutput("sigmoid", Phi_sigmoid);

    // 3. Gaussian basis.
    Matrix Phi_gauss = calcGaussBasisFeaturesM(X, gaussCenters, gauss_s);
    for (auto &row : Phi_gauss)
        row.insert(row.begin(), 1.0);
    runLRandOutput("gaussian", Phi_gauss);

    // (Optionally, produce a grid for decision boundary plotting.)
}

////////////////////////////////////////////////////////////////////////////////
// Titanic Experiment: Load train and test CSVs and run LR (and GDA) on Titanic data.
////////////////////////////////////////////////////////////////////////////////
void runTitanicExperiment() {
    cout << "=== Running Titanic Experiment ===" << endl;
    
    // Load training data from "train_Titanic.csv"
    std::ifstream trainFile("./datasets/Titanic_Classification/train_Titanic.csv");
    if (!trainFile.is_open()) {
        cerr << "Error: Unable to open train_Titanic.csv" << endl;
        return;
    }
    string line;
    std::getline(trainFile, line); // header
    Matrix X_train;
    vector<int> y_train;
    while (std::getline(trainFile, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        vector<string> tokens;
        string token;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        // Expected columns: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        if (tokens.size() < 12) continue;
        try {
            int survived = std::stoi(tokens[1]);
            double pclass = std::stod(tokens[2]);
            double age = std::stod(tokens[5]);
            double fare = std::stod(tokens[9]);
            X_train.push_back({pclass, age, fare});
            y_train.push_back(survived);
        } catch (...) {
            continue; // skip rows with conversion problems
        }
    }
    trainFile.close();
    
    // Load test data from "test_Titanic.csv" (assume similar format, but without Survived column or with it for evaluation)
    std::ifstream testFile("./datasets/Titanic_Classification/test_Titanic.csv");
    if (!testFile.is_open()) {
        cerr << "Error: Unable to open test_Titanic.csv" << endl;
        return;
    }
    std::getline(testFile, line); // header
    Matrix X_test;
    vector<int> y_test;
    while (std::getline(testFile, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        vector<string> tokens;
        string token;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        // For test data, assume columns: PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        // (No Survived label is given in some competitions but here assume some file contains true labels.)
        if (tokens.size() < 11) continue;
        // Here, we assume if the test CSV does include Survived (for evaluation) it is, say, the second column.
        // Adjust indices as needed.
        try {
            // For this example, we assume test data has the same format including true label in position 1.
            int survived = std::stoi(tokens[1]); // adjust if necessary
            double pclass = std::stod(tokens[2]);
            double age = std::stod(tokens[5]);
            double fare = std::stod(tokens[9]);
            X_test.push_back({pclass, age, fare});
            y_test.push_back(survived);
        } catch (...) {
            continue;
        }
    }
    testFile.close();
    
    // For Titanic we will use a simple (bias+linear) basis.
    Matrix Phi_train = genPolyFeatures(X_train, 1);
    Matrix Phi_test = genPolyFeatures(X_test, 1);
    
    // Logistic Regression on Titanic.
    LogisticRegression lr(0.01, 10000, 1e-6);
    lr.fit(Phi_train, y_train);
    vector<int> preds_lr = lr.predict(Phi_test);
    double acc_lr = computeAccuracy(y_test, preds_lr);
    cout << "Titanic Logistic Regression Accuracy: " << acc_lr << "%" << endl;
    {
        std::ofstream file("./results/titanic_lr_predictions.csv");
        file << "Pclass,Age,Fare,true_label,predicted_label\n";
        for (size_t i = 0; i < X_test.size(); i++) {
            file << X_test[i][0] << "," << X_test[i][1] << "," << X_test[i][2]
                 << "," << y_test[i] << "," << preds_lr[i] << "\n";
        }
    }
    vector<vector<int>> cm_lr = confusionMatrix(y_test, preds_lr);
    writeMatrixToCSV("./results/titanic_lr_confusion_matrix.csv", cm_lr);
    
    // Gaussian Discriminant Analysis on Titanic.
    GaussianDiscriminantAnalysis gda;
    gda.fit(X_train, y_train);
    vector<int> preds_gda = gda.predict(X_test);
    double acc_gda = computeAccuracy(y_test, preds_gda);
    cout << "Titanic GDA Accuracy: " << acc_gda << "%" << endl;
    {
        std::ofstream file("./results/titanic_gda_predictions.csv");
        file << "Pclass,Age,Fare,true_label,predicted_label\n";
        for (size_t i = 0; i < X_test.size(); i++) {
            file << X_test[i][0] << "," << X_test[i][1] << "," << X_test[i][2]
                 << "," << y_test[i] << "," << preds_gda[i] << "\n";
        }
    }
    vector<vector<int>> cm_gda = confusionMatrix(y_test, preds_gda);
    writeMatrixToCSV("./results/titanic_gda_confusion_matrix.csv", cm_gda);
}

////////////////////////////////////////////////////////////////////////////////
// Main function: run all experiments sequentially.
////////////////////////////////////////////////////////////////////////////////
int main() {
    runSyntheticExperiment();
    runIrisExperiment();
    runTitanicExperiment();
    return 0;
}