#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "./src/logisticRegression.h"
#include "./src/polyReg.h"
#include "./src/sigmoidalBasis.h"
#include "./src/gaussianBasis.h"
#include "./src/gaussianDiscriminantAnalysis.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using Matrix = std::vector<std::vector<double>>;

double computeAccuracy(const vector<int> &actual, const vector<int> &predicted)
{
    int correct = 0;
    for (size_t i = 0; i < actual.size(); i++)
    {
        if (actual[i] == predicted[i])
            correct++;
    }
    return 100.0 * correct / actual.size();
}

vector<vector<int>> confusionMatrix(const vector<int> &actual, const vector<int> &predicted)
{
    vector<vector<int>> cm(2, vector<int>(2, 0));
    for (size_t i = 0; i < actual.size(); i++)
    {
        if (actual[i] == 0)
        {
            if (predicted[i] == 0)
                cm[0][0]++;
            else
                cm[0][1]++;
        }
        else
        {
            if (predicted[i] == 1)
                cm[1][1]++;
            else
                cm[1][0]++;
        }
    }
    return cm;
}

void writeVectorToCSV(const string &filename, const vector<double> &vec, const string &header)
{
    std::ofstream file(filename);
    file << header << "\n";
    for (size_t i = 0; i < vec.size(); i++)
    {
        file << i << "," << vec[i] << "\n";
    }
    file.close();
}

void writeMatrixToCSV(const string &filename, const vector<vector<int>> &mat)
{
    std::ofstream file(filename);
    for (const auto &row : mat)
    {
        for (size_t j = 0; j < row.size(); j++)
        {
            file << row[j] << ((j < row.size() - 1) ? "," : "\n");
        }
    }
    file.close();
}

void initSummaryCSV()
{
    std::ofstream summaryFile("./results/summary.csv");
    summaryFile << "Experiment,Method,Accuracy\n";
    summaryFile.close();
}

void appendSummaryResult(const string &experiment, const string &method, double accuracy)
{
    std::ofstream summaryFile("./results/summary.csv", std::ios::app);
    summaryFile << experiment << "," << method << "," << accuracy << "\n";
    summaryFile.close();
}

vector<string> parseCSVLine(const string &line) {
    vector<string> result;
    bool inQuotes = false;
    string field;
    for (char c : line) {
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            result.push_back(field);
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    result.push_back(field);
    return result;
}

void runSyntheticExperiment()
{
    cout << "=== Running Synthetic Experiment ===" << endl;

    const int N = 200;
    Matrix X;
    vector<int> y;
    std::default_random_engine generator(42);
    std::uniform_real_distribution<double> uniformDist(0.0, 10.0);

    for (int i = 0; i < N; i++)
    {
        double x1 = uniformDist(generator);
        double x2 = uniformDist(generator);
        X.push_back({x1, x2});
        y.push_back((x1 + x2 > 10.0) ? 1 : 0);
    }

    const int polyDegree = 2;
    vector<double> data1D;
    for (const auto &row : X)
        data1D.push_back(row[0]);
    int numCenters = 5;
    double s_value = 1.0;
    vector<double> sigmoidCenters = genSigmoidCenters(data1D, numCenters);
    vector<double> gaussCenters = genCenters(data1D, numCenters);
    double gauss_s = 1.0;

    auto runLRandOutput = [&](const string &basisName, const Matrix &Phi)
    {
        LogisticRegression lr(0.001, 1000, 1e-6);
        lr.fit(Phi, y);
        vector<int> predictions = lr.predict(Phi);
        double acc = computeAccuracy(y, predictions);
        cout << basisName << " basis accuracy: " << acc << "%" << endl;
        appendSummaryResult("Synthetic", basisName, acc);

        {
            std::ofstream file("./results/synthetic_" + basisName + "_predictions.csv");
            file << "x1,x2,true_label,predicted_label\n";
            for (size_t i = 0; i < X.size(); i++)
            {
                file << X[i][0] << "," << X[i][1] << "," << y[i] << "," << predictions[i] << "\n";
            }
        }
        writeVectorToCSV("./results/synthetic_" + basisName + "_cost_history.csv", lr.getCostHistory(), "Iteration,Cost");
        vector<vector<int>> cm = confusionMatrix(y, predictions);
        writeMatrixToCSV("./results/synthetic_" + basisName + "_confusion_matrix.csv", cm);
    };

    Matrix Phi_poly = genPolyFeatures(X, polyDegree);
    runLRandOutput("poly", Phi_poly);

    Matrix Phi_sigmoid = calcSigmoidalBasisFeaturesM(X, sigmoidCenters, s_value);
    for (auto &row : Phi_sigmoid)
        row.insert(row.begin(), 1.0);
    runLRandOutput("sigmoid", Phi_sigmoid);

    Matrix Phi_gauss = calcGaussBasisFeaturesM(X, gaussCenters, gauss_s);
    for (auto &row : Phi_gauss)
        row.insert(row.begin(), 1.0);
    runLRandOutput("gaussian", Phi_gauss);
}

void runIrisExperiment()
{
    cout << "=== Running Iris Experiment ===" << endl;

    std::ifstream file("./datasets/Iris_Classification/Iris_Classification.csv");
    if (!file.is_open())
    {
        cerr << "Error: Unable to open Iris_Classification.csv" << endl;
        return;
    }
    string line;
    std::getline(file, line);

    Matrix X;
    vector<int> y;
    int count = 0;
    while (std::getline(file, line) && count < 100)
    {
        if (line.empty()) continue;
        std::istringstream ss(line);
        vector<string> tokens;
        string token;
        while (std::getline(ss, token, ','))
        {
            tokens.push_back(token);
        }
        if (tokens.size() < 6)
            continue;
        double petalLen = std::stod(tokens[3]);
        double petalWid = std::stod(tokens[4]);
        int label = (tokens[5] == "Iris-setosa") ? 0 : 1;
        X.push_back({petalLen, petalWid});
        y.push_back(label);
        count++;
    }
    file.close();

    const int polyDegree = 2;
    vector<double> data1D;
    for (const auto &row : X)
        data1D.push_back(row[0]);
    int numCenters = 5;
    double s_value = 1.0;
    vector<double> sigmoidCenters = genSigmoidCenters(data1D, numCenters);
    vector<double> gaussCenters = genCenters(data1D, numCenters);
    double gauss_s = 1.0;

    auto runLRandOutput = [&](const string &basisName, const Matrix &Phi)
    {
        LogisticRegression lr(0.001, 1000, 1e-6);
        lr.fit(Phi, y);
        vector<int> predictions = lr.predict(Phi);
        double acc = computeAccuracy(y, predictions);
        cout << "Iris " << basisName << " basis accuracy: " << acc << "%" << endl;
        appendSummaryResult("Iris", basisName, acc);

        {
            std::ofstream file("./results/iris_" + basisName + "_predictions.csv");
            file << "PetalLength,PetalWidth,true_label,predicted_label\n";
            for (size_t i = 0; i < X.size(); i++)
            {
                file << X[i][0] << "," << X[i][1] << "," << y[i] << "," << predictions[i] << "\n";
            }
        }
        writeVectorToCSV("./results/iris_" + basisName + "_cost_history.csv", lr.getCostHistory(), "Iteration,Cost");
        vector<vector<int>> cm = confusionMatrix(y, predictions);
        writeMatrixToCSV("./results/iris_" + basisName + "_confusion_matrix.csv", cm);
    };

    Matrix Phi_poly = genPolyFeatures(X, polyDegree);
    runLRandOutput("poly", Phi_poly);

    Matrix Phi_sigmoid = calcSigmoidalBasisFeaturesM(X, sigmoidCenters, s_value);
    for (auto &row : Phi_sigmoid)
        row.insert(row.begin(), 1.0);
    runLRandOutput("sigmoid", Phi_sigmoid);

    Matrix Phi_gauss = calcGaussBasisFeaturesM(X, gaussCenters, gauss_s);
    for (auto &row : Phi_gauss)
        row.insert(row.begin(), 1.0);
    runLRandOutput("gaussian", Phi_gauss);
}

// void runTitanicExperiment() {
//     cout << "=== Running Titanic Experiment (Split-Version) ===" << endl;
    
//     std::ifstream file("./datasets/Titanic_Classification/train_Titanic.csv");
//     if (!file.is_open()) {
//         cerr << "Error: Unable to open train_Titanic.csv" << endl;
//         return;
//     }
//     string line;
//     std::getline(file, line);

//     Matrix X;
//     vector<int> y;
//     while (std::getline(file, line)) {
//         if (line.empty()) continue;
//         vector<string> tokens = parseCSVLine(line);
//         if (tokens.size() < 12) continue;
//         try {
//             int survived = std::stoi(tokens[1]);
//             double pclass = std::stod(tokens[2]);
//             double age = tokens[5].empty() ? 0.0 : std::stod(tokens[5]);
//             double fare = tokens[9].empty() ? 0.0 : std::stod(tokens[9]);
//             X.push_back({pclass, age, fare});
//             y.push_back(survived);
//         } catch (...) {
//             continue;
//         }
//     }
//     file.close();
    
//     if (X.empty()) {
//         cerr << "No valid Titanic data was loaded." << endl;
//         return;
//     }
    
//     size_t total = X.size();
//     vector<size_t> indices(total);
//     std::iota(indices.begin(), indices.end(), 0);
//     std::shuffle(indices.begin(), indices.end(), std::default_random_engine(42));
    
//     size_t trainSize = static_cast<size_t>(total * 0.7);
//     Matrix X_train, X_test;
//     vector<int> y_train, y_test;
//     for (size_t i = 0; i < indices.size(); i++) {
//         if (i < trainSize) {
//             X_train.push_back(X[indices[i]]);
//             y_train.push_back(y[indices[i]]);
//         } else {
//             X_test.push_back(X[indices[i]]);
//             y_test.push_back(y[indices[i]]);
//         }
//     }
    
//     cout << "Titanic dataset: " << total << " samples; " 
//          << X_train.size() << " training, " 
//          << X_test.size() << " testing." << endl;
    
//     Matrix Phi_train = genPolyFeatures(X_train, 1);
//     Matrix Phi_test = genPolyFeatures(X_test, 1);
    
//     LogisticRegression lr(0.01, 10000, 1e-6);
//     lr.fit(Phi_train, y_train);
//     vector<int> preds_lr = lr.predict(Phi_test);
//     double acc_lr = computeAccuracy(y_test, preds_lr);
//     cout << "Titanic Logistic Regression Accuracy: " << acc_lr << "%" << endl;
//     appendSummaryResult("Titanic", "LR", acc_lr);
    
//     {
//         std::ofstream out("./results/titanic_lr_predictions.csv");
//         out << "Pclass,Age,Fare,true_label,predicted_label\n";
//         for (size_t i = 0; i < X_test.size(); i++) {
//             out << X_test[i][0] << "," << X_test[i][1] << "," << X_test[i][2]
//                 << "," << y_test[i] << "," << preds_lr[i] << "\n";
//         }
//     }
//     vector<vector<int>> cm_lr = confusionMatrix(y_test, preds_lr);
//     writeMatrixToCSV("./results/titanic_lr_confusion_matrix.csv", cm_lr);
    
//     GaussianDiscriminantAnalysis gda;
//     gda.fit(X_train, y_train);
//     vector<int> preds_gda = gda.predict(X_test);
//     double acc_gda = computeAccuracy(y_test, preds_gda);
//     cout << "Titanic GDA Accuracy: " << acc_gda << "%" << endl;
//     appendSummaryResult("Titanic", "GDA", acc_gda);
    
//     {
//         std::ofstream out("./results/titanic_gda_predictions.csv");
//         out << "Pclass,Age,Fare,true_label,predicted_label\n";
//         for (size_t i = 0; i < X_test.size(); i++) {
//             out << X_test[i][0] << "," << X_test[i][1] << "," << X_test[i][2]
//                 << "," << y_test[i] << "," << preds_gda[i] << "\n";
//         }
//     }
//     vector<vector<int>> cm_gda = confusionMatrix(y_test, preds_gda);
//     writeMatrixToCSV("./results/titanic_gda_confusion_matrix.csv", cm_gda);
// }

int main()
{
    initSummaryCSV();
    
    runSyntheticExperiment();
    runIrisExperiment();
    // runTitanicExperiment();
    return 0;
}
