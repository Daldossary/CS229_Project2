#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "./src/gaussianDiscriminantAnalysis.h"
#include "./src/logisticRegression.h"
#include "./src/polyReg.h"

using std::cout;
using std::endl;
using std::vector;
using Matrix = std::vector<std::vector<double>>;

double computeAccuracy(const vector<int>& y_true, const vector<int>& y_pred) {
    if (y_true.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); i++) {
        if (y_true[i] == y_pred[i])
            correct++;
    }
    return 100.0 * correct / y_true.size();
}

vector<vector<int>> confusionMatrix(const vector<int>& y_true, const vector<int>& y_pred) {
    vector<vector<int>> cm(2, vector<int>(2, 0));
    for (size_t i = 0; i < y_true.size(); i++) {
        if (y_true[i] == 0) {
            if (y_pred[i] == 0)
                cm[0][0]++;
            else
                cm[0][1]++;
        } else {
            if (y_pred[i] == 1)
                cm[1][1]++;
            else
                cm[1][0]++;
        }
    }
    return cm;
}

int main() {
    cout << "==== Testing Gaussian Discriminant Analysis (GDA) ====" << endl;

    const int N = 200;
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
    
    int degree = 2;
    Matrix Phi = genPolyFeatures(X, degree);
    
    LogisticRegression logReg(0.01, 10000, 1e-6);
    logReg.fit(Phi, y);
    vector<int> preds_logReg = logReg.predict(Phi);
    double acc_logReg = computeAccuracy(y, preds_logReg);
    cout << "Logistic Regression Accuracy: " << acc_logReg << "%" << endl;
    
    GaussianDiscriminantAnalysis gda;
    gda.fit(X, y);
    vector<int> preds_gda = gda.predict(X);
    double acc_gda = computeAccuracy(y, preds_gda);
    cout << "Gaussian Discriminant Analysis Accuracy: " << acc_gda << "%" << endl;
    
    vector<vector<int>> cm = confusionMatrix(y, preds_gda);
    
    {
        std::ofstream file("./results/gda_predictions.csv");
        file << "x,true_label,predicted_label\n";
        for (size_t i = 0; i < X.size(); i++) {
            file << X[i][0] << "," << y[i] << "," << preds_gda[i] << "\n";
        }
    }
    {
        std::ofstream file("./results/gda_confusion_matrix.csv");
        file << " ,Predicted_0,Predicted_1\n";
        file << "True_0," << cm[0][0] << "," << cm[0][1] << "\n";
        file << "True_1," << cm[1][0] << "," << cm[1][1] << "\n";
    }
    {
        std::ofstream file("./results/logReg_predictions.csv");
        file << "x,true_label,predicted_label\n";
        for (size_t i = 0; i < X.size(); i++) {
            file << X[i][0] << "," << y[i] << "," << preds_logReg[i] << "\n";
        }
    }
    {
        auto means = gda.getClassMeans();
        auto priors = gda.getClassPriors();
        auto classLabels = gda.getClasses();
        std::ofstream file("./results/gda_class_means.csv");
        file << "Class,Prior";
        for (size_t j = 0; j < means[0].size(); j++) {
            file << ",mu_" << j;
        }
        file << "\n";
        for (size_t idx = 0; idx < means.size(); idx++) {
            file << classLabels[idx] << "," << priors[idx];
            for (size_t j = 0; j < means[idx].size(); j++) {
                file << "," << means[idx][j];
            }
            file << "\n";
        }
        file.close();
    }
    {
        Matrix cov = gda.getCovariance();
        std::ofstream file("./results/gda_shared_covariance.csv");
        for (size_t i = 0; i < cov.size(); i++) {
            for (size_t j = 0; j < cov[i].size(); j++) {
                file << cov[i][j];
                if (j < cov[i].size()-1)
                    file << ",";
            }
            file << "\n";
        }
        file.close();
    }
    
    return 0;
}
