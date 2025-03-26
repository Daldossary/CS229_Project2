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

// Helper: Compute classification accuracy.
double computeAccuracy(const vector<int>& y_true, const vector<int>& y_pred) {
    if (y_true.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); i++) {
        if (y_true[i] == y_pred[i])
            correct++;
    }
    return 100.0 * correct / y_true.size();
}

// Helper: Compute confusion matrix for binary classification,
// where matrix[0][0]=True Negatives, [0][1]=False Positives,
//       matrix[1][0]=False Negatives, [1][1]=True Positives.
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

    // Generate a synthetic binary classification dataset.
    // 200 points uniformly from [0, 10]: label = 1 if x > 5; else 0.
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
    
    // For logistic regression, apply a polynomial basis expansion (degree 2).
    int degree = 2;
    Matrix Phi = genPolyFeatures(X, degree);
    
    // Train logistic regression.
    LogisticRegression logReg(0.01, 10000, 1e-6);
    logReg.fit(Phi, y);
    vector<int> preds_logReg = logReg.predict(Phi);
    double acc_logReg = computeAccuracy(y, preds_logReg);
    cout << "Logistic Regression Accuracy: " << acc_logReg << "%" << endl;
    
    // Train GDA on the raw features (without expansion).
    GaussianDiscriminantAnalysis gda;
    gda.fit(X, y);
    vector<int> preds_gda = gda.predict(X);
    double acc_gda = computeAccuracy(y, preds_gda);
    cout << "Gaussian Discriminant Analysis Accuracy: " << acc_gda << "%" << endl;
    
    // Compute confusion matrix for GDA.
    vector<vector<int>> cm = confusionMatrix(y, preds_gda);
    
    // Write GDA predictions to CSV.
    {
        std::ofstream file("./results/gda_predictions.csv");
        file << "x,true_label,predicted_label\n";
        for (size_t i = 0; i < X.size(); i++) {
            file << X[i][0] << "," << y[i] << "," << preds_gda[i] << "\n";
        }
    }
    // Write confusion matrix for GDA.
    {
        std::ofstream file("./results/gda_confusion_matrix.csv");
        file << " ,Predicted_0,Predicted_1\n";
        file << "True_0," << cm[0][0] << "," << cm[0][1] << "\n";
        file << "True_1," << cm[1][0] << "," << cm[1][1] << "\n";
    }
    // Write logistic regression predictions for comparison.
    {
        std::ofstream file("./results/logReg_predictions.csv");
        file << "x,true_label,predicted_label\n";
        for (size_t i = 0; i < X.size(); i++) {
            file << X[i][0] << "," << y[i] << "," << preds_logReg[i] << "\n";
        }
    }
    // Write GDA model parameters (class means, shared covariance, and priors) to CSV files.
    {
        auto means = gda.getClassMeans();
        auto priors = gda.getClassPriors();
        auto classLabels = gda.getClasses();  // NEW: retrieve class labels
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