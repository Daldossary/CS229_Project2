// testExperimentsTuning.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <execution>   // For parallel execution policies

#include "./src/matrixOperations.h"   // Provides Matrix alias and functions (transposeM, multMs, invertM)
#include "./src/polyReg.h"            // Provides genPolyFeatures (univariate) and genMultiVarPolyFeatures (multivariate)
#include "./src/closedFormSol.h"      // For ML solution (closedFormSingleVar)
#include "./src/bayesianLinearReg.h"  // Provides BayesianLinearRegression class
#include "./src/statFunctions.h"      // Provides calcMean, calcStdDev, etc.
#include "./src/parseCSV.h"           // Provides parseCSV() using a delimiter

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;
using Matrix = std::vector<std::vector<double>>;

//***********************************************************************
// Helper: Normalize a matrix column‐wise (z‐score standardization)
// Uses parallel for_each.
Matrix normalizeMatrix(const Matrix &X) {
    size_t m = X.size();
    if (m == 0) return X; 
    size_t n = X[0].size();
    Matrix X_norm = X; // copy

    // Create an index vector for columns [0..n-1]
    vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t j) {
        vector<double> col;
        col.reserve(m);
        for (size_t i = 0; i < m; i++) {
            col.push_back(X[i][j]);
        }
        double mean = calcMean(col);
        double stddev = calcStdDev(col);
        if (stddev == 0) stddev = 1.0;
        for (size_t i = 0; i < m; i++) {
            X_norm[i][j] = (X[i][j] - mean) / stddev;
        }
    });
    return X_norm;
}

// Helper: Parse CSV with a given delimiter
Matrix parseCSVWithDelimiter(const string &filename, char delimiter, bool skipHeader = true) {
    std::ifstream file(filename);
    Matrix data;
    if (!file) {
        cerr << "File " << filename << " not found." << endl;
        return data;
    }
    string line;
    if (skipHeader && std::getline(file, line))
        ; // header skipped
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        vector<double> row;
        std::istringstream ss(line);
        string token;
        while (std::getline(ss, token, delimiter)) {
            try {
                double value = std::stod(token);
                row.push_back(value);
            } catch (...) {
                row.push_back(0.0);
            }
        }
        data.push_back(row);
    }
    return data;
}

//***********************************************************************
// Define a simple structure to hold tuning results
struct TuningResult {
    double best_alpha;
    double best_beta;
    double best_val_mse; // validation mean squared error
    // We store the best hyperparameters so we can later refit the model.
};

// Compute Mean Squared Error.
double computeMSE(const vector<double>& y_true, const vector<double>& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return y_true.empty() ? 0.0 : sum / y_true.size();
}

//***********************************************************************
// Hyperparameter tuning for Bayesian Linear Regression
// We split the provided training set (Phi_train, t_train) into (say) 80% for tuning-train and 20% for validation.
TuningResult tuneBayesianHyperparameters(const Matrix &Phi_train, const vector<double> &t_train) {
    size_t N = Phi_train.size();
    size_t n_train = static_cast<size_t>(N * 0.8);
    Matrix Phi_tuneTrain(Phi_train.begin(), Phi_train.begin() + n_train);
    vector<double> t_tuneTrain(t_train.begin(), t_train.begin() + n_train);
    Matrix Phi_val(Phi_train.begin() + n_train, Phi_train.end());
    vector<double> t_val(t_train.begin() + n_train, t_train.end());

    // Candidate hyperparameters for α and β
    vector<double> candidate_alphas = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10};
    vector<double> candidate_betas  = {1e1, 1e2, 1e3, 1e4};

    double best_mse = 1e12;
    double best_alpha = candidate_alphas[0];
    double best_beta = candidate_betas[0];

    for (double alpha : candidate_alphas) {
        for (double beta : candidate_betas) {
            BayesianLinearRegression model(alpha, beta);
            model.fit(Phi_tuneTrain, t_tuneTrain);
            vector<double> pred_val = model.predict(Phi_val);
            double mse = computeMSE(t_val, pred_val);
            if (mse < best_mse) {
                best_mse = mse;
                best_alpha = alpha;
                best_beta = beta;
            }
        }
    }
    TuningResult result;
    result.best_alpha = best_alpha;
    result.best_beta = best_beta;
    result.best_val_mse = best_mse;
    return result;
}

//***********************************************************************
// Experiment on synthetic regression dataset with hyperparameter tuning
void runSyntheticExperimentTuning() {
    cout << "=== Synthetic Dataset Experiment with Hyperparameter Tuning ===" << endl;
    
    const int N_train = 50;
    const int N_test = 20;
    vector<vector<double>> X_train, X_test;
    vector<double> t_train;
    
    std::default_random_engine generator(42);
    std::uniform_real_distribution<double> uniformDist(0.0, 10.0);
    std::normal_distribution<double> noise(0.0, 1.0);
    
    // Generate training data: true function: t = 3*x + 2 + noise
    for (int i = 0; i < N_train; i++) {
        double x = uniformDist(generator);
        X_train.push_back({x});
        double t_val = 3.0 * x + 2.0 + noise(generator);
        t_train.push_back(t_val);
    }
    
    // Generate test data (sorted)
    for (int i = 0; i < N_test; i++) {
        double x = 10.0 * i / (N_test - 1);
        X_test.push_back({x});
    }
    
    // Use polynomial basis expansion (degree 2)
    int degree = 2;
    Matrix Phi_train = genPolyFeatures(X_train, degree);
    Matrix Phi_test = genPolyFeatures(X_test, degree);

    // Grid-search over hyperparameters using a split of Phi_train
    TuningResult tuneRes = tuneBayesianHyperparameters(Phi_train, t_train);
    
    // Output the best hyperparameters to CSV
    {
        std::ofstream tuningFile("./results/synthetic_tuning_results.csv");
        tuningFile << "BestAlpha,BestBeta,ValidationMSE\n";
        tuningFile << tuneRes.best_alpha << "," << tuneRes.best_beta << "," << tuneRes.best_val_mse << "\n";
        tuningFile.close();
    }
    
    // Refit the Bayesian model on the full training set using the best hyperparameters
    BayesianLinearRegression blr(tuneRes.best_alpha, tuneRes.best_beta);
    blr.fit(Phi_train, t_train);
    
    // Get predictions and predictive variances on test data
    vector<double> pred_test = blr.predict(Phi_test);
    vector<double> predVar_test = blr.predictiveVariance(Phi_test);
    
    // Output predictions to CSV
    {
        std::ofstream file("./results/synthetic_bayesian_predictions_tuned.csv");
        file << "x,true,prediction,variance,lower95,upper95\n";
        for (size_t i = 0; i < X_test.size(); i++){
            double x = X_test[i][0];
            double true_val = 3.0 * x + 2.0;  // true function (without noise)
            double pred = pred_test[i];
            double var = predVar_test[i];
            double stddev = std::sqrt(var);
            double lower95 = pred - 1.96 * stddev;
            double upper95 = pred + 1.96 * stddev;
            file << x << "," << true_val << "," << pred << "," << var
                 << "," << lower95 << "," << upper95 << "\n";
        }
    }
    
    // Output Bayesian posterior weights
    {
        vector<double> weights = blr.getWeights();
        std::ofstream file("./results/synthetic_bayesian_weights_tuned.csv");
        file << "Index,Weight\n";
        for (size_t i = 0; i < weights.size(); i++)
            file << i << "," << weights[i] << "\n";
    }
    
    // Compute training MSE for reporting
    vector<double> pred_train = blr.predict(Phi_train);
    double mse_train = computeMSE(t_train, pred_train);
    {
        std::ofstream file("./results/synthetic_error_metrics_tuned.csv");
        file << "Method,TrainingMSE\n";
        file << "Bayesian_Tuned," << mse_train << "\n";
    }
    
    cout << "Synthetic experiment completed. Best hyperparameters: α = " << tuneRes.best_alpha
         << ", β = " << tuneRes.best_beta << ". Results written to the results folder." << endl;
}

//***********************************************************************
// Experiment on provided Wine Quality (red) regression dataset with tuning
void runWineQualityExperimentTuning() {
    cout << "=== Wine Quality (Red) Experiment with Hyperparameter Tuning ===" << endl;
    
    string filename = "./datasets/WineQuality_Red_Regression/winequality-red_Regression.csv";
    // Wine dataset uses ';' as delimiter.
    Matrix wineData = parseCSVWithDelimiter(filename, ';', true);
    
    // In the wine dataset, assume the last column is target (quality) and the rest are features.
    Matrix X_wine;
    vector<double> t_wine;
    for (size_t i = 0; i < wineData.size(); i++) {
        if (wineData[i].empty()) continue;
        vector<double> features(wineData[i].begin(), wineData[i].end() - 1);
        X_wine.push_back(features);
        t_wine.push_back(wineData[i].back());
    }
    
    // Normalize features
    Matrix X_wine_norm = normalizeMatrix(X_wine);
    
    // Use multivariate polynomial features (degree 2)
    int wineDegree = 2;
    Matrix Phi_wine = genMultiVarPolyFeatures(X_wine_norm, wineDegree);
    
    // Split dataset into training and test sets (80% training, 20% test)
    size_t total = Phi_wine.size();
    size_t train_size = static_cast<size_t>(total * 0.8);
    Matrix Phi_wine_train(Phi_wine.begin(), Phi_wine.begin() + train_size);
    Matrix Phi_wine_test(Phi_wine.begin() + train_size, Phi_wine.end());
    vector<double> t_wine_train(t_wine.begin(), t_wine.begin() + train_size);
    vector<double> t_wine_test(t_wine.begin() + train_size, t_wine.end());
    
    // Hyperparameter tuning (grid-search on training set split)
    TuningResult tuneRes = tuneBayesianHyperparameters(Phi_wine_train, t_wine_train);
    
    // Output tuning results
    {
        std::ofstream file("./results/wine_red_tuning_results.csv");
        file << "BestAlpha,BestBeta,ValidationMSE\n";
        file << tuneRes.best_alpha << "," << tuneRes.best_beta << "," << tuneRes.best_val_mse << "\n";
    }
    
    // Refit model on full wine training set using best hyperparameters
    BayesianLinearRegression blr_wine(tuneRes.best_alpha, tuneRes.best_beta);
    blr_wine.fit(Phi_wine_train, t_wine_train);
    
    // Predictions on test set
    vector<double> pred_test = blr_wine.predict(Phi_wine_test);
    vector<double> predVar_test = blr_wine.predictiveVariance(Phi_wine_test);
    
    // Write test predictions to CSV (include original normalized features for reference)
    {
        std::ofstream file("./results/wine_red_bayesian_predictions_tuned.csv");
        // For multivariate data, we output features as a combined string (or as individual columns if preferred)
        file << "features,true,prediction,variance,lower95,upper95\n";
        for (size_t i = 0; i < Phi_wine_test.size(); i++){
            // Write normalized features (comma‐separated)
            for (size_t j = 0; j < X_wine_norm[train_size + i].size(); j++) {
                file << X_wine_norm[train_size + i][j];
                if (j < X_wine_norm[train_size + i].size() - 1)
                    file << " ";
            }
            double pred = pred_test[i];
            double var = predVar_test[i];
            double stddev = std::sqrt(var);
            double lower95 = pred - 1.96 * stddev;
            double upper95 = pred + 1.96 * stddev;
            file << "," << t_wine_test[i] << "," << pred << "," << var
                 << "," << lower95 << "," << upper95 << "\n";
        }
    }
    
    // Write model weights
    {
        vector<double> weights = blr_wine.getWeights();
        std::ofstream file("./results/wine_red_bayesian_weights_tuned.csv");
        file << "Index,Weight\n";
        for (size_t i = 0; i < weights.size(); i++)
            file << i << "," << weights[i] << "\n";
    }
    
    // Compute training MSE for wine training set
    vector<double> pred_train = blr_wine.predict(Phi_wine_train);
    double mse_train = computeMSE(t_wine_train, pred_train);
    {
        std::ofstream file("./results/wine_red_error_metrics_tuned.csv");
        file << "Method,TrainingMSE\n";
        file << "Bayesian_Tuned," << mse_train << "\n";
    }
    
    cout << "Wine quality (red) experiment completed. Best hyperparameters: α = " << tuneRes.best_alpha 
         << ", β = " << tuneRes.best_beta << ". Results written to the results folder." << endl;
}

//***********************************************************************
int main() {
    runSyntheticExperimentTuning();
    runWineQualityExperimentTuning();
    
    cout << "All experiments with hyperparameter tuning completed. Check the results folder." << endl;
    return 0;
}