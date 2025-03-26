#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <execution>  // For parallel execution
#include "./src/matrixOperations.h"         // Provides Matrix alias and functions: transposeM(), multMs(), invertM()
#include "./src/polyReg.h"                  // Provides genPolyFeatures() and genMultiVarPolyFeatures()
#include "./src/closedFormSol.h"            // Provides closedFormSingleVar() for ML solution
#include "./src/bayesianLinearReg.h"        // Provides the BayesianLinearRegression class
#include "./src/statFunctions.h"            // Provides calcMean(), calcStdDev(), etc.
#include "./src/parseCSV.h"                 // Provides parseCSV() with comma delimiter

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;
using Matrix = std::vector<std::vector<double>>;

//---------------------------------------------------------------------
// normalizeMatrix: standardize each column (zero mean, unit variance)
// Modified to update columns in parallel.
Matrix normalizeMatrix(const Matrix &X) {
    size_t m = X.size();
    if (m == 0) return X;
    size_t n = X[0].size();
    Matrix X_norm = X; // copy original data

    // For each column j, use parallel execution.
    vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t j) {
        vector<double> col;
        col.reserve(m);
        for (size_t i = 0; i < m; i++)
            col.push_back(X[i][j]);
        double mean = calcMean(col);
        double stddev = calcStdDev(col);
        if (stddev == 0) stddev = 1; // avoid division by zero
        for (size_t i = 0; i < m; i++) {
            X_norm[i][j] = (X[i][j] - mean) / stddev;
        }
    });
    
    return X_norm;
}

//---------------------------------------------------------------------
// parseCSVWithDelimiter: similar to parseCSV but using specified delimiter.
Matrix parseCSVWithDelimiter(const string &filename, char delimiter, bool skipHeader = true) {
    std::ifstream file(filename);
    Matrix data;
    if (!file) {
        cerr << "File " << filename << " not found." << endl;
        return data;
    }
    string line;
    if (skipHeader && std::getline(file, line)) {
        // header skipped
    }
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

// Compute Mean Squared Error.
double computeMSE(const vector<double>& y_true, const vector<double>& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return y_true.empty() ? 0 : sum / y_true.size();
}

//---------------------------------------------------------------------
// runSyntheticExperiment: Generate synthetic data, compute features,
//   train both ML and Bayesian models, and write outputs to CSV files.
void runSyntheticExperiment() {
    cout << "Running experiments on synthetic regression dataset..." << endl;
    const int N_train = 50;
    const int N_test = 20;
    vector<vector<double>> X_train, X_test;
    vector<double> t_train;
    
    std::default_random_engine generator(42);
    std::uniform_real_distribution<double> uniformDist(0.0, 10.0);
    std::normal_distribution<double> noise(0.0, 1.0);

    // Generate training data: t = 3*x + 2 + noise.
    for (int i = 0; i < N_train; i++) {
        double x = uniformDist(generator);
        X_train.push_back({ x });
        double t_val = 3.0 * x + 2.0 + noise(generator);
        t_train.push_back(t_val);
    }
    // Generate test inputs (sorted)
    for (int i = 0; i < N_test; i++) {
        double x = 10.0 * i / (N_test - 1);
        X_test.push_back({ x });
    }
    int degree = 2;
    Matrix Phi_train = genPolyFeatures(X_train, degree);
    Matrix Phi_test = genPolyFeatures(X_test, degree);
    
    // ----- Maximum Likelihood (ML) Regression -----
    vector<double> w_ml = closedFormSingleVar(Phi_train, t_train);
    vector<double> pred_test_ml(Phi_test.size(), 0.0);
    {
        // Parallelize prediction computation over test samples.
        vector<size_t> idx(Phi_test.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](size_t i) {
            double pred = 0.0;
            for (size_t j = 0; j < w_ml.size(); j++){
                pred += Phi_test[i][j] * w_ml[j];
            }
            pred_test_ml[i] = pred;
        });
    }
    vector<double> pred_train_ml(Phi_train.size(), 0.0);
    {
        vector<size_t> idx(Phi_train.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](size_t i) {
            double pred = 0.0;
            for (size_t j = 0; j < w_ml.size(); j++){
                pred += Phi_train[i][j] * w_ml[j];
            }
            pred_train_ml[i] = pred;
        });
    }
    double mse_ml = computeMSE(t_train, pred_train_ml);
    
    // ----- Bayesian Regression -----
    double alpha = 1e-3, beta = 1e+3;
    BayesianLinearRegression blr(alpha, beta);
    blr.fit(Phi_train, t_train);
    vector<double> pred_test_bayes(Phi_test.size(), 0.0);
    vector<size_t> idx_test(Phi_test.size());
    std::iota(idx_test.begin(), idx_test.end(), 0);
    std::for_each(std::execution::par, idx_test.begin(), idx_test.end(), [&](size_t i) {
        // Use the predict() method per row:
        // (Since our predict() is internally a loop, one could also parallelize that loop
        // over test instances if desired.)
        vector<vector<double>> singleRow = { Phi_test[i] };
        // We do a direct computation here:
        double pred = 0.0;
        vector<double> bayes_w = blr.getWeights();
        for (size_t j = 0; j < bayes_w.size(); j++) {
            pred += bayes_w[j] * Phi_test[i][j];
        }
        pred_test_bayes[i] = pred;
    });
    vector<double> pred_train_bayes = blr.predict(Phi_train);
    double mse_bayes = computeMSE(t_train, pred_train_bayes);
    vector<double> predVar_test = blr.predictiveVariance(Phi_test);
    
    // ----- Write outputs to CSV files -----
    {
        std::ofstream file_ml("./results/ml_predictions.csv");
        file_ml << "x,true,prediction\n";
        for (size_t i = 0; i < X_test.size(); i++){
            double x = X_test[i][0];
            double true_val = 3.0 * x + 2.0; // true function without noise
            file_ml << x << "," << true_val << "," << pred_test_ml[i] << "\n";
        }
    }
    {
        std::ofstream file_bayes("./results/bayesian_predictions.csv");
        file_bayes << "x,true,prediction,variance,lower95,upper95\n";
        for (size_t i = 0; i < X_test.size(); i++){
            double x = X_test[i][0];
            double true_val = 3.0 * x + 2.0;
            double pred = pred_test_bayes[i];
            double var = predVar_test[i];
            double stddev = std::sqrt(var);
            double lower95 = pred - 1.96 * stddev;
            double upper95 = pred + 1.96 * stddev;
            file_bayes << x << "," << true_val << "," << pred << "," << var
                       << "," << lower95 << "," << upper95 << "\n";
        }
    }
    {
        std::ofstream file_ml_w("./results/ml_weights.csv");
        file_ml_w << "Index,Weight\n";
        for (size_t i = 0; i < w_ml.size(); i++){
            file_ml_w << i << "," << w_ml[i] << "\n";
        }
    }
    {
        std::ofstream file_bayes_w("./results/bayesian_weights.csv");
        vector<double> bayes_w = blr.getWeights();
        file_bayes_w << "Index,Weight\n";
        for (size_t i = 0; i < bayes_w.size(); i++){
            file_bayes_w << i << "," << bayes_w[i] << "\n";
        }
    }
    {
        std::ofstream file_err("./results/error_metrics.csv");
        file_err << "Method,MSE\n";
        file_err << "ML," << mse_ml << "\n";
        file_err << "Bayesian," << mse_bayes << "\n";
    }
    
    cout << "Synthetic experiments completed." << endl;
}

//---------------------------------------------------------------------
// runWineQualityExperiment: Load the provided wine quality (red) dataset,
// normalize the features, perform a multivariate polynomial expansion,
// split into train/test (80/20), run both ML and Bayesian regression,
// and output predictions, weights, and error metrics.
void runWineQualityExperiment() {
    cout << "Running experiments on provided regression dataset (wine quality - red)..." << endl;
    string redFile = "./datasets/WineQuality_Red_Regression/winequality-red_Regression.csv";
    // Wine quality files are delimited by semicolons.
    Matrix wine_red = parseCSVWithDelimiter(redFile, ';', true);
    
    // In this dataset the last column is the target ("quality")
    Matrix X_wine;
    vector<double> t_wine;
    for (size_t i = 0; i < wine_red.size(); i++) {
        if (wine_red[i].empty()) continue;
        vector<double> row(wine_red[i].begin(), wine_red[i].end() - 1);
        X_wine.push_back(row);
        t_wine.push_back(wine_red[i].back());
    }
    // Normalize features column‐wise.
    Matrix X_wine_norm = normalizeMatrix(X_wine);
    
    // Use multivariate polynomial expansion (degree 2).
    int wineDegree = 2;
    Matrix Phi_wine = genMultiVarPolyFeatures(X_wine_norm, wineDegree);
    
    // Split dataset into 80% training, 20% test.
    size_t total = Phi_wine.size();
    size_t train_size = static_cast<size_t>(total * 0.8);
    Matrix Phi_wine_train(Phi_wine.begin(), Phi_wine.begin() + train_size);
    Matrix Phi_wine_test(Phi_wine.begin() + train_size, Phi_wine.end());
    vector<double> t_wine_train(t_wine.begin(), t_wine.begin() + train_size);
    vector<double> t_wine_test(t_wine.begin() + train_size, t_wine.end());
    
    // ----- ML Regression on Wine Dataset -----
    vector<double> w_ml_wine = closedFormSingleVar(Phi_wine_train, t_wine_train);
    vector<double> pred_train_ml_wine(Phi_wine_train.size(), 0.0);
    vector<double> pred_test_ml_wine(Phi_wine_test.size(), 0.0);
    {
        vector<size_t> idx(train_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](size_t i) {
            double sum = 0.0;
            for (size_t j = 0; j < w_ml_wine.size(); j++) {
                sum += Phi_wine_train[i][j] * w_ml_wine[j];
            }
            pred_train_ml_wine[i] = sum;
        });
    }
    {
        vector<size_t> idx(Phi_wine_test.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](size_t i) {
            double sum = 0.0;
            for (size_t j = 0; j < w_ml_wine.size(); j++) {
                sum += Phi_wine_test[i][j] * w_ml_wine[j];
            }
            pred_test_ml_wine[i] = sum;
        });
    }
    double mse_ml_wine = computeMSE(t_wine_train, pred_train_ml_wine);
    
    // ----- Bayesian Regression on Wine Dataset -----
    double alpha = 1e-3, beta = 1e+3;
    BayesianLinearRegression blr_wine(alpha, beta);
    blr_wine.fit(Phi_wine_train, t_wine_train);
    vector<double> pred_train_bayes_wine = blr_wine.predict(Phi_wine_train);
    vector<double> pred_test_bayes_wine = blr_wine.predict(Phi_wine_test);
    vector<double> predVar_test_wine = blr_wine.predictiveVariance(Phi_wine_test);
    double mse_bayes_wine = computeMSE(t_wine_train, pred_train_bayes_wine);
    
    // ----- Write outputs for Wine Regression Experiment -----
    {
        std::ofstream file_ml("./results/wine_red_ml_predictions.csv");
        // Output: we first output the original normalized feature values,
        // then the true target and the ML prediction.
        file_ml << "features,true,prediction\n";
        for (size_t i = 0; i < Phi_wine_test.size(); i++) {
            // Write features (comma–separated)
            for (size_t j = 0; j < X_wine_norm[train_size + i].size(); j++) {
                file_ml << X_wine_norm[train_size + i][j];
                if (j != X_wine_norm[train_size + i].size() - 1)
                    file_ml << ",";
            }
            file_ml << "," << t_wine_test[i] << "," << pred_test_ml_wine[i] << "\n";
        }
    }
    {
        std::ofstream file_bayes("./results/wine_red_bayesian_predictions.csv");
        file_bayes << "features,true,prediction,variance,lower95,upper95\n";
        for (size_t i = 0; i < Phi_wine_test.size(); i++) {
            for (size_t j = 0; j < X_wine_norm[train_size + i].size(); j++) {
                file_bayes << X_wine_norm[train_size + i][j] << ",";
            }
            double pred = pred_test_bayes_wine[i];
            double var = predVar_test_wine[i];
            double stddev = std::sqrt(var);
            double lower95 = pred - 1.96 * stddev;
            double upper95 = pred + 1.96 * stddev;
            file_bayes << t_wine_test[i] << "," << pred << "," << var
                       << "," << lower95 << "," << upper95 << "\n";
        }
    }
    {
        std::ofstream file_ml_w("./results/wine_red_ml_weights.csv");
        file_ml_w << "Index,Weight\n";
        for (size_t i = 0; i < w_ml_wine.size(); i++){
            file_ml_w << i << "," << w_ml_wine[i] << "\n";
        }
    }
    {
        std::ofstream file_bayes_w("./results/wine_red_bayesian_weights.csv");
        vector<double> bayes_wine = blr_wine.getWeights();
        file_bayes_w << "Index,Weight\n";
        for (size_t i = 0; i < bayes_wine.size(); i++){
            file_bayes_w << i << "," << bayes_wine[i] << "\n";
        }
    }
    {
        std::ofstream file_err("./results/wine_red_error_metrics.csv");
        file_err << "Method,MSE\n";
        file_err << "ML," << mse_ml_wine << "\n";
        file_err << "Bayesian," << mse_bayes_wine << "\n";
    }
    cout << "Wine quality experiment completed." << endl;
}

//---------------------------------------------------------------------
int main() {
    runSyntheticExperiment();
    runWineQualityExperiment();
    cout << "All experiments (synthetic and wine) completed. Check the results folder." << endl;
    return 0;
}