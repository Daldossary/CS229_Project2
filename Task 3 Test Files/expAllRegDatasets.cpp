/*
 * File: expAllBasisFunctions.cpp
 * Description:
 *   This file runs Bayesian regression with hyperparameter tuning on three datasets
 *   (Synthetic, Wine Quality [Red], Forest Fires) using three basis functions:
 *
 *     - Polynomial Basis Functions
 *     - Sigmoidal Basis Functions
 *     - Gaussian Basis Functions
 *
 *   For each dataset the code:
 *     1. Loads (or generates) and normalizes the data.
 *     2. For each basis function, computes the design (feature‐expanded) matrix:
 *          • For Polynomial: If the data is univariate use genPolyFeatures() 
 *            otherwise use genMultiVarPolyFeatures().
 *          • For Sigmoidal and Gaussian: (Working on the first feature only)
 *            – Generate centers (via genSigmoidCenters() or genCenters())
 *            – Compute the corresponding basis features (using calcSigmoidalBasisFeaturesM()
 *              or calcGaussBasisFeaturesM())
 *            – Prepend a bias term to each row.
 *     3. Splits the design matrix into training (80%) and test (20%) portions.
 *     4. Performs grid search tuning of hyperparameters (α and β) on the training split.
 *     5. Trains Bayesian linear regression and computes predictions (and predictive variances)
 *        on the test set. It then calculates 95% confidence intervals.
 *     6. Writes all outputs (tuning results, predictions, posterior (weight) parameters,
 *        error metrics) to CSV files in the ./results/ folder.
 *     7. For each dataset the best (lowest training MSE) basis function is determined and
 *        a summary CSV file is saved.
 *
 *   Compilation (from the project root):
 *     g++ expAllBasisFunctions.cpp ./src/matrixOperations.cpp ./src/polyReg.cpp ./src/bayesianLinearReg.cpp ./src/statFunctions.cpp ./src/sigmoidalBasis.cpp ./src/gaussianBasis.cpp -o expAllBasisFunctions -std=c++17
 *
 *   Usage:
 *     ./expAllBasisFunctions
 *
 * Requirements addressed:
 *   - Language: vanilla C++17 using only standard libraries.
 *   - Re-use: CSV parsing, matrix and vector operations functions are re-used.
 *   - Parallelism: Normalization uses std::for_each with std::execution::par.
 *   - Documentation: The code is well-commented.
 *   - Visualization: All results are written to CSV files for later plotting.
 */

 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <vector>
 #include <string>
 #include <cmath>
 #include <random>
 #include <algorithm>
 #include <numeric>
 #include <execution>  // C++17 parallel algorithms
 
 // Re-used source files (your own implementations)
 #include "./src/matrixOperations.h"    // Provides transposeM(), multMs(), invertM(), and the Matrix alias.
 #include "./src/polyReg.h"               // Provides genPolyFeatures() and genMultiVarPolyFeatures()
 #include "./src/sigmoidalBasis.h"        // Provides calcSigmoidalBasisFeaturesM(), genSigmoidCenters()
 #include "./src/gaussianBasis.h"         // Provides calcGaussBasisFeaturesM(), genCenters()
 #include "./src/bayesianLinearReg.h"     // Provides bayesianLinearRegression class.
 #include "./src/statFunctions.h"         // Provides calcMean(), calcStdDev(), etc.
 
 using std::cout;
 using std::cerr;
 using std::endl;
 using std::string;
 using std::vector;
 using Matrix = std::vector<std::vector<double>>;
 
 /*---------------------- Helper Functions ---------------------------*/
 
 // Parse a CSV file using the given delimiter.
 // This version first reads the header line (to record the expected number of tokens)
 // and then, for each subsequent line, only keeps as many tokens as in the header.
 Matrix parseCSVWithDelimiter(const string &filename, char delimiter, bool skipHeader = true) {
     std::ifstream file(filename);
     Matrix data;
     if (!file) {
         cerr << "Error: File " << filename << " not found." << endl;
         return data;
     }
     string line;
     size_t expectedTokens = 0;
     if (skipHeader && std::getline(file, line)) {
         std::istringstream headerStream(line);
         string token;
         while (std::getline(headerStream, token, delimiter)) {
             expectedTokens++;
         }
     }
     while (std::getline(file, line)) {
         if (line.empty()) continue;
         std::istringstream ss(line);
         string token;
         vector<double> row;
         size_t count = 0;
         while (std::getline(ss, token, delimiter)) {
             // Only take expectedTokens tokens (ignore extra tokens if any)
             if (expectedTokens > 0 && count >= expectedTokens)
                 break;
             try {
                 double value = std::stod(token);
                 row.push_back(value);
             } catch (...) {
                 row.push_back(0.0);
             }
             count++;
         }
         // Only add rows that have at least one token
         if (!row.empty())
             data.push_back(row);
     }
     return data;
 }
   
 // Normalize the matrix X column‑wise using z‑score standardization.
 // Uses parallel execution with std::for_each and std::execution::par.
 Matrix normalizeMatrix(const Matrix &X) {
     size_t m = X.size();
     if (m == 0) return X;
     size_t n = X[0].size();
     Matrix X_norm = X; // create a copy of X
 
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
   
 // Extract the first column from X and return it as a new matrix.
 Matrix extractFirstColumnMatrix(const Matrix &X) {
     Matrix X_first;
     for (const auto &row : X) {
         if (!row.empty())
             X_first.push_back({row[0]});
     }
     return X_first;
 }
   
 // Extract the first column of X into a vector.
 vector<double> extractFirstColumnVector(const Matrix &X) {
     vector<double> v;
     for (const auto &row : X) {
         if (!row.empty())
             v.push_back(row[0]);
     }
     return v;
 }
   
 // Compute Mean Squared Error between y_true and y_pred.
 double computeMSE(const vector<double>& y_true, const vector<double>& y_pred) {
     double sum = 0.0;
     for (size_t i = 0; i < y_true.size(); i++) {
         double diff = y_true[i] - y_pred[i];
         sum += diff * diff;
     }
     return y_true.empty() ? 0.0 : sum / y_true.size();
 }
   
 // Structure to hold hyperparameter tuning results.
 struct TuningResult {
     double best_alpha;
     double best_beta;
     double best_val_mse;
 };
   
 // Perform grid search for hyperparameter tuning, splitting the training data
 // into 80% tuning-training and 20% validation.
 TuningResult tuneBayesianHyperparameters(const Matrix &Phi_train, const vector<double> &t_train) {
     size_t N = Phi_train.size();
     size_t n_train = static_cast<size_t>(N * 0.8);
     Matrix Phi_tuneTrain(Phi_train.begin(), Phi_train.begin() + n_train);
     vector<double> t_tuneTrain(t_train.begin(), t_train.begin() + n_train);
     Matrix Phi_val(Phi_train.begin() + n_train, Phi_train.end());
     vector<double> t_val(t_train.begin() + n_train, t_train.end());
 
     vector<double> candidate_alphas = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10};
     vector<double> candidate_betas  = {1e1, 1e2, 1e3, 1e4};
 
     double best_mse = 1e12;
     double best_alpha = candidate_alphas[0];
     double best_beta = candidate_betas[0];
 
     for (double alpha : candidate_alphas) {
         for (double beta : candidate_betas) {
             bayesianLinearRegression model(alpha, beta);
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
     return TuningResult{best_alpha, best_beta, best_mse};
 }
   
 // Structure to store the experimental result for one run.
 struct ExperimentResult {
     string dataset;
     string basisType;
     double trainingMSE;
     double bestAlpha;
     double bestBeta;
 };
   
 /*---------------------- Experiment Functions ---------------------------*/
 
 // Run experiments for three basis functions on a given dataset.
 // Parameters:
 //   datasetName: identifier used in output file names ("synthetic", "wine_red", "forestfires").
 //   X: normalized feature matrix.
 //   t: target vector.
 //   isUnivariate: true if original data is univariate (used for Polynomial basis).
 //   polyDegree: degree for polynomial expansion.
 //   numCenters: number of centers for sigmoidal and Gaussian bases.
 //   s: scale parameter for sigmoidal and Gaussian bases.
 vector<ExperimentResult> runBasisExperiments(const string &datasetName, const Matrix &X, const vector<double> &t,
                                                bool isUnivariate, int polyDegree, int numCenters, double s) {
     vector<ExperimentResult> results;
     vector<string> basisTypes = {"Polynomial", "Sigmoidal", "Gaussian"};
   
     for (const auto &basis : basisTypes) {
         Matrix Phi;  // design (feature-expanded) matrix
       
         if (basis == "Polynomial") {
             if (isUnivariate)
                 Phi = genPolyFeatures(const_cast<Matrix&>(X), polyDegree);
             else
                 Phi = genMultiVarPolyFeatures(const_cast<Matrix&>(X), polyDegree);
         }
         else if (basis == "Sigmoidal") {
             // Use only the first feature.
             Matrix X_first = extractFirstColumnMatrix(X);
             vector<double> dataVector = extractFirstColumnVector(X);
             vector<double> centers = genSigmoidCenters(dataVector, numCenters);
             Phi = calcSigmoidalBasisFeaturesM(X_first, centers, s);
             // Prepend bias term.
             for (auto &row : Phi) {
                 row.insert(row.begin(), 1.0);
             }
         }
         else if (basis == "Gaussian") {
             Matrix X_first = extractFirstColumnMatrix(X);
             vector<double> dataVector = extractFirstColumnVector(X);
             vector<double> centers = genCenters(dataVector, numCenters);
             Phi = calcGaussBasisFeaturesM(X_first, centers, s);
             // Prepend bias term.
             for (auto &row : Phi) {
                 row.insert(row.begin(), 1.0);
             }
         }
       
         // Split Phi (and targets) into training (80%) and test (20%).
         size_t total = Phi.size();
         size_t train_size = static_cast<size_t>(total * 0.8);
         Matrix Phi_train(Phi.begin(), Phi.begin() + train_size);
         Matrix Phi_test(Phi.begin() + train_size, Phi.end());
         vector<double> t_train(t.begin(), t.begin() + train_size);
         vector<double> t_test(t.begin() + train_size, t.end());
       
         // Hyperparameter tuning.
         TuningResult tuneRes = tuneBayesianHyperparameters(Phi_train, t_train);
         {
             std::ofstream file("./results/" + datasetName + "_" + basis + "_tuning_results.csv");
             file << "BestAlpha,BestBeta,ValidationMSE\n";
             file << tuneRes.best_alpha << "," << tuneRes.best_beta << "," << tuneRes.best_val_mse << "\n";
         }
       
         // Train Bayesian regression on training set.
         bayesianLinearRegression model(tuneRes.best_alpha, tuneRes.best_beta);
         model.fit(Phi_train, t_train);
       
         // Get predictions and predictive variances on test set.
         vector<double> pred_test = model.predict(Phi_test);
         vector<double> predVar_test = model.predictiveVar(Phi_test);
       
         {
             std::ofstream file("./results/" + datasetName + "_" + basis + "_bayesian_predictions_tuned.csv");
             file << "features,true,prediction,variance,lower95,upper95\n";
             for (size_t i = 0; i < Phi_test.size(); i++){
                 if (basis == "Polynomial") {
                     // For polynomial basis, output original (normalized) features.
                     for (size_t j = 0; j < X[i + train_size].size(); j++){
                         file << X[i + train_size][j];
                         if (j < X[i + train_size].size() - 1)
                             file << " ";
                     }
                 } else {
                     // For Sigmoidal and Gaussian, output the first feature.
                     file << X[i + train_size][0];
                 }
                 double pred = pred_test[i];
                 double var = predVar_test[i];
                 double stddev = std::sqrt(var);
                 double lower95 = pred - 1.96 * stddev;
                 double upper95 = pred + 1.96 * stddev;
                 file << "," << t_test[i] << "," << pred << "," << var << "," << lower95 << "," << upper95 << "\n";
             }
         }
       
         {
             vector<double> weights = model.getWeights();
             std::ofstream file("./results/" + datasetName + "_" + basis + "_bayesian_weights_tuned.csv");
             file << "Index,Weight\n";
             for (size_t i = 0; i < weights.size(); i++){
                 file << i << "," << weights[i] << "\n";
             }
         }
       
         vector<double> pred_train = model.predict(Phi_train);
         double mse_train = computeMSE(t_train, pred_train);
         {
             std::ofstream file("./results/" + datasetName + "_" + basis + "_error_metrics_tuned.csv");
             file << "Method,TrainingMSE\n";
             file << "Bayesian_Tuned," << mse_train << "\n";
         }
       
         cout << datasetName << " experiment (" << basis << " basis) completed. Best Alpha = " 
              << tuneRes.best_alpha << ", Best Beta = " << tuneRes.best_beta
              << ", Training MSE = " << mse_train << endl;
       
         ExperimentResult er{datasetName, basis, mse_train, tuneRes.best_alpha, tuneRes.best_beta};
         results.push_back(er);
     }
     return results;
 }
   
 /*---------------------- Dataset-Specific Experiment Functions ---------------------------*/
 
// Synthetic dataset: Generate univariate data: t = 3*x + 2 + noise.
void runSyntheticFullExperiment(vector<ExperimentResult> &overallResults) {
    cout << "=== Running Synthetic Regression Experiments with All Basis Functions ===" << endl;
    const int N_train = 50;
    const int N_test = 20;
    vector<vector<double>> X_train, X_test;
    vector<double> t_train, t_test;  // Create separate target vectors for training and test
    std::default_random_engine generator(42);
    std::uniform_real_distribution<double> uniformDist(0.0, 10.0);
    std::normal_distribution<double> noise(0.0, 1.0);

    // Generate training data.
    for (int i = 0; i < N_train; i++) {
         double x = uniformDist(generator);
         X_train.push_back({x});
         double t_val = 3.0 * x + 2.0 + noise(generator);
         t_train.push_back(t_val);
    }

    // Generate test data.
    for (int i = 0; i < N_test; i++) {
         double x = 10.0 * i / (N_test - 1);
         X_test.push_back({x});
         double true_t = 3.0 * x + 2.0; // Use the noise-free function for test targets
         t_test.push_back(true_t);
    }

    // Combine training and test feature matrices.
    Matrix X_full = X_train;
    X_full.insert(X_full.end(), X_test.begin(), X_test.end());
    // Combine target vectors so that t_full matches the number of rows in X_full.
    vector<double> t_full = t_train;
    t_full.insert(t_full.end(), t_test.begin(), t_test.end());

    // Now both X_full and t_full have (N_train + N_test = 70) rows.
    vector<ExperimentResult> res = runBasisExperiments("synthetic", X_full, t_full, true, 2, 5, 1.0);

    // Choose the best basis according to lowest training MSE.
    ExperimentResult best = res[0];
    for (const auto &r : res) {
         if (r.trainingMSE < best.trainingMSE)
             best = r;
    }
    {
         std::ofstream file("./results/synthetic_best_summary.csv");
         file << "BasisType,TrainingMSE,Alpha,Beta\n";
         file << best.basisType << "," << best.trainingMSE << "," << best.bestAlpha << "," << best.bestBeta << "\n";
    }
    overallResults.insert(overallResults.end(), res.begin(), res.end());
    cout << "Synthetic best basis: " << best.basisType << " with Training MSE = " << best.trainingMSE << endl;
}
   
 // Wine Quality (Red) dataset.
 void runWineQualityExperiment(vector<ExperimentResult> &overallResults) {
     cout << "=== Running Wine Quality (Red) Regression Experiments with All Basis Functions ===" << endl;
     string filename = "./datasets/WineQuality_Red_Regression/winequality-red_Regression.csv";
     Matrix wineData = parseCSVWithDelimiter(filename, ';', true);
     if (wineData.empty()) {
         cerr << "Error loading wine dataset." << endl;
         return;
     }
     Matrix X;
     vector<double> t;
     for (size_t i = 0; i < wineData.size(); i++) {
         if (wineData[i].empty())
             continue;
         vector<double> features(wineData[i].begin(), wineData[i].end() - 1);
         X.push_back(features);
         t.push_back(wineData[i].back());
     }
     Matrix X_norm = normalizeMatrix(X);
     vector<ExperimentResult> res = runBasisExperiments("wine_red", X_norm, t, false, 2, 5, 1.0);
   
     ExperimentResult best = res[0];
     for (const auto &r : res) {
         if (r.trainingMSE < best.trainingMSE)
             best = r;
     }
     {
         std::ofstream file("./results/wine_red_best_summary.csv");
         file << "BasisType,TrainingMSE,Alpha,Beta\n";
         file << best.basisType << "," << best.trainingMSE << "," << best.bestAlpha << "," << best.bestBeta << "\n";
     }
     overallResults.insert(overallResults.end(), res.begin(), res.end());
     cout << "Wine Quality (Red) best basis: " << best.basisType << " with Training MSE = " << best.trainingMSE << endl;
 }
   
 // Forest Fires dataset.
 void runForestFiresExperiment(vector<ExperimentResult> &overallResults) {
     cout << "=== Running Forest Fires Regression Experiments with All Basis Functions ===" << endl;
     string filename = "./datasets/ForestFires_Regression/forestfires.csv";
     Matrix data = parseCSVWithDelimiter(filename, ',', true);
     if (data.empty()) {
         cerr << "Error loading forest fires dataset." << endl;
         return;
     }
     Matrix X;
     vector<double> t;
     for (size_t i = 0; i < data.size(); i++) {
         if (data[i].empty())
             continue;
         vector<double> features(data[i].begin(), data[i].end() - 1);
         X.push_back(features);
         t.push_back(data[i].back());
     }
     Matrix X_norm = normalizeMatrix(X);
     // Use a higher polynomial degree (4) to better capture nonlinearity.
     vector<ExperimentResult> res = runBasisExperiments("forestfires", X_norm, t, false, 4, 5, 1.0);
   
     ExperimentResult best = res[0];
     for (const auto &r : res) {
         if (r.trainingMSE < best.trainingMSE)
             best = r;
     }
     {
         std::ofstream file("./results/forestfires_best_summary.csv");
         file << "BasisType,TrainingMSE,Alpha,Beta\n";
         file << best.basisType << "," << best.trainingMSE << "," << best.bestAlpha << "," << best.bestBeta << "\n";
     }
     overallResults.insert(overallResults.end(), res.begin(), res.end());
     cout << "Forest Fires best basis: " << best.basisType << " with Training MSE = " << best.trainingMSE << endl;
 }
   
 /*---------------------- Main ---------------------------*/
 int main() {
    try {
        vector<ExperimentResult> overallResults;
        runSyntheticFullExperiment(overallResults);
        runWineQualityExperiment(overallResults);
        runForestFiresExperiment(overallResults);
       
        // Write an overall summary file with results from all experiments.
        {
            std::ofstream file("./results/overall_best_summary.csv");
            file << "Dataset,BasisType,TrainingMSE,Alpha,Beta\n";
            for (const auto &res : overallResults) {
                file << res.dataset << "," << res.basisType << "," << res.trainingMSE << ","
                     << res.bestAlpha << "," << res.bestBeta << "\n";
            }
        }
       
        cout << "All experiments completed. Check the './results/' folder for outputs." << endl;
    } catch (const std::exception &e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}