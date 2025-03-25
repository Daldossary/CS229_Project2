#include <iostream>
#include <fstream>
#include "./src/gaussianBasis.h"
#include <vector>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main() {
    cout << "Testing Gaussian Basis Functions" << endl;
    
    // ------------------ Test 1: Single Value Transformation ------------------
    double x = 5.0;
    vector<double> fixedCenters = {2.0, 4.0, 6.0, 8.0};
    double s = 1.0;
    
    vector<double> features = computeGaussianBasisFeatures(x, fixedCenters, s);
    cout << "Gaussian features for x = " << x << ":" << endl;
    for (size_t i = 0; i < features.size(); ++i) {
        cout << "Center " << fixedCenters[i] << ": " << features[i] << endl;
    }
    
    // ------------------ Test 2: Automatic Center Generation ------------------
    vector<double> sampleData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int numCenters = 5;
    vector<double> autoCenters = generateCenters(sampleData, numCenters);
    cout << "Automatically generated centers:" << endl;
    for (double center : autoCenters) {
        cout << center << " ";
    }
    cout << endl;
    
    // ------------------ Test 3: Gaussian Basis Matrix Transformation --------
    // Create a sample matrix X (each row is a univariate data point).
    Matrix X = { {1.0}, {3.0}, {5.0}, {7.0}, {9.0} };
    Matrix transformed = computeGaussianBasisFeaturesMatrix(X, autoCenters, s);
    
    // Write the transformed features to a CSV file for visualization.
    string csvFilename = "./results/gaussian_basis_features.csv";
    std::ofstream outFile(csvFilename);
    if (!outFile) {
        cout << "Error: Could not open file " << csvFilename << " for writing." << endl;
        return 1;
    }
    
    // Write header.
    outFile << "x";
    for (size_t j = 0; j < autoCenters.size(); ++j) {
        outFile << ",phi" << j;
    }
    outFile << "\n";
    
    // Write each data value and its corresponding Gaussian features.
    for (size_t i = 0; i < X.size(); ++i) {
        outFile << X[i][0];
        for (size_t j = 0; j < transformed[i].size(); ++j) {
            outFile << "," << transformed[i][j];
        }
        outFile << "\n";
    }
    
    outFile.close();
    cout << "Gaussian basis features written to " << csvFilename << endl;
    
    return 0;
}