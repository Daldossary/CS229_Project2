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
    
    // --- single value transformation test --- //
    double x = 5.0; // single data point
    vector<double> fixedCenters = {2.0, 4.0, 6.0, 8.0}; // fixed centers
    double s = 1.0; // common slope for all basis functions
    
    vector<double> features = calcGaussBasisFeatures(x, fixedCenters, s); // calculate features
    cout << "Gaussian features for x = " << x << ":" << endl; // print features
    for (size_t i = 0; i < features.size(); ++i) {
        cout << "Center " << fixedCenters[i] << ": " << features[i] << endl; 
    }
    
    // --- automatic center generation test --- //
    vector<double> sampleData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; // sample data
    int numCenters = 5; // number of centers
    vector<double> autoCenters = genCenters(sampleData, numCenters); // generate centers
    cout << "Automatically generated centers:" << endl; // print centers
    for (double center : autoCenters) {
        cout << center << " ";
    }
    cout << endl;
    
    // --- matrix transformation test --- //
    Matrix X = { {1.0}, {3.0}, {5.0}, {7.0}, {9.0} }; // sample matrix X (each row is univariate data point).
    Matrix transformed = calcGaussBasisFeaturesM(X, autoCenters, s); // calculate features

    
    // --- writing transformed features to CSV file for visualization --- //
    string csvFilename = "./results/gaussian_basis_features.csv";
    std::ofstream outFile(csvFilename);
    if (!outFile) {
        cout << "Error: Could not open file " << csvFilename << " for writing." << endl;
        return 1;
    }
    
    // writing a header.
    outFile << "x";
    for (size_t j = 0; j < autoCenters.size(); ++j) {
        outFile << ",phi" << j;
    }
    outFile << "\n";
    
    // writing each data value and its gaussian features.
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