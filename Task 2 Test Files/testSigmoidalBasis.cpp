#include <iostream>
#include <fstream>
#include "./src/sigmoidalBasis.h"
#include <vector>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main() {
    cout << "Testing Sigmoidal Basis Functions" << endl;
    
    // --- single value transformation test --- //
    double x = 5.0; // single data point
    vector<double> fixedCenters = {2.0, 4.0, 6.0, 8.0}; // fixed centers
    double s = 1.0; // common slope for all basis functions
    vector<double> features = calcSigmoidalBasisFeatures(x, fixedCenters, s); // calculate features
    
    cout << "Sigmoidal features for x = " << x << ":" << endl; // print features
    for (size_t i = 0; i < features.size(); ++i) { // loop through features
        cout << "Center " << fixedCenters[i] << ": " << features[i] << endl; // print feature
    }
    
    // --- automatic center generation test --- //
    vector<double> sampleData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; // sample data
    int numCenters = 5; // number of centers
    vector<double> autoCenters = genSigmoidCenters(sampleData, numCenters); // generate centers
    
    cout << "Automatically generated centers:" << endl; // print centers
    for (double center : autoCenters) { // loop through centers
        cout << center << " "; // print center
    }
    cout << endl; // print newline
    
    // --- matrix transformation test --- //
    Matrix X = { {1.0}, {3.0}, {5.0}, {7.0}, {9.0} }; // sample matrix X (each row is univariate data point)
    Matrix transformed = calcSigmoidalBasisFeaturesM(X, autoCenters, s); // calculate features


    // --- writing transformed features to CSV file for visualization --- //
    string csvFilename = "./results/sigmoidal_basis_features.csv";
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
    
    // writing each data value and its sigmoidal features.
    for (size_t i = 0; i < X.size(); ++i) {
        outFile << X[i][0];
        for (size_t j = 0; j < transformed[i].size(); ++j) {
            outFile << "," << transformed[i][j];
        }
        outFile << "\n";
    }
    
    outFile.close();
    cout << "Sigmoidal basis features written to " << csvFilename << endl;
    
    return 0;
}