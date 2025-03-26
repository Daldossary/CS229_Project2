#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "./src/basisFunctionInterface.h"

using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::vector;

int main() {

    //sample input matrix X (univariate data points: one feature per row)
    Matrix X = { {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}, {10.0} };
    
    // display menu and get user choice
    cout << "Select the basis function for the regression model:" << endl;
    cout << "1. Polynomial Basis Function" << endl;
    cout << "2. Gaussian Basis Function" << endl;
    cout << "3. Sigmoidal Basis Function" << endl;
    cout << "Enter your choice (1/2/3): ";
    
    int choiceInput = 0; // user choice
    cin >> choiceInput; // read user choice
    basisFunctionType selectedBasis = static_cast<basisFunctionType>(choiceInput); // convert to enum
    
    double p1 = 0.0, p2 = 0.0;
    
    // ask for parameters based on choice.
    switch (selectedBasis) {
        case basisFunctionType::POLYNOMIAL:
            cout << "Enter polynomial degree (integer): ";
            cin >> p1;
            cout << "Polynomial basis function selected with degree " << p1 << endl;
            break;
        case basisFunctionType::GAUSSIAN:
            cout << "Enter number of centers for Gaussian basis (integer): ";
            cin >> p1;
            cout << "Enter scale (s) for Gaussian basis: ";
            cin >> p2;
            cout << "Gaussian basis function selected with " << p1 << " centers and scale " << p2 << endl;
            break;
        case basisFunctionType::SIGMOIDAL:
            cout << "Enter number of centers for Sigmoidal basis (integer): ";
            cin >> p1;
            cout << "Enter slope parameter for Sigmoidal basis: ";
            cin >> p2;
            cout << "Sigmoidal basis function selected with " << p1 << " centers and slope " << p2 << endl;
            break;
        default:
            cout << "Invalid selection. Exiting." << endl;
            return 1;
    }
    
    // apply transform on input features using selected basis function
    Matrix transformed = transformFeatures(X, selectedBasis, p1, p2);
    
    // output transformed feature matrix to CSV file for visualization
    string outputFilename = "./results/basis_function_choice.csv";
    std::ofstream outFile(outputFilename);
    if (!outFile) {
        cout << "Error: Could not open file " << outputFilename << " for writing." << endl;
        return 1;
    }
    
    // writing header: 
    // first column "x"
    // one column per basis function (phi0, phi1, etc...)
    outFile << "x";
    if (!transformed.empty()) {
        for (size_t j = 0; j < transformed[0].size(); ++j)
            outFile << ",phi" << j;
        outFile << "\n";
    
        // Write one row per data point.
        for (size_t i = 0; i < transformed.size(); ++i) {
            outFile << X[i][0];
            for (size_t j = 0; j < transformed[i].size(); ++j)
                outFile << "," << transformed[i][j];
            outFile << "\n";
        }
    }
    
    outFile.close();
    cout << "Transformed features written to " << outputFilename << endl;
    
    return 0;
}