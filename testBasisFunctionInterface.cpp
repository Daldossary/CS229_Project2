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
    // Prepare a sample input matrix X (univariate data points: one feature per row).
    Matrix X = { {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}, {10.0} };
    
    // Display menu and get user selection.
    cout << "Select the basis function to use for the regression model:" << endl;
    cout << "1. Polynomial" << endl;
    cout << "2. Gaussian" << endl;
    cout << "3. Sigmoidal" << endl;
    cout << "Enter your choice (1/2/3): ";
    
    int choiceInput = 0;
    cin >> choiceInput;
    BasisFunctionType selectedFunction = static_cast<BasisFunctionType>(choiceInput);
    
    double param1 = 0.0, param2 = 0.0;
    
    // Ask for additional parameters based on the choice.
    switch (selectedFunction) {
        case BasisFunctionType::POLYNOMIAL:
            cout << "Enter polynomial degree (integer): ";
            cin >> param1;
            cout << "Polynomial basis function selected with degree " << param1 << endl;
            break;
        case BasisFunctionType::GAUSSIAN:
            cout << "Enter number of centers for Gaussian basis (integer): ";
            cin >> param1;
            cout << "Enter scale (s) for Gaussian basis: ";
            cin >> param2;
            cout << "Gaussian basis function selected with " << param1 << " centers and scale " << param2 << endl;
            break;
        case BasisFunctionType::SIGMOIDAL:
            cout << "Enter number of centers for Sigmoidal basis (integer): ";
            cin >> param1;
            cout << "Enter slope parameter for Sigmoidal basis: ";
            cin >> param2;
            cout << "Sigmoidal basis function selected with " << param1 << " centers and slope " << param2 << endl;
            break;
        default:
            cout << "Invalid selection. Exiting." << endl;
            return 1;
    }
    
    // Transform the input features using the selected basis function.
    Matrix transformed = transformFeatures(X, selectedFunction, param1, param2);
    
    // Output the transformed feature matrix to a CSV file for visualization.
    string outputFilename = "./results/basis_function_choice.csv";
    std::ofstream outFile(outputFilename);
    if (!outFile) {
        cout << "Error: Could not open file " << outputFilename << " for writing." << endl;
        return 1;
    }
    
    // Write header: first column "x", then one column per basis function (phi0, phi1, ...)
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