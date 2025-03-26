#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <execution>
#include <algorithm>
#include <numeric>
#include "./src/parseCSV.h"
#include "./src/statFunctions.h"

using Matrix = std::vector<std::vector<double>>;
using std::string;
using std::vector;
using std::cout;
using std::endl;

void writeSummaryStatistics(const Matrix &data, const string &outputFilename) {
    if(data.empty()) { // Check if data is empty
        cout << "No data loaded. Nothing to write to " << outputFilename << endl;
        return;
    }

    size_t numCols = data[0].size(); // Number of columns in dataset
    vector<double> means(numCols, 0.0); // vector for storing means
    vector<double> variances(numCols, 0.0); // vector for storing variances
    vector<double> stdDevs(numCols, 0.0); // vector for storing standard deviations

    // vector of indices for columns
    vector<size_t> indices(numCols);
    std::iota(indices.begin(), indices.end(), 0);

    // using parallel processing for_each over indices
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t col) {
        vector<double> columnData; // vector for storing data in column
        for (const auto &row : data) { // iterate over rows
            if (col < row.size()) // check if column index is within row size
                columnData.push_back(row[col]); // add data to columnData
        }
        means[col] = calcMean(columnData); // calculate mean
        variances[col] = calcVar(columnData); // calculate variance
        stdDevs[col] = calcStdDev(columnData); // calculate standard deviation
    });

    // writing summary statistics to csv file
    std::ofstream outFile("./results/" + outputFilename);
    if (!outFile) {
        cout << "Error: Could not open " << outputFilename << " for writing." << endl;
        return;
    }
    outFile << "Feature,Mean,Variance,StdDev\n";
    for (size_t i = 0; i < numCols; ++i) {
        outFile << i << "," << means[i] << "," << variances[i] << "," << stdDevs[i] << "\n";
    }
    cout << "Summary statistics written to ./results/" << outputFilename << endl;
}

void processDataset(const string &datasetPath, const string &outputFileName) {
    Matrix data = parseCSV(datasetPath);
    if(data.empty()){
        cout << "No data loaded from " << datasetPath << endl;
        return;
    }
    writeSummaryStatistics(data, outputFileName);
}

int main() {
    // processing each dataset
    processDataset("./datasets/Iris_Classification/Iris_Classification.csv", "iris_summary.csv");
    processDataset("./datasets/Titanic_Classification/train_Titanic.csv", "titanic_summary.csv");
    processDataset("./datasets/WineQuality_Red_Regression/winequality-red_Regression.csv", "wine_red_summary.csv");
    processDataset("./datasets/WineQuality_White_Regression/winequality-white_Regression.csv", "wine_white_summary.csv");

    cout << "All datasets processed. Check the results folder." << endl;
    return 0;
}