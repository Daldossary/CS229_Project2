#include "parseCSV.h"
#include <iostream>
#include <fstream>
#include <sstream>

Matrix parseCSV(const string& filename, bool skipHeader) {
    std::ifstream file(filename);
    if (!file) {
        printf("File not found.\n");
        return {}; // no file = no data
    }
    
    Matrix data;
    std::string currentline;

    // skipping the header line if true
    if (skipHeader) {
        std::getline(file, currentline);
    }
    
    while (std::getline(file, currentline)) {
        if (!currentline.empty()) {
            vector<double> row;
            std::istringstream inputStream(currentline);
            string token;
            while (std::getline(inputStream, token, ',')) {
                try {
                    double value = std::stod(token);
                    row.push_back(value);
                } catch (std::exception& e) {
                    // printf("Error parsing file.\n");
                    row.push_back(0.0);
                }
            }
            data.push_back(row);
        }
    }
    return data;
}