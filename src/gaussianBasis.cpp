#include "gaussianBasis.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <execution>

// generate a single gaussian basis function
double gaussianBasisFunction(double x, double mu, double s) {
    return std::exp(- ((x - mu) * (x - mu)) / (2 * s * s)); // calc and return the gaussian basis function
}

// calcs the gaussian basis functions for a single input value
vector<double> calcGaussBasisFeatures(double x, const vector<double>& centers, double s) {
    vector<double> features; // create vector to store features
    features.reserve(centers.size()); // reserve space for the features
    for (double mu : centers) { // iterate over the centers
        features.push_back(gaussianBasisFunction(x, mu, s)); // calc feature and add to vector
    }
    return features; // return the vector of features
}

// calcs the gaussian basis functions for a matrix of input values
Matrix calcGaussBasisFeaturesM(const Matrix &X, const vector<double>& centers, double s) {
    size_t numData = X.size(); // get number of data points
    size_t numCenters = centers.size(); // get number of centers
    Matrix transformed(numData, vector<double>(numCenters, 0.0)); // create matrix to store transformed data
    
    // index vector [0, 1, 2, ..., numData-1] for parallel iteration.
    vector<size_t> indices(numData);
    std::iota(indices.begin(), indices.end(), 0); 
    
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) { // iterate over the data points
        if (!X[i].empty()) { // check if the data point is not empty
            double x = X[i][0];  // assumes univariate data (one feature per data point)
            for (size_t j = 0; j < numCenters; ++j) { // iterate over centers
                transformed[i][j] = gaussianBasisFunction(x, centers[j], s); // calc and store transformed data
            }
        }
    });
    
    return transformed; // return the transformed data
}

// generate a set of centers
vector<double> genCenters(const vector<double>& data, int numCenters) {
    if (data.empty() || numCenters <= 0) // check if data is empty or numCenters is less than or equal to 0
        return {}; // return empty vector
        
    double minVal = *std::min_element(data.begin(), data.end()); // get min value in the data
    double maxVal = *std::max_element(data.begin(), data.end()); // get max value in the data
    vector<double> centers(numCenters, 0.0); // create vector to store centers
    
    if (numCenters == 1) { // check if numCenters is 1
        centers[0] = (minVal + maxVal) / 2.0; // set center to average of min and max
    } else {
        double interval = (maxVal - minVal) / (numCenters - 1); // calc interval between centers
        for (int i = 0; i < numCenters; ++i) { // iterate over number of centers
            centers[i] = minVal + i * interval; // set center value
            // center val is the min val plus i times the interval, which is the distance between centers
        }
    }
    
    return centers; // return vector of centers
}