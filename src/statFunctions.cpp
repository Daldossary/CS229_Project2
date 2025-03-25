#include "statFunctions.h"
#include <vector>
#include <cmath> 

using std::vector;

// create an alias for a 2D vector of doubles called matrix.
using Matrix = vector<vector<double>>;


// calc mean of data vector.
double calcMean(vector<double>& data) {
    double sum = 0.0;                       
    for (double val : data) {              
        sum += val;                     
    }
    return data.empty() ? 0.0 : sum / data.size();
}

// calc variance of data vector.
double calcVar(vector<double>& data) {
    double mean = calcMean(data);     
    double sumSqDiff = 0.0;  
    for (double val : data) {    
        double diff = val - mean;      
        sumSqDiff += diff * diff;            
    }
    return data.empty() ? 0.0 : sumSqDiff / data.size();
}

// calc std dev of data vector.
double calcStdDev(vector<double>& data) {
    double var = calcVar(data);    
    return std::sqrt(var);  
}
