#include "matrixOperations.h"
#include <vector>
#include <string>
#include <cmath> 

using std::string;
using std::vector;

// create an alias for a 2D vector of doubles called matrix.
using Matrix = vector<vector<double>>;


// matrix multiplication.
vector<vector<double>> multMs(vector<vector<double>>& m1, vector<vector<double>>& m2) {

    size_t m1Rows = m1.size();               
    size_t m1Cols = m1[0].size();     
    size_t m2Rows = m2.size();          
    size_t m2Cols = m2[0].size();

    if (m1Cols != m2Rows) {
        printf("Matrix dimensions are incompatible for multiplication.");
    }

    vector<vector<double>> result(m1Rows, vector<double>(m2Cols, 0.0));

    //looping over rows of m1
    for (size_t i = 0; i < m1Rows; ++i) { 
        //looping over columns of m2             
        for (size_t j = 0; j < m2Cols; ++j) {
            //looping over columns of m1 or rows of m2
            for (size_t k = 0; k < m1Cols; ++k) {
                //performing the multiplication
                result[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    return result;  
}

// matrix transpose.
vector<vector<double>> transposeM(vector<vector<double>>& m) {
    
    size_t rows = m.size();            
    size_t cols = m[0].size();      

    // swap dimensions for resulting transpose m.
    vector<vector<double>> mTransposed(cols, vector<double>(rows, 0.0));

    //loop over rows of m
    for (size_t i = 0; i < rows; ++i) {      
        //loop over columns of m
        for (size_t j = 0; j < cols; ++j) {
            //perform transpose by flipping the indices.
            mTransposed[j][i] = m[i][j];
        }
    }
    return mTransposed;
}

// calc inverse m through gauss. elimin.
vector<vector<double>> invertM(vector<vector<double>>& m) {
    size_t n = m.size();

    // checking for sq m
    for (auto& row : m) {
        if (row.size() != n) {
            printf("Matrix is not square.");
        }
    }

    // augmented matrix -> [matrix | identity]
    vector<vector<double>> aug(n, vector<double>(2 * n, 0.0));
    for (size_t i = 0; i < n; ++i) { // i = rows
        for (size_t j = 0; j < n; ++j) { // j = columns
            aug[i][j] = m[i][j]; // copying m into aug
        }
        aug[i][n + i] = 1.0; // add identity m
        // i = rows, n+i = m columns then i
    }

    // gauss. elimin.
    for (size_t i = 0; i < n; ++i) { // go through rows

        // taking diagElem.
        double diagElement = aug[i][i];

        // check for singular m.
        if (diagElement == 0.0) {
            printf("Matrix is singular and cannot be inverted.");
        }

        // make diagElem = 1 by diving by row (aka diag norm.)
        for (size_t j = 0; j < 2 * n; ++j) { // go through columns
            aug[i][j] /= diagElement;
        }

        // make other vals in column = 0 (aka row reduc.)
        for (size_t k = 0; k < n; ++k) { //iterate over all rows that are not i
            if (k != i) {
                double factor = aug[k][i]; 
                for (size_t j = 0; j < 2 * n; ++j) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // getting inverse from aug m.
    vector<vector<double>> inverse(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            inverse[i][j] = aug[i][n + j];
        }
    }

    return inverse;
}
