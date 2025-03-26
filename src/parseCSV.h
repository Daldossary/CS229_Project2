#ifndef PARSECSV_H
#define PARSECSV_H

#include <vector>
#include <string>

using std::vector;
using std::string;
using Matrix = vector<vector<double>>;

// changed so that it can skip headers automatically
Matrix parseCSV(const string& filename, bool skipHeader = true);

#endif