#ifndef PARSECSV_H
#define PARSECSV_H

#include <vector>
#include <string>

using std::vector;
using std::string;
using Matrix = vector<vector<double>>;

// By default, skipHeader is set to true
Matrix parseCSV(const string& filename, bool skipHeader = true);

#endif