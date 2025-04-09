#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <vector>
using std::vector;
using Matrix = std::vector<std::vector<double>>;

class LogisticRegression {
public:
    LogisticRegression(double learning_rate = 0.01, int max_iter = 10000, double tol = 1e-6);

    void fit(const Matrix &X, const vector<int> &y);

    vector<double> predictProb(const Matrix &X) const;

    vector<int> predict(const Matrix &X) const;

    vector<double> getWeights() const;
    
    vector<double> getCostHistory() const;

private:
    vector<double> weights_;
    double learning_rate_;
    int max_iter_;
    double tol_;
    
    vector<double> cost_history_;

    double sigmoid(double z) const;

    double computeCost(const Matrix &X, const vector<int> &y) const;
};

#endif
