#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>


struct DATA {
	std::vector<double> inputs;
	int labelIndex;

	DATA() {}
	DATA(std::vector<double> i, int l) : inputs(i), labelIndex(l) { }
};



struct CnnData {
	Eigen::MatrixXd inputs;
	int labelIndex;

	CnnData() { }
	CnnData(Eigen::MatrixXd i, int l) : inputs(i), labelIndex(l) { }
};



namespace Utils {
	
	double Normalize(double value, double min, double max, double initial = -1.0, double final = 1.0);
	int IndexOfMaxValue(std::vector<double> arr);
	int IndexOfMinValue(std::vector<double> arr);
}





