#pragma once

#include <vector>


struct DATA {
	std::vector<double> inputs;
	int labelIndex;

	DATA() {}
	DATA(std::vector<double> i, int l) : inputs(i), labelIndex(l) { }
};



namespace Utils {
	
	double Normalize(double value, double min, double max, double initial = -1.0, double final = 1.0);
	int IndexOfMaxValue(std::vector<double> arr);
	int IndexOfMinValue(std::vector<double> arr);
}


