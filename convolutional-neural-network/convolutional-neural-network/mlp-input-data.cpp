#include "mlp-input-data.h"
#include <algorithm>

double Utils::Normalize(double value, double min, double max, double initial, double final)
{
	return (final - initial) * ((value - min) / (max - min)) + initial;
}



int Utils::IndexOfMaxValue(std::vector<double> arr)
{
	std::vector<double>::iterator it = std::max_element(arr.begin(), arr.end());
	int index = std::distance(arr.begin(), it);
	return index;
}



int Utils::IndexOfMinValue(std::vector<double> arr)
{
	std::vector<double>::iterator it = std::min_element(arr.begin(), arr.end());
	int index = std::distance(arr.begin(), it);
	return index;
}