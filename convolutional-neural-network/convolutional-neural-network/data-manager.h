#pragma once

#include <iostream>
#include "mlp-input-data.h"

class DataManager {

protected:
	std::vector<Data> _allReceivedDatas;
	std::vector<Data> _trainingSet;
	std::vector<Data> _testingSet;

public:
	DataManager() { }
	~DataManager() { }
};