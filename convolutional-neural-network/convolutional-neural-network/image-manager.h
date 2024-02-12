#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "mlp-input-data.h"

class ImageManager {
	private:
	cv::Mat _image;
	std::vector<DATA> _trainingSet;

	void ExtractDatas(cv::Mat& image);
	static int MaxChannel(const cv::Vec3d& pixel);

	public:
	ImageManager(std::string filePath);
	~ImageManager();

	std::vector<DATA> TrainigSet();
	std::vector<std::vector<double>> ProcessingSet();

};