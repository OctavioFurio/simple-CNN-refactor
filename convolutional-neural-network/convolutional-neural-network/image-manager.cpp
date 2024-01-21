#include "image-manager.h"




ImageManager::ImageManager(std::string filePath)
{
	_image = cv::imread(filePath);

	assert(_image.data == nullptr  ||  _image.rows != 28  ||  _image.cols != 28);

	_image.convertTo(_image, CV_64FC3);

	ExtractDatas(_image);
}



ImageManager::~ImageManager()
{
}



std::vector<DATA> ImageManager::TrainigSet()
{
	return _trainingSet;
}



std::vector<std::vector<double>> ImageManager::ProcessingSet()
{
	std::vector<std::vector<double>> processingSet;
	
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			
			double normalizedI = Utils::Normalize(i, 0.0, 27.0, -1.0, 1.0);
			double normalizedJ = Utils::Normalize(j, 0.0, 27.0, -1.0, 1.0);

			processingSet.push_back( {normalizedI,  normalizedJ} );

		}
	}

	return processingSet;
}



void ImageManager::ExtractDatas(cv::Mat& image)
{
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			cv::Vec3d pixel = image.at<cv::Vec3d>(i, j);

			if (pixel != cv::Vec3d{ 255.0, 255.0, 255.0 }) {
				double normalizedI = Utils::Normalize(i, 0.0, 27.0, -1.0, 1.0);
				double normalizedJ = Utils::Normalize(j, 0.0, 27.0, -1.0, 1.0);

				int labelIndex = MaxChannel(pixel);

				std::cout << "[" << pixel[0] << " ; " << pixel[1] << " ; " << pixel[2] << "] -> ";
				std::cout << labelIndex << "  ";
				std::cout << i << "x" << j << "\n";

				_trainingSet.push_back(DATA{ {normalizedI,  normalizedJ}, labelIndex });
			}

		}
	}
}



int ImageManager::MaxChannel(const cv::Vec3d& pixel)
{
	int maxChannel = 0;

	if (pixel[1] >= pixel[maxChannel]) { maxChannel = 1; }

	if (pixel[2] >= pixel[maxChannel]) { maxChannel = 2; }

	return maxChannel;
}
