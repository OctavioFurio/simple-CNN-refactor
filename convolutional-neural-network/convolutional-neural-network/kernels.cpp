#include "kernels.h"

Eigen::MatrixXd Kernel3x3::Ones(size_t row, size_t col)
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd::Ones(row, col);
	return kernel;
}



Eigen::MatrixXd Kernel3x3::Zero(size_t row, size_t col)
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd::Zero(row, col);
	return kernel;
}



Eigen::MatrixXd Kernel3x3::Constant(size_t row, size_t col, const double& val)
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd::Constant(row, col, val);
	return kernel;
}



Eigen::MatrixXd Kernel3x3::RandomCloseToZero(size_t row, size_t col)
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<double> distribution(-0.2, 0.2);

	double smallestDouble = std::numeric_limits<double>::min();

	Eigen::MatrixXd kernel = Eigen::MatrixXd(row, col);
	kernel << distribution(generator);

	std::cout << "\n\n" << kernel << "\n\n\n";

	return kernel;
}



Eigen::MatrixXd Kernel3x3::Identity()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		0, 0, 0,
		0, 1, 0,
		0, 0, 0;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::Laplacian()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		0, 1, 0,
		1, -4, 1,
		0, 1, 0;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::EdgeEnhancement()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::AverageBlur() // BoxBlur
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		1.0 / 9, 1.0 / 9, 1.0 / 9,
		1.0 / 9, 1.0 / 9, 1.0 / 9,
		1.0 / 9, 1.0 / 9, 1.0 / 9;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::HorizontalLineDetection()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		-1, -1, -1,
		2, 2, 2,
		-1, -1, -1;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::VerticalLineDetection()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		-1, 2, -1,
		-1, 2, -1,
		-1, 2, -1;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::RobinsonCompassNorth()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::RobinsonCompassNorthWest()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		0, 1, 2,
		-1, 0, 1,
		-2, -1, 0;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::RobinsonCompassWest()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		1, 2, 1,
		0, 0, 0,
		-1, -2, -1;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::RobinsonCompassSouthWest()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		2, 1, 0,
		1, 0, -1,
		0, -1, -2;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::RobinsonCompassSouth()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		1, 0, -1,
		2, 0, -2,
		1, 0, -1;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::RobinsonCompassSouthEast()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		0, -1, -2,
		1, 0, -1,
		2, 1, 0;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::RobinsonCompassEast()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::RobinsonCompassNorthEast()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		2, -1, 0,
		-1, 0, 1,
		0, 1, 2;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::KrischCompassNorth()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		-3, -3, 5,
		-3, 0, 5,
		-3, -3, 5;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::KrischCompassNorthWest()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		-3, 5, 5,
		-3, 0, 5,
		-3, -3, -3;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::KrischCompassWest()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		5, 5, 5,
		-3, 0, -3,
		-3, -3, -3;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::KrischCompassSouthWest()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		5, 5, -3,
		5, 0, -3,
		-3, -3, -3;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::KrischCompassSouth()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		5, -3, -3,
		5, 0, -3,
		5, -3, -3;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::KrischCompassSouthEast()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		-3, -3, -3,
		5, 0, -3,
		5, 5, -3;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::KrischCompassEast()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		-3, -3, -3,
		-3, 0, -3,
		5, 5, 5;

	return kernel;
}



Eigen::MatrixXd Kernel3x3::KrischCompassNorthEast()
{
	Eigen::MatrixXd kernel = Eigen::MatrixXd(3, 3);
	kernel <<
		-3, -3, -3,
		-3, 0, 5,
		-3, 5, 5;

	return kernel;
}