#include "pooling-operation.h"

/// -----------------
/// AveragePooling
/// -----------------

AveragePooling::AveragePooling(size_t rows, size_t cols)
	: _rows(rows), _cols(cols)
{
	_unitaryMatrix = Eigen::MatrixXd::Ones(_rows, _cols);
}



Matrix& AveragePooling::FowardPooling(Eigen::MatrixXd& input)
{
	const int rows = (int)std::ceil(input.rows() / _rows);
	const int cols = (int)std::ceil(input.cols() / _cols);

	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			double sum = input.block(i * _rows, j * _cols, _rows, _cols).cwiseProduct(_unitaryMatrix).sum();
			result(i, j) = sum / (double)(_rows * _cols);
		}
	}

	return result;
}



size_t AveragePooling::Rows()
{
	return _rows;
}



size_t AveragePooling::Cols()
{
	return _cols;
}



Matrix AveragePooling::BackwardPooling(Eigen::MatrixXd& input)
{
	return Matrix();
}






/// -----------------
/// MaxPooling
/// -----------------

MaxPooling::MaxPooling(size_t rows, size_t cols)
	: _rows(rows), _cols(cols)
{
}



Matrix& MaxPooling::FowardPooling(Eigen::MatrixXd& input)
{
	const int rows = input.rows() / _rows;
	const int cols = input.cols() / _cols;

	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(rows, cols);

	_backwardMatrix = Eigen::MatrixXd::Zero(input.rows(), input.cols());

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			Eigen::Index maxRow, maxCol;

			double maxElement = input.block(i * _rows, j * _cols, _rows, _cols).maxCoeff(&maxRow, &maxCol);
			result(i, j) = maxElement;

			_backwardMatrix(i * _rows + maxRow, j * _cols + maxCol) = 1.0;
		}
	}

	return result;
}



Matrix MaxPooling::BackwardPooling(Eigen::MatrixXd& input)
{
	for (int i = 0; i < input.rows(); i++)
		for (int j = 0; j < input.cols(); j++) 
			_backwardMatrix.block(i * _rows, j * _cols, _rows, _cols) *= input(i, j);

	return _backwardMatrix;
}



size_t MaxPooling::Rows()
{
	return _rows;
}



size_t MaxPooling::Cols()
{
	return _cols;
}




/// -----------------
/// Don't do Pooling / NoPool
/// -----------------

DontPooling::DontPooling()
{
}

Matrix& DontPooling::FowardPooling(Eigen::MatrixXd& input)
{
	return input;
}

size_t DontPooling::Rows()
{
	return 1;
}

size_t DontPooling::Cols()
{
	return 1;
}

Matrix DontPooling::BackwardPooling(Eigen::MatrixXd& input)
{
	return input;
}