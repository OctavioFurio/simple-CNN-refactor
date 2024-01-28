#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

using Matrix = Eigen::MatrixXd;



class IPooling {
	public:
		virtual Matrix& FowardPooling(Eigen::MatrixXd& input) = 0;
		virtual Matrix BackwardPooling(Eigen::MatrixXd& input) = 0;
		virtual size_t Rows() = 0;
		virtual size_t Cols() = 0;
};



class DontPooling : public IPooling {
	public:
		DontPooling();
		virtual Matrix& FowardPooling(Eigen::MatrixXd& input) override;
		virtual Matrix BackwardPooling(Eigen::MatrixXd& input) override;
		virtual size_t Rows() override;
		virtual size_t Cols() override;
		
};
 


class AveragePooling : public IPooling {
	protected:
		Eigen::MatrixXd _unitaryMatrix;
		size_t _rows;
		size_t _cols;

	public:
		AveragePooling(size_t rows, size_t cols);

		virtual Matrix& FowardPooling(Eigen::MatrixXd& input) override;
		virtual Matrix BackwardPooling(Eigen::MatrixXd& input) override;
		virtual size_t Rows() override;
		virtual size_t Cols() override;
};




class MaxPooling : public IPooling {
	protected:
		size_t _rows;
		size_t _cols;
		Eigen::MatrixXd _backwardMatrix;

	public:
		MaxPooling(size_t rows, size_t cols);

		virtual Matrix& FowardPooling(Eigen::MatrixXd& input) override;
		virtual Matrix BackwardPooling(Eigen::MatrixXd& input) override;
		virtual size_t Rows() override;
		virtual size_t Cols() override;
};


