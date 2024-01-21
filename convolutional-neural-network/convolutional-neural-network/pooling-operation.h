#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

using Matrix = Eigen::MatrixXd;



class IPooling {
	public:
		virtual Matrix Pooling(Eigen::MatrixXd& input) = 0;
};



class DontPooling : public IPooling {
	public:
		DontPooling();
		virtual Matrix Pooling(Eigen::MatrixXd& input) override;
};



class MaxPooling : public IPooling {
	protected:
		Eigen::MatrixXd _unitaryMatrix;
		unsigned _rows;
		unsigned _cols;

	public:
		MaxPooling(unsigned rows, unsigned cols);
		virtual Matrix Pooling(Eigen::MatrixXd& input) override;
};




class AveragePooling : public IPooling {
	protected:
		unsigned _rows;
		unsigned _cols;

	public:
		AveragePooling(unsigned rows, unsigned cols);
		virtual Matrix Pooling(Eigen::MatrixXd& input) override;
};


