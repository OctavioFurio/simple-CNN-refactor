#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <limits>



namespace Kernel3x3 {

    

    Eigen::MatrixXd Ones(size_t row, size_t col);

    Eigen::MatrixXd Zero(size_t row, size_t col);

    Eigen::MatrixXd Constant(size_t row, size_t col, const double& val);

    Eigen::MatrixXd RandomCloseToZero(size_t row, size_t col);

    Eigen::MatrixXd Identity();

    Eigen::MatrixXd Laplacian();

    Eigen::MatrixXd EdgeEnhancement();

    Eigen::MatrixXd AverageBlur();

    Eigen::MatrixXd HorizontalLineDetection();

    Eigen::MatrixXd VerticalLineDetection();

    Eigen::MatrixXd RobinsonCompassNorth();
    Eigen::MatrixXd RobinsonCompassNorthWest();
    Eigen::MatrixXd RobinsonCompassWest();
    Eigen::MatrixXd RobinsonCompassSouthWest();
    Eigen::MatrixXd RobinsonCompassSouth();
    Eigen::MatrixXd RobinsonCompassSouthEast();
    Eigen::MatrixXd RobinsonCompassEast();
    Eigen::MatrixXd RobinsonCompassNorthEast();

    Eigen::MatrixXd KrischCompassNorth();
    Eigen::MatrixXd KrischCompassNorthWest();
    Eigen::MatrixXd KrischCompassWest();
    Eigen::MatrixXd KrischCompassSouthWest();
    Eigen::MatrixXd KrischCompassSouth();
    Eigen::MatrixXd KrischCompassSouthEast();
    Eigen::MatrixXd KrischCompassEast();
    Eigen::MatrixXd KrischCompassNorthEast();
    

}


