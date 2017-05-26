#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	//borrowed from Udacity Course
	if (estimations.size() != ground_truth.size()
		|| estimations.size() == 0) {
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i) {

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse / estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3, 4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check division by zero
	if ((px == 0)&(py == 0)) {
		cout << "divide by zero error" << endl;
		return Hj;
	}

	//compute the Jacobian matrix
	float pxpy2 = pow(px, 2) + pow(py, 2);
	Hj << px / pow(pxpy2, 0.5), py / pow(pxpy2, 0.5), 0., 0.,
		-py / pxpy2, px / pxpy2, 0., 0.,
		py*(vx*py - vy*px) / pow(pxpy2, (3 / 2)), px*(vy*px - vx*py) / pow(pxpy2, (3 / 2)), px / pow(pxpy2, 0.5), py / pow(pxpy2, 0.5);
	
	return Hj;
}

VectorXd Tools::CalcHx(const VectorXd& x_state) {
	//init the return vector h(x')
	VectorXd zpred(3);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check division by zero
	if ((px == 0)&(py == 0)) {
		cout << "divide by zero error, can only mean your target is at your origin, space-time internal error" << endl;
		return zpred;
	}

	//compute the h(x') matrix
	float pxpy2 = pow(px, 2) + pow(py, 2);
	zpred << pow(pxpy2, 0.5),
		atan2(py, px),
		(px*vx + py*vy) / pow(pxpy2, 0.5);
	
	//cout <<"zpred:\n " << zpred << endl;

	return zpred;
}

VectorXd Tools::PolarToCart(const VectorXd& x_state) {
	//init the return vector cartesian coords
	VectorXd cartesian(4);

	//recover state parameters
	float rho = x_state(0);
	float phi = x_state(1);
	float rhodot = x_state(2);	

	//compute the cartesian coords
	float px;
	float py;
	
	// deriving px and py from polar coords:
	float denom = pow(1 + pow(tan(phi), 2), 0.5);
	px = rho / denom;
	py = tan(phi)*px;
	cartesian << px, py, 0, 0;

	return cartesian;
}
