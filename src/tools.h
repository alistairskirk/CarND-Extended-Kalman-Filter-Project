#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * A helper method to calculate Jacobians.
  */
  MatrixXd CalculateJacobian(const VectorXd& x_state);

  /**
  * A helper method to calculate h(x').
  */
  VectorXd CalcHx(const VectorXd& x_state);

  /**
  * A helper method to calculate Polar to Cartesian.
  */
  VectorXd PolarToCart(const VectorXd& x_state);
};

#endif /* TOOLS_H_ */
