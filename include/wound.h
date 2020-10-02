/*

WOUND

This code is the implementation of the wound model
Particularly, the Dalai Lama Wound Healing, or DaLaWoHe

*/

#ifndef wound_h
#define wound_h

#include <vector>
#include <Eigen/Dense> // most of the vector functions I will need inside of an element
using namespace Eigen;


//========================================================//
// RESIDUAL AND TANGENT
//
void evalWound(
double dt,
const std::vector<Matrix2d> &ip_Jac,
const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
const std::vector<double> &node_rho_0, const std::vector<double> &node_c_0,
std::vector<Matrix2d> &ip_strain_pi, std::vector<Matrix2d> &ip_stress_pi, std::vector<Vector2d> &ip_lamdaE_pi,
const std::vector<double> &ip_phif_0,const std::vector<Vector2d> &ip_a0_0,const std::vector<double> &ip_kappa_0, const std::vector<Vector2d> &ip_lamdaP_0,
const std::vector<double> &node_rho, const std::vector<double> &node_c,
std::vector<double> &ip_phif, std::vector<Vector2d> &ip_a0, std::vector<double> &ip_kappa, std::vector<Vector2d> &ip_lamdaP,
const std::vector<Vector2d> &node_x,
VectorXd &Re_x,MatrixXd &Ke_x_x,MatrixXd &Ke_x_rho,MatrixXd &Ke_x_c,
VectorXd &Re_rho,MatrixXd &Ke_rho_x, MatrixXd &Ke_rho_rho,MatrixXd &Ke_rho_c,
VectorXd &Re_c,MatrixXd &Ke_c_x,MatrixXd &Ke_c_rho,MatrixXd &Ke_c_c);
//
//========================================================//



//========================================================//
// EVAL SOURCE AND FLUX 
//
void evalFluxesSources(const std::vector<double> &global_parameters, double phif,Vector2d a0,double kappa,Vector2d lampdaP,
Matrix2d FF,double rho, double c, Vector2d Grad_rho, Vector2d Grad_c,
Matrix2d & SS,Vector2d &Q_rho,double &S_rho, Vector2d &Q_c,double &S_c);
//
//========================================================//



//========================================================//
// LOCAL PROBLEM: structural update
//
void localWoundProblem(
double dt, const std::vector<double> &local_parameters,
double C,double rho,const Matrix2d &CC,
double phif_0, const Vector2d &a0_0, double kappa_0, const Vector2d &lamdaP_0,
double &phif, Vector2d &a0, double &kappa, Vector2d &lamdaP,
VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadC);
//
//========================================================//



//========================================================//
// LOCAL PROBLEM: structural update
//
void localWoundProblemExplicit(
        double dt, const std::vector<double> &local_parameters,
        double C,double rho,const Matrix2d &FF,
        const double &phif_0, const Vector2d &a0_0, const double &kappa_0, const Vector2d &lamdaP_0,
        double &phif, Vector2d &a0, double &kappa, Vector2d &lamdaP,
        VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadC);

void evalForwardEulerUpdate(double local_dt, const std::vector<double> &local_parameters, double c,double rho,const Matrix2d &FF, const Matrix2d &CC,
                            const double &phif, const Vector2d &a0, const double &kappa, const Vector2d &lamdaP,
                            double &phif_dot, Vector2d &a0_dot, double &kappa_dot, Vector2d &lamdaP_dot);
//
//========================================================//



// ========================================================//
// OUTPUT FUNCTION: eval FF at coordinates xi, eta
//
Matrix2d evalWoundFF(
double dt,
const std::vector<Matrix2d> &ip_Jac,
const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
const std::vector<double> &node_X,
std::vector<double> &ip_phif, std::vector<Vector2d> &ip_a0, std::vector<double> &ip_kappa, std::vector<Vector2d> &ip_lamdaP,
const std::vector<Vector2d> &node_x,const std::vector<double> &node_rho, const std::vector<double> &node_c,
double xi, double eta);
//
//========================================================//


/////////////////////////////////////////////////////////////////////////////////////////
// GEOMETRY and ELEMENT ROUTINES
/////////////////////////////////////////////////////////////////////////////////////////

//-----------------------------//
// Jacobians, at all ip and xi,eta
//
std::vector<Matrix2d> evalJacobian(const std::vector<Vector2d> node_X);
Matrix2d evalJacobian(const std::vector<Vector2d> node_X, double xi, double eta);
//
//-----------------------------//

//-----------------------------//
// Integration points
//
std::vector<Vector3d> LineQuadriIP();
//
//-----------------------------//

//-----------------------------//
// Basis functions
//
std::vector<double> evalShapeFunctionsR(double xi,double eta);
std::vector<double> evalShapeFunctionsRxi(double xi,double eta);
std::vector<double> evalShapeFunctionsReta(double xi,double eta);
std::vector<double> evalShapeFunctionsQuadraticR(double xi,double eta);
std::vector<double> evalShapeFunctionsQuadraticRxi(double xi,double eta);
std::vector<double> evalShapeFunctionsQuadraticReta(double xi,double eta);
//
//-----------------------------//



/////////////////////////////////////////////////////////////////////////////////////////
// NUMERICAL CHECKS
/////////////////////////////////////////////////////////////////////////////////////////

//-----------------------------//
// Eval strain energy 
//
void evalPsif(const std::vector<double> &global_parameters,double kappa, double I1e,double I4e,double &Psif,double &Psif1,double &Psif4);
//
//-----------------------------//

//-----------------------------//
// Eval passive reference stress
//
void evalSS(const std::vector<double> &global_parameters, double phif, Vector2d a0, double kappa, double lamdaP_a,double lamdaP_s,const Matrix2d &CC,double rho, double c, Matrix2d &SSpas,Matrix2d &SSact, Matrix2d&SSpres);
//
//-----------------------------//


//-----------------------------//
// Eval residuals only
//
void evalWoundRes(
double dt,
const std::vector<Matrix2d> &ip_Jac,
const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
const std::vector<double> &node_rho_0, const std::vector<double> &node_c_0,
const std::vector<double> &ip_phif_0,const std::vector<Vector2d> &ip_a0_0,const std::vector<double> &ip_kappa_0, const std::vector<Vector2d> &ip_lamdaP_0,
const std::vector<double> &node_rho, const std::vector<double> &node_c,
std::vector<double> &ip_phif, std::vector<Vector2d> &ip_a0, std::vector<double> &ip_kappa, std::vector<Vector2d> &ip_lamdaP,
const std::vector<Vector2d> &node_x,
VectorXd &Re_x,VectorXd &Re_rho,VectorXd &Re_c);
//
//-----------------------------//


#endif