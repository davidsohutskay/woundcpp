/*

WOUND

This code is the implementation of the DaLaWoHe

*/

//#define EIGEN_USE_MKL_ALL
#include <omp.h>
#include "wound.h"

#include <iostream>
#include <math.h> 
#include <stdexcept> 


//--------------------------------------------------------//
// RESIDUAL AND TANGENT
//--------------------------------------------------------//

// ELEMENT RESIDUAL AND TANGENT
void evalWound(
double dt,
const std::vector<Matrix2d> &ip_Jac,
const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
const std::vector<double> &node_rho_0, const std::vector<double> &node_c_0,
std::vector<Matrix2d> &ip_strain, std::vector<Matrix2d> &ip_stress, std::vector<Vector2d> &ip_lamdaE,
const std::vector<double> &ip_phif_0,const std::vector<Vector2d> &ip_a0_0,const std::vector<double> &ip_kappa_0, const std::vector<Vector2d> &ip_lamdaP_0,
const std::vector<double> &node_rho, const std::vector<double> &node_c,
std::vector<double> &ip_phif, std::vector<Vector2d> &ip_a0, std::vector<double> &ip_kappa, std::vector<Vector2d> &ip_lamdaP,
const std::vector<Vector2d> &node_x,
VectorXd &Re_x,MatrixXd &Ke_x_x,MatrixXd &Ke_x_rho,MatrixXd &Ke_x_c,
VectorXd &Re_rho,MatrixXd &Ke_rho_x, MatrixXd &Ke_rho_rho,MatrixXd &Ke_rho_c,
VectorXd &Re_c,MatrixXd &Ke_c_x,MatrixXd &Ke_c_rho,MatrixXd &Ke_c_c)
{
    Eigen::initParallel();
	//std::cout<<"element routine\n";
	//---------------------------------//
	// INPUT
	//  dt: time step
	//	elem_jac_IP: jacobians at the integration points, needed for the deformation grad
	//  matParam: material parameters
	//  Xi_t: global fields at previous time step
	//  Theta_t: structural fields at previous time steps
	//  Xi: current guess of the global fields
	//  Theta: current guess of the structural fields
	//	node_x: deformed positions
	//
	// OUTPUT
	//  Re: all residuals
	//  Ke: all tangents
	//
	// Algorithm
	//  0. Loop over integration points
	//	1. F,rho,c,nabla_rho,nabla_c: deformation at IP
	//  2. LOCAL NEWTON -> update the current guess of the structural parameters
	//  3. Fe,Fp
	//	4. Se_pas,Se_act,S
	//	5. Qrho,Srho,Qc,Sc
	//  6. Residuals
	//  7. Tangents
	//---------------------------------//
	
	
	
	//---------------------------------//
	// PARAMETERS
	// 
    double k0 = global_parameters[0]; // neo hookean
    double kf = global_parameters[1]; // stiffness of collagen
    double k2 = global_parameters[2]; // nonlinear exponential
    double t_rho = global_parameters[3]; // force of fibroblasts
    double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
    double K_t = global_parameters[5]; // saturation of collagen on force
    double K_t_c = global_parameters[6]; // saturation of chemical on force
    double D_rhorho = global_parameters[7]; // diffusion of cells
    double D_rhoc = global_parameters[8]; // diffusion of chemotactic gradient
    double D_cc = global_parameters[9]; // diffusion of chemical
    double p_rho =global_parameters[10]; // production of fibroblasts naturally
    double p_rho_c = global_parameters[11]; // production enhanced by the chem
    double p_rho_theta = global_parameters[12]; // mechanosensing
    double K_rho_c= global_parameters[13]; // saturation of cell production by chemical
    double K_rho_rho = global_parameters[14]; // saturation of cell by cell
    double d_rho = global_parameters[15] ;// decay of cells
    double vartheta_e = global_parameters[16]; // physiological state of area stretch
    double gamma_theta = global_parameters[17]; // sensitivity of heviside function
    double p_c_rho = global_parameters[18];// production of C by cells
    double p_c_thetaE = global_parameters[19]; // coupling of elastic and chemical
    double K_c_c = global_parameters[20];// saturation of chem by chem
    double d_c = global_parameters[21]; // decay of chemical
	//std::cout<<"read all global parameters\n";
	//
	//---------------------------------//

    // voigt table
    Vector3d voigt_table_I_i(0,1,0);
    Vector3d voigt_table_I_j(0,1,1);
    Vector3d voigt_table_J_k(0,1,0);
    Vector3d voigt_table_J_l(0,1,1);
	
	//---------------------------------//
	// GLOBAL VARIABLES
	// Initialize the residuals to zero and declare some global stuff
	Re_x.setZero();
	Re_rho.setZero();
	Re_c.setZero();
	Ke_x_x.setZero();
	Ke_x_rho.setZero();
	Ke_x_c.setZero();
	Ke_rho_x.setZero();
	Ke_rho_rho.setZero();
	Ke_rho_c.setZero();
	Ke_c_x.setZero();
	Ke_c_rho.setZero();
	Ke_c_c.setZero();
	int n_nodes = node_x.size();
	std::vector<Vector2d> Ebasis; Ebasis.clear();
	Matrix2d Rot90;Rot90<<0.,-1.,1.,0.;
	Ebasis.push_back(Vector2d(1.,0.)); Ebasis.push_back(Vector2d(0.,1.));
	//---------------------------------//
	
	
	
	//---------------------------------//
	// LOOP OVER INTEGRATION POINTS
	//---------------------------------//
	
	// array with integration points
	std::vector<Vector3d> IP = LineQuadriIP();
	//std::cout<<"loop over integration points\n";
	for(int ip=0;ip<IP.size();ip++)
	{
	
		//---------------------------------//
		// EVALUATE FUNCTIONS 
		//
		// coordinates of the integration point in parent domain
		double xi = IP[ip](0);
		double eta = IP[ip](1);
		// weight of the integration point
		double wip = IP[ip](2);
		double Jac = 1./ip_Jac[ip].determinant(); // instead of Jacobian, J^(-T) is stored
		//std::cout<<"integration point: "<<xi<<", "<<eta<<"; "<<wip<<"; "<<Jac<<"\n";
		//
        std::vector<double> R;
        std::vector<double> Rxi;
        std::vector<double> Reta;
		if(n_nodes == 4){
            // eval linear shape functions (4 of them)
            R = evalShapeFunctionsR(xi,eta);
            // eval derivatives
            Rxi = evalShapeFunctionsRxi(xi,eta);
            Reta = evalShapeFunctionsReta(xi,eta);
		}
		else if(n_nodes == 8){
            // eval quadratic shape functions (4 of them)
            R = evalShapeFunctionsQuadraticR(xi,eta);
            // eval derivatives
            Rxi = evalShapeFunctionsQuadraticRxi(xi,eta);
            Reta = evalShapeFunctionsQuadraticReta(xi,eta);
		}
		else{
            throw std::runtime_error("Wrong number of nodes in element!");
		}
		//
		// declare variables and gradients at IP
		std::vector<Vector2d> dRdXi;dRdXi.clear();
		Vector2d dxdxi,dxdeta;
		dxdxi.setZero();dxdeta.setZero();
		double rho_0=0.; Vector2d drho0dXi; drho0dXi.setZero();
		double rho=0.; Vector2d drhodXi; drhodXi.setZero();
		double c_0=0.; Vector2d dc0dXi; dc0dXi.setZero();
		double c=0.; Vector2d dcdXi; dcdXi.setZero();
		//
		for(int ni=0;ni<n_nodes;ni++)
		{
			dRdXi.push_back(Vector2d(Rxi[ni],Reta[ni]));
			
			dxdxi += node_x[ni]*Rxi[ni];
			dxdeta += node_x[ni]*Reta[ni];
			
			rho_0 += node_rho_0[ni]*R[ni];
			drho0dXi(0) += node_rho_0[ni]*Rxi[ni];
			drho0dXi(1) += node_rho_0[ni]*Reta[ni];
			
			rho += node_rho[ni]*R[ni];
			drhodXi(0) += node_rho[ni]*Rxi[ni];
			drhodXi(1) += node_rho[ni]*Reta[ni];
			
			c_0 += node_c_0[ni]*R[ni];
			dc0dXi(0) += node_c_0[ni]*Rxi[ni];
			dc0dXi(1) += node_c_0[ni]*Reta[ni];

			c += node_c[ni]*R[ni];
			dcdXi(0) += node_c[ni]*Rxi[ni];
			dcdXi(1) += node_c[ni]*Reta[ni];
		}
		//
		//---------------------------------//



		//---------------------------------//
		// EVAL GRADIENTS
		//
		// Deformation gradient and strain
		// assemble the columns
		Matrix2d dxdXi; dxdXi<<dxdxi(0),dxdeta(0),dxdxi(1),dxdeta(1);
		// F = dxdX 
        Matrix2d FF = dxdXi*ip_Jac[ip].transpose();
		// the strain
		Matrix2d Identity;Identity<<1,0,0,1;
		Matrix2d EE = 0.5*(FF.transpose()*FF - Identity);
        ip_strain[ip] = EE;
        Matrix2d CC = FF.transpose()*FF;
		Matrix2d CCinv = CC.inverse();
		//
		// Gradient of concentrations in current configuration
		Matrix2d dXidx = dxdXi.inverse();
		Vector2d grad_rho0 = dXidx.transpose()*drho0dXi;
		Vector2d grad_rho  = dXidx.transpose()*drhodXi;
		Vector2d grad_c0   = dXidx.transpose()*dc0dXi;
		Vector2d grad_c    = dXidx.transpose()*dcdXi;
		//
		// Gradient of concentrations in reference configuration
		Vector2d Grad_rho0 = ip_Jac[ip]*drho0dXi;
		Vector2d Grad_rho = ip_Jac[ip]*drhodXi;
		Vector2d Grad_c0 = ip_Jac[ip]*dc0dXi;
		Vector2d Grad_c = ip_Jac[ip]*dcdXi;
		//
		// Gradient of basis functions for the nodes in reference
		std::vector<Vector2d> Grad_R;Grad_R.clear();
		// Gradient of basis functions in deformed configuration
		std::vector<Vector2d> grad_R;grad_R.clear();
        for(int ni=0;ni<n_nodes;ni++)
        {
            Grad_R.push_back(ip_Jac[ip]*dRdXi[ni]);
            grad_R.push_back(dXidx.transpose()*dRdXi[ni]);
        }
		//
		//---------------------------------//

		//std::cout<<"deformation gradient\n"<<FF<<"\n";
		//std::cout<<"rho0: "<<rho_0<<", rho: "<<rho<<"\n";
		//std::cout<<"c0: "<<c_0<<", c: "<<c<<"\n";
		//std::cout<<"gradient of rho: "<<Grad_rho<<"\n";
		//std::cout<<"gradient of c: "<<Grad_c<<"\n";



		//---------------------------------//
		// LOCAL NEWTON: structural problem
		//
        //VectorXd dThetadCC(18);dThetadCC.setZero();
		VectorXd dThetadCC(24);dThetadCC.setZero();
		VectorXd dThetadrho(6);dThetadrho.setZero();
		VectorXd dThetadc(6);dThetadc.setZero();
		//std::cout<<"Local variables before update:\nphif0 = "<<ip_phif_0[ip]<<"\nkappa_0 = "<<ip_kappa_0[ip]<<"\na0_0 = ["<<ip_a0_0[ip](0)<<","<<ip_a0_0[ip](1)<<"]\nlamdaP_0 = ["<<ip_lamdaP_0[ip](0)<<","<<ip_lamdaP_0[ip](1)<<"]\n";
		localWoundProblem(dt,local_parameters,c,rho,CC,ip_phif_0[ip],ip_a0_0[ip],ip_kappa_0[ip],ip_lamdaP_0[ip],ip_phif[ip],ip_a0[ip],ip_kappa[ip],ip_lamdaP[ip],dThetadCC,dThetadrho,dThetadc);
        //localWoundProblemExplicit(dt,local_parameters,c,rho,FF,ip_phif_0[ip],ip_a0_0[ip],ip_kappa_0[ip],ip_lamdaP_0[ip],ip_phif[ip],ip_a0[ip],ip_kappa[ip],ip_lamdaP[ip],dThetadCC,dThetadrho,dThetadc);
		//
		// rename variables to make it easier to track
		double phif_0 = ip_phif_0[ip];
		Vector2d a0_0 = ip_a0_0[ip];
		double kappa_0 = ip_kappa_0[ip];
		Vector2d lamdaP_0 = ip_lamdaP_0[ip];
		double phif = ip_phif[ip];
		Vector2d a0 = ip_a0[ip];
		double kappa = ip_kappa[ip];
		Vector2d lamdaP = ip_lamdaP[ip];
		double lamdaP_a_0 = lamdaP_0(0);
		double lamdaP_s_0 = lamdaP_0(1);
		double lamdaP_a = lamdaP(0);
		double lamdaP_s = lamdaP(1);
		//std::cout<<"Local variables after update:\nphif0 = "<<phif_0<<",	phif = "<<phif<<"\nkappa_0 = "<<kappa_0<<",	kappa = "<<kappa<<"\na0_0 = ["<<a0_0(0)<<","<<a0_0(1)<<"],	a0 = ["<<a0(0)<<","<<a0(1)<<"]\nlamdaP_0 = ["<<lamdaP_0(0)<<","<<lamdaP_0(1)<<"],	lamdaP = ["<<lamdaP(0)<<","<<lamdaP(1)<<"]\n";
		// make sure the update preserved length
		double norma0 = sqrt(a0.dot(a0));
		if(fabs(norma0-1.)>0.001){std::cout<<"update did not preserve unit length of a0\n";}
		ip_a0[ip] = a0/(sqrt(a0.dot(a0)));
		a0 = a0/(sqrt(a0.dot(a0)));
		//
		// unpack the derivatives wrt CC
        Matrix2d dphifdCC; dphifdCC.setZero();
        Matrix2d da0xdCC;da0xdCC.setZero();
        Matrix2d da0ydCC;da0ydCC.setZero();
        Matrix2d dkappadCC; dkappadCC.setZero();
        Matrix2d dlamdaP_adCC; dlamdaP_adCC.setZero();
        Matrix2d dlamdaP_sdCC; dlamdaP_sdCC.setZero();
		// remember dThetatCC: 4 phi, 4 a0x, 4 a0y, 4 kappa, 4 lamdaPa, 4 lamdaPs
		dphifdCC(0,0) = dThetadCC(0);
		dphifdCC(0,1) = dThetadCC(1);
		dphifdCC(1,0) = dThetadCC(2);
		dphifdCC(1,1) = dThetadCC(3);
		da0xdCC(0,0) = dThetadCC(4);
		da0xdCC(0,1) = dThetadCC(5);
		da0xdCC(1,0) = dThetadCC(6);
		da0xdCC(1,1) = dThetadCC(7);
		da0ydCC(0,0) = dThetadCC(8);
		da0ydCC(0,1) = dThetadCC(9);
		da0ydCC(1,0) = dThetadCC(10);
		da0ydCC(1,1) = dThetadCC(11);
		dkappadCC(0,0) = dThetadCC(12);
		dkappadCC(0,1) = dThetadCC(13);
		dkappadCC(1,0) = dThetadCC(14);
		dkappadCC(1,1) = dThetadCC(15);
        dlamdaP_adCC(0,0) = dThetadCC(16);
        dlamdaP_adCC(0,1) = dThetadCC(17);
        dlamdaP_adCC(1,0) = dThetadCC(18);
        dlamdaP_adCC(1,1) = dThetadCC(19);
        dlamdaP_sdCC(0,0) = dThetadCC(20);
        dlamdaP_sdCC(0,1) = dThetadCC(21);
        dlamdaP_sdCC(1,0) = dThetadCC(22);
        dlamdaP_sdCC(1,1) = dThetadCC(23);
//        for (int II=0; II<3; II++){
//            int ii = voigt_table_I_i(II);
//            int jj = voigt_table_I_j(II);
//            dphifdCC(ii,jj) = dThetadCC(0+II);
//            da0xdCC(ii,jj) = dThetadCC(3+II);
//            da0ydCC(ii,jj) = dThetadCC(6+II);
//            dkappadCC(ii,jj) = dThetadCC(9+II);
//            dlamdaP_adCC(ii,jj) = dThetadCC(12+II);
//            dlamdaP_sdCC(ii,jj) = dThetadCC(15+II);
//            if(ii!=jj){
//                dphifdCC(jj,ii) = dThetadCC(0+II);
//                da0xdCC(jj,ii) = dThetadCC(3+II);
//                da0ydCC(jj,ii) = dThetadCC(6+II);
//                dkappadCC(jj,ii) = dThetadCC(9+II);
//                dlamdaP_adCC(jj,ii) = dThetadCC(12+II);
//                dlamdaP_sdCC(jj,ii) = dThetadCC(15+II);
//            }
//        }
		// unpack the derivatives wrt rho
		double dphifdrho = dThetadrho(0);
		double da0xdrho  = dThetadrho(1);
		double da0ydrho  = dThetadrho(2);
		double dkappadrho  = dThetadrho(3);
		double dlamdaP_adrho  = dThetadrho(4);
		double dlamdaP_sdrho  = dThetadrho(5);
		// unpack the derivatives wrt c
		double dphifdc = dThetadc(0);
		double da0xdc  = dThetadc(1);
		double da0ydc  = dThetadc(2);
		double dkappadc  = dThetadc(3);
		double dlamdaP_adc  = dThetadc(4);
		double dlamdaP_sdc  = dThetadc(5);
		//
		//---------------------------------//
		//std::cout<<"\n"<<dThetadCC<<"\n";


		//---------------------------------//
		// CALCULATE SOURCE AND FLUX
		//
		// Update kinematics
		CCinv = CC.inverse();
		// re-compute basis a0, s0
		Matrix2d Rot90;Rot90<<0.,-1.,1.,0.;
		Vector2d s0 = Rot90*a0;
		// fiber tensor in the reference
		Matrix2d a0a0 = a0*a0.transpose();
		Matrix2d s0s0 = s0*s0.transpose();
		Matrix2d A0 = kappa*Identity + (1-2.*kappa)*a0a0;
		Vector2d a = FF*a0;
		Matrix2d A = kappa*FF*FF.transpose() + (1.-2.0*kappa)*a*a.transpose();
		double trA = A(0,0) + A(1,1);
		Matrix2d hat_A = A/trA;
		// recompute split
		Matrix2d FFg = lamdaP_a*(a0a0) + lamdaP_s*(s0s0);
		double thetaP = lamdaP_a*lamdaP_s;
		Matrix2d FFginv = (1./lamdaP_a)*(a0a0) + (1./lamdaP_s)*(s0s0);
		Matrix2d FFe = FF*FFginv;
		//std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
		// elastic strain
		Matrix2d CCe = FFe.transpose()*FFe;
        // Elastic stretches of the directions a and s
        double Ce_aa = a0.transpose()*CCe*a0;
        double Ce_ss = s0.transpose()*CCe*s0;
        double lamdaE_a = sqrt(Ce_aa);
        double lamdaE_s = sqrt(Ce_ss);
        ip_lamdaE[ip] = Vector2d(lamdaE_a, lamdaE_s);
		// invariant of the elastic strain
		double I1e = CCe(0,0) + CCe(1,1);
		double I4e = a0.dot(CCe*a0);
		// calculate the normal stretch
		double thetaE = sqrt(CCe.determinant());
		double theta = thetaE*thetaP;
		//std::cout<<"split of the determinants. theta = thetaE*thetaB = "<<theta<<" = "<<thetaE<<"*"<<thetaP<<"\n";
		double lamda_N = 1./thetaE;
		double I4tot = a0.dot(CC*a0);
		// Second Piola Kirchhoff stress tensor
		// passive elastic
		double Psif = (kf/(2.*k2))*(exp( k2*pow((kappa*I1e + (1-2*kappa)*I4e -1),2))-1);
		double Psif1 = 2*k2*kappa*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif;
		double Psif4 = 2*k2*(1-2*kappa)*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif;
		//Matrix2d SSe_pas = k0*Identity + phif*(Psif1*Identity + Psif4*a0a0);
        Matrix2d SSe_pas = k0*Identity + phif*(Psif1*Identity + Psif4*a0a0);
		// pull back to the reference
		Matrix2d SS_pas = thetaP*FFginv*SSe_pas*FFginv;
		// magnitude from systems bio
		double traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho;
		//Matrix2d SS_act = (thetaP*traction_act*phif/trA)*A0;
        Matrix2d SS_act = (thetaP*traction_act*phif/(trA*(K_t*K_t+phif*phif)))*A0;
		// total stress, don't forget the pressure
		double pressure = -k0*lamda_N*lamda_N;
		Matrix2d SS_pres = pressure*thetaP*CCinv;
		//std::cout<<"stresses.\nSSpas\n"<<SS_pas<<"\nSS_act\n"<<SS_act<<"\nSS_pres"<<SS_pres<<"\n";
		//Matrix2d SS = SS_pas + SS_pres + SS_act;
        Matrix2d SS = SS_pas + SS_pres + SS_act;
        ip_stress[ip] = SS;
		Vector3d SS_voigt = Vector3d(SS(0,0),SS(1,1),SS(0,1));
		// Flux and Source terms for the rho and the C
		Vector2d Q_rho = -D_rhorho*CCinv*Grad_rho - D_rhoc*rho*CCinv*Grad_c;
		Vector2d Q_c = -D_cc*CCinv*Grad_c;
        //Vector2d Q_rho = -3*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0*Grad_rho/trA - 3*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0*Grad_c/trA;
        //Vector2d Q_c = -3*(D_cc-phif*(D_cc-D_cc/10))*A0*Grad_c/trA;
		// mechanosensing 
		double He = 1./(1.+exp(-gamma_theta*(thetaE - vartheta_e)));
		double S_rho = (p_rho + p_rho_c*c/(K_rho_c+c)+p_rho_theta*He)*(1-rho/K_rho_rho)*rho - d_rho*rho;
		// heviside function for elastic response of the chemical
		double S_c = (p_c_rho*c+ p_c_thetaE*He)*(rho/(K_c_c+c)) - d_c*c;
		//std::cout<<"flux of celss, Q _rho\n"<<Q_rho<<"\n";
		//std::cout<<"source of cells, S_rho: "<<S_rho<<"\n";
		//std::cout<<"flux of chemical, Q _c\n"<<Q_c<<"\n";
		//std::cout<<"source of chemical, S_c: "<<S_c<<"\n";
		//---------------------------------//
		
		
		
		//---------------------------------//
		// ADD TO THE RESIDUAL
		//
		Matrix2d deltaFF,deltaCC;
		Vector3d deltaCC_voigt;
		for(int nodei=0;nodei<n_nodes;nodei++){
			for(int coordi=0;coordi<2;coordi++){
				// alternatively, define the deltaCC
				deltaFF = Ebasis[coordi]*Grad_R[nodei].transpose();
				deltaCC = deltaFF.transpose()*FF + FF.transpose()*deltaFF;
				deltaCC_voigt = Vector3d(deltaCC(0,0),deltaCC(1,1),2.*deltaCC(1,0));
				Re_x(nodei*2+coordi) += Jac*SS_voigt.dot(deltaCC_voigt);
			}
            // Element residuals for rho and c
            Re_rho(nodei) += Jac*(((rho-rho_0)/dt - S_rho)*R[nodei] - Grad_R[nodei].dot(Q_rho));
            Re_c(nodei) += Jac*(((c-c_0)/dt - S_c)*R[nodei] - Grad_R[nodei].dot(Q_c));
		}
		//
		//---------------------------------//


		
		//---------------------------------//
		// TANGENTS
		//---------------------------------//
		
		
		
		//---------------------------------//
		// NUMERICAL DERIVATIVES
		//
		// NOTE:
		// the chain rule for structural parameters is a mess, might as well use 
		// numerical tangent, partially.
		// proposed: numerical tangent wrt structural parameters
		// then use the derivatives dThetadCC, dThetadrho, dThetadC
		// derivative wrt to CC is done analytically
		//
		double epsilon = 1e-7;
		//
		// structural parameters
		double phif_plus = phif + epsilon;
		double phif_minus= phif - epsilon;
		Vector2d a0_plus_x= a0 + epsilon*Ebasis[0];
		Vector2d a0_minus_x = a0 - epsilon*Ebasis[0];
		Vector2d a0_plus_y= a0 + epsilon*Ebasis[1];
		Vector2d a0_minus_y= a0 - epsilon*Ebasis[1];
		double kappa_plus = kappa + epsilon;
		double kappa_minus =kappa - epsilon;
		Vector2d lamdaP_plus_a = lamdaP + epsilon*Ebasis[0];
		Vector2d lamdaP_minus_a = lamdaP - epsilon*Ebasis[0];
		Vector2d lamdaP_plus_s = lamdaP + epsilon*Ebasis[1];
		Vector2d lamdaP_minus_s = lamdaP - epsilon*Ebasis[1];
		//
		// fluxes and sources
		Matrix2d SS_plus,SS_minus;
		Vector2d Q_rho_plus,Q_rho_minus;
		Vector2d Q_c_plus,Q_c_minus;
		double S_rho_plus,S_rho_minus;
		double S_c_plus,S_c_minus;
		//
		// phif
		evalFluxesSources(global_parameters,phif_plus,a0,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
		evalFluxesSources(global_parameters,phif_minus,a0,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
		Matrix2d dSSdphif = (1./(2.*epsilon))*(SS_plus-SS_minus);
		Vector2d dQ_rhodphif = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
		Vector2d dQ_cdphif = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
		double dS_rhodphif = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
		double dS_cdphif = (1./(2.*epsilon))*(S_c_plus-S_c_minus);		
		//
		// a0x
		evalFluxesSources(global_parameters,phif,a0_plus_x,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
		evalFluxesSources(global_parameters,phif,a0_minus_x,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
		Matrix2d dSSda0x = (1./(2.*epsilon))*(SS_plus-SS_minus);
		Vector2d dQ_rhoda0x = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
		Vector2d dQ_cda0x = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
		double dS_rhoda0x = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
		double dS_cda0x = (1./(2.*epsilon))*(S_c_plus-S_c_minus);		
		//
		// a0y
		evalFluxesSources(global_parameters,phif,a0_plus_y,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
		evalFluxesSources(global_parameters,phif,a0_minus_y,kappa,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
		Matrix2d dSSda0y = (1./(2.*epsilon))*(SS_plus-SS_minus);
		Vector2d dQ_rhoda0y = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
		Vector2d dQ_cda0y = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
		double dS_rhoda0y = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
		double dS_cda0y = (1./(2.*epsilon))*(S_c_plus-S_c_minus);		
		//
		// kappa
		evalFluxesSources(global_parameters,phif,a0,kappa_plus,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
		evalFluxesSources(global_parameters,phif,a0,kappa_minus,lamdaP,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
		Matrix2d dSSdkappa = (1./(2.*epsilon))*(SS_plus-SS_minus);
		Vector2d dQ_rhodkappa = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
		Vector2d dQ_cdkappa = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
		double dS_rhodkappa = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
		double dS_cdkappa = (1./(2.*epsilon))*(S_c_plus-S_c_minus);		
		//
		// lamdaP_a
		evalFluxesSources(global_parameters,phif,a0,kappa,lamdaP_plus_a,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
		evalFluxesSources(global_parameters,phif,a0,kappa,lamdaP_minus_a,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
		Matrix2d dSSdlamdaPa = (1./(2.*epsilon))*(SS_plus-SS_minus);
		Vector2d dQ_rhodlamdaPa = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
		Vector2d dQ_cdlamdaPa = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
		double dS_rhodlamdaPa = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
		double dS_cdlamdaPa = (1./(2.*epsilon))*(S_c_plus-S_c_minus);		
		//
		// lamdaP_s
		evalFluxesSources(global_parameters,phif,a0,kappa,lamdaP_plus_s,FF,rho,c,Grad_rho,Grad_c, SS_plus,Q_rho_plus,S_rho_plus,Q_c_plus,S_c_plus);
		evalFluxesSources(global_parameters,phif,a0,kappa,lamdaP_minus_s,FF,rho,c,Grad_rho,Grad_c, SS_minus,Q_rho_minus,S_rho_minus,Q_c_minus,S_c_minus);
		Matrix2d dSSdlamdaPs = (1./(2.*epsilon))*(SS_plus-SS_minus);
		Vector2d dQ_rhodlamdaPs = (1./(2.*epsilon))*(Q_rho_plus-Q_rho_minus);
		Vector2d dQ_cdlamdaPs = (1./(2.*epsilon))*(Q_c_plus-Q_c_minus);
		double dS_rhodlamdaPs = (1./(2.*epsilon))*(S_rho_plus-S_rho_minus);
		double dS_cdlamdaPs = (1./(2.*epsilon))*(S_c_plus-S_c_minus);		
		//
		//---------------------------------//
		


		//---------------------------------//
		// MECHANICS TANGENT
		//
		double Psif11 = 2*k2*kappa*kappa*Psif+2*k2*kappa*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif1 ;
		double Psif14 = 2*k2*kappa*(1-2*kappa)*I4e*Psif + 2*k2*kappa*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif4;
		double Psif41 = 2*k2*(1-2*kappa)*kappa*Psif + 2*k2*(1-2*kappa)*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif1;
		double Psif44 = 2*k2*(1-2*kappa)*(1-2*kappa)*Psif + 2*k2*(1-2*kappa)*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif4;
		std::vector<double> dSSpasdCC_explicit(16,0.);				
		int ii,jj,kk,ll,pp,rr,ss,tt;
		for(ii=0;ii<2;ii++){
			for(jj=0;jj<2;jj++){
				for(kk=0;kk<2;kk++){
					for(ll=0;ll<2;ll++){
						for(pp=0;pp<2;pp++){
							for(rr=0;rr<2;rr++){
								for(ss=0;ss<2;ss++){
									for(tt=0;tt<2;tt++){
										dSSpasdCC_explicit[ii*8+jj*4+kk*2+ll] += theta*theta/2.*FFginv(ii,pp)*(FFginv(ss,kk)*FFginv(ll,tt)+FFginv(ss,ll)*FFginv(kk,tt))*FFginv(rr,jj)*(
																phif*(Psif11*Identity(pp,rr)*Identity(ss,tt) + Psif14*Identity(pp,rr)*a0a0(ss,tt)
															  +Psif41*a0a0(pp,rr)*Identity(ss,tt) + Psif44*a0a0(pp,rr)*a0a0(ss,tt)));

//                                        dSSpasdCC_explicit[ii*8+jj*4+kk*2+ll] += theta*theta/2.*FFginv(ii,pp)*(FFginv(ss,kk)*FFginv(ll,tt)+FFginv(ss,ll)*FFginv(kk,tt))*FFginv(rr,jj)*(
//                                                (Psif11*Identity(pp,rr)*Identity(ss,tt) + Psif14*Identity(pp,rr)*a0a0(ss,tt)
//                                                      +Psif41*a0a0(pp,rr)*Identity(ss,tt) + Psif44*a0a0(pp,rr)*a0a0(ss,tt)));
									}
								}
							}
						}
					}
				}	
			}
		}		
		//--------------------------------------------------//		
		// build DD, the voigt version of CCCC = dSS_dCC
		// some things needed first
		// only derivatives with respect to CC explicitly
		Matrix2d dthetadCC = 0.5*theta*CCinv;	
		Matrix2d dpresdCC = 2*k0*lamda_N*thetaP/(theta*theta)*dthetadCC;
		Matrix2d dtrAdCC = kappa*Identity + (1-2*kappa)*a0a0;
		Matrix3d DDpres,DDpas,DDact,DDstruct,DDtot;
		
		//--------------------------------------------------//		
		// CHECKING
		//Matrix2d CC_p,CC_m,SSpas_p,SSpas_m,SSpres_p,SSpres_m,SSact_p,SSact_m;
		//Matrix3d DDpres_num,DDpas_num,DDact_num;
		//--------------------------------------------------//		
		
		for(int II=0;II<3;II++){
			for(int JJ=0;JJ<3;JJ++){
				ii = voigt_table_I_i(II);
				jj = voigt_table_I_j(II);
				kk = voigt_table_J_k(JJ);
				ll = voigt_table_J_l(JJ);

				// pressure, explicit  only
				DDpres(II,JJ) = pressure*thetaP*(-0.5*(CCinv(ii,kk)*CCinv(jj,ll)+CCinv(ii,ll)*CCinv(jj,kk)))
								+thetaP*CCinv(ii,jj)*dpresdCC(kk,ll);
				
				// passive, explicit only
				DDpas(II,JJ) = dSSpasdCC_explicit[ii*8+jj*4+kk*2+ll];
				
				// active , explicit only
				DDact(II,JJ) = -1.0*phif*traction_act*thetaP/(trA*trA*(K_t*K_t+phif*phif))*dtrAdCC(kk,ll)*A0(ii,jj);
				
				// structural
				DDstruct(II,JJ) = dSSdphif(ii,jj)*dphifdCC(kk,ll) + dSSda0x(ii,jj)*da0xdCC(kk,ll)+ dSSda0y(ii,jj)*da0ydCC(kk,ll)
								+dSSdkappa(ii,jj)*dkappadCC(kk,ll)+dSSdlamdaPa(ii,jj)*dlamdaP_adCC(kk,ll)+dSSdlamdaPs(ii,jj)*dlamdaP_sdCC(kk,ll);
				
				// TOTAL. now include the structural parameters
				DDtot(II,JJ) = DDpres(II,JJ)+DDpas(II,JJ)+DDact(II,JJ)+DDstruct(II,JJ);
                //DDtot(II,JJ) = DDpres(II,JJ)+ phif*(DDpas(II,JJ)) + DDact(II,JJ) + DDstruct(II,JJ);
				
				//--------------------------------------------------//		
				// CHECKING
				// the pressure, explicit only, means
				//CC_p = CC + 0.5*epsilon*Ebasis[kk]*Ebasis[ll].transpose()+ 0.5*epsilon*Ebasis[ll]*Ebasis[kk].transpose();
				//CC_m = CC - 0.5*epsilon*Ebasis[kk]*Ebasis[ll].transpose()- 0.5*epsilon*Ebasis[ll]*Ebasis[kk].transpose();
				//evalSS(global_parameters,phif,a0,kappa,lamdaP_a,lamdaP_s,CC_p,rho,c,SSpas_p,SSact_p,SSpres_p);
				//evalSS(global_parameters,phif,a0,kappa,lamdaP_a,lamdaP_s,CC_m,rho,c,SSpas_m,SSact_m,SSpres_m);
				//DDpas_num(II,JJ) = (1./(2.0*epsilon))*(SSpas_p(ii,jj)-SSpas_m(ii,jj));
				//DDpres_num(II,JJ) = (1./(2.0*epsilon))*(SSpres_p(ii,jj)-SSpres_m(ii,jj));
				//DDact_num(II,JJ) = (1./(2.0*epsilon))*(SSact_p(ii,jj)-SSact_m(ii,jj));
				//--------------------------------------------------//		
				
			}
		}
		
		//--------------------------------------------------//		
		// CHECKING
		//std::cout<<"\n"<<DDact<<"\n"<<DDpas<<"\n";
		//std::cout<<"comparing\nDD_pas\n";
		//std::cout<<DDpas<<"\nDD_pas_num\n"<<DDpas_num<<"\n";
		//std::cout<<"comparing\nDD_pres\n";
		//std::cout<<DDpres<<"\nDD_pres_num\n"<<DDpres_num<<"\n";
		//std::cout<<"comparing\nDD_act\n";
		//std::cout<<DDact<<"\nDD_act_num\n"<<DDact_num<<"\n";
		//
		//--------------------------------------------------//		
		
		// Derivatives for rho and c
		//
		double dtractiondrho = (t_rho + t_rho_c*c/(K_t_c + c));
		// Matrix2d dSSdrho_explicit = (thetaP*dtractiondrho*phif/trA)*(kappa*Identity+(1-2*kappa)*a0a0);
		Matrix2d dSSdrho_explicit = (thetaP*dtractiondrho*phif/(trA*(K_t*K_t+phif*phif)))*(kappa*Identity+(1-2*kappa)*a0a0);
		Matrix2d dSSdrho = dSSdrho_explicit + dSSdphif*dphifdrho + dSSda0x*da0xdrho + dSSda0y*da0ydrho
							+dSSdkappa*dkappadrho + dSSdlamdaPa*dlamdaP_adrho + dSSdlamdaPs*dlamdaP_sdrho;
		Vector3d dSSdrho_voigt(dSSdrho(0,0),dSSdrho(1,1),dSSdrho(0,1));
		//Matrix2d dsigma_actdc = phif*(t_rho_c/(K_t_c + c)-t_rho_c*c/pow((K_t_c + c),2))*rho*hat_A;
		double dtractiondc = (t_rho_c/(K_t_c + c)-t_rho_c*c/pow((K_t_c + c),2))*rho;
		//Matrix2d dSSdc_explicit = (thetaP*dtractiondc*phif/trA)*(kappa*Identity+(1-2*kappa)*a0a0);
        Matrix2d dSSdc_explicit = (thetaP*dtractiondc*phif/(trA*(K_t*K_t+phif*phif)))*(kappa*Identity+(1-2*kappa)*a0a0);
		Matrix2d dSSdc = dSSdc_explicit + dSSdphif*dphifdc + dSSda0x*da0xdc + dSSda0y*da0ydc
							+dSSdkappa*dkappadc + dSSdlamdaPa*dlamdaP_adc + dSSdlamdaPs*dlamdaP_sdc;							
		Vector3d dSSdc_voigt(dSSdc(0,0),dSSdc(1,1),dSSdc(0,1));
		//
		// some other declared variables
		Matrix2d linFF,linCC,lindeltaCC;
		Vector3d linCC_voigt,lindeltaCC_voigt;
		//
		// Loop over nodes and coordinates twice and assemble the corresponding entry
		for(int nodei=0;nodei<n_nodes;nodei++){
			for(int coordi=0;coordi<2;coordi++){
				deltaFF = Ebasis[coordi]*Grad_R[nodei].transpose();
				deltaCC = deltaFF.transpose()*FF + FF.transpose()*deltaFF;
				deltaCC_voigt = Vector3d(deltaCC(0,0),deltaCC(1,1),2.*deltaCC(1,0));
				for(int nodej=0;nodej<n_nodes;nodej++){
					for(int coordj=0;coordj<2;coordj++){
					
						//-----------//
						// Ke_X_X
						//-----------//
						
						// material part of the tangent
						linFF =  Ebasis[coordj]*Grad_R[nodej].transpose();
						linCC = linFF.transpose()*FF + FF.transpose()*linFF;
						linCC_voigt = Vector3d(linCC(0,0),linCC(1,1),2.*linCC(1,0));
						//
						Ke_x_x(nodei*2+coordi,nodej*2+coordj) += Jac*deltaCC_voigt.dot(DDtot*linCC_voigt);
						//
						// geometric part of the tangent
						lindeltaCC = deltaFF.transpose()*linFF + linFF.transpose()*deltaFF;
						lindeltaCC_voigt = Vector3d(lindeltaCC(0,0),lindeltaCC(1,1),2.*lindeltaCC(0,1));
						//
						Ke_x_x(nodei*2+coordi,nodej*2+coordj) += Jac*SS_voigt.dot(lindeltaCC_voigt);
						
					}
					
					//-----------//
					// Ke_x_rho
					//-----------//
					 
					Ke_x_rho(nodei*2+coordi,nodej) += Jac*dSSdrho_voigt.dot(deltaCC_voigt)*R[nodej];
					
					//-----------//
					// Ke_x_c
					//-----------//
					
					Ke_x_c(nodei*2+coordi,nodej) += Jac*dSSdc_voigt.dot(deltaCC_voigt)*R[nodej];
					
				}
			}
		}		



		//-----------------//
		// RHO and C
		//-----------------//
		
		// Derivatives of flux and source terms wrt CC, rho, and C
        std::vector<double> dQ_rhodCC_explicit(8,0.);
        std::vector<double> dQ_cdCC_explicit(8,0.);
        for(int ii=0;ii<2;ii++) {
            for(int jj=0;jj<2;jj++) {
                for(int kk=0;kk<2;kk++) {
                    for(int ll=0;ll<2;ll++) {
                        // These are third order tensors, but there are two contractions from matrix multiplication
//                        dQ_rhodCC_explicit[ii*4+kk*2+ll] += -1.0*(-3*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0(ii,jj)*Grad_rho(jj)
//                                                                  - 3*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0(ii,jj)*Grad_c(jj))*dtrAdCC(kk,ll) / (trA*trA);
//
//                        dQ_cdCC_explicit[ii*4+kk*2+ll] += -1.0*(-3*(D_cc-phif*(D_cc-D_cc/10))*A0(ii,jj)*Grad_c(jj))
//                                                          *dtrAdCC(kk,ll)/(trA*trA);

                        dQ_rhodCC_explicit[ii*4+kk*2+ll] += -1.0*(-0.5)*D_rhorho*(CCinv(ii,kk)*CCinv(jj,ll)+CCinv(jj,kk)*CCinv(ii,ll))*Grad_rho(jj)
                                                            -1.0*(-0.5)*D_cc*rho*(CCinv(ii,kk)*CCinv(jj,ll)+CCinv(jj,kk)*CCinv(ii,ll))*Grad_c(jj);

                        dQ_cdCC_explicit[ii*4+kk*2+ll] += -1.0*(-0.5)*D_cc*(CCinv(ii,kk)*CCinv(jj,ll)+CCinv(jj,kk)*CCinv(ii,ll))*Grad_c(jj);
                    }
                }
            }
        }
        // Put into a Voigt form (2x3)
        MatrixXd dQ_rhodCC_voigt(2,3); dQ_rhodCC_voigt.setZero();
        MatrixXd dQ_cdCC_voigt(2,3); dQ_cdCC_voigt.setZero();
        for(int II=0;II<2;II++){
            for(int JJ=0;JJ<3;JJ++) {
                // We can use the same Voigt tables, but only need three of them,
                // since we have contracted on jj, we will only take the first three entries of II
                int ii = voigt_table_I_i(II);
                int kk = voigt_table_J_k(JJ);
                int ll = voigt_table_J_l(JJ);

                dQ_rhodCC_voigt(II,JJ) = dQ_rhodCC_explicit[ii*4+kk*2+ll] + dQ_rhodphif(ii)*dphifdCC(kk,ll)
                                         + dQ_rhoda0x(ii)*da0xdCC(kk,ll) + dQ_rhoda0y(ii)*da0ydCC(kk,ll)
                                         +dQ_rhodkappa(ii)*dkappadCC(kk,ll) + dQ_rhodlamdaPa(ii)*dlamdaP_adCC(kk,ll)
                                         + dQ_rhodlamdaPs(ii)*dlamdaP_sdCC(kk,ll);

                dQ_cdCC_voigt(II,JJ) = dQ_cdCC_explicit[ii*4+kk*2+ll] + dQ_cdphif(ii)*dphifdCC(kk,ll)
                                       + dQ_cda0x(ii)*da0xdCC(kk,ll) + dQ_cda0y(ii)*da0ydCC(kk,ll)
                                       +dQ_cdkappa(ii)*dkappadCC(kk,ll) + dQ_cdlamdaPa(ii)*dlamdaP_adCC(kk,ll)
                                       + dQ_cdlamdaPs(ii)*dlamdaP_sdCC(kk,ll);
            }
        }
        //std::cout<<"\ndQ_rhodCC_voigt\n"<<dQ_rhodCC_voigt<<"\ndQ_cdCC_voigt\n"<<dQ_cdCC_voigt<<"\n";
		// explicit linearizations. In this case no dependence on structural parameters.
        //Matrix2d linQ_rhodGradrho = -3*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0/trA;
        //Vector2d linQ_rhodrho = -3*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*A0*Grad_c/trA;
        //Matrix2d linQ_rhodGradc = -3*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0/trA;
        //Matrix2d linQ_cdGradc = -3*(D_cc-phif*(D_cc-D_cc/10))*A0/trA;
        Matrix2d linQ_rhodGradrho = -D_rhorho*CCinv;
        Vector2d linQ_rhodrho = -D_rhoc*CCinv*Grad_c;
        Matrix2d linQ_rhodGradc = -D_rhoc*rho*CCinv;
        Matrix2d linQ_cdGradc = -D_cc*CCinv;
		//
		// explicit derivatives of source terms
		double dS_rhodrho_explicit = (p_rho + p_rho_c*c/(K_rho_c+c)+p_rho_theta*He)*(1-rho/K_rho_rho) - d_rho + rho*(p_rho + p_rho_c*c/(K_rho_c+c)+p_rho_theta*He)*(-1./K_rho_rho);
		double dS_rhodc_explicit = (1-rho/K_rho_rho)*rho*(p_rho_c/(K_rho_c+c) - p_rho_c*c/((K_rho_c+c)*(K_rho_c+c)));
		double dS_cdrho_explicit = (p_c_rho*c + p_c_thetaE*He)*(1./(K_c_c+c));
		double dS_cdc_explicit = -d_c + (p_c_rho*c + p_c_thetaE*He)*(-rho/((K_c_c+c)*(K_c_c+c))) + (rho/(K_c_c+c))*p_c_rho;
		// total derivatives
		double dS_rhodrho = dS_rhodrho_explicit + dS_rhodphif*dphifdrho + dS_rhoda0x*da0xdrho + dS_rhoda0y*da0ydrho
							+dS_rhodkappa*dkappadrho+dS_rhodlamdaPa*dlamdaP_adrho+dS_rhodlamdaPs*dlamdaP_sdrho;
		double dS_rhodc = dS_rhodc_explicit + dS_rhodphif*dphifdc + dS_rhoda0x*da0xdc + dS_rhoda0y*da0ydc
							+dS_rhodkappa*dkappadc+dS_rhodlamdaPa*dlamdaP_adc+dS_rhodlamdaPs*dlamdaP_sdc;
		double dS_cdrho = dS_cdrho_explicit + dS_cdphif*dphifdrho + dS_cda0x*da0xdrho + dS_cda0y*da0ydrho
							+dS_cdkappa*dkappadrho+dS_cdlamdaPa*dlamdaP_adrho+dS_cdlamdaPs*dlamdaP_sdrho;							
		double dS_cdc = dS_cdc_explicit + dS_cdphif*dphifdc + dS_cda0x*da0xdc + dS_cda0y*da0ydc
							+dS_cdkappa*dkappadc+dS_cdlamdaPa*dlamdaP_adc+dS_cdlamdaPs*dlamdaP_sdc;
		// wrt Mechanics
		Matrix2d dHedCC_explicit = -1./pow((1.+exp(-gamma_theta*(thetaE - vartheta_e))),2)*(exp(-gamma_theta*(thetaE - vartheta_e)))*(-gamma_theta)*(1./thetaP)*(dthetadCC);
		Matrix2d dS_rhodCC_explicit = (1-rho/K_rho_rho)*rho*p_rho_theta*dHedCC_explicit;
		Matrix2d dS_cdCC_explicit = (rho/(K_c_c+c))*(p_c_thetaE*dHedCC_explicit);
		Matrix2d dS_rhodCC = dS_rhodCC_explicit + dS_rhodphif*dphifdCC + dS_rhoda0x*da0xdCC + dS_rhoda0y*da0ydCC
							+dS_rhodkappa*dkappadCC+dS_rhodlamdaPa*dlamdaP_adCC+dS_rhodlamdaPs*dlamdaP_sdCC;
		Vector3d dS_rhodCC_voigt(dS_rhodCC(0,0),dS_rhodCC(1,1),dS_rhodCC(0,1));
		Matrix2d dS_cdCC = dS_cdCC_explicit + dS_cdphif*dphifdCC + dS_cda0x*da0xdCC + dS_cda0y*da0ydCC
							+dS_cdkappa*dkappadCC+dS_cdlamdaPa*dlamdaP_adCC+dS_cdlamdaPs*dlamdaP_sdCC;
		Vector3d dS_cdCC_voigt(dS_cdCC(0,0),dS_cdCC(1,1),dS_cdCC(0,1));
		//		
		for(int nodei=0;nodei<n_nodes;nodei++){
			for(int nodej=0;nodej<n_nodes;nodej++){
				for(int coordj=0;coordj<2;coordj++){

					linFF =  Ebasis[coordj]*Grad_R[nodej].transpose();
					linCC = linFF.transpose()*FF + FF.transpose()*linFF;
					linCC_voigt = Vector3d(linCC(0,0),linCC(1,1),2.*linCC(0,1));
									
					//-----------//
					// Ke_rho_X
					//-----------//

					Ke_rho_x(nodei,nodej*2+coordj)+= -(R[nodei]*dS_rhodCC_voigt.dot(linCC_voigt) + Grad_R[nodei].dot(dQ_rhodCC_voigt*linCC_voigt))*Jac;
					
					//-----------//
					// Ke_c_X
					//-----------//

					Ke_c_x(nodei,nodej*2+coordj)+= -(R[nodei]*dS_cdCC_voigt.dot(linCC_voigt) + Grad_R[nodei].dot(dQ_cdCC_voigt*linCC_voigt))*Jac;
					
				}
				
				//-----------//
				// Ke_rho_rho
				//-----------//
				
				Ke_rho_rho(nodei,nodej) += Jac*(R[nodei]*R[nodej]/dt - 1.* R[nodei]*dS_rhodrho*R[nodej] - 1.* Grad_R[nodei].dot(linQ_rhodGradrho*Grad_R[nodej] + linQ_rhodrho*R[nodej]));
				
				//-----------//
				// Ke_rho_c
				//-----------//
				
				Ke_rho_c(nodei,nodej) += Jac*(-1.*R[nodei]*dS_rhodc*R[nodej] - 1.* Grad_R[nodei].dot(linQ_rhodGradc*Grad_R[nodej]));
				
				//-----------//
				// Ke_c_rho
				//-----------//
		
				Ke_c_rho(nodei,nodej) += Jac*(-1.*R[nodei]*dS_cdrho*R[nodej]);
				
				//-----------//
				// Ke_c_c
				//-----------//
				
				Ke_c_c(nodei,nodej) += Jac*(R[nodei]*R[nodej]/dt -1.* R[nodei]*dS_cdc*R[nodej] - 1.*Grad_R[nodei].dot(linQ_cdGradc*Grad_R[nodej]));
			}
		}
		
	} // END INTEGRATION loop	
}


//========================================================//
// EVAL SOURCE AND FLUX 
//========================================================//

// Sources and Fluxes are :stress, biological fluxes and sources

void evalFluxesSources(const std::vector<double> &global_parameters, double phif,Vector2d a0,double kappa,Vector2d lamdaP,
Matrix2d FF,double rho, double c, Vector2d Grad_rho, Vector2d Grad_c,
Matrix2d & SS,Vector2d &Q_rho,double &S_rho, Vector2d &Q_c,double &S_c)
{
    double k0 = global_parameters[0]; // neo hookean
    double kf = global_parameters[1]; // stiffness of collagen
    double k2 = global_parameters[2]; // nonlinear exponential
    double t_rho = global_parameters[3]; // force of fibroblasts
    double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
    double K_t = global_parameters[5]; // saturation of collagen on force
    double K_t_c = global_parameters[6]; // saturation of chemical on force
    double D_rhorho = global_parameters[7]; // diffusion of cells
    double D_rhoc = global_parameters[8]; // diffusion of chemotactic gradient
    double D_cc = global_parameters[9]; // diffusion of chemical
    double p_rho =global_parameters[10]; // production of fibroblasts naturally
    double p_rho_c = global_parameters[11]; // production enhanced by the chem
    double p_rho_theta = global_parameters[12]; // mechanosensing
    double K_rho_c= global_parameters[13]; // saturation of cell production by chemical
    double K_rho_rho = global_parameters[14]; // saturation of cell by cell
    double d_rho = global_parameters[15] ;// decay of cells
    double vartheta_e = global_parameters[16]; // physiological state of area stretch
    double gamma_theta = global_parameters[17]; // sensitivity of heviside function
    double p_c_rho = global_parameters[18];// production of C by cells
    double p_c_thetaE = global_parameters[19]; // coupling of elastic and chemical
    double K_c_c = global_parameters[20];// saturation of chem by chem
    double d_c = global_parameters[21]; // decay of chemical

	Matrix2d CC = FF.transpose()*FF;
	Matrix2d CCinv = CC.inverse();
	Matrix2d Identity;Identity<<1.,0.,0.,1.;
	// Update kinematics. 
	// fiber tensor in the reference
	Matrix2d a0a0 = a0*a0.transpose();
	Matrix2d Rot90;Rot90<<0,-1,1,0;
	Vector2d s0 = Rot90*a0;
	Matrix2d s0s0 = s0*s0.transpose();
	Matrix2d A0 = kappa*Identity + (1-2.*kappa)*a0a0;
	Vector2d a = FF*a0;
	Matrix2d A = kappa*FF*FF.transpose() + (1.-2.0*kappa)*a*a.transpose();
	double trA = A(0,0) + A(1,1);
	Matrix2d hat_A = A/trA;
	// recompute the split
	double lamdaP_a = lamdaP(0);
	double lamdaP_s = lamdaP(1);
	Matrix2d FFg = lamdaP_a*(a0a0) + lamdaP_s*(s0s0);
	double thetaP = lamdaP_a*lamdaP_s;
	Matrix2d FFginv = (1./lamdaP_a)*(a0a0) + (1./lamdaP_s)*(s0s0);
	Matrix2d FFe = FF*FFginv;
	// elastic strain
	Matrix2d CCe = FFe.transpose()*FFe;
	// invariant of the elastic strain
	double I1e = CCe(0,0) + CCe(1,1);
	double I4e = a0.dot(CCe*a0);
	// calculate the normal stretch
	double thetaE = sqrt(CCe.determinant());
	double theta = thetaE*thetaP;
	double lamda_N = 1./thetaE;
	// Second Piola Kirchhoff stress tensor
	// passive elastic
	double Psif = (kf/(2.*k2))*(exp( k2*pow((kappa*I1e + (1-2*kappa)*I4e -1),2))-1);
	double Psif1 = 2*k2*kappa*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif;
	double Psif4 = 2*k2*(1-2*kappa)*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif;
	//Matrix2d SSe_pas = k0*Identity + phif*(Psif1*Identity + Psif4*a0a0);
    Matrix2d SSe_pas = k0*Identity + phif*(Psif1*Identity + Psif4*a0a0);
	// pull back to the reference
	Matrix2d SS_pas = thetaP*FFginv*SSe_pas*FFginv;
	// magnitude from systems bio
	double traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho;
	//Matrix2d SS_act = (thetaP*traction_act*phif/trA)*(kappa*Identity+(1-2*kappa)*a0a0);
	Matrix2d SS_act = (thetaP*traction_act*phif/(trA*(K_t*K_t+phif*phif)))*A0;
	// total stress, don't forget the pressure
	double pressure = -k0*lamda_N*lamda_N;
	Matrix2d SS_pres = pressure*thetaP*CCinv;
	//SS = SS_pas + SS_act + SS_pres;
    SS = (SS_pas) + SS_pres + SS_act;
	// Flux and Source terms for the rho and the C
	Q_rho = -D_rhorho*CCinv*Grad_rho - D_rhoc*rho*CCinv*Grad_c;
	Q_c = -D_cc*CCinv*Grad_c;
    //Q_rho = -3*(D_rhorho-phif*(D_rhorho-D_rhorho/10))*A0*Grad_rho/trA - 3*(D_rhoc-phif*(D_rhoc-D_rhoc/10))*rho*A0*Grad_c/trA;
    //Q_c = -3*(D_cc-phif*(D_cc-D_cc/10))*A0*Grad_c/trA;
	double He = 1./(1.+exp(-gamma_theta*(thetaE - vartheta_e)));
	S_rho = (p_rho + p_rho_c*c/(K_rho_c+c)+p_rho_theta*He)*(1-rho/K_rho_rho)*rho - d_rho*rho;
	S_c = (p_c_rho*c+ p_c_thetaE*He)*(rho/(K_c_c+c)) - d_c*c;
}




//========================================================//
// LOCAL PROBLEM: structural update
//========================================================//

void localWoundProblem(
double dt, const std::vector<double> &local_parameters,
double c,double rho,const Matrix2d &CC,
double phif_0, const Vector2d &a0_0, double kappa_0, const Vector2d &lamdaP_0,
double &phif, Vector2d &a0, double &kappa, Vector2d &lamdaP,
VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadc)
{



	//---------------------------------//
	// 
	// INPUT
	// 	matParam: material parameters
	//	rho: value of the cell density at the point
	//	c: concentration at the point
	//	CC: deformation at the point
	//	Theta_t: value of the parameters at previous time step
	//
	// OUTPUT
	//	Theta: value of the parameters at the current time step
	//	dThetadCC: derivative of Theta wrt global mechanics (CC)
	// 	dThetadrho: derivative of Theta wrt global rho
	// 	dThetadc: derivative of Theta wrt global C
	//
	//---------------------------------//


	//---------------------------------//
	// Parameters
	//
	// collagen fraction
	double p_phi = local_parameters[0]; // production by fibroblasts, natural rate
	double p_phi_c = local_parameters[1]; // production up-regulation, weighted by C and rho
	double p_phi_theta = local_parameters[2]; //production regulated by stretch
	double K_phi_c = local_parameters[3]; // saturation of C effect on deposition
	double K_phi_rho = local_parameters[4]; // saturation of collagen fraction itself
	double d_phi = local_parameters[5]; // rate of degradation
	double d_phi_rho_c = local_parameters[6]; // rate of degradation
	//
	// fiber alignment
	double tau_omega = local_parameters[7]; // time constant for angular reorientation
	//
	// dispersion parameter
	double tau_kappa = local_parameters[8]; // time constant
	double gamma_kappa = local_parameters[9]; // exponent of the principal stretch ratio
	// 
	// permanent contracture/growth
	double tau_lamdaP_a = local_parameters[10]; // time constant for direction a
	double tau_lamdaP_s = local_parameters[11]; // time constant for direction s
	//
	double gamma_theta = local_parameters[12]; // exponent of the Heaviside function
	double vartheta_e = local_parameters[13]; // mechanosensing response
	//
	// solution parameters
	double tol_local = local_parameters[14]; // local tolerance
	double max_local_iter = local_parameters[15]; // max local iter
	//
	// other local stuff
	Matrix2d Identity;Identity<<1.,0.,0.,1.;
	// Might as well just pass this guy directly
	//Matrix2d CC = FF.transpose()*FF; // right Cauchy-Green deformation tensor
	double theta = sqrt(CC.determinant());
	double theta_e;
	double H_theta;
	Matrix2d CCinv = CC.inverse();
	double PIE = 3.14159;
	//
	//---------------------------------//



	//---------------------------------//
	// Preprocess the Newton
	//
	// initial guess for local newton
	phif = phif_0;
	// make sure it is unit length
	a0 = a0_0/(sqrt(a0_0.dot(a0_0)));
	kappa = kappa_0;
	lamdaP = lamdaP_0;
	//
	// initialize the residual and iterations
	int iter = 0;
	double residuum0 = 1.;
	double residuum = 1.;	
	//
	// Declare Variables
	//
	// some global
	std::vector<Vector2d> Ebasis; Ebasis.clear();
	Matrix2d Rot90;Rot90<<0.,-1.,1.,0.;
	Matrix2d Rot180;Rot180<<-1.,0.,0.,-1.;
	Ebasis.push_back(Vector2d(1,0)); Ebasis.push_back(Vector2d(0,1));
	//
	// residual
	VectorXd RR_local(6);
	double R_phif,R_kappa; Vector2d R_lamdaP,R_a0;
	double phif_dot_plus,phif_dot;
	Vector2d s0;
	double Ce_aa,Ce_ss,Ce_as,lamda0, lamda1, sinVartheta,omega;
	Matrix2d Romega; Vector2d Rot_a0_0;
	double kappa_dot;
	double lamdaE_a,lamdaE_s;
	Vector2d lamdaP_dot;
	//
	// tangent
	MatrixXd KK_local(6,6);
	double dtheta_edlamdaP_a,dtheta_edlamdaP_s;
	double dH_thetadlamdaP_a,dH_thetadlamdaP_s;
	double dRphifdphif,dRphifdlamdaP_a,dRphifdlamdaP_s;
	Vector2d dphifplusdlamdaP;
	Vector2d dRa0dphif; Matrix2d dRa0da0, dRa0dlamdaP;
	double dRkappadphif,dRkappadkappa;
	Vector2d dRkappada0, dRkappadlamdaP,dRlamdaPdphif;
	Vector2d dRlamdaPada0, dRlamdaPsda0;
	double dRlamdaPadlamdaPa,dRlamdaPadlamdaPs,dRlamdaPsdlamdaPa,dRlamdaPsdlamdaPs;
	double dphifplusdphif;
	Matrix2d dRomegadomega ;
	double domegadphif;
	Vector2d dCe_aada0,dCe_ssda0,dCe_asda0;
	double dCe_aada0x,dCe_aada0y,dCe_ssda0x,dCe_ssda0y,dCe_asda0x,dCe_asda0y;
	Vector2d dlamda1da0, dlamda0da0,dsinVarthetada0,domegada0;
	double aux00,aux01;
	double dlamda1dCe_aa,dlamda1dCe_ss,dlamda1dCe_as;
	double dlamda0dCe_aa,dlamda0dCe_ss,dlamda0dCe_as;
	double dsinVarthetadlamda1,dsinVarthetadCe_aa,dsinVarthetadCe_ss,dsinVarthetadCe_as;
	Vector2d dCe_aadlamdaP,dCe_ssdlamdaP,dCe_asdlamdaP;
	Vector2d dlamda1dlamdaP,dlamda0dlamdaP;
	Vector2d dsinVarthetadlamdaP, domegadlamdaP;
	//
	VectorXd SOL_local(6);
	//
	//---------------------------------//
	
	
	
	//---------------------------------//
	// NEWTON LOOP
	//---------------------------------//
	while(residuum>tol_local && iter<max_local_iter){
		//std::cout<<"iter : "<<iter<<"\n";
		
		//----------//
		// RESIDUALS
		//----------//
		
		RR_local.setZero();
		
		// collagen fraction residual
		// heaviside functions
		// theta_e = theta/theta_p
		theta = sqrt(CC.determinant());
		theta_e = theta/(lamdaP(0)*lamdaP(1));
		H_theta = 1./(1+exp(-gamma_theta*(theta_e-vartheta_e)));
		if(H_theta<0.002){H_theta=0;}
		//std::cout<<"H_theta: "<<H_theta<<"\n";
		phif_dot_plus = (p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*H_theta)*(rho/(K_phi_rho+phif));
		//std::cout<<"phidotplus: "<<phif_dot_plus<<"\n";
		phif_dot = phif_dot_plus - (d_phi + c*rho*d_phi_rho_c)*phif;
		R_phif = (-phif + phif_0)/dt + phif_dot;
		
		//std::cout<<"Collagen fraction residual.\nphif_0= "<<phif_0<<", phif = "<<phif<<"\n";
		//std::cout<<"phif_dot_plus = "<<phif_dot_plus<<", phif_dot = "<<phif_dot<<"\n";
		//std::cout<<"R_phif = "<<R_phif<<"\n";
		
		// fiber orientation residual
		// given the current guess of the direction make the choice of s
		s0(0)=-a0(1); s0(1) = a0(0);
		// compute the principal eigenvalues and eigenvectors of Ce as a function of 
		// the structural variables, namely lamdaP and a0
		Ce_aa = (1./(lamdaP(0)*lamdaP(0)))*(CC(0,0)*a0(0)*a0(0)+2*CC(0,1)*a0(0)*a0(1)+CC(1,1)*a0(1)*a0(1));
		Ce_as = (1./(lamdaP(0)*lamdaP(1)))*(CC(0,0)*a0(0)*s0(0)+CC(0,1)*a0(0)*s0(1)+CC(1,0)*s0(0)*a0(1)+CC(1,1)*a0(1)*s0(1));
		Ce_ss = (1./(lamdaP(1)*lamdaP(1)))*(CC(0,0)*s0(0)*s0(0)+2*CC(0,1)*s0(0)*s0(1)+CC(1,1)*s0(1)*s0(1));
		lamda1 = ((Ce_aa + Ce_ss) + sqrt( (Ce_aa-Ce_ss)*(Ce_aa-Ce_ss) + 4*Ce_as*Ce_as))/2.; // the eigenvalue is a squared number by notation
		lamda0 = ((Ce_aa + Ce_ss) - sqrt( (Ce_aa-Ce_ss)*(Ce_aa-Ce_ss) + 4*Ce_as*Ce_as))/2.; // the eigenvalue is a squared number by notation
		if(fabs(lamda1-lamda0)<1e-7 || fabs(lamda1-Ce_aa)<1e-7){
			// equal eigenvalues means multiple of identity -> you can't possibly reorient. 
			// or, eigenvector in the direction of a0 already -> no need to reorient since you are already there
			sinVartheta = 0.;
		}else{
			// if eigenvalues are not the same and the principal eigenvalue is not already in the direction of a0
			sinVartheta = (lamda1-Ce_aa)/sqrt(Ce_as*Ce_as + (lamda1-Ce_aa)*(lamda1-Ce_aa));
		}

		// Alternative is to do in the original coordinates
		double lamdaP_a = lamdaP(0);
		double lamdaP_s = lamdaP(1);
		double a0x = a0(0);
		double a0y = a0(1);
		double s0x = s0(0);
		double s0y = s0(1);
		double C00 = CC(0,0);
		double C01 = CC(0,1);
		double C11 = CC(1,1);
		double Ce00 = (-a0x*a0y*(lamdaP_a - lamdaP_s)*(C01*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a) - C11*a0x*a0y*(lamdaP_a - lamdaP_s)) + (C00*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a) - C01*a0x*a0y*(lamdaP_a - lamdaP_s))*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
		double Ce01 = (a0x*a0y*(lamdaP_a - lamdaP_s)*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - (a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a)*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
		double Ce11 = (a0x*a0y*(lamdaP_a - lamdaP_s)*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - (a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
		double Tr = Ce00+Ce11;
		double Det = Ce00*Ce11 - Ce01*Ce01;
		//lamda1 = Tr/2. + sqrt(Tr*Tr/4. - Det);
		//lamda0 = Tr/2. - sqrt(Tr*Tr/4. - Det);
		/*
		if(fabs(lamda1-lamda0)<1e-7){
			// equal eigenvalues, no way to reorient
			sinVartheta = 0;
		}else if(fabs(Ce01)<1e-7 && fabs(Ce00-lamda1)>1e-7){
			sinVartheta = (-Ce01*a0y + a0x*(-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/2)/(Ce00-lamda1);
		}else{
			sinVartheta = (Ce01*a0x - a0y*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/2)/(sqrt((Ce01*Ce01)+(Ce11-lamda1)*(Ce11-lamda1)));
		} */
		//std::cout<<"iter = "<<iter<<", lamda1 = "<<lamda1<<", lamda0 = "<<lamda0<<", sinVartheta = "<<sinVartheta<<"\n";
		// Compute the angular velocity
		omega = ((2.*PIE*phif_dot_plus)/(tau_omega))*lamda1*sinVartheta; // lamda1 is already squared by notation
		//omega = -((2.*PIE*phif_dot_plus)/(tau_omega))*lamda1*sinVartheta*sinVartheta; // always move opposite to sinvartheta<pi/2
		// compute the rotation tensor
		Romega(0,0) = cos(omega*dt); Romega(0,1) = -sin(omega*dt); 
		Romega(1,0) = sin(omega*dt); Romega(1,1) = cos(omega*dt);
		// rotate the previous fiber
		Rot_a0_0 = Romega*a0_0;
		// residual
		if(fabs(omega)<1e-8){
			a0 = Rot_a0_0;
		}
		R_a0 = a0 - Rot_a0_0;
		
		//std::cout<<"Fiber direction residual.\na0_0 = ["<<a0_0(0)<<","<<a0_0(1)<<"], a0 = ["<<a0(0)<<","<<a0(1)<<"]\n";
		//std::cout<<"CCe (in a,s system) = ["<<Ce_aa<<","<<Ce_as<<","<<Ce_ss<<"]\n";
		//std::cout<<"sinVartheta = "<<sinVartheta<<", lamda1 = "<<lamda1<<", lamda0 = "<<lamda0<<"\n";
		//std::cout<<"Romega\n"<<Romega<<"\n";
		
		// dispersion residual
		kappa_dot = (phif_dot_plus/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/2.  -kappa);
		R_kappa = (-kappa+kappa_0)/dt + kappa_dot;
		
		// permanent deformation residual
		// elastic stretches of the directions a and s
		lamdaE_a = sqrt(Ce_aa);
		lamdaE_s = sqrt(Ce_ss);
		lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-1)/tau_lamdaP_a;
		lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-1)/tau_lamdaP_s;
		R_lamdaP= (1./dt)*(-lamdaP +lamdaP_0) + lamdaP_dot;
		
		// Assemble into the residual vector
		RR_local(0) = R_phif;
		RR_local(1) = R_a0(0);
		RR_local(2) = R_a0(1);
		RR_local(3) = R_kappa;
		RR_local(4) = R_lamdaP(0);
		RR_local(5) = R_lamdaP(1);
		
		//----------//
		// TANGENT
		//----------//
		
		KK_local.setZero();
		
		// Tangent of phif
		// derivative of the phifdotplus
		dphifplusdphif = (p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*H_theta)*(-rho/((K_phi_rho+phif)*(K_phi_rho+phif)));
		dRphifdphif = -1./dt + dphifplusdphif - d_phi;
		dtheta_edlamdaP_a = -theta/(lamdaP(0)*lamdaP(0)*lamdaP(1));
		dtheta_edlamdaP_s = -theta/(lamdaP(0)*lamdaP(1)*lamdaP(1));
		dH_thetadlamdaP_a = -1.0*H_theta*H_theta*exp(-gamma_theta*(theta_e-vartheta_e))*(-gamma_theta*(dtheta_edlamdaP_a));
		dH_thetadlamdaP_s= -1.0*H_theta*H_theta*exp(-gamma_theta*(theta_e-vartheta_e))*(-gamma_theta*(dtheta_edlamdaP_s));
		dRphifdlamdaP_a = p_phi_theta*dH_thetadlamdaP_a*(rho/(K_phi_rho+phif));
		dRphifdlamdaP_s = p_phi_theta*dH_thetadlamdaP_s*(rho/(K_phi_rho+phif));
		dphifplusdlamdaP =Vector2d(p_phi_theta*dH_thetadlamdaP_a*(rho/(K_phi_rho+phif)),p_phi_theta*dH_thetadlamdaP_s*(rho/(K_phi_rho+phif)));
		
		//std::cout<<"Collagen fraction tangent.\n";
		//std::cout<<"dphifplusdphif = "<<dphifplusdphif<<"\n";
		//std::cout<<"dRphifdphif = "<<dRphifdphif<<"\n";
		
		// Tangent of a0
		// derivative of the rotation matrix wrt omega
		dRomegadomega(0,0) = -sin(omega*dt)*dt; dRomegadomega(0,1) = -cos(omega*dt)*dt;
		dRomegadomega(1,0) = cos(omega*dt)*dt;  dRomegadomega(1,1)= -sin(omega*dt)*dt;
		// derivative of the omega angular velocity wrt phif
		domegadphif = (2.*PIE*dphifplusdphif)*lamda1*sinVartheta/(tau_omega);
		//domegadphif = -(2.*PIE*dphifplusdphif)*lamda1*sinVartheta*sinVartheta/(tau_omega);
		// chain rule for derivative of residual wrt phif
		dRa0dphif = (-dRomegadomega*a0_0)*domegadphif;

		// derivative of R_a0 wrt to a0 needs some pre-calculations
		// derivatives of Ce wrt a0
		dCe_aada0.setZero(); dCe_asda0.setZero(); dCe_ssda0.setZero();
		for(int alpha=0; alpha<2; alpha++){
			for(int beta=0; beta<2; beta++){
				dCe_aada0 += (CC(alpha,beta)/(lamdaP(0)*lamdaP(0)))*(a0(alpha)*Ebasis[beta] + a0(beta)*Ebasis[alpha]);
				dCe_asda0 += (CC(alpha,beta)/(lamdaP(0)*lamdaP(1)))*(s0(beta)*Ebasis[alpha]+ a0(alpha)*Rot90*Ebasis[beta]);
				dCe_ssda0 += (CC(alpha,beta)/(lamdaP(1)*lamdaP(1)))*(s0(alpha)*Rot90*Ebasis[beta] + s0(beta)*Rot90*Ebasis[alpha]);	
			}
		}
		// close form 
		dCe_aada0x = (1./(lamdaP(0)*lamdaP(0)))*(2*CC(0,0)*a0(0)+2*CC(0,1)*a0(1));
		dCe_aada0y = (1./(lamdaP(0)*lamdaP(0)))*(2*CC(0,1)*a0(0)+2*CC(1,1)*a0(1));
		dCe_asda0x = (1./(lamdaP(0)*lamdaP(1)))*(-1.*CC(0,0)*a0(1)+2*CC(0,1)*a0(0)+CC(1,1)*a0(1));
		dCe_asda0y = (1./(lamdaP(0)*lamdaP(1)))*(-1.*CC(0,0)*a0(0)-2*CC(0,1)*a0(1)+CC(1,1)*a0(0));
		dCe_ssda0x = (1./(lamdaP(1)*lamdaP(1)))*(-2.*CC(0,1)*a0(1)+2*CC(1,1)*a0(0));
		dCe_ssda0y = (1./(lamdaP(1)*lamdaP(1)))*(2*CC(0,0)*a0(1)-2*CC(0,1)*a0(0));
		dCe_aada0(0) = dCe_aada0x;
		dCe_aada0(1) = dCe_aada0y;
		dCe_asda0(0) = dCe_asda0x;
		dCe_asda0(1) = dCe_asda0y;
		dCe_ssda0(0) = dCe_ssda0x;
		dCe_ssda0(1) = dCe_ssda0y;
		// derivatives of the principal stretches wrt a0
		aux00 = sqrt( (Ce_aa-Ce_ss)*(Ce_aa-Ce_ss) + 4*Ce_as*Ce_as);
		if(aux00>1e-7){
			dlamda1da0 = (1./2.)*(dCe_aada0+dCe_ssda0) + (Ce_aa - Ce_ss)*(dCe_aada0 - dCe_ssda0)/aux00/2. + 2*Ce_as*dCe_asda0/aux00;
			dlamda0da0 = (1./2.)*(dCe_aada0+dCe_ssda0) - (Ce_aa - Ce_ss)*(dCe_aada0 - dCe_ssda0)/aux00/2. - 2*Ce_as*dCe_asda0/aux00;
			// Aternatively, do all the chain rule
			dlamda1dCe_aa = 0.5 + 0.5*(Ce_aa - Ce_ss)/aux00;
			dlamda1dCe_ss = 0.5 - 0.5*(Ce_aa - Ce_ss)/aux00;
			dlamda1dCe_as = 2*Ce_as/aux00;
			dlamda0dCe_aa = 0.5 - 0.5*(Ce_aa - Ce_ss)/aux00;
			dlamda0dCe_ss = 0.5 + 0.5*(Ce_aa - Ce_ss)/aux00;
			dlamda0dCe_as = -2.*Ce_as/aux00;
			//
		}else{
			dlamda1da0 = (1./2.)*(dCe_aada0+dCe_ssda0);
			dlamda0da0 = (1./2.)*(dCe_aada0+dCe_ssda0);
			// Aternatively, do all the chain rule
			dlamda1dCe_aa = 0.5 ;
			dlamda1dCe_ss = 0.5 ;
			dlamda1dCe_as = 0;
			dlamda0dCe_aa = 0.5;
			dlamda0dCe_ss = 0.5;
			dlamda0dCe_as = 0;
		}
		dlamda1da0 = dlamda1dCe_aa*dCe_aada0 + dlamda1dCe_as*dCe_asda0 + dlamda1dCe_ss*dCe_ssda0;
		dlamda0da0 = dlamda0dCe_aa*dCe_aada0 + dlamda0dCe_as*dCe_asda0 + dlamda0dCe_ss*dCe_ssda0;
		// derivative of sinVartheta
		if(fabs(lamda1-Ce_aa)<1e-7 || fabs(lamda1-lamda0)<1e-7){
			// derivative is zero
			dsinVarthetada0 = Vector2d(0.,0.);
		}else{
			aux01 = sqrt(Ce_as*Ce_as + (lamda1-Ce_aa)*(lamda1-Ce_aa));
			dsinVarthetada0 = (1./aux01)*(dlamda1da0 - dCe_aada0) - (lamda1-Ce_aa)/(2*aux01*aux01*aux01)*(2*Ce_as*dCe_asda0+2*(lamda1-Ce_aa)*(dlamda1da0-dCe_aada0));		
			dsinVarthetadlamda1 = 1./aux01 - (lamda1-Ce_aa)/(2*aux01*aux01*aux01)*(2*(lamda1-Ce_aa));
			// total derivative, ignore the existence of lamda 1 here, think sinVartheta(Ce_aa,Ce_as,Ce_ss), meaning chain rule dude
			dsinVarthetadCe_aa = -1./aux01 + (lamda1-Ce_aa)/(aux01*aux01*aux01)*((lamda1-Ce_aa))   + dsinVarthetadlamda1*dlamda1dCe_aa; 
			dsinVarthetadCe_ss = dsinVarthetadlamda1*dlamda1dCe_ss;
			dsinVarthetadCe_as = -(lamda1-Ce_aa)/(aux01*aux01*aux01)*Ce_as + dsinVarthetadlamda1*dlamda1dCe_as;
			//
			dsinVarthetada0 = ((1./aux01) - ((lamda1 - Ce_aa)*(lamda1-Ce_aa)/(aux01*aux01*aux01)))*dlamda1da0 + ((-1./aux01) + ((lamda1 - Ce_aa)*(lamda1 - Ce_aa)/(aux01*aux01*aux01)))*dCe_aada0 - ((lamda1-Ce_aa)/(aux01*aux01*aux01))*Ce_as*dCe_asda0;
		}
		// residual of the Ra0 wrt a0, it is a 2x2 matrix
		domegada0 = ((2.*PIE*phif_dot_plus)/(tau_omega))*(sinVartheta*dlamda1da0+lamda1*dsinVarthetada0);
		
		// Alternative, do everything with respect to cartesian coordinates for Ra0
		double dlamda1dCe00, dlamda1dCe01, dlamda1dCe11;
		double dlamda0dCe00, dlamda0dCe01, dlamda0dCe11;
		if(fabs(Tr*Tr/4. - Det)<1e-7){
			dlamda1dCe00 = 0.5;
			dlamda1dCe01 = 0.0;
			dlamda1dCe11 = 0.5;
			dlamda0dCe00 = 0.5;
			dlamda0dCe01 = 0.0;
			dlamda0dCe11 = 0.5;
		}else{
			dlamda1dCe00 = (Ce00/4 - Ce11/4)/sqrt(-Ce00*Ce11 + Ce01*Ce01 + Tr*Tr/4) + 0.5;
			dlamda1dCe01 = 2*Ce01/sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr);
			dlamda1dCe11 = -(Ce00/4 - Ce11/4)/sqrt(-Ce00*Ce11 + Ce01*Ce01 + Tr*Tr/4) + 0.5;
			dlamda0dCe00 = -(Ce00/4 - Ce11/4)/sqrt(-Ce00*Ce11 + Ce01*Ce01 + Tr*Tr/4) + 0.5;
			dlamda0dCe01 = -2*Ce01/sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr);
			dlamda0dCe11 = (Ce00/4 - Ce11/4)/sqrt(-Ce00*Ce11 + Ce01*Ce01 + Tr*Tr/4) + 0.5;
		}
		double dCe00da0x = 4.0*C00*a0x*a0x*a0x/(lamdaP_a*lamdaP_a) + 4.0*C00*a0x*a0y*a0y/(lamdaP_a*lamdaP_s) - 6.0*C01*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) + 6.0*C01*a0x*a0x*a0y/(lamdaP_a*lamdaP_a) - 2.0*C01*a0y*a0y*a0y/(lamdaP_s*lamdaP_s) + 2.0*C01*a0y*a0y*a0y/(lamdaP_a*lamdaP_s) + 2.0*C11*a0x*a0y*a0y/(lamdaP_s*lamdaP_s) - 4.0*C11*a0x*a0y*a0y/(lamdaP_a*lamdaP_s) + 2.0*C11*a0x*a0y*a0y/(lamdaP_a*lamdaP_a);
		double dCe00da0y = 4.0*C00*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) + 4.0*C00*a0y*a0y*a0y/(lamdaP_s*lamdaP_s) - 2.0*C01*a0x*a0x*a0x/(lamdaP_a*lamdaP_s) + 2.0*C01*a0x*a0x*a0x/(lamdaP_a*lamdaP_a) - 6.0*C01*a0x*a0y*a0y/(lamdaP_s*lamdaP_s) + 6.0*C01*a0x*a0y*a0y/(lamdaP_a*lamdaP_s) + 2.0*C11*a0x*a0x*a0y/(lamdaP_s*lamdaP_s) - 4.0*C11*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) + 2.0*C11*a0x*a0x*a0y/(lamdaP_a*lamdaP_a);
		double dCe01da0x = (1.0*a0x*a0y*(lamdaP_a - lamdaP_s)*(1.0*C01*a0y*(lamdaP_a - lamdaP_s) - 2.0*C11*a0x*lamdaP_a) - 2.0*a0x*lamdaP_s*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) + 1.0*a0y*(lamdaP_a - lamdaP_s)*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - 1.0*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a)*(1.0*C00*a0y*(lamdaP_a - lamdaP_s) - 2.0*C01*a0x*lamdaP_a))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
		double dCe01da0y = (1.0*a0x*a0y*(lamdaP_a - lamdaP_s)*(1.0*C01*a0x*(lamdaP_a - lamdaP_s) - 2.0*C11*a0y*lamdaP_s) + 1.0*a0x*(lamdaP_a - lamdaP_s)*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - 2.0*a0y*lamdaP_a*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - 1.0*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a)*(1.0*C00*a0x*(lamdaP_a - lamdaP_s) - 2.0*C01*a0y*lamdaP_s))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
		double dCe11da0x = 2.0*C00*a0x*a0y*a0y/(lamdaP_s*lamdaP_s) - 4.0*C00*a0x*a0y*a0y/(lamdaP_a*lamdaP_s) + 2.0*C00*a0x*a0y*a0y/(lamdaP_a*lamdaP_a) - 6.0*C01*a0x*a0x*a0y/(lamdaP_s*lamdaP_s) + 6.0*C01*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) - 2.0*C01*a0y*a0y*a0y/(lamdaP_a*lamdaP_s) + 2.0*C01*a0y*a0y*a0y/(lamdaP_a*lamdaP_a) + 4.0*C11*a0x*a0x*a0x/(lamdaP_s*lamdaP_s) + 4.0*C11*a0x*a0y*a0y/(lamdaP_a*lamdaP_s);
		double dCe11da0y = 2.0*C00*a0x*a0x*a0y/(lamdaP_s*lamdaP_s) - 4.0*C00*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) + 2.0*C00*a0x*a0x*a0y/(lamdaP_a*lamdaP_a) - 2.0*C01*a0x*a0x*a0x/(lamdaP_s*lamdaP_s) + 2.0*C01*a0x*a0x*a0x/(lamdaP_a*lamdaP_s) - 6.0*C01*a0x*a0y*a0y/(lamdaP_a*lamdaP_s) + 6.0*C01*a0x*a0y*a0y/(lamdaP_a*lamdaP_a) + 4.0*C11*a0x*a0x*a0y/(lamdaP_a*lamdaP_s) + 4.0*C11*a0y*a0y*a0y/(lamdaP_a*lamdaP_a);
		//std::cout<<"derivatives dlamda1dCe\n"<<dlamda1dCe00<<", "<<dlamda1dCe01<<", "<<dlamda1dCe11<<"\n";
		//std::cout<<"derivatives dlamda0dCe\n"<<dlamda0dCe00<<", "<<dlamda0dCe01<<", "<<dlamda0dCe11<<"\n";
		//std::cout<<"derivatives dCeda0x\n"<<dCe00da0x<<", "<<dCe01da0x<<", "<<dCe11da0x<<"\n";
		//std::cout<<"derivatives dCeda0y\n"<<dCe00da0y<<", "<<dCe01da0y<<", "<<dCe11da0y<<"\n";
		//double dDetda0x = 1.0*a0x*(8.0*C00*C11*pow(a0x,6) + 24.0*C00*C11*pow(a0x,4)*a0y*a0y + 24.0*C00*C11*a0x*a0x*pow(a0y,4) + 8.0*C00*C11*pow(a0y,6) - 8.0*C01*C01*pow(a0x,6) - 24.0*C01*C01*pow(a0x,4)*a0y*a0y - 24.0*C01*C01*a0x*a0x*pow(a0y,4) - 8.0*C01*C01*pow(a0y,6))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
		//double dDetda0y = 1.0*a0y*(8.0*C00*C11*pow(a0x,6) + 24.0*C00*C11*pow(a0x,4)*a0y*a0y + 24.0*C00*C11*a0x*a0x*pow(a0y,4) + 8.0*C00*C11*pow(a0y,6) - 8.0*C01*C01*pow(a0x,6) - 24.0*C01*C01*pow(a0x,4)*a0y*a0y - 24.0*C01*C01*a0x*a0x*pow(a0y,4) - 8.0*C01*C01*pow(a0y,6))/(lamdaP_a*lamdaP_a*lamdaP_s*lamdaP_s);
		
		//dlamda1da0(0) = dlamda1dCe00*dCe00da0x + dlamda1dCe01*dCe01da0x + dlamda1dCe11*dCe11da0x;
		//dlamda1da0(1) = dlamda1dCe00*dCe00da0y + dlamda1dCe01*dCe01da0y + dlamda1dCe11*dCe11da0y;
		//dlamda0da0(0) = dlamda0dCe00*dCe00da0x + dlamda0dCe01*dCe01da0x + dlamda0dCe11*dCe11da0x;
		//dlamda0da0(1) = dlamda0dCe00*dCe00da0y + dlamda0dCe01*dCe01da0y + dlamda0dCe11*dCe11da0y;

		//std::cout<<"dlamda1da0 =\n"<<dlamda1da0<<"\n";
		//std::cout<<"dlamda0da0 =\n"<<dlamda0da0<<"\n";
		//
		double dsinVarthetada0x_exp, dsinVarthetada0y_exp, dsinVarthetadCe00_exp, dsinVarthetadCe01_exp, dsinVarthetadCe11_exp;
		if(fabs(lamda1-lamda0)<1e-7){
			// equal eigenvalues, no way to reorient
			dsinVarthetada0x_exp =0.0;
			dsinVarthetada0y_exp =0.0;
			dsinVarthetadCe00_exp =0.0;
			dsinVarthetadCe01_exp =0.0;
			dsinVarthetadCe11_exp =0.0;
		}else if(fabs(Ce01)<1e-7 && fabs(Ce00-lamda1)>1e-7){
			dsinVarthetada0x_exp = -1;
			dsinVarthetada0y_exp = 2*Ce01/(-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
			dsinVarthetadCe00_exp = Ce01*a0y*(-2*Ce00 + 2*Ce11 + 2.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/(pow((-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2)*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
			dsinVarthetadCe01_exp = 2*a0y*(-4*Ce01*Ce01 + (-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/(pow((-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2)*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
			dsinVarthetadCe11_exp = Ce01*a0y*(2*Ce00 - 2*Ce11 - 2.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/(pow((-1.0*Ce00 + Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2)*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
		}else{
			dsinVarthetada0x_exp = 2*Ce01/sqrt(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) );
			dsinVarthetada0y_exp = (-Ce00 + Ce11 - sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/sqrt(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) );
			dsinVarthetadCe00_exp = (-2*a0y*(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) )*(2*Ce00 - 2*Ce11 + 2.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)) + (-2*Ce01*a0x + a0y*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)))*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))*(4*Ce00 - 4*Ce11 + 4.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)))/(4*pow((4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) ),(3/2) )*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
			dsinVarthetadCe01_exp = 2*(-2*Ce01*a0y*(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) )*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr) + 2*Ce01*(-2*Ce01*a0x + a0y*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)))*(Ce00 - 1.0*Ce11 + 2*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr) + a0x*(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) )*(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))/(pow((4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) ),(3/2) )*(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
			dsinVarthetadCe11_exp = (2*a0y*(4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) )*(2*Ce00 - 2*Ce11 + 2.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)) + (2*Ce01*a0x - a0y*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)))*(Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr))*(4*Ce00 - 4*Ce11 + 4.0*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)))/(4*pow((4*Ce01*Ce01 + pow((Ce00 - 1.0*Ce11 + sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr)),2) ),(3/2))*sqrt(-4*Ce00*Ce11 + 4*Ce01*Ce01 + Tr*Tr));
		}
		//
		//dsinVarthetada0(0) = dsinVarthetada0x_exp + dsinVarthetadCe00_exp*dCe00da0x + dsinVarthetadCe01_exp*dCe01da0x + dsinVarthetadCe11_exp*dCe11da0x;
		//dsinVarthetada0(1) = dsinVarthetada0y_exp + dsinVarthetadCe00_exp*dCe00da0y + dsinVarthetadCe01_exp*dCe01da0y + dsinVarthetadCe11_exp*dCe11da0y;
		//std::cout<<"dsinVarthetada0 = \n"<<dsinVarthetada0<<"\n";
		//
		//domegada0 = ((2.*PIE*phif_dot_plus)/(tau_omega))*(sinVartheta*dlamda1da0+lamda1*dsinVarthetada0);
		//domegada0 = -((2.*PIE*phif_dot_plus)/(tau_omega))*(sinVartheta*sinVartheta*dlamda1da0+2*lamda1*sinVartheta*dsinVarthetada0);
		//std::cout<<"domegada0 = \n"<<domegada0<<"\n";
		//
		dRa0da0 = 	Identity + (-dRomegadomega*a0_0)*domegada0.transpose();
		

		// derivative of Ra0 wrt the dispersion is zero. 
		// derivative of Ra0 wrt lamdaP requires some calculations
		dCe_aadlamdaP.setZero();
		dCe_asdlamdaP.setZero();
		dCe_ssdlamdaP.setZero();
		for(int alpha=0; alpha<2; alpha++){
			for(int beta=0; beta<2; beta++){
				dCe_aadlamdaP(0) += -2*CC(alpha,beta)*(a0(alpha)*a0(beta))/(lamdaP(0)*lamdaP(0)*lamdaP(0));
				dCe_asdlamdaP(0) += -CC(alpha,beta)*(s0(beta)*a0(alpha))/(lamdaP(1)*lamdaP(0)*lamdaP(0));
				dCe_asdlamdaP(1) += -CC(alpha,beta)*(s0(beta)*a0(alpha))/(lamdaP(0)*lamdaP(1)*lamdaP(1));
				dCe_ssdlamdaP(1) += -2*CC(alpha,beta)*(s0(alpha)*s0(beta))/(lamdaP(1)*lamdaP(1)*lamdaP(1));
			}
		}
		// closed form
		dCe_aadlamdaP(0) = -(2.0*CC(0,0)*a0x*a0x + 4.0*CC(0,1)*a0x*a0y + 2.0*CC(1,1)*a0y*a0y)/pow(lamdaP_a,3);
		dCe_aadlamdaP(1) = 0.;
		dCe_asdlamdaP(0) = (CC(0,0)*a0x*a0y - CC(0,1)*a0x*a0x + CC(0,1)*a0y*a0y - CC(1,1)*a0x*a0y)/(lamdaP_a*lamdaP_a*lamdaP_s);
		dCe_asdlamdaP(1) = (CC(0,0)*a0x*a0y - CC(0,1)*a0x*a0x + CC(0,1)*a0y*a0y - CC(1,1)*a0x*a0y)/(lamdaP_a*lamdaP_s*lamdaP_s);
		dCe_ssdlamdaP(0) = 0.;
		dCe_ssdlamdaP(1) = (-2.0*CC(0,0)*a0y*a0y + 4.0*CC(0,1)*a0x*a0y - 2.0*CC(1,1)*a0x*a0x)/pow(lamdaP_s,3);
		// derivative of principal stretches wrt lampdaP
		if(aux00>1e-7){
			dlamda1dlamdaP = (1./2.)*(dCe_aadlamdaP + dCe_ssdlamdaP)+(1./(4*aux00))*(2.*(Ce_aa-Ce_ss)*(dCe_aadlamdaP - dCe_ssdlamdaP)+8.*Ce_as*dCe_asdlamdaP);
			dlamda0dlamdaP = (1./2.)*(dCe_aadlamdaP + dCe_ssdlamdaP)-(1./(4*aux00))*(2.*(Ce_aa-Ce_ss)*(dCe_aadlamdaP - dCe_ssdlamdaP)+8.*Ce_as*dCe_asdlamdaP);
		}else{
			dlamda1dlamdaP = (1./2.)*(dCe_aadlamdaP + dCe_ssdlamdaP);
			dlamda0dlamdaP = (1./2.)*(dCe_aadlamdaP + dCe_ssdlamdaP);
		}
		dlamda1dlamdaP = dlamda1dCe_aa*dCe_aadlamdaP + dlamda1dCe_as*dCe_asdlamdaP + dlamda1dCe_ss*dCe_ssdlamdaP;
		dlamda0dlamdaP = dlamda0dCe_aa*dCe_aadlamdaP + dlamda0dCe_as*dCe_asdlamdaP + dlamda0dCe_ss*dCe_ssdlamdaP;
		// derivative of sinvartheta wrt lamdaP
		//dsinVarthetadlamdaP = dsinVarthetadCe_aa*dCe_aadlamdaP + dsinVarthetadCe_as*dCe_asdlamdaP + dsinVarthetadCe_ss*dCe_ssdlamdaP;
		if(fabs(lamda1-Ce_aa)<1e-7 || fabs(lamda1-lamda0)<1e-7){
			// derivative is zero
			dsinVarthetadlamdaP = Vector2d(0.,0.);
		}else{
			dsinVarthetadlamdaP = ((1./aux01) - ((lamda1 - Ce_aa)/(aux01*aux01*aux01)))*dlamda1dlamdaP + ((-1./aux01) + ((lamda1 - Ce_aa)/(aux01*aux01*aux01)))*dCe_aadlamdaP - (1./(aux01*aux01*aux01))*Ce_as*dCe_asdlamdaP;
		}
		// Alternative, do everything with respect to original cartesian coordinates
		double dCe00dlamdaPa = -1.0*a0x*(2.0*C00*pow(a0x,3)*lamdaP_s + 2.0*C00*a0x*a0y*a0y*lamdaP_a - 2.0*C01*a0x*a0x*a0y*lamdaP_a + 4.0*C01*a0x*a0x*a0y*lamdaP_s + 2.0*C01*pow(a0y,3)*lamdaP_a - 2.0*C11*a0x*a0y*a0y*lamdaP_a + 2.0*C11*a0x*a0y*a0y*lamdaP_s)/(pow(lamdaP_a,3)*lamdaP_s);
		double dCe00dlamdaPs = -1.0*a0y*(2.0*C00*a0x*a0x*a0y*lamdaP_s + 2.0*C00*pow(a0y,3)*lamdaP_a - 2.0*C01*pow(a0x,3)*lamdaP_s - 4.0*C01*a0x*a0y*a0y*lamdaP_a + 2.0*C01*a0x*a0y*a0y*lamdaP_s + 2.0*C11*a0x*a0x*a0y*lamdaP_a - 2.0*C11*a0x*a0x*a0y*lamdaP_s)/(lamdaP_a*pow(lamdaP_s,3));
		double dCe01dlamdaPa = 1.0*(a0x*a0x*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) + a0x*a0y*a0y*(lamdaP_a - lamdaP_s)*(C01*a0x + C11*a0y) + a0x*a0y*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) - a0y*(C00*a0x + C01*a0y)*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a))/(pow(lamdaP_a,3)*lamdaP_s);
		double dCe01dlamdaPs = 1.0*(-a0x*a0x*a0y*(lamdaP_a - lamdaP_s)*(C01*a0y - C11*a0x) - a0x*a0y*(C01*a0x*a0y*(lamdaP_a - lamdaP_s) - C11*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)) + a0x*(C00*a0y - C01*a0x)*(a0x*a0x*lamdaP_s + a0y*a0y*lamdaP_a) + a0y*a0y*(C00*a0x*a0y*(lamdaP_a - lamdaP_s) - C01*(a0x*a0x*lamdaP_a + a0y*a0y*lamdaP_s)))/(lamdaP_a*pow(lamdaP_s,3));
		double dCe11dlamdaPa = 1.0*a0y*(2.0*C00*a0x*a0x*a0y*lamdaP_a - 2.0*C00*a0x*a0x*a0y*lamdaP_s - 2.0*C01*pow(a0x,3)*lamdaP_a + 2.0*C01*a0x*a0y*a0y*lamdaP_a - 4.0*C01*a0x*a0y*a0y*lamdaP_s - 2.0*C11*a0x*a0x*a0y*lamdaP_a - 2.0*C11*pow(a0y,3)*lamdaP_s)/(pow(lamdaP_a,3)*lamdaP_s);
		double dCe11dlamdaPs = -1.0*a0x*(2.0*C00*a0x*a0y*a0y*lamdaP_a - 2.0*C00*a0x*a0y*a0y*lamdaP_s - 4.0*C01*a0x*a0x*a0y*lamdaP_a + 2.0*C01*a0x*a0x*a0y*lamdaP_s - 2.0*C01*pow(a0y,3)*lamdaP_s + 2.0*C11*pow(a0x,3)*lamdaP_a + 2.0*C11*a0x*a0y*a0y*lamdaP_s)/(lamdaP_a*pow(lamdaP_s,3));
		//double dDetdlamdaPa = (-2.0*C00*C11*pow(a0x,8) - 8.0*C00*C11*pow(a0x,6)*a0y*a0y - 12.0*C00*C11*pow(a0x,4)*pow(a0y,4) - 8.0*C00*C11*a0x*a0x*pow(a0y,6) - 2.0*C00*C11*pow(a0y,8) + 2.0*C01*C01*pow(a0x,8) + 8.0*C01*C01*pow(a0x,6)*a0y*a0y + 12.0*C01*C01*pow(a0x,4)*pow(a0y,4) + 8.0*C01*C01*a0x*a0x*pow(a0y,6) + 2.0*C01*C01*pow(a0y,8) )/(pow(lamdaP_a,3)*lamdaP_s*lamdaP_s);
		//double dDetdlamdaPs = (-2.0*C00*C11*pow(a0x,8) - 8.0*C00*C11*pow(a0x,6)*a0y*a0y - 12.0*C00*C11*pow(a0x,4)*pow(a0y,4) - 8.0*C00*C11*a0x*a0x*pow(a0y,6) - 2.0*C00*C11*pow(a0y,8) + 2.0*C01*C01*pow(a0x,8) + 8.0*C01*C01*pow(a0x,6)*a0y*a0y + 12.0*C01*C01*pow(a0x,4)*pow(a0y,4) + 8.0*C01*C01*a0x*a0x*pow(a0y,6) + 2.0*C01*C01*pow(a0y,8) )/(lamdaP_a*lamdaP_a*pow(lamdaP_s,3));
		//

		//dlamda1dlamdaP(0) = dlamda1dCe00*dCe00dlamdaPa + dlamda1dCe01*dCe01dlamdaPa +  dlamda1dCe11*dCe11dlamdaPa;
		//dlamda1dlamdaP(1) = dlamda1dCe00*dCe00dlamdaPs + dlamda1dCe01*dCe01dlamdaPs +  dlamda1dCe11*dCe11dlamdaPs;
		//dlamda0dlamdaP(0) = dlamda0dCe00*dCe00dlamdaPa + dlamda0dCe01*dCe01dlamdaPa +  dlamda0dCe11*dCe11dlamdaPa;
		//dlamda0dlamdaP(1) = dlamda0dCe00*dCe00dlamdaPs + dlamda0dCe01*dCe01dlamdaPs +  dlamda0dCe11*dCe11dlamdaPs;

		//std::cout<<"dlamda1dlamdaP =\n"<<dlamda1dlamdaP<<"\n";
		//std::cout<<"dlamda0dlamdaP =\n"<<dlamda0dlamdaP<<"\n";
		
		//dsinVarthetadlamdaP(0) =  dsinVarthetadCe00_exp*dCe00dlamdaPa + dsinVarthetadCe01_exp*dCe01dlamdaPa + dsinVarthetadCe11_exp*dCe11dlamdaPa;	
		//dsinVarthetadlamdaP(1) =  dsinVarthetadCe00_exp*dCe00dlamdaPs + dsinVarthetadCe01_exp*dCe01dlamdaPs + dsinVarthetadCe11_exp*dCe11dlamdaPs;	
		
		//std::cout<<"dsinVarthetadlamdaP = \n"<<dsinVarthetadlamdaP<<"\n";
		// derivative of omega wrt lamdaP
		domegadlamdaP = (2*PIE*phif_dot_plus/tau_omega)*(lamda1*dsinVarthetadlamdaP + sinVartheta*dlamda1dlamdaP)+((2.*PIE*dphifplusdlamdaP)/(tau_omega))*lamda1*sinVartheta;
		//domegadlamdaP = -(2*PIE*phif_dot_plus/tau_omega)*(2*lamda1*sinVartheta*dsinVarthetadlamdaP + sinVartheta*sinVartheta*dlamda1dlamdaP)+((-2.*PIE*dphifplusdlamdaP)/(tau_omega))*lamda1*sinVartheta*sinVartheta;
		//std::cout<<"domegadlamdaP = \n"<<domegadlamdaP<<"\n";
		// and finally, derivative of Ra0 wrt lamdaP
		dRa0dlamdaP = (dRomegadomega*a0_0)*domegadlamdaP.transpose();
		
		// Tangent of dispersion
		dRkappadphif = (1./tau_kappa)*(pow(lamda0/lamda1,gamma_kappa)/2.-kappa)*dphifplusdphif;
		dRkappada0 =(phif_dot_plus/tau_kappa)*((gamma_kappa/2.)*pow(lamda0/lamda1,gamma_kappa-1))*((1./lamda1)*dlamda0da0-(lamda0/(lamda1*lamda1))*dlamda1da0);
		dRkappadkappa = -1/dt - (phif_dot_plus/tau_kappa);
		dRkappadlamdaP = (phif_dot_plus/tau_kappa)*((gamma_kappa/2.)*pow(lamda0/lamda1,gamma_kappa-1))*((1./lamda1)*dlamda0dlamdaP-(lamda0/(lamda1*lamda1))*dlamda1dlamdaP)+(dphifplusdlamdaP/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/2.  -kappa);
		
		// Tangent of lamdaP
		// derivative wrt phif
		dRlamdaPdphif.setZero();
		dRlamdaPdphif(0) = ((lamdaE_a-1.)/tau_lamdaP_a)*dphifplusdphif;
		dRlamdaPdphif(1) = ((lamdaE_s-1.)/tau_lamdaP_s)*dphifplusdphif;
		// derivative wrt fiber direction
		dRlamdaPada0 = (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a*lamdaE_a))*dCe_aada0;
		dRlamdaPsda0 = (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s*lamdaE_s))*dCe_ssda0;
		// no dependence on the fiber dispersion
		// derivative wrt the lamdaP
		dRlamdaPadlamdaPa = -1./dt + (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a*lamdaE_a))*dCe_aadlamdaP(0)+dphifplusdlamdaP(0)*(lamdaE_a-1)/tau_lamdaP_a;
		dRlamdaPadlamdaPs = 		    (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a*lamdaE_a))*dCe_aadlamdaP(1)+dphifplusdlamdaP(1)*(lamdaE_a-1)/tau_lamdaP_a;
		dRlamdaPsdlamdaPa = 		    (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s*lamdaE_s))*dCe_ssdlamdaP(0)+dphifplusdlamdaP(0)*(lamdaE_s-1)/tau_lamdaP_s;
		dRlamdaPsdlamdaPs = -1./dt + (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s*lamdaE_s))*dCe_ssdlamdaP(1)+dphifplusdlamdaP(1)*(lamdaE_s-1)/tau_lamdaP_s;
		
		// Assemble into the tangent matrix.
		// phif
		KK_local(0,0) = dRphifdphif;  KK_local(0,4) = dRphifdlamdaP_a;  KK_local(0,5) = dRphifdlamdaP_s;
		// a0
		KK_local(1,0) = dRa0dphif(0); KK_local(1,1) = dRa0da0(0,0);KK_local(1,2) = dRa0da0(0,1); KK_local(1,4) = dRa0dlamdaP(0,0);KK_local(1,5) = dRa0dlamdaP(0,1);
		KK_local(2,0) = dRa0dphif(1); KK_local(2,1) = dRa0da0(1,0);KK_local(2,2) = dRa0da0(1,1); KK_local(2,4) = dRa0dlamdaP(1,0);KK_local(2,5) = dRa0dlamdaP(1,1);
		// kappa
		KK_local(3,0) = dRkappadphif; KK_local(3,1) = dRkappada0(0);KK_local(3,2) = dRkappada0(1);KK_local(3,3) = dRkappadkappa; KK_local(3,4) = dRkappadlamdaP(0);KK_local(3,5) = dRkappadlamdaP(1);
		// lamdaP
		KK_local(4,0) = dRlamdaPdphif(0);KK_local(4,1) = dRlamdaPada0(0);KK_local(4,2) = dRlamdaPada0(1); KK_local(4,4) =dRlamdaPadlamdaPa; KK_local(4,5) =dRlamdaPadlamdaPs;
		KK_local(5,0) = dRlamdaPdphif(1);KK_local(5,1) = dRlamdaPsda0(0);KK_local(5,2) = dRlamdaPsda0(1); KK_local(5,4) =dRlamdaPsdlamdaPa; KK_local(5,5) =dRlamdaPsdlamdaPs;
		
		//----------//
		// SOLVE
		//----------//
		
		//std::cout<<"SOLVE.\nRR_local\n"<<RR_local<<"\nKK_local\n"<<KK_local<<"\n";
		
		double normRR = sqrt(RR_local.dot(RR_local));
		residuum = normRR;
		// solve
		SOL_local = KK_local.lu().solve(-RR_local); 
		
		//std::cout<<"SOL_local\n"<<SOL_local<<"\n";
		// update the solution
		double normSOL = sqrt(SOL_local.dot(SOL_local));
		phif += SOL_local(0);
		a0(0) += SOL_local(1);
		a0(1) += SOL_local(2);
		kappa += SOL_local(3);
		lamdaP(0) += SOL_local(4);
		lamdaP(1) += SOL_local(5);
		// normalize a0
		a0 = a0/sqrt(a0.dot(a0));
		//std::cout<<"norm(RR): "<<residuum<<"\n";
		//std::cout<<"norm(SOL): "<<normSOL<<"\n";
		iter += 1;
//		if(normRR > 1e-4 && iter == max_local_iter){
//			std::cout<<"no local convergence\nlamda1: "<<lamda1<<", lamda0: "<<lamda0<<", lamdaP:"<<lamdaP(0)<<","<<lamdaP(1)<<",a0:"<<a0(0)<<","<<a0(1)<<"Ce_aa: "<<Ce_aa<<","<<Ce_as<<","<<Ce_ss<<"\n";
//			std::cout<<"Ce-lamda:"<<fabs(lamda1-Ce_aa)<<"\n";
//			std::cout<<"aux"<<aux00<<"\n";
//			std::cout<<"sinVartheta: "<<sinVartheta<<"\n";
//			std::cout<<"Res\n"<<RR_local<<"\nSOL_local\n"<<SOL_local<<"\n";
//			//throw std::runtime_error("sorry pal ");
//		}
		
	} // END OF WHILE LOOP OF LOCAL NEWTON
	//a0 = a0/sqrt(a0.dot(a0));
	//std::cout<<"Finish local Newton.\niter = "<<iter<<", residuum = "<<residuum<<"\n";
	//std::cout<<"lamda1: "<<lamda1<<", lamda0: "<<lamda0<<", lamdaP:"<<lamdaP(0)<<","<<lamdaP(1)<<",a0:"<<a0(0)<<","<<a0(1)<<"Ce_aa: "<<Ce_aa<<","<<Ce_as<<","<<Ce_ss<<"\n";
	//std::cout<<"Ce-lamda:"<<fabs(lamda1-Ce_aa)<<"\n";
	//std::cout<<"sinVartheta: "<<sinVartheta<<"\n";

	//-----------------------------------//
	// WOUND TANGENTS FOR GLOBAL PROBLEM
	//-----------------------------------//	

	//----------//
	// MECHANICs
	//----------//

	// explicit derivatives of Ce wrt CC
	Matrix2d dCe_aadCC;dCe_aadCC.setZero();
	Matrix2d dCe_ssdCC;dCe_ssdCC.setZero();
	Matrix2d dCe_asdCC;dCe_asdCC.setZero();
	for(int coordi=0;coordi<2;coordi++){
		for(int coordj=0;coordj<2;coordj++){
			dCe_aadCC(coordi,coordj) += a0(coordi)*a0(coordj)/(lamdaP(0)*lamdaP(0));
			dCe_ssdCC(coordi,coordj) += s0(coordi)*s0(coordj)/(lamdaP(1)*lamdaP(1));
			dCe_asdCC(coordi,coordj) += a0(coordi)*s0(coordj)/(lamdaP(0)*lamdaP(1));
		}
	}
	
	// Explicit derivatives of residuals wrt CC
	
	// phif
	Matrix2d dRphifdCC;dRphifdCC.setZero(); 
	double dH_thetadtheta = -1.*H_theta*H_theta*exp(-gamma_theta*(theta_e-vartheta_e))*(-gamma_theta/(lamdaP(0)*lamdaP(1)));
	Matrix2d dthetadCC = (1./2)*theta*CCinv;
	dRphifdCC = p_phi_theta*dH_thetadtheta*(rho/(K_phi_rho+phif))*dthetadCC;
	Matrix2d dphifdotplusdCC = p_phi_theta*dH_thetadtheta*(rho/(K_phi_rho+phif))*dthetadCC;
	
	// a0
	// preprocessing
	Matrix2d dlamda1dCC;
	Matrix2d dlamda0dCC;
	dlamda1dCC = dlamda1dCe_aa*dCe_aadCC + dlamda1dCe_ss*dCe_ssdCC + dlamda1dCe_as*dCe_asdCC;
	dlamda0dCC = dlamda0dCe_aa*dCe_aadCC + dlamda0dCe_ss*dCe_ssdCC + dlamda0dCe_as*dCe_asdCC;
	Matrix2d dsinVarthetadCC;
	dsinVarthetadCC = dsinVarthetadCe_aa*dCe_aadCC+dsinVarthetadCe_ss*dCe_ssdCC+dsinVarthetadCe_as*dCe_asdCC;
	Matrix2d domegadCC;
	domegadCC = (2*PIE*phif_dot_plus/tau_omega)*(lamda1*dsinVarthetadCC + sinVartheta*dlamda1dCC)+ ((2.*PIE*dphifdotplusdCC)/(tau_omega))*lamda1*sinVartheta;
	// a0x
	Matrix2d dRa0xdCC;dRa0xdCC.setZero();
	dRa0xdCC = (-dRomegadomega *a0_0)(0)*domegadCC;
	// a0y
	Matrix2d dRa0ydCC;dRa0ydCC.setZero();
	dRa0ydCC = (-dRomegadomega *a0_0)(1)*domegadCC;
	
	// kappa
	Matrix2d dRkappadCC;
	dRkappadCC = (dphifdotplusdCC/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/2.  -kappa)+(phif_dot_plus/tau_kappa)*((gamma_kappa/2.)*pow(lamda0/lamda1,gamma_kappa-1))*((1./lamda1)*dlamda0dCC-(lamda0/(lamda1*lamda1))*dlamda1dCC);

	// lamdaP
	Matrix2d dRlamdaP_adCC;
	Matrix2d dRlamdaP_sdCC;
	dRlamdaP_adCC = (phif_dot_plus/tau_lamdaP_a)*(1./(2*lamdaE_a))*dCe_aadCC+dphifdotplusdCC*(lamdaE_a-1.)/tau_lamdaP_a;
	dRlamdaP_sdCC = (phif_dot_plus/tau_lamdaP_s)*(1./(2*lamdaE_s))*dCe_ssdCC+dphifdotplusdCC*(lamdaE_s-1.)/tau_lamdaP_s;
	
	// Assemble dRThetadCC
	// count is phi=4, a0x = 4, a0y = 4, kappa = 4, lamdaP_a =4, lamdaP_s = 4
	VectorXd dRThetadCC(24);dRThetadCC.setZero();
	// phi
	dRThetadCC(0) = dRphifdCC(0,0); dRThetadCC(1) = dRphifdCC(0,1); dRThetadCC(2) = dRphifdCC(1,0); dRThetadCC(3) = dRphifdCC(1,1);
	// a0x 
	dRThetadCC(4) = dRa0xdCC(0,0);  dRThetadCC(5) = dRa0xdCC(0,1);  dRThetadCC(6)  = dRa0xdCC(1,0);   dRThetadCC(7) = dRa0xdCC(1,1);
	// a0y
	dRThetadCC(8) = dRa0ydCC(0,0);  dRThetadCC(9) = dRa0ydCC(0,1);  dRThetadCC(10) = dRa0ydCC(1,0);  dRThetadCC(11) = dRa0ydCC(1,1);
	// kappa
	dRThetadCC(12) = dRkappadCC(0,0);  dRThetadCC(13) = dRkappadCC(0,1);  dRThetadCC(14) = dRkappadCC(1,0);  dRThetadCC(15) = dRkappadCC(1,1);
	// lamdaP_a
	dRThetadCC(16) = dRlamdaP_adCC(0,0);  dRThetadCC(17) = dRlamdaP_adCC(0,1);  dRThetadCC(18) = dRlamdaP_adCC(1,0);  dRThetadCC(19) = dRlamdaP_adCC(1,1);
	// lamdaP_s
	dRThetadCC(20) = dRlamdaP_sdCC(0,0);  dRThetadCC(21) = dRlamdaP_sdCC(0,1);  dRThetadCC(22) = dRlamdaP_sdCC(1,0);  dRThetadCC(23) = dRlamdaP_sdCC(1,1);
	
	// Assemble KK_local_extended
	
	MatrixXd dRThetadTheta_ext(24,24);dRThetadTheta_ext.setZero();
	for(int kkj=0;kkj<6;kkj++){
		// phi
		dRThetadTheta_ext(0,kkj*4+0) = KK_local(0,kkj);
		dRThetadTheta_ext(1,kkj*4+1) = KK_local(0,kkj);
		dRThetadTheta_ext(2,kkj*4+2) = KK_local(0,kkj);
		dRThetadTheta_ext(3,kkj*4+3) = KK_local(0,kkj);
		// a0x
		dRThetadTheta_ext(4,kkj*4+0) = KK_local(1,kkj);
		dRThetadTheta_ext(5,kkj*4+1) = KK_local(1,kkj);
		dRThetadTheta_ext(6,kkj*4+2) = KK_local(1,kkj);
		dRThetadTheta_ext(7,kkj*4+3) = KK_local(1,kkj);
		// a0y
		dRThetadTheta_ext(8,kkj*4+0) =  KK_local(2,kkj);
		dRThetadTheta_ext(9,kkj*4+1) =  KK_local(2,kkj);
		dRThetadTheta_ext(10,kkj*4+2) = KK_local(2,kkj);
		dRThetadTheta_ext(11,kkj*4+3) = KK_local(2,kkj);
		// kappa
		dRThetadTheta_ext(12,kkj*4+0) = KK_local(3,kkj);
		dRThetadTheta_ext(13,kkj*4+1) = KK_local(3,kkj);
		dRThetadTheta_ext(14,kkj*4+2) = KK_local(3,kkj);
		dRThetadTheta_ext(15,kkj*4+3) = KK_local(3,kkj);
		// lamdaP_a
		dRThetadTheta_ext(16,kkj*4+0) = KK_local(4,kkj);
		dRThetadTheta_ext(17,kkj*4+1) = KK_local(4,kkj);
		dRThetadTheta_ext(18,kkj*4+2) = KK_local(4,kkj);
		dRThetadTheta_ext(19,kkj*4+3) = KK_local(4,kkj);
		// lamdaP_s
		dRThetadTheta_ext(20,kkj*4+0) = KK_local(5,kkj);
		dRThetadTheta_ext(21,kkj*4+1) = KK_local(5,kkj);
		dRThetadTheta_ext(22,kkj*4+2) = KK_local(5,kkj);
		dRThetadTheta_ext(23,kkj*4+3) = KK_local(5,kkj);
	}
	
	// SOLVE for the dThetadCC
	dThetadCC = dRThetadTheta_ext.lu().solve(-dRThetadCC);
	
	
	
	//----------//
	// RHO
	//----------//
	
	// Explicit derivatives of the residuals with respect to rho
	
	// phi
	double dRphidrho = (p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*H_theta)*(1./(K_phi_rho+phif)) - d_phi_rho_c*c*phif;
	double dphif_dot_plusdrho = (p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*H_theta)*(1./(K_phi_rho+phif));
	
	// a0
	double domegadrho = ((2.*PIE*dphif_dot_plusdrho)/(tau_omega))*lamda1*sinVartheta; 
	Vector2d dRa0drho = (-dRomegadomega*a0_0)*domegadrho;
	
	// kappa
	double dRkappadrho =(dphif_dot_plusdrho/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/2.  -kappa);
	
	// lamdaP
	Vector2d dRlamdaPdrho;
	dRlamdaPdrho(0) =  dphif_dot_plusdrho*(lamdaE_a-1)/tau_lamdaP_a;
	dRlamdaPdrho(1) =  dphif_dot_plusdrho*(lamdaE_s-1)/tau_lamdaP_s;
	
	// Aseemble in one vector
	VectorXd dRThetadrho(6);
	dRThetadrho(0) = dRphidrho;
	dRThetadrho(1) = dRa0drho(0);
	dRThetadrho(2) = dRa0drho(1);
	dRThetadrho(3) = dRkappadrho;
	dRThetadrho(4) = dRlamdaPdrho(0);
	dRThetadrho(5) = dRlamdaPdrho(1);
	
	// the tangent matrix in this case remains the same as KK_local
	dThetadrho = KK_local.lu().solve(-dRThetadrho);
	
	
	
	//----------//
	// c
	//----------//

	// Explicit derivatives of the residuals with respect to c
	
	// phi
	double dphif_dot_plusdc = (rho/(K_phi_rho+phif))*((p_phi_c)/(K_phi_c+c) - (p_phi_c*c)/((K_phi_c+c)*(K_phi_c+c)));
	double dRphidc = dphif_dot_plusdc - d_phi_rho_c*rho*phif;
	
	// a0
	double domegadc = ((2.*PIE*dphif_dot_plusdc)/(tau_omega))*lamda1*sinVartheta; 
	Vector2d dRa0dc = (-dRomegadomega*a0_0)*domegadc;
	
	// kappa
	double dRkappadc =(dphif_dot_plusdc/tau_kappa)*( pow(lamda0/lamda1,gamma_kappa)/2.  -kappa);
	
	// lamdaP
	Vector2d dRlamdaPdc;
	dRlamdaPdc(0) =  dphif_dot_plusdc*(lamdaE_a-1)/tau_lamdaP_a;
	dRlamdaPdc(1) =  dphif_dot_plusdc*(lamdaE_s-1)/tau_lamdaP_s;
	
	// Aseemble in one vector
	VectorXd dRThetadc(6);
	dRThetadc(0) = dRphidc;
	dRThetadc(1) = dRa0dc(0);
	dRThetadc(2) = dRa0dc(1);
	dRThetadc(3) = dRkappadc;
	dRThetadc(4) = dRlamdaPdc(0);
	dRThetadc(5) = dRlamdaPdc(1);
	
	// the tangent matrix in this case remains the same as KK_local
	dThetadc = KK_local.lu().solve(-dRThetadc);

}

//========================================================//
// EXPLICIT LOCAL PROBLEM: structural update
//========================================================//
void localWoundProblemExplicit(
        double dt, const std::vector<double> &local_parameters,
        double c,double rho,const Matrix2d &FF,
        const double &phif_0, const Vector2d &a0_0, const double &kappa_0, const Vector2d &lamdaP_0,
        double &phif, Vector2d &a0, double &kappa, Vector2d &lamdaP,
        VectorXd &dThetadCC, VectorXd &dThetadrho, VectorXd &dThetadc)
{
    //---------------------------------//
    //
    // INPUT
    // 	matParam: material parameters
    //	rho: value of the cell density at the point
    //	c: concentration at the point
    //	CC: deformation at the point
    //	Theta_t: value of the parameters at previous time step
    //
    // OUTPUT
    //	Theta: value of the parameters at the current time step
    //	dThetadCC: derivative of Theta wrt global mechanics (CC)
    // 	dThetadrho: derivative of Theta wrt global rho
    // 	dThetadc: derivative of Theta wrt global C
    //
    //---------------------------------//

    //---------------------------------//
    // Parameters
    //
    // collagen fraction
    double p_phi = local_parameters[0]; // production by fibroblasts, natural rate
    double p_phi_c = local_parameters[1]; // production up-regulation, weighted by C and rho
    double p_phi_theta = local_parameters[2]; //production regulated by stretch
    double K_phi_c = local_parameters[3]; // saturation of C effect on deposition
    double K_phi_rho = local_parameters[4]; // saturation of collagen fraction itself
    double d_phi = local_parameters[5]; // rate of degradation
    double d_phi_rho_c = local_parameters[6]; // rate of degradation
    //
    // fiber alignment
    double tau_omega = local_parameters[7]; // time constant for angular reorientation
    //
    // dispersion parameter
    double tau_kappa = local_parameters[8]; // time constant
    double gamma_kappa = local_parameters[9]; // exponent of the principal stretch ratio
    //
    // permanent contracture/growth
    double tau_lamdaP_a = local_parameters[10]; // time constant for direction a
    double tau_lamdaP_s = local_parameters[11]; // time constant for direction s
    //
    double gamma_theta = local_parameters[12]; // exponent of the Heaviside function
    double vartheta_e = local_parameters[13]; // mechanosensing response
    //
    // solution parameters
    double tol_local = local_parameters[14]; // local tolerance
    double time_step_ratio = local_parameters[15]; // max local iter or time step ratio
    //
    // other local stuff
    double local_dt = dt/time_step_ratio;
    //
    double PIE = 3.14159;
    Matrix2d Rot90;Rot90<<0.,-1.,1.,0.;

    // Use these to get at the six elements of CC that we need
    Vector3d voigt_table_I_i(0,1,0);
    Vector3d voigt_table_I_j(0,1,1);

    //---------------------------------//
    // KINEMATICS
    //---------------------------------//
    Matrix2d CC = FF.transpose()*FF;
    Matrix2d CCinv = CC.inverse();
    // re-compute basis a0, s0, n0. (If not always vertical, could find n0 with cross product or rotations. Be careful of sign.)
    // fiber tensor in the reference
    Matrix2d a0a0, s0s0;
    // recompute split
    Matrix2d FFg, FFginv, FFe;
    // elastic strain
    Matrix2d CCe, CCeinv;
    //
    //---------------------------------//

    //---------------------------------//
    // LOCAL EXPLICIT FORWARD-EULER ITERATION
    //---------------------------------//
    //
    // Declare Variables
    // New phi, a0x, a0y, kappa, lamdaPa, lamdaPs
    phif = phif_0;
    // make sure it is unit length
    Vector2d s0 = Rot90*a0;
    a0 = a0_0/(sqrt(a0_0.dot(a0_0)));
    s0 = Rot90*a0;
    kappa = kappa_0;
    lamdaP = lamdaP_0;
    //
    double phif_dot, kappa_dot;
    Vector2d lamdaP_dot, a0_dot;
    //
    VectorXd dThetadCC_num(18); dThetadCC_num.setZero();
    VectorXd dThetadrho_num(6); dThetadrho_num.setZero();
    VectorXd dThetadc_num(6); dThetadc_num.setZero();

    //std::ofstream myfile;
    //myfile.open("FE_results.csv");

    for(int step=0;step<time_step_ratio;step++){
        // Save results
        //myfile << std::fixed << std::setprecision(10) << local_dt*step << "," << phif << "," << kappa << "," << a0(0) << "," << a0(1) << "," << lamdaP(0) << "," << lamdaP(1) << "\n";
        //std::cout << "\n a0.dot(s0) " << a0.dot(s0) << "\n  a0.dot(n0)" << a0.dot(n0) << "\n s0.dot(n0)" << s0.dot(n0) << "\n";

        // fiber tensor in the reference
        a0a0 = a0*a0.transpose();
        s0s0 = s0*s0.transpose();
        // recompute split
        FFg = lamdaP(0)*(a0a0) + lamdaP(1)*(s0s0);
        FFginv = (1./lamdaP(0))*(a0a0) + (1./lamdaP(1))*(s0s0);
        //Matrix2d FFe = FF*FFginv;
        // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
        // elastic strain
        CCe = FFginv*CC*FFginv;
        CCeinv = CCe.inverse();
        // Jacobian of the deformations
        double Jp = lamdaP(0)*lamdaP(1);
        double Je = sqrt(CCe.determinant());
        double J = Je*Jp;

        // Eigenvalues
        // Use .compute() for the QR algorithm, .computeDirect() for an explicit trig
        // QR may be more accurate, but explicit is faster
        // Eigenvalues
        SelfAdjointEigenSolver<Matrix2d> eigensolver;
        eigensolver.compute(CCe);
        Vector2d lamda = eigensolver.eigenvalues();
        Matrix2d vectors = eigensolver.eigenvectors();
        double lamdamax = lamda(1);
        double lamdamin = lamda(0);
        Vector2d vectormax = vectors.col(1);
        Vector2d vectormin = vectors.col(0);
        if (a0.dot(vectormax) < 0) {
            vectormax = -vectormax;
        }
        // If CC is the identity matrix, the eigenvectors are arbitrary which is problematic.
        // Beware the matrix becoming singular. Need to perturb.
        double epsilon = 1e-7;
        double delta = 1e-7;
        if(abs(lamdamin-lamdamax) < epsilon ){
            lamdamax = lamdamax*(1+delta);
            lamdamin = lamdamin*(1-delta);
        }
        //std::cout << "\n vectormax" << vectormax << "\n lamdaMax" << lamdamax << "\n";

        // Mechanosensing
        double He = 1./(1.+exp(-gamma_theta*(Je - vartheta_e)));
        //if(He<0.002){He=0;}

        //----------------------------//
        // 2D FORWARD-EULER EQUATIONS
        //----------------------------//
        // Collagen density PHI
        double phif_dot_plus = (p_phi + (p_phi_c*c)/(K_phi_c+c) + p_phi_theta*He)*(rho/(K_phi_rho+phif));
        //std::cout<<"phidotplus: "<<phif_dot_plus<<"\n";
        phif_dot = phif_dot_plus - (d_phi + c*rho*d_phi_rho_c)*phif;

        // Principal direction A0
        // Alternatively, see Menzel (NOTE THAT THE THIRD COMPONENT IS THE LARGEST ONE)
        // a0 = a0 + local_dt*(((2.*PIE*phif_dot_plus)/(tau_omega))*lamda(2)*(Identity-a0a0)*(vectors.col(2)));
        a0_dot = ((2.*PIE*phif_dot_plus)/(tau_omega))*lamdamax*(Matrix2d::Identity()-a0a0)*vectormax;

        // Dispersion KAPPA
        // kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamda(2)/lamda(1),gamma_kappa)/2. - kappa);
        kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamdamin/lamdamax,gamma_kappa)/2. - kappa);

        // elastic stretches of the directions a and s
        double Ce_aa = a0.transpose()*CCe*a0;
        double Ce_ss = s0.transpose()*CCe*s0;
        double lamdaE_a = sqrt(Ce_aa);
        double lamdaE_s = sqrt(Ce_ss);

        // Permanent deformation LAMDAP
        lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-1)/tau_lamdaP_a;
        lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-1)/tau_lamdaP_s;

        //----------------------------------------//
        // CALCULATE GLOBAL CHAIN RULE DERIVATIVES
        //----------------------------------------//

        // Calculate derivatives of eigenvalues and eigenvectors
        Matrix2d dCCdCC; dCCdCC.setZero();
        Matrix3d LHS; LHS.setZero();
        Vector3d RHS,SOL; RHS.setZero(), SOL.setZero();
        //std::vector<Vector2d> dvectordCCe(9,Vector2d::Zero());
        //std::vector<Vector2d> dvectordCC(9,Vector2d::Zero());
        std::vector<Matrix2d> dvectormaxdCCe(2,Matrix2d::Zero());
        std::vector<Matrix2d> dvectormaxdCC(2,Matrix2d::Zero());

        // We actually only  need one eigenvector so an outer loop is not needed, but if we want more just change to 3.
        // Create matrix for calculation of derivatives of eigenvalues and vectors.
        LHS << CCe(0,0) - lamdamax, CCe(0,1), -vectormax(0),
                CCe(1,0), CCe(1,1) - lamdamax, -vectormax(1),
                vectormax(0), vectormax(1), 0;
        // CC is symmetric so we actually only need 6 components.
        //std::cout<<"\n"<<MM<<"\n"<<MM.determinant()<<"\n";
        for (int ii=0; ii<2; ii++){
            for (int jj=0; jj<2; jj++) {
                // Create vector for right hand side. It is the product of an elementary matrix with the eigenvector.
                RHS.setZero();
                RHS(ii) = -vectormax(jj);
                // Solve
                SOL = LHS.lu().solve(RHS);
                dvectormaxdCCe[0](ii,jj) = SOL(0); // II counts the Voigt components of CCe, index has the eigenvector components
                dvectormaxdCCe[1](ii,jj) = SOL(1);
                //dlamdamaxdCCe[II] = SOL(2);
            }
        }

        for (int ii=0; ii<2; ii++){
            for (int jj=0; jj<2; jj++) {
                for (int kk=0; kk<2; kk++){
                    for (int ll=0; ll<2; ll++) {
                        for (int mm=0; mm<2; mm++) {
                            dvectormaxdCC[mm](kk,ll) = dvectormaxdCCe[mm](ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                        }
                    }
                }
            }
        }

        // Alternatively for the eigenvalue we can use the rule from Holzapfel
        // But we still need dCCedCC for the chain rule
        Matrix2d dlamdamaxdCCe = vectormax*vectormax.transpose();
        Matrix2d dlamdamindCCe = vectormin*vectormin.transpose();

        // Multiply by dCCdCCe to get dlamdadCC
        Matrix2d dlamdamaxdCC; Matrix2d dlamdamindCC;
        dlamdamaxdCC.setZero(); dlamdamindCC.setZero();
        for (int ii=0; ii<2; ii++){
            for (int jj=0; jj<2; jj++) {
                for (int kk=0; kk<2; kk++){
                    for (int ll=0; ll<2; ll++) {
                        dlamdamaxdCC(kk,ll) = dlamdamaxdCCe(ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                        dlamdamindCC(kk,ll) = dlamdamindCCe(ii,jj)*(FFginv(ii,jj)*FFginv(kk,ll));
                    }
                }
            }
        }

        // Calculate derivative of lamdaE wrt CC. This will involve an elementary matrix.
        Matrix2d dlamdaE_a_dCC; dlamdaE_a_dCC.setZero();
        Matrix2d dlamdaE_s_dCC; dlamdaE_s_dCC.setZero();
        // Matrix multiplication is associative, so d(a0*FFginv)*CC*(FFginv*a0)/dCC
        // is the outer product of the two vectors we get from a0*FFginv
        // and the symmetry makes the calculation easier
        for (int ii=0; ii<2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
                for (int kk = 0; kk < 2; kk++) {
                    for (int ll = 0; ll < 2; ll++) {
                        dlamdaE_a_dCC(jj,kk) = (a0(ii) * FFginv(ii,jj)) * (FFginv(kk,ll) * a0(ll));
                        dlamdaE_s_dCC(jj,kk) = (s0(ii) * FFginv(ii,jj)) * (FFginv(kk,ll) * s0(ll));
                    }
                }
            }
        }
        // Calculate derivative of He wrt to CC. If this is the same H, this is the same as in the main code.
        Matrix2d dHedCC_explicit, dphifdotplusdCC; dHedCC_explicit.setZero(); dphifdotplusdCC.setZero();
        phif_dot_plus = (p_phi + (p_phi_c*c)/(K_phi_c+c) + p_phi_theta*He)*(rho/(K_phi_rho+phif));
        dHedCC_explicit = (-1./pow((1.+exp(-gamma_theta*(Je - vartheta_e))),2))*(exp(-gamma_theta*(Je - vartheta_e)))*(-gamma_theta)*(J*CCinv/(2*Jp));
        dphifdotplusdCC = p_phi_theta*dHedCC_explicit*(rho/(K_phi_rho+phif));
        //std::cout<<"RHO " << rho << " p_phi_theta " << p_phi_theta << " dHedCC_explicit " << dHedCC_explicit << " CCinv " << CCinv;

        //----------//
        // X
        //----------//
        // Explicit derivatives of the local variables with respect to CC (phi, a0, kappa, lamdap)
        // Really, there are 8*9 = 72 components.
        // Then, remember that CC is symmetric so we actually only need 8*6 = 48 unique values.
        std::vector<Matrix2d> da0dCC(2,Matrix2d::Zero());
        for (int ii=0; ii<2; ii++){
            for (int jj=0; jj<2; jj++) {
                for (int kk=0; kk<2; kk++){
                    for (int ll=0; ll<2; ll++) {
                        da0dCC[kk](ii,jj) += (2.*PIE/tau_omega)*((dphifdotplusdCC(ii,jj)*lamdamax*(Matrix2d::Identity()(kk,ll)-a0a0(kk,ll))*(vectormax(ll)))
                                                                 + (phif_dot_plus*dlamdamaxdCC(ii,jj)*(Matrix2d::Identity()(kk,ll)-a0a0(kk,ll))*(vectormax(ll)))
                                                                 + (phif_dot_plus*lamdamax*((Matrix2d::Identity()(kk,ll)-a0a0(kk,ll))*dvectormaxdCC[ll](ii,jj))));
                    }
                }
            }
        }


        for (int II=0; II<3; II++){
            int ii = voigt_table_I_i(II);
            int jj = voigt_table_I_j(II);
            // phif
            dThetadCC(0+II) += local_dt*dphifdotplusdCC(ii,jj);
            // a0x, a0y, a0z
            dThetadCC(3+II) += local_dt*da0dCC[0](ii,jj);
            dThetadCC(6+II) += local_dt*da0dCC[1](ii,jj);
            // kappa
            dThetadCC(9+II) += (local_dt/(tau_kappa))*((dphifdotplusdCC(ii,jj)*(pow(lamdamin/lamdamax,gamma_kappa)/2. - kappa))
                                                        + ((phif_dot_plus/2.)*(pow(dlamdamindCC(ii,jj)/lamdamax,gamma_kappa) - pow(lamdamin*dlamdamaxdCC(ii,jj)/(lamdamax*lamdamax),gamma_kappa))));
            // lamdaPa, lamdaPs, lamdaPn
            dThetadCC(12+II) += (local_dt/tau_lamdaP_a)*((dphifdotplusdCC(ii,jj)*(lamdaE_a-1)) + (phif_dot_plus*(dlamdaE_a_dCC(ii,jj))));
            dThetadCC(15+II) += (local_dt/tau_lamdaP_s)*((dphifdotplusdCC(ii,jj)*(lamdaE_s-1)) + (phif_dot_plus*(dlamdaE_s_dCC(ii,jj))));
        }

        //----------//
        // RHO
        //----------//

        // Explicit derivatives of the local variables with respect to rho
        // Assemble in one vector (phi, a0x, a0y, kappa, lamdap1, lamdap2)
        double dphifdotplusdrho = ((p_phi + (p_phi_c*c)/(K_phi_c+c)+p_phi_theta*He)*(1/(K_phi_rho+phif)));
        dThetadrho(0) += local_dt*(dphifdotplusdrho - (c*d_phi_rho_c)*phif);
        dThetadrho(1) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix2d::Identity()-a0a0)*(vectormax))(0)*dphifdotplusdrho;
        dThetadrho(2) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix2d::Identity()-a0a0)*(vectormax))(1)*dphifdotplusdrho;
        dThetadrho(3) += local_dt*(1/tau_kappa)*( pow(lamdamin/lamdamax,gamma_kappa)/2. - kappa)*dphifdotplusdrho;
        dThetadrho(4) += local_dt*((lamdaE_a-1)/tau_lamdaP_a)*dphifdotplusdrho;
        dThetadrho(5) += local_dt*((lamdaE_s-1)/tau_lamdaP_s)*dphifdotplusdrho;

        //----------//
        // c
        //----------//

        // Explicit derivatives of the local variables with respect to c
        // Assemble in one vector (phi, a0x, a0y, kappa, lamdap1, lamdap2)
        //double dphifdotplusdc = ((p_phi_c*K_phi_c)/(pow(K_phi_c+c,2)))*(rho/(K_phi_rho+phif_0));
        double dphifdotplusdc = (rho/(K_phi_rho+phif))*((p_phi_c)/(K_phi_c+c) - (p_phi_c*c)/((K_phi_c+c)*(K_phi_c+c)));
        dThetadc(0) += local_dt*(dphifdotplusdc - (rho*d_phi_rho_c)*phif);
        dThetadc(1) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix2d::Identity()-a0a0)*(vectormax))(0)*dphifdotplusdc;
        dThetadc(2) += local_dt*(((2.*PIE)/(tau_omega))*lamdamax*(Matrix2d::Identity()-a0a0)*(vectormax))(1)*dphifdotplusdc;
        dThetadc(3) += local_dt*(1/tau_kappa)*( pow(lamdamin/lamdamax,gamma_kappa)/2. - kappa)*dphifdotplusdc;
        dThetadc(4) += local_dt*((lamdaE_a-1)/tau_lamdaP_a)*dphifdotplusdc;
        dThetadc(5) += local_dt*((lamdaE_s-1)/tau_lamdaP_s)*dphifdotplusdc;

        //---------------------//
        // COMPARE WITH NUMERICAL
        //---------------------//
        /*
        // Calculate numerical derivatives
        //std::cout << "Last iteration" << "\n";
        double phif_dot_plus_num; double phif_dot_minus;
        double kappa_dot_plus; double kappa_dot_minus;
        Vector2d a0_dot_plus; Vector2d a0_dot_minus;
        Vector2d lamdaP_dot_plus; Vector2d lamdaP_dot_minus;
        epsilon = 1e-7;
        double rho_plus = rho + epsilon;
        double rho_minus = rho - epsilon;
        double c_plus = c + epsilon;
        double c_minus = c - epsilon;
        Matrix2d CC_plus, CC_minus;

        // Call update function with plus and minus to get numerical derivatives
        evalForwardEulerUpdate(local_dt, local_parameters, c, rho_plus, FF, CC, phif, a0, s0, kappa, lamdaP, phif_dot_plus_num, a0_dot_plus, kappa_dot_plus, lamdaP_dot_plus);
        evalForwardEulerUpdate(local_dt, local_parameters, c, rho_minus, FF, CC,phif, a0, s0, kappa, lamdaP, phif_dot_minus, a0_dot_minus, kappa_dot_minus, lamdaP_dot_minus);
        dThetadrho_num(0) += local_dt*(1./(2.*epsilon))*(phif_dot_plus_num-phif_dot_minus);
        dThetadrho_num(1) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(0)-a0_dot_minus(0));
        dThetadrho_num(2) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(1)-a0_dot_minus(1));
        dThetadrho_num(3) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(2)-a0_dot_minus(2));
        dThetadrho_num(4) += local_dt*(1./(2.*epsilon))*(kappa_dot_plus-kappa_dot_minus);
        dThetadrho_num(5) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(0)-lamdaP_dot_minus(0));
        dThetadrho_num(6) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(1)-lamdaP_dot_minus(1));
        dThetadrho_num(7) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(2)-lamdaP_dot_minus(2));

        evalForwardEulerUpdate(local_dt, local_parameters, c_plus, rho, FF, CC, phif, a0, s0, kappa, lamdaP, phif_dot_plus_num, a0_dot_plus, kappa_dot_plus, lamdaP_dot_plus);
        evalForwardEulerUpdate(local_dt, local_parameters, c_minus, rho, FF, CC, phif, a0, s0, kappa, lamdaP, phif_dot_minus, a0_dot_minus, kappa_dot_minus, lamdaP_dot_minus);
        dThetadc_num(0) += local_dt*(1./(2.*epsilon))*(phif_dot_plus_num-phif_dot_minus);
        dThetadc_num(1) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(0)-a0_dot_minus(0));
        dThetadc_num(2) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(1)-a0_dot_minus(1));
        dThetadc_num(3) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(2)-a0_dot_minus(2));
        dThetadc_num(4) += local_dt*(1./(2.*epsilon))*(kappa_dot_plus-kappa_dot_minus);
        dThetadc_num(5) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(0)-lamdaP_dot_minus(0));
        dThetadc_num(6) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(1)-lamdaP_dot_minus(1));
        dThetadc_num(7) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(2)-lamdaP_dot_minus(2));

        for (int II=0; II<6; II++){
            int ii = voigt_table_I(II);
            int jj = voigt_table_J(II);

            CC_plus = CC;
            CC_minus = CC;
            CC_plus(ii,jj) += epsilon;
            CC_minus(ii,jj) -= epsilon;
            evalForwardEulerUpdate(local_dt, local_parameters, c, rho, FF, CC_plus, phif, a0, s0, kappa, lamdaP, phif_dot_plus_num, a0_dot_plus, kappa_dot_plus, lamdaP_dot_plus);
            evalForwardEulerUpdate(local_dt, local_parameters, c, rho, FF, CC_minus, phif, a0, s0, kappa, lamdaP, phif_dot_minus, a0_dot_minus, kappa_dot_minus, lamdaP_dot_minus);

            // phif
            dThetadCC_num(0+II) += local_dt*(1./(2.*epsilon))*(phif_dot_plus_num-phif_dot_minus);
            // a0x, a0y, a0z
            dThetadCC_num(6+II) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(0)-a0_dot_minus(0));
            dThetadCC_num(12+II) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(1)-a0_dot_minus(1));
            dThetadCC_num(18+II) += local_dt*(1./(2.*epsilon))*(a0_dot_plus(2)-a0_dot_minus(2));
            // kappa
            dThetadCC_num(24+II) += local_dt*(1./(2.*epsilon))*(kappa_dot_plus-kappa_dot_minus);
            // lamdaPa, lamdaPs, lamdaPn
            dThetadCC_num(30+II) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(0)-lamdaP_dot_minus(0));
            dThetadCC_num(36+II) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(1)-lamdaP_dot_minus(1));
            dThetadCC_num(42+II) += local_dt*(1./(2.*epsilon))*(lamdaP_dot_plus(2)-lamdaP_dot_minus(2));
        }*/

        //---------------------//
        // UPDATE VARIABLES
        //---------------------//
        // Collagen density PHI
        phif = phif + local_dt*(phif_dot);

        // Principal direction A0
        Vector2d a0_00 = a0;
        a0 = a0 + local_dt*a0_dot;
        // normalize a0
        a0 = a0/sqrt(a0.dot(a0));
        //std::cout << "\n a0_00: " << a0_00 << "\n a0: " << a0 << "\n";

        // These should retain their normality, but we can renormalize
        s0 = Rot90*a0;
        s0 = s0/sqrt(s0.dot(s0));

        // These should retain their orthogonality, but we could refacor with modified Gram-Schmidt if they become non-orthogonal
//        n = size(V,1);
//        k = size(V,2);
//        U = zeros(n,k);
//        U(:,1) = V(:,1)/sqrt(V(:,1)'*V(:,1));
//        for i = 2:k
//        U(:,i) = V(:,i);
//        for j = 1:i-1
//        U(:,i) = U(:,i) - ( U(:,j)'*U(:,i) )/( U(:,j)'*U(:,j) )*U(:,j);
//        end
//        U(:,i) = U(:,i)/sqrt(U(:,i)'*U(:,i));
//        end

        // Dispersion KAPPA
        kappa = kappa + local_dt*(kappa_dot);

        // Permanent deformation LAMDAP
        lamdaP = lamdaP + local_dt*(lamdaP_dot);

        //std::cout << "\nphif: " << phif << ", kappa: " << kappa << ", lamdaP:" << lamdaP(0) << "," << lamdaP(1) << "," << lamdaP(2)
        //          << ",a0:" << a0(0) << "," << a0(1) << "," << a0(2) << ",s0:" << s0(0) << "," << s0(1) << "," << s0(2) << ",n0:" << n0(0) << "," << n0(1) << "," << n0(2) << "\n";
    }
    //myfile.close();



    //std::cout<<"lamda1: "<<lamda2d(1)<<", lamda0: "<<lamda2d(0)<<", lamdaP:"<<lamdaP(0)<<","<<lamdaP(1)<<",a0:"<<a0(0)<<","<<a0(1)<<"Ce_aa: "<<Ce_aa<<","<<Ce_ss<<"\n";
    //std::cout<<"\n CC \n"<<CCproj;
    //std::cout << "\n" << dThetadCC_num << "\n";
    //std::cout << "\n" << dThetadrho_num << "\n";
    //std::cout << "\n" << dThetadc_num << "\n";
}

void evalForwardEulerUpdate(double local_dt, const std::vector<double> &local_parameters, double c,double rho,const Matrix2d &FF, const Matrix2d &CC,
                            const double &phif, const Vector2d &a0, const double &kappa, const Vector2d &lamdaP,
                            double &phif_dot, Vector2d &a0_dot, double &kappa_dot, Vector2d &lamdaP_dot)
{
    //---------------------------------//
    // Parameters
    //
    // collagen fraction
    double p_phi = local_parameters[0]; // production by fibroblasts, natural rate
    double p_phi_c = local_parameters[1]; // production up-regulation, weighted by C and rho
    double p_phi_theta = local_parameters[2]; //production regulated by stretch
    double K_phi_c = local_parameters[3]; // saturation of C effect on deposition
    double K_phi_rho = local_parameters[4]; // saturation of collagen fraction itself
    double d_phi = local_parameters[5]; // rate of degradation
    double d_phi_rho_c = local_parameters[6]; // rate of degradation
    //
    // fiber alignment
    double tau_omega = local_parameters[7]; // time constant for angular reorientation
    //
    // dispersion parameter
    double tau_kappa = local_parameters[8]; // time constant
    double gamma_kappa = local_parameters[9]; // exponent of the principal stretch ratio
    //
    // permanent contracture/growth
    double tau_lamdaP_a = local_parameters[10]; // time constant for direction a
    double tau_lamdaP_s = local_parameters[11]; // time constant for direction s
    //
    double gamma_theta = local_parameters[12]; // exponent of the Heaviside function
    double vartheta_e = local_parameters[13]; // mechanosensing response
    //
    // solution parameters
    double tol_local = local_parameters[14]; // local tolerance
    double max_local_iter = local_parameters[15]; // max local iter
    //

    double PIE = 3.14159;
    Matrix2d Rot90;Rot90<<0.,-1.,1.,0.;

    Vector2d s0 = Rot90*a0;
    //std::cout<<"a0"<<"\n"<<a0<<"\n"<<"s0"<<"\n"<<s0<<"\n"<<"n0"<<"\n"<<n0<<'\n';
    // fiber tensor in the reference
    Matrix2d a0a0 = a0*a0.transpose();
    Matrix2d s0s0 = s0*s0.transpose();
    // recompute split
    Matrix2d FFg = lamdaP(0)*(a0a0) + lamdaP(1)*(s0s0);
    Matrix2d FFginv = (1./lamdaP(0))*(a0a0) + (1./lamdaP(1))*(s0s0);
    //Matrix2d FFe = FF*FFginv;
    // std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
    // elastic strain
    Matrix2d CCe = FFginv*CC*FFginv;
    Matrix2d CCeinv = CCe.inverse();
    // Jacobian of the deformations
    double Jp = lamdaP(0)*lamdaP(1)*1.;
    double Je = sqrt(CCe.determinant());
    double J = Je*Jp;

    // Eigenvalues
    // Use .compute() for the QR algorithm, .computeDirect() for an explicit trig
    // QR may be more accurate, but explicit is faster
    // Eigenvalues
    SelfAdjointEigenSolver<Matrix2d> eigensolver;
    eigensolver.compute(CCe);
    Vector2d lamda = eigensolver.eigenvalues();
    Matrix2d vectors = eigensolver.eigenvectors();
    double lamdamax = lamda(2);
    double lamdamed = lamda(1);
    double lamdamin = lamda(0);
    Vector2d vectormax = vectors.col(2);
    Vector2d vectormed = vectors.col(1);
    Vector2d vectormin = vectors.col(0);
    if (a0.dot(vectormax) < 0) {
        vectormax = -vectormax;
    }

    double He = 1./(1.+exp(-gamma_theta*(Je - vartheta_e)));
    //if(He<0.002){He=0;}

    //----------------//
    // 2D FORWARD-EULER EQUATIONS
    //----------------//
    // Collagen density PHI
    double phif_dot_plus = (p_phi + (p_phi_c*c)/(K_phi_c+c) + p_phi_theta*He)*(rho/(K_phi_rho+phif));
    //std::cout<<"phidotplus: "<<phif_dot_plus<<"\n";
    phif_dot = phif_dot_plus - (d_phi + c*rho*d_phi_rho_c)*phif;

    // Principal direction A0
    // Alternatively, see Menzel (NOTE THAT THE THIRD COMPONENT IS THE LARGEST ONE)
    a0_dot = ((2.*PIE*phif_dot_plus)/(tau_omega))*lamdamax*(Matrix2d::Identity()-a0a0)*vectormax;

    // Dispersion KAPPA
    // kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamda(2)/lamda(1),gamma_kappa)/2. - kappa);
    kappa_dot = (phif_dot_plus/tau_kappa)*(pow(lamdamed/lamdamax,gamma_kappa)/3. - kappa);

    // elastic stretches of the directions a and s
    double Ce_aa = a0.transpose()*CCe*a0;
    double Ce_ss = s0.transpose()*CCe*s0;
    double lamdaE_a = sqrt(Ce_aa);
    double lamdaE_s = sqrt(Ce_ss);

    // Permanent deformation LAMDAP
    lamdaP_dot(0) = phif_dot_plus*(lamdaE_a-1)/tau_lamdaP_a;
    lamdaP_dot(1) = phif_dot_plus*(lamdaE_s-1)/tau_lamdaP_s;
}

//--------------------------------------------------------//
// PRINTING ROUTINES
//--------------------------------------------------------//
//
// The point of these functions is to print stuff. For instance for an element I can print the 
// average at the center of the stress or other fields 
// ELEMENT RESIDUAL AND TANGENT
Matrix2d evalWoundFF(
double dt,
const std::vector<Matrix2d> &ip_Jac,
const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
const std::vector<double> &node_X,
std::vector<double> &ip_phif, std::vector<Vector2d> &ip_a0, std::vector<double> &ip_kappa, std::vector<Vector2d> &ip_lamdaP,
const std::vector<Vector2d> &node_x,const std::vector<double> &node_rho, const std::vector<double> &node_c,
double xi, double eta)
{
	// return the deformation gradient
	//---------------------------------//
	// PARAMETERS
	// 
	double k0 = global_parameters[0]; // neo hookean
	double kf = global_parameters[1]; // stiffness of collagen
	double k2 = global_parameters[2]; // nonlinear exponential
	double t_rho = global_parameters[3]; // force of fibroblasts
	double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
	double K_t_c = global_parameters[5]; // saturation of chemical on force
	double D_rhorho = global_parameters[6]; // diffusion of cells
	double D_rhoc = global_parameters[7]; // diffusion of chemotactic gradient
	double D_cc = global_parameters[8]; // diffusion of chemical
	double p_rho =global_parameters[9]; // production of fibroblasts naturally
	double p_rho_c = global_parameters[10]; // production enhanced by the chem
	double p_rho_theta = global_parameters[11]; // mechanosensing
	double K_rho_c= global_parameters[12]; // saturation of cell production by chemical
	double K_rho_rho = global_parameters[13]; // saturation of cell by cell
	double d_rho = global_parameters[14] ;// decay of cells
	double theta_phy = global_parameters[15]; // physiological state of area stretch
	double gamma_c_thetaE = global_parameters[16]; // sensitivity of heviside function
	double p_c_rho = global_parameters[17];// production of C by cells
	double p_c_thetaE = global_parameters[18]; // coupling of elastic and chemical
	double K_c_c = global_parameters[19];// saturation of chem by chem
	double d_c = global_parameters[20]; // decay of chemical
	//
	//---------------------------------//
	
	int n_nodes = node_X.size();
	std::vector<Vector2d> Ebasis; Ebasis.clear();
	Matrix2d Rot90;Rot90<<0.,-1.,1.,0.;
	Ebasis.push_back(Vector2d(1.,0.)); Ebasis.push_back(Vector2d(0.,1.));
	//---------------------------------//
	
	
	//---------------------------------//
	// EVALUATE FUNCTIONS
	//
	// evaluate jacobian [actually J^(-T) ]
	Matrix2d Jac_iT = evalJacobian(node_x,xi,eta);
	//
	// eval linear shape functions (4 of them)
	std::vector<double> R = evalShapeFunctionsR(xi,eta);
	// eval derivatives
	std::vector<double> Rxi = evalShapeFunctionsRxi(xi,eta);
	std::vector<double> Reta = evalShapeFunctionsReta(xi,eta);
	//
	// declare variables and gradients at IP
	std::vector<Vector2d> dRdXi;dRdXi.clear();
	Vector2d dxdxi,dxdeta;
	dxdxi.setZero();dxdeta.setZero();
	double rho=0.; Vector2d drhodXi; drhodXi.setZero();
	double c=0.; Vector2d dcdXi; dcdXi.setZero();
	//
	for(int ni=0;ni<n_nodes;ni++)
	{
		dRdXi.push_back(Vector2d(Rxi[ni],Reta[ni]));
		
		dxdxi += node_x[ni]*Rxi[ni];
		dxdeta += node_x[ni]*Reta[ni];
				
		rho += node_rho[ni]*R[ni];
		drhodXi(0) += node_rho[ni]*Rxi[ni];
		drhodXi(1) += node_rho[ni]*Reta[ni];

		c += node_c[ni]*R[ni];
		dcdXi(0) += node_c[ni]*Rxi[ni];
		dcdXi(1) += node_c[ni]*Reta[ni];
	}
	//
	//---------------------------------//


	//---------------------------------//
	// EVAL GRADIENTS
	//
	// Deformation gradient and strain
	// assemble the columns
	Matrix2d dxdXi; dxdXi<<dxdxi(0),dxdeta(0),dxdxi(1),dxdeta(1);
	// F = dxdX 
	Matrix2d FF = Jac_iT*dxdXi;
	//
	// Gradient of concentrations in current configuration
	Matrix2d dXidx = dxdXi.inverse();
	Vector2d grad_rho  = dXidx.transpose()*drhodXi;
	Vector2d grad_c    = dXidx.transpose()*dcdXi;
	//
	// Gradient of concentrations in reference
	Vector2d Grad_rho = Jac_iT*drhodXi;
	Vector2d Grad_c = Jac_iT*dcdXi;
	
	return FF;
}

//--------------------------------------------------------//
// GEOMETRY and ELEMENT ROUTINES
//--------------------------------------------------------//

//-----------------------------//
// Jacobians
//-----------------------------//

std::vector<Matrix2d> evalJacobian(const std::vector<Vector2d> node_X)
{
	// The gradient of the shape functions with respect to the reference coordinates

	// Vector with 4 elements, each element is the inverse transpose of the 
	// Jacobian at the corresponding integration point of the linear quadrilateral

	std::vector<Matrix2d> ip_Jac;
    int elem_size = node_X.size();
	// LOOP OVER THE INTEGRATION POINTS
	std::vector<Vector3d> IP = LineQuadriIP();
	int n_IP = IP.size();
	for(int ip=0;ip<n_IP;ip++){
	
		// evaluate basis functions derivatives
		// coordinates of the integration point in parent domain
		double xi = IP[ip](0);
		double eta = IP[ip](1);
		
		// eval shape functions
        std::vector<double> R;
        // eval derivatives
        std::vector<double> Rxi;
        std::vector<double> Reta;
        if(elem_size == 4){
            R = evalShapeFunctionsR(xi,eta);
            Rxi = evalShapeFunctionsRxi(xi,eta);
            Reta = evalShapeFunctionsReta(xi,eta);
        }
        else if(elem_size == 8){
            R = evalShapeFunctionsQuadraticR(xi,eta);
            Rxi = evalShapeFunctionsQuadraticRxi(xi,eta);
            Reta = evalShapeFunctionsQuadraticReta(xi,eta);
        }
        else{
            throw std::runtime_error("Wrong number of nodes in element!");
        }

		// sum over the 4 nodes
		Vector2d dXdxi;dXdxi.setZero();
		Vector2d dXdeta;dXdeta.setZero();
		for(int ni=0;ni<elem_size;ni++)
		{
			dXdxi += Rxi[ni]*node_X[ni];
			dXdeta += Reta[ni]*node_X[ni];
		}
		// put them in one column
		Matrix2d Jac; Jac<<dXdxi(0),dXdeta(0),dXdxi(1),dXdeta(1);
		// invert and transpose it 
		Matrix2d Jac_iT = (Jac.inverse()).transpose();
		// save this for the vector
		ip_Jac.push_back(Jac_iT);
	}
	// return the vector with all the inverse jacobians
	return ip_Jac;
}

Matrix2d evalJacobian(const std::vector<Vector2d> node_X, double xi, double eta)
{
	// eval the inverse Jacobian at given xi and eta coordinates
    int elem_size = node_X.size();
	//
    // eval shape functions
    std::vector<double> R;
    // eval derivatives
    std::vector<double> Rxi;
    std::vector<double> Reta;
    if(elem_size == 4){
        R = evalShapeFunctionsR(xi,eta);
        Rxi = evalShapeFunctionsRxi(xi,eta);
        Reta = evalShapeFunctionsReta(xi,eta);
    }
    else if(elem_size == 8){
        R = evalShapeFunctionsQuadraticR(xi,eta);
        Rxi = evalShapeFunctionsQuadraticRxi(xi,eta);
        Reta = evalShapeFunctionsQuadraticReta(xi,eta);
    }
    else{
        throw std::runtime_error("Wrong number of nodes in element!");
    }

	// sum over the 4 nodes
	Vector2d dXdxi;dXdxi.setZero();
	Vector2d dXdeta;dXdeta.setZero();
	for(int ni=0;ni<elem_size;ni++)
	{
		dXdxi += Rxi[ni]*node_X[ni];
		dXdeta += Reta[ni]*node_X[ni];
	}
	// put them in one column
	Matrix2d Jac; Jac<<dXdxi(0),dXdeta(0),dXdxi(1),dXdeta(1);
	// invert and transpose it 
	Matrix2d Jac_iT = (Jac.inverse()).transpose();
	return Jac_iT;
}

//-----------------------------//
// Integration points
//-----------------------------//

std::vector<Vector3d> LineQuadriIP()
{
	// return the integration points of the quadratic hex element
	std::vector<Vector3d> IP;
	std::vector<double> pIP = {-sqrt(3.)/3.,sqrt(3.)/3.};
	std::vector<double> wIP = {1.,1.};
	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			IP.push_back(Vector3d(pIP[i],pIP[j],wIP[i]*wIP[j]));
		}
	}
	return IP;
}

//-----------------------------//
// Basis functions
//-----------------------------//

std::vector<double> evalShapeFunctionsR(double xi,double eta)
{
	std::vector<Vector2d> node_Xi = {Vector2d(-1.,-1.),Vector2d(+1.,-1.),Vector2d(+1.,+1.),Vector2d(-1.,+1.)};
	std::vector<double> R;
	for(int i=0;i<node_Xi.size();i++)
	{
		R.push_back((1./4.)*(1+xi*node_Xi[i](0))*(1+eta*node_Xi[i](1)));
	}
	return R;
}
std::vector<double> evalShapeFunctionsRxi(double xi,double eta)
{
	std::vector<Vector2d> node_Xi = {Vector2d(-1.,-1.),Vector2d(+1.,-1.),Vector2d(+1.,+1.),Vector2d(-1.,+1.)};
	std::vector<double> Rxi;
	for(int i=0;i<node_Xi.size();i++)
	{
		Rxi.push_back((1./4.)*(node_Xi[i](0))*(1+eta*node_Xi[i](1)));
	}
	return Rxi;
}
std::vector<double> evalShapeFunctionsReta(double xi,double eta)
{
	std::vector<Vector2d> node_Xi = {Vector2d(-1.,-1.),Vector2d(+1.,-1.),Vector2d(+1.,+1.),Vector2d(-1.,+1.)};
	std::vector<double> Reta;
	for(int i=0;i<node_Xi.size();i++)
	{
		Reta.push_back((1./4.)*(1+xi*node_Xi[i](0))*(node_Xi[i](1)));
	}
	return Reta;
}

//-----------------------------//
// Quadratic Basis functions
//-----------------------------//

std::vector<double> evalShapeFunctionsQuadraticR(double xi,double eta)
{
    std::vector<double> R;
    R.push_back((1./4.)*(1+xi*(-1))*(1+eta*(-1)) - (1./4.)*(1-xi*xi)*(1+eta*(-1)) - (1./4.)*(1+xi*(-1))*(1-eta*eta)); // Node 1,3,5,7 -> -1,-1
    R.push_back((1./2.)*(1-xi*xi)*(1+eta*(-1))); // Node 2,6
    R.push_back((1./4.)*(1+xi*(+1))*(1+eta*(-1)) - (1./4.)*(1-xi*xi)*(1+eta*(-1)) - (1./4.)*(1+xi*(+1))*(1-eta*eta)); // Node 1,3,5,7
    R.push_back((1./2.)*(1+xi*(+1))*(1-eta*eta)); // Node 4,8
    R.push_back((1./4.)*(1+xi*(+1))*(1+eta*(+1)) - (1./4.)*(1-xi*xi)*(1+eta*(+1)) - (1./4.)*(1+xi*(+1))*(1-eta*eta)); // Node 1,3,5,7
    R.push_back((1./2.)*(1-xi*xi)*(1+eta*(+1))); // Node 2,6
    R.push_back((1./4.)*(1+xi*(-1))*(1+eta*(+1)) - (1./4.)*(1-xi*xi)*(1+eta*(+1)) - (1./4.)*(1+xi*(-1))*(1-eta*eta)); // Node 1,3,5,7
    R.push_back((1./2.)*(1+xi*(-1))*(1-eta*eta)); // Node 4,8
    return R;
}
std::vector<double> evalShapeFunctionsQuadraticRxi(double xi,double eta)
{
    std::vector<double> Rxi;
    Rxi.push_back((1./4.)*((-1))*(1+eta*(-1)) - (1./4.)*(-2*xi)*(1+eta*(-1)) - (1./4.)*((-1))*(1-eta*eta)); // Node 1,3,5,7
    Rxi.push_back((1./2.)*(-2*xi)*(1+eta*(-1))); // Node 2,6
    Rxi.push_back((1./4.)*((+1))*(1+eta*(-1)) - (1./4.)*(-2*xi)*(1+eta*(-1)) - (1./4.)*((+1))*(1-eta*eta)); // Node 1,3,5,7
    Rxi.push_back((1./2.)*((+1))*(1-eta*eta)); // Node 4,8
    Rxi.push_back((1./4.)*((+1))*(1+eta*(+1)) - (1./4.)*(-2*xi)*(1+eta*(+1)) - (1./4.)*((+1))*(1-eta*eta)); // Node 1,3,5,7
    Rxi.push_back((1./2.)*(-2*xi)*(1+eta*(+1))); // Node 2,6
    Rxi.push_back((1./4.)*((-1))*(1+eta*(+1)) - (1./4.)*(-2*xi)*(1+eta*(+1)) - (1./4.)*((-1))*(1-eta*eta)); // Node 1,3,5,7
    Rxi.push_back((1./2.)*((-1))*(1-eta*eta)); // Node 4,8
    return Rxi;
}
std::vector<double> evalShapeFunctionsQuadraticReta(double xi,double eta)
{
    std::vector<double> Reta;
    Reta.push_back((1./4.)*(1+xi*(-1))*((-1)) - (1./4.)*(1-xi*xi)*((-1)) - (1./4.)*(1+xi*(-1))*(-2*eta)); // Node 1,3,5,7
    Reta.push_back((1./2.)*(1-xi*xi)*((-1))); // Node 2,6
    Reta.push_back((1./4.)*(1+xi*(+1))*((-1)) - (1./4.)*(1-xi*xi)*((-1)) - (1./4.)*(1+xi*(+1))*(-2*eta)); // Node 1,3,5,7
    Reta.push_back((1./2.)*(1+xi*(+1))*(-2*eta)); // Node 4,8
    Reta.push_back((1./4.)*(1+xi*(+1))*((+1)) - (1./4.)*(1-xi*xi)*((+1)) - (1./4.)*(1+xi*(+1))*(-2*eta)); // Node 1,3,5,7
    Reta.push_back((1./2.)*(1-xi*xi)*((+1))); // Node 2,6
    Reta.push_back((1./4.)*(1+xi*(-1))*((+1)) - (1./4.)*(1-xi*xi)*((+1)) - (1./4.)*(1+xi*(-1))*(-2*eta)); // Node 1,3,5,7
    Reta.push_back((1./2.)*(1+xi*(-1))*(-2*eta)); // Node 4,8
    return Reta;
}

//-------------------------------//
// Functions for numerical tests
//-------------------------------//

void evalPsif(const std::vector<double> &global_parameters,double kappa, double I1e,double I4e,double &Psif,double &Psif1,double &Psif4)
{
	// unpack material constants
	//---------------------------------//
	// PARAMETERS
	// 
	double k0 = global_parameters[0]; // neo hookean
	double kf = global_parameters[1]; // stiffness of collagen
	double k2 = global_parameters[2]; // nonlinear exponential
	// passive elastic
	Psif = (kf/(2.*k2))*(exp( k2*pow((kappa*I1e + (1-2*kappa)*I4e -1),2))-1);
	Psif1 = 2*k2*kappa*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif;
	Psif4 = 2*k2*(1-2*kappa)*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif;
}

void evalSS(const std::vector<double> &global_parameters, double phif, Vector2d a0, double kappa, double lamdaP_a,double lamdaP_s,const Matrix2d &CC,double rho, double c, Matrix2d &SSpas,Matrix2d &SSact, Matrix2d&SSpres)
{
	Matrix2d Identity;Identity<<1,0,0,1;
	Matrix2d a0a0 = a0*a0.transpose();
	Matrix2d Rot90; Rot90<<0,-1,1,0;
	Vector2d s0 = Rot90*a0;
	Matrix2d s0s0 = s0*s0.transpose();
	Matrix2d FFg = lamdaP_a*(a0a0) + lamdaP_s*(s0s0);
	Matrix2d FFginv = 1./lamdaP_a*(a0a0) + 1./lamdaP_s*(s0s0);
	Matrix2d CCinv = CC.inverse();
	double theta  = sqrt(CC.determinant());
	double thetaP = lamdaP_a*lamdaP_s;
	double thetaE = theta/thetaP;
	double lamda_N = 1./thetaE;
	Matrix2d CCe = FFginv*CC*FFginv;
	double I4tot = a0.dot(CC*a0);
	double trA = kappa*(CC(0,0)+CC(1,1)) + (1-2*kappa)*I4tot;
	double k0 = global_parameters[0]; // neo hookean
	double kf = global_parameters[1]; // stiffness of collagen
	double k2 = global_parameters[2]; // nonlinear exponential
	double t_rho = global_parameters[3]; // force of fibroblasts
	double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
	double K_t_c = global_parameters[5]; // saturation of chemical on force
	double Psif,Psif1,Psif4;
	double I1e = CCe(0,0)+CCe(1,1);
	double I4e = a0.dot(CCe*a0);
	evalPsif(global_parameters,kappa,I1e,I4e,Psif,Psif1,Psif4);
	Matrix2d SSe_pas = k0*Identity + phif*(Psif1*Identity + Psif4*a0a0);
	SSpas = thetaP*FFginv*SSe_pas*FFginv;
	double traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho;
	SSact = (thetaP*traction_act*phif/trA)*(kappa*Identity+(1-2*kappa)*a0a0);
	double pressure = -k0*lamda_N*lamda_N;
	SSpres = pressure*thetaP*CCinv;
}

// ELEMENT RESIDUAL ONLY
void evalWoundRes(
double dt,
const std::vector<Matrix2d> &ip_Jac,
const std::vector<double> &global_parameters,const std::vector<double> &local_parameters,
const std::vector<double> &node_rho_0, const std::vector<double> &node_c_0,
const std::vector<double> &ip_phif_0,const std::vector<Vector2d> &ip_a0_0,const std::vector<double> &ip_kappa_0, const std::vector<Vector2d> &ip_lamdaP_0,
const std::vector<double> &node_rho, const std::vector<double> &node_c,
std::vector<double> &ip_phif, std::vector<Vector2d> &ip_a0, std::vector<double> &ip_kappa, std::vector<Vector2d> &ip_lamdaP,
const std::vector<Vector2d> &node_x,
VectorXd &Re_x,VectorXd &Re_rho,VectorXd &Re_c)
{

	
	//---------------------------------//
	// INPUT
	//  dt: time step
	//	elem_jac_IP: jacobians at the integration points, needed for the deformation grad
	//  matParam: material parameters
	//  Xi_t: global fields at previous time step
	//  Theta_t: structural fields at previous time steps
	//  Xi: current guess of the global fields
	//  Theta: current guess of the structural fields
	//	node_x: deformed positions
	//
	// OUTPUT
	//  Re: all residuals
	//
	// Algorithm
	//  0. Loop over integration points
	//	1. F,rho,c,nabla_rho,nabla_c: deformation at IP
	//  2. LOCAL NEWTON -> update the current guess of the structural parameters
	//  3. Fe,Fp
	//	4. Se_pas,Se_act,S
	//	5. Qrho,Srho,Qc,Sc
	//  6. Residuals
	//---------------------------------//
	
	//---------------------------------//
	// PARAMETERS
	// 
	double k0 = global_parameters[0]; // neo hookean
	double kf = global_parameters[1]; // stiffness of collagen
	double k2 = global_parameters[2]; // nonlinear exponential
	double t_rho = global_parameters[3]; // force of fibroblasts
	double t_rho_c = global_parameters[4]; // force of myofibroblasts enhanced by chemical
	double K_t_c = global_parameters[5]; // saturation of chemical on force
	double D_rhorho = global_parameters[6]; // diffusion of cells
	double D_rhoc = global_parameters[7]; // diffusion of chemotactic gradient
	double D_cc = global_parameters[8]; // diffusion of chemical
	double p_rho =global_parameters[9]; // production of fibroblasts naturally
	double p_rho_c = global_parameters[10]; // production enhanced by the chem
	double p_rho_theta = global_parameters[11]; // mechanosensing
	double K_rho_c= global_parameters[12]; // saturation of cell production by chemical
	double K_rho_rho = global_parameters[13]; // saturation of cell by cell
	double d_rho = global_parameters[14] ;// decay of cells
	double theta_phy = global_parameters[15]; // physiological state of area stretch
	double gamma_c_thetaE = global_parameters[16]; // sensitivity of heviside function
	double p_c_rho = global_parameters[17];// production of C by cells
	double p_c_thetaE = global_parameters[18]; // coupling of elastic and chemical
	double K_c_c = global_parameters[19];// saturation of chem by chem
	double d_c = global_parameters[20]; // decay of chemical
	//std::cout<<"read all global parameters\n";
	//
	//---------------------------------//
	
	
	
	//---------------------------------//
	// GLOBAL VARIABLES
	// Initialize the residuals to zero and declare some global stuff
	Re_x.setZero();
	Re_rho.setZero();
	Re_c.setZero();
	int n_nodes = node_x.size();
	std::vector<Vector2d> Ebasis; Ebasis.clear();
	Matrix2d Rot90;Rot90<<0.,-1.,1.,0.;
	Ebasis.push_back(Vector2d(1.,0.)); Ebasis.push_back(Vector2d(0.,1.));
	//---------------------------------//
	
	
	
	//---------------------------------//
	// LOOP OVER INTEGRATION POINTS
	//---------------------------------//
	
	// array with integration points
	std::vector<Vector3d> IP = LineQuadriIP();
	//std::cout<<"loop over integration points\n";
	for(int ip=0;ip<IP.size();ip++)
	{
	
		//---------------------------------//
		// EVALUATE FUNCTIONS
		//
		// coordinates of the integration point in parent domain
		double xi = IP[ip](0);
		double eta = IP[ip](1);
		// weight of the integration point
		double wip = IP[ip](2);
		double Jac = 1./ip_Jac[ip].determinant();
		//std::cout<<"integration point: "<<xi<<", "<<eta<<"; "<<wip<<"; "<<Jac<<"\n";
		//
        std::vector<double> R;
        std::vector<double> Rxi;
        std::vector<double> Reta;
        if(n_nodes == 4){
            // eval linear shape functions (4 of them)
            R = evalShapeFunctionsR(xi,eta);
            // eval derivatives
            Rxi = evalShapeFunctionsRxi(xi,eta);
            Reta = evalShapeFunctionsReta(xi,eta);
        }
        else if(n_nodes == 8){
            // eval quadratic shape functions (4 of them)
            R = evalShapeFunctionsQuadraticR(xi,eta);
            // eval derivatives
            Rxi = evalShapeFunctionsQuadraticRxi(xi,eta);
            Reta = evalShapeFunctionsQuadraticReta(xi,eta);
        }
        else{
            std::cout << "Wrong number of nodes!";
        }
		//
		// declare variables and gradients at IP
		std::vector<Vector2d> dRdXi;dRdXi.clear();
		Vector2d dxdxi,dxdeta;
		dxdxi.setZero();dxdeta.setZero();
		double rho_0=0.; Vector2d drho0dXi; drho0dXi.setZero();
		double rho=0.; Vector2d drhodXi; drhodXi.setZero();
		double c_0=0.; Vector2d dc0dXi; dc0dXi.setZero();
		double c=0.; Vector2d dcdXi; dcdXi.setZero();
		//
		for(int ni=0;ni<n_nodes;ni++)
		{
			dRdXi.push_back(Vector2d(Rxi[ni],Reta[ni]));
			
			dxdxi += node_x[ni]*Rxi[ni];
			dxdeta += node_x[ni]*Reta[ni];
			
			rho_0 += node_rho_0[ni]*R[ni];
			drho0dXi(0) += node_rho_0[ni]*Rxi[ni];
			drho0dXi(1) += node_rho_0[ni]*Reta[ni];
			
			rho += node_rho[ni]*R[ni];
			drhodXi(0) += node_rho[ni]*Rxi[ni];
			drhodXi(1) += node_rho[ni]*Reta[ni];
			
			c_0 += node_c_0[ni]*R[ni];
			dc0dXi(0) += node_c_0[ni]*Rxi[ni];
			dc0dXi(1) += node_c_0[ni]*Reta[ni];

			c += node_c[ni]*R[ni];
			dcdXi(0) += node_c[ni]*Rxi[ni];
			dcdXi(1) += node_c[ni]*Reta[ni];
		}
		//
		//---------------------------------//



		//---------------------------------//
		// EVAL GRADIENTS
		//
		// Deformation gradient and strain
		// assemble the columns
		Matrix2d dxdXi; dxdXi<<dxdxi(0),dxdeta(0),dxdxi(1),dxdeta(1);
		// F = dxdX 
		Matrix2d FF = ip_Jac[ip]*dxdXi;
		// the strain
		Matrix2d Identity;Identity<<1,0,0,1;
		Matrix2d EE = 0.5*(FF.transpose()*FF - Identity);
		Matrix2d CC = FF.transpose()*FF;
		Matrix2d CCinv = CC.inverse();
		//
		// Gradient of concentrations in current configuration
		Matrix2d dXidx = dxdXi.inverse();
		Vector2d grad_rho0 = dXidx.transpose()*drho0dXi;
		Vector2d grad_rho  = dXidx.transpose()*drhodXi;
		Vector2d grad_c0   = dXidx.transpose()*dc0dXi;
		Vector2d grad_c    = dXidx.transpose()*dcdXi;
		//
		// Gradient of concentrations in reference
		Vector2d Grad_rho0 = ip_Jac[ip]*drho0dXi;
		Vector2d Grad_rho = ip_Jac[ip]*drhodXi;
		Vector2d Grad_c0 = ip_Jac[ip]*dc0dXi;
		Vector2d Grad_c = ip_Jac[ip]*dcdXi;
		//
		// Gradient of basis functions for the nodes in reference
		std::vector<Vector2d> Grad_R;Grad_R.clear();
		//
		// Gradient of basis functions in deformed configuration
		std::vector<Vector2d> grad_R;grad_R.clear();
        for(int ni=0;ni<n_nodes;ni++)
        {
            Grad_R.push_back(ip_Jac[ip]*dRdXi[ni]);
            grad_R.push_back(dXidx.transpose()*dRdXi[ni]);
        }
		//
		//---------------------------------//
		//std::cout<<"deformation gradient\n"<<FF<<"\n";
		//std::cout<<"rho0: "<<rho_0<<", rho: "<<rho<<"\n";
		//std::cout<<"c0: "<<c_0<<", c: "<<c<<"\n";
		//std::cout<<"gradient of rho: "<<Grad_rho<<"\n";
		//std::cout<<"gradient of c: "<<Grad_c<<"\n";

		//---------------------------------//
		// LOCAL NEWTON: structural problem
		//
		VectorXd dThetadCC(24);dThetadCC.setZero();
		VectorXd dThetadrho(6);dThetadrho.setZero();
		VectorXd dThetadc(6);dThetadc.setZero();
		//std::cout<<"Local variables before update:\nphif0 = "<<ip_phif_0[ip]<<"\nkappa_0 = "<<ip_kappa_0[ip]<<"\na0_0 = ["<<ip_a0_0[ip](0)<<","<<ip_a0_0[ip](1)<<"]\nlamdaP_0 = ["<<ip_lamdaP_0[ip](0)<<","<<ip_lamdaP_0[ip](1)<<"]\n";
		//localWoundProblem(dt,local_parameters,c,rho,CC,ip_phif_0[ip],ip_a0_0[ip],ip_kappa_0[ip],ip_lamdaP_0[ip],ip_phif[ip],ip_a0[ip],ip_kappa[ip],ip_lamdaP[ip],dThetadCC,dThetadrho,dThetadc);
		//
		// rename variables to make it easier
		double phif_0 = ip_phif_0[ip];
		Vector2d a0_0 = ip_a0_0[ip];
		double kappa_0 = ip_kappa_0[ip];
		Vector2d lamdaP_0 = ip_lamdaP_0[ip];
		double phif = ip_phif[ip];
		Vector2d a0 = ip_a0[ip];
		double kappa = ip_kappa[ip];
		Vector2d lamdaP = ip_lamdaP[ip];
		double lamdaP_a_0 = lamdaP_0(0);
		double lamdaP_s_0 = lamdaP_0(1);
		double lamdaP_a = lamdaP(0);
		double lamdaP_s = lamdaP(1);
		//std::cout<<"Local variables after update:\nphif0 = "<<phif_0<<",	phif = "<<phif<<"\nkappa_0 = "<<kappa_0<<",	kappa = "<<kappa<<"\na0_0 = ["<<a0_0(0)<<","<<a0_0(1)<<"],	a0 = ["<<a0(0)<<","<<a0(1)<<"]\nlamdaP_0 = ["<<lamdaP_0(0)<<","<<lamdaP_0(1)<<"],	lamdaP = ["<<lamdaP(0)<<","<<lamdaP(1)<<"]\n";
		// make sure the update preserved length
		double norma0 = sqrt(a0.dot(a0));
		if(fabs(norma0-1.)>0.001){std::cout<<"update did not preserve unit length of a0\n";}
		ip_a0[ip] = a0/(sqrt(a0.dot(a0)));
		a0 = a0/(sqrt(a0.dot(a0)));
		//
		// unpack the derivatives wrt CC
		// remember dThetatCC: 4 phi, 4 a0x, 4 a0y, 4 kappa, 4 lamdaPa, 4 lamdaPs
		Matrix2d dlamdaP_adCC; dlamdaP_adCC.setZero();
		dlamdaP_adCC(0,0) = dThetadCC(16); 
		dlamdaP_adCC(0,1) = dThetadCC(17);
		dlamdaP_adCC(1,0) = dThetadCC(18);
		dlamdaP_adCC(1,1) = dThetadCC(19);
		Matrix2d dlamdaP_sdCC; dlamdaP_sdCC.setZero();
		dlamdaP_sdCC(0,0) = dThetadCC(20);
		dlamdaP_sdCC(0,1) = dThetadCC(21);
		dlamdaP_sdCC(1,0) = dThetadCC(22);
		dlamdaP_sdCC(1,1) = dThetadCC(23);
		Matrix2d dphifdCC; dphifdCC.setZero();
		dphifdCC(0,0) = dThetadCC(0);
		dphifdCC(0,1) = dThetadCC(1);
		dphifdCC(1,0) = dThetadCC(2);
		dphifdCC(1,1) = dThetadCC(3);
		Matrix2d da0xdCC;da0xdCC.setZero();
		da0xdCC(0,0) = dThetadCC(4);
		da0xdCC(0,1) = dThetadCC(5);
		da0xdCC(1,0) = dThetadCC(6);
		da0xdCC(1,1) = dThetadCC(7);
		Matrix2d da0ydCC;da0ydCC.setZero();
		da0ydCC(0,0) = dThetadCC(8);
		da0ydCC(0,1) = dThetadCC(9);
		da0ydCC(1,0) = dThetadCC(10);
		da0ydCC(1,1) = dThetadCC(11);
		Matrix2d dkappadCC; dkappadCC.setZero();
		dkappadCC(0,0) = dThetadCC(12);
		dkappadCC(0,1) = dThetadCC(13);
		dkappadCC(1,0) = dThetadCC(14);
		dkappadCC(1,1) = dThetadCC(15);
		// unpack the derivatives wrt rho
		double dphifdrho = dThetadrho(0);
		double da0xdrho  = dThetadrho(1);
		double da0ydrho  = dThetadrho(2);
		double dkappadrho  = dThetadrho(3);
		double dlamdaP_adrho  = dThetadrho(4);
		double dlamdaP_sdrho  = dThetadrho(5);
		// unpack the derivatives wrt c
		double dphifdc = dThetadc(0);
		double da0xdc  = dThetadc(1);
		double da0ydc  = dThetadc(2);
		double dkappadc  = dThetadc(3);
		double dlamdaP_adc  = dThetadc(4);
		double dlamdaP_sdc  = dThetadc(5);
		//
		//---------------------------------//
		
		
		
		//---------------------------------//
		// CALCULATE SOURCE AND FLUX
		//
		// Update kinematics
		CCinv = CC.inverse();
		// re-compute basis a0, s0
		Matrix2d Rot90;Rot90<<0.,-1.,1.,0.;
		Vector2d s0 = Rot90*a0;
		// fiber tensor in the reference
		Matrix2d a0a0 = a0*a0.transpose();
		Matrix2d s0s0 = s0*s0.transpose();
		Matrix2d A0 = kappa*Identity + (1-2.*kappa)*a0a0;
		Vector2d a = FF*a0;
		Matrix2d A = kappa*FF*FF.transpose() + (1.-2.0*kappa)*a*a.transpose();
		double trA = A(0,0) + A(1,1);
		Matrix2d hat_A = A/trA;
		// recompute split
		Matrix2d FFg = lamdaP_a*(a0a0) + lamdaP_s*(s0s0);
		double thetaP = lamdaP_a*lamdaP_s;
		Matrix2d FFginv = (1./lamdaP_a)*(a0a0) + (1./lamdaP_s)*(s0s0);
		Matrix2d FFe = FF*FFginv;
		//std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
		// elastic strain
		Matrix2d CCe = FFe.transpose()*FFe;
		// invariant of the elastic strain
		double I1e = CCe(0,0) + CCe(1,1);
		double I4e = a0.dot(CCe*a0);
		// calculate the normal stretch
		double thetaE = sqrt(CCe.determinant());
		double theta = thetaE*thetaP;
		//std::cout<<"split of the determinants. theta = thetaE*thetaB = "<<theta<<" = "<<thetaE<<"*"<<thetaP<<"\n";
		double lamda_N = 1./thetaE;
		double I4tot = a0.dot(CC*a0);
		// Second Piola Kirchhoff stress tensor
		// passive elastic
		double Psif = (kf/(2.*k2))*(exp( k2*pow((kappa*I1e + (1-2*kappa)*I4e -1),2))-1);
		double Psif1 = 2*k2*kappa*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif;
		double Psif4 = 2*k2*(1-2*kappa)*(kappa*I1e + (1-2*kappa)*I4e -1)*Psif;
		Matrix2d SSe_pas = k0*Identity + phif*(Psif1*Identity + Psif4*a0a0);
		// pull back to the reference
		Matrix2d SS_pas = thetaP*FFginv*SSe_pas*FFginv;
		// magnitude from systems bio
		double traction_act = (t_rho + t_rho_c*c/(K_t_c + c))*rho;
		Matrix2d SS_act = (thetaP*traction_act*phif/trA)*A0;
		// total stress, don't forget the pressure
		double pressure = -k0*lamda_N*lamda_N;
		Matrix2d SS_pres = pressure*thetaP*CCinv;
		//std::cout<<"stresses.\nSSpas\n"<<SS_pas<<"\nSS_act\n"<<SS_act<<"\nSS_pres"<<SS_pres<<"\n";
		Matrix2d SS = SS_pas + SS_act + SS_pres;
		Vector3d SS_voigt = Vector3d(SS(0,0),SS(1,1),SS(0,1));
		// Flux and Source terms for the rho and the C
		Vector2d Q_rho = -D_rhorho*CCinv*Grad_rho - D_rhoc*rho*CCinv*Grad_c;
		Vector2d Q_c = -D_cc*CCinv*Grad_c;
		// mechanosensing 
		double He = 1./(1.+exp(-gamma_c_thetaE*(thetaE + theta_phy)));
		double S_rho = (p_rho + p_rho_c*c/(K_rho_c+c)+p_rho_theta*He)*(1-rho/K_rho_rho)*rho - d_rho*rho;
		// heviside function for elastic response of the chemical
		double S_c = (p_c_rho*c+ p_c_thetaE*He)*(rho/(K_c_c+c)) - d_c*c;
		//std::cout<<"flux of celss, Q _rho\n"<<Q_rho<<"\n";
		//std::cout<<"source of cells, S_rho: "<<S_rho<<"\n";
		//std::cout<<"flux of chemical, Q _c\n"<<Q_c<<"\n";
		//std::cout<<"source of chemical, S_c: "<<S_c<<"\n";
		//---------------------------------//
		
		
		
		//---------------------------------//
		// ADD TO THE RESIDUAL
		//
		Matrix2d deltaFF,deltaCC;
		Vector3d deltaCC_voigt;
        for(int nodei=0;nodei<n_nodes;nodei++){
            for(int coordi=0;coordi<2;coordi++){
                // alternatively, define the deltaCC
                deltaFF = Ebasis[coordi]*Grad_R[nodei].transpose();
                deltaCC = deltaFF.transpose()*FF + FF.transpose()*deltaFF;
                deltaCC_voigt = Vector3d(deltaCC(0,0),deltaCC(1,1),2.*deltaCC(1,0));
                Re_x(nodei*2+coordi) += Jac*SS_voigt.dot(deltaCC_voigt);
            }
            // Element residuals for rho and c
            Re_rho(nodei) += Jac*(((rho-rho_0)/dt - S_rho)*R[nodei] - Grad_R[nodei].dot(Q_rho));
            Re_c(nodei) += Jac*(((c-c_0)/dt - S_c)*R[nodei] - Grad_R[nodei].dot(Q_c));
        }
		//
		//---------------------------------//
		
	}
	
	
}
		
