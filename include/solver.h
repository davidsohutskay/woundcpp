/*
	Pre and Post processing functions
	Solver functions
	Struct and classes for the problem definition
*/

#ifndef solver_h
#define solver_h

#include <vector>
#include <map>
#include <Eigen/Dense> // most of the vector functions I will need inside of an element
using namespace Eigen;

// Structure for the problem
struct tissue{

	// connectivity (topology)
	int n_node;
	int n_quadri;
	int n_IP;
	std::vector<std::vector<int> > LineQuadri;
	std::vector<int> boundaryNodes;
    std::vector<int> boundary_flag;

	// reference geometry
	//
	// nodal values
	std::vector<Vector2d> node_X;
	std::vector<double> node_rho_0;
	std::vector<double> node_c_0;
	//
	// integration point values
	// order by element then by integration point of the element
	std::vector<double> ip_phif_0;
	std::vector<Vector2d> ip_a0_0;
	std::vector<double> ip_kappa_0;
	std::vector<Vector2d> ip_lamdaP_0;
	
	// deformed geometry
	//
	// nodal values
	std::vector<Vector2d> node_x;
	std::vector<double> node_rho;
	std::vector<double> node_c;
	//
	// integration point values
	std::vector<double> ip_phif;
	std::vector<Vector2d> ip_a0;
	std::vector<double> ip_kappa;
	std::vector<Vector2d> ip_lamdaP;
    std::vector<Vector2d> ip_lamdaE;
    std::vector<Matrix2d> ip_strain;
    std::vector<Matrix2d> ip_stress;

	// boundary conditions
	//
	// essential boundary conditions for displacement
	std::map<int,double>  eBC_x;
	// essential boundary conditions for concentrations
	std::map<int,double>  eBC_rho;
	std::map<int,double>  eBC_c;
	//
	// traction boundary conditions for displacements
	std::map<int,double> nBC_x;
	// traction boundary conditions for concentrations
	std::map<int,double> nBC_rho;
	std::map<int,double> nBC_c;
	
	// degree of freedom maps
	//
	// displacements
	std::vector< int > dof_fwd_map_x;
	//
	// concentrations
	std::vector< int > dof_fwd_map_rho;
	std::vector< int > dof_fwd_map_c;
	
	// all dof inverse map
	std::vector< std::vector<int> > dof_inv_map;
	
	// material parameters
	std::vector<double> global_parameters;
	std::vector<double> local_parameters;
	
	// internal element constant (jacobians at IP)
	std::vector<std::vector<Matrix2d> > elem_jac_IP;
	
	// parameters for the simulation
	int n_dof;
	double time_final;
	double time_step;
	double time;
	double tol;
	int max_iter;
	
};


//-------------------------------------------------//
// PRE PROCESS
//-------------------------------------------------//

//----------------------------//
// FILL DOF
//----------------------------//
//
// now I have the mesh and 
// =>> Somehow filled in the essential boundary conditions
// so I create the dof maps.  
void fillDOFmap(tissue &myTissue);

//----------------------------//
// EVAL JAC
//----------------------------//
//
// eval the jacobians that I use later on in the element subroutines
void evalElemJacobians(tissue &myTissue);

//-------------------------------------------------//
// SOLVER
//-------------------------------------------------//

//----------------------------//
// SPARSE SOLVER
//----------------------------//
//
void sparseWoundSolver(tissue &myTissue, std::string filename, int save_freq,const std::vector<int> &save_node,const std::vector<int> &save_ip);

// dense solver in previous version of wound.cpp
// void denseWoundSolver(tissue &myTissue, std::string filename, int save_freq);

//-------------------------------------------------//
// IO
//-------------------------------------------------//

//----------------------------//
// READ ABAQUS
//----------------------------//
//
// read in an Abaqus input file with mesh and fill in structures
void readAbaqusInput(const char* filename, tissue &myTissue);

//----------------------------//
// REAd OWN FILE
//----------------------------//
//
tissue readTissue(const char* filename);

//----------------------------//
// WRITE PARAVIEW
//----------------------------//
//
// write the paraview file with 
//	RHO, C, PHI, THETA_B at the nodes
void writeParaview(tissue &myTissue, const char* filename, const char* filename2);

//----------------------------//
// WRITE OWN FILE
//----------------------------//
//
void writeTissue(tissue &myTissue, const char* filename,double time);

//----------------------------//
// WRITE NODE
//----------------------------//
//
// write the node information
// DEFORMED x, RHO, C
void writeNode(tissue &myTissue,const char* filename,int nodei,double time);

//----------------------------//
// WRITE IP
//----------------------------//
//
// write integration point information
// just ip variables
// PHI A0 KAPPA LAMBDA 
void writeIP(tissue &myTissue,const char* filename,int ipi,double time);

//----------------------------//
// WRITE ELEMENT
//----------------------------//
//
// write an element information to a text file. write the average
// of variables at the center in the following order
//	PHI A0 KAPPA LAMDA_B RHO C
void writeElement(tissue &myTissue,const char* filename,int elemi,double time);

#endif