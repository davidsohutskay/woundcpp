
/*

RESULTS
circular wound problem.

Read a quad mesh defined by myself.
Then apply boundary conditions.
Solve.

*/

//#define EIGEN_USE_MKL_ALL
#include <omp.h>
#include "wound.h"
#include "solver.h"
#include "myMeshGenerator.h"

#include <boost/algorithm/string.hpp>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <math.h>
#include <string>
#include <time.h>

#include <Eigen/Dense>
using namespace Eigen;

double frand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


int main(int argc, char *argv[])
{
    Eigen::initParallel();
    std::cout<<"\nResults 3: full domain simulations \n";
    srand (time(NULL));

//    // Open file
//    std::string readfilename = "samples_to_run.txt";
//    std::ifstream readfile;
//    readfile.open(readfilename.c_str());
//    // Check if object is valid
//    if(!readfile)
//    {
//        std::cerr << "Cannot open the file!"<<std::endl;
//        return 1;
//    }
//
//    // Create empty vectors
//    std::vector<double> kv_vector;
//    std::vector<double> k0_vector;
//
//    // Read the next line from File untill it reaches the end.
//    for (std::string line; std::getline(readfile, line); ){
//        // Split the string into numbers
//        std::vector<std::string> strs;
//        boost::split(strs,line,boost::is_any_of(" "));
//
//        // Print the line
//        std::cout << std::stod(strs[0]) << std::endl;
//        std::cout << std::stod(strs[1]) << std::endl;
//
//        // Push to the empty vectors we made
//        kv_vector.push_back(std::stod(strs[0]));
//        k0_vector.push_back(std::stod(strs[1]));
//    }
//
//    // Check the length for samples
//    std::cout << "\n" << kv_vector.size() << "\n";
//    std::cout << "\n" << kv_vector.size() << "\n";
//
//    // Run each set of parameters separately in the loop
//    readfile.close();

    for(int sample = 0; sample < 1; sample++){
        //---------------------------------//
        // GLOBAL PARAMETERS
        //
        // for nondimensionalization
        // for normalization
        double t_max = 7*24*2;
        double x_length = 37.5;
        double rho_phys = 1000.; // [cells/mm^3]
        double stress_phys = 0.005;
        double c_max = 1.0e-4; // [g/mm3] from tgf beta review, 5e-5g/mm3 was good for tissues
        //
        double k0 = 0.0511; // neo hookean for skin, used previously, in MPa
        double kf = 0.015; // stiffness of collagen in MPa, from previous paper
        double k2 = 0.048; // nonlinear exponential coefficient, non-dimensional
        double t_rho = 0.005; // 0.0045 force of fibroblasts in MPa, this is per cell. so, in an average sense this is the production by the natural density
        double t_rho_c = 10*t_rho; // 0.045 force of myofibroblasts enhanced by chemical, I'm assuming normalized chemical, otherwise I'd have to add a normalizing constant
        double K_t = 0.3; // Saturation of mechanical force by collagen
        double K_t_c = 1/10.; // saturation of chemical on force. this can be calculated from steady state
        double D_rhorho = 0.0833*t_max/(x_length*x_length); // diffusion of cells in [mm^2/hour], not normalized
        double D_rhoc = (-1.66e-12/c_max)*t_max/(x_length*x_length); // diffusion of chemotactic gradient, an order of magnitude greater than random walk [mm^2/hour], not normalized
        double D_cc = 0.01208*t_max/(x_length*x_length); // 0.15 diffusion of chemical TGF, not normalized.
        double p_rho = 0.034/2*t_max; // in 1/hour production of fibroblasts naturally, proliferation rate, not normalized, based on data of doubling rate from commercial use
        double p_rho_c = p_rho/2; // production enhanced by the chem, if the chemical is normalized, then suggest two fold,
        double p_rho_theta = p_rho/2; // enhanced production by theta
        double K_rho_c= 1/10.; // saturation of cell proliferation by chemical, this one is definitely not crucial, just has to be small enough <cmax
        double K_rho_rho = 10000/rho_phys; // saturation of cell by cell, from steady state
        double d_rho = p_rho*(1-1/K_rho_rho); // decay of cells, keeps equilibrium
        double vartheta_e = 2.0; // physiological state of area stretch
        double gamma_theta = 5.0; // sensitivity of heviside function
        double p_c_rho = 90.0e-16*t_max/c_max/c_max;// production of c by cells in g/cells/h
        double p_c_thetaE = 300.0e-16*t_max/c_max/c_max; // coupling of elastic and chemical, three fold
        double K_c_c = 1./c_max;// saturation of chem by chem, from steady state
        double d_c = 0.01*t_max; // 0.01 decay of chemical in 1/hours
        //---------------------------------//
        std::vector<double> global_parameters = {k0,kf,k2,t_rho,t_rho_c,K_t,K_t_c,D_rhorho,D_rhoc,D_cc,p_rho,p_rho_c,p_rho_theta,K_rho_c,K_rho_rho,d_rho,vartheta_e,gamma_theta,p_c_rho,p_c_thetaE,K_c_c,d_c};

        //---------------------------------//
        // LOCAL PARAMETERS
        //
        // collagen fraction
        double p_phi = 0.002*t_max; // production by fibroblasts, natural rate in percent/hour, 5% per day
        double p_phi_c = p_phi; // production up-regulation, weighted by C and rho
        double p_phi_theta = p_phi; // mechanosensing upregulation. no need to normalize by Hmax since Hmax = 1
        double K_phi_c = 0.0001/c_max; // saturation of C effect on deposition. RANDOM?
        double d_phi = 0.000970*t_max; // rate of degradation, in the order of the wound process, 100 percent in one year for wound, means 0.000116 effective per hour means degradation = 0.002 - 0.000116
        double d_phi_rho_c = 0.5*d_phi; // 0.000194; // degradation coupled to chemical and cell density to maintain phi equilibrium
        double K_phi_rho = p_phi/d_phi - 1; // saturation of collagen fraction itself, from steady state
        //
        //
        // fiber alignment
        double tau_omega = 1000000*10./(K_phi_rho+1); // time constant for angular reorientation, think 100 percent in one year
        //
        // dispersion parameter
        double tau_kappa = 1000000*1./(K_phi_rho+1); // time constant, on the order of a year
        double gamma_kappa = 5.; // exponent of the principal stretch ratio
        //
        // permanent contracture/growth
        double tau_lamdaP_a = 1000000*0.05/(K_phi_rho+1); // time constant for direction a, on the order of a year
        double tau_lamdaP_s = 1000000*0.05/(K_phi_rho+1); // time constant for direction s, on the order of a year
        //
        // solution parameters
        double tol_local = 1e-5; // local tolerance
        double max_local_iter = 35; // max local iter (implicit) or time step ratio (explicit)
        //
        std::vector<double> local_parameters = {p_phi,p_phi_c,p_phi_theta,K_phi_c,K_phi_rho,d_phi,d_phi_rho_c,tau_omega,tau_kappa,gamma_kappa,tau_lamdaP_a,tau_lamdaP_s,gamma_theta,vartheta_e,tol_local,max_local_iter};
        //
        // other local stuff
        double PIE = 3.14159;
        //---------------------------------//

        //---------------------------------//
        // values for the wound
        double rho_wound = 0; // [cells/mm^3]
        double c_wound = 1.0; //1.0e-4;
        double phif0_wound= 0.01;
        double kappa0_wound = 0.4;
        double a0x = frand(0,1.);
        double a0y = sqrt(1-a0x*a0x);
        Vector2d a0_wound;a0_wound<<a0x,a0y;
        Vector2d lamda0_wound;lamda0_wound<<1.,1.;
        //---------------------------------//


        //---------------------------------//
        // values for the healthy
        double rho_healthy = 1.0; //1000; // [cells/mm^3]
        double c_healthy = 0.0;
        double phif0_healthy= 1.;
        double kappa0_healthy = 0.4;
        Vector2d a0_healthy;a0_healthy<<1.0,0.0;
        a0_wound = a0_healthy;
        Vector2d lamda0_healthy;lamda0_healthy<<1.,1.;
        //---------------------------------//


        //---------------------------------//
        // create mesh (only nodes and elements)
        std::cout<<"Going to create the mesh\n";
        //std::vector<double> rectangleDimensions = {0.0,75.0,0.0,75.0};
        std::vector<double> rectangleDimensions = {-1.0,1.0,-1.0,1.0};
        std::vector<int> meshResolution = {40,40};
        QuadMesh myMesh = myRectangleMesh(rectangleDimensions, meshResolution);
        //QuadMesh myMesh = myQuadraticRectangleMesh(rectangleDimensions, meshResolution);
        //QuadMesh myMesh = myMultiBlockMesh(rectangleDimensions, meshResolution);
        //std::string mesh_filename = "COMSOL_wound_mesh_2D_75mmx15mm.mphtxt";
        //QuadMesh myMesh = readCOMSOLInput(mesh_filename,rectangleDimensions,meshResolution);
        //QuadMesh myMesh = readParaviewInput(mesh_filename,rectangleDimensions,meshResolution);

        std::cout<<"Created the mesh with "<<myMesh.n_nodes<<" nodes and "<<myMesh.boundary_flag.size()<<" boundaries and "<<myMesh.n_elements<<" elements\n";
        // print the mesh
        std::cout<<"nodes\n";
        for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
            std::cout<<myMesh.nodes[nodei](0)<<","<<myMesh.nodes[nodei](1)<<"\n";
        }
        std::cout<<"elements\n";
        for(int elemi=0;elemi<myMesh.n_elements;elemi++){
            for(int nodei=0;nodei<myMesh.elements[elemi].size();nodei++){
                std::cout<<myMesh.elements[elemi][nodei]<<" ";
            }
            std::cout<<"\n";
        }
        std::cout<<"boundary\n";
        for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
            std::cout<<myMesh.boundary_flag[nodei]<<"\n";
        }
        // create the other fields needed in the tissue struct.
        int n_elem = myMesh.n_elements;
        int n_node = myMesh.n_nodes;
        int elem_size = myMesh.elements[0].size();
        // integration points
        std::vector<Vector3d> IP = LineQuadriIP();
        int IP_size = IP.size();
        std::cout << IP_size;
        //
        // global fields rho and c initial conditions
        std::vector<double> node_rho0(n_node,rho_healthy);
        std::vector<double> node_c0 (n_node,c_healthy);
        //
        // values at the integration points
        std::vector<double> ip_phi0(n_elem*IP_size,phif0_healthy);
        std::vector<Vector2d> ip_a00(n_elem*IP_size,a0_healthy);
        std::vector<double> ip_kappa0(n_elem*IP_size,kappa0_healthy);
        std::vector<Vector2d> ip_lamda0(n_elem*IP_size,lamda0_healthy);
        //
        // define ellipse
        double tol_boundary = 0.01; //0.1;
        double x_center = 0.0; //37.5;
        double y_center = 0.0; //37.5;
        double x_axis = 0.2; //7.5 + tol_boundary;
        double y_axis = 0.2; //7.5 + tol_boundary;
        double alpha_ellipse = 0.;

        //std::vector<double> ellipse = {x_center, y_center, x_axis, y_axis, alpha_ellipse};
        //conformMesh2Ellipse(myMesh, ellipse);

        // boundary conditions and definition of the wound
        std::map<int,double> eBC_x;
        std::map<int,double> eBC_rho;
        std::map<int,double> eBC_c;
        for(int nodei=0;nodei<n_node;nodei++){
            double x_coord = myMesh.nodes[nodei](0);
            double y_coord = myMesh.nodes[nodei](1);
            // check
            if(myMesh.boundary_flag[nodei]!=0){
                // insert the boundary condition for displacement
                std::cout<<"fixing node "<<nodei<<"\n";
                eBC_x.insert ( std::pair<int,double>(nodei*2+0,myMesh.nodes[nodei](0)) ); // x coordinate
                eBC_x.insert ( std::pair<int,double>(nodei*2+1,myMesh.nodes[nodei](1)) ); // y coordinate
                // insert the boundary condition for rho
                eBC_rho.insert ( std::pair<int,double>(nodei,rho_healthy) );
                // insert the boundary condition for c
                eBC_c.insert   ( std::pair<int,double>(nodei,c_healthy) );
            }
            // check if it is in the center of the wound
            double check_ellipse = pow((x_coord-x_center)*cos(alpha_ellipse)+(y_coord-y_center)*sin(alpha_ellipse),2)/(x_axis*x_axis) +\
						pow((x_coord-x_center)*sin(alpha_ellipse)+(y_coord-y_center)*cos(alpha_ellipse),2)/(y_axis*y_axis) ;
            double distance = sqrt(pow((x_coord-x_center),2) + pow((y_coord-y_center),2));
            double scaled_distance = 5.*(distance - 0.1);
            double smoother_step = (pow(scaled_distance,3) * (6. * pow(scaled_distance,2) - 15. * scaled_distance + 10.));
            //std::cout << distance << "\n";
            if(distance < 0.1){ //check_ellipse<=1 distance <= x_axis/2
                // inside
                std::cout<<"wound node "<<nodei<<"\n";
                node_rho0[nodei] = rho_wound;
                node_c0[nodei] = c_wound;
            }
            else if(distance < 0.3){
                // transition
                std::cout<<"transition node "<<nodei<<"\n";
                node_rho0[nodei] = rho_healthy*smoother_step;
                node_c0[nodei] = c_wound*(1 - smoother_step);
            }
            // else we are outside and already set to healthy values
        }

        for(int elemi=0;elemi<n_elem;elemi++){
            for(int ip=0;ip<IP_size;ip++)
            {
                double xi = IP[ip](0);
                double eta = IP[ip](1);
                // weight of the integration point
                double wip = IP[ip](2);
                std::vector<double> R;
                if(elem_size == 4){
                    R = evalShapeFunctionsR(xi,eta);
                }
                else if(elem_size == 8){
                    R = evalShapeFunctionsQuadraticR(xi,eta);
                }
                else{
                    throw std::runtime_error("Wrong number of nodes in element!");
                }

                Vector2d X_IP;X_IP.setZero();
                for(int nodej=0;nodej<elem_size;nodej++){
                    X_IP += R[nodej]*myMesh.nodes[myMesh.elements[elemi][nodej]];
                }

                //std::cout<<" IP node " << ip << " reference coordinates: " <<xi<< " " <<eta << " "<<"\n";
                //std::cout<<"Element " << elemi <<" IP node " << ip << " coordinates: " <<X_IP(0)<< " " <<X_IP(1) << " "<<"\n";
                double check_ellipse_ip = pow((X_IP(0)-x_center)*cos(alpha_ellipse)+(X_IP(1)-y_center)*sin(alpha_ellipse),2)/(x_axis*x_axis) +\
                        pow((X_IP(0)-x_center)*sin(alpha_ellipse)+(X_IP(1)-y_center)*cos(alpha_ellipse),2)/(y_axis*y_axis) ;
                double distance = sqrt(pow((X_IP(0)-x_center),2) + pow((X_IP(1)-y_center),2));
                double scaled_distance = 5.*(distance - 0.1);
                double smoother_step = (pow(scaled_distance,3) * (6. * pow(scaled_distance,2) - 15. * scaled_distance + 10.));
                if(distance < 0.1){ //check_ellipse<=1 distance <= x_axis/2
                    // wound
                    std::cout<<"IP wound node: "<<IP_size*elemi+ip<<"\n";
                    ip_phi0[elemi*IP_size+ip] = phif0_wound;
                    ip_a00[elemi*IP_size+ip] = a0_wound;
                    ip_kappa0[elemi*IP_size+ip] = kappa0_wound;
                    ip_lamda0[elemi*IP_size+ip] = lamda0_wound;
                }
                else if(distance < 0.3){
                    // transition
                    std::cout<<"IP transition node: "<<IP_size*elemi+ip<<"\n";
                    ip_phi0[elemi*IP_size+ip] = phif0_healthy*smoother_step;
                    ip_a00[elemi*IP_size+ip] = a0_wound;
                    ip_kappa0[elemi*IP_size+ip] = kappa0_wound;
                    ip_lamda0[elemi*IP_size+ip] = lamda0_wound;
                }
                // else we are outside and already set to healthy values
            }
        }

        // no neumann boundary conditions.
        std::map<int,double> nBC_x;
        std::map<int,double> nBC_rho;
        std::map<int,double> nBC_c;

        // initialize my tissue
        tissue myTissue;
        // connectivity
        myTissue.LineQuadri = myMesh.elements;
        // parameters
        myTissue.global_parameters = global_parameters;
        myTissue.local_parameters = local_parameters;
        myTissue.boundary_flag = myMesh.boundary_flag;
        //
        myTissue.node_X = myMesh.nodes;
        myTissue.node_x = myMesh.nodes;
        myTissue.node_rho_0 = node_rho0;
        myTissue.node_rho = node_rho0;
        myTissue.node_c_0 = node_c0;
        myTissue.node_c = node_c0;
        myTissue.ip_phif_0 = ip_phi0;
        myTissue.ip_phif = ip_phi0;
        myTissue.ip_a0_0 = ip_a00;
        myTissue.ip_a0 = ip_a00;
        myTissue.ip_kappa_0 = ip_kappa0;
        myTissue.ip_kappa = ip_kappa0;
        myTissue.ip_lamdaP_0 = ip_lamda0;
        myTissue.ip_lamdaP = ip_lamda0;
        myTissue.ip_lamdaE = ip_lamda0;
        std::vector<Matrix2d> ip_strain(myMesh.n_elements*IP_size,Matrix2d::Identity(2,2));
        std::vector<Matrix2d> ip_stress(myMesh.n_elements*IP_size,Matrix2d::Zero(2,2));
        myTissue.ip_strain = ip_strain;
        myTissue.ip_stress = ip_stress;
        //
        myTissue.eBC_x = eBC_x;
        myTissue.eBC_rho = eBC_rho;
        myTissue.eBC_c = eBC_c;
        myTissue.nBC_x = nBC_x;
        myTissue.nBC_rho = nBC_rho;
        myTissue.nBC_c = nBC_c;
        myTissue.time_final = 1.0; //24*15;
        myTissue.time_step = 1.0/40/175; //0.0002;
        myTissue.tol = 1e-8;
        myTissue.max_iter = 25;
        myTissue.n_node = myMesh.n_nodes;
        myTissue.n_quadri = myMesh.n_elements;
        myTissue.n_IP = IP_size*myMesh.n_elements;
        //
        std::cout<<"filling dofs...\n";
        fillDOFmap(myTissue);
        std::cout<<"going to eval jacobians...\n";
        evalElemJacobians(myTissue);
        //

        //print out the Jacobians
        std::cout<<"element jacobians\nJacobians= ";
        std::cout<<myTissue.elem_jac_IP.size()<<"\n";
        for(int i=0;i<myTissue.elem_jac_IP.size();i++){
            std::cout<<"element: "<<i<<"\n";
            for(int j=0;j<IP_size;j++){
                std::cout<<"ip; "<<j<<"\n"<<myTissue.elem_jac_IP[i][j]<<"\n";
            }
        }
        // print out the forward dof map
        std::cout<<"Total :"<<myTissue.n_dof<<" dof\n";
        for(int i=0;i<myTissue.dof_fwd_map_x.size();i++){
            std::cout<<"x node*2+coord: "<<i<<", dof: "<<myTissue.dof_fwd_map_x[i]<<"\n";
        }
        for(int i=0;i<myTissue.dof_fwd_map_rho.size();i++){
            std::cout<<"rho node: "<<i<<", dof: "<<myTissue.dof_fwd_map_rho[i]<<"\n";
        }
        for(int i=0;i<myTissue.dof_fwd_map_c.size();i++){
            std::cout<<"c node: "<<i<<", dof: "<<myTissue.dof_fwd_map_c[i]<<"\n";
        }
        //
        //
        std::cout<<"going to start solver\n";
        // save a node and an integration point to a file
        std::vector<int> save_node;save_node.clear();
        std::vector<int> save_ip;save_ip.clear();


        std::stringstream ss;
        ss << sample;
        std::string filename = "sample_" + ss.str() + "_paraview_ouput_";


        //----------------------------------------------------------//
        // SOLVE
        sparseWoundSolver(myTissue, filename, 175,save_node,save_ip);
        //----------------------------------------------------------//
    }

    return 0;
}