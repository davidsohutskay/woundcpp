// mesh generator for a simple quadrilateral domain

// no need to have a header for this. I will just have a function

//#define EIGEN_USE_MKL_ALL
#include <omp.h>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include "myMeshGenerator.h"
#include <iostream>
// a simple quad mesh generation, really stupid one
QuadMesh myRectangleMesh(const std::vector<double> &rectangleDimensions, const std::vector<int> &meshResolution)
{
    Eigen::initParallel();
	// number of points in the x and y directions
	int n_x_points = meshResolution[0];
	int n_y_points = meshResolution[1];

	// dimensions of the mesh in the x and y direction
	double x_init = rectangleDimensions[0];
	double x_final = rectangleDimensions[1];
	double y_init = rectangleDimensions[2];
	double y_final = rectangleDimensions[3];
	int n_nodes = n_x_points*n_y_points;
	int n_elems = (n_x_points-1)*(n_y_points-1);
	std::cout<<"Going to create a mesh of "<<n_nodes<<" nodes and "<<n_elems<<" elements\n";
	std::cout<<"X0 = "<<x_init<<", XF = "<<x_final<<", Y0 = "<<y_init<<", YF = "<<y_final<<"\n";
	std::vector<Vector2d> NODES(n_nodes,Vector2d(0.,0.));
	std::vector<int> elem0 = {0,0,0,0};
	std::vector<std::vector<int> > ELEMENTS(n_elems,elem0);
	std::vector<int> BOUNDARIES(n_x_points*n_y_points,0);
	// create the nodes row by row
	for(int i=0;i<n_x_points;i++){
		for(int j=0;j<n_y_points;j++){
			double x_coord = x_init+ i*(x_final-x_init)/(n_x_points-1);
			double y_coord = y_init+ j*(y_final-y_init)/(n_y_points-1);
			NODES[j*n_x_points+i](0) = x_coord;
			NODES[j*n_x_points+i](1) = y_coord;
			if(i==0 || i==n_x_points-1 || j==0 || j==n_y_points-1){
				BOUNDARIES[j*n_x_points+i]=1;
			}
		}
	}
	std::cout<<"... filled nodes...\n";
	// create the connectivity
	for(int i=0;i<n_x_points-1;i++){
		for(int j=0;j<n_y_points-1;j++){		
			ELEMENTS[j*(n_x_points-1)+i][0] = j*n_x_points+i;
			ELEMENTS[j*(n_x_points-1)+i][1] = j*n_x_points+i+1;
			ELEMENTS[j*(n_x_points-1)+i][2] = (j+1)*n_x_points+i+1;
			ELEMENTS[j*(n_x_points-1)+i][3] = (j+1)*n_x_points+i;
		}
	}
	QuadMesh myMesh;
	myMesh.nodes = NODES;
	myMesh.elements = ELEMENTS;
	myMesh.boundary_flag = BOUNDARIES;
	myMesh.n_nodes = n_x_points*n_y_points;
	myMesh.n_elements = (n_x_points-1)*(n_y_points-1);
	return myMesh;
}

// a simple quad mesh generation, really stupid one
QuadMesh myQuadraticRectangleMesh(const std::vector<double> &rectangleDimensions, const std::vector<int> &meshResolution)
{
    Eigen::initParallel();
    // number of points in the x and y directions
    int n_x_points = meshResolution[0]*2+1;
    int n_y_points = meshResolution[1]*2+1;

    // dimensions of the mesh in the x and y direction
    double x_init = rectangleDimensions[0];
    double x_final = rectangleDimensions[1];
    double y_init = rectangleDimensions[2];
    double y_final = rectangleDimensions[3];
    //int n_nodes = n_x_points*n_y_points/2 + (n_x_points-1)*n_y_points/4;
    //int n_elems = (n_x_points-1)*(n_y_points-1);
    //std::cout<<"Going to create a mesh of "<<n_nodes<<" nodes and "<<n_elems<<" elements\n";
    std::cout<<"X0 = "<<x_init<<", XF = "<<x_final<<", Y0 = "<<y_init<<", YF = "<<y_final<<"\n";
    std::vector<Vector2d> NODES;
    //std::vector<int> elem0 = {0,0,0,0,0,0,0,0};
    std::vector<std::vector<int> > ELEMENTS;
    std::vector<int> BOUNDARIES;
    // create the nodes row by row
    for(int i=0;i<n_y_points;i++){
        for(int j=0;j<n_x_points;j++){
            if(!(i%2==1 && j%2==1)){
                double x_coord = x_init + j*(x_final-x_init)/(n_x_points-1);
                double y_coord = y_init + i*(y_final-y_init)/(n_y_points-1);
                Vector2d nodei = Vector2d(x_coord,y_coord);
                NODES.push_back(nodei);
                if(round(x_coord)==x_init){
                    BOUNDARIES.push_back(1);
                }
                else if(round(x_coord)==x_final){
                    BOUNDARIES.push_back(2);
                }
                else if(round(y_coord)==y_init){
                    BOUNDARIES.push_back(3);
                }
                else if(round(y_coord)==y_final){
                    BOUNDARIES.push_back(4);
                }
                else{
                    BOUNDARIES.push_back(0);
                }
            }
        }
    }
    std::cout<<"... filled nodes...\n";
    // create the connectivity
    for(int j=0;j<n_y_points-2;j+=2){
        for(int i=0;i<n_x_points-2;i+=2){
            // Push nodes.
            std::vector<int> elemi; elemi.clear();
            elemi.push_back(((j/2)*n_x_points) + (j/2)*((n_x_points+1)/2) + (i));
            elemi.push_back(((j/2)*n_x_points) + (j/2)*((n_x_points+1)/2) + (i+1));
            elemi.push_back(((j/2)*n_x_points) + (j/2)*((n_x_points+1)/2) + (i+2));
            elemi.push_back(((j/2+1)*n_x_points) + (j/2)*((n_x_points+1)/2) + (i/2+1) );
            elemi.push_back(((j/2+1)*n_x_points) + (j/2+1)*((n_x_points+1)/2) + (i+2));
            elemi.push_back(((j/2+1)*n_x_points) + (j/2+1)*((n_x_points+1)/2) + (i+1));
            elemi.push_back(((j/2+1)*n_x_points) + (j/2+1)*((n_x_points+1)/2) + (i));
            elemi.push_back(((j/2+1)*n_x_points) + (j/2)*((n_x_points+1)/2) + (i/2));
            //std::cout<<"\n";
            ELEMENTS.push_back(elemi);
        }
    }
    QuadMesh myMesh;
    myMesh.nodes = NODES;
    myMesh.elements = ELEMENTS;
    myMesh.boundary_flag = BOUNDARIES;
    myMesh.n_nodes = NODES.size();
    myMesh.n_elements = ELEMENTS.size();
    return myMesh;
}

//---------------------------------------//
// MULTIBLOCK MESH
//---------------------------------------//
//
QuadMesh myMultiBlockMesh(const std::vector<double> &hexDimensions, const std::vector<int> &meshResolution)
{
    // number of points in the x and y directions
    int n_x_points = meshResolution[0];
    int n_y_points = meshResolution[1];

    // dimensions of the mesh in the x, y, and z direction
    double x_init = hexDimensions[0];
    double x_final = hexDimensions[1];
    double y_init = hexDimensions[2];
    double y_final = hexDimensions[3];
    int n_nodes = n_x_points*n_y_points;;
    int n_elems = (n_x_points-1)*(n_y_points-1);
    std::cout<<"Going to create a mesh of "<<n_nodes<<" nodes and "<<n_elems<<" elements\n";
    std::cout<<"X0 = "<<x_init<<", XF = "<<x_final<<", Y0 = "<<y_init<<", YF = "<<y_final<< "\n";
    std::vector<Vector2d> NODES(n_nodes,Vector2d(0.,0.));
    std::vector<int> elem0 = {0,0,0,0};
    std::vector<std::vector<int> > ELEMENTS(n_elems,elem0);
    std::vector<int> BOUNDARIES(n_x_points*n_y_points,0);
    // create the nodes row by row
    for(int j=0;j<n_y_points;j++){
        for(int k=0;k<n_x_points;k++){
            // std::cout << "Node Iter" << i*n_x_points*n_y_points+j*n_x_points+k << "\n";
            double x_coord, y_coord;
            double inner = 1./3; // What size of the inner zone
            double inner_res = 1./2; // What percent of the lines should be in the inner zone
            double outer = (1-inner)/2.;
            double outer_res = (inner_res)/2;
            // x coord
            if(k<=(n_x_points-1)/4){
                x_coord = x_init + k*(outer)*(x_final-x_init)/(outer_res*(n_x_points-1));
            }
            else if(k<=3*(n_x_points-1)/4){
                x_coord = x_init + (outer)*(x_final-x_init) + (k - (n_x_points-1)/4.)*(inner)*(x_final-x_init)/((n_x_points-1)/2.);
            }
            else{
                x_coord = x_init + (outer+inner)*(x_final-x_init) + (k - 3*(n_x_points-1)/4.)*(outer)*(x_final-x_init)/((n_x_points-1)/4.);
            }
            // y coord
            if(j<=(n_y_points-1)/4){
                y_coord = y_init + j*(outer)*(y_final-y_init)/((n_y_points-1)/4.);
            }
            else if(j<=3*(n_y_points-1)/4){
                y_coord = y_init + (outer)*(y_final-y_init) + (j - (n_y_points-1)/4.)*(inner)*(y_final-y_init)/((n_y_points-1)/2.);
            }
            else{
                y_coord = y_init + (outer+inner)*(y_final-y_init) + (j - 3*(n_y_points-1)/4.)*(outer)*(y_final-y_init)/((n_y_points-1)/4.);
            }

            // std::cout << "X = " << x_coord << ", " << "Y = " << y_coord << ", Z = " << z_coord << " \n";
            NODES[j*n_x_points+k](0) = x_coord;
            NODES[j*n_x_points+k](1) = y_coord;

            //-------------//
            // ALTERNATIVE BOUNDARIES VERSION
            //-------------//
            // Be careful. Make sure you remember which nodes are part of which face
            // Apply Dirichlet BCs first, so the corner nodes become part of those faces
            //
            if(k==0){ // x = 0
                BOUNDARIES[j*n_x_points+k]=1;
            }
            else if (k==n_x_points-1){ // x = end
                BOUNDARIES[j*n_x_points+k]=1;
            }
            else if(j==0){ // y = 0
                BOUNDARIES[j*n_x_points+k]=1;
            }
            else if (j==n_y_points-1){ // y = end
                BOUNDARIES[j*n_x_points+k]=1;
            }
        }
    }
    std::cout<<"... filled nodes...\n";
    // create the 3D element connectivity
    for(int j=0;j<n_y_points-1;j++){
        for (int k=0;k<n_x_points-1;k++){
            ELEMENTS[j*(n_x_points-1)+k][0] = j*n_x_points+k;
            ELEMENTS[j*(n_x_points-1)+k][1] = j*n_x_points+k+1;
            ELEMENTS[j*(n_x_points-1)+k][2] = (j+1)*n_x_points+k+1;
            ELEMENTS[j*(n_x_points-1)+k][3] = (j+1)*n_x_points+k;
        }
    }
    // create the 2D boundary connectivity
    /// loop over boundaries

    QuadMesh myMesh;
    myMesh.nodes = NODES;
    myMesh.elements = ELEMENTS;
    myMesh.boundary_flag = BOUNDARIES;
    myMesh.n_nodes = n_x_points*n_y_points;
    myMesh.n_elements = (n_x_points-1)*(n_y_points-1);
    return myMesh;
}

//---------------------------------------//
// READ PARAVIEW
//---------------------------------------//
//
// read in the Paraview file and generate the mesh and fill in
QuadMesh readParaviewInput(const std::string& filename, const std::vector<double> &rectangleDimensions, const std::vector<int> &meshResolution)
{
    //std::cout << filename;
    std::cout << "Importing a mesh of from Paraview \n";
    std::vector<Vector2d> NODES;
    std::vector<std::vector<int> > ELEMENTS;
    std::vector<int> BOUNDARIES;
    int n_nodes = 0;
    int n_elements = 0;
    // READ NODES
    std::ifstream myfile;
    myfile.open(filename.c_str());
    std::string keyword_node = "double"; // or "float"
    //std::cout << myfile.is_open();
    if (myfile.is_open()){
        // read in until you find the keyword *NODE
        for (std::string line; std::getline(myfile, line); ){
            //std::cout << line << std::endl;
            // check for the keyword
            std::size_t found = line.find(keyword_node);
            if (found!=std::string::npos){
                // found the beginning of the nodes, so keep looping until you get '*'
                for ( std::string node_line; std::getline(myfile, node_line); ){
                    //std::cout << node_line << std::endl;
                    //std::cout << node_line.size() << std::endl;
                    std::size_t foundend = node_line.find("METADATA");
                    if(node_line.size() == 1 || node_line.empty() || foundend!=std::string::npos){
                        break;
                    }
                    std::vector<std::string> strs;
                    boost::split(strs,node_line,boost::is_any_of(" "));
                    Vector3d nodei;
                    Vector2d nodei2d;
                    for(int i = 0; i < strs.size()/3; i++){
                        nodei = Vector3d(std::stod(strs[0+3*i]),std::stod(strs[1+3*i]),std::stod(strs[2+3*i]));
                        nodei = (nodei-Vector3d(0.6,0.6,0.))*1000;
                        nodei2d = Vector2d(nodei(0),nodei(1));
                        NODES.push_back(nodei2d);
                        n_nodes += 1;

                        //-------------//
                        // ALTERNATIVE BOUNDARIES VERSION
                        //-------------//
                        // Be careful. Make sure you remember which nodes are part of which face
                        // Apply Dirichlet BCs first, so the corner nodes become part of those faces
                        //
                        int bound = 0;
                        if(round(nodei(0))==rectangleDimensions[0]){ // x = 0
                            bound = 1;
                        }
                        else if (round(nodei(0))==rectangleDimensions[1]){ // x = end
                            bound = 2;
                        }
                        else if(round(nodei(1))==rectangleDimensions[2]){ // y = 0
                            bound = 3;
                        }
                        else if (round(nodei(1))==rectangleDimensions[3]){ // y = end
                            bound = 4;
                        }
                        BOUNDARIES.push_back(bound);
                    }
                }
            }
        }
    }
    else{
        std::cout<<"\nFailed to open file.\n";
    }
    myfile.close();


    // READ ELEMENTS
    myfile.open(filename);
    std::string keyword_element = "CELLS";
    if (myfile.is_open()){
        // read in until you find the keyword ELEMENT
        for (std::string line; std::getline(myfile, line); ){
            // check for the keyword
            std::size_t found = line.find(keyword_element);
            if (found!=std::string::npos){
                // found the beginning of the nodes, so keep looping until you get '*'
                for ( std::string element_line; std::getline(myfile, element_line); ){
                    if(element_line.size() == 1 || element_line.empty()){
                        break;
                    }
                    // std::cout << element_line << std::endl;
                    // the nodes for the C3D8 element
                    // also remember that abaqus has node numbering starting in 1
                    std::vector<std::string> strs1;
                    boost::split(strs1,element_line,boost::is_any_of(" "));
                    std::vector<int> elemi; elemi.clear();
                    // Push nodes. PARAVIEW starts at the second number
                    elemi.push_back(std::stoi(strs1[1]));
                    elemi.push_back(std::stoi(strs1[5]));
                    elemi.push_back(std::stoi(strs1[2]));
                    elemi.push_back(std::stoi(strs1[6]));
                    elemi.push_back(std::stoi(strs1[3]));
                    elemi.push_back(std::stoi(strs1[7]));
                    elemi.push_back(std::stoi(strs1[4]));
                    elemi.push_back(std::stoi(strs1[8]));
                    //std::cout<<"\n";
                    ELEMENTS.push_back(elemi);
                    n_elements += 1;
                }
            }
        }
    }
    myfile.close();

    QuadMesh myMesh;
    myMesh.nodes = NODES;
    myMesh.elements = ELEMENTS;
    myMesh.boundary_flag = BOUNDARIES;
    myMesh.n_nodes = n_nodes;
    myMesh.n_elements = n_elements;
    //std::cout << "\n NODES \n" << myMesh.n_nodes << "\n ELEMENTS \n" << myMesh.n_elements << "\n BOUNDARIES \n" << BOUNDARIES.size() << "\n";
    return myMesh;
}

//---------------------------------------//
// READ COMSOL
//---------------------------------------//
//
// read in the COMSOL file and generate the mesh and fill in
QuadMesh readCOMSOLInput(const std::string& filename, const std::vector<double> &rectangleDimensions, const std::vector<int> &meshResolution)
{
    //std::cout << filename;
    std::cout << "Importing a mesh of from COMSOL \n";
    std::vector<Vector2d> NODES;
    std::vector<std::vector<int> > ELEMENTS;
    std::vector<int> BOUNDARIES;
    // READ NODES
    std::ifstream myfile;
    myfile.open(filename.c_str());
    std::string keyword_node = "Mesh point coordinates";
    //std::cout << myfile.is_open();
    if (myfile.is_open()){
        // read in until you find the keyword *NODE
        for (std::string line; std::getline(myfile, line); ){
            //std::cout << line << std::endl;
            // check for the keyword
            std::size_t found = line.find(keyword_node);
            if (found!=std::string::npos){
                // found the beginning of the nodes, so keep looping until you get '*'
                for ( std::string node_line; std::getline(myfile, node_line); ){
                    std::cout << node_line << std::endl;
                    //std::cout << node_line.size() << std::endl;
                    if(node_line.size() == 1 || node_line.empty()){
                        break;
                    }
                    std::vector<std::string> strs;
                    boost::split(strs,node_line,boost::is_any_of(" "));
                    Vector2d nodei = Vector2d(std::stod(strs[0]),std::stod(strs[1]));
                    NODES.push_back(nodei);

//                    //-------------//
//                    // ALTERNATIVE BOUNDARIES VERSION
//                    //-------------//
//                    // Be careful. Make sure you remember which nodes are part of which face
//                    // Apply Dirichlet BCs first, so the corner nodes become part of those faces
//                    //
//                    if(round(nodei(0))==rectangleDimensions[0]){ // x = 0
//                        BOUNDARIES.push_back(1);
//                    }
//                    else if (round(nodei(0))==rectangleDimensions[1]){ // x = end
//                        BOUNDARIES.push_back(2);
//                    }
//                    else if(round(nodei(1))==rectangleDimensions[2]){ // y = 0
//                        BOUNDARIES.push_back(3);
//                    }
//                    else if (round(nodei(1))==rectangleDimensions[3]){ // y = end
//                        BOUNDARIES.push_back(4);
//                    }
//                    else{
//                        BOUNDARIES.push_back(0);
//                    }

                    //-------------//
                    // CIRCULAR DOMAIN BOUNDARIES VERSION
                    //-------------//
                    // Be careful. Make sure you remember which nodes are part of which face
                    // Apply Dirichlet BCs first, so the corner nodes become part of those faces
                    //
                    double distance = sqrt(pow((nodei(0) - 0.0),2) + pow((nodei(1) - 0.0),2));
                    double tolerance = 0.5;
                    if (round(distance) >= rectangleDimensions[1]/2 - tolerance){ // r = end of domain
                        BOUNDARIES.push_back(1);
                    }
                    else{
                        BOUNDARIES.push_back(0);
                    }
                }
            }
        }
    }
    else{
        std::cout<<"\nFailed to open file.\n";
    }
    myfile.close();


    // READ ELEMENTS
    myfile.open(filename);
    std::string keyword_element = "4 # number of nodes per element";
    if (myfile.is_open()){
        // read in until you find the keyword ELEMENT
        for (std::string line; std::getline(myfile, line); ){
            // check for the keyword
            std::size_t found = line.find(keyword_element);
            if (found!=std::string::npos){
                // Skip two lines to get to element connectivity
                getline (myfile,line); getline (myfile,line);
                // found the beginning of the nodes, so keep looping until you get '*'
                for ( std::string element_line; std::getline(myfile, element_line); ){
                    if(element_line.size() == 1 || element_line.empty()){
                        break;
                    }
                    std::cout << element_line << std::endl;
                    // the nodes for the C3D8 element
                    // also remember that abaqus has node numbering starting in 1
                    std::vector<std::string> strs1;
                    boost::split(strs1,element_line,boost::is_any_of(" "));
                    std::vector<int> elemi; elemi.clear();
                    // Push nodes. COMSOL has a weird format, so this makes it correct for Paraview
                    elemi.push_back(std::stoi(strs1[0]));
                    elemi.push_back(std::stoi(strs1[1]));
                    elemi.push_back(std::stoi(strs1[3]));
                    elemi.push_back(std::stoi(strs1[2]));
                    //std::cout<<"\n";
                    ELEMENTS.push_back(elemi);
                }
            }
        }
    }
    myfile.close();

    QuadMesh myMesh;
    myMesh.nodes = NODES;
    myMesh.elements = ELEMENTS;
    myMesh.boundary_flag = BOUNDARIES;
    myMesh.n_nodes = NODES.size();
    myMesh.n_elements = ELEMENTS.size();
    //std::cout << "\n NODES \n" << myMesh.n_nodes << "\n ELEMENTS \n" << myMesh.n_elements << "\n BOUNDARIES \n" << BOUNDARIES.size() << "\n";
    return myMesh;
}


double distanceX2E(std::vector<double> &ellipse, double x_coord, double y_coord,double mesh_size)
{
	// given a point and the geometry of an ellipse, give me the
	// distance along the x axis towards the ellipse
	double x_center = ellipse[0];
	double y_center = ellipse[1];
	double x_axis = ellipse[2];
	double y_axis = ellipse[3];
	double alpha = ellipse[4];
	
	// equation of the ellipse 
	double x_ellipse_1 = (pow(x_axis,2)*x_center*pow(sin(alpha),2) + pow(x_axis,2)*y_center*sin(2*alpha)/2 - pow(x_axis,2)*y_coord*sin(2*alpha)/2 \
						+ x_center*pow(y_axis,2)*pow(cos(alpha),2) + pow(y_axis,2)*y_center*sin(2*alpha)/2 - pow(y_axis,2)*y_coord*sin(2*alpha)/2 -\
						sqrt(pow(x_axis,2)*pow(y_axis,2)*(pow(x_axis,2)*pow(sin(alpha),2) - pow(y_axis,2)*pow(sin(alpha),2) + pow(y_axis,2) \
				- 4*pow(y_center,2)*pow(sin(alpha),4) + 4*pow(y_center,2)*pow(sin(alpha),2) - pow(y_center,2) + 8*y_center*y_coord*pow(sin(alpha),4)\
				 - 8*y_center*y_coord*pow(sin(alpha),2) + 2*y_center*y_coord - 4*pow(y_coord,2)*pow(sin(alpha),4) + 4*pow(y_coord,2)*pow(sin(alpha),2)\
				  - pow(y_coord,2))))/(pow(x_axis,2)*pow(sin(alpha),2) + pow(y_axis,2)*pow(cos(alpha),2));
	double x_ellipse_2 = (pow(x_axis,2)*x_center*pow(sin(alpha),2) + pow(x_axis,2)*y_center*sin(2*alpha)/2 - pow(x_axis,2)*y_coord*sin(2*alpha)/2 \
						+ x_center*pow(y_axis,2)*pow(cos(alpha),2) + pow(y_axis,2)*y_center*sin(2*alpha)/2 - pow(y_axis,2)*y_coord*sin(2*alpha)/2 +\
						sqrt(pow(x_axis,2)*pow(y_axis,2)*(pow(x_axis,2)*pow(sin(alpha),2) - pow(y_axis,2)*pow(sin(alpha),2) + pow(y_axis,2) \
				- 4*pow(y_center,2)*pow(sin(alpha),4) + 4*pow(y_center,2)*pow(sin(alpha),2) - pow(y_center,2) + 8*y_center*y_coord*pow(sin(alpha),4)\
				 - 8*y_center*y_coord*pow(sin(alpha),2) + 2*y_center*y_coord - 4*pow(y_coord,2)*pow(sin(alpha),4) + 4*pow(y_coord,2)*pow(sin(alpha),2)\
				  - pow(y_coord,2))))/(pow(x_axis,2)*pow(sin(alpha),2) + pow(y_axis,2)*pow(cos(alpha),2));
	// which is closer?
	double distance1 = fabs(x_ellipse_1 - x_coord);
	double distance2 = fabs(x_ellipse_2 - x_coord);
	if(distance1<distance2 && distance1<mesh_size){
		return x_ellipse_1-x_coord;
	}else if(distance2<mesh_size){
		return x_ellipse_2-x_coord;
	}
	return 0;
}

// conform the mesh to a given ellipse
void conformMesh2Ellipse(QuadMesh &myMesh, std::vector<double> &ellipse)
{
	// the ellipse is defined by center, axis, and angle
	double x_center = ellipse[0];
	double y_center = ellipse[1];
	double x_axis = ellipse[2];
	double y_axis = ellipse[3];
	double alpha_ellipse = ellipse[4];
	
	// loop over the mesh nodes 
	double x_coord,y_coord,check,d_x2e,mesh_size;
	mesh_size = (myMesh.nodes[1](0)-myMesh.nodes[0](0))/1.1;
	for(int i=0;i<myMesh.n_nodes;i++){
		// if the point is inside check if it is close, if it is in a certain 
		// range smaller than mesh size then move it the ellipse along x. ta ta
		x_coord = myMesh.nodes[i](0);
		y_coord = myMesh.nodes[i](1);
		check = pow((x_coord-x_center)*cos(alpha_ellipse)+(y_coord-y_center)*sin(alpha_ellipse),2)/(x_axis*x_axis) +\
				pow((x_coord-x_center)*sin(alpha_ellipse)+(y_coord-y_center)*cos(alpha_ellipse),2)/(y_axis*y_axis) ;
		if(check>1){
			// calculate the distance to the ellipse along x axis
			d_x2e = distanceX2E(ellipse,x_coord,y_coord,mesh_size);
			myMesh.nodes[i](0) += d_x2e;
		}
	}					
}