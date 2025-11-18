#ifndef FDTD_2D_H
#define FDTD_2D_H

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <json/json.h>
#include <functional>
#include <Mesh2D.h>

// =========================
// Class Declaration
// =========================
class FDTD_2D {
  public:
    // Constructor / Destructor
    FDTD_2D();
    ~FDTD_2D();

    // Initialization    
    void compute_grid_parameters();
    void get_fields(std::vector<EzNode>, std::vector<HxNode>, std::vector<HyNode>);
    void initialize_fields();
    void init_PML();
      
    // Main solver loop
    void noPML_TMz();
    void TMz_mesh_update();  
    void TMz_mesh_update_CPML();
    void set_domain_size(int, int, int);
    void set_PML_parameters();
    void saveMeshToVTK(const std::string& type, std::string& filename) const ;
    
  private:    
  
    double dx, dy;
		double h_coeff, e_coeff;		

    // X boundaries
    std::vector<double> sigma_x_r, sigma_x_l;
    std::vector<double> kappa_x_r, kappa_x_l;
    std::vector<double> a_x_r, a_x_l;
    std::vector<double> b_x_r, b_x_l;
    std::vector<double> c_x_r, c_x_l;
    std::vector<double> Da_x_r, Da_x_l;
    std::vector<double> Db_x_r, Db_x_l;

    // Y boundaries
    std::vector<double> sigma_y_t, sigma_y_b;
    std::vector<double> kappa_y_t, kappa_y_b;
    std::vector<double> a_y_t, a_y_b;
    std::vector<double> b_y_t, b_y_b;
    std::vector<double> c_y_t, c_y_b;
    std::vector<double> Da_y_t, Da_y_b;
    std::vector<double> Db_y_t, Db_y_b;
  
    // --- Update coefficients ---
    std::vector<double> Ca_x, Ca_y, Ca_z, Cb_x, Cb_y, Cb_z;
      
    // --- Source ---    
    double source(double);    
    std::function<double(double)> source_fn; 
    void setup_source();  
    void getSourceID();    

    //----------------------------------- Using the MESH2D library -----------------------------------//
    std::vector<EzNode> Ez_nodes;
    std::vector<HxNode> Hx_nodes;
    std::vector<HyNode> Hy_nodes;
    int Ez_domain_size, Hx_domain_size, Hy_domain_size;
    int source_ID;

};

#endif // FDTD_2D_H
