#include <fdtd_2d.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <fstream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <json/json.h>
#include <sstream>     
#include <Mesh2D.h>
#include <global_variables.h>
#include <set>

#include <cuda_runtime.h>


FDTD_2D::FDTD_2D() {
  compute_grid_parameters();
  setup_source(); 
}

FDTD_2D::~FDTD_2D() = default;

void FDTD_2D::getSourceID() {
  double min_dist2 = std::numeric_limits<double>::max(); // squared distance
  int closest_id = -1;

  for (auto &n : Ez_nodes) {
    double dx = n.x - source_point[0];
    double dy = n.y - source_point[1];
    double dist2 = dx*dx + dy*dy;
    if (dist2 < min_dist2) {
      min_dist2 = dist2;
      closest_id = n.nodeID;
    }
  }

  source_ID = closest_id;
}

void FDTD_2D::compute_grid_parameters() {

  dx = step_size;
  dy = step_size;  
  
  h_coeff = dt / (sqrt(MU_0*EPSILON_0));
  e_coeff = dt / (sqrt(MU_0*EPSILON_0));
  std::cout<<"Completed grid computation."<<std::endl;
}

void FDTD_2D::get_fields(std::vector<EzNode> Ez_nodes_, std::vector<HxNode> Hx_nodes_, std::vector<HyNode> Hy_nodes_) {
  Ez_nodes = Ez_nodes_;
  Hx_nodes = Hx_nodes_;
  Hy_nodes = Hy_nodes_;
}

void FDTD_2D::setup_source() {
  
  if (source_type == "sin") {
    source_fn = [=](double t){ return amplitude * sin(omega * t); };
  }
  else if (source_type == "gaussian") {
    source_fn = [=](double t){ return amplitude * exp(-pow((t - pulse_delay)/pulse_width, 2.0)); };    
  }
  else if (source_type == "gaussian_sin") {    
    source_fn = [=](double t){ return amplitude * sin(omega * t) * exp(-pow((t - pulse_delay)/pulse_width, 2.0)); };
  }
}

double FDTD_2D::source(double time) {
  return source_fn(time);
}

void FDTD_2D::set_domain_size(int Ez_domain_size_, int Hx_domain_size_, int Hy_domain_size_) {
  Ez_domain_size = Ez_domain_size_;
  Hx_domain_size = Hx_domain_size_;
  Hy_domain_size = Hy_domain_size_;
}


void FDTD_2D::set_PML_parameters() {

  // double grading_exponent = 5; double kappa_max = 5.0; double a_max = 1;

  // Option A: scale conductivity by a factor
  double sigma_scale = 10.0;

  double pml_cond_e = -sigma_scale * (grading_exponent + 1) * log(1e-12) * c0 * EPSILON_0 / (2 * pml_size);
  int maxCellID = pml_size / step_size;

  // Resize helper
  auto resizePML = [&](auto &sigma, auto &kappa, auto &a, auto &b, auto &c, auto &Da, auto &Db) {
    sigma.resize(maxCellID, 0);
    kappa.resize(maxCellID, 0);
    a.resize(maxCellID, 0);
    b.resize(maxCellID, 0);
    c.resize(maxCellID, 0);
    Da.resize(maxCellID, 0);
    Db.resize(maxCellID, 0);
  };

  resizePML(sigma_x_r, kappa_x_r, a_x_r, b_x_r, c_x_r, Da_x_r, Db_x_r);
  resizePML(sigma_x_l, kappa_x_l, a_x_l, b_x_l, c_x_l, Da_x_l, Db_x_l);
  resizePML(sigma_y_t, kappa_y_t, a_y_t, b_y_t, c_y_t, Da_y_t, Db_y_t);
  resizePML(sigma_y_b, kappa_y_b, a_y_b, b_y_b, c_y_b, Da_y_b, Db_y_b);

  // Compute helper
  auto computePML = [&](auto &sigma, auto &kappa, auto &a, auto &b, auto &c, auto &Da, auto &Db) {
    for (int i = 0; i < maxCellID; i++) {
      double x = (i * step_size) / pml_size;                      
      sigma[i] = pml_cond_e * pow(x, grading_exponent);
      kappa[i] = 1.0 + (kappa_max - 1.0) * pow(x, grading_exponent);
      a[i] = a_max * (1.0 - x);
      b[i] = exp(-(sigma[i] / kappa[i] + a[i]) * dt / EPSILON_0);
      c[i] = (sigma[i] * (b[i] - 1.0)) / (sigma[i] + a[i] * kappa[i]) / kappa[i];
    }

    for (int i = 0; i < maxCellID; ++i) {
      double s = sigma[i];
      double denom = 1.0 + (s * dt) / (2.0 * MU_0);
      Da[i] = (1.0 - (s * dt) / (2.0 * MU_0)) / denom;
      Db[i] = (dt / (sqrt(MU_0 * EPSILON_0))) / denom;
    }
  };

  // Right/top (no flip)
  computePML(sigma_x_r, kappa_x_r, a_x_r, b_x_r, c_x_r, Da_x_r, Db_x_r);
  computePML(sigma_y_t, kappa_y_t, a_y_t, b_y_t, c_y_t, Da_y_t, Db_y_t);  
 
  sigma_x_l = sigma_x_r;  sigma_y_b = sigma_y_t;
  kappa_x_l = kappa_x_r;  kappa_y_b = kappa_y_t;
  a_x_l     = a_x_r;      a_y_b     = a_y_t;
  b_x_l     = b_x_r;      b_y_b     = b_y_t;
  c_x_l     = c_x_r;      c_y_b     = c_y_t;
  Da_x_l    = Da_x_r;     Da_y_b    = Da_y_t;
  Db_x_l    = Db_x_r;     Db_y_b    = Db_y_t;

}


// ---- global CUDA variables ----
__constant__ double dx_c;
__constant__ double dy_c;
__constant__ double h_coeff_c;
__constant__ double e_coeff_c;

__global__ void update_Ez(double *Ez, double *Hy, double *Hx, int *ez_left, int *ez_right, int *ez_top, int *ez_bottom, int Ez_size)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < Ez_size) 
  {
    int L = ez_left[id]; int R = ez_right[id]; int B = ez_bottom[id]; int T = ez_top[id];
    double dHy_dx = (Hy[R] - Hy[L]) / dx_c;   // dx_c from __constant__
    double dHx_dy = (Hx[T] - Hx[B]) / dy_c;   // dy_c from __constant__ 
    Ez[id] += e_coeff_c * (dHy_dx - dHx_dy);  // e_coeff_c from __constant__
  }
}

__global__ void update_Hx(double *Hx, double *Ez, int *top, int *bottom, int Hx_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Hx_size) {
      int T = top[i];
      int B = bottom[i];
      double curlEz = (Ez[T] - Ez[B]) / dy_c;    // dy_c is __constant__
      Hx[i] -= h_coeff_c * curlEz;
    }
}

__global__ void update_Hy(double *Hy, double *Ez, int *left, int *right, int Hy_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Hy_size) {      
      int L = left[i];
      int R = right[i];
      double curlEz = (Ez[R] - Ez[L]) / dx_c;     // dx_c is __constant__
      Hy[i] += h_coeff_c * curlEz;    
    }
}

__global__ void gpu_source_injection(double *Ez, int source_ID, double time, double omega)
{
  double src = sin(time*omega);
  Ez[source_ID] += src;
}

void FDTD_2D::TMz_mesh_update() {
  
  getSourceID();

  // Make arrays of Ez, Hx and Hy CPU copy   
  double* Hy_host = new double[Hy_domain_size + 1];
  double* Hx_host = new double[Hx_domain_size + 1];
  double* Ez_host = new double[Ez_domain_size + 1];

  // ----------------------
  // CPU neighbour arrays
  // ----------------------
  int* hy_left_neighbour    = new int[Hy_domain_size];
  int* hy_right_neighbour   = new int[Hy_domain_size];

  int* hx_top_neighbour     = new int[Hx_domain_size];
  int* hx_bottom_neighbour  = new int[Hx_domain_size];

  int* ez_top_neighbour     = new int[Ez_domain_size];
  int* ez_bottom_neighbour  = new int[Ez_domain_size];
  int* ez_left_neighbour    = new int[Ez_domain_size];
  int* ez_right_neighbour   = new int[Ez_domain_size];

  // ----------------------
  // Fill CPU neighbour arrays
  // ----------------------
  for (int i = 0; i < Hy_domain_size; i++) {
    hy_left_neighbour[i]  = Hy_nodes[i].ez_left_id;
    hy_right_neighbour[i] = Hy_nodes[i].ez_right_id;
    if(hy_left_neighbour[i] > Hy_domain_size) hy_left_neighbour[i] = Hy_domain_size;
    if(hy_right_neighbour[i] > Hy_domain_size) hy_right_neighbour[i] = Hy_domain_size;          
  }

  for (int i = 0; i < Hx_domain_size; i++) {
    hx_top_neighbour[i]    = Hx_nodes[i].ez_top_id;
    hx_bottom_neighbour[i] = Hx_nodes[i].ez_bottom_id;

    if (hx_top_neighbour[i]    > Hx_domain_size) hx_top_neighbour[i]    = Hx_domain_size;
    if (hx_bottom_neighbour[i] > Hx_domain_size) hx_bottom_neighbour[i] = Hx_domain_size;
  }


  for (int i = 0; i < Ez_domain_size; i++) {
    ez_top_neighbour[i]    = Ez_nodes[i].hx_top_id;
    ez_bottom_neighbour[i] = Ez_nodes[i].hx_bottom_id;
    ez_left_neighbour[i]   = Ez_nodes[i].hy_left_id;
    ez_right_neighbour[i]  = Ez_nodes[i].hy_right_id;

    if (ez_top_neighbour[i]    > Ez_domain_size) ez_top_neighbour[i]    = Ez_domain_size;
    if (ez_bottom_neighbour[i] > Ez_domain_size) ez_bottom_neighbour[i] = Ez_domain_size;
    if (ez_left_neighbour[i]   > Ez_domain_size) ez_left_neighbour[i]   = Ez_domain_size;
    if (ez_right_neighbour[i]  > Ez_domain_size) ez_right_neighbour[i]  = Ez_domain_size;
  }

  // ----------------------
  // GPU pointers
  // ----------------------
  double *d_Hy = nullptr;
  double *d_Hx = nullptr;
  double *d_Ez = nullptr;

  int *d_hy_left_neighbour    = nullptr;  int *d_hx_top_neighbour     = nullptr;
  int *d_hy_right_neighbour   = nullptr;  int *d_hx_bottom_neighbour  = nullptr;

  int *d_ez_top_neighbour     = nullptr;
  int *d_ez_bottom_neighbour  = nullptr;
  int *d_ez_left_neighbour    = nullptr;
  int *d_ez_right_neighbour   = nullptr;

  // ----------------------
  // GPU allocation
  // ----------------------

  size_t byte_size_hy = Hy_domain_size * sizeof(int);
  size_t byte_size_hx = Hx_domain_size * sizeof(int);
  size_t byte_size_ez = Ez_domain_size * sizeof(int);

  size_t hy_byte_double = (Hy_domain_size + 1) * sizeof(double);
  size_t hx_byte_double = (Hx_domain_size + 1) * sizeof(double);
  size_t ez_byte_double = (Ez_domain_size + 1) * sizeof(double);

  cudaMalloc(&d_Hx, hx_byte_double);
  cudaMalloc(&d_Hy, hy_byte_double);
  cudaMalloc(&d_Ez, ez_byte_double);

  cudaMalloc(&d_hy_left_neighbour,    byte_size_hy);
  cudaMalloc(&d_hy_right_neighbour,   byte_size_hy);
  cudaMalloc(&d_hx_top_neighbour,     byte_size_hx);
  cudaMalloc(&d_hx_bottom_neighbour,  byte_size_hx);
  cudaMalloc(&d_ez_top_neighbour,     byte_size_ez);
  cudaMalloc(&d_ez_bottom_neighbour,  byte_size_ez);
  cudaMalloc(&d_ez_left_neighbour,    byte_size_ez);
  cudaMalloc(&d_ez_right_neighbour,   byte_size_ez);

  // ----------------------
  // CPU → GPU copy
  // ----------------------
  cudaMemset(d_Hy, 0, hy_byte_double);
  cudaMemset(d_Hx, 0, hx_byte_double);
  cudaMemset(d_Ez, 0, ez_byte_double);

  cudaMemcpy(d_hy_left_neighbour, hy_left_neighbour, byte_size_hy, cudaMemcpyHostToDevice);
  cudaMemcpy(d_hy_right_neighbour,   hy_right_neighbour, byte_size_hy, cudaMemcpyHostToDevice);
  cudaMemcpy(d_hx_top_neighbour,     hx_top_neighbour,  byte_size_hx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_hx_bottom_neighbour,  hx_bottom_neighbour, byte_size_hx, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ez_top_neighbour,     ez_top_neighbour, byte_size_ez, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ez_bottom_neighbour,  ez_bottom_neighbour, byte_size_ez, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ez_left_neighbour,    ez_left_neighbour, byte_size_ez, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ez_right_neighbour,   ez_right_neighbour, byte_size_ez, cudaMemcpyHostToDevice);


  // Copying constants on GPU

  cudaMemcpyToSymbol(dx_c, &dx, sizeof(double));
  cudaMemcpyToSymbol(dy_c, &dy, sizeof(double));
  cudaMemcpyToSymbol(h_coeff_c, &h_coeff, sizeof(double));
  cudaMemcpyToSymbol(e_coeff_c, &e_coeff, sizeof(double));

  int THREADS = 128;
  int hy_blocks = (Hy_domain_size + THREADS - 1) / THREADS;
  int hx_blocks = (Hx_domain_size + THREADS - 1) / THREADS;
  int ez_blocks = (Ez_domain_size + THREADS - 1) / THREADS;

  for (int t = 0; t < N_time_steps; t++) {

    // --- Update Ez on CPU---
    // for (int Ez_node_ID = 0; Ez_node_ID < Ez_domain_size; Ez_node_ID++) {
    //   auto &n = Ez_nodes[Ez_node_ID];
    //   double dHy_dx = (Hy_nodes[n.hy_right_id].fieldValue - Hy_nodes[n.hy_left_id].fieldValue) / dx;        
    //   double dHx_dy = (Hx_nodes[n.hx_top_id].fieldValue - Hx_nodes[n.hx_bottom_id].fieldValue) / dy;            
    //   n.fieldValue += e_coeff * (dHy_dx - dHx_dy);
    // }

    update_Ez<<<ez_blocks, THREADS>>>(d_Ez, d_Hy, d_Hx, d_ez_left_neighbour, d_ez_right_neighbour, d_ez_top_neighbour, d_ez_bottom_neighbour, Ez_domain_size);
    cudaDeviceSynchronize();  

    // --- Update Hx on CPU---
    // for (int Hx_node_ID = 0; Hx_node_ID < Hx_domain_size; Hx_node_ID++) {
    //   auto &n = Hx_nodes[Hx_node_ID];
    //   double curlEz = (Ez_nodes[n.ez_top_id].fieldValue - Ez_nodes[n.ez_bottom_id].fieldValue) / dy;              
    //   n.fieldValue -= h_coeff * curlEz;
    // }

    update_Hx<<<hx_blocks, THREADS>>>(d_Hx, d_Ez, d_hx_top_neighbour, d_hx_bottom_neighbour, Hx_domain_size);
    cudaDeviceSynchronize();  

    // // --- Update Hy on CPU---
    //Replace with update Hy kernel
    // for (int Hy_node_ID = 0; Hy_node_ID < Hy_domain_size; Hy_node_ID++) {
    //   auto &n = Hy_nodes[Hy_node_ID];
    //   double curlEz = (Ez_nodes[n.ez_right_id].fieldValue - Ez_nodes[n.ez_left_id].fieldValue) / dx;    
    //   n.fieldValue += h_coeff * curlEz;
    // }

    // // --- Update Hy on GPU ---
    update_Hy<<<hy_blocks, THREADS>>>(d_Hy, d_Ez, d_hy_left_neighbour, d_hy_right_neighbour, Hy_domain_size);    
    cudaDeviceSynchronize();  

    // --- Source Injection on CPU --- 
    double time = t * dt;
    // double src = sin(omega * time);
    // Ez_nodes[source_ID].fieldValue += src;
    
    //Source injection on GPU
    gpu_source_injection<<<1, 1>>>(d_Ez, source_ID, time, omega);
    cudaDeviceSynchronize();  
    
    std::cout << "Time step = " << t << std::endl;
  }
  
  // cudaDeviceSynchronize();    
  cudaMemcpy(Hy_host, d_Hy, hy_byte_double, cudaMemcpyDeviceToHost);
  cudaMemcpy(Hx_host, d_Hx, hx_byte_double, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ez_host, d_Ez, ez_byte_double, cudaMemcpyDeviceToHost);

  //copy back Ez, Hx and Hy info on CPU

  // for (int i = 0; i < 10; i++) {
  //   cout<<Hy_host[i]<<endl;
  // }

  // for (int Hy_node_ID = 0; Hy_node_ID < 10; Hy_node_ID++) {
  //   auto &n = Hy_nodes[Hy_node_ID];
  //   cout<<n.fieldValue<<endl;
  // }

  // ----------------------
  // CUDA free
  // ----------------------
  cudaFree(d_hy_left_neighbour);
  cudaFree(d_hy_right_neighbour);
  cudaFree(d_hx_top_neighbour);
  cudaFree(d_hx_bottom_neighbour);
  cudaFree(d_ez_top_neighbour);
  cudaFree(d_ez_bottom_neighbour);
  cudaFree(d_ez_left_neighbour);
  cudaFree(d_ez_right_neighbour);
  cudaFree(d_Hx);
  cudaFree(d_Hy);
  cudaFree(d_Ez);

  // ----------------------
  // CPU free
  // ----------------------
  delete[] hy_left_neighbour;
  delete[] hy_right_neighbour;
  delete[] hx_top_neighbour;
  delete[] hx_bottom_neighbour;
  delete[] ez_top_neighbour;
  delete[] ez_bottom_neighbour;
  delete[] ez_left_neighbour;
  delete[] ez_right_neighbour;
  delete[] Hy_host;
  delete[] Hx_host;
  delete[] Ez_host;
}

