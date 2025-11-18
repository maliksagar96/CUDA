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
// #include <vtkWriter.h>
#include <global_variables.h>
#include <set>

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

// void FDTD_2D::saveMeshToVTK(const std::string& type, std::string& filename) const {
//   VTKQuadWriter vtk2elements;
//   std::vector<double> coords;
//   std::vector<int> connectivity;
//   std::vector<double> field;
//   std::string fieldName;

//   if (type == "Ez") {
//     fieldName = "Ez";
//     // ---- Ez export ----
//     for (int i = 0; i < Ez_domain_size; i++) {
//       const auto &n = Ez_nodes[i];
//       coords.push_back(n.x);
//       coords.push_back(n.y);
//       coords.push_back(0.0);
//     }

//     for (int i = 0; i < Ez_domain_size; i++) {
//       const auto &n = Ez_nodes[i];
//       if (n.right != -1 && n.top != -1) {
//         const auto &rightNode = Ez_nodes[n.right];
//         const auto &topNode   = Ez_nodes[n.top];
//         if (topNode.right != -1) {
//           const auto &topRight = Ez_nodes[topNode.right];
//           connectivity.push_back(n.nodeID);
//           connectivity.push_back(rightNode.nodeID);
//           connectivity.push_back(topRight.nodeID);
//           connectivity.push_back(topNode.nodeID);
//           field.push_back(n.fieldValue);
//         }
//       }
//     }
//   }
//   else if (type == "Hx") {
//     fieldName = "Hx";
//     // ---- Hx export ----
//     for (int i = 0; i < Hx_domain_size; i++) {
//       const auto &n = Hx_nodes[i];
//       coords.push_back(n.x);
//       coords.push_back(n.y);
//       coords.push_back(0.0);
//     }

//     for (int i = 0; i < Hx_domain_size; i++) {
//       const auto &n = Hx_nodes[i];
//       if (n.right != -1 && n.top != -1) {
//         const auto &rightNode = Hx_nodes[n.right];
//         const auto &topNode   = Hx_nodes[n.top];
//         if (topNode.right != -1) {
//           const auto &topRight = Hx_nodes[topNode.right];
//           connectivity.push_back(n.nodeID);
//           connectivity.push_back(rightNode.nodeID);
//           connectivity.push_back(topRight.nodeID);
//           connectivity.push_back(topNode.nodeID);
//           field.push_back(n.fieldValue);
//         }
//       }
//     }
//   }
//   else if (type == "Hy") {
//     fieldName = "Hy";
//     // ---- Hy export ----
//     for (int i = 0; i < Hy_domain_size; i++) {
//       const auto &n = Hy_nodes[i];
//       coords.push_back(n.x);
//       coords.push_back(n.y);
//       coords.push_back(0.0);
//     }

//     for (int i = 0; i < Hy_domain_size; i++) {
//       const auto &n = Hy_nodes[i];
//       if (n.right != -1 && n.top != -1) {
//         const auto &rightNode = Hy_nodes[n.right];
//         const auto &topNode   = Hy_nodes[n.top];
//         if (topNode.right != -1) {
//           const auto &topRight = Hy_nodes[topNode.right];
//           connectivity.push_back(n.nodeID);
//           connectivity.push_back(rightNode.nodeID);
//           connectivity.push_back(topRight.nodeID);
//           connectivity.push_back(topNode.nodeID);
//           field.push_back(n.fieldValue);
//         }
//       }
//     }
//   }
//   else {
//     throw std::runtime_error("Unknown type: " + type);
//   }

//   vtk2elements.set_points(coords);
//   vtk2elements.set_cells(connectivity);
//   vtk2elements.add_scalar(field, fieldName);
//   vtk2elements.write_vtk(filename);
// }

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

void FDTD_2D::TMz_mesh_update() {

  getSourceID();

  for (int t = 0; t < N_time_steps; t++) {

    // --- Update Ez ---
    for (int Ez_node_ID = 0; Ez_node_ID < Ez_domain_size; Ez_node_ID++) {
      auto &n = Ez_nodes[Ez_node_ID];
      double dHy_dx = (Hy_nodes[n.hy_right_id].fieldValue - Hy_nodes[n.hy_left_id].fieldValue) / dx;        
      double dHx_dy = (Hx_nodes[n.hx_top_id].fieldValue - Hx_nodes[n.hx_bottom_id].fieldValue) / dy;            
      n.fieldValue += e_coeff * (dHy_dx - dHx_dy);
    }

    // --- Update Hx ---
    for (int Hx_node_ID = 0; Hx_node_ID < Hx_domain_size; Hx_node_ID++) {
      auto &n = Hx_nodes[Hx_node_ID];
      double curlEz = (Ez_nodes[n.ez_top_id].fieldValue - Ez_nodes[n.ez_bottom_id].fieldValue) / dy;              
      n.fieldValue -= h_coeff * curlEz;
    }

    // --- Update Hy ---
    for (int Hy_node_ID = 0; Hy_node_ID < Hy_domain_size; Hy_node_ID++) {
      auto &n = Hy_nodes[Hy_node_ID];
      double curlEz = curlEz = (Ez_nodes[n.ez_right_id].fieldValue - Ez_nodes[n.ez_left_id].fieldValue) / dx;    
      n.fieldValue += h_coeff * curlEz;
    }

    // --- Source injection ---
    // double time = t * dt;  
    // Ez_nodes[source_ID].fieldValue += source(time);
    // // Ez_nodes[source_ID].fieldValue += sin(time);

    double time = t * dt;
    double pulse_width = 10 * dt;              // smoothness control
    double pulse_delay = 6.0 * pulse_width;    // shift so it starts near zero
    double src = sin(omega * time) * exp(-pow((time - pulse_delay)/pulse_width, 2.0));

    Ez_nodes[source_ID].fieldValue += amplitude * src;

    std::cout << "Time step = " << t << std::endl;
    // if (t % data_capture_interval == 0) {
    //   std::string type = "Ez";
    //   std::string filename = "data/" + type + "/" + type + "_" + std::to_string(t) + ".vtk";
    //   saveMeshToVTK(type, filename);
    // }
  }
  
  for (int Hy_node_ID = 0; Hy_node_ID < 10; Hy_node_ID++) {
    auto &n = Hy_nodes[Hy_node_ID];
    cout<<n.fieldValue<<endl;
  }

}


// void FDTD_2D::TMz_mesh_update_CPML() {
  
//   getSourceID();
//   std::set<int> unique_indices;

//   for (int t = 0; t < N_time_steps; t++) {

//     // --- Update Ez ---
//     for (int Ez_node_ID = 0; Ez_node_ID < Ez_domain_size; Ez_node_ID++) {
//       auto &n = Ez_nodes[Ez_node_ID];
//       double dHy_dx = (Hy_nodes[n.hy_right_id].fieldValue - Hy_nodes[n.hy_left_id].fieldValue) / dx;    
//       double dHx_dy = (Hx_nodes[n.hx_top_id].fieldValue - Hx_nodes[n.hx_bottom_id].fieldValue) / dy;            
//       n.fieldValue += e_coeff * (dHy_dx - dHx_dy);
//     }

// // helper lambda (put near top of function or in class)
// auto clamp_index = [](int idx, int max_idx) {
//   if (idx < 0) return 0;
//   if (idx > max_idx) return max_idx;
//   return idx;
// };

// // --- Psi_Hx_y update (bottom + top safe) ---
// for (int Hx_node_ID = 0; Hx_node_ID < Hx_domain_size; Hx_node_ID++) {
//   auto &n = Hx_nodes[Hx_node_ID];
//   double curlEz = (Ez_nodes[n.ez_top_id].fieldValue - Ez_nodes[n.ez_bottom_id].fieldValue) / dy;

//   if (n.y > domain_size[3]) { // top PML
//     int raw = static_cast<int>(std::floor((n.y - domain_size[3]) / step_size));
//     int index = clamp_index(raw, static_cast<int>(b_y_t.size()) - 1);
//     n.Psi_Hx_y = b_y_t[index] * n.Psi_Hx_y + c_y_t[index] * curlEz;
//   }
//   else if (n.y < domain_size[1]) { // bottom PML
//     int raw = static_cast<int>(std::floor((domain_size[1] - n.y) / step_size));
//     int index = clamp_index(raw, static_cast<int>(b_y_b.size()) - 1);
//     n.Psi_Hx_y = b_y_b[index] * n.Psi_Hx_y + c_y_b[index] * curlEz;
//   }
//   else {
//     n.Psi_Hx_y = 0.0;
//   }
// }

// // --- Update Hx (with checks) ---
// for (int Hx_node_ID = 0; Hx_node_ID < Hx_domain_size; Hx_node_ID++) {
//   auto &n = Hx_nodes[Hx_node_ID];
//   double curlEz = (Ez_nodes[n.ez_top_id].fieldValue - Ez_nodes[n.ez_bottom_id].fieldValue) / dy;

//   if (n.y > domain_size[3]) { // top PML
//     int raw = static_cast<int>(std::floor((n.y - domain_size[3]) / step_size));
//     int index = clamp_index(raw, static_cast<int>(Da_y_t.size()) - 1);

//     // safety: avoid divide by zero
//     double kappa = (kappa_y_t[index] == 0.0) ? 1.0 : kappa_y_t[index];

//     n.fieldValue = Da_y_t[index] * n.fieldValue - Db_y_t[index] * (curlEz / kappa + n.Psi_Hx_y);
//   }
//   else if (n.y < domain_size[1]) { // bottom PML
//     int raw = static_cast<int>(std::floor((domain_size[1] - n.y) / step_size));
//     int index = clamp_index(raw, static_cast<int>(Da_y_b.size()) - 1);

//     double kappa = (kappa_y_b[index] == 0.0) ? 1.0 : kappa_y_b[index];

//     n.fieldValue = Da_y_b[index] * n.fieldValue - Db_y_b[index] * (curlEz / kappa + n.Psi_Hx_y);
//   }
//   else {
//     n.fieldValue -= h_coeff * curlEz;
//   }  
  
// }

// // --- Psi_Hy_x update (left + right safe) ---
// for (int Hy_node_ID = 0; Hy_node_ID < Hy_domain_size; Hy_node_ID++) {
//   auto &n = Hy_nodes[Hy_node_ID];
//   double curlEz = (Ez_nodes[n.ez_right_id].fieldValue - Ez_nodes[n.ez_left_id].fieldValue) / dx;

//   if (n.x > domain_size[2]) { // right PML
//     int raw = static_cast<int>(std::floor((n.x - domain_size[2]) / step_size));
//     int index = clamp_index(raw, static_cast<int>(b_x_r.size()) - 1);
//     n.Psi_Hy_x = b_x_r[index] * n.Psi_Hy_x + c_x_r[index] * curlEz;
//   }
//   else if (n.x < domain_size[0]) { // left PML
//     int raw = static_cast<int>(std::floor((domain_size[0] - n.x) / step_size));
//     int index = clamp_index(raw, static_cast<int>(b_x_l.size()) - 1);
//     n.Psi_Hy_x = b_x_l[index] * n.Psi_Hy_x + c_x_l[index] * curlEz;
//   }
//   else {
//     n.Psi_Hy_x = 0.0;
//   }
// }

// // --- Update Hy (with checks) ---
// for (int Hy_node_ID = 0; Hy_node_ID < Hy_domain_size; Hy_node_ID++) {
//   auto &n = Hy_nodes[Hy_node_ID];
//   double curlEz = (Ez_nodes[n.ez_right_id].fieldValue - Ez_nodes[n.ez_left_id].fieldValue) / dx;

//   if (n.x > domain_size[2]) { // right PML
//     int raw = static_cast<int>(std::floor((n.x - domain_size[2]) / step_size));
//     int index = clamp_index(raw, static_cast<int>(Da_x_r.size()) - 1);

//     double kappa = (kappa_x_r[index] == 0.0) ? 1.0 : kappa_x_r[index];

//     n.fieldValue = Da_x_r[index] * n.fieldValue + Db_x_r[index] * (curlEz / kappa + n.Psi_Hy_x);
//   }
//   else if (n.x < domain_size[0]) { // left PML
//     int raw = static_cast<int>(std::floor((domain_size[0] - n.x) / step_size));
//     int index = clamp_index(raw, static_cast<int>(Da_x_l.size()) - 1);

//     double kappa = (kappa_x_l[index] == 0.0) ? 1.0 : kappa_x_l[index];

//     n.fieldValue = Da_x_l[index] * n.fieldValue + Db_x_l[index] * (curlEz / kappa + n.Psi_Hy_x);
//   }
//   else {
//     n.fieldValue += h_coeff * curlEz;
//   }

//   if (!std::isfinite(n.fieldValue)) {
//     std::cerr << "Hy NaN/Inf at node " << Hy_node_ID << " x=" << n.x << "\n";
//     n.fieldValue = 0.0;
//   }
// }
//     // --- Source injection ---
//     // double time = t * dt;  
//     // Ez_nodes[source_ID].fieldValue += source(time);
//     // // Ez_nodes[source_ID].fieldValue += sin(time);

//     double time = t * dt;
//     double pulse_width = 30 * dt;              // smoothness control
//     double pulse_delay = 6.0 * pulse_width;    // shift so it starts near zero
//     double src = sin(omega * time) * exp(-pow((time - pulse_delay)/pulse_width, 2.0));
//     // double src = sin(omega * time);

//     Ez_nodes[source_ID].fieldValue += amplitude * src;

//     std::cout << "Time step = " << t << std::endl;
//     if (t % data_capture_interval == 0) {
//       std::string type = "Ez";
//       std::string filename = "data/" + type + "/" + type + "_" + std::to_string(t) + ".vtk";
//       saveMeshToVTK(type, filename);
//     }
//   }
// }


/*

0.00437602
0.00820882
0.0111079
0.012884
0.0134916
0.0130541
0.0119575
0.0107431
0.00989492
0.00983963


*/