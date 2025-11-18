void FDTD_2D::TMz_mesh_update() {
  
  getSourceID();

  // Make arrays of Ez, Hx and Hy CPU copy   
  double* Hy_host = new double[Hy_domain_size];
  double* Hx_host = new double[Hx_domain_size];
  double* Ez_host = new double[Ez_domain_size];

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
  }

  for (int i = 0; i < Hx_domain_size; i++) {
    hx_top_neighbour[i]    = Hx_nodes[i].ez_top_id;
    hx_bottom_neighbour[i] = Hx_nodes[i].ez_bottom_id;
  }

  for (int i = 0; i < Ez_domain_size; i++) {
    ez_top_neighbour[i]    = Ez_nodes[i].hy_top_id;
    ez_bottom_neighbour[i] = Ez_nodes[i].hy_bottom_id;
    ez_left_neighbour[i]   = Ez_nodes[i].hx_left_id;
    ez_right_neighbour[i]  = Ez_nodes[i].hx_right_id;
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

  size_t hy_byte_double = Hy_domain_size * sizeof(double);
  size_t hx_byte_double = Hx_domain_size * sizeof(double);
  size_t ez_byte_double = Ez_domain_size * sizeof(double);

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

  int THREADS = 256;
  int hy_blocks = (Hy_domain_size + THREADS - 1) / THREADS;
  int hx_blocks = (Hx_domain_size + THREADS - 1) / THREADS;
  int ez_blocks = (Ez_domain_size + THREADS - 1) / THREADS;

  for (int t = 0; t < N_time_steps; t++) {

    // --- Update Ez on CPU---
    for (int Ez_node_ID = 0; Ez_node_ID < Ez_domain_size; Ez_node_ID++) {
      auto &n = Ez_nodes[Ez_node_ID];
      double dHy_dx = (Hy_nodes[n.hy_right_id].fieldValue - Hy_nodes[n.hy_left_id].fieldValue) / dx;        
      double dHx_dy = (Hx_nodes[n.hx_top_id].fieldValue - Hx_nodes[n.hx_bottom_id].fieldValue) / dy;            
      n.fieldValue += e_coeff * (dHy_dx - dHx_dy);
    }

    // --- Update Hx on CPU---
    for (int Hx_node_ID = 0; Hx_node_ID < Hx_domain_size; Hx_node_ID++) {
      auto &n = Hx_nodes[Hx_node_ID];
      double curlEz = (Ez_nodes[n.ez_top_id].fieldValue - Ez_nodes[n.ez_bottom_id].fieldValue) / dy;              
      n.fieldValue -= h_coeff * curlEz;
    }

    // --- Update Hy on CPU---
    //Replace with update Hy kernel
    // for (int Hy_node_ID = 0; Hy_node_ID < Hy_domain_size; Hy_node_ID++) {
    //   auto &n = Hy_nodes[Hy_node_ID];
    //   double curlEz = (Ez_nodes[n.ez_right_id].fieldValue - Ez_nodes[n.ez_left_id].fieldValue) / dx;    
    //   n.fieldValue += h_coeff * curlEz;
    // }

    // --- Update Hy on GPU ---
    update_Hy<<<hy_blocks, THREADS>>>(d_Hy, d_Ez, d_hy_left_neighbour, d_hy_right_neighbour, Hy_domain_size);
    

    // --- Source Injection on CPU --- 
    double time = t * dt;
    double pulse_width = 10 * dt;              // smoothness control
    double pulse_delay = 6.0 * pulse_width;    // shift so it starts near zero
    double src = sin(omega * time) * exp(-pow((time - pulse_delay)/pulse_width, 2.0));
    Ez_nodes[source_ID].fieldValue += amplitude * src;
    // std::cout << "Time step = " << t << std::endl;

    //Source injection on GPU
    gpu_source_injection<<<1,1>>>(time, pulse_width, pulse_delay, omega, amplitude);

    cudaDeviceSynchronize();  

  }
  
  //copy back Ez, Hx and Hy info on CPU

  for (int Hy_node_ID = 0; Hy_node_ID < 10; Hy_node_ID++) {
    auto &n = Hy_nodes[Hy_node_ID];
    cout<<n.fieldValue<<endl;
  }

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
