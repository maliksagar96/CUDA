#include <global_variables.h>
#include <iostream>
#include <json/json.h>
#include <cmath>

const double EPSILON_0 = 8.8541878128e-12;
const double MU_0      = 1.256637062e-6;
const double c0        = 2.99792458e8;

int N_time_steps;
double frequency;

int data_capture_interval;
std::string source_type;
std::string STEP_filename;
double pml_size;
double simulation_size[4];
int cells_per_wavelength;
double amplitude;
double CFL;
double source_point[2];      // source location (x,y)
double pulse_delay;
double pulse_width;
double center_wavelength;
double omega;
double step_size;
double dt;
double domain_size[4];
double kappa_max;
double a_max;
double grading_exponent;
double sigma_scale;

void read_json(const Json::Value& root) {
	STEP_filename = root["filename"].asString();
	frequency = root["frequency"].asDouble();
	N_time_steps = root["Iterations"].asInt();
	data_capture_interval = root["data_capture_interval"].asInt();
  
  source_type = root["source_type"].asString();

  pml_size = root["pml_region"].asDouble();  

  domain_size[0] = root["domain_region"][0].asDouble();
  domain_size[1] = root["domain_region"][1].asDouble();
  domain_size[2] = root["domain_region"][2].asDouble();
  domain_size[3] = root["domain_region"][3].asDouble();

	simulation_size[0] = domain_size[0] - pml_size;
  simulation_size[1] = domain_size[1] - pml_size;
  simulation_size[2] = domain_size[2] + pml_size;
  simulation_size[3] = domain_size[3] + pml_size;

	cells_per_wavelength = root["cells_per_wavelength"].asInt();
  amplitude = root["amplitude"].asDouble();
  CFL = root["CFL"].asDouble();

  pulse_width = 1.0 / frequency;
  pulse_delay = 3*pulse_width; 

  source_point[0] = root["source_point"][0].asDouble();
  source_point[1] = root["source_point"][1].asDouble();

  grading_exponent = root["grading_exponent"].asDouble();
  kappa_max = root["kappa_max"].asDouble();
  a_max = root["a_max"].asDouble();
  sigma_scale = root["sigma_scale"].asDouble();

	if((source_point[0] < simulation_size[0] || source_point[0] > simulation_size[2] || 
      source_point[1] < simulation_size[1] || source_point[1] > simulation_size[3])) {
    std::cerr<<"Source point not inside the simualtion domain."<<std::endl;
    exit(1);
  }

  center_wavelength = c0/frequency;      
  omega = 2 * M_PI * frequency;
  step_size = center_wavelength / cells_per_wavelength;
  dt = CFL / (c0 * sqrt((1.0/(step_size*step_size)) + (1.0/(step_size*step_size))));

	std::cout<<"************JSON Parameters***********************"<<std::endl;
  std::cout<<"Frequency = "<<frequency<<std::endl;
  std::cout<<"Iterations = "<<N_time_steps<<std::endl;
  std::cout<<"data_capture_interval = "<<data_capture_interval<<std::endl;
  std::cout<<"simulation_size = "<<simulation_size[0]<<", "<<simulation_size[1]<<", "<<simulation_size[2]<<", "<<simulation_size[3]<<std::endl;
  std::cout<<"source_point = "<<source_point[0]<<", "<<source_point[1]<<std::endl;
  std::cout<<"cells_per_wavelength = "<<cells_per_wavelength<<std::endl;
  std::cout<<"amplitude = "<<amplitude<<std::endl;
  std::cout<<"CFL = "<<CFL<<std::endl;
  std::cout<<"pml_size = "<<pml_size<<std::endl;
  std::cout<<"Simulation time = "<<dt<<std::endl;
  std::cout<<"Pulse delay = "<<pulse_delay<<std::endl;
  std::cout<<"Reading JSON file completed."<<std::endl;

}