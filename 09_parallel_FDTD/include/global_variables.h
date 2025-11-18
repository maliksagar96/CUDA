#pragma once
#include <iostream>
#include <json/json.h>

extern const double EPSILON_0;
extern const double MU_0;
extern const double c0;

extern int N_time_steps;

extern double frequency;
extern int data_capture_interval;
extern std::string source_type;
extern std::string STEP_filename;
extern double pulse_delay;
extern double pulse_width;

extern double pml_size;

extern double a_max;
extern double kappa_max;
extern double grading_exponent;
extern double sigma_scale;

extern double simulation_size[4];
extern int cells_per_wavelength;
extern double amplitude;
extern double CFL;

extern double source_point[2];      // source location (x,y)

void read_json(const Json::Value& root);

extern double center_wavelength;
extern double omega;
extern double step_size;
extern double dt;

extern double domain_size[4];
