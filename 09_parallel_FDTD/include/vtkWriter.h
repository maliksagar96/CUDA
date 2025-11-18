#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include <iostream>
#include <vector> 
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>

class VTKUnstructuredWriter {
public:
  VTKUnstructuredWriter(){
    init();
  };
  ~VTKUnstructuredWriter();
  
  /**
   * @brief Set the points of the mesh.
   * 
   */
  virtual void set_points(std::vector<double>&);

  /**
   * @brief Set the connectivity of the mesh.
   * 
   * @param connectivity A vector of integers representing the connectivity of the mesh.
   */
  virtual void set_cells(std::vector<int>&);

  /**
   * @brief Add a scalar field to the mesh.
   * 
   * @param scalars A vector of doubles representing the scalar values.
   * @param name The name of the scalar field.
   */
  virtual void add_scalar(std::vector<double>&, std::string);

  /**
   * @brief Add a vector field to the mesh.
   * 
   * @param vectors A vector of vectors representing the vector values.
   * @param name The name of the vector field.
   */
  virtual void add_vector(std::vector<std::vector<double>>&, std::string);
  /**
   * @brief Write the mesh to a VTK file.
   * 
   * @param filename The name of the output VTK file.
   */
  virtual void write_vtk(std::string);    

protected:
  void init();
  vtkSmartPointer<vtkPoints> points;
  vtkSmartPointer<vtkCellArray> cells;
  vtkSmartPointer<vtkUnstructuredGrid> grid;
};

// Derived classes for all tri mesh types
class VTKTriangleWriter : public VTKUnstructuredWriter {
public:
  void set_cells(std::vector<int>&) override;  
};

// Derived classes for all quad mesh types
class VTKQuadWriter : public VTKUnstructuredWriter {
public:
  void set_cells(std::vector<int>&) override;  
};


#endif
