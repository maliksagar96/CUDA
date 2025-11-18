#include <vtkWriter.h>
#include <vector> 
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkTriangle.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h> 
#include <vtkCellData.h>
#include <iostream>

VTKUnstructuredWriter::~VTKUnstructuredWriter() = default;

void VTKUnstructuredWriter::init() {

  points = vtkSmartPointer<vtkPoints>::New();
  cells = vtkSmartPointer<vtkCellArray>::New();
  grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
}

void VTKUnstructuredWriter::set_points(std::vector<double>& coordinates) {
  points = vtkSmartPointer<vtkPoints>::New();
  for (std::size_t i = 0; i + 2 < coordinates.size(); i += 3) {
      points->InsertNextPoint(coordinates[i], coordinates[i + 1], coordinates[i + 2]);
  }
  grid->SetPoints(points);
}

void VTKUnstructuredWriter::set_cells(std::vector<int>& connectivity) {
  std::cout<<"No defualt definition.\n";
}

void VTKUnstructuredWriter::add_scalar(std::vector<double>& scalar, std::string scalarName) {
  vtkSmartPointer<vtkDoubleArray> dataArray = vtkSmartPointer<vtkDoubleArray>::New();
  dataArray->SetName(scalarName.c_str());
  dataArray->SetNumberOfComponents(1);
  dataArray->SetNumberOfTuples(scalar.size());

  for (vtkIdType i = 0; i < scalar.size(); ++i) {
    dataArray->SetValue(i, scalar[i]);
  }

  grid->GetCellData()->AddArray(dataArray);
}

void VTKUnstructuredWriter::add_vector(std::vector<std::vector<double>>& vect, std::string vectName) {
  if (vect[0].size() != 3) {
    throw std::runtime_error("Each vector must have exactly 3 components.");
  }

  vtkSmartPointer<vtkDoubleArray> vectorArray = vtkSmartPointer<vtkDoubleArray>::New();
  vectorArray->SetName(vectName.c_str());
  vectorArray->SetNumberOfComponents(3);
  vtkIdType numTuples = vect.size();
  vectorArray->SetNumberOfTuples(numTuples);

  for (vtkIdType i = 0; i < numTuples; ++i) {
    vectorArray->SetTuple3(i, vect[i][0], vect[i][1], vect[i][2]);
  }

  grid->GetCellData()->AddArray(vectorArray);
}

void VTKUnstructuredWriter::write_vtk(std::string fname){
  vtkSmartPointer<vtkUnstructuredGridWriter> writer = vtkSmartPointer<vtkUnstructuredGridWriter>::New();
  writer->SetFileName(fname.c_str());
  writer->SetInputData(grid);
  writer->Write();
  return;
}

/**********For Triangle Files**********/
void VTKTriangleWriter::set_cells(std::vector<int>& connectivity) {
  cells = vtkSmartPointer<vtkCellArray>::New();
  for(int i = 0;i+2<connectivity.size();i+=3) {
    cells->InsertNextCell(3);
    cells->InsertCellPoint(connectivity[i]); cells->InsertCellPoint(connectivity[i+1]); cells->InsertCellPoint(connectivity[i+2]);
  }
  grid->SetCells(VTK_TRIANGLE, cells);    
}

/**********For Quad Files**********/
void VTKQuadWriter::set_cells(std::vector<int>& connectivity) {
  for (int i = 0; i + 3 < connectivity.size(); i += 4) {
  cells->InsertNextCell(4);
  cells->InsertCellPoint(connectivity[i]);
  cells->InsertCellPoint(connectivity[i+1]);
  cells->InsertCellPoint(connectivity[i+2]);
  cells->InsertCellPoint(connectivity[i+3]);
}
grid->SetCells(VTK_QUAD, cells);
  
}
