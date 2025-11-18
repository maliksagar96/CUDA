#pragma once

#include <STEPControl_Reader.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Face.hxx>
#include <gp_Pnt.hxx>
#include <vector>
#include <string>

class HxNode;
class HyNode;
class EzNode;

class BaseNode {
public:
  BaseNode(double x_, double y_, int id)
    : x(x_), y(y_), nodeID(id),
      left(-1), right(-1), top(-1), bottom(-1), fieldValue(0) {}
  virtual ~BaseNode() = default;

  int nodeID;
  double x, y;
  double fieldValue;
  gp_Pnt nodePoint;  
  int left, right, top, bottom;
};

class EzNode : public BaseNode {
public:
  EzNode(double x, double y, int id) 
    : BaseNode(x, y, id),
      hx_top_id(-1), hx_bottom_id(-1),
      hy_left_id(-1), hy_right_id(-1) {}
  int hx_top_id, hx_bottom_id, hy_left_id, hy_right_id;
};

class HxNode : public BaseNode {
public:
  HxNode(double x, double y, int id) 
    : BaseNode(x, y, id),
      ez_top_id(-1), ez_bottom_id(-1) {}
  int ez_top_id, ez_bottom_id;
  double Psi_Hx_y = 0.0;

};

class HyNode : public BaseNode {
public:
  HyNode(double x, double y, int id) 
    : BaseNode(x, y, id),
      ez_left_id(-1), ez_right_id(-1) {}
  int ez_left_id, ez_right_id;
  double Psi_Hy_x = 0.0;
};

class Mesh2D {
  public:
    Mesh2D(const std::string& stepFile, double cellSize);
    bool readSTEP();                              // read STEP file and store TopoDS_Face
    void generateUniformGrid();                   // create uniform grid of nodes
    void filterNodesOutsideGeometry();            // keep only inside nodes
    void linkNodeNeighbors();                     // set left/right/top/bottom
    void linkCrossNeighbors();
    void set_PML_parameters();
    void add_ghost_layer();
    void check_nullptr();
    void check_connects();
    void saveMeshToVTK(const std::string&, const std::string&) const;
    void save_Hx_MeshToVTK(const std::string& filename) const;    
    const std::vector<EzNode>& get_Ez_Nodes() const {return Ez_nodes;}
    const std::vector<HxNode>& get_Hx_Nodes() const {return Hx_nodes;}
    const std::vector<HyNode>& get_Hy_Nodes() const {return Hy_nodes;} 
    int get_Ez_domain_size() {return Ez_domain_size;}   
    int get_Hx_domain_size() {return Hx_domain_size;}   
    int get_Hy_domain_size() {return Hy_domain_size;}   
    int get_Ez_ghost_cells() {return Ez_ghost_cells;}
    int get_Hx_ghost_cells() {return Hx_ghost_cells;}
    int get_Hy_ghost_cells() {return Hy_ghost_cells;}    
    void generateEdges();

  private:    
    bool isPointInside(const gp_Pnt& p) const;
    std::string stepFile_;
    double cellSize_;
    TopoDS_Face face_;                            
    std::vector<EzNode> Ez_nodes;
    std::vector<HxNode> Hx_nodes;
    std::vector<HyNode> Hy_nodes;     
    
    

    int Ez_domain_size, Hx_domain_size, Hy_domain_size;
    int Ez_ghost_cells, Hx_ghost_cells, Hy_ghost_cells;
};
