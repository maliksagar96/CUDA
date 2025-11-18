// #include <vtkWriter.h>
#include <Mesh2D.h>
#include <TopoDS.hxx>           
#include <TopoDS_Shape.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <TopExp_Explorer.hxx>
#include <BRepBndLib.hxx>
#include <Bnd_Box.hxx>
#include <STEPControl_Reader.hxx>
#include <BRepTools.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <TopAbs_State.hxx>
#include <gp_Pnt.hxx>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <array>
#include <global_variables.h>

Mesh2D::Mesh2D(const std::string& stepFile, double cellSize)
  : stepFile_(stepFile), cellSize_(cellSize) {}

// ------------------- Read STEP file ------------------- //
bool Mesh2D::readSTEP() {
  STEPControl_Reader reader;
  if (reader.ReadFile(stepFile_.c_str()) != IFSelect_RetDone) {
    std::cerr << "Cannot read STEP file: " << stepFile_ << std::endl;
    return false;
  }

  reader.TransferRoots();
  TopoDS_Shape shape = reader.OneShape();

  // Assume single face (2D sheet)
  for (TopExp_Explorer faceExp(shape, TopAbs_FACE); faceExp.More(); faceExp.Next()) {
    face_ = TopoDS::Face(faceExp.Current());  // <<--- use TopoDS::Face
    return true;
  }


  std::cerr << "No face found in STEP file." << std::endl;
  return false;
}

// ------------------- Check if point is inside face ------------------- // 
bool Mesh2D::isPointInside(const gp_Pnt& p) const {
  BRepClass_FaceClassifier classifier(face_, p, 1e-7);
  TopAbs_State state = classifier.State();
  return (state == TopAbs_IN || state == TopAbs_ON);
}

// ------------------- Generate uniform grid -------------------
void Mesh2D::generateUniformGrid() {
  double xmin = simulation_size[0], xmax = simulation_size[2];
  double ymin = simulation_size[1], ymax = simulation_size[3];

  int id = 0;
  for (double x = xmin; x <= xmax; x += cellSize_) {
    for (double y = ymin; y <= ymax; y += cellSize_) {
      EzNode n_Ez(x, y, id);   
      HxNode n_Hx(x, y - cellSize_/2, id);
      HyNode n_Hy(x - cellSize_/2, y, id);
      Ez_nodes.push_back(n_Ez);
      Hx_nodes.push_back(n_Hx);
      Hy_nodes.push_back(n_Hy);

      id++;
    }
  }
}

// ------------------- Filter cells outside .STEP geometry -------------------//
void Mesh2D::filterNodesOutsideGeometry() {
  std::vector<EzNode> outside_Ez_Nodes;
  std::vector<HxNode> outside_Hx_Nodes;
  std::vector<HyNode> outside_Hy_Nodes;
  
  int id = 0;
  for (auto &n : Ez_nodes) {
    gp_Pnt p(n.x, n.y, 0);  

    if (!isPointInside(p)) {
      n.nodeID = id;        
      n.nodePoint = p;      
      outside_Ez_Nodes.push_back(n);            
      id++;
    }
  }

  id = 0;
  for (auto &n : Hx_nodes) {
    gp_Pnt p(n.x, n.y, 0);  

    if (!isPointInside(p)) {
      n.nodeID = id;        
      n.nodePoint = p;      
      outside_Hx_Nodes.push_back(n);            
      id++;
    }
  }

  id = 0;
  for (auto &n : Hy_nodes) {
    gp_Pnt p(n.x, n.y, 0);  

    if (!isPointInside(p)) {
      n.nodeID = id;        
      n.nodePoint = p;      
      outside_Hy_Nodes.push_back(n);            
      id++;
    }
  }

  Ez_nodes = outside_Ez_Nodes;
  Hx_nodes = outside_Hx_Nodes;
  Hy_nodes = outside_Hy_Nodes;

}

template <typename NodeType>
void linkNeighbors(std::vector<NodeType> &nodes,
                   std::unordered_map<std::string, int> &lookup,
                   double cellSize) {
  auto makeKey = [](double x, double y) {
    return std::to_string(x) + "_" + std::to_string(y);
  };

  // build lookup: key → index
  for (int i = 0; i < (int)nodes.size(); i++) {
    lookup[makeKey(nodes[i].x, nodes[i].y)] = i;
  }

  // assign neighbors
  for (int i = 0; i < (int)nodes.size(); i++) {
    auto &n = nodes[i];
    n.left   = lookup.count(makeKey(n.x - cellSize, n.y)) ? lookup[makeKey(n.x - cellSize, n.y)] : -1;
    n.right  = lookup.count(makeKey(n.x + cellSize, n.y)) ? lookup[makeKey(n.x + cellSize, n.y)] : -1;
    n.top    = lookup.count(makeKey(n.x, n.y + cellSize)) ? lookup[makeKey(n.x, n.y + cellSize)] : -1;
    n.bottom = lookup.count(makeKey(n.x, n.y - cellSize)) ? lookup[makeKey(n.x, n.y - cellSize)] : -1;
  }
}

void Mesh2D::linkNodeNeighbors() {
  std::unordered_map<std::string, int> Ez_lookup;
  std::unordered_map<std::string, int> Hx_lookup;
  std::unordered_map<std::string, int> Hy_lookup;

  linkNeighbors(Ez_nodes, Ez_lookup, cellSize_);
  linkNeighbors(Hx_nodes, Hx_lookup, cellSize_);
  linkNeighbors(Hy_nodes, Hy_lookup, cellSize_);
}

void Mesh2D::linkCrossNeighbors() {
  auto makeKey = [](double x, double y) {
    return std::to_string(x) + "_" + std::to_string(y);
  };

  // --- lookup maps (store index, not pointer) ---
  std::unordered_map<std::string, int> Ez_lookup;
  std::unordered_map<std::string, int> Hx_lookup;
  std::unordered_map<std::string, int> Hy_lookup;

  for (int i = 0; i < (int)Ez_nodes.size(); i++)
    Ez_lookup[makeKey(Ez_nodes[i].x, Ez_nodes[i].y)] = i;
  for (int i = 0; i < (int)Hx_nodes.size(); i++)
    Hx_lookup[makeKey(Hx_nodes[i].x, Hx_nodes[i].y)] = i;
  for (int i = 0; i < (int)Hy_nodes.size(); i++)
    Hy_lookup[makeKey(Hy_nodes[i].x, Hy_nodes[i].y)] = i;

  // --- Ez -> Hx / Hy ---
  for (auto &n : Ez_nodes) {
    n.hx_top_id    = Hx_lookup.count(makeKey(n.x, n.y + cellSize_/2)) ? Hx_lookup[makeKey(n.x, n.y + cellSize_/2)] : -1;
    n.hx_bottom_id = Hx_lookup.count(makeKey(n.x, n.y - cellSize_/2)) ? Hx_lookup[makeKey(n.x, n.y - cellSize_/2)] : -1;

    n.hy_left_id   = Hy_lookup.count(makeKey(n.x - cellSize_/2, n.y)) ? Hy_lookup[makeKey(n.x - cellSize_/2, n.y)] : -1;
    n.hy_right_id  = Hy_lookup.count(makeKey(n.x + cellSize_/2, n.y)) ? Hy_lookup[makeKey(n.x + cellSize_/2, n.y)] : -1;
  }

  // --- Hx -> Ez ---
  for (auto &n : Hx_nodes) {
    n.ez_top_id    = Ez_lookup.count(makeKey(n.x, n.y + cellSize_/2)) ? Ez_lookup[makeKey(n.x, n.y + cellSize_/2)] : -1;
    n.ez_bottom_id = Ez_lookup.count(makeKey(n.x, n.y - cellSize_/2)) ? Ez_lookup[makeKey(n.x, n.y - cellSize_/2)] : -1;
  }

  // --- Hy -> Ez ---
  for (auto &n : Hy_nodes) {
    n.ez_left_id   = Ez_lookup.count(makeKey(n.x - cellSize_/2, n.y)) ? Ez_lookup[makeKey(n.x - cellSize_/2, n.y)] : -1;
    n.ez_right_id  = Ez_lookup.count(makeKey(n.x + cellSize_/2, n.y)) ? Ez_lookup[makeKey(n.x + cellSize_/2, n.y)] : -1;
  }
}

void Mesh2D::add_ghost_layer() {
  Ez_domain_size = Ez_nodes.size();
  Hx_domain_size = Hx_nodes.size();
  Hy_domain_size = Hy_nodes.size();

  std::cout << "---------------------------------\n";
  std::cout << "Ez_domain_size = " << Ez_domain_size << "\n";
  std::cout << "Hx_domain_size = " << Hx_domain_size << "\n";
  std::cout << "Hy_domain_size = " << Hy_domain_size << "\n";
  std::cout << "---------------------------------\n";

  // Stable copies of domain nodes
  std::vector<HxNode> Hx_copy_nodes = Hx_nodes;
  std::vector<HyNode> Hy_copy_nodes = Hy_nodes;

  int hx_id = 0, hy_id = 0, ez_id = 0;

  // --- Ez nodes: create missing Hx/Hy ghosts ---
  for (auto &n : Ez_nodes) {
    if (n.hx_bottom_id == -1) {
      Hx_nodes.emplace_back(n.x, n.y - cellSize_ / 2, Hx_domain_size + hx_id);
      n.hx_bottom_id = Hx_nodes.back().nodeID;
      hx_id++;
    }
    if (n.hx_top_id == -1) {
      Hx_nodes.emplace_back(n.x, n.y + cellSize_ / 2, Hx_domain_size + hx_id);
      n.hx_top_id = Hx_nodes.back().nodeID;
      hx_id++;
    }
    if (n.hy_left_id == -1) {
      Hy_nodes.emplace_back(n.x - cellSize_ / 2, n.y, Hy_domain_size + hy_id);
      n.hy_left_id = Hy_nodes.back().nodeID;
      hy_id++;
    }
    if (n.hy_right_id == -1) {
      Hy_nodes.emplace_back(n.x + cellSize_ / 2, n.y, Hy_domain_size + hy_id);
      n.hy_right_id = Hy_nodes.back().nodeID;
      hy_id++;
    }
  }

  // --- Hx nodes: create missing Ez ghosts ---
  for (int i = 0; i < Hx_domain_size; i++) {
    const auto &orig = Hx_copy_nodes[i];
    if (orig.ez_bottom_id == -1) {
      Ez_nodes.emplace_back(orig.x, orig.y - cellSize_ / 2, Ez_domain_size + ez_id);
      Hx_nodes[i].ez_bottom_id = Ez_nodes.back().nodeID;
      ez_id++;
    }
    if (orig.ez_top_id == -1) {
      Ez_nodes.emplace_back(orig.x, orig.y + cellSize_ / 2, Ez_domain_size + ez_id);
      Hx_nodes[i].ez_top_id = Ez_nodes.back().nodeID;
      ez_id++;
    }
  }

  // --- Hy nodes: create missing Ez ghosts ---
  for (int i = 0; i < Hy_domain_size; i++) {
    const auto &orig = Hy_copy_nodes[i];
    if (orig.ez_left_id == -1) {
      Ez_nodes.emplace_back(orig.x - cellSize_ / 2, orig.y, Ez_domain_size + ez_id);
      Hy_nodes[i].ez_left_id = Ez_nodes.back().nodeID;
      ez_id++;
    }
    if (orig.ez_right_id == -1) {
      Ez_nodes.emplace_back(orig.x + cellSize_ / 2, orig.y, Ez_domain_size + ez_id);
      Hy_nodes[i].ez_right_id = Ez_nodes.back().nodeID;
      ez_id++;
    }
  }

  // record ghost counts
  Ez_ghost_cells = Ez_nodes.size() - Ez_domain_size;
  Hx_ghost_cells = Hx_nodes.size() - Hx_domain_size;
  Hy_ghost_cells = Hy_nodes.size() - Hy_domain_size;
}


// void Mesh2D::saveMeshToVTK(const std::string& type, const std::string& filename) const {
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
