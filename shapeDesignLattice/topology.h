#pragma once
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/disable_warnings.h>
#include <vector>
#include <memory>
#include "beam.h"
#include "MOGWO.h"
#include "GWOtest.h"
typedef std::array<std::size_t, 3> Facet;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3  Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef Mesh::Face_index face_descriptor;

void locateArcs(std::vector<BeamPlugin*> &beams, Triangle* triList, QuadFace* quadList, ArcType* arcList, short* flag,int nodeid);
void setSegNum(std::vector<BeamPlugin*> &beams, float chordError, short* flag,int& totalSegNum,int& totalArcSampleNum);
int getTopology(std::vector<BeamPlugin*> &beams,Junction* J,Triangle* triList,QuadFace* quadList,short* flag,float* longestLength);
void findAdjTriangles(Triangle* triList, int triNum);
void findQuadFaces(Triangle* triList, QuadFace* quadList, int triNum, bool &inverted);
float measureSingleCost(Triangle* triList, int triNum, QuadFace* quadList, int quadNum, GreyWolves gws, int nVar,bool &inverted);
void measureDistance(Triangle* &triList, QuadFace* &quadList,int triNum,int quadNum,int nObj, GreyWolves& gws, int nVar);
float** getRotationMatrix(float3 diskNorm, float theta, float3 transl);
int mod(float a, float b);
struct Construct {
	Mesh& mesh;
	template < typename PointIterator>
	Construct(Mesh& mesh, PointIterator b, PointIterator e)
		: mesh(mesh)
	{
		for (; b != e; ++b) {
			boost::graph_traits<Mesh>::vertex_descriptor v;
			v = add_vertex(mesh);
			mesh.point(v) = *b;
		}
	}
	Construct& operator=(const Facet f)
	{
		typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;
		typedef boost::graph_traits<Mesh>::vertices_size_type size_type;
		mesh.add_face(vertex_descriptor(static_cast<size_type>(f[0])),
			vertex_descriptor(static_cast<size_type>(f[1])),
			vertex_descriptor(static_cast<size_type>(f[2])));
		return *this;
	}
	Construct&
		operator*() { return *this; }
	Construct&
		operator++() { return *this; }
	Construct
		operator++(int) { return *this; }
};