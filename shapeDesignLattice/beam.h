#pragma once
#include<Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define PI 3.14159265358979323846
using namespace std;
struct Triangle;
struct QuadFace;

////法线
//typedef struct Normal
//{
//	double nx;
//	double ny;
//	double nz;
//} Normal;
// 
//点
typedef struct Point
{
	short id;
	bool isVisited;
	double x;
	double y;
	double z;
} Point;
 
//面
typedef struct Face
{
	float3 normal;
	float3 p1;
	float3 p2;
	float3 p3;
	short inverted;
	short type;
	//short flag;
} Face;
//边
typedef struct Edge
{
	float3 p[2];
	float3 middle;
} Edge;
//三角形
struct Triangle
{
    public:
		
		//~Triangle();
		int triId,nodeid;
		float3 normal;
		float3 originNormal;
		float3 p[3];

		int triWithBeamid[3];
		Triangle* adjTriangles[3];
		short adjTriNum;
		QuadFace* adjQuadFaces[3];
		short adjQuadNum;
		//连接的三个杆的圆心坐标
		float3 o[3];
		float3 diskNorm[3];
		//每个三角形的最小距离
		float cost; float projSeg;
		//区分quadFace是否共点
		short flag;
		float rotationAngle[3];
		short inverted[2];
		//__device__ Triangle()
		//{
		//}
		//__device__ Triangle(Triangle &T)
		//{
		//	triId = T.triId;
		//	/*normal = T.normal;
		//	originNormal = T.originNormal;
		//	for (int i = 0; i < 3; i++)
		//	{
		//		p[i] = T.p[i];
		//		o[i] = T.o[i];
		//		diskNorm[i] = T.diskNorm[i];
		//	}
		//	triWithBeamid = T.triWithBeamid;
		//	adjTriangles = T.adjTriangles;
		//	adjTriNum = T.adjTriNum;
		//	adjQuadFaces = T.adjQuadFaces;
		//	adjQuadNum = T.adjQuadNum;
		//	cost = T.cost;*/
		//}
};
struct ArcType
{
	float3 st;
	float3 ed;
	float3 o;
	float theta;
	int sampleNum;
	float segTheta;
	float3 diskNorm;
	float3* sample;
};
struct QuadFace
{
public:
	
	//~QuadFace();
	int quadId,nodeid;
	int quadWithBeamid[2];
	Triangle t[4];
	//std::vector<QuadFace> adjQuadFaces;
	float width[2];
	ArcType* arc[2];
	short F[2];
	float COS[4];
	short inverted[6];
	//short convergeF;
	/*__device__ QuadFace();
	__device__ QuadFace(const QuadFace& Q)
	{
		quadId = Q.quadId;
		quadWithBeamid = Q.quadWithBeamid;
		for (int i = 0; i < 2; i++)
		{
			t[i] = Q.t[i];
			width[i] = Q.width[i];
		}
	}*/
};



struct BeamPlugin
{
    public:
		/*BeamPlugin();
		~BeamPlugin();*/

		bool BeamConstruct(Edge etmp,double r);
		float** getMatrix();
		void BeamTransform();

		int beamId;
		double cutLength,cutOtherLength;
		float convexLength[2];
		float longestLength[2];
		float minAngle[2];
		double radius, length,originLength;
		int samples;
		Edge axis;
		short arcTriList[2][50];
		short arcNum[2];
		ArcType* arcArray[2][50];
		int nodeid[2];
		//short convergeF[2];
		//std::vector<Eigen::Vector3i> tris1Vec, tris2Vec;//记录两个端点处的三角形并按序排列
		//std::vector<Eigen::Vector3d> end1Vec, end2Vec;//记录两个端点处的细分节点位置
		//std::vector<Face> f;
};

struct Junction
{
    public:
		Junction();
		~Junction();
		//bool JunctionConstruct();

		int nodeId;
		float3 position;//用以判断beam的哪个端点连接到这个junction
		//int numOfBeams;
		//std::vector<Face> f;//节点处的所有面
		short triNum;
		short quadNum;
		Face* f;

};


struct EigenCUDA
{
public:
	int a;
	int* b = NULL;
	Eigen::Vector3d p;
};
typedef bool(*Compare)(BeamPlugin a, BeamPlugin b);
struct Heap {
	BeamPlugin b[40];
	int heapSize;
	Compare cmp;
};

struct NodeStatistic {
	int degree;
	int converge;
	int iterNum;
	float face_angle;
	float face_angle_variance;
	float disToSphere,circleDis;
};

struct Data {
	BeamPlugin** BatchBeams;
	short* BatchFlag;
	Triangle* AllMesh;
	QuadFace* AllQuad;
	int batchIdx;
	int batchTriNum;
	int batchQuadNum;
	int batchBeamNum;
};