#pragma once
#include "beam.h"
#include<vector>
#include<Eigen/Dense>
#include <Eigen/Geometry>

#include <random>
#include <ctime>
#include <memory>
#define INF 100000000
struct GreyWolves
{
    public:
		//GreyWolves();
		//~GreyWolves();
		//int triNum,disNum;
		//double Velocity = 0;
		//std::vector<Triangle*> triangles;
		//std::vector<Eigen::Vector3d> Position;
		//std::vector<float> rotationAngle;//真正的Position，一开始随机为一个旋转角度
		float* rotationAngle=NULL;
		float* chaoticC=NULL;
		float* chaoticA=NULL;
		float singleCost;
		bool inverted;
		//std::vector<float> Cost;
		float Cost[2];
		bool Dominated;
		int GridIndex;
		//vector<int> GridSubIndex;
		int GridSubIndex[2];
		int test[2];
		/*vector<float> chaoticC;
		vector<float> chaoticA;*/
};
struct GwsForTriangulation
{
	int separatePos[180];
	float singleCost;
};
struct GwsForShape
{
	float offset[40];
	float singleCost;
};
float3 ComputeNegativeCircumcenter(float3 P[]);
float3 Subtraction(float3 a, float3 b);
float Norm(float3 a);
float Dot(float3 a, float3 b);
float3 make_norm(float3 a);
float4 Multiply(float** m, float4 a);
float3 cross(float3 a, float3 b);
float* computeX123(float* leaderRotationAngle, GreyWolves &gws, int nVar, float a, int it);
void GreyWolfOptimizer(int nVar, Triangle* &triangles, QuadFace* &quads, int triNum,int quadNum,int batchIdx);
void MultiObjectiveGreyWolfOptimizer(int nVar, int nObj, Triangle* &triangles, QuadFace* &quadList,int triNum, int quadNum);
void GWOforQuadTriangulation(int n, int m, float3* sample_n, float3* sample_m, Triangle* resultTri, int triNum, bool isCircle);
void shapeOptimization(std::vector<BeamPlugin*> &beams, float3 node, Triangle* triList, int triNum, short* flag,float* longestLength);
void GWO2(BeamPlugin** beams, int beamNum, short* flag, int nVar, Triangle* triangles,
	QuadFace* quads, int triNum, int quadNum,int batchIdx,int nodeId);