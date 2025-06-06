#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"curand_kernel.h"
#include <stdio.h>
#include "malloc.h"
#include <Eigen/Dense>


extern "C" void call_SearchAgentsComputing(int nVar,GreyWolves* gws, int GreyWolves_num, Triangle* triList, int triNum, QuadFace* quadList, int quadNum);

extern "C" void call_testEigenCUDA(int GreyWolves_num, GreyWolves* gws);

extern "C" void call_NodeComputing(int plines, BeamPlugin** G, int* beamNumVec, float3* positions, 
	Triangle* AllMesh, QuadFace* AllQuad,  Point* Alldots,
	 short* AllFlag, NodeStatistic* statis,int batchIdx,int batchSize,int batchNum);

extern "C" void call_locateArcs_setSegNum(int plines, BeamPlugin** G, int* beamNumVec, Triangle* AllTriangle,
	QuadFace* AllQuad, ArcType* AllArc, short* AllFlag, int* totalSampleNum, int* totalArcSampleNum, float ce);

extern "C" void call_beamTriangulation(int elines, BeamPlugin** beamVector, float3* Allsample, ArcType* AllbeamArc, Triangle* AllresultTri);

extern "C" void call_quadTriangulation(int totalQuadNum, QuadFace* AllQuad, Triangle* AllQuadresultTri);

extern "C" void call_latticePreprocessing(int plines, BeamPlugin** G, int* beamNumVec, float3* positions,int elines, BeamPlugin** beamVector, float threshold);