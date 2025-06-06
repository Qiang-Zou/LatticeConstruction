#pragma once
#include "MOGWO.h"
#include "topology.h"
#define NOMINMAX
//#include <graphics.h>		// 引用 EasyX 绘图库头文件
#include <conio.h>
#include <math.h>
//#include <windows.h>
#include <stdio.h>
//#include <easyx.h>	
//#include "drawCoordinateSystem.h"
#include "queue.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include"curand_kernel.h"
#include "device_launch_parameters.h"
#include "cutil.h"
#include "call_cuda.h"

#define PI 3.14159265

//#define PARALLEL
//#define TEST
//GreyWolves::GreyWolves()
//{
//	//disNum = triNum / 2 * 3;
//
//}

//GreyWolves::~GreyWolves()
//{
//
//}

bool Dominates(float* x, float* y,int nObj)
{
	//bool dom = true;
	int costNum = nObj;
	/*if (x[0] < y[0]|| x[4] < y[4])
		return false;*/
	/*if (x[0]>=y[0]&&x[4]>=y[4])
		return true;*/

	for (int i = 0; i < costNum; i++)
	{
		
		if (x[i] > y[i])
		{
			return false;
		}
	}
	for (int i = 0; i < costNum; i++)
	{
		if (x[i] < y[i])
		{
			return true;
		}
	}
	return false;
	/*if (x[1] < y[1])
		return false;

	if (x[1] > y[1])
		return true;*/
}

void DetermineDomination(std::vector<GreyWolves> &gws,int GreyWolves_num,int nObj)
{
	for (int i = 0; i < GreyWolves_num; i++)
	{
		gws[i].Dominated = false;
		for (int j = 0; j <= i - 1; j++)
		{
			if (!gws[j].Dominated)
			{
				if (Dominates(gws[i].Cost, gws[j].Cost,nObj))
				{
					gws[j].Dominated = true;
				}
				else if (Dominates(gws[j].Cost, gws[i].Cost,nObj))
				{
					gws[i].Dominated = true;
					break;
				}
			}
		}
	}
}
std::vector<GreyWolves> GetNonDominatedParticles(std::vector<GreyWolves> &gws,int GreyWolves_num,int nVar)
{
	std::vector<GreyWolves> Archive;
	for (int i = 0; i < GreyWolves_num; i++)
	{
		if (!gws[i].Dominated)
		{
			GreyWolves A;
			A = gws[i];
			A.rotationAngle = new float[nVar];
			for (int j = 0; j < nVar; j++)
			{
				A.rotationAngle[j] = gws[i].rotationAngle[j];
			}

			Archive.push_back(A);
		}
	}
	return Archive;
}
std::vector<float>* GetCosts(std::vector<GreyWolves> Archive,int nObj)
{
	//int nObj = Archive[0].Cost.size();
	int ArchiveSize = Archive.size();
	std::vector<float>* cost = new std::vector<float>[nObj];
	for (int i = 0; i < nObj; i++)
	{
		for (int j = 0; j < ArchiveSize; j++)
		{
			cost[i].push_back(Archive[j].Cost[i]);
		}
	}
	return cost;
}
std::vector<float> * CreateHypercubes(std::vector<float> *Archive_costs, int nObj,int nGrid, float alpha)
{
	std::vector<float>* G = new std::vector<float>[nObj];
	for (int i = 0; i < nObj; i++)
	{
		float min_cj = *std::min_element(Archive_costs[i].begin(), Archive_costs[i].end());
		float max_cj = *std::max_element(Archive_costs[i].begin(), Archive_costs[i].end());
		float dcj = alpha * (max_cj - min_cj);
		min_cj = min_cj - dcj;
		max_cj = max_cj + dcj;
		//等差数列
		float d = (max_cj - min_cj) / (nGrid - 2);
		for (int j = 0; j < nGrid - 1; j++)
		{
			float tmp = min_cj + j * d;
			G[i].push_back(tmp);
		}
		G[i].push_back(INF);
	}
	return G;
}
void GetGridIndex(GreyWolves &particle,std::vector<float>* G,int nObj)
{
	float* c = particle.Cost;
	//int nObj = c.size();
	int nGrid = G[0].size();

	//10x10x10...共nObj个10相乘的超空间
	//SubIndex
	//int* SubIndex = new int[nObj];
	int SubIndex[2];
	for (int i = 0; i < nObj; i++)
	{
		for (int j = 0; j < nGrid; j++)
		{
			if (c[i] < G[i][j])
			{
				SubIndex[i] = j;//得到行列索引
			}
		}
	}
	//particle.GridSubIndex = new int[nObj];
	//particle.GridSubIndex.resize(nObj);
	for (int i = 0; i < nObj; i++)
	{
		
		particle.GridSubIndex[i] = SubIndex[i];
	}
	
	//计算线性索引
	int Index=0;
	for (int i = 0; i < nObj; i++)
	{
		Index += SubIndex[i] * pow(nGrid,nObj-1-i);
	}
	particle.GridIndex = Index;

	//delete[] SubIndex;
}
void GetOccupiedCells(std::vector<GreyWolves> Archive, std::vector<int> &occ_cell_index, std::vector<int> &occ_cell_member_count)
{
	std::vector<int> GridIndices;
	int ArchiveSize = Archive.size();
	for (int i = 0; i < ArchiveSize; i++)
	{
		GridIndices.push_back(Archive[i].GridIndex);
	}
	occ_cell_index = GridIndices;
	sort(occ_cell_index.begin(), occ_cell_index.end());
	auto it = unique(occ_cell_index.begin(), occ_cell_index.end());
	occ_cell_index.erase(it, occ_cell_index.end());

	int m = occ_cell_index.size();
	for (int k = 0; k < m; k++)
	{
		int cnt = 0;
		for (int i = 0; i < ArchiveSize; i++)
		{
			if (GridIndices[i] == occ_cell_index[k])
				cnt++;
		}
		occ_cell_member_count.push_back(cnt);
	}
}
GreyWolves SelectLeader(std::vector<GreyWolves> Archive, float beta,int &h)
{
	std::vector<int> occ_cell_index;
	std::vector<int> occ_cell_member_count;
	GetOccupiedCells(Archive, occ_cell_index, occ_cell_member_count);

	std::vector<float> p;
	int m = occ_cell_member_count.size();
	for (int i = 0; i < m; i++)
	{
		float pTmp = pow((float)occ_cell_member_count[i], -beta);
		p.push_back(pTmp);
	}
	float sum = accumulate(p.begin(), p.end(),0.0);
	for (int i = 0; i < m; i++)
	{
		p[i] = p[i] / sum;
	}
	//计算p的累积和
	std::vector<float> c; c.resize(m);
	for (int i = 0; i < m; i++)
	{
		c[i] = 0;
		for (int j = 0; j <= i; j++)
		{
			c[i] += p[j];
		}
	}
	std::default_random_engine e;
	std::uniform_real_distribution<float> u(0, 1);
	e.seed(time(0));
	float r = u(e);//返回从区间 (0,1) 的均匀分布中得到的随机标量
	int idx;
	for (idx = 0; idx < m; idx++)
	{
		if (r <= c[idx])
		{
			break;
		}
	}
	int selected_cell_index = occ_cell_index[idx];//选中的线性索引

	std::vector<int> GridIndices;
	int ArchiveSize = Archive.size();
	for (int i = 0; i < ArchiveSize; i++)
	{
		GridIndices.push_back(Archive[i].GridIndex);
	}

	std::vector<int> selected_cell_members;
	for (int i = 0; i < ArchiveSize; i++)
	{
		if (selected_cell_index == GridIndices[i])
		{
			selected_cell_members.push_back(i);
		}
	}
	int n = selected_cell_members.size();
	//srand(time(0));
	std::default_random_engine e2;
	std::uniform_int_distribution<int> u2(0, n-1); // 左闭右闭区间
	e2.seed(time(0));
	int selected_memebr_index = u2(e2);
	h = selected_cell_members[selected_memebr_index];
	return Archive[h];
}


float* computeX123(float* leaderRotationAngle,GreyWolves &gws,int nVar,float a,int it)
{
	
	//% Eq.(3.4) in the paper
	float* c = (float*)malloc(nVar*sizeof(float));
	
	/*if (it == 0)
	{
		gws.chaoticC.clear();
		gws.chaoticA.clear();
	}*/
	//srand((unsigned)time(NULL));
	
	for (int i = 0; i < nVar; i++)
	{
		float rndC;
		if (it == 0)
		{
			/*std::default_random_engine e;
			std::uniform_real_distribution<> u(0.0, nextafter(1.0, DBL_MAX));
			e.seed(time(0));*/
			//Eigen::MatrixXd MC = Eigen::MatrixXd::Random(1, 1);
			curandState devStates;
			/*srand(time(0));
			int seed = rand();*/
			curand_init((unsigned long long)clock(), i, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//curand_init((unsigned long long)(seed*1e9), i, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution

			rndC = abs(RANDOM); //if(rndC==1.0)cout << rndC << endl;
			c[i] = (2 * rndC);
			//printf("%f ", RANDOM);
			//gws.chaoticC.push_back(rndC);
		}
		//else
		//{
		//	rndC = cos(0.5*acos(gws.chaoticC[i]));
		//	//rndC = 0.5*gws.chaoticC[i] * (1 - gws.chaoticC[i]);
		//	c.push_back(2 * rndC);
		//	gws.chaoticC[i] = rndC;
		//}
		
	}
	float* D = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		D[i] = (abs(c[i] * leaderRotationAngle[i] - gws.rotationAngle[i]));
	}
	float* A = (float*)malloc(nVar * sizeof(float));
	
	for (int i = 0; i < nVar; i++)
	{
		float rndA;
		if (it == 0)
		{
			/*std::default_random_engine e;
			std::uniform_real_distribution<> u(0.0, nextafter(1.0, DBL_MAX));
			e.seed(time(0));*/
			//srand(time(0));
			//Eigen::MatrixXd MA = Eigen::MatrixXd::Random(1, 1);
			curandState devStates;
			/*srand(time(0));
			int seed = rand();*/
			curand_init((unsigned long long)clock(), i, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//curand_init((unsigned long long)(seed*1e9), i, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution

			rndA = abs(RANDOM);
			A[i] = (2 * a*rndA - a);
			//gws.chaoticA.push_back(rndA);
		}
		//else
		//{
		//	rndA = cos(0.5*acos(gws.chaoticA[i]));
		//	//rndA = 0.5*gws.chaoticA[i] * (1 - gws.chaoticA[i]);
		//	A.push_back(2 * a*rndA - a);
		//	gws.chaoticA[i] = rndA;
		//}
		
	}
	float* X = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		X[i] = (leaderRotationAngle[i] - A[i] * abs(D[i]));
		//printf("%f ", X[i]);
	}
	free(c);
	free(D);
	free(A);
	
	return X;
}
void DeleteFromRep(std::vector<GreyWolves> &Archive,int EXTRA,int gamma)
{
	for (int k = 0; k < EXTRA; k++)
	{
		std::vector<int> occ_cell_index;
		std::vector<int> occ_cell_member_count;
		GetOccupiedCells(Archive, occ_cell_index, occ_cell_member_count);

		std::vector<float> p;
		int m = occ_cell_member_count.size();
		for (int i = 0; i < m; i++)
		{
			float pTmp = pow((float)occ_cell_member_count[i], gamma);
			p.push_back(pTmp);
		}
		float sum = accumulate(p.begin(), p.end(), 0.0);
		for (int i = 0; i < m; i++)
		{
			p[i] = p[i] / sum;
		}

		//计算p的累积和
		std::vector<float> c; c.resize(m);
		for (int i = 0; i < m; i++)
		{
			c[i] = 0;
			for (int j = 0; j <= i; j++)
			{
				c[i] += p[j];
			}
		}
		std::default_random_engine e;
		std::uniform_real_distribution<float> u(0, 1);
		e.seed(time(0));
		float r = u(e);//返回从区间 (0,1) 的均匀分布中得到的随机标量
		int idx;
		for (idx = 0; idx < m; idx++)
		{
			if (r <= c[idx])
			{
				break;
			}
		}
		int selected_cell_index = occ_cell_index[idx];//选中的线性索引

		std::vector<int> GridIndices;
		int ArchiveSize = Archive.size();
		for (int i = 0; i < ArchiveSize; i++)
		{
			GridIndices.push_back(Archive[i].GridIndex);
		}

		std::vector<int> selected_cell_members;
		for (int i = 0; i < ArchiveSize; i++)
		{
			if (selected_cell_index == GridIndices[i])
			{
				selected_cell_members.push_back(i);
			}
		}
		int n = selected_cell_members.size();
		//srand(time(0));
		std::default_random_engine e2;
		std::uniform_int_distribution<int> u2(0, n - 1); // 左闭右闭区间
		e2.seed(time(0));
		int selected_memebr_index = u2(e2);
		int h = selected_cell_members[selected_memebr_index];

		auto iter = Archive.begin() + h;
		Archive.erase(iter);
	}
}
GreyWolves selectSolution(Triangle* &triList,int triNum, QuadFace* &quadList,int &quadNum,
	std::vector<GreyWolves> Archive,float beta,Triangle* &triangles,int &H,int nObj,int nVar)
{
	//取得Alpha
	//int H;
	GreyWolves Alpha = SelectLeader(Archive, beta, H);

	//计算cost
	//根据旋转角度计算三角形顶点坐标
	//拷贝一份triangles，暂时不能真正旋转
	/*std::vector<Triangle*> triList;
	std::vector<QuadFace*> quadList;*/
	//int triNum = triangles.size();
	//Triangle* triTmp = new Triangle[triNum];
	//vector<shared_ptr<Triangle>> triTmp;
	//triTmp.resize(triNum);
	triList = (Triangle*)malloc(triNum * sizeof(Triangle));
	quadList = (QuadFace*)malloc(quadNum*sizeof(QuadFace));
	
	for (int j = 0; j < triNum; j++)
	{
		triList[j] = triangles[j];
		//triList.push_back(&triTmp[j]);
	}
	//计算在当前三角形位置上旋转随机角度后的位置
	for (int j = 0; j < triNum; j++)
	{
		Triangle t = triList[j];
		for (int k = 0; k < 3; k++)
		{
			float3 diskNorm = t.diskNorm[k];
			float theta = Alpha.rotationAngle[j * 3 + k];
			float3 O = t.o[k];
			float** m = getRotationMatrix(diskNorm, theta, O);

			float3 rotateV = make_float3(t.p[k].x - O.x, t.p[k].y - O.y, t.p[k].z - O.z);
			float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
			RV = Multiply(m , RV);
			t.p[k] = make_float3(RV.x,RV.y,RV.z);
		}
		float3 e1 = make_float3(t.p[1].x - t.p[0].x, t.p[1].y - t.p[0].y, t.p[1].z - t.p[0].z);
		float3 e2 = make_float3(t.p[2].x - t.p[0].x, t.p[2].y - t.p[0].y, t.p[2].z - t.p[0].z);
		t.normal = cross(e1,e2);
		t.normal = make_norm(t.normal);
		triList[j] = t;

		//triList.push_back(triTmp[j]);
	}
	//计算cost
	bool inverted = false;
	findAdjTriangles(triList,triNum);//否则找到的邻居三角形还是以前的
	findQuadFaces(triList, quadList,triNum, inverted);
	measureDistance(triList, quadList,triNum,quadNum,nObj,Alpha,nVar);
	Alpha.inverted = inverted;
	/*free(triList);
	free(quadList);*/

	return Alpha;
}
float measureShapeCost(float originR,std::vector<BeamPlugin*> &beams,float3 node, Triangle* triList, int triNum,GwsForShape gws,int nVar,short* flag)
{
	int beamNum = beams.size();
	//计算到原始球面的距离
	float disToSphere = 0;
	for (int i = 0; i < nVar; i++)
	{
		BeamPlugin* b = beams[i];
		//int idx = flag[i] - 1;
		/*float offset = gws.offset[i];
		float dis = abs(Norm(Subtraction(b->axis.p[idx], node))+offset-originR);*/
		disToSphere += gws.offset[i];
	}

	//计算连接三角形和三个圆面的法线夹角
	/*float disDot = 0;
	for (int i = 0; i < triNum; i++)
	{
		Triangle* t = &triList[i];
		BeamPlugin* b[3];
		float3 v[3];
		float3 diskNorm[3];
		for (int j = 0; j < 3; j++)
		{
			b[j] = beams[t->triWithBeamid[j]];
			diskNorm[j] = make_float3(-t->diskNorm[j].x, -t->diskNorm[j].y, -t->diskNorm[j].z);
			float offset = gws.offset[t->triWithBeamid[j]];
			int idx = flag[t->triWithBeamid[j]] - 1;
			v[j].x = b[j]->axis.p[idx].x+diskNorm[j].x*offset;
			v[j].y = b[j]->axis.p[idx].y + diskNorm[j].y*offset;
			v[j].z = b[j]->axis.p[idx].z + diskNorm[j].z*offset;
		}
		float3 e1 = Subtraction(v[1], v[0]);
		float3 e2 = Subtraction(v[2], v[0]);
		float3 normal = make_norm(cross(e1, e2));
		for (int j = 0; j < 3; j++)
		{
			disDot += pow(Dot(diskNorm[j], normal) - 1, 2);
		}
	}*/
	float circleDis = 0,circleAngle=0;
	//for (int i = 0; i < beamNum - 1; i++)
	//{
	//	for (int j = i + 1; j < beamNum; j++)
	//	{
	//		BeamPlugin b1 = *beams[i];
	//		BeamPlugin b2 = *beams[j];
	//		float3 direct1 = make_norm(Subtraction(b1.axis.p[flag[i] - 1], node));
	//		float3 direct2 = make_norm(Subtraction(b2.axis.p[flag[j] - 1], node));
	//		int idx1 = flag[i] - 1, idx2 = flag[j] - 1;

	//		b1.axis.p[idx1].x = b1.axis.p[idx1].x + direct1.x*gws.offset[i];
	//		b1.axis.p[idx1].y = b1.axis.p[idx1].y + direct1.y*gws.offset[i];
	//		b1.axis.p[idx1].z = b1.axis.p[idx1].z + direct1.z*gws.offset[i];

	//		b2.axis.p[idx2].x = b2.axis.p[idx2].x + direct2.x*gws.offset[j];
	//		b2.axis.p[idx2].y = b2.axis.p[idx2].y + direct2.y*gws.offset[j];
	//		b2.axis.p[idx2].z = b2.axis.p[idx2].z + direct2.z*gws.offset[j];

	//		float3 cross1 = cross(direct1, direct2);
	//		if (cross1.x == 0 && cross1.y == 0 && cross1.z == 0)
	//		{
	//			circleDis += Norm(Subtraction(b1.axis.p[idx1], b2.axis.p[idx2]));
	//			continue;
	//		}
	//		float3 v1 = make_norm(cross(cross1, direct1));
	//		//printf("%f %f %f\n", v1.x, v1.y, v1.z);
	//		v1 = make_float3(b1.radius*v1.x, b1.radius*v1.y, b1.radius*v1.z);
	//		v1 = make_float3(b1.axis.p[idx1].x + v1.x, b1.axis.p[idx1].y + v1.y, b1.axis.p[idx1].z + v1.z);

	//		float3 cross2 = cross(direct2, direct1);
	//		float3 v2 = make_norm(cross(cross2, direct2));
	//		//printf("%f %f %f\n", v2.x, v2.y, v2.z);
	//		v2 = make_float3(b2.radius*v2.x, b2.radius*v2.y, b2.radius*v2.z);
	//		v2 = make_float3(b2.axis.p[idx2].x + v2.x, b2.axis.p[idx2].y + v2.y, b2.axis.p[idx2].z + v2.z);

	//		circleDis += Norm(Subtraction(v1, v2));
	//		
	//		
	//	}
	//}
	float needCut = 0;
	for (int i = 0; i < triNum; i++)
	{
		Triangle* t = &triList[i];
		BeamPlugin b[3];
		//float3 v[3];
		float3 diskNorm[3];
		for (int j = 0; j < 3; j++)
		{
			b[j] = *beams[t->triWithBeamid[j]];
			diskNorm[j] = make_float3(-t->diskNorm[j].x, -t->diskNorm[j].y, -t->diskNorm[j].z);
			float offset = gws.offset[t->triWithBeamid[j]];
			int idx = flag[t->triWithBeamid[j]] - 1;
			b[j].axis.p[idx].x = b[j].axis.p[idx].x + diskNorm[j].x*offset;
			b[j].axis.p[idx].y = b[j].axis.p[idx].y + diskNorm[j].y*offset;
			b[j].axis.p[idx].z = b[j].axis.p[idx].z + diskNorm[j].z*offset;
		}
		for (int j = 0; j < 2; j++)
		{
			for (int k = j+1; k < 3; k++)
			{
				int idx1 = flag[t->triWithBeamid[j]] - 1, idx2 = flag[t->triWithBeamid[k]] - 1;

				float3 direct1 = make_norm(Subtraction(b[j].axis.p[flag[idx1] - 1], node));
				float3 direct2 = make_norm(Subtraction(b[k].axis.p[flag[idx2] - 1], node));

				float3 cross1 = cross(direct1, direct2);
				if (cross1.x == 0 && cross1.y == 0 && cross1.z == 0)
				{
					circleDis += Norm(Subtraction(b[j].axis.p[idx1], b[k].axis.p[idx2]));
					continue;
				}
				float3 v1 = make_norm(cross(cross1, direct1));
				//printf("%f %f %f\n", v1.x, v1.y, v1.z);
				v1 = make_float3(b[j].radius*v1.x, b[j].radius*v1.y, b[j].radius*v1.z);
				v1 = make_float3(b[j].axis.p[idx1].x + v1.x, b[j].axis.p[idx1].y + v1.y, b[j].axis.p[idx1].z + v1.z);

				float3 cross2 = cross(direct2, direct1);
				float3 v2 = make_norm(cross(cross2, direct2));
				//printf("%f %f %f\n", v2.x, v2.y, v2.z);
				v2 = make_float3(b[k].radius*v2.x, b[k].radius*v2.y, b[k].radius*v2.z);
				v2 = make_float3(b[k].axis.p[idx2].x + v2.x, b[k].axis.p[idx2].y + v2.y, b[k].axis.p[idx2].z + v2.z);

				float3 OtoV1 = make_norm(Subtraction(v1, b[j].axis.p[idx1]));
				float3 V1toV2 = make_norm(Subtraction(v2, v1));
				float3 OtoV2 = make_norm(Subtraction(v2, b[k].axis.p[idx2]));
				float3 V2toV1 = make_norm(Subtraction(v1, v2));

				circleAngle += pow(Dot(OtoV1,V1toV2)-1,2)+pow(Dot(OtoV2,V2toV1)-1,2);
				if(Dot(t->diskNorm[j],V1toV2)<0||Dot(t->diskNorm[k],V2toV1)<0)
					circleDis += Norm(Subtraction(v1, v2));
				
				//if (j == k)continue;
				double r1 = b[j].radius, r2 = b[k].radius;

				double cosVal = Dot(diskNorm[j], diskNorm[k]) / (Norm(diskNorm[j])*Norm(diskNorm[k]));
				double angle = acos(cosVal);

				double cutj = Norm(Subtraction(b[j].axis.p[flag[t->triWithBeamid[j]] - 1], node));
				double cutk = Norm(Subtraction(b[k].axis.p[flag[t->triWithBeamid[k]] - 1], node));
				double length1 = sqrt(pow(r1, 2) + pow(cutj, 2));
				double itoj = angle - atan(r1 / cutj);
				double cut1 = length1 * cos(itoj);
				if (cut1 > cutk)
				{
					needCut += cut1 - cutk;
					cutk = cut1;
					/*intersectionLength[j] = cut1;
					stk.push(j);
					stk2.push(j);
					break;*/
								
				}

				double length2 = sqrt(pow(r2, 2) + pow(cutk, 2));
				double jtoi = angle - atan(r2 / cutk);
				double cut2 = length2 * cos(jtoi);
				if (cut2 > cutj)
				{
					needCut += cut2 - cutj;
					cutj = cut2;
					/*intersectionLength[i] = cut2;
					stk.push(i);
					stk2.push(j);
					break;*/
				}
			}
		}
	}

	//printf("%f %f %f\n", disToSphere, circleDis,needCut);
	return 2.3*disToSphere + 1*circleDis;
}
float* shapeComputeX123(float* leaderPos, GwsForShape &gws, int nVar, float a)
{

	//% Eq.(3.4) in the paper
	float* c = (float*)malloc(nVar * sizeof(float));

	/*if (it == 0)
	{
		gws.chaoticC.clear();
		gws.chaoticA.clear();
	}*/
	//srand((unsigned)time(NULL));

	for (int i = 0; i < nVar; i++)
	{
		float rndC;
		//if (it == 0)
		{
			
			curandState devStates;
			/*srand(time(0));
			int seed = rand();*/
			curand_init((unsigned long long)clock(), i, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//curand_init((unsigned long long)(seed*1e9), i, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution

			rndC = abs(RANDOM); //if(rndC==1.0)cout << rndC << endl;
			c[i] = (2 * rndC);
			//printf("%f ", RANDOM);
			//gws.chaoticC.push_back(rndC);
		}
		//else
		//{
		//	rndC = cos(0.5*acos(gws.chaoticC[i]));
		//	//rndC = 0.5*gws.chaoticC[i] * (1 - gws.chaoticC[i]);
		//	c.push_back(2 * rndC);
		//	gws.chaoticC[i] = rndC;
		//}

	}
	float* D = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		D[i] = (abs(c[i] * leaderPos[i] - gws.offset[i]));
	}
	float* A = (float*)malloc(nVar * sizeof(float));

	for (int i = 0; i < nVar; i++)
	{
		float rndA;
		//if (it == 0)
		{
			
			curandState devStates;
			/*srand(time(0));
			int seed = rand();*/
			curand_init((unsigned long long)clock(), i, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//curand_init((unsigned long long)(seed*1e9), i, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution

			rndA = abs(RANDOM);
			A[i] = (2 * a*rndA - a);
			//gws.chaoticA.push_back(rndA);
		}
		//else
		//{
		//	rndA = cos(0.5*acos(gws.chaoticA[i]));
		//	//rndA = 0.5*gws.chaoticA[i] * (1 - gws.chaoticA[i]);
		//	A.push_back(2 * a*rndA - a);
		//	gws.chaoticA[i] = rndA;
		//}

	}
	float* X = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		X[i] = (leaderPos[i] - A[i] * abs(D[i]));
		//printf("%f ", X[i]);
	}
	free(c);
	free(D);
	free(A);

	return X;
}
void shapeOptimization(std::vector<BeamPlugin*> &beams,float3 node,Triangle* triList,int triNum,short* flag,float* longestLength)
{
	int nVar = beams.size();

	int GreyWolves_num = 100;
	float MaxIt = 100;

	float* Alpha_pos = (float*)malloc(nVar * sizeof(float));// new float[nVar];
	memset(Alpha_pos, 0, nVar * sizeof(float));
	float Alpha_score = INF;

	float* Beta_pos = (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	memset(Beta_pos, 0, nVar * sizeof(float));
	float Beta_score = INF;

	float* Delta_pos = (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	memset(Delta_pos, 0, nVar * sizeof(float));
	float Delta_score = INF;

	GwsForShape* gws = (GwsForShape*)malloc(GreyWolves_num * sizeof(GwsForShape));
	//原始球半径计算
	float originR = 0;
	for (int i = 0; i < GreyWolves_num; i++)
	{
		for (int j = 0; j < nVar; j++)
		{
			curandState devStates;
			curand_init((unsigned long long)clock(), j, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//if (longestLength[j] - Norm(Subtraction(beams[j]->axis.p[flag[j] - 1], node)) < 0)printf("%f\n", longestLength[j] - Norm(Subtraction(beams[j]->axis.p[flag[j] - 1], node)));
			RANDOM *= longestLength[j]-Norm(Subtraction(beams[j]->axis.p[flag[j]-1],node));
			gws[i].offset[j] = RANDOM;
			if (beams[j]->radius > originR)
				originR = beams[j]->radius;
		}
	}
	float* record = (float*)malloc(MaxIt * sizeof(float));
	int total_iter = 0;
	//main loop
	for (float it = 0; it < MaxIt; it++)
	{
		float a = 2 - it * ((2) / MaxIt);
		for (int i = 0; i < GreyWolves_num; i++)
		{
			//计算cost
			gws[i].singleCost = measureShapeCost(originR, beams, node, triList, triNum, gws[i], nVar, flag);
			float fitness = gws[i].singleCost;
			//Update Alpha, Beta, and Delta
			if (fitness < Alpha_score)
			{
				Alpha_score = fitness;
				memcpy(Alpha_pos, gws[i].offset, nVar * sizeof(float));
			}
			if (fitness > Alpha_score && fitness < Beta_score)
			{
				Beta_score = fitness;
				memcpy(Beta_pos, gws[i].offset, nVar * sizeof(float));
			}
			if (fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score)
			{
				Delta_score = fitness;
				memcpy(Delta_pos, gws[i].offset, nVar * sizeof(float));
			}
		}
		for (int i = 0; i < GreyWolves_num; i++)
		{
			int It = 0;
			
			float* X1 = shapeComputeX123(Alpha_pos, gws[i], nVar, a);
			float* X2 = shapeComputeX123(Beta_pos, gws[i], nVar, a);
			float* X3 = shapeComputeX123(Delta_pos, gws[i], nVar, a);

			for (int k = 0; k < nVar; k++)
			{
				gws[i].offset[k] = (X1[k] + X2[k] + X3[k]) / 3;
				gws[i].offset[k] = std::min(std::max((double)(gws[i].offset[k]), 0.0), (double)(longestLength[k] - Norm(Subtraction(beams[k]->axis.p[flag[k] - 1], node))));
				//printf("%f ", gws[i].rotationAngle[k]);
			}
			//printf("\n");
			free(X1); free(X2); free(X3);
		}

		printf("%f\n", Alpha_score);
		record[total_iter] = Alpha_score;
		total_iter++;
		while (it == MaxIt - 1)
		{
			if (abs(record[(int)MaxIt - 1] - record[(int)MaxIt - 11]) < 1)
			{
				for (int j = 0; j < nVar; j++)
				{
					BeamPlugin* b = beams[j];
					float offset = Alpha_pos[j]; printf("%f ", offset);
					float3 direction;
					if (flag[j] == 1)
					{
						direction = make_norm(Subtraction(b->axis.p[1], b->axis.p[0]));
						b->axis.p[0].x += direction.x*offset;
						b->axis.p[0].y += direction.y*offset;
						b->axis.p[0].z += direction.z*offset;
					}
					else
					{
						direction = make_norm(Subtraction(b->axis.p[0], b->axis.p[1]));
						b->axis.p[1].x += direction.x*offset;
						b->axis.p[1].y += direction.y*offset;
						b->axis.p[1].z += direction.z*offset;
					}
					b->length -= offset;
				}
				printf("\n");
				break;
			}
			else
			{

				/*triList.clear();
				quadList.clear();*/
				it = -1;
				MaxIt = 100;
				//if(record.size()==MaxIt)
				free(record);
				record = (float*)malloc(MaxIt * sizeof(float));
				total_iter = 0;
				//record.resize(MaxIt);
				break;

			}
		}
	}
	free(gws);
	free(record);
	free(Alpha_pos);
	//free(Alpha.rotationAngle);
	free(Beta_pos);
	//free(Beta.rotationAngle);
	free(Delta_pos);
}
//CPU端，若是非凸，则难以收敛
void GreyWolfOptimizer(int nVar, Triangle* &triangles, QuadFace* &quads,int triNum,int quadNum,int batchIdx)
{
	//int triNum = triangles.size();
	int GreyWolves_num = 100;
	float MaxIt = 100;

	bool once_inverted = true;
	float* uninverted_pos = (float*)malloc(nVar * sizeof(float));
	memset(uninverted_pos, 0, nVar * sizeof(float));

	bool Alpha_inverted = false;
	float* Alpha_pos = (float*)malloc(nVar * sizeof(float));// new float[nVar];
	memset(Alpha_pos, 0, nVar * sizeof(float));
	/*for (int i = 0; i < nVar; i++)cout << Alpha_pos[i] << " ";
	cout << endl;*/
	float Alpha_score = INF;
	//GreyWolves Alpha; Alpha.rotationAngle = (float*)malloc(nVar * sizeof(float));// new float[nVar];
	//for (int i = 0; i < nVar; i++)
	//	Alpha.rotationAngle[i] = 0;

	float* Beta_pos = (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	memset(Beta_pos, 0, nVar * sizeof(float));
	/*for (int i = 0; i < nVar; i++)cout << Beta_pos[i] << " ";
	cout << endl;*/
	float Beta_score = INF;
	//GreyWolves Beta; Beta.rotationAngle = (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	//for (int i = 0; i < nVar; i++)
	//	Beta.rotationAngle[i] = 0;

	float* Delta_pos = (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	memset(Delta_pos, 0, nVar * sizeof(float));
	/*for (int i = 0; i < nVar; i++)cout << Delta_pos[i] << " ";
	cout << endl;*/
	float Delta_score = INF;
	//GreyWolves Delta; Delta.rotationAngle = (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	//for (int i = 0; i < nVar; i++)
	//	Delta.rotationAngle[i] = 0;

	//std::vector<GreyWolves> gws;
	//gws.resize(GreyWolves_num);
	GreyWolves* gws =(GreyWolves*)malloc(GreyWolves_num*sizeof(GreyWolves));
	//Initialization
	//std::vector<shared_ptr<Triangle>> triList;
	Triangle* triList;
	//std::vector<shared_ptr<QuadFace>> quadList;
	QuadFace* quadList;

	
	
	for (int i = 0; i < GreyWolves_num; i++)
	{
		gws[i].rotationAngle = (float*)malloc(nVar*sizeof(float));
		for (int j = 0; j < nVar; j++)
		{
			////初始化position即随机旋转角
			//std::default_random_engine e;
			//std::uniform_real_distribution<float> u(-PI, PI);
			//e.seed(time(0));
			curandState devStates;
			//srand(time(0));
			//int seed = rand();
			curand_init((unsigned long long)clock(), j, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//Eigen::MatrixXd R = Eigen::MatrixXd::Random(1, 1);
			//curand_init((unsigned long long)(seed*1e9), 0, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution
			RANDOM = (RANDOM - 0.5) * 2 * PI;
			//float RANDOM = (abs(R(0,0)) - 0.5) * 2 * PI;
			//cout << i * nVar + j << " " << host_rA[i*nVar + j] << endl;
			gws[i].rotationAngle[j] = RANDOM;
		}
		//gws[i].rA = gws[i].rotationAngle.data();
	}
	
	float* record = (float*)malloc(MaxIt*sizeof(float));
	float* tmpRecord = (float*)malloc(20*sizeof(float));
	int tmp_iter = 0;
	/*vector<float> Y;
	vector<bool> invertFlag;*/

	int total_iter = 0;
	//int total_cnt = 0;
	//record.resize(MaxIt*3);
	//main loop
	for (float it = 0; it < MaxIt; it++)
	{
		float a = 2 - it * ((2) / MaxIt);

#ifdef TEST
		int nodeNum = 100;
		GreyWolves* hostec = (GreyWolves*)malloc(nodeNum * sizeof(GreyWolves));
		for (int i = 0; i < nodeNum; i++)hostec[i].rotationAngle = (float*)malloc(nVar * sizeof(float));
		GreyWolves* cudaec;
		CUDA_SAFE_CALL(cudaMallocManaged((void **)&cudaec, nodeNum * sizeof(GreyWolves)));
		//CUDA_SAFE_CALL(cudaMemcpy((void*)cudaec, (void*)hostec, nodeNum * sizeof(GreyWolves), cudaMemcpyHostToDevice));
		float* d_rotation[100];
		float* h_tmp[100];
		int* d_test[100];
		int* h_tmp2[100];
		for (int i = 0; i < nodeNum; i++)
		{
			h_tmp[i] = hostec[i].rotationAngle;
			h_tmp2[i] = hostec[i].test;

			CUDA_SAFE_CALL(cudaMalloc((void **)&(d_rotation[i]), nVar * sizeof(float)));
			CUDA_SAFE_CALL(cudaMemcpy((void*)(d_rotation[i]), (void*)(hostec[i].rotationAngle), nVar * sizeof(float), cudaMemcpyHostToDevice));
			hostec[i].rotationAngle = d_rotation[i];

			/*CUDA_SAFE_CALL(cudaMalloc((void **)&(d_test[i]), 2 * sizeof(int)));
			CUDA_SAFE_CALL(cudaMemcpy((void*)(d_test[i]), (void*)(hostec[i].test), 2 * sizeof(int), cudaMemcpyHostToDevice));
			hostec[i].test = d_test[i];*/
			/*CUDA_SAFE_CALL(cudaMalloc((void **)&(cudaec[i].rotationAngle), nVar * sizeof(float)));
			CUDA_SAFE_CALL(cudaMemcpy((void*)(cudaec[i].rotationAngle), (void*)(hostec[i].rotationAngle), nVar * sizeof(float), cudaMemcpyHostToDevice));*/
		}
		CUDA_SAFE_CALL(cudaMemcpy((void*)cudaec, (void*)hostec, nodeNum * sizeof(GreyWolves), cudaMemcpyHostToDevice));
		call_testEigenCUDA(nodeNum,cudaec);
		CUDA_SAFE_CALL(cudaMemcpy((void*)hostec, (void*)cudaec, nodeNum * sizeof(GreyWolves), cudaMemcpyDeviceToHost));
		for (int i = 0; i < nodeNum; i++)
		{
			hostec[i].rotationAngle = h_tmp[i];
			//float* d_rotation;
			//CUDA_SAFE_CALL(cudaMalloc((void **)&(d_rotation), nVar * sizeof(float)));
			CUDA_SAFE_CALL(cudaMemcpy((void*)(hostec[i].rotationAngle), (void*)(d_rotation[i]), nVar * sizeof(float), cudaMemcpyDeviceToHost));
		}
		cout << hostec[10].test[1] <<" "<<hostec[12].rotationAngle[8]<< endl;
#endif

#ifdef PARALLEL
		QuadFace* hostquadList = (QuadFace*)malloc(quadNum * sizeof(QuadFace));
		/*for (int i = 0; i < quadNum; i++)
		{
			hostquadList[i].quadId = 0;
			hostquadList[i].quadWithBeamid[0] = 0;
			hostquadList[i].quadWithBeamid[1] = 0;
			for (int j = 0; j < 3; j++)
			{
				hostquadList[i].t[0].p[j] = Eigen::Vector3d(0,0,0);
				hostquadList[i].t[1].p[j] = Eigen::Vector3d(0, 0, 0);
			}
			hostquadList[i].t[0].normal = Eigen::Vector3d(0, 0, 0);
			hostquadList[i].t[1].normal = Eigen::Vector3d(0, 0, 0);
			hostquadList[i].width[0] = 0;
			hostquadList[i].width[1] = 0;
		}*/
		CUDA_SAFE_CALL(cudaMalloc((void **)&quadList, quadNum*sizeof(QuadFace)));
		CUDA_SAFE_CALL(cudaMemcpy((void*)quadList, (void*)hostquadList, quadNum * sizeof(QuadFace), cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpy((void*)hostquadList, (void*)quadList, quadNum * sizeof(QuadFace), cudaMemcpyDeviceToHost));
		
		CUDA_SAFE_CALL(cudaMalloc((void **)&triList,triNum* sizeof(Triangle)));
		Triangle* d_adjT[50];
		Triangle* h_tmp[50];
		QuadFace* d_adjQ[50];
		QuadFace* h_tmp2[50];
		for (int i = 0; i < triNum; i++)
		{
			h_tmp[i] = triangles[i].adjTriangles;
			h_tmp2[i] = triangles[i].adjQuadFaces;

			CUDA_SAFE_CALL(cudaMalloc((void **)&(d_adjT[i]), 3 * sizeof(Triangle)));
			CUDA_SAFE_CALL(cudaMemcpy((void*)(d_adjT[i]), (void*)(triangles[i].adjTriangles), 3 * sizeof(Triangle), cudaMemcpyHostToDevice));
			triangles[i].adjTriangles = d_adjT[i];

			CUDA_SAFE_CALL(cudaMalloc((void **)&(d_adjQ[i]), 3 * sizeof(QuadFace)));
			CUDA_SAFE_CALL(cudaMemcpy((void*)(d_adjQ[i]), (void*)(triangles[i].adjQuadFaces), 3 * sizeof(QuadFace), cudaMemcpyHostToDevice));
			triangles[i].adjQuadFaces = d_adjQ[i];
		}
		CUDA_SAFE_CALL(cudaMemcpy((void*)triList, (void*)triangles, triNum * sizeof(Triangle), cudaMemcpyHostToDevice));
		for (int i = 0; i < triNum; i++)
		{
			triangles[i].adjTriangles = h_tmp[i];
			triangles[i].adjQuadFaces = h_tmp2[i];
		}

		GreyWolves* cudagws; 
		CUDA_SAFE_CALL(cudaMalloc((void **)&cudagws, GreyWolves_num * sizeof(GreyWolves)));
		float* d_rotation[256];
		float* h_tmp3[256];
		for (int i = 0; i < GreyWolves_num; i++)
		{
			h_tmp3[i] = gws[i].rotationAngle;

			CUDA_SAFE_CALL(cudaMalloc((void **)&(d_rotation[i]), nVar * sizeof(float)));
			CUDA_SAFE_CALL(cudaMemcpy((void*)(d_rotation[i]), (void*)(gws[i].rotationAngle), nVar * sizeof(float), cudaMemcpyHostToDevice));
			gws[i].rotationAngle = d_rotation[i];
		}
		CUDA_SAFE_CALL(cudaMemcpy((void*)cudagws, (void*)gws, GreyWolves_num * sizeof(GreyWolves), cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpy((void*)hostgws, (void*)cudagws, GreyWolves_num * sizeof(GreyWolves), cudaMemcpyDeviceToHost));
		call_SearchAgentsComputing(nVar, cudagws, GreyWolves_num, triList, triNum, quadList, quadNum);
		/*cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
			printf("%s.\n",cudaGetErrorString(cudaerr));*/
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		//free(gws);
		//GreyWolves* hostgws = (GreyWolves*)malloc(GreyWolves_num * sizeof(GreyWolves));
		CUDA_SAFE_CALL(cudaMemcpy((void*)gws, (void*)cudagws, GreyWolves_num * sizeof(GreyWolves), cudaMemcpyDeviceToHost));
		for (int i = 0; i < GreyWolves_num; i++)
		{
			gws[i].rotationAngle = h_tmp3[i];
			CUDA_SAFE_CALL(cudaMemcpy((void*)(gws[i].rotationAngle), (void*)(d_rotation[i]), nVar * sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaFree(d_rotation[i]));

			float fitness = gws[i].singleCost;
			//Update Alpha, Beta, and Delta
			if (fitness < Alpha_score)
			{
				Alpha_score = fitness;
				copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Alpha_pos); //Alpha_pos = gws[i].rotationAngle;
				Alpha = gws[i];
				Alpha_inverted = gws[i].inverted;
			}
			if (fitness > Alpha_score && fitness < Beta_score)
			{
				Beta_score = fitness;
				copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Beta_pos); //Beta_pos = gws[i].rotationAngle;
				Beta = gws[i];
			}
			if (fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score)
			{
				Delta_score = fitness;
				copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Delta_pos); //Delta_pos = gws[i].rotationAngle;
				Delta = gws[i];
			}
		}
		for (int i = 0; i < triNum; i++)
		{
			CUDA_SAFE_CALL(cudaFree(d_adjT[i]));
			CUDA_SAFE_CALL(cudaFree(d_adjQ[i]));
		}
		CUDA_SAFE_CALL(cudaFree(quadList));
		CUDA_SAFE_CALL(cudaFree(triList));
		CUDA_SAFE_CALL(cudaFree(cudagws));
		free(hostquadList);
#else
		for (int i = 0; i < GreyWolves_num; i++)
		{
			//根据旋转角度计算三角形顶点坐标
		    //拷贝一份triangles，暂时不能真正旋转
			/*int triListSize = triList.size();
			vector<Triangle> triTmpList;
			triTmpList.resize(triListSize);
			for (int v = 0; v < triListSize; v++)
			{
				triList[v]->adjTriangles.clear();
				triList[v]->adjQuadFaces.clear();
				triTmpList[v] = *triList[v];
			}*/

			/*triList.clear();
			quadList.clear();

			triList.resize(triNum);*/
			quadList = (QuadFace*)malloc(quadNum * sizeof(QuadFace));
			triList = (Triangle*)malloc(triNum * sizeof(Triangle));
				/*triList[j] = make_shared<Triangle>();*/
			for (int j = 0; j < triNum; j++)
			{
				triList[j] = triangles[j];
			}
			//计算在当前三角形位置上旋转随机角度后的位置
			for (int j = 0; j < triNum; j++)
			{
				//if (triListSize > 0)
				//{
				//	std::default_random_engine e;
				//	std::uniform_real_distribution<float> u(-1, 1);
				//	e.seed(time(0));
				//	float r = u(e);//返回从区间 (-1,1) 的均匀分布中得到的随机标量
				//	Triangle prevT = triTmpList[j];
				//	if (prevT.cost < 1.4&&r>0)
				//	{
				//		*triList[j] = prevT;
				//		for (int k = 0; k < 3; k++)
				//		{
				//			gws[i].rotationAngle[j * 3 + k] = gws[mod(i-1,GreyWolves_num)].rotationAngle[j * 3 + k];
				//		}
				//		continue;
				//	}
				//}
				Triangle t = triList[j];
				for (int k = 0; k < 3; k++)
				{
					float3 diskNorm = t.diskNorm[k];
					float theta = gws[i].rotationAngle[j * 3 + k];
					float3 O = t.o[k];
					float** m = getRotationMatrix(diskNorm, theta, O);

					float3 rotateV = make_float3(t.p[k].x - O.x, t.p[k].y - O.y, t.p[k].z - O.z);
					float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
					RV = Multiply(m, RV);
					t.p[k] = make_float3(RV.x, RV.y, RV.z);

					for (int f = 0; f < 4; f++)
					{
						free(m[f]);// = (float*)malloc(4 * sizeof(float));
					}
					free(m);
				}
				float3 e1 = make_float3(t.p[1].x - t.p[0].x, t.p[1].y - t.p[0].y, t.p[1].z - t.p[0].z);
				float3 e2 = make_float3(t.p[2].x - t.p[0].x, t.p[2].y - t.p[0].y, t.p[2].z - t.p[0].z);
				t.normal = cross(e1, e2);
				t.normal = make_norm(t.normal);
				triList[j] = t;
			}
			//计算cost
			bool inverted = false;
			findAdjTriangles(triList,triNum);//否则找到的邻居三角形还是以前的
			findQuadFaces(triList, quadList,triNum, inverted);
			gws[i].singleCost = measureSingleCost(triList,triNum,quadList,quadNum,gws[i],nVar,inverted);
			gws[i].inverted = inverted;

			float fitness = gws[i].singleCost;
			//Update Alpha, Beta, and Delta
			if (fitness < Alpha_score)
			{
				Alpha_score = fitness;
				//copy(gws[i].rotationAngle, gws[i].rotationAngle+nVar, Alpha_pos); //Alpha_pos = gws[i].rotationAngle;
				memcpy(Alpha_pos, gws[i].rotationAngle, nVar * sizeof(float));
				/*for (int k = 0; k < nVar; k++)cout << Alpha_pos[k] << " "<< gws[i].rotationAngle[k];
				cout << endl;*/
				//Alpha = gws[i];
				Alpha_inverted = gws[i].inverted;
				if (Alpha_inverted == false)
				{
					memcpy(uninverted_pos, Alpha_pos, nVar * sizeof(float));
					once_inverted = false;
				}
			}
			if (fitness > Alpha_score && fitness < Beta_score)
			{
				Beta_score = fitness;
				//copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Beta_pos); //Beta_pos = gws[i].rotationAngle;
				memcpy(Beta_pos, gws[i].rotationAngle, nVar * sizeof(float));
				/*for (int i = 0; i < nVar; i++)cout << Beta_pos[i] << " ";
				cout << endl;*/
				//Beta = gws[i];
			}
			if (fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score)
			{
				Delta_score = fitness;
				//copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Delta_pos); //Delta_pos = gws[i].rotationAngle;
				memcpy(Delta_pos, gws[i].rotationAngle, nVar * sizeof(float));
				/*for (int i = 0; i < nVar; i++)cout << Delta_pos[i] << " ";
				cout << endl;*/
				//Delta = gws[i];
			}
			
			//for (int v = 0; v < triNum; v++)
			//{
			//	
			//	free(triList[v].adjTriangles);
			//	free(triList[v].adjQuadFaces);
			//	//free(triList[v]);
			//}
			free(triList);
			free(quadList);
			//for (int v = 0; v < quadNum; v++)
			//{
			//	free(quadList[v]);
			//}
			    
		}
#endif
		for (int i = 0; i < GreyWolves_num; i++)
		{
			int It=0;
			if (total_iter > 10)
			{
				It = 0;
				/*if (record.back() == *(record.end() - 20))
				{
					It = 0;
				}
				else
				{
					It = it;
				}*/
			}
			float* X1 = computeX123(Alpha_pos, gws[i], nVar, a,It);
			float* X2 = computeX123(Beta_pos, gws[i], nVar, a,It);
			float* X3 = computeX123(Delta_pos, gws[i], nVar, a,It);

			for (int k = 0; k < nVar; k++)
			{
				gws[i].rotationAngle[k] = (X1[k] + X2[k] + X3[k]) / 3;
				gws[i].rotationAngle[k] = std::min(std::max((double)(gws[i].rotationAngle[k]), -PI), PI);
				//printf("%f ", gws[i].rotationAngle[k]);
			}
			//printf("\n");
			free(X1); free(X2); free(X3);
		}

		printf("%d %f batch: %d\n", Alpha_inverted, Alpha_score,batchIdx);
		record[total_iter] = Alpha_score;
		tmpRecord[tmp_iter] = Alpha_score;
		/*Y.push_back(Alpha_score);
		invertFlag.push_back(Alpha_inverted);*/
		tmp_iter++;
		total_iter++;
		//total_cnt++;

		//重新随机化，加速收敛
		if (tmp_iter > 10&& tmp_iter<=20)
		{
			if (tmpRecord[tmp_iter-1] == *(tmpRecord + tmp_iter - 10))
			{
				for (int i = 0; i < GreyWolves_num; i++)
				{
					//gws[i].rotationAngle.clear();
					for (int j = 0; j < nVar; j++)
					{
						////初始化position即随机旋转角
						//std::default_random_engine e;
						//std::uniform_real_distribution<float> u(-PI, PI);
						//e.seed(time(0));
						curandState devStates;
						//srand(time(0));
						//int seed = rand();
						curand_init((unsigned long long)clock(), j, 0, &devStates);// initialize the state
						float RANDOM = curand_uniform(&devStates);// uniform distribution
						//Eigen::MatrixXd R = Eigen::MatrixXd::Random(1, 1);
						//curand_init((unsigned long long)(seed*1e9), 0, 0, &devStates2);// initialize the state
						//float RANDOM = curand_uniform(&devStates2);// uniform distribution
						RANDOM = (RANDOM - 0.5) * 2 * PI;
						//float RANDOM = (abs(R(0,0)) - 0.5) * 2 * PI;

						gws[i].rotationAngle[j] = RANDOM;
					}

				}
				//for (int j = 0; j < nVar; j++)
				//{
				//	
				//	Eigen::MatrixXd R = Eigen::MatrixXd::Random(1, 1);
				//	float RANDOM = (abs(R(0, 0)) - 0.5) * 2 * PI;
				//	/*float RANDOM2 = (abs(R(1, 0)) - 0.5) * 2 * PI;
				//	float RANDOM3 = (abs(R(2, 0)) - 0.5) * 2 * PI;*/
				//	Alpha_pos[j] = RANDOM;
				//	/*Beta_pos[j] = RANDOM2;
				//	Delta_pos[j] = RANDOM3;*/
				//}
				free(tmpRecord);
				tmpRecord = (float*)malloc(20 * sizeof(float));
				tmp_iter = 0;
			}
		}
		if (tmp_iter >= 20)
		{
			free(tmpRecord);
			tmpRecord = (float*)malloc(20 * sizeof(float));
			tmp_iter = 0;
		}

		while (it == MaxIt - 1)
		{
			//判断是否翻转
			if ((Alpha_inverted==false&&abs(record[(int)MaxIt-1]-record[(int)MaxIt-11])<1))
			{
				for (int j = 0; j < triNum; j++)
				{
					Triangle t = triangles[j];
					for (int k = 0; k < 3; k++)
					{
						float3 diskNorm = t.diskNorm[k];
						float theta = Alpha_pos[j * 3 + k];
						float3 O = t.o[k];
						float** m = getRotationMatrix(diskNorm, theta, O);

						float3 rotateV = make_float3(t.p[k].x - O.x, t.p[k].y - O.y, t.p[k].z - O.z);
						float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
						RV = Multiply(m, RV);
						t.p[k] = make_float3(RV.x, RV.y, RV.z);

						for (int f = 0; f < 4; f++)
						{
							free(m[f]);// = (float*)malloc(4 * sizeof(float));
						}
						free(m);
					}
					float3 e1 = make_float3(t.p[1].x - t.p[0].x, t.p[1].y - t.p[0].y, t.p[1].z - t.p[0].z);
					float3 e2 = make_float3(t.p[2].x - t.p[0].x, t.p[2].y - t.p[0].y, t.p[2].z - t.p[0].z);
					t.normal = cross(e1, e2);
					t.normal = make_norm(t.normal);
					triangles[j] = t;
				}
				//计算cost
				bool inverted = false;
				findAdjTriangles(triangles,triNum);//否则找到的邻居三角形还是以前的
				findQuadFaces(triangles, quads,triNum, inverted);

				/*triangles = triList;
				quads = quadList;*/
				break;
			}
			else if (once_inverted == false && abs(record[(int)MaxIt - 1] - record[(int)MaxIt - 11]) < 1)
			{
				for (int j = 0; j < triNum; j++)
				{
					Triangle t = triangles[j];
					for (int k = 0; k < 3; k++)
					{
						float3 diskNorm = t.diskNorm[k];
						float theta = uninverted_pos[j * 3 + k];
						float3 O = t.o[k];
						float** m = getRotationMatrix(diskNorm, theta, O);

						float3 rotateV = make_float3(t.p[k].x - O.x, t.p[k].y - O.y, t.p[k].z - O.z);
						float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
						RV = Multiply(m, RV);
						t.p[k] = make_float3(RV.x, RV.y, RV.z);

						for (int f = 0; f < 4; f++)
						{
							free(m[f]);// = (float*)malloc(4 * sizeof(float));
						}
						free(m);
					}
					float3 e1 = make_float3(t.p[1].x - t.p[0].x, t.p[1].y - t.p[0].y, t.p[1].z - t.p[0].z);
					float3 e2 = make_float3(t.p[2].x - t.p[0].x, t.p[2].y - t.p[0].y, t.p[2].z - t.p[0].z);
					t.normal = cross(e1, e2);
					t.normal = make_norm(t.normal);
					triangles[j] = t;
				}
				//计算cost
				bool inverted = false;
				findAdjTriangles(triangles, triNum);//否则找到的邻居三角形还是以前的
				findQuadFaces(triangles, quads, triNum, inverted);

				/*triangles = triList;
				quads = quadList;*/
				break;
			}
			else
			{

				/*triList.clear();
				quadList.clear();*/
				it = -1;
				MaxIt = 100;
				//if(record.size()==MaxIt)
				free(record);
				record = (float*)malloc(MaxIt * sizeof(float));
				total_iter = 0;
				//record.resize(MaxIt);
				break;

			}

		}
	}

	for (int i = 0; i < GreyWolves_num; i++)
	{
		free(gws[i].rotationAngle);
	}
	free(gws);
	free(record);
	free(tmpRecord);
	free(uninverted_pos);
	free(Alpha_pos);
	//free(Alpha.rotationAngle);
	free(Beta_pos);
	//free(Beta.rotationAngle);
	free(Delta_pos);
	//free(Delta.rotationAngle);
	//CoordinateSystem cs;
	//coordinateSystemInit(&cs);
	//int YSize = Y.size();
	//float *a = new float[YSize];
	//copy(Y.begin(),Y.end(), a);

	//bool* inF = new bool[YSize];
	//copy(invertFlag.begin(),invertFlag.end(),inF);
	///*int j = 0;
	//for (float i = 0; i < 2 * 3.14; i += 3.14 / 500) {
	//	a[j++] = 10 * sin(i);
	//}*/
	//plotFloat(&cs, a,inF, YSize, RED);
	//_getch();				// 按任意键继续
	//closegraph();			// 关闭图形界面

}
//CPU端，若是非凸，则难以收敛，比老版更差
void GWO2(BeamPlugin** beams, int beamNum, short* flag, int nVar, Triangle* triangles,
	QuadFace* quads, int triNum, int quadNum,int batchIdx, int nodeId)
{
	float MaxIt = 100;

	 bool Alpha_inverted;
	Alpha_inverted = false;
	float* Alpha_pos = (float*)malloc((nVar+beamNum) * sizeof(float));// new float[nVar];
	memset(Alpha_pos, 0, (nVar+beamNum) * sizeof(float));
	 float Alpha_score;
	Alpha_score = INF;

	float* Beta_pos = (float*)malloc((nVar + beamNum) * sizeof(float)); //new float[nVar];
	memset(Beta_pos, 0, (nVar + beamNum) * sizeof(float));
	 float Beta_score;
	Beta_score = INF;

	float* Delta_pos = (float*)malloc((nVar + beamNum) * sizeof(float)); //new float[nVar];
	memset(Delta_pos, 0, (nVar + beamNum) * sizeof(float));
	 float Delta_score;
	Delta_score = INF;


	 GreyWolves gws[128];
	 float fitness[128];

	Triangle triList[100];
	QuadFace quadList[150];
	for (int i = 0; i < 128; i++)
	{
		gws[i].rotationAngle = (float*)malloc((nVar + beamNum) * sizeof(float));
		/*gws[i].chaoticA = (float*)malloc((nVar + beamNum) * sizeof(float));
		gws[i].chaoticC = (float*)malloc((nVar + beamNum) * sizeof(float));*/
	}
	
	//for (int j = 0; j < nVar; j++)
	//{
	//	////初始化position即随机旋转角
	//	//std::default_random_engine e;
	//	//std::uniform_real_distribution<float> u(-PI, PI);
	//	//e.seed(time(0));
	//	curandState devStates;
	//	/*srand(time(0));
	//	int seed = rand();*/
	//	curand_init((unsigned long long)clock(), j, 0, &devStates);// initialize the state
	//	float RANDOM = curand_uniform(&devStates);// uniform distribution
	//	//curand_init((unsigned long long)(seed*1e6), 0, 0, &devStates2);// initialize the state
	//	//float RANDOM = curand_uniform(&devStates2);// uniform distribution
	//	RANDOM = RANDOM * 2 * PI;
	//	//cout << i * nVar + j << " " << host_rA[i*nVar + j] << endl;
	//	gws[lane_id].rotationAngle[j] = RANDOM;
	//}
	for (int i = 0; i < 128; i++)
	{
		int cnt = 0;
		for (int j = 0; j < beamNum; j++)
		{
			BeamPlugin* b = beams[j];
			int idx = flag[j] - 1;
			int rndThetaNum = b->arcNum[idx];
			float thetaSum = 0;
			for (int k = 0; k <= rndThetaNum; k++)
			{
				curandState devStates;
				curand_init((unsigned long long)clock(), cnt + k, 0, &devStates);// initialize the state
				float RANDOM = curand_uniform(&devStates);// uniform distribution
				//RANDOM = RANDOM * 2 * PI;
				if (k != 0)
				{
					gws[i].rotationAngle[cnt + k] = (RANDOM + 0.5) * (-2 * PI / rndThetaNum);
					//gws[lane_id].rotationAngle[cnt + k] = Min(Max((double)(gws[lane_id].rotationAngle[cnt + k]), -2*PI/rndThetaNum*2), 0);
				}
				if (k != 0)
					thetaSum += gws[i].rotationAngle[cnt + k];
				if (k == 0)
					gws[i].rotationAngle[cnt + k] = (RANDOM - 0.5) * PI * 2;
				//gws[lane_id].rotationAngle[cnt + k] = -RANDOM * 2 * PI;
			}
			for (int k = 1; k <= rndThetaNum; k++)
			{
				gws[i].rotationAngle[cnt + k] /= thetaSum;
				gws[i].rotationAngle[cnt + k] *= -2 * PI;
			}
			cnt += rndThetaNum + 1;
		}
	}
	

	float record[100];
	float tmpRecord[20];
	int tmp_iter = 0;

	int total_iter = -1;
	int force_iter = 0;
	//record.resize(MaxIt*3);
	//main loop
	for (float it = 0; it < MaxIt; it++)
	{
		float a = 2 - it * ((2) / MaxIt);
		for (int i = 0; i < 128; i++)
		{
			for (int j = 0; j < triNum; j++)
			{
				triList[j] = triangles[j];
			}
			int cnt = 0;
			for (int j = 0; j < beamNum; j++)
			{
				BeamPlugin* b = beams[j];
				int idx = flag[j] - 1;
				Triangle* tri = triList + b->arcTriList[idx][0];
				float3 prevP;
				for (int v = 0; v < 3; v++)
				{
					if (tri->triWithBeamid[v] == j)
					{
						float3 diskNorm = tri->diskNorm[v];
						float theta = gws[i].rotationAngle[cnt];
						float3 O = tri->o[v];
						float **m=getRotationMatrix(diskNorm, theta, O);

						float3 rotateV = make_float3(tri->p[v].x - O.x, tri->p[v].y - O.y, tri->p[v].z - O.z);
						float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
						RV = Multiply(m, RV);
						tri->p[v] = make_float3(RV.x, RV.y, RV.z);
						prevP = tri->p[v];
						for (int f = 0; f < 4; f++)
						{
							free(m[f]);// = (float*)malloc(4 * sizeof(float));
						}
						free(m);
						break;
					}

				}
				for (int k = 1; k < b->arcNum[idx]; k++)
				{
					//Triangle* tri = triList + b->arcTriList[idx][k];
					Triangle* tri2 = triList + b->arcTriList[idx][k];
					for (int v = 0; v < 3; v++)
					{
						if (tri2->triWithBeamid[v] == j)
						{
							float3 diskNorm = tri2->diskNorm[v];
							float theta = gws[i].rotationAngle[cnt + k];
							float3 O = tri2->o[v];
							float** m = getRotationMatrix(diskNorm, theta, O);

							float3 rotateV = make_float3(prevP.x - O.x, prevP.y - O.y, prevP.z - O.z);
							float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
							RV = Multiply(m, RV);
							tri2->p[v] = make_float3(RV.x, RV.y, RV.z);
							prevP = tri2->p[v];
							for (int f = 0; f < 4; f++)
							{
								free(m[f]);// = (float*)malloc(4 * sizeof(float));
							}
							free(m);
							break;
						}
					}

					//tri = tri2;

				}
				cnt += b->arcNum[idx] + 1;
			}
			//所有三角形的normal更新
			for (int j = 0; j < triNum; j++)
			{
				Triangle* t = &triList[j];
				float3 e1 = make_float3(t->p[1].x - t->p[0].x, t->p[1].y - t->p[0].y, t->p[1].z - t->p[0].z);
				float3 e2 = make_float3(t->p[2].x - t->p[0].x, t->p[2].y - t->p[0].y, t->p[2].z - t->p[0].z);
				t->normal = cross(e1, e2);
				t->normal = make_norm(t->normal);
			}
			//计算cost
			bool inverted = false;
			findAdjTriangles(triList, triNum);//否则找到的邻居三角形还是以前的
			findQuadFaces(triList, quadList, triNum, inverted);
			gws[i].singleCost = measureSingleCost(triList, triNum, quadList, quadNum, gws[i], nVar, inverted);
			gws[i].inverted = inverted;
			float fitness = gws[i].singleCost;
			//Update Alpha, Beta, and Delta
			if (fitness < Alpha_score)
			{
				Alpha_score = fitness;
				//copy(gws[i].rotationAngle, gws[i].rotationAngle+nVar, Alpha_pos); //Alpha_pos = gws[i].rotationAngle;
				memcpy(Alpha_pos, gws[i].rotationAngle, (nVar + beamNum) * sizeof(float));
				/*for (int k = 0; k < nVar; k++)cout << Alpha_pos[k] << " "<< gws[i].rotationAngle[k];
				cout << endl;*/
				//Alpha = gws[i];
				Alpha_inverted = gws[i].inverted;
				
			}
			if (fitness > Alpha_score && fitness < Beta_score)
			{
				Beta_score = fitness;
				//copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Beta_pos); //Beta_pos = gws[i].rotationAngle;
				memcpy(Beta_pos, gws[i].rotationAngle, (nVar + beamNum) * sizeof(float));
				/*for (int i = 0; i < nVar; i++)cout << Beta_pos[i] << " ";
				cout << endl;*/
				//Beta = gws[i];
			}
			if (fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score)
			{
				Delta_score = fitness;
				//copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Delta_pos); //Delta_pos = gws[i].rotationAngle;
				memcpy(Delta_pos, gws[i].rotationAngle, (nVar + beamNum) * sizeof(float));
				/*for (int i = 0; i < nVar; i++)cout << Delta_pos[i] << " ";
				cout << endl;*/
				//Delta = gws[i];
			}
			

			
		}
		
		total_iter++;
		for (int i = 0; i < 128; i++)
		{
			int It = 0;
			if (total_iter > 10)
			{
				//It = 0;
				//if (record.back() == *(record.end() - 20))
				if (abs(record[total_iter - 1] - record[total_iter - 10]) < 0.5)
				{
					It = it; //printf("chaotic!\n");
				}
				else
				{
					It = 0;
				}
			}
			float* X1= computeX123(Alpha_pos, gws[i], nVar + beamNum, a, 0);
			float* X2= computeX123(Beta_pos, gws[i], nVar + beamNum, a, 0);
			float* X3= computeX123(Delta_pos, gws[i], nVar + beamNum, a, 0);
			
			
			

			int cnt = 0;
			for (int j = 0; j < beamNum; j++)
			{
				BeamPlugin* b = beams[j];
				int idx = flag[j] - 1;
				int rndThetaNum = b->arcNum[idx];
				float thetaSum = 0;
				for (int k = 0; k <= rndThetaNum; k++)
				{
					gws[i].rotationAngle[cnt + k] = (X1[cnt + k] + X2[cnt + k] + X3[cnt + k]) / 3;
					if (k > 0)
						gws[i].rotationAngle[cnt + k] = min(max((double)(gws[i].rotationAngle[cnt + k]), -2 * PI / rndThetaNum * 1.4), -2 * PI / rndThetaNum * 0.2);
					if (k == 0)
						gws[i].rotationAngle[cnt + k] = min(max((double)(gws[i].rotationAngle[cnt + k]), -PI), PI);
					if (k > 0)
						thetaSum += gws[i].rotationAngle[cnt + k];
				}
				//curandState devStates;
				//curand_init((unsigned long long)clock(), j, 0, &devStates);// initialize the state
				//float RANDOM = curand_uniform(&devStates);// uniform distribution
				//RANDOM *= -2 * PI;
				//thetaSum += RANDOM;
				//if (thetaSum < -2 * PI)
				{
					for (int k = 1; k <= rndThetaNum; k++)
					{
						gws[i].rotationAngle[cnt + k] /= thetaSum;
						gws[i].rotationAngle[cnt + k] *= -2 * PI;
					}
				}
				cnt += rndThetaNum + 1;
			}
			free(X1); free(X2); free(X3);
		}
		
		/*for (int k = 0; k < nVar; k++)
		{
			gws[lane_id].rotationAngle[k] = (X1[k] + X2[k] + X3[k]) / 3;
			gws[lane_id].rotationAngle[k] = Min(Max((double)(gws[lane_id].rotationAngle[k]), -PI), PI);
		}*/
		//if (lane_id == 0)
		{
			printf("%d %f batch:%d %d\n", Alpha_inverted, Alpha_score,batchIdx,nodeId);
		}

		record[total_iter] = Alpha_score;
		tmpRecord[tmp_iter] = Alpha_score;
		/*Y.push_back(Alpha_score);
		invertFlag.push_back(Alpha_inverted);*/
		tmp_iter++;

		if (Alpha_inverted == true)
			force_iter++;
		//if (it == MaxIt - 1)
		if (total_iter > 50)
		{
			//判断是否翻转
			if (Alpha_inverted == false && abs(record[(int)total_iter - 1] - record[(int)total_iter - 50]) < 1)
			{
				//if (lane_id == 0)
				{
					/*statis[nodeId].converge = true;
					statis[nodeId].iterNum = force_iter;*/
					//printf("iteration number: %d\n", force_iter);
					//for (int j = 0; j < nVar; j++)
					//	Alpha_pos[j] = 0.2;
					//	//printf("%f ", Alpha_pos[j]);
					//printf("\n");
					int cnt = 0;
					for (int j = 0; j < beamNum; j++)
					{
						BeamPlugin* b = beams[j];
						int idx = flag[j] - 1;
						//b->convergeF[idx] = 1;
						Triangle* tri = triangles + b->arcTriList[idx][0];
						float3 prevP;
						for (int v = 0; v < 3; v++)
						{
							if (tri->triWithBeamid[v] == j)
							{
								float3 diskNorm = tri->diskNorm[v];
								float theta = Alpha_pos[cnt];
								float3 O = tri->o[v];
								float** m = getRotationMatrix(diskNorm, theta, O);

								float3 rotateV = make_float3(tri->p[v].x - O.x, tri->p[v].y - O.y, tri->p[v].z - O.z);
								float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
								RV = Multiply(m, RV);
								tri->p[v] = make_float3(RV.x, RV.y, RV.z);
								prevP = tri->p[v];
								for (int f = 0; f < 4; f++)
								{
									free(m[f]);// = (float*)malloc(4 * sizeof(float));
								}
								free(m);
								break;
							}

						}
						for (int k = 1; k < b->arcNum[idx]; k++)
						{
							//Triangle* tri = triList + b->arcTriList[idx][k];
							Triangle* tri2 = triangles + b->arcTriList[idx][k];
							for (int v = 0; v < 3; v++)
							{
								if (tri2->triWithBeamid[v] == j)
								{
									float3 diskNorm = tri2->diskNorm[v];
									float theta = Alpha_pos[cnt + k];
									float3 O = tri2->o[v];
									float** m = getRotationMatrix(diskNorm, theta, O);

									float3 rotateV = make_float3(prevP.x - O.x, prevP.y - O.y, prevP.z - O.z);
									float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
									RV = Multiply(m, RV);
									tri2->p[v] = make_float3(RV.x, RV.y, RV.z);
									prevP = tri2->p[v];
									for (int f = 0; f < 4; f++)
									{
										free(m[f]);// = (float*)malloc(4 * sizeof(float));
									}
									free(m);
									break;
								}
							}

							//tri = tri2;

						}
						cnt += b->arcNum[idx] + 1;
					}
					//所有三角形的normal更新
					for (int j = 0; j < triNum; j++)
					{
						Triangle* t = &triangles[j];
						float3 e1 = make_float3(t->p[1].x - t->p[0].x, t->p[1].y - t->p[0].y, t->p[1].z - t->p[0].z);
						float3 e2 = make_float3(t->p[2].x - t->p[0].x, t->p[2].y - t->p[0].y, t->p[2].z - t->p[0].z);
						t->normal = cross(e1, e2);
						t->normal = make_norm(t->normal);
						t->flag = 0;
					}
					/*for (int j = 0; j < quadNum; j++)
					{
						QuadFace* q = &quads[j];
						q->convergeF = 1;
					}*/
					//计算cost
					bool inverted = false;
					findAdjTriangles(triangles, triNum);//否则找到的邻居三角形还是以前的
					findQuadFaces(triangles, quads, triNum, inverted);
					measureSingleCost(triangles, triNum, quads, quadNum, gws[0], nVar, inverted);
					printf("inverted: %d\n", inverted);
					/*triangles = triList;
					quads = quadList;*/

				}
				break;
			}
			else if (it == MaxIt - 1)
			{

				/*triList.clear();
				quadList.clear();*/
				it = -1;
				MaxIt = 100;
				//if(record.size()==MaxIt)
				//free(record);
				//record = (float*)malloc(MaxIt * sizeof(float));
				total_iter = -1;
				//record.resize(MaxIt);
				//break;

			}
			//else if (force_iter >= 5000 && it == MaxIt - 1)
			
		}


	}
	for (int i = 0; i < 128; i++)
	{
		free(gws[i].rotationAngle);
	}
	//free(gws);
	free(Alpha_pos);
	//free(Alpha.rotationAngle);
	free(Beta_pos);
	//free(Beta.rotationAngle);
	free(Delta_pos);
}
void pickPos(int* pos, int n,int m)
{
	int max_value;
	int min_value;
	int Pos = 0;
	for (int i = 0; i < m - 1; i++)
	{
		curandState devStates;
		curand_init((unsigned long long)clock(), i, 0, &devStates);// initialize the state
		float RANDOM = curand_uniform(&devStates);// uniform distribution

		max_value = min(n-(m-1-i),n - 1);
		min_value = max(i, Pos);
		if (max_value == min_value)
		{
			Pos = max_value;
			pos[i] = Pos;
			continue;
		}
		RANDOM *= (max_value - min_value + 0.999999);
		RANDOM += min_value;
		Pos = (int)(RANDOM);
		pos[i] = Pos;
	}
}
float measureVariance(int n,int m,Triangle* resultTri,int triNum,int* pos)
{
	//float variance = 0;

	//float avg0 = 0, avg1 = 0;
	////int flag = 1;
	//for (int i = 0; i < triNum; i++)
	//{
	//	avg0 += Norm(Subtraction(resultTri[i].p[2], resultTri[i].p[0]));
	//	if (i < n - 1)
	//		avg1 += Norm(Subtraction(resultTri[i].p[1], resultTri[i].p[0]));
	//	else
	//	{
	//		avg1 += Norm(Subtraction(resultTri[i].p[2], resultTri[i].p[1]));
	//	}
	//}
	//avg0 = avg0 / triNum;
	/*avg1 = avg1 / (n + m - 1);
	float v0 = 0, v1 = 0;
	flag = 1;
	for (int i = 0; i < triNum; i++)
	{
		v0 += pow(Norm(Subtraction(resultTri[i].p[2], resultTri[i].p[0])) - avg0, 2);
		if (i < n - m)
			v1 += pow(Norm(Subtraction(resultTri[i].p[1], resultTri[i].p[0])) - avg1, 2);
		else
		{
			if (flag)
			{
				v1 += pow(Norm(Subtraction(resultTri[i].p[2], resultTri[i].p[1])) - avg1, 2);
				v1 += pow(Norm(Subtraction(resultTri[i + 1].p[1], resultTri[i + 1].p[0])) - avg1, 2);
				flag = 0;
			}
			else
				flag = 1;
		}
	}
	v0 = v0 / triNum;
	v1 = v1 / (n + m - 1);
	variance = v0 + v1;
	return variance;*/
	//float thetaCost = 0;
	float variance = 0,v0=0,v1=0;
	float numfor_m = (float)((n - 1)*1.0) / m;
	float numfor_n = (float)((m - 1)*1.0) / n;
	int pos_idx = 0;
	int triSt = 0, triEd = pos[0];
	for (int i = 0; i < m; i++)
	{
		int tmp_triNum = triEd - triSt;
		v0 += pow(tmp_triNum - numfor_m, 2);
		triSt = triEd;
		if (i < m - 2)
		{
			triEd = pos[pos_idx + 1];
			pos_idx++;
		}
		else
		{
			triEd = n - 1;
		}
	}
	pos_idx = 0;
	int remain_n = n;
	for (; pos_idx < m - 1; pos_idx++)
	{
		int tmp_triNum = 1;
		while(pos_idx + 1 < m - 1)
		{
			if (pos[pos_idx] == pos[pos_idx + 1])
			{
				tmp_triNum++;
				pos_idx++;
			}
			else
				break;
		}
		v1 += pow(tmp_triNum - numfor_n, 2);
		remain_n --;
	}
	for (int i = 0; i < remain_n; i++)
	{
		v1 += pow(numfor_n, 2);
	}
	v0 = v0 / m;
	v1 = v1 / n;
	variance = v0 + v1;

	float cost=0;
	for (int i = 0; i < triNum; i++)
	{
		float3 e1 = Subtraction(resultTri[i].p[1], resultTri[i].p[0]);
		float3 e2 = Subtraction(resultTri[i].p[2], resultTri[i].p[0]);
		e1 = make_norm(e1);
		e2 = make_norm(e2);
		float theta1 = acos(min(max(Dot(e1,e2), -1.0f), 1.0f));
		e1 = Subtraction(resultTri[i].p[2], resultTri[i].p[1]);
		e2 = Subtraction(resultTri[i].p[2], resultTri[i].p[0]);
		e1 = make_norm(e1);
		e2 = make_norm(e2);
		float theta2 = acos(min(max(Dot(e1, e2), -1.0f), 1.0f));
		cost += pow(theta1 - theta2,2);
	}
	return variance;
}
float* computeX123(int* leaderPos, GwsForTriangulation &gws, int nVar, float a)
{

	//% Eq.(3.4) in the paper
	float* c = (float*)malloc(nVar * sizeof(float));

	/*if (it == 0)
	{
		gws.chaoticC.clear();
		gws.chaoticA.clear();
	}*/
	//srand((unsigned)time(NULL));

	for (int i = 0; i < nVar; i++)
	{
		float rndC;
		//if (it == 0)
		{
			/*std::default_random_engine e;
			std::uniform_real_distribution<> u(0.0, nextafter(1.0, DBL_MAX));
			e.seed(time(0));*/
			//Eigen::MatrixXd MC = Eigen::MatrixXd::Random(1, 1);
			curandState devStates;
			/*srand(time(0));
			int seed = rand();*/
			curand_init((unsigned long long)clock(), i, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//curand_init((unsigned long long)(seed*1e9), i, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution

			rndC = abs(RANDOM); //if(rndC==1.0)cout << rndC << endl;
			c[i] = (2 * rndC);
			//printf("%f ", RANDOM);
			//gws.chaoticC.push_back(rndC);
		}
		//else
		//{
		//	rndC = cos(0.5*acos(gws.chaoticC[i]));
		//	//rndC = 0.5*gws.chaoticC[i] * (1 - gws.chaoticC[i]);
		//	c.push_back(2 * rndC);
		//	gws.chaoticC[i] = rndC;
		//}

	}
	float* D = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		D[i] = (abs(c[i] * leaderPos[i] - gws.separatePos[i]));
	}
	float* A = (float*)malloc(nVar * sizeof(float));

	for (int i = 0; i < nVar; i++)
	{
		float rndA;
		//if (it == 0)
		{
			/*std::default_random_engine e;
			std::uniform_real_distribution<> u(0.0, nextafter(1.0, DBL_MAX));
			e.seed(time(0));*/
			//srand(time(0));
			//Eigen::MatrixXd MA = Eigen::MatrixXd::Random(1, 1);
			curandState devStates;
			/*srand(time(0));
			int seed = rand();*/
			curand_init((unsigned long long)clock(), i, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//curand_init((unsigned long long)(seed*1e9), i, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution

			rndA = abs(RANDOM);
			A[i] = (2 * a*rndA - a);
			//gws.chaoticA.push_back(rndA);
		}
		//else
		//{
		//	rndA = cos(0.5*acos(gws.chaoticA[i]));
		//	//rndA = 0.5*gws.chaoticA[i] * (1 - gws.chaoticA[i]);
		//	A.push_back(2 * a*rndA - a);
		//	gws.chaoticA[i] = rndA;
		//}

	}
	float* X = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		X[i] = (leaderPos[i] - A[i] * abs(D[i]));
		//printf("%f ", X[i]);
	}
	free(c);
	free(D);
	free(A);

	return X;
}
void GWOforQuadTriangulation(int n,int m, float3* sample_n,float3* sample_m,Triangle* resultTri,int triNum,bool isCircle)
{
	int GreyWolves_num = 100;
	float MaxIt = 100;

	int* Alpha_pos = (int*)malloc((m-1) * sizeof(int));
	pickPos(Alpha_pos, n, m);
	float Alpha_score = INF;

	int* Beta_pos = (int*)malloc((m - 1) * sizeof(int));
	pickPos(Beta_pos, n, m);
	float Beta_score = INF;

	int* Delta_pos = (int*)malloc((m - 1) * sizeof(int));
	pickPos(Delta_pos, n, m);
	float Delta_score = INF;

	GwsForTriangulation* gws = (GwsForTriangulation*)malloc(GreyWolves_num * sizeof(GwsForTriangulation));

	for (int i = 0; i < GreyWolves_num; i++)
	{
		//gws[i].separatePos = (int*)malloc((m - 1) * sizeof(int));
		pickPos(gws[i].separatePos, n, m);
	}
	int m_pos = 0;
	if (isCircle)
	{
		float3 point = sample_n[0];
		float dis = Norm(Subtraction(point, sample_m[m-1]));
		for (int i = 1; i < m; i++)
		{
			float tmp = Norm(Subtraction(point, sample_m[m-1-i]));
			if (tmp < dis)
			{
				m_pos = i;
				dis = tmp;
			}
		}
	}
	//main loop
	for (float it = 0; it < MaxIt; it++)
	{
		float a = 2 - it * (2 / MaxIt);
		for (int i = 0; i < GreyWolves_num; i++)
		{
			bool skip = false;
			int* pos = gws[i].separatePos;
			int triSt_idx = 0, triEd_idx =pos[0];
			int tri_cnt = 0;
			int pos_idx = 0;
			for (int j = m_pos; j < m+ m_pos; j++)
			{
				int idx;
				if (isCircle)
					idx = mod(j, m - 1);
				else
					idx = mod(j, m);
				int tmp_triNum = triEd_idx - triSt_idx;
				if (tmp_triNum > n)
				{
					skip = true;
					break;
				}
				for (int k = 0; k < tmp_triNum; k++)
				{
					Triangle* tri = &resultTri[tri_cnt++];
					tri->p[0] = sample_n[triSt_idx];
					tri->p[1] = sample_m[m-1-idx];
					tri->p[2] = sample_n[triSt_idx + 1];
					tri->normal = cross(Subtraction(tri->p[1], tri->p[0]), Subtraction(tri->p[2], tri->p[0]));
					triSt_idx++;
				}
				triSt_idx = triEd_idx;
				if (j < m+m_pos - 2)
				{
					triEd_idx = pos[pos_idx + 1];
					pos_idx++;
				}
				else if (j == m+m_pos - 2)
				{
					triEd_idx = n - 1;
					continue;
				}
				
			}
			if (skip)
			{
				//skip = false;
				continue;
			}
			pos_idx = 0;
			for (int j = m_pos; j < m - 1+m_pos; j++)
			{
				int idx = mod(j, m - 1);
				Triangle* tri = &resultTri[tri_cnt++];
				tri->p[0] = sample_m[m-1-(idx + 1)];
				tri->p[1] = sample_n[pos[pos_idx]];
				tri->p[2] = sample_m[m-1-idx];
				tri->normal = cross(Subtraction(tri->p[1], tri->p[0]), Subtraction(tri->p[2], tri->p[0]));
				
				pos_idx++;
			}
			gws[i].singleCost = measureVariance(n, m, resultTri, (n - m) + (m - 1) * 2,pos);
			float fitness = gws[i].singleCost;
			if (fitness < Alpha_score)
			{
				Alpha_score = fitness;
				memcpy(Alpha_pos, gws[i].separatePos, (m - 1) * sizeof(int));
			}
			if (fitness > Alpha_score && fitness < Beta_score)
			{
				Beta_score = fitness;
				memcpy(Beta_pos, gws[i].separatePos, (m - 1) * sizeof(int));
			}
			if (fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score)
			{
				Delta_score = fitness;
				memcpy(Delta_pos, gws[i].separatePos, (m - 1) * sizeof(int));
			}
		}

		for (int i = 0; i < GreyWolves_num; i++)
		{
			float* X1 = computeX123(Alpha_pos, gws[i], m - 1, a);
			float* X2 = computeX123(Beta_pos, gws[i], m - 1, a);
			float* X3 = computeX123(Delta_pos, gws[i], m - 1, a);

			int max_value;
			int min_value;
			int Pos = 0;
			for (int k = 0; k < m - 1; k++)
			{
				gws[i].separatePos[k] = (int)((X1[k] + X2[k] + X3[k]) / 3);

				max_value = min(n - (m - 1 - k), n - 1);
				min_value = max(k, Pos);
				if (max_value == min_value)
				{
					Pos = max_value;
					gws[i].separatePos[k] = Pos;
					continue;
				}
				gws[i].separatePos[k] = min(max(gws[i].separatePos[k], min_value), max_value);
				Pos = gws[i].separatePos[k];
				
			}
			free(X1); free(X2); free(X3);
		}
		//printf("%f\n", Alpha_score);

		if (it == MaxIt - 1)
		{
			int* pos = Alpha_pos;
			int triSt_idx = 0, triEd_idx = pos[0];
			int tri_cnt = 0;
			int pos_idx = 0;
			for (int j = m_pos; j < m + m_pos; j++)
			{
				int idx;
				if (isCircle)
					idx = mod(j, m - 1);
				else
					idx = mod(j, m);
				int tmp_triNum = triEd_idx - triSt_idx;
				for (int k = 0; k < tmp_triNum; k++)
				{
					Triangle* tri = &resultTri[tri_cnt++];
					tri->p[0] = sample_n[triSt_idx];
					tri->p[1] = sample_m[m - 1 - idx];
					tri->p[2] = sample_n[triSt_idx + 1];
					tri->normal = cross(Subtraction(tri->p[1], tri->p[0]), Subtraction(tri->p[2], tri->p[0]));
					tri->normal = make_norm(tri->normal);
					if (isCircle)
					{
						float3 tmp = tri->p[0];
						tri->p[0] = tri->p[2];
						tri->p[2] = tmp;
						tri->normal = make_float3(-tri->normal.x, -tri->normal.y, -tri->normal.z);
					}
					triSt_idx++;
				}
				triSt_idx = triEd_idx;
				if (j < m + m_pos - 2)
				{
					triEd_idx = pos[pos_idx + 1];
					pos_idx++;
				}
				else if (j == m + m_pos - 2)
				{
					triEd_idx = n - 1;
					continue;
				}

			}
			pos_idx = 0;
			for (int j = m_pos; j < m - 1 + m_pos; j++)
			{
				int idx = mod(j, m - 1);
				Triangle* tri = &resultTri[tri_cnt++];
				tri->p[0] = sample_m[m - 1 - (idx + 1)];
				tri->p[1] = sample_n[pos[pos_idx]];
				tri->p[2] = sample_m[m - 1 - idx];
				tri->normal = cross(Subtraction(tri->p[1], tri->p[0]), Subtraction(tri->p[2], tri->p[0]));
				tri->normal = make_norm(tri->normal);
				
				if (isCircle)
				{
					float3 tmp = tri->p[0];
					tri->p[0] = tri->p[2];
					tri->p[2] = tmp;
					tri->normal = make_float3(-tri->normal.x, -tri->normal.y, -tri->normal.z);
					
				}
				pos_idx++;
			}
		}
	}
	/*for (int i = 0; i < GreyWolves_num; i++)
	{
		free(gws[i].separatePos);
	}*/
	free(gws);
	free(Alpha_pos);
	free(Beta_pos);
	free(Delta_pos);

}
void GWOforBeamTriangulation()
{

}
void MultiObjectiveGreyWolfOptimizer(int nVar, int nObj, Triangle* &triangles,QuadFace* &quads,int triNum,int quadNum)
{
	int GreyWolves_num = 100;
	float MaxIt = 100;
	int Archive_size = 100;

	float alpha = 0.1;
	int nGrid = 10;
	float beta = 4;
	int gamma = 2;
	std::vector<GreyWolves> gws;
	gws.resize(GreyWolves_num);
	//GreyWolves* gws = new GreyWolves[GreyWolves_num];
	//Initialization
	for (int i = 0; i < GreyWolves_num; i++)
	{
		gws[i].rotationAngle = new float[nVar];
		
		for (int j = 0; j < nVar; j++)
		{
			//初始化position即随机旋转角
			curandState devStates;
			//srand(time(0));
			//int seed = rand();
			curand_init((unsigned long long)clock(), j, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//Eigen::MatrixXd R = Eigen::MatrixXd::Random(1, 1);
			//curand_init((unsigned long long)(seed*1e9), 0, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution
			RANDOM = (RANDOM - 0.5) * 2 * PI;
			gws[i].rotationAngle[j] = RANDOM;
		}
		//根据旋转角度计算三角形顶点坐标
		//拷贝一份triangles，暂时不能真正旋转
		Triangle* triList;
		QuadFace* quadList;
		//int triNum = triangles.size();
		//Triangle* triT = new Triangle[triNum];
		//vector<shared_ptr<Triangle>> triTmp;
		//triTmp.resize(triNum);
		triList = (Triangle*)malloc(triNum * sizeof(Triangle));
		quadList = (QuadFace*)malloc(quadNum * sizeof(QuadFace));
		
		for (int j = 0; j < triNum; j++)
		{
			triList[j] = triangles[j];
			//triList.push_back(&triTmp[j]);
		}
		//计算在当前三角形位置上旋转随机角度后的位置
		for (int j = 0; j < triNum; j++)
		{
			Triangle t = triList[j];
			for (int k = 0; k < 3; k++)
			{
				float3 diskNorm = t.diskNorm[k];
				float theta = gws[i].rotationAngle[j * 3 + k];
				float3 O = t.o[k];
				float** m = getRotationMatrix(diskNorm, theta, O);

				float3 rotateV = make_float3(t.p[k].x - O.x, t.p[k].y - O.y, t.p[k].z - O.z);
				float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
				RV = Multiply(m, RV);
				t.p[k] = make_float3(RV.x, RV.y, RV.z);
			}
			float3 e1 = make_float3(t.p[1].x - t.p[0].x, t.p[1].y - t.p[0].y, t.p[1].z - t.p[0].z);
			float3 e2 = make_float3(t.p[2].x - t.p[0].x, t.p[2].y - t.p[0].y, t.p[2].z - t.p[0].z);
			t.normal = cross(e1, e2);
			t.normal = make_norm(t.normal);
			triList[j] = t;
			
			//triList.push_back(triTmp[j]);
		}
		//计算cost
		bool inverted = false;
		findAdjTriangles(triList,triNum);//否则找到的邻居三角形还是以前的
		findQuadFaces(triList, quadList,triNum,inverted);
		measureDistance(triList,quadList,triNum,quadNum,nObj,gws[i],nVar);
		gws[i].inverted = inverted;

		//delete[]triTmp;
		free(triList);
		free(quadList);
	}

	DetermineDomination(gws, GreyWolves_num,nObj);
	std::vector<GreyWolves> Archive = GetNonDominatedParticles(gws,GreyWolves_num,nVar);
	//create hypercubes
	std::vector<float> *Archive_costs = GetCosts(Archive,nObj);
	std::vector<float> *G = CreateHypercubes(Archive_costs,nObj, nGrid, alpha);
	
	int ArchiveSize = Archive.size();
	for (int i = 0; i < ArchiveSize; i++)
	{
		GetGridIndex(Archive[i],G,nObj);
	}

	float* tmpRecord = (float*)malloc(20 * sizeof(float));
	float* tmpRecord2 = (float*)malloc(20 * sizeof(float));
	int tmp_iter = 0;

	//main loop
	for (float it = 0; it < MaxIt; it++)
	{
		float a = 2 - it * ((2) / MaxIt);
		for (int i = 0; i < GreyWolves_num; i++)
		{
			int h1, h2,h3;
			GreyWolves Delta = SelectLeader(Archive, beta,h1);
			GreyWolves Beta = SelectLeader(Archive, beta,h2);
			GreyWolves Alpha = SelectLeader(Archive, beta,h3);

			//这里就是exclude操作
			std::vector<GreyWolves> rep2, rep3;
			if (Archive.size() > 1)
			{
				rep2 = Archive;
				auto iter = rep2.begin() + h1;
				rep2.erase(iter);
				Beta = SelectLeader(rep2, beta,h2);
			}
			if (Archive.size() > 2)
			{
				rep3 = rep2;
				auto iter = rep3.begin() + h2;
				rep3.erase(iter);
				Alpha = SelectLeader(rep3, beta,h3);
			}

			float* X1 = computeX123(Delta.rotationAngle,gws[i],nVar,a,0);
			float* X2 = computeX123(Beta.rotationAngle, gws[i], nVar, a, 0);
			float* X3 = computeX123(Alpha.rotationAngle, gws[i], nVar, a, 0);
			for (int k = 0; k < nVar; k++)
			{
				gws[i].rotationAngle[k] = (X1[k] + X2[k] + X3[k]) / 3;
				gws[i].rotationAngle[k] = std::min(std::max((double)(gws[i].rotationAngle[k]), -PI), PI);
			}
			//计算cost
			//根据旋转角度计算三角形顶点坐标
		    //拷贝一份triangles，暂时不能真正旋转
			Triangle* triList;
			QuadFace* quadList;
			//int triNum = triangles.size();
			//Triangle* triTmp = new Triangle[triNum];
			//vector<shared_ptr<Triangle>> triTmp;
			//triTmp.resize(triNum);
			triList = (Triangle*)malloc(triNum * sizeof(Triangle));
			quadList = (QuadFace*)malloc(quadNum * sizeof(QuadFace));
			
			for (int j = 0; j < triNum; j++)
			{
				triList[j] = triangles[j];
				//triList.push_back(&triTmp[j]);
			}
			//计算在当前三角形位置上旋转随机角度后的位置
			for (int j = 0; j < triNum; j++)
			{
				Triangle t = triList[j];
				for (int k = 0; k < 3; k++)
				{
					float3 diskNorm = t.diskNorm[k];
					float theta = gws[i].rotationAngle[j * 3 + k];
					float3 O = t.o[k];
					float** m = getRotationMatrix(diskNorm, theta, O);

					float3 rotateV = make_float3(t.p[k].x - O.x, t.p[k].y - O.y, t.p[k].z - O.z);
					float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
					RV = Multiply(m, RV);
					t.p[k] = make_float3(RV.x, RV.y, RV.z);
				}
				float3 e1 = make_float3(t.p[1].x - t.p[0].x, t.p[1].y - t.p[0].y, t.p[1].z - t.p[0].z);
				float3 e2 = make_float3(t.p[2].x - t.p[0].x, t.p[2].y - t.p[0].y, t.p[2].z - t.p[0].z);
				t.normal = cross(e1, e2);
				t.normal = make_norm(t.normal);
				triList[j] = t;

				//triList.push_back(triTmp[j]);
			}
			//计算cost
			bool inverted = false;
			findAdjTriangles(triList,triNum);//否则找到的邻居三角形还是以前的
			findQuadFaces(triList,quadList,triNum,inverted);
			measureDistance(triList,quadList,triNum,quadNum,nObj,gws[i],nVar);
			gws[i].inverted = inverted;

			//delete []triTmp;
			free(triList);
			free(quadList);
		}

		DetermineDomination(gws, GreyWolves_num,nObj);
		std::vector<GreyWolves> non_dominated_wolves = GetNonDominatedParticles(gws, GreyWolves_num,nVar);

		Archive.insert(Archive.end(), non_dominated_wolves.begin(), non_dominated_wolves.end());

		DetermineDomination(Archive, Archive.size(),nObj);
		Archive = GetNonDominatedParticles(Archive, Archive.size(),nVar);

		int ArchiveSize = Archive.size();
		for (int i = 0; i < ArchiveSize; i++)
		{
			
			GetGridIndex(Archive[i], G,nObj);
		}

		if (ArchiveSize > Archive_size)
		{
			int EXTRA = ArchiveSize - Archive_size;
			DeleteFromRep(Archive,EXTRA,gamma);

			delete[] Archive_costs;
			Archive_costs = GetCosts(Archive,nObj);

			delete[] G;
			G = CreateHypercubes(Archive_costs, nObj, nGrid, alpha);
		}
		//取得Alpha
		Triangle* triList;
		QuadFace* quadList;
		int H;
		GreyWolves Alpha = selectSolution(triList,triNum,quadList,quadNum,Archive, beta, triangles,H,nObj,nVar);
		std::cout <<Alpha.inverted<<" "<< Alpha.Cost[0] << " " << Alpha.Cost[1]  << std::endl;

		tmpRecord[tmp_iter] = Alpha.Cost[0];
		tmpRecord2[tmp_iter] = Alpha.Cost[1];
		/*Y.push_back(Alpha_score);
		invertFlag.push_back(Alpha_inverted);*/
		tmp_iter++;
		//重新随机化，加速收敛
		if (tmp_iter > 10 && tmp_iter <= 20)
		{
			if ((tmpRecord[tmp_iter - 1] == *(tmpRecord + tmp_iter - 7))|| (tmpRecord2[tmp_iter - 1] == *(tmpRecord2 + tmp_iter - 7)))
			{
				for (int i = 0; i < GreyWolves_num; i++)
				{
					//gws[i].rotationAngle.clear();
					for (int j = 0; j < nVar; j++)
					{
						////初始化position即随机旋转角
						//std::default_random_engine e;
						//std::uniform_real_distribution<float> u(-PI, PI);
						//e.seed(time(0));
						curandState devStates;
						//srand(time(0));
						//int seed = rand();
						curand_init((unsigned long long)clock(), j, 0, &devStates);// initialize the state
						float RANDOM = curand_uniform(&devStates);// uniform distribution
						//Eigen::MatrixXd R = Eigen::MatrixXd::Random(1, 1);
						//curand_init((unsigned long long)(seed*1e9), 0, 0, &devStates2);// initialize the state
						//float RANDOM = curand_uniform(&devStates2);// uniform distribution
						RANDOM = (RANDOM - 0.5) * 2 * PI;
						//float RANDOM = (abs(R(0,0)) - 0.5) * 2 * PI;

						gws[i].rotationAngle[j] = RANDOM;
					}

				}
				
				free(tmpRecord);
				free(tmpRecord2);
				tmpRecord = (float*)malloc(20 * sizeof(float));
				tmpRecord2 = (float*)malloc(20 * sizeof(float));
				tmp_iter = 0;
			}
		}
		if (tmp_iter >= 20)
		{
			free(tmpRecord);
			free(tmpRecord2);
			tmpRecord = (float*)malloc(20 * sizeof(float));
			tmpRecord2 = (float*)malloc(20 * sizeof(float));
			tmp_iter = 0;
		}

		std::vector<GreyWolves> rep;
		rep = Archive;

		
		while (it == MaxIt - 1)
		{
			//if (Alpha.Cost[0]>0.45 && Alpha.Cost[1] > 0.45)
			if(Alpha.inverted==false)
			{
				for (int i = 0; i < triNum; i++)
				{
					triangles[i] = triList[i];
				}
				for (int i = 0; i < quadNum; i++)
				{
					quads[i] = quadList[i];
				}
				
				break;
			}
			else
			{
				
				/*delete[] triList;
				delete[] quadList;*/
				/*triList.clear();
				quadList.clear();*/
				//这里就是exclude操作
				std::vector<GreyWolves> rep2;
				if (rep.size() > 1)
				{
					free(triList);
					free(quadList);

					rep2 = rep;
					auto iter = rep2.begin() + H;
					rep2.erase(iter);
					//GreyWolves Beta = SelectLeader(rep2, beta, h2);
					Alpha = selectSolution(triList,triNum, quadList,quadNum, rep2, beta, triangles, H,nObj,nVar);
					std::cout << rep2.size() << " " << Alpha.inverted << " "<< Alpha.Cost[0] << " " << Alpha.Cost[1] << std::endl;
					rep = rep2;
				}
				else
				{
					it = -1;
					MaxIt=100;
					break;
				}
				
			}
			
		}
		free(triList);
		free(quadList);
	}
}