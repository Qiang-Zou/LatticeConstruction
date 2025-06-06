#include<cstdio>
#include<iostream>
#include<vector>
#include<cmath>
#include<queue>
#include "beam.h"
#include "topology.h"
//#include "convexHull.h"
//#include "convhull_3d.h"
#include<cstring>
#include<Eigen/Dense>
#include <Eigen/Geometry>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil.h"
#include "call_cuda.h"
#include<tbb/tbb.h>
#define CONVHULL_3D_ENABLE
#define READFILE
//#define _CRT_SECURE_NO_WARNINGS
#define PARALLEL_NODE


using namespace Eigen;
using namespace std;
bool convertToPLY(string path, vector<Face> face)
{
	std::ofstream file(path);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << path << std::endl;
		return false;
	}
	int fNum = face.size();
	int vNum = 3 * fNum;
	// PLY header
	file << "ply\n"
		<< "format ascii 1.0\n"
		<< "element vertex "<<vNum<<"\n"
		<< "property float x\n"
		<< "property float y\n"
		<< "property float z\n"
		<< "element face "<<fNum<<"\n"
		<< "property list uchar int vertex_index\n"
		<< "property uchar red\n"
		<< "property uchar green\n"
		<< "property uchar blue\n"
		<< "end_header\n";
	for (int i = 0; i < face.size(); i++)
	{
		file << face[i].p1.x <<" "<< face[i].p1.y <<" "<< face[i].p1.z << endl;
		file << face[i].p2.x << " " << face[i].p2.y << " " << face[i].p2.z << endl;
		file << face[i].p3.x << " " << face[i].p3.y << " " << face[i].p3.z << endl;
	}
	for (int i = 0; i < face.size(); i++)
	{
		if (face[i].type==1)
		{
			file << "3 "<< 3*i<<" "<<3*i+1<<" "<<3*i+2<<" "<<"0 68 255" << endl;     //blue
		}
		else if(face[i].type==2)
		{
			file << "3 " << 3 * i << " " << 3 * i + 1 << " " << 3 * i + 2 << " " << "0 205 134" << endl;  // Green
		}
		else if (face[i].type == 3)
		{
			file << "3 " << 3 * i << " " << 3 * i + 1 << " " << 3 * i + 2 << " " << "255 204 153" << endl; //Red
		}
	}
	file.close();
	return true;
}
bool convertToSTL(string path, vector<Face> face)
{
	char head[128];//文件头
	strcpy(head, "solid ");
	strcat(head, "mystl");
	FILE* fw = fopen(path.c_str(), "w");
	if (fw != NULL)
	{
		fprintf(fw, "%s\n", head);//先写入文件头
		for (int i = 0; i < face.size(); i++)
		{
			fprintf(fw, "%s %lf %lf %lf\n", "facet normal", face[i].normal.x, face[i].normal.y, face[i].normal.z);
			fprintf(fw, "%s\n", "outor loop");
			fprintf(fw, "%s %lf %lf %lf\n", "vertex ", face[i].p1.x, face[i].p1.y, face[i].p1.z);
			fprintf(fw, "%s %lf %lf %lf\n", "vertex ", face[i].p2.x, face[i].p2.y, face[i].p2.z);
			fprintf(fw, "%s %lf %lf %lf\n", "vertex ", face[i].p3.x, face[i].p3.y, face[i].p3.z);
			fprintf(fw, "%s\n", "endloop");
			fprintf(fw, "%s\n", "endfacet");

		}

		fprintf(fw, "%s\n", "endsolid");
	}
	else
		return false;

	return true;
}
bool processTQB(string path,Triangle* AllTriangle,int totalMeshNum, QuadFace* AllQuad,int totalQuadNum,ArcType* AllArc,short* AllFlag, vector<BeamPlugin*> G[], int* beamNumVec, int plines,BeamPlugin** beamVector,int elines)
{
	for (int i = 1; i <= plines; i++)
	{
		int st = 0;
		int meshSt = 0;
		int quadSt = 0;
		int arcSt = 0;
		int flagSt = 0;
		for (int j = 1; j < i; j++)
		{
			st += beamNumVec[j];
			int meshNumincre = 8 + (beamNumVec[j] - 6) * 2;
			if (beamNumVec[j] < 4)meshNumincre = 0;
			meshSt += meshNumincre;
			int quadNumincre = 6 + 3 * ((meshNumincre - 4) / 2);
			quadSt += quadNumincre;
			arcSt += 2 * quadNumincre;
			if (beamNumVec[j] >= 4)flagSt += beamNumVec[j];
		}
		
		locateArcs(G[i], AllTriangle + meshSt, AllQuad + quadSt, AllArc + arcSt, AllFlag + flagSt,i);
	}
	std::ofstream file(path);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << path << std::endl;
		return false;
	}
	//write triangles
	file << totalMeshNum << endl;
	for (int i = 0; i < totalMeshNum; i++)
	{
		Triangle* t = AllTriangle + i;
		file << t->p[0].x << " " << t->p[0].y << " " << t->p[0].z <<" "<<t->nodeid<< endl;
		file << t->p[1].x << " " << t->p[1].y << " " << t->p[1].z <<" "<<t->nodeid<< endl;
		file << t->p[2].x << " " << t->p[2].y << " " << t->p[2].z <<" "<<t->nodeid<< endl;
	}
	//write quads
	file << totalQuadNum << endl;
	for (int i = 0; i < totalQuadNum; i++)
	{
		QuadFace* q = AllQuad + i;
		//two arcs
		for (int j = 0; j < 2; j++)
		{
			file << q->arc[j]->st.x << " " << q->arc[j]->st.y << " " << q->arc[j]->st.z <<" "<<q->nodeid<< endl;
			file << q->arc[j]->ed.x << " " << q->arc[j]->ed.y << " " << q->arc[j]->ed.z << " " << q->nodeid << endl;
			float** m = getRotationMatrix(q->arc[j]->diskNorm, q->arc[j]->theta / 2, q->arc[j]->o);
			float3 rotateV = make_float3(q->arc[j]->st.x - q->arc[j]->o.x,
				q->arc[j]->st.y - q->arc[j]->o.y, q->arc[j]->st.z - q->arc[j]->o.z);
			float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
			RV = Multiply(m, RV);
			float3 midpoint = make_float3(RV.x, RV.y, RV.z);
			file << midpoint.x << " " << midpoint.y << " " << midpoint.z << " " << q->nodeid << endl;
			for (int f = 0; f < 4; f++)
			{
				free(m[f]);// = (float*)malloc(4 * sizeof(float));
			}
			free(m);
		}
		//two straight lines
		file<< q->arc[0]->st.x << " " << q->arc[0]->st.y << " " << q->arc[0]->st.z << " " << q->nodeid << endl;
		file<< q->arc[1]->ed.x << " " << q->arc[1]->ed.y << " " << q->arc[1]->ed.z << " " << q->nodeid << endl;

		file<< q->arc[0]->ed.x << " " << q->arc[0]->ed.y << " " << q->arc[0]->ed.z << " " << q->nodeid << endl;
		file<< q->arc[1]->st.x << " " << q->arc[1]->st.y << " " << q->arc[1]->st.z << " " << q->nodeid << endl;
	}
	//write beams
	file << elines << endl;
	for (int i = 1; i <= elines; i++)
	{
		BeamPlugin* b = beamVector[i];
		file << b->axis.p[0].x << " " << b->axis.p[0].y << " " << b->axis.p[0].z <<" "<<b->nodeid[0]<<" "<<b->nodeid[1]<< endl;
		file << b->axis.p[1].x << " " << b->axis.p[1].y << " " << b->axis.p[1].z << " " << b->nodeid[0] << " " << b->nodeid[1] << endl;
		file << b->radius * 2 << " " << b->length << " " << b->nodeid[0] << " " << b->nodeid[1] << endl;
	}
	return true;
}
__device__  float3 cross(float3 a, float3 b)
{
	float3 result;
	result.x = a.y*b.z - b.y*a.z;
	result.y = -(a.x*b.z - b.x*a.z);
	result.z = a.x*b.y - b.x*a.y;
	return result;
}
__device__  float4 Multiply(float** m, float4 a)
{
	float4 result;
	result.x = m[0][0] * a.x + m[0][1] * a.y + m[0][2] * a.z + m[0][3] * a.w;
	result.y = m[1][0] * a.x + m[1][1] * a.y + m[1][2] * a.z + m[1][3] * a.w;
	result.z = m[2][0] * a.x + m[2][1] * a.y + m[2][2] * a.z + m[2][3] * a.w;
	result.w = m[3][0] * a.x + m[3][1] * a.y + m[3][2] * a.z + m[3][3] * a.w;

	return result;
}
__device__  float3 make_norm(float3 a)
{
	float norm = sqrt(pow(a.x,2)+pow(a.y,2)+pow(a.z,2));
	float3 result = make_float3(a.x / norm, a.y / norm, a.z / norm);
	return result;
}
__device__  float Dot(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__  float Norm(float3 a)
{
	return sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2));
}
__device__ float3 Subtraction(float3 a, float3 b)
{
	float3 result = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	return result;
}
__device__ float3 ComputeNegativeCircumcenter(float3 P[])
{
	float D = (2 * pow(Norm(cross(Subtraction(P[0], P[1]), Subtraction(P[1], P[2]))), 2));

	float alpha = pow(Norm(Subtraction(P[1], P[2])), 2)*(Dot(Subtraction(P[0], P[1]), Subtraction(P[0], P[2])));
	alpha = alpha / D;

	float beta = pow(Norm(Subtraction(P[0], P[2])), 2)*(Dot(Subtraction(P[1], P[0]), Subtraction(P[1], P[2])));
	beta = beta / D;

	float gamma = pow(Norm(Subtraction(P[0], P[1])), 2)*(Dot(Subtraction(P[2], P[0]), Subtraction(P[2], P[1])));
	gamma = gamma / D;

	float3 result = make_float3(-(alpha*P[0].x + beta * P[1].x + gamma * P[2].x), -(alpha*P[0].y + beta * P[1].y + gamma * P[2].y),
		-(alpha*P[0].z + beta * P[1].z + gamma * P[2].z));

	return result;
}
void recursiveOptimalCut(vector<BeamPlugin*> &beams,int i, int size,float3* direct,double* intersectionLength)
{
	//optimal cut
	//for (int i = cutId; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (j == i)continue;
			BeamPlugin b1 = *beams[i];
			BeamPlugin b2 = *beams[j];
			double r1 = b1.radius, r2 = b2.radius;

			double cosVal = Dot(direct[i], direct[j]) / (Norm(direct[i])*Norm(direct[j]));
			double angle = acos(cosVal);
			/*if (angle >= PI / 2)
				continue;*/

			double length1 = sqrt(pow(r1, 2) + pow(intersectionLength[i], 2));
			double itoj = angle - atan(r1 / intersectionLength[i]);
			double cut1 = length1 * cos(itoj);
			if (cut1 > intersectionLength[j])
			{
				intersectionLength[j] = cut1;
				recursiveOptimalCut(beams, j, size, direct, intersectionLength);
			}

			double length2 = sqrt(pow(r2, 2) + pow(intersectionLength[j], 2));
			double jtoi = angle - atan(r2 / intersectionLength[j]);
			double cut2 = length2 * cos(jtoi);
			if (cut2 > intersectionLength[i])
			{
				intersectionLength[i] = cut2;
				recursiveOptimalCut(beams, i, size, direct, intersectionLength);
			}
		}
	}
}
struct cmp
{
	bool operator()(BeamPlugin* a, BeamPlugin* b)
	{
		return a->cutLength < b->cutLength;
	}
};
bool CMP(BeamPlugin* a, BeamPlugin* b)
{
	return a->cutLength < b->cutLength;
}
bool CMP2(BeamPlugin a, BeamPlugin b)
{
	return a.cutOtherLength < b.cutOtherLength;
}
void computeIntersectionLength(vector<BeamPlugin*> &beams, float3 startP,float* longestLength)
{
	//priority_queue<BeamPlugin*, vector<BeamPlugin*>, cmp>N[60];
	vector<BeamPlugin>N[60];
	bool visited[60] = { false };
	//std::cout << startP[0] << " " << startP[1] << " " << startP[2] << std::endl;
	//更新axis的起点和length
	int size = beams.size();
	BeamPlugin* tmpBeams = (BeamPlugin*)malloc(size * sizeof(BeamPlugin));
	for (int i = 0; i < size; i++)
	{
		tmpBeams[i] = *beams[i];
		tmpBeams[i].beamId = i;
		tmpBeams[i].cutLength = 0;
	}
	double threshold = 0.2;
	double* intersectionLength = new double[size];
	fill(intersectionLength, intersectionLength+size,0);
	float3* direct = new float3[size];
	for (int i = 0; i < size - 1; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			BeamPlugin *b1 = &tmpBeams[i];
			BeamPlugin *b2 = &tmpBeams[j];
			double r1 = b1->radius, r2 = b2->radius;
			Edge e1 = b1->axis; Edge e2 = b2->axis;
			float3 st, ed;
			if (sqrt(pow(startP.x - b1->axis.p[0].x, 2) + pow(startP.y - b1->axis.p[0].y, 2) + pow(startP.z - b1->axis.p[0].z, 2))<1e-6)
			{
				st = b1->axis.p[0];
				ed = b1->axis.p[1];
			}
			else
			{
				st = b1->axis.p[1];
				ed = b1->axis.p[0];
			}
			float3 direc1 = make_float3(ed.x - st.x, ed.y - st.y, ed.z - st.z);
			direc1 = make_norm(direc1); direct[i] = direc1;
			if (sqrt(pow(startP.x - b2->axis.p[0].x, 2) + pow(startP.y - b2->axis.p[0].y, 2) + pow(startP.z - b2->axis.p[0].z, 2)) < 1e-6)
			{
				st = b2->axis.p[0];
				ed = b2->axis.p[1];
			}
			else
			{
				st = b2->axis.p[1];
				ed = b2->axis.p[0];
			}
			float3 direc2 = make_float3(ed.x - st.x, ed.y - st.y, ed.z - st.z);
			direc2 = make_norm(direc2);  direct[j] = direc2;
			double cosVal = Dot(direc1,direc2) / (Norm(direc1)*Norm(direc2));
			if (cosVal + 1 < 1e-1)
			{
				continue;
			}
			double angle = acos(cosVal);//是弧度角
			double beta = atan((r1+r2*cos(angle))/(r2*sin(angle)));
			double alpha = beta + angle - PI / 2;
			double ltmp1 = r1/tan(alpha), ltmp2 = tan(beta)*r2;
			if (ltmp1 > 0 || ltmp2 > 0)
			{
				BeamPlugin b2tmp = *b2;
				b2tmp.cutOtherLength = ltmp1;
				b2tmp.cutLength = ltmp2;
				N[i].push_back(b2tmp);
				BeamPlugin b1tmp = *b1;
				b1tmp.cutOtherLength = ltmp2;
				b1tmp.cutLength = ltmp1;
				N[j].push_back(b1tmp);
			}
			if (ltmp1 > b1->cutLength)
			{
				//intersectionLength[i] = ltmp1;
				b1->cutLength = ltmp1;
			}
			if (ltmp2 > b2->cutLength)
			{
				//intersectionLength[j] = ltmp2;
				b2->cutLength = ltmp2;
			}
			
		}
	}
	//priority_queue<BeamPlugin*,vector<BeamPlugin*>,cmp>pq;
	vector<BeamPlugin*>pq;
	for (int i = 0; i < size; i++)
	{
		longestLength[i] = tmpBeams[i].cutLength+threshold*2;
		pq.push_back(&tmpBeams[i]);
		if (!N[i].empty())
		{
			make_heap(N[i].begin(), N[i].end(), CMP2);
			
		}
	}
	make_heap(pq.begin(), pq.end(), CMP);
	while (!pq.empty())
	{
		BeamPlugin* self = pq[0];
		int beamid = self->beamId;
		
		while (!N[beamid].empty())
		{
			//BeamPlugin* self = &tmpBeams[beamid];
			BeamPlugin neighbor = N[beamid][0];
			while (visited[neighbor.beamId])
			{
				pop_heap(N[beamid].begin(), N[beamid].end(), CMP2);
				N[beamid].pop_back();
				if (!N[beamid].empty())
					neighbor = N[beamid][0];
				else
					break;
			}
			if (N[beamid].empty())
			{
				visited[beamid] = true;
				pop_heap(pq.begin(), pq.end(), CMP);
				pq.pop_back();
				break;
			}
			if (self->cutLength != N[beamid][0].cutOtherLength)
			{
				self->cutLength = N[beamid][0].cutOtherLength;
				make_heap(pq.begin(), pq.end(), CMP);
				break;
			}


			if(self->cutLength>intersectionLength[beamid])
			intersectionLength[beamid] = self->cutLength;
			while (!N[beamid].empty())
			{
				double cosVal = Dot(direct[beamid], direct[neighbor.beamId]) / (Norm(direct[beamid])*Norm(direct[neighbor.beamId]));
				if (cosVal < 0)
				{
					//重新算neighbor要cut的长度，可能更短了
					double theta1 = acos(min(max(cosVal, -1.0), 1.0));
					double L1 = (self->radius + neighbor.radius*cosVal) / sin(theta1);
					double theta2 = acos(min(max(cosVal, -1.0), 1.0)) - PI / 2;
					double tmp = intersectionLength[beamid] / cos(theta2);
					double L2=0;
					if (tmp < neighbor.radius)
					{
						double tmp2 = neighbor.radius - tmp;
						L2 = tmp2 / tan(theta2);
					}
					double L = min(L1, L2);//取两者中较小值
					if (L > intersectionLength[neighbor.beamId])
						intersectionLength[neighbor.beamId] = L;
				}
				pop_heap(N[beamid].begin(), N[beamid].end(), CMP2);
				N[beamid].pop_back();
				if (!N[beamid].empty())
					neighbor = N[beamid][0];
				else
					break;
				while (visited[neighbor.beamId])
				{
					pop_heap(N[beamid].begin(), N[beamid].end(), CMP2);
					N[beamid].pop_back();
					if (!N[beamid].empty())
						neighbor = N[beamid][0];
					else
						break;
				}
				
			}
			
			
			if (N[beamid].empty())
			{
				visited[beamid] = true;
				pop_heap(pq.begin(), pq.end(), CMP);
				pq.pop_back();
				break;
			}
			/*if (cosVal<0 && neighbor.cutLength>intersectionLength[neighbor.beamId])
				intersectionLength[neighbor.beamId] = neighbor.cutLength;*/
			//if (self->cutLength > 0 && neighbor->cutLength > 0)
			//{
			//	
			//	if (abs(pow(self->cutLength, 2) + pow(self->radius, 2) - pow(neighbor->cutLength, 2) - pow(neighbor->radius, 2))<1e-6)
			//	{
			//		intersectionLength[beamid] = self->cutLength;
			//		//pq.pop();
			//		if(cosVal<0&&neighbor->cutLength>intersectionLength[neighbor->beamId])
			//		intersectionLength[neighbor->beamId] = neighbor->cutLength;
			//	}
			//	else
			//	{
			//		//看neighbor是否与当前杆相交，若相交，需要再切
			//		if (cosVal < 0)
			//		{
			//			double theta = acos(min(max(cosVal, -1.0), 1.0))-PI/2;
			//			double tmp = self->cutLength / cos(theta);
			//			if (tmp < neighbor->radius)
			//			{
			//				double tmp2 = neighbor->radius - tmp;
			//				double L = tmp2 / tan(theta);
			//				if (L > intersectionLength[neighbor->beamId])
			//					intersectionLength[neighbor->beamId] = L;
			//			}
			//		}
			//	}
			//}
			//else if(abs(neighbor->cutLength)<1e-6)
			//{
			//	if (cosVal < 0)
			//	{
			//		double theta = acos(min(max(cosVal, -1.0), 1.0)) - PI / 2;
			//		double R = self->cutLength*tan(theta);
			//		if (abs(R - self->radius) < 1e-6)
			//		{
			//			intersectionLength[beamid] = self->cutLength;
			//			//pq.pop();
			//		}
			//	}
			//}
			/*pop_heap(N[beamid].begin(), N[beamid].end(), CMP);
			N[beamid].pop_back();*/
			//N[beamid].pop();
		}
		/*visited[beamid] = true;
		pop_heap(pq.begin(), pq.end(), CMP);
		pq.pop_back();*/
		//while (!pq.empty() && pq[0]->beamId == b->beamId)
		//{
		//	pop_heap(pq.begin(), pq.end(), CMP);
		//	pq.pop_back();
		//	//pq.pop();
		//}
		//for (int i = 0; i < size; i++)
		//{
		//	if (visited[i] == false)
		//	{
		//		if (!N[i].empty()&&N[i][0]->beamId == b->beamId)
		//		{
		//			pop_heap(N[i].begin(), N[i].end(), CMP);
		//			N[i].pop_back();
		//			//N[i].pop();
		//			while (!N[i].empty() && visited[N[i][0]->beamId])
		//			{
		//				pop_heap(N[i].begin(), N[i].end(), CMP);
		//				N[i].pop_back();
		//				//N[i].pop();
		//			}
		//			if (!N[i].empty())
		//			{
		//				pq.push_back(N[i][0]);
		//				push_heap(pq.begin(), pq.end(), CMP);
		//			}
		//		}
		//	}
		//	
		//}
		
	}
	/*BeamPlugin *a = new BeamPlugin(); 
	BeamPlugin *b = new BeamPlugin();
	BeamPlugin *c = new BeamPlugin();
	a->cutLength = 1; b->cutLength = 3; c->cutLength = 5; vector<BeamPlugin*>V;
	V.push_back(a); V.push_back(b); V.push_back(c); 
	make_heap(V.begin(), V.end(), CMP);
	a->cutLength = 6;
	make_heap(V.begin(), V.end(), CMP);
	while (!V.empty())
	{
		cout << V[0]->cutLength << "\n";
		pop_heap(V.begin(),V.end(),CMP);
		V.pop_back();
	}*/

	//optimal cut
	/*for(int i=0;i<size;i++)
		recursiveOptimalCut(beams, i, size, direct, intersectionLength);*/
	//stack<int> stk; stack<int> stk2;
	//for (int cutId = 0; cutId < size; cutId++)
	//{
	//	//recursiveOptimalCut(beams, i, size, direct, intersectionLength);
	//	stk.push(cutId);
	//	while (!stk.empty())
	//	{
	//		int i = stk.top();
	//		int j = 0;
	//		if (stk2.size() == stk.size())
	//		{
	//			j = stk2.top();
	//			j++;
	//			stk2.pop();
	//		}
	//		for (; j < size; j++)
	//		{
	//			if (j == i)continue;
	//			BeamPlugin b1 = *beams[i];
	//			BeamPlugin b2 = *beams[j];
	//			double r1 = b1.radius, r2 = b2.radius;

	//			double cosVal = Dot(direct[i], direct[j]) / (Norm(direct[i])*Norm(direct[j]));
	//			double angle = acos(cosVal);
	//			/*if (angle >= PI / 2)
	//				continue;*/

	//			double length1 = sqrt(pow(r1, 2) + pow(intersectionLength[i], 2));
	//			double itoj = angle - atan(r1 / intersectionLength[i]);
	//			double cut1 = length1 * cos(itoj);
	//			if (cut1 > intersectionLength[j])
	//			{
	//				intersectionLength[j] = cut1;
	//				stk.push(j);
	//				stk2.push(j);
	//				break;
	//				//recursiveOptimalCut(beams, j, size, direct, intersectionLength);
	//			}

	//			double length2 = sqrt(pow(r2, 2) + pow(intersectionLength[j], 2));
	//			double jtoi = angle - atan(r2 / intersectionLength[j]);
	//			double cut2 = length2 * cos(jtoi);
	//			if (cut2 > intersectionLength[i])
	//			{
	//				intersectionLength[i] = cut2;
	//				stk.push(i);
	//				stk2.push(j);
	//				break;
	//				//recursiveOptimalCut(beams, i, size, direct, intersectionLength);
	//			}
	//		}
	//		if (j == size)
	//		{
	//			stk.pop();
	//		}
	//	}
	//}
	
	//for (int i = 0; i < size; i++)
	//{
	//	for (int j = 0; j < size; j++)
	//	{
	//		if (j == i)continue;
	//		BeamPlugin b1 = *beams[i];
	//		BeamPlugin b2 = *beams[j];
	//		double r1 = b1.radius, r2 = b2.radius;

	//		double cosVal = Dot(direct[i],direct[j]) / (Norm(direct[i])*Norm(direct[j]));
	//		double angle = acos(cosVal);
	//		/*if (angle >= PI / 2)
	//			continue;*/

	//		double length1 = sqrt(pow(r1, 2) + pow(intersectionLength[i], 2));
	//		double itoj = angle - atan(r1 / intersectionLength[i]);
	//		double cut1 = length1 * cos(itoj);
	//		if (cut1 > intersectionLength[j])
	//			intersectionLength[j] = cut1;

	//		double length2 = sqrt(pow(r2, 2) + pow(intersectionLength[j], 2));
	//		double jtoi = angle - atan(r2 / intersectionLength[j]);
	//		double cut2 = length2 * cos(jtoi);
	//		if (cut2 > intersectionLength[i])
	//			intersectionLength[i] = cut2;
	//	}
	//}
	//double maxLength = intersectionLength[0];
	for (int i = 0; i < size; i++)
	{
		intersectionLength[i] += threshold;
		/*if (intersectionLength[i] > maxLength)
			maxLength = intersectionLength[i];*/
	}
	//maxLength += threshold;
	//找到推进距离之后更新axis的起点和length
	for (int i = 0; i < size; i++)
	{
		BeamPlugin &b = *beams[i];
		float3 start,end;
		int flag = 0;
		if (sqrt(pow(startP.x - b.axis.p[0].x, 2) + pow(startP.y - b.axis.p[0].y, 2) + pow(startP.z - b.axis.p[0].z, 2)) < 1e-6)
		{
			start = b.axis.p[0];
			end = b.axis.p[1];
			//std::cout << b.axis.p1[0]<<" "<< b.axis.p1[1]<<" "<< b.axis.p1[2]<<" "<<b.axis.p2[0]<<" "<< b.axis.p2[1] << " "<<b.axis.p2[2];
		}
		else
		{
			start = b.axis.p[1];
			end = b.axis.p[0];
			flag = 1;
			//std::cout << b.axis.p2[0] << " " << b.axis.p2[1] << " " << b.axis.p2[2] << " " << b.axis.p1[0] << " " << b.axis.p1[1] << " " << b.axis.p1[2];
		}
		float3 direction = make_float3(end.x-start.x,end.y-start.y,end.z-start.z);
		direction = make_norm(direction);
		start.x += direction.x* intersectionLength[i];
		start.y += direction.y* intersectionLength[i];
		start.z += direction.z* intersectionLength[i];
		if (!flag)
		{
			b.axis.p[0] = start;
			//std::cout<< " "<< maxLength<<std::endl;
		}
		else
		{
			b.axis.p[1] = start;
			//std::cout<<" " << maxLength << std::endl;
		}
		b.length -= intersectionLength[i];
	}

}
vector<Face> faces;
vector<Face> allFaces;
void figFaces(Triangle* AllMesh, int triNum)
{
	for (int i = 0; i < triNum; i++)
	{
		Triangle* triTmp = AllMesh + i;
		if (i >= 42 && i <= 51)
		{
			for (int k = 0; k < 3; k++)
			{
				float3 diskNorm = triTmp->diskNorm[k];
				curandState devStates;
				curand_init((unsigned long long)clock(), i, 0, &devStates);// initialize the state
				float RANDOM = curand_uniform(&devStates);// uniform distribution
				float theta = RANDOM;
				float3 O = triTmp->o[k];
				float** m;
				m = getRotationMatrix(diskNorm, theta, O);
				float3 rotateV = make_float3(triTmp->p[k].x - O.x, triTmp->p[k].y - O.y, triTmp->p[k].z - O.z);
				float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
				RV = Multiply(m, RV);
				triTmp->p[k] = make_float3(RV.x, RV.y, RV.z);
			}
			
		}
		if (triTmp->flag == 0)
		{
			Face ftmp;
			ftmp.p1 = triTmp->p[0];
			ftmp.p2 = triTmp->p[1];
			ftmp.p3 = triTmp->p[2];
			ftmp.normal = triTmp->normal;
			faces.push_back(ftmp);
		}
	}
}
void generateFaces(Triangle* AllMesh,int triNum,short type)
{
	for (int i = 0; i < triNum; i++)
	{
		Triangle* triTmp = AllMesh + i;
		Face ftmp;
		ftmp.p1 = triTmp->p[0];
		ftmp.p2 = triTmp->p[1];
		ftmp.p3 = triTmp->p[2];
		ftmp.normal = triTmp->normal;
		ftmp.inverted = triTmp->inverted[0];
		ftmp.type = type;
		if (triTmp->flag == 0)
		{
			
			faces.push_back(ftmp);
		}
		//allFaces.push_back(ftmp);
		
	}
	/*for (int i = 0; i < quadNum; i++)
	{
		QuadFace* quadTmp = AllQuad + i;
		Face ftmp;
		if (quadTmp->t[0].flag == 0)
		{
			ftmp.p1 = quadTmp->t[0].p[0];
			ftmp.p2 = quadTmp->t[0].p[1];
			ftmp.p3 = quadTmp->t[0].p[2];
			ftmp.normal = quadTmp->t[0].normal;
			faces.push_back(ftmp);
		}
		if (quadTmp->t[1].flag == 0)
		{
			ftmp.p1 = quadTmp->t[1].p[0];
			ftmp.p2 = quadTmp->t[1].p[1];
			ftmp.p3 = quadTmp->t[1].p[2];
			ftmp.normal = quadTmp->t[1].normal;
			faces.push_back(ftmp);
		}
	}*/
}
void tbbGWO(int idx,Triangle* host_AllMesh,QuadFace* host_AllQuad,int* beamNumVec, BeamPlugin** G,short* host_AllFlag)
{
	int meshSt = 0;
	int quadSt = 0;
	//int arcSt = 0;
	int flagSt = 0;
	for (int j = 1; j < idx; j++)
	{
		int meshNumincre = 8 + (beamNumVec[j] - 6) * 2;
		if (beamNumVec[j] < 4)meshNumincre = 0;
		meshSt += meshNumincre;
		int quadNumincre = 6 + 3 * ((meshNumincre - 4) / 2);
		quadSt += quadNumincre;
		//arcSt += 2 * quadNumincre;
		if (beamNumVec[j] >= 4)flagSt += beamNumVec[j];
	}
	//if (G[i].size() > 1)
	{
		BeamPlugin** beams = G + flagSt;
		int triNum = 8 + (beamNumVec[idx] - 6) * 2;
		int quadNum = 6 + 3 * ((triNum - 4) / 2);
		Triangle* triangles = host_AllMesh + meshSt;
		QuadFace* quads = host_AllQuad + quadSt;
		short* flag = host_AllFlag + flagSt;
		//getTopology(G[i], &nodeVector[i], AllTriangle + meshSt, AllQuad + quadSt, AllFlag + flagSt, AllLength + flagSt);
		if (triangles[0].inverted[0] == 3);
			//GWO2(beams, beamNumVec[idx], flag, triNum * 3 + beamNumVec[idx], triangles, quads, triNum, quadNum);
			//GreyWolfOptimizer(triNum * 3, triangles, quads, triNum, quadNum);
	}
}
int main(int argc, char *argv[])
{
	tbb::parallel_invoke([]() {cout << "Hello " << endl; },
		[]() {cout << "TBB!" << endl; });

	
	float3* point = new float3[10281];
	vector<BeamPlugin*> G[10281];//存储拓扑关系
	int* beamNumVec;
	//int* meshNumVec;

	FILE *fe, *fp;
	fe = fopen("D:\\shapeDesignLattice\\Data\\testSmallAngle.edge", "r");
	fp = fopen("D:\\shapeDesignLattice\\Data\\testSmallAngle.node", "r");

	string path = "D:\\shapeDesignLattice\\Results\\CylinderBeam.stl";
	// string path = "D:\\MutiResolution\\Cylinder.stl";
	// convertToSTL(path,b1.f);
#ifdef READFILE
	if (fe != NULL && fp != NULL)
	{

		int plines, dim, patri, pmk;
		fscanf(fp, "%d %d %d %d", &plines, &dim, &patri, &pmk);
		//Junction* nodeVector = new Junction[plines + 1];// (Junction*)malloc((plines + 1) * sizeof(Junction));
		float3* host_positions = new float3[plines + 1];
		
		
		beamNumVec = new int[plines + 1];
		//meshNumVec = new int[plines + 1];
		int totalMeshNum = 0;
		int totalQuadNum = 0;
		int totalFlagNum = 0;
		//nodeVector.resize(plines+1);
		for (int i = 1; i <= plines; i++)
		{
			int pline;
			double vx, vy, vz;
			fscanf(fp, "%d %lf %lf %lf", &pline, &vx, &vy, &vz);
			//printf("%d %lf %lf %lf\n",pline,vx,vy,vz);
			point[i].x = vx*100;
			point[i].y = vy*100;
			point[i].z = vz*100;

			beamNumVec[i] = 0;
			//meshNumVec[i] = -4;
		}

		int elines, emark;
		fscanf(fe, "%d %d", &elines, &emark);
		//vector<BeamPlugin*> beamVector;
		//beamVector.resize(elines+1);
		BeamPlugin** beamVector = (BeamPlugin**)malloc((elines+1) * sizeof(BeamPlugin*));
		int updatedElines = elines; int eidx = 1;
		std::default_random_engine e;
		std::uniform_real_distribution<float> u(0.08, 0.16);
		e.seed(time(0)); //vector<int>checkG[7500];
		for (int i = 1; i <= elines; i++)
		{
			int eline, id1, id2, emk;
			
			float r=0.12;
			//float r = u(e); //printf("%f\n", r);
			fscanf(fe, "%d %d %d %d %f", &eline, &id1, &id2, &emk,&r); //id1++; id2++;

			/*if ((beamNumVec[id1] > 8 || beamNumVec[id2] > 8)&&emk==-1)
			{
				updatedElines--;
				continue;
			}*/
			/*if ((beamNumVec[id1] > 30 || beamNumVec[id2] > 30))
			{
				updatedElines--;
				continue;
			}*/

			Edge etmp;
			etmp.p[0] = point[id1];
			etmp.p[1] = point[id2];
			printf("%f\n", Norm(Subtraction(etmp.p[1], etmp.p[0])));
			float edgeLength = Norm(Subtraction(etmp.p[1], etmp.p[0]));
			/*if (edgeLength < 200)r = 0.08;
			if (edgeLength < 120)r = 0.05;
			if (edgeLength < 80)r = 0.03;*/
			//if (edgeLength < 1000)r = 0.3;
			/*if (edgeLength < 700)r = 0.3;
			if (edgeLength < 400)r = 0.15;*/
			/*if (edgeLength < 150)r = 0.02;
			if (edgeLength < 100)r = 0.01;*/
			/*if (edgeLength < 35)r = 0.01;
			if (edgeLength < 15)r = 0.005;*/
			if (abs(edgeLength - 99.684751) < 1e-4 || abs(edgeLength - 96.321869) < 1e-4 || abs(edgeLength - 126.296877) < 1e-4 || abs(edgeLength - 149.394969) < 1e-4 || abs(edgeLength - 76.754511) < 1e-4 || abs(edgeLength - 74.429033) < 1e-4 || abs(edgeLength - 103.309783) < 1e-4 || abs(edgeLength - 85.913043) < 1e-4 || abs(edgeLength - 134.599552) < 1e-4 || abs(edgeLength - 126.553071) < 1e-4 || abs(edgeLength - 35.635094) < 1e-4 || abs(edgeLength - 41.967350) < 1e-4 || abs(edgeLength - 54.251526) < 1e-4 || abs(edgeLength - 53.324937) < 1e-4 || abs(edgeLength - 53.470341) < 1e-4 || abs(edgeLength - 44.077942) < 1e-4 || abs(edgeLength - 52.751317) < 1e-4 || abs(edgeLength - 33.709738) < 1e-4 || abs(edgeLength - 19.172227) < 1e-4 || abs(edgeLength - 27.165291) < 1e-4 ||abs(edgeLength- 51.502611)<1e-4||abs(edgeLength- 31.694096)<1e-5||abs(edgeLength- 66.018105)<1e-4||abs(edgeLength- 42.081882)<1e-5||abs(edgeLength- 56.883881)<1e-5||abs(edgeLength- 52.297329)<1e-5||abs(edgeLength - 51.505483) < 1e-4||abs(edgeLength- 21.819664)<1e-4||abs(edgeLength - 41.763945) < 1e-4 || abs(edgeLength - 27.232736) < 1e-4||abs(edgeLength- 44.078725)<1e-4||abs(edgeLength- 41.653680)<1e-5 || abs(edgeLength - 26.689088) < 1e-5 || abs(edgeLength - 39.342680) < 1e-5 || abs(edgeLength - 22.855256) < 1e-5 || abs(edgeLength - 41.407813) < 1e-5 || abs(edgeLength - 29.224399) < 1e-4 || abs(edgeLength - 23.731346) < 1e-4 || abs(edgeLength - 49.242787) < 1e-4)
			{
				updatedElines--;
				continue;
			}
			/*if (find(checkG[id1].begin(), checkG[id1].end(), id2) != checkG[id1].end() || find(checkG[id2].begin(), checkG[id2].end(), id1) != checkG[id2].end())
			{
				printf("duplicated edge!\n");
			}
			checkG[id1].push_back(id2); checkG[id2].push_back(id1);
			if (id1 > plines || id2 > plines || id1 <= 0 || id2 <= 0)printf("wrong id!\n");*/
			//if (edgeLength > 50 && edgeLength < 110)r = 4;
			////if (edgeLength <= 200&&edgeLength>100)r = 6;
			//if (edgeLength <= 110 && edgeLength > 70)r = 0.02;
			//else if (edgeLength <= 70 && edgeLength > 35)r = 0.005;
			////else if (edgeLength <= 45 && edgeLength > 18)r = 0.0018;
			//else if (edgeLength <= 35)r = 0.005;
			//if (edgeLength <= 40 && edgeLength > 39)
			//{
			//	updatedElines--;
			//	continue;
			//}
			//if (edgeLength <= 110.2 && edgeLength > 110)
			//{
			//	updatedElines--;
			//	continue;
			//}
			//if (edgeLength > 600)r = 20;
			//else if (edgeLength <= 600 && edgeLength > 400)r = 13;
			//else if (edgeLength<=400&&edgeLength > 200)r = 7;
			//else if (edgeLength <= 200 && edgeLength > 100)r = 3.5;
			////else if (edgeLength <= 100 && edgeLength > 100)r = 2.5;
			////else if (edgeLength > 65 && edgeLength <= 100)r = 1.6;
			//else if (edgeLength <= 100 && edgeLength > 50)r = 2;
			//else if (edgeLength <= 50 && edgeLength > 20)r = 1;
			//else
			//	r =1;
			//r = Norm(Subtraction(etmp.p[1], etmp.p[0])) / 30;
			BeamPlugin *b = new BeamPlugin();
			b->BeamConstruct(etmp,r*100);
			//b.BeamTransform();
			//b->convergeF[0] = b->convergeF[1] = 0;
			b->beamId = eidx;
			b->minAngle[0] =b->minAngle[1]= INF;
			b->arcNum[0] = 0; b->arcNum[1] = 0;
			b->convexLength[0] = b->convexLength[1] = 0;
			beamVector[eidx] = b;
			eidx++;

			G[id1].push_back(b);
			//Junction *j1 = new Junction(); j1->nodeId = id1; j1->position = point[id1];
			//nodeVector[id1] = *j1;
			host_positions[id1] = point[id1];// j1->position;
			
			G[id2].push_back(b);
			//Junction *j2 = new Junction(); j2->nodeId = id2; j2->position = point[id2];
			//nodeVector[id2] = *j2;
			host_positions[id2] = point[id2];// j2->position;

			beamNumVec[id1]++;
			beamNumVec[id2]++;
			/*meshNumVec[id1] += 2;
			meshNumVec[id2] += 2;*/
		}
		elines = updatedElines;
		
		for (int i = 1; i < plines + 1; i++)
		{
			//if (G[i].size() > 1)
			{
				if (beamNumVec[i] < 4)printf("wrong!\n");
				int meshNumincre= 8 + (beamNumVec[i] - 6) * 2;
				if (beamNumVec[i] < 4)meshNumincre = 0;
				if (meshNumincre > 100)printf("too many!\n");
				totalMeshNum += meshNumincre;
				int quadNumincre= 6 + 3 * ((meshNumincre - 4) / 2);
				if (quadNumincre > 150)printf("too many quads!\n");
				totalQuadNum += quadNumincre;
				totalFlagNum += beamNumVec[i];
			}
			//statics[beamNumVec[i]]++;
		}
		
#ifdef PARALLEL_NODE
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1400 * 1024 * 1024);
		size_t HeapValue;
		cudaDeviceGetLimit(&HeapValue, cudaLimitMallocHeapSize);
		cout << "HeapValue: " << HeapValue << endl;

		cudaDeviceSetLimit(cudaLimitStackSize, 8192);
		size_t StackValue;
		cudaDeviceGetLimit(&StackValue, cudaLimitStackSize);
		cout << "StackValue: " << StackValue << endl;

		int length = elines * 2;
		BeamPlugin** hostG = (BeamPlugin**)malloc(length*sizeof(BeamPlugin*));
		int cnt = 0;
		for (int i = 1; i <= plines; i++)
		{
			int n = G[i].size();
			for (int j = 0; j < n; j++)
			{
				hostG[cnt] = G[i][j];
				cnt++;
			}
			
		}
		map<int, int>mp;
		BeamPlugin** cudaG;
		BeamPlugin** cudaTmp = new BeamPlugin*[length];

		BeamPlugin** cuda_beamVector;
		CUDA_SAFE_CALL(cudaMallocManaged((void **)&cuda_beamVector, (elines + 1) * sizeof(BeamPlugin*)));
		/*for (int i = 1; i <= elines; i++)
		{
			CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_beamVector[i], sizeof(BeamPlugin)));
			CUDA_SAFE_CALL(cudaMemcpy((void*)cuda_beamVector[i], (void*)beamVector[i], sizeof(BeamPlugin), cudaMemcpyHostToDevice));
		}*/
		
		bool* visited = new bool[elines+1];
		fill(visited, visited + elines+1, false);
		CUDA_SAFE_CALL(cudaMallocManaged((void **)&cudaG, length * sizeof(BeamPlugin*)));
		for (int i = 0; i < length; i++)
		{
			if (!visited[hostG[i]->beamId])
			{
				CUDA_SAFE_CALL(cudaMalloc((void **)&cudaG[i], sizeof(BeamPlugin)));
				CUDA_SAFE_CALL(cudaMemcpy((void*)cudaG[i], (void*)hostG[i], sizeof(BeamPlugin), cudaMemcpyHostToDevice));

				visited[hostG[i]->beamId] = true;
				mp[hostG[i]->beamId] = i;
				cuda_beamVector[hostG[i]->beamId] = cudaG[i];
			}
			else
			{
				int idx = mp[hostG[i]->beamId];
				cudaG[i] = cudaG[idx];

			}
			cudaTmp[i] = cudaG[i];//用于DeviceToHost
		}
		delete visited;

		int* cuda_beamNumVec;
		CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_beamNumVec, (plines + 1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy((void*)cuda_beamNumVec, (void*)beamNumVec, (plines + 1) * sizeof(int), cudaMemcpyHostToDevice));

		/*int* cuda_meshNumVec;
		CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_meshNumVec, (plines + 1) * sizeof(int)));
		CUDA_SAFE_CALL(cudaMemcpy((void*)cuda_meshNumVec, (void*)meshNumVec, (plines + 1) * sizeof(int), cudaMemcpyHostToDevice));*/
		//int* host_triNums = new int[plines + 1];

		float3* cuda_positions;
		CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_positions, (plines + 1) * sizeof(float3)));
		CUDA_SAFE_CALL(cudaMemcpy((void*)cuda_positions, (void*)host_positions, (plines + 1) * sizeof(float3), cudaMemcpyHostToDevice));

		/*Face** cuda_f;
		CUDA_SAFE_CALL(cudaMallocManaged((void **)&cuda_f, (plines + 1) * sizeof(Face*)));
		Face** host_f = new Face*[plines + 1];*/
		Triangle* AllMesh;
		CUDA_SAFE_CALL(cudaMallocManaged((void **)&AllMesh, (totalMeshNum) * sizeof(Triangle)));

		Triangle* AllTriList;
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllTriList, totalMeshNum * sizeof(Triangle)));

		QuadFace* AllQuad;
		CUDA_SAFE_CALL(cudaMallocManaged((void **)&AllQuad, (totalQuadNum) * sizeof(QuadFace)));

		QuadFace* AllQuadList;
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllQuadList, totalQuadNum * sizeof(QuadFace)));

		Point* dots;
		CUDA_SAFE_CALL(cudaMalloc((void **)&dots, length * sizeof(Point)));

		/*float* Allc, *AllD, *AllA, *AllX1,*AllX2,*AllX3;
		CUDA_SAFE_CALL(cudaMalloc((void **)&Allc, totalMeshNum *3* sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllD, totalMeshNum * 3 * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllA, totalMeshNum * 3 * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllX1, totalMeshNum * 3 * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllX2, totalMeshNum * 3 * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllX3, totalMeshNum * 3 * sizeof(float)));*/

		ArcType* AllArc;
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllArc, 2*totalQuadNum * sizeof(ArcType)));

		short* AllFlag;
		CUDA_SAFE_CALL(cudaMallocManaged((void **)&AllFlag, totalFlagNum * sizeof(short)));

		/*float* AllLength;
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllLength, totalFlagNum * sizeof(float)));*/


		NodeStatistic* Statis;
		CUDA_SAFE_CALL(cudaMalloc((void **)&Statis,(plines+1)* sizeof(NodeStatistic)));
		
		int batchSize = 7449; int batchNum = ceil((float)plines / batchSize); printf("%d\n", batchNum);
		int batchIdx = 0;
		tbb::parallel_pipeline(64,
			tbb::make_filter<void, int>(
				tbb::filter_mode::serial_in_order,
				[&](tbb::flow_control& fc)->int {
					if (batchIdx == batchNum) {
						fc.stop();
						return -1;
					}
					return batchIdx++;
				}
				)&
			tbb::make_filter<int,Data>(
				tbb::filter_mode::serial_in_order,
				[&](int batchIdx)->Data {
					call_NodeComputing(plines, cudaG, cuda_beamNumVec, cuda_positions,
						AllMesh, AllQuad, dots, AllFlag, Statis,batchIdx,batchSize,batchNum);
					int offset = batchIdx * batchSize;
					int meshNumincre = 0,quadNumincre=0,flagNumincre=0;
					for (int j = 1; j <= offset; j++)
					{
						int tmpIncre = 8 + (beamNumVec[j] - 6) * 2;
						meshNumincre += tmpIncre;
						quadNumincre+= 6 + 3 * ((tmpIncre - 4) / 2);
						flagNumincre += beamNumVec[j];
					}
					Data dat;
					int offset2;
					if (batchIdx == batchNum - 1)
						offset2 = plines;
					else
						offset2 = (batchIdx + 1) * batchSize;
					
					int triNum = 0, quadNum = 0,flagNum=0;
					for (int j = offset+1; j <= offset2; j++)
					{
						int tmpIncre = 8 + (beamNumVec[j] - 6) * 2;
						triNum += tmpIncre;
						quadNum+= 6 + 3 * ((tmpIncre - 4) / 2);
						flagNum += beamNumVec[j];
					}
					dat.batchIdx = batchIdx;
					dat.batchTriNum = triNum;
					dat.batchQuadNum = quadNum;
					dat.batchBeamNum = flagNum;
					dat.AllMesh = (Triangle*)malloc(triNum*sizeof(Triangle));
					dat.AllQuad = (QuadFace*)malloc(quadNum*sizeof(QuadFace));
					dat.BatchFlag = (short*)malloc(flagNum * sizeof(short));
					dat.BatchBeams = (BeamPlugin**)malloc(flagNum * sizeof(BeamPlugin*));
					CUDA_SAFE_CALL(cudaMemcpy((void*)dat.AllMesh, (void*)(AllMesh+meshNumincre), triNum * sizeof(Triangle), cudaMemcpyDeviceToHost));
					CUDA_SAFE_CALL(cudaMemcpy((void*)dat.AllQuad, (void*)(AllQuad+quadNumincre), quadNum * sizeof(QuadFace), cudaMemcpyDeviceToHost));
					CUDA_SAFE_CALL(cudaMemcpy((void*)dat.BatchFlag, (void*)(AllFlag + flagNumincre), flagNum * sizeof(short), cudaMemcpyDeviceToHost));
					int cnt = 0; bool* visited = new bool[elines + 1]; fill(visited, visited + elines + 1, false);
					map<int, int>mp2;
					for (int j = offset + 1; j <= offset2; j++)
					{
						int n = G[j].size();
						for (int k = 0; k < n; k++)
						{
							if (!visited[hostG[flagNumincre+cnt]->beamId])
							{
								dat.BatchBeams[cnt] = (BeamPlugin*)malloc(sizeof(BeamPlugin));
								CUDA_SAFE_CALL(cudaMemcpy((void*)(dat.BatchBeams[cnt]), (void*)cudaTmp[flagNumincre + cnt], sizeof(BeamPlugin), cudaMemcpyDeviceToHost));
							
								visited[hostG[flagNumincre + cnt]->beamId] = true;
								mp2[hostG[flagNumincre + cnt]->beamId] = cnt;
							
							}
							else
							{
								int idx = mp2[hostG[flagNumincre + cnt]->beamId];
								dat.BatchBeams[cnt] = dat.BatchBeams[idx];
							}
							
							cnt++;
						}
						

					}
					delete visited;
					return dat;
				}
				)&
					//流水线并行，内部节点还可以并行
			tbb::make_filter<Data, Data>(
				tbb::filter_mode::parallel,
				[&](Data dat)->Data {
					int offset = dat.batchIdx * batchSize;
					int offset2;
					if (dat.batchIdx == batchNum - 1)
						offset2 = plines;
					else
						offset2 = (dat.batchIdx + 1) * batchSize;

					tbb::parallel_for(offset+1, offset2+1, [&](int i) {
						int meshSt = 0;
						int quadSt = 0;
						//int arcSt = 0;
						int flagSt = 0;
						for (int j = offset+1; j < i; j++)
						{
							int meshNumincre = 8 + (beamNumVec[j] - 6) * 2;
							if (beamNumVec[j] < 4)meshNumincre = 0;
							meshSt += meshNumincre;
							int quadNumincre = 6 + 3 * ((meshNumincre - 4) / 2);
							quadSt += quadNumincre;
							//arcSt += 2 * quadNumincre;
							if (beamNumVec[j] >= 4)flagSt += beamNumVec[j];
						}
						if (G[i].size() > 1)
						{
							int triNum = 8 + (beamNumVec[i] - 6) * 2;
							int quadNum = 6 + 3 * ((triNum - 4) / 2);
							Triangle* triangles = dat.AllMesh + meshSt;
							QuadFace* quads = dat.AllQuad + quadSt;
							short* flag = dat.BatchFlag + flagSt;
							BeamPlugin** beams = dat.BatchBeams + flagSt;
							//getTopology(G[i], &nodeVector[i], AllTriangle + meshSt, AllQuad + quadSt, AllFlag + flagSt, AllLength + flagSt);
							if (triangles[0].inverted[0] == 3)
								GWO2(beams, beamNumVec[i], flag, triNum * 3, triangles, quads, triNum, quadNum,dat.batchIdx,i);
								//GreyWolfOptimizer(triNum * 3, triangles, quads, triNum, quadNum,dat.batchIdx);
						}
						});
					free(dat.BatchFlag);bool* visited = new bool[dat.batchBeamNum]; fill(visited, visited + dat.batchBeamNum, false);
					for (int j = 0; j < dat.batchBeamNum; j++)
					{
						if (!visited[j])
						{
							
							for (int k = j + 1; k < dat.batchBeamNum; k++)
							{
								if (!visited[k] && dat.BatchBeams[j]->beamId == dat.BatchBeams[k]->beamId)
								{
									free(dat.BatchBeams[j]);
									dat.BatchBeams[j] = NULL;
									dat.BatchBeams[k] = NULL;
									visited[j] = true;
									visited[k] = true;
									break;
								}
							}
							
						}
					}
					free(dat.BatchBeams); delete visited;
					return dat;
				}
				)&

			tbb::make_filter<Data, void>(
				tbb::filter_mode::serial_in_order,
				[&](Data dat)->void {
					int offset = dat.batchIdx * batchSize;
					int meshNumincre = 0, quadNumincre = 0;
					for (int j = 1; j <= offset; j++)
					{
						int tmpIncre = 8 + (beamNumVec[j] - 6) * 2;
						meshNumincre += tmpIncre;
						quadNumincre += 6 + 3 * ((tmpIncre - 4) / 2);
					}
					CUDA_SAFE_CALL(cudaMemcpy((void*)(AllMesh + meshNumincre), (void*)(dat.AllMesh), dat.batchTriNum * sizeof(Triangle), cudaMemcpyHostToDevice));
					CUDA_SAFE_CALL(cudaMemcpy((void*)(AllQuad + quadNumincre), (void*)(dat.AllQuad), dat.batchQuadNum * sizeof(QuadFace), cudaMemcpyHostToDevice));
					free(dat.AllMesh); free(dat.AllQuad); 
				}
				)
					);
		//call_latticePreprocessing(plines, cudaG, cuda_beamNumVec, cuda_positions,elines, cuda_beamVector, 0.18 * 23);
		/*call_NodeComputing(plines,cudaG,cuda_beamNumVec, cuda_positions, 
			AllMesh, AllQuad, dots, AllFlag,Statis);*/

		NodeStatistic* host_Statis = (NodeStatistic*)malloc((plines+1)*sizeof(NodeStatistic));
		CUDA_SAFE_CALL(cudaMemcpy((void*)host_Statis, (void*)Statis, (plines + 1) * sizeof(NodeStatistic), cudaMemcpyDeviceToHost));
		cudaFree(Statis);
		int sta1[50] = { 0 }; int sta2[50] = { 0 }; int angleRange1[20] = { 0 }; int angleRange2[20] = { 0 };
		int ratio1[40] = { 0 }; int ratio2[40] = { 0 };
		float convergedNum = 0;
		for (int i = 1; i <= plines; i++)
		{
			if (host_Statis[i].converge)
			{
				sta1[host_Statis[i].degree]++;
				int idx = host_Statis[i].face_angle_variance / (PI / 40);
				angleRange1[idx]++;
				float ratio = host_Statis[i].circleDis/host_Statis[i].disToSphere;
				//printf("%f\n", ratio);
				if (host_Statis[i].disToSphere != 0)
				{
					int r_id = ratio / 0.2;
					ratio1[r_id]++;
				}
				
				convergedNum++;
			}
			else
			{
				sta2[host_Statis[i].degree]++;
				int idx = host_Statis[i].face_angle_variance / (PI / 40);
				angleRange2[idx]++;
				float ratio = host_Statis[i].circleDis/host_Statis[i].disToSphere;
				//printf("%f\n", ratio);
				if (host_Statis[i].disToSphere != 0)
				{
					int r_id = ratio / 0.2;
					ratio2[r_id]++;
				}
				
			}
		}
		free(host_Statis);
		ofstream oFile;
		oFile.open("node_degree.csv", ios::out | ios::trunc);
		oFile << "converged node degree" << "," << "number" << endl;
		for (int i = 0; i < 50; i++)
		{
			if (sta1[i] > 0)
			{
				oFile << i <<","<< sta1[i] << endl;
			}
		}
		oFile << "non-converged node degree" << "," << "number" << endl;
		for (int i = 0; i < 50; i++)
		{
			if (sta2[i] > 0)
			{
				oFile << i <<","<< sta2[i] << endl;
			}
		}
		oFile.close();


		ofstream oFile2;
		oFile2.open("ratio.csv", ios::out | ios::trunc);
		oFile2 << "converged ratio" << "," << "number" << endl;
		//oFile2 << setiosflags(ios::fixed) << setprecision(2);
		for (int i = 0; i < 40; i++)
		{
			if (ratio1[i] > 0)
			{
				oFile2 << i * 0.2 << "-" << (i + 1)*0.2 << "," << ratio1[i] << endl;
			}
		}
		oFile2<< "non-converged ratio" << "," << "number" << endl;
		for (int i = 0; i < 40; i++)
		{
			if (ratio2[i] > 0)
			{
				oFile2<< i * 0.2 << "-" << (i + 1)*0.2 << "," << ratio2[i] << endl;
			}
		}
		oFile2 << convergedNum / plines << endl;
		oFile2.close();
		//CUDA_SAFE_CALL(cudaMemcpy((void*)hostG, (void*)cudaG, length * sizeof(BeamPlugin), cudaMemcpyDeviceToHost));
		for (int i = 0; i < length; i++)
		{
			CUDA_SAFE_CALL(cudaMemcpy((void*)hostG[i], (void*)cudaTmp[i], sizeof(BeamPlugin), cudaMemcpyDeviceToHost));
		}
		Triangle* host_AllMesh = new Triangle[totalMeshNum];
		CUDA_SAFE_CALL(cudaMemcpy((void*)host_AllMesh, (void*)AllMesh, totalMeshNum*sizeof(Triangle), cudaMemcpyDeviceToHost));
		QuadFace* host_AllQuad = new QuadFace[totalQuadNum];
		CUDA_SAFE_CALL(cudaMemcpy((void*)host_AllQuad, (void*)AllQuad, totalQuadNum * sizeof(QuadFace), cudaMemcpyDeviceToHost));
		short* host_AllFlag = new short[totalFlagNum];
		CUDA_SAFE_CALL(cudaMemcpy((void*)host_AllFlag, (void*)AllFlag, totalFlagNum * sizeof(short), cudaMemcpyDeviceToHost));

		ArcType* host_AllArc = (ArcType*)malloc(2 * totalQuadNum * sizeof(ArcType));
		
		//CUDA_SAFE_CALL(cudaMemcpy((void*)AllMesh, (void*)host_AllMesh, totalMeshNum * sizeof(Triangle), cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpy((void*)AllQuad, (void*)host_AllQuad, totalQuadNum * sizeof(QuadFace), cudaMemcpyHostToDevice));
		//figFaces(host_AllMesh, totalMeshNum);
		generateFaces(host_AllMesh, totalMeshNum,1);
		string txtPath= "D:\\shapeDesignLattice\\Results\\CylinderBeam.txt";
		processTQB(txtPath, host_AllMesh, totalMeshNum, host_AllQuad, totalQuadNum, host_AllArc, host_AllFlag, G, beamNumVec, plines, beamVector, elines);
		delete host_AllMesh; delete host_AllQuad; delete host_AllFlag; free(host_AllArc);
		printf("chord error: \n");
		float ce;
		scanf("%f", &ce);
		int* totalSampleNum;
		int* totalArcSampleNum;
		CUDA_SAFE_CALL(cudaMalloc((void **)&totalSampleNum, sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(totalSampleNum,0, sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&totalArcSampleNum, sizeof(int)));
		CUDA_SAFE_CALL(cudaMemset(totalArcSampleNum, 0, sizeof(int)));
		call_locateArcs_setSegNum(plines, cudaG, cuda_beamNumVec, AllMesh, AllQuad, AllArc, AllFlag, totalSampleNum, totalArcSampleNum, ce);
		cudaDeviceSynchronize();
		cudaFree(AllMesh); cudaFree(AllTriList); cudaFree(AllQuadList); cudaFree(dots); cudaFree(cuda_positions); cudaFree(cuda_beamNumVec);
		//cudaFree(AllA); cudaFree(Allc); cudaFree(AllD); cudaFree(AllX1); cudaFree(AllX2); cudaFree(AllX3);
		//将GPU中对G的b的arcNum的修改同步到CPU端，从而使beamVector指向的相同的b改变
		for (int i = 0; i < length; i++)
		{
			CUDA_SAFE_CALL(cudaMemcpy((void*)hostG[i], (void*)cudaTmp[i], sizeof(BeamPlugin), cudaMemcpyDeviceToHost));
		}
		int* host_totalSampleNum = new int[1];
		int* host_totalArcSampleNum = new int[1];
		CUDA_SAFE_CALL(cudaMemcpy((void*)host_totalSampleNum, (void*)totalSampleNum, sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy((void*)host_totalArcSampleNum, (void*)totalArcSampleNum, sizeof(int), cudaMemcpyDeviceToHost));
		printf("%d %d\n", (host_totalSampleNum[0] - 2 * elines), (host_totalArcSampleNum[0] - 2 * totalQuadNum));

		float3* Allsample;
		CUDA_SAFE_CALL(cudaMalloc((void **)&Allsample, host_totalSampleNum[0]*sizeof(float3)));
		ArcType* AllbeamArc;
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllbeamArc, 2 * elines*sizeof(ArcType)));
		Triangle* AllresultTri;
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllresultTri, (host_totalSampleNum[0] - 2 * elines) * sizeof(Triangle)));
		/*BeamPlugin** cuda_beamVector;
		CUDA_SAFE_CALL(cudaMallocManaged((void **)&cuda_beamVector, (elines+1) * sizeof(BeamPlugin*)));
		for (int i = 1; i <= elines; i++)
		{
			CUDA_SAFE_CALL(cudaMalloc((void **)&cuda_beamVector[i], sizeof(BeamPlugin)));
			CUDA_SAFE_CALL(cudaMemcpy((void*)cuda_beamVector[i], (void*)beamVector[i], sizeof(BeamPlugin), cudaMemcpyHostToDevice));
		}*/
		call_beamTriangulation(elines, cuda_beamVector, Allsample, AllbeamArc, AllresultTri);
		cudaDeviceSynchronize();
		Triangle* host_AllresultTri = new Triangle[(host_totalSampleNum[0] - 2 * elines)];
		CUDA_SAFE_CALL(cudaMemcpy((void*)host_AllresultTri, (void*)AllresultTri, (host_totalSampleNum[0] - 2 * elines) * sizeof(Triangle), cudaMemcpyDeviceToHost));
		generateFaces(host_AllresultTri, (host_totalSampleNum[0] - 2 * elines),3);
		cudaFree(AllresultTri); //cudaFree(cuda_beamVector);
		delete host_AllresultTri;

		Triangle* AllQuadresultTri;
		CUDA_SAFE_CALL(cudaMalloc((void **)&AllQuadresultTri, (host_totalArcSampleNum[0] - 2 * totalQuadNum) * sizeof(Triangle)));
		call_quadTriangulation(totalQuadNum, AllQuad, AllQuadresultTri);
		cudaDeviceSynchronize();
		
		Triangle* host_AllQuadresultTri = new Triangle[(host_totalArcSampleNum[0] - 2 * totalQuadNum)];
		CUDA_SAFE_CALL(cudaMemcpy((void*)host_AllQuadresultTri, (void*)AllQuadresultTri, (host_totalArcSampleNum[0] - 2 * totalQuadNum) * sizeof(Triangle), cudaMemcpyDeviceToHost));
		
		
		generateFaces(host_AllQuadresultTri, (host_totalArcSampleNum[0] - 2 * totalQuadNum),2);
		delete host_AllQuadresultTri;
		/*CUDA_SAFE_CALL(cudaMemcpy((void*)nodeVector, (void*)cuda_nodeVector, (plines + 1) * sizeof(Junction), cudaMemcpyDeviceToHost));*/
		//CUDA_SAFE_CALL(cudaMemcpy((void*)host_triNums, (void*)cuda_triNums, (plines + 1) * sizeof(int), cudaMemcpyDeviceToHost));
		//for (int i = 1; i <= plines; i++)
		//{
		//	host_f[i] = cuda_f[i];
		//	nodeVector[i].triNum = host_triNums[i]; cout << host_triNums[i] << endl;
		//	nodeVector[i].f = (Face*)malloc(host_triNums[i] * sizeof(Face));// new Face[host_triNums[i]];
		//	Face* test = new Face[host_triNums[i]];
		//	CUDA_SAFE_CALL(cudaMemcpy((void*)(test), (void*)host_f[i], host_triNums[i] *sizeof(Face), cudaMemcpyDeviceToHost));
		//}
		/*cnt = 0;
		for (int i = 1; i <= plines; i++)
		{
			int n = G[i].size();
			if (n > 1)
			{
				for (int j = 0; j < n; j++)
				{
					G[i][j] = hostG[cnt];
					cnt++;
				}
			}	
		}*/
		//for (int i = 1; i <= plines; i++)
		//{
		//	if (G[i].size() > 1)
		//	{
		//		//计算junction的faces
		//		//getTopology(G[i], &nodeVector[i]);
		//		//computeConvexhull(G[i], nodeVector[i]);

		//		//faces.insert(faces.end(), (nodeVector[i]).f, (nodeVector[i]).f+ (nodeVector[i]).triNum);
		//	}
		//}
#else

        
		Triangle* AllTriangle = (Triangle*)malloc(totalMeshNum * sizeof(Triangle));
		QuadFace* AllQuad = (QuadFace*)malloc(totalQuadNum * sizeof(QuadFace));
		ArcType* AllArc = (ArcType*)malloc(2 * totalQuadNum * sizeof(ArcType));
		short* AllFlag = (short*)malloc(totalFlagNum * sizeof(short));
		float* AllLength = (float*)malloc(totalFlagNum * sizeof(float));

		for (int i = 1; i <= plines; i++)
		{
			int meshSt = 0;
			int quadSt = 0;
			//int arcSt = 0;
			int flagSt = 0;
			for (int j = 1; j < i; j++)
			{
				int meshNumincre = 8 + (beamNumVec[j] - 6) * 2;
				if (beamNumVec[j] < 4)meshNumincre = 0;
				meshSt += meshNumincre;
				int quadNumincre = 6 + 3 * ((meshNumincre - 4) / 2);
				quadSt += quadNumincre;
				//arcSt += 2 * quadNumincre;
				if(beamNumVec[j]>=4)flagSt += beamNumVec[j];
			}
			if (G[i].size() > 1)
			{
				computeIntersectionLength(G[i],nodeVector[i].position,AllLength+flagSt);
				//计算junction的faces
				getTopology(G[i],&nodeVector[i],AllTriangle+meshSt,AllQuad+quadSt,AllFlag+flagSt,AllLength+flagSt);
				//computeConvexhull(G[i], nodeVector[i]);

				//faces.insert(faces.end(),(nodeVector[i]).f, (nodeVector[i]).f+ (nodeVector[i]).triNum);
			}
		}
		//printf("chord error: \n");
		//float ce;
		//scanf("%f", &ce);
		//int totalSampleNum = 0;
		//int totalArcSampleNum = 0;
		//for (int i = 1; i <= plines; i++)
		//{
		//	int meshSt = 0;
		//	int quadSt = 0;
		//	int arcSt = 0;
		//	int flagSt = 0;
		//	for (int j = 1; j < i; j++)
		//	{
		//		int meshNumincre = 8 + (beamNumVec[j] - 6) * 2;
		//		if (beamNumVec[j] < 4)meshNumincre = 0;
		//		meshSt += meshNumincre;
		//		int quadNumincre = 6 + 3 * ((meshNumincre - 4) / 2);
		//		quadSt += quadNumincre;
		//		arcSt += 2 * quadNumincre;
		//		if (beamNumVec[j] >= 4)flagSt += beamNumVec[j];
		//	}
		//	if (G[i].size() > 1)
		//	{
		//		locateArcs(G[i], AllTriangle + meshSt, AllQuad + quadSt, AllArc + arcSt, AllFlag + flagSt);
		//		setSegNum(G[i], ce, AllFlag + flagSt, totalSampleNum, totalArcSampleNum);
		//	}
		//}
		//float3* Allsample = (float3*)malloc(totalSampleNum * sizeof(float3));
		//ArcType* AllbeamArc = (ArcType*)malloc(2 * elines * sizeof(ArcType));
		//Triangle* AllresultTri = (Triangle*)malloc((totalSampleNum - 2 * elines) * sizeof(Triangle));
		//for (int i = 1; i <= elines; i++)
		//{
		//	BeamPlugin* b = beamVector[i];
		//	int beamArcSt = 0;
		//	int beamSampleSt = 0;
		//	int beamTriSt = 0; 
		//	for (int k = 1; k < i; k++)
		//	{
		//		beamSampleSt += AllbeamArc[beamArcSt].sampleNum + AllbeamArc[beamArcSt + 1].sampleNum;
		//		beamTriSt += AllbeamArc[beamArcSt].sampleNum + AllbeamArc[beamArcSt + 1].sampleNum - 2;
		//		beamArcSt += 2;
		//		
		//	}
		//	ArcType* beamArc[2];
		//	for (int idx = 0; idx < 2; idx++)
		//	{
		//		beamArc[idx] = AllbeamArc + beamArcSt + idx;
		//		beamArc[idx]->sample = Allsample + beamSampleSt;
		//		beamArc[idx]->sampleNum = 0;
		//		int smallArcSt = 0;
		//		for (int j = 0; j < b->arcNum[idx]; j++)
		//		{
		//			beamArc[idx]->sampleNum += b->arcArray[idx][j]->sampleNum - 1;
		//			//计算旋转后的采样点位置
		//			b->arcArray[idx][j]->sample = &Allsample[beamSampleSt + smallArcSt];
		//			Allsample[beamSampleSt+smallArcSt] = b->arcArray[idx][j]->st;
		//			int k = 1;
		//			for (; k <= b->arcArray[idx][j]->sampleNum - 2; k++)
		//			{
		//				float** m = getRotationMatrix(b->arcArray[idx][j]->diskNorm, b->arcArray[idx][j]->segTheta*k, b->arcArray[idx][j]->o);
		//				float3 rotateV = make_float3(Allsample[beamSampleSt + smallArcSt].x - b->arcArray[idx][j]->o.x,
		//					Allsample[beamSampleSt + smallArcSt].y - b->arcArray[idx][j]->o.y, Allsample[beamSampleSt + smallArcSt].z - b->arcArray[idx][j]->o.z);
		//				float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
		//				RV = Multiply(m, RV);
		//				Allsample[beamSampleSt + smallArcSt+k] = make_float3(RV.x, RV.y, RV.z);

		//				for (int f = 0; f < 4; f++)
		//				{
		//					free(m[f]);// = (float*)malloc(4 * sizeof(float));
		//				}
		//				free(m);
		//			}
		//			if (j == b->arcNum[idx] - 1)
		//			{
		//				Allsample[beamSampleSt + smallArcSt + k] = b->arcArray[idx][j]->ed;
		//			}
		//			smallArcSt += b->arcArray[idx][j]->sampleNum - 1;
		//		}
		//		beamArc[idx]->sampleNum++;
		//		beamSampleSt += beamArc[idx]->sampleNum;
		//	}
		//	int resultTriNum = beamArc[0]->sampleNum + beamArc[1]->sampleNum - 2;
		//	if (beamArc[0]->sampleNum > beamArc[1]->sampleNum)
		//	{
		//		GWOforQuadTriangulation(beamArc[0]->sampleNum, beamArc[1]->sampleNum, beamArc[0]->sample, beamArc[1]->sample, AllresultTri + beamTriSt, resultTriNum, true);
		//	}
		//	else
		//	{
		//		GWOforQuadTriangulation(beamArc[1]->sampleNum, beamArc[0]->sampleNum, beamArc[1]->sample, beamArc[0]->sample, AllresultTri + beamTriSt, resultTriNum, true);
		//	}
		//}
		//Triangle* AllQuadresultTri = (Triangle*)malloc((totalArcSampleNum - 2 * totalQuadNum) * sizeof(Triangle));
		//for (int i = 0; i < totalQuadNum; i++)
		//{
		//	QuadFace* q = AllQuad+i;
		//	int sampleSt = 0;
		//	for (int j = 0; j < i; j++)
		//	{
		//		QuadFace* qtmp = AllQuad + j;
		//		sampleSt += qtmp->arc[0]->sampleNum + qtmp->arc[1]->sampleNum - 2;
		//	}
		//	int resultTriNum = q->arc[0]->sampleNum + q->arc[1]->sampleNum - 2;
		//	if (q->arc[0]->sampleNum > q->arc[1]->sampleNum)
		//	{
		//		GWOforQuadTriangulation(q->arc[0]->sampleNum, q->arc[1]->sampleNum, q->arc[0]->sample, q->arc[1]->sample, AllQuadresultTri + sampleSt, resultTriNum, false);
		//	}
		//	else
		//	{
		//		GWOforQuadTriangulation(q->arc[1]->sampleNum, q->arc[0]->sampleNum, q->arc[1]->sample, q->arc[0]->sample, AllQuadresultTri + sampleSt, resultTriNum, false);
		//	}
		//}
		//generateFaces(AllresultTri, (totalSampleNum - 2 * elines));
		//generateFaces(AllQuadresultTri, (totalArcSampleNum - 2 * totalQuadNum));
#endif
		//for (int i = 1; i <= elines; i++)
		//{
		//	(*beamVector[i]).BeamTransform();
		//	//faces.insert(faces.end(),(*beamVector[i]).f.begin(), (*beamVector[i]).f.end());
		//}
		convertToSTL(path, faces);
		string plyPath = "D:\\shapeDesignLattice\\Results\\CylinderBeam.ply";
		convertToPLY(plyPath, faces);
	}
#endif
	
	return 0;
}
