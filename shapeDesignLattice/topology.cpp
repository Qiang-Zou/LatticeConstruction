#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include <array>
#include "topology.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/disable_warnings.h>
#include <vector>
#include <set>
#include <map>
//#include "convhull_3d.h"
#include "Header/DotCloud.h"
#include "Header/Triangulation.h"

//#define cgal
#define PI 3.14159265
typedef std::array<std::size_t, 3> Facet;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3  Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef Mesh::Face_index face_descriptor;
typedef Mesh::Vertex_index vertex_descriptor;

int mod(float a, float b);

void getTriangle(BeamPlugin *b[], Junction* &J,Triangle* &triList,int triId)
{
	float3 finalP[3];
	int flag[3] = { 0 };
	
	for (int i = 0; i < 3; i++)
	{
		BeamPlugin *btmp = b[i];
		if (pow(J->position.x - btmp->axis.p[0].x, 2) + pow(J->position.y - btmp->axis.p[0].y, 2) + pow(J->position.z - btmp->axis.p[0].z, 2)
			< pow(J->position.x - btmp->axis.p[1].x, 2) + pow(J->position.y - btmp->axis.p[1].y, 2) + pow(J->position.z - btmp->axis.p[1].z, 2))
			flag[i] = 1;
		else
			flag[i] = 2;
		
		////取采样点
		//float step = 2 * acos(-1) / 60;
		//for (int j = 0; j < 60; j++)
		//{
		//	Point p;
		//	if (flag[i] == 1)
		//	{
		//		p[0] = btmp->length / 2;
		//	}
		//	else if (flag[i] == 2)
		//	{
		//		p[0] = -btmp->length / 2;
		//	}
		//	p[1] = btmp->radius * sin(j*step);
		//	p[2] = btmp->radius * cos(j*step);
		//	Eigen::Vector4d np(p[0],p[1],p[2],1);
		//	np = m * np;
		//	p[0] = np[0];
		//	p[1] = np[1];
		//	p[2] = np[2];
		//	s[i].push_back(p);
		//	//std::cout << i << "圆: " << p[0] << " " << p[1] << " " << p[2] << std::endl;
		//}
	}

	float3 diskNorm[3];//圆盘法线方向
	for (int i = 0; i < 3; i++)
	{
		float3 o1;
		
		if (flag[i] == 1)
		{
			o1 = b[i]->axis.p[0];
			diskNorm[i] = make_float3(b[i]->axis.p[0].x - b[i]->axis.p[1].x, b[i]->axis.p[0].y - b[i]->axis.p[1].y, b[i]->axis.p[0].z - b[i]->axis.p[1].z);
		}
		else if (flag[i] == 2)
		{
			o1 = b[i]->axis.p[1];
			diskNorm[i] = make_float3(b[i]->axis.p[1].x - b[i]->axis.p[0].x, b[i]->axis.p[1].y - b[i]->axis.p[0].y, b[i]->axis.p[1].z - b[i]->axis.p[0].z);// b[i]->axis.p2 - b[i]->axis.p1;
		}
		diskNorm[i] = make_norm(diskNorm[i]);
		
		//计算另外两个圆心
		float3 o2, o3;
		if (flag[(i + 1) % 3] == 1)
		{
			o2 = b[(i + 1) % 3]->axis.p[0];
		}
		else if (flag[(i + 1) % 3] == 2)
		{
			o2 = b[(i + 1) % 3]->axis.p[1];
		}
		if (flag[(i + 2) % 3] == 1)
		{
			o3 = b[(i + 2) % 3]->axis.p[0];
		}
		else if (flag[(i + 2) % 3] == 2)
		{
			o3 = b[(i + 2) % 3]->axis.p[1];
		}
	

		//计算变换后的三角形法线，之后尝试直接把三角形法线变换过来
		float3 normal;
		normal.x = (o2.y - o1.y)*(o3.z - o1.z) - (o2.z - o1.z)*(o3.y - o1.y);
		normal.y = (o2.z - o1.z)*(o3.x - o1.x) - (o2.x - o1.x)*(o3.z - o1.z);
		normal.z = (o2.x - o1.x)*(o3.y - o1.y) - (o2.y - o1.y)*(o3.x - o1.x);

		normal = make_norm(normal);

		float cosVal = Dot(normal,diskNorm[i]);
		float theta = acos(min(max(cosVal,-1.0f),1.0f));
		float s = b[i]->radius / sin(theta);
		finalP[i] = make_float3(s * normal.x + o1.x - s * cos(theta)*diskNorm[i].x, 
			s * normal.y + o1.y - s * cos(theta)*diskNorm[i].y, s * normal.z + o1.z - s * cos(theta)*diskNorm[i].z);

		/*Eigen::Vector3d o1o2 = o2 - o1; o1o2.normalize();
		Eigen::Vector3d o1o3 = o3 - o1; o1o3.normalize();
		cosVal = o1o2.dot(diskNorm[i]);
		theta = acos(cosVal);
		s = b[i]->radius / sin(theta);
		Eigen::Vector3d proj1 = s * o1o2 + o1 - s * cos(theta)*diskNorm[i];
		proj1 = proj1 - o1;

		cosVal = o1o3.dot(diskNorm[i]);
		theta = acos(cosVal);
		s = b[i]->radius / sin(theta);
		Eigen::Vector3d proj2 = s * o1o3 + o1 - s * cos(theta)*diskNorm[i];
		proj2 = proj2 - o1;

		Eigen::Vector3d proj = proj1 + proj2;
		proj.normalize();
		if (proj.dot(normal) < 0)
		{
			proj = -proj;
		}
		cosVal = proj.dot(diskNorm[i]);
		theta = acos(cosVal);
		s = b[i]->radius / sin(theta);
		finalP[i] = s * proj + o1 - s * cos(theta)*diskNorm[i];*/
		//std::cout << proj << std::endl;
		//判断和三角形法线方向是否一致
	/*	if (proj.dot(normal) < 0)
		{
			proj1.normalize();
			proj2.normalize();
			float angle = acos(proj1.dot(proj2));
			Eigen::Vector3d judge = proj1.cross(proj2);
			if (judge.dot(diskNorm) > 0)
			{

			}
		}*/
		//std::cout << proj << std::endl;

		//Eigen::Vector3d o2o1(o1[0]-o2[0],o1[1]-o2[1],o1[2]-o2[2]);
		//Eigen::Vector3d o3o1(o1[0] - o3[0], o1[1] - o3[1], o1[2] - o3[2]);
		////o2o1.normalize(); o3o1.normalize();
		//Eigen::Vector3d addV = o2o1 + o3o1;
		//addV.normalize();
		//if (normV.dot(xAxis) > 0)//角度小于90°
		//{
		//	if (addV.dot(xAxis) > 0)
		//		addV = -addV;
		//}
		//else
		//{
		//	if (addV.dot(xAxis) < 0)
		//		addV = -addV;
		//}
		
	}
	/*int minPerimeter = 1000000;
	Point A, B, C;
	for (int i = 0; i < 60; i++)
	{
		for (int j = 0; j < 60; j++)
		{
			for (int k = 0; k < 60; k++)
			{
				Point a = s[0][i], b = s[1][j], c = s[2][k];
				int perimeter = sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2)) +
					sqrt(pow(a[0] - c[0], 2) + pow(a[1] - c[1], 2) + pow(a[2] - c[2], 2)) +
					sqrt(pow(b[0] - c[0], 2) + pow(b[1] - c[1], 2) + pow(b[2] - c[2], 2));
				if (perimeter < minPerimeter)
				{
					minPerimeter = perimeter;
					A = a; B = b, C = c;
				}
			}
		}
	}*/
	/*for (int i = 0; i < 3; i++)
	{
		if (flag[i] == 1)
		{
			b[i]->tris1Vec.push_back(Eigen::Vector3i(b[i]->beamId,b[(i+1)%3]->beamId,b[(i+2)%3]->beamId));
		}
		else if (flag[i] == 2)
		{
			b[i]->tris2Vec.push_back(Eigen::Vector3i(b[i]->beamId, b[(i + 1) % 3]->beamId, b[(i + 2) % 3]->beamId));
		}
	}
	for (int i = 0; i < 3; i++)
	{
		if (flag[i] == 1)
		{
			b[i]->end1Vec.push_back(finalP[i]);
		}
		else if (flag[i] == 2)
		{
			b[i]->end2Vec.push_back(finalP[i]);
		}
	}*/
	
	Triangle &triTmp = triList[triId];
	triTmp.p[0] = finalP[0];
	triTmp.p[1] = finalP[1];
	triTmp.p[2] = finalP[2];
	float3 e1 = make_float3(finalP[1].x - finalP[0].x, finalP[1].y - finalP[0].y, finalP[1].z - finalP[0].z);
	float3 e2 = make_float3(finalP[2].x - finalP[0].x, finalP[2].y - finalP[0].y, finalP[2].z - finalP[0].z);
	triTmp.normal = cross(e1,e2);
	triTmp.normal = make_norm(triTmp.normal);
	triTmp.triId = triId;
	for (int i = 0; i < 3; i++)
	{
		//triTmp.triWithBeamid[i] = b[i]->beamId;
		triTmp.diskNorm[i] = diskNorm[i];
	}
	/*triTmp.o[0] = mp[b[0]->beamId];
	triTmp.o[1] = mp[b[1]->beamId];
	triTmp.o[2] = mp[b[2]->beamId];*/
	e1 = make_float3(triTmp.o[1].x - triTmp.o[0].x, triTmp.o[1].y - triTmp.o[0].y, triTmp.o[1].z - triTmp.o[0].z);
	e2 = make_float3(triTmp.o[2].x - triTmp.o[0].x, triTmp.o[2].y - triTmp.o[0].y, triTmp.o[2].z - triTmp.o[0].z);
	triTmp.originNormal = cross(e1,e2);
	triTmp.originNormal = make_norm(triTmp.originNormal);

	//triTmp.adjTriangles = (Triangle*)malloc(3*sizeof(Triangle));
	//triTmp.adjQuadFaces = (QuadFace*)malloc(3*sizeof(QuadFace));
	triTmp.adjTriNum = 0;
	triTmp.adjQuadNum = 0;

	triTmp.flag = 0;

	/*if (triId == 0)
		triList = (Triangle*)malloc(sizeof(Triangle));
	else
		triList = (Triangle*)realloc(triList, (triId + 1) * sizeof(Triangle));*/
	
	//triList[triId] = triTmp;
	//triList.push_back(triTmp);
	/*Face ftmp;
	ftmp.p1 = finalP[0];
	ftmp.p2 = finalP[1];
	ftmp.p3 = finalP[2];
	Eigen::Vector3d v1 = ftmp.p1;
	Eigen::Vector3d v2 = ftmp.p2;
	Eigen::Vector3d v3 = ftmp.p3;
	ftmp.normal[0] = (v2[1] - v1[1])*(v3[2] - v1[2]) - (v2[2] - v1[2])*(v3[1] - v1[1]);
	ftmp.normal[1] = (v2[2] - v1[2])*(v3[0] - v1[0]) - (v2[0] - v1[0])*(v3[2] - v1[2]);
	ftmp.normal[2] = (v2[0] - v1[0])*(v3[1] - v1[1]) - (v2[1] - v1[1])*(v3[0] - v1[0]);
	J->f.push_back(ftmp);*/
}
//float computeDistance(Triangle* t1, Triangle* t2)
//{
//	std::vector<float> Dis;
//	//在两个法向上的距离
//	float proj[6];
//	Eigen::Vector3d normal = t1->normal;
//	for (int i = 0; i < 3; i++)
//	{
//		proj[i] = t1->p[i].dot(normal);
//	}
//	for (int i = 3; i < 6; i++)
//	{
//		proj[i] = t2->p[i-3].dot(normal);
//	}
//	float min1 = *std::min_element(proj,proj+3);
//	float max1 = *std::max_element(proj,proj+3);
//	float min2 = *std::min_element(proj+3, proj + 6);
//	float max2 = *std::max_element(proj+3, proj + 6);
//
//	float dis = std::max(min1 - max2, min2 - max1);
//	//Dis.push_back(dis);
//
//	normal = t2->normal;
//	for (int i = 0; i < 3; i++)
//	{
//		proj[i] = t1->p[i].dot(normal);
//	}
//	for (int i = 3; i < 6; i++)
//	{
//		proj[i] = t2->p[i-3].dot(normal);
//	}
//	min1 = *std::min_element(proj, proj + 3);
//	max1 = *std::max_element(proj, proj + 3);
//	min2 = *std::min_element(proj + 3, proj + 6);
//	max2 = *std::max_element(proj + 3, proj + 6);
//	dis = std::max(min1 - max2, min2 - max1);
//	//Dis.push_back(dis);
//
//	//判断是否共面
//	Eigen::Vector3d Cross = t1->normal.cross(t2->normal);
//	if (Cross != Eigen::Vector3d(0, 0, 0))
//	{
//		for (int i = 0; i < 3; i++)
//		{
//			Eigen::Vector3d e1 = t1->p[(i + 1) % 3] - t1->p[i];
//			for (int j = 0; j < 3; j++)
//			{
//				Eigen::Vector3d e2 = t2->p[(j + 1) % 3] - t2->p[j];
//				normal = e1.cross(e2);
//				normal.normalize();
//				for (int k = 0; k < 3; k++)
//				{
//					proj[k] = t1->p[k].dot(normal);
//				}
//				for (int k = 3; k < 6; k++)
//				{
//					proj[k] = t2->p[k-3].dot(normal);
//				}
//				min1 = *std::min_element(proj, proj + 3);
//				max1 = *std::max_element(proj, proj + 3);
//				min2 = *std::min_element(proj + 3, proj + 6);
//				max2 = *std::max_element(proj + 3, proj + 6);
//				dis = std::max(min1 - max2, min2 - max1);
//				Dis.push_back(dis);
//			}
//		}
//	}
//	else//可能平行或共面，当作共面处理
//	{
//		for (int i = 0; i < 6; i++)
//		{
//			if (i < 3)
//			{
//				normal = (t1->p[(i + 1) % 3] - t1->p[i]).cross(t1->normal);
//				normal.normalize();
//			}
//			else
//			{
//				normal = (t2->p[(i - 3 + 1) % 3] - t2->p[i - 3]).cross(t2->normal);
//				normal.normalize();
//			}
//			for (int k = 0; k < 3; k++)
//			{
//				proj[k] = t1->p[k].dot(normal);
//			}
//			for (int k = 3; k < 6; k++)
//			{
//				proj[k] = t2->p[k-3].dot(normal);
//			}
//			min1 = *std::min_element(proj, proj + 3);
//			max1 = *std::max_element(proj, proj + 3);
//			min2 = *std::min_element(proj + 3, proj + 6);
//			max2 = *std::max_element(proj + 3, proj + 6);
//			dis = std::max(min1 - max2, min2 - max1);
//			Dis.push_back(dis);
//		}
//	}
//
//	float resultD = *std::max_element(Dis.begin(), Dis.end());
//	return resultD;
//
//}
void host_isCross(float3 O, float3 A, float3 B, float &proj)
{
	proj = 0;
	float3 AO = Subtraction(O, A);
	float3 AB = Subtraction(B, A);
	float r = Dot(AO, AB) / (Norm(AO) * Norm(AB));
	proj = r;
	/*if (r > 0 && r < 1)
	{
		float3 ab = make_norm(AB);
		proj = Dot(AO, ab) * 2;
		return true;
	}
	else
		return false;*/
}
float measureSingleCost(Triangle* triList,int triNum, QuadFace* quadList, int quadNum,GreyWolves gws,int nVar,bool &inverted)
{
	float centerDis = 0;
	float angleDis = 0;
	float max_N1N2dis = 0;
	int quadCnt = 0;
	//vector<float> costN1N2;
	//vector<float> costT;
	//int triNum = triList.size();
	float projSeg = 0, projSeg2 = 0;
	float triEdgeLength = 0;
	float BF1 = 0;
	float BF = 0;
	for (int i = 0; i < triNum; i++)
	{
		//costT.clear();
		Triangle* t = triList + i;
		float3 originCenter = make_float3((t->o[0].x + t->o[1].x + t->o[2].x) / 3, (t->o[0].y + t->o[1].y + t->o[2].y) / 3,
			(t->o[0].z + t->o[1].z + t->o[2].z) / 3);
		float3 center = make_float3((t->p[0].x + t->p[1].x + t->p[2].x) / 3, (t->p[0].y + t->p[1].y + t->p[2].y) / 3,
			(t->p[0].z + t->p[1].z + t->p[2].z) / 3);
		centerDis += Norm(Subtraction(originCenter, center));
		float originAngle[3];
		for (int j = 0; j < 3; j++)
		{
			float3 e1 = Subtraction(t->o[mod(j + 1, 3)], t->o[j]);
			float3 e2 = Subtraction(t->o[mod(j + 2, 3)], t->o[j]);
			originAngle[j] = acos(Dot(e1, e2) / (Norm(e1)*Norm(e2)));

		}
		float triAngle[3];
		for (int j = 0; j < 3; j++)
		{
			float3 e1 = Subtraction(t->p[mod(j + 1, 3)], t->p[j]);
			float3 e2 = Subtraction(t->p[mod(j + 2, 3)], t->p[j]);
			triAngle[j] = acos(Dot(e1, e2) / (Norm(e1)*Norm(e2)));
		}
		for (int j = 0; j < 3; j++)
		{
			angleDis += abs(originAngle[j] - triAngle[j]);
		}
		//计算每个三角形和相邻四边面的法线角度差
		for (int j = 0; j < 3; j++)
		{
			QuadFace* adjQ = t->adjQuadFaces[j];
			//找公共边
			//for (int k = 0; k < 3; k++)
			//{
			//	int tid = t->triWithBeamid[k];
			//	if (tid != adjQ->quadWithBeamid[0] && tid != adjQ->quadWithBeamid[1])
			//	{
			//		//k-1和k+1为公共边
			//		//检查QuadFace的哪个三角形是直接和当前三角形相邻
			//		float3 tp1 = t->p[mod(k - 1, 3)];
			//		float3 tp2 = t->p[mod(k + 1, 3)];
			//		for (int v = 0; v < 4; v++)
			//		{
			//			Triangle* quadT = &(adjQ->t[v]);
			//			if (quadT->flag == 0)
			//			{
			//				quadCnt++;
			//				int cnt = 0;
			//				for (int x = 0; x < 3; x++)
			//				{
			//					float3 qtp = quadT->p[x];
			//					if (Norm(Subtraction(tp1, qtp)) < 1e-6)
			//						cnt++;
			//					if (Norm(Subtraction(tp2, qtp)) < 1e-6)
			//						cnt++;
			//				}
			//				if (cnt == 2)
			//				{
			//					float COS = Dot(t->normal, quadT->normal);
			//					//float theta = acos(min(max(Dot(t->normal, quadT->normal), -1.0f), 1.0f));//acos(cuda_Dot(t->normal,quadT->normal));
			//				//costT.push_back(theta);
			//				//costN1N2.push_back(theta);
			//					max_N1N2dis += pow(COS - 1, 2);
			//					if (COS<0 && t->inverted[v/2] == 1)//翻转
			//					{
			//						float x = -COS;
			//						BF1 += -(1000 / x)*log(1 - x);
			//					}
			//					//break;
			//				}
			//				/*else
			//				{
			//					float theta = acos(min(max(Dot(t->normal, quadT->normal), -1.0f), 1.0f));
			//					max_N1N2dis += pow(cos(theta) - 1, 2);
			//				}*/
			//				t->cost = max(t->cost, acos(min(max(Dot(t->normal, quadT->normal), -1.0f), 1.0f)));
			//			}

			//			//int cnt = 0;
			//			//for (int x = 0; x < 3; x++)
			//			//{
			//			//	Eigen::Vector3d qtp = quadT.p[x];
			//			//	if (pow(tp1[0] - qtp[0], 2) + pow(tp1[1] - qtp[1], 2) + pow(tp1[2] - qtp[2], 2) < 1e-6)
			//			//		cnt++;
			//			//	if (pow(tp2[0] - qtp[0], 2) + pow(tp2[1] - qtp[1], 2) + pow(tp2[2] - qtp[2], 2) < 1e-6)
			//			//		cnt++;
			//			//}
			//			//if (cnt == 2)//即相邻三角形
			//			//{
			//			//	float theta = acos(t.normal.dot(quadT.normal));
			//			//	costT.push_back(theta);
			//			//	costN1N2.push_back(theta);
			//			//	break;
			//			//}
			//		}
			//	}
			//}
			float3 n_diskNorm = make_float3(-t->diskNorm[j].x, -t->diskNorm[j].y, -t->diskNorm[j].z);
			float3 line1 = Subtraction(t->p[mod(j + 1, 3)], t->p[j]);
			float3 line2 = Subtraction(t->p[mod(j + 2, 3)], t->p[j]);
			if (Dot(n_diskNorm, t->normal) > 0 || Dot(line1, n_diskNorm) > 0 || Dot(line2, n_diskNorm) > 0)
			{

				float cosTheta = Dot(t->diskNorm[j], line1) / Norm(line1);
				float length1 = cosTheta * Norm(line1);
				float3 newDiskNorm1 = make_float3(t->diskNorm[j].x*length1, t->diskNorm[j].y*length1, t->diskNorm[j].z*length1);

				/*float cosTheta = cuda_Dot(n_diskNorm, t->normal);
				float theta = acos(Min(Max(cosTheta, -1.0f), 1.0f));
				float length1 = sin(theta)*cuda_Norm(line1);
				float3 newDiskNorm1 = make_float3(t->diskNorm[j].x*length1, t->diskNorm[j].y*length1, t->diskNorm[j].z*length1);*/
				float3 sub1 = Subtraction(line1, newDiskNorm1);
				float3 projL1 = make_float3(sub1.x + t->p[j].x, sub1.y + t->p[j].y, sub1.z + t->p[j].z);

				cosTheta = Dot(t->diskNorm[j], line2) / Norm(line2);
				float length2 = cosTheta * Norm(line2);
				float3 newDiskNorm2 = make_float3(t->diskNorm[j].x*length2, t->diskNorm[j].y*length2, t->diskNorm[j].z*length2);
				/*float length2 = sin(theta)*cuda_Norm(line2);
				float3 newDiskNorm2 = make_float3(t->diskNorm[j].x*length2, t->diskNorm[j].y*length2, t->diskNorm[j].z*length2);*/
				float3 sub2 = Subtraction(line2, newDiskNorm2);
				float3 projL2 = make_float3(sub2.x + t->p[j].x, sub2.y + t->p[j].y, sub2.z + t->p[j].z);

				float projLength1, projLength2;
				host_isCross(t->o[j], t->p[j], projL1, projLength1);
				host_isCross(t->o[j], t->p[j], projL2, projLength2);
				if (projLength1 > 0 && projLength2 < 0)
					projSeg += projLength1;
				else if (projLength1 < 0 && projLength2>0)
					projSeg += projLength2;
				else
					projSeg += projLength1 + projLength2;
				//判断是否本来需要optimal cut
				//float3 vjtoj_1 = cuda_Subtraction(t->p[cuda_mod(j+1,3)],t->p[j]);
				if (Dot(line1, n_diskNorm) > 0 || Dot(line2, n_diskNorm) > 0)
				{
					if (projLength1 > 0 || projLength2 > 0)
					{
						//t->inverted = 2;
						inverted = true; projSeg2 += projLength1; projSeg2 += projLength2;
						t->projSeg += projLength1 + projLength2;
						float bf = projLength1 + projLength2;
						if (projLength1 < 0)
							bf = projLength2;
						else if (projLength2 < 0)
							bf = projLength1;
						bf = bf / 2;
						BF += -(1000 / bf)*log(1 - bf);
						//BF = 10000;
					}
				}
			}
			else if (Dot(n_diskNorm, t->normal) < 0)
			{
				float cosTheta = Dot(t->diskNorm[j], line1) / Norm(line1);
				float length1 = cosTheta * Norm(line1);
				float3 newDiskNorm1 = make_float3(t->diskNorm[j].x * length1, t->diskNorm[j].y * length1, t->diskNorm[j].z * length1);

				/*float cosTheta = cuda_Dot(n_diskNorm, t->normal);
				float theta = acos(Min(Max(cosTheta, -1.0f), 1.0f));
				float length1 = sin(theta)*cuda_Norm(line1);
				float3 newDiskNorm1 = make_float3(t->diskNorm[j].x*length1, t->diskNorm[j].y*length1, t->diskNorm[j].z*length1);*/
				float3 sub1 = Subtraction(line1, newDiskNorm1);
				float3 projL1 = make_float3(sub1.x + t->p[j].x, sub1.y + t->p[j].y, sub1.z + t->p[j].z);

				cosTheta = Dot(t->diskNorm[j], line2) / Norm(line2);
				float length2 = cosTheta * Norm(line2);
				float3 newDiskNorm2 = make_float3(t->diskNorm[j].x * length2, t->diskNorm[j].y * length2, t->diskNorm[j].z * length2);
				/*float length2 = sin(theta)*cuda_Norm(line2);
				float3 newDiskNorm2 = make_float3(t->diskNorm[j].x*length2, t->diskNorm[j].y*length2, t->diskNorm[j].z*length2);*/
				float3 sub2 = Subtraction(line2, newDiskNorm2);
				float3 projL2 = make_float3(sub2.x + t->p[j].x, sub2.y + t->p[j].y, sub2.z + t->p[j].z);

				float projLength1, projLength2;
				host_isCross(t->o[j], t->p[j], projL1, projLength1);
				host_isCross(t->o[j], t->p[j], projL2, projLength2);
				if (projLength1 > 0 && projLength2 < 0)
					projSeg += -projLength2;
				else if (projLength1 < 0 && projLength2>0)
					projSeg += -projLength1;
				else
					projSeg += -(projLength1 + projLength2);
			}


		}
		for (int j = 0; j < 3; j++)
		{
			float l = Norm(Subtraction(t->p[mod(j + 1, 3)], t->p[j]));
			triEdgeLength += l;
		}
		//t.cost = *std::max_element(costT.begin(), costT.end());

	}
	for (int i = 0; i < quadNum; i++)
	{
		QuadFace* qTmp = &quadList[i];
		for (int j = 0; j < 4; j++)
		{
			float COS = qTmp->COS[j];
			max_N1N2dis += pow(COS - 1, 2);
			if (COS < 0 && qTmp->inverted[j / 2] == 1)
			{
				float x = -COS;
				BF1 += -(1000 / x) * log(1 - x);
			}
			else if ((j == 0 || j == 3) && COS < 0 && qTmp->inverted[2] == 1)
			{
				float x = -COS;
				BF1 += -(1000 / x) * log(1 - x);
			}
			else if ((j == 1 || j == 2) && COS < 0 && qTmp->inverted[3] == 1)
			{
				float x = -COS;
				BF1 += -(1000 / x) * log(1 - x);
			}
			else if ((j == 0 || j == 2) && COS < 0 && qTmp->inverted[4] == 1)
			{
				float x = -COS;
				BF1 += -(1000 / x) * log(1 - x);
			}
			else if ((j == 1 || j == 3) && COS < 0 && qTmp->inverted[5] == 1)
			{
				float x = -COS;
				BF1 += -(1000 / x) * log(1 - x);
			}
		}
		//if (qTmp->t[0].flag == 0 && qTmp->t[1].flag == 0)
		//{
		//	float COS = Dot(qTmp->t[0].normal, qTmp->t[1].normal);
		//	//float theta = acos(min(max(Dot(qTmp->t[0].normal, qTmp->t[1].normal), -1.0f), 1.0f));//acos(cuda_Dot(qTmp->t[0].normal,qTmp->t[1].normal));
		//	//costN1N2.push_back(theta);
		//	max_N1N2dis += pow(COS - 1, 2);
		//	//if (COS < 0 && qTmp->t[0].inverted[0] == 1)//翻转
		//	//{
		//	//	float x = -COS;
		//	//	BF1 += -(1000 / x) * log(1 - x);
		//	//}
		//}
		//if (qTmp->t[2].flag == 0 && qTmp->t[3].flag == 0)
		//{
		//	float COS = Dot(qTmp->t[2].normal, qTmp->t[3].normal);
		//	//float theta = acos(min(max(Dot(qTmp->t[2].normal, qTmp->t[3].normal), -1.0f), 1.0f));//acos(cuda_Dot(qTmp->t[0].normal,qTmp->t[1].normal));
		//	//costN1N2.push_back(theta);
		//	max_N1N2dis += pow(COS - 1, 2);
		//	//if (COS < 0 && qTmp->t[2].inverted[0] == 1)//翻转
		//	//{
		//	//	float x = -COS;
		//	//	BF1 += -(1000 / x) * log(1 - x);
		//	//}
		//}
	}

	/*for (int i = 0; i < nVar; i++)
	{
		max_N1N2dis += pow(cos(abs(gws.rotationAngle[i])) - 1, 2);
	}*/
	//float max_N1N2dis = *std::max_element(costN1N2.begin(), costN1N2.end());

	/*int costNum = costN1N2.size();
	for (int i = 0; i < costNum; i++)
		max_N1N2dis += pow(cos(costN1N2[i]) - 1, 2);*/

		/*int costQNum = costQ.size();
		for (int i = 0; i < costQNum; i++)
			max_N1N2dis += pow(cos(costQ[i]) - 1, 2);*/

			/*for (int i = 0; i < nVar; i++)
			{
				max_N1N2dis += 0.6*pow(gws.rotationAngle[i], 2);
			}*/
	float variance = 0; float quadAngle = 0;
	for (int i = 0; i < quadNum; i++)
	{
		QuadFace* qTmp = &quadList[i];
		float avg = (qTmp->width[0] + qTmp->width[1]) / 2;
		float v = (pow(qTmp->width[0] - avg, 2) + pow(qTmp->width[1] - avg, 2)) / 2;
		variance += v;

		/*if (qTmp->t[0].flag == 0 && qTmp->t[1].flag == 0)
		{
			float3 e1 = cuda_Subtraction(qTmp->t[0].p[1], qTmp->t[0].p[0]);
			float3 e2 = cuda_Subtraction(qTmp->t[0].p[2], qTmp->t[0].p[1]);
			float3 e3 = cuda_Subtraction(qTmp->t[1].p[2], qTmp->t[1].p[1]);
			float3 e4 = cuda_Subtraction(qTmp->t[1].p[0], qTmp->t[1].p[2]);
			e1 = cuda_make_norm(e1);
			e2 = cuda_make_norm(e2);
			e3 = cuda_make_norm(e3);
			e4 = cuda_make_norm(e4);
			float theta1 = acos(Min(Max(cuda_Dot(e1, e2), -1.0f), 1.0f));
			float theta2 = acos(Min(Max(cuda_Dot(e2, e3), -1.0f), 1.0f));
			float theta3 = acos(Min(Max(cuda_Dot(e3, e4), -1.0f), 1.0f));
			float theta4 = acos(Min(Max(cuda_Dot(e1, e4), -1.0f), 1.0f));
			quadAngle += pow(cos(theta1), 2);
			quadAngle += pow(cos(theta2), 2);
			quadAngle += pow(cos(theta3), 2);
			quadAngle += pow(cos(theta4), 2);
		}*/
	}
	int cnt = 0; float thetaPenalty = 0;
	/*for (int i = 0; i < beamNum; i++)
	{
		BeamPlugin* b = beams[i];
		int idx = flag[i]-1;
		for (int j = 1; j <= b->arcNum[idx]; j++)
		{
			float theta = gws.rotationAngle[cnt + j];
			thetaPenalty += exp(-pow(theta, 2) * 15);
		}
		cnt += b->arcNum[idx] + 1;
	}*/
	//printf("%f %f %f %f\n", max_N1N2dis, variance/50, projSeg/6,BF);
	//max_N1N2dis += variance/50;
	//max_N1N2dis /= quadCnt;
	//printf("%f %f %f %f\n", max_N1N2dis, thetaPenalty*10/100, projSeg / 40/2, variance / 60/3);
	//max_N1N2dis *= 100;
	//max_N1N2dis += thetaPenalty*50;
	//max_N1N2dis += (projSeg/40/2);
	//max_N1N2dis += variance / 60/3;
	//max_N1N2dis += triEdgeLength / 10;
	//max_N1N2dis += BF;
	//max_N1N2dis += quadAngle;
	//printf("%f %f %f\n", max_N1N2dis, projSeg2 * 3, BF);
	/*if (inverted == true)
		max_N1N2dis += BF;*/
		//printf("%f %f %f\n", centerDis/10, angleDis/2, max_N1N2dis);
		//printf("%f %f\n", BF, BF1/4);
		//printf("%f %f %f %f %f %f\n", max_N1N2dis, projSeg / 4, BF, BF1 / 4, centerDis / 5, angleDis);
	return max_N1N2dis + projSeg + BF + BF1;
	//vector<float> costN1N2,costQ;
	//vector<float> costT;
	////int triNum = triList.size();
	//for (int i = 0; i < triNum; i++)
	//{
	//	costT.clear();
	//	Triangle t = triList[i];
	//	//新加一个三角形法线与原法线的角度
	//	//costN1N2.push_back(acos(min(max(Dot(t.normal, t.originNormal),-1.0f),1.0f)));
	//	/*if (!(acos(min(max(Dot(t.normal, t.originNormal), -1.0f), 1.0f)) >=0&& acos(min(max(Dot(t.normal, t.originNormal), -1.0f), 1.0f)) <=PI))
	//	{
	//		std::cout << Dot(t.normal, t.originNormal) << " " << acos(Dot(t.normal, t.originNormal))<<" "<< "wrong angle!!!" << std::endl;
	//	}*/
	//	//计算每个三角形和相邻四边面的法线角度差
	//	for (int j = 0; j < 3; j++)
	//	{
	//		QuadFace adjQ = *(t.adjQuadFaces[j]);
	//		//找公共边
	//		for (int k = 0; k < 3; k++)
	//		{
	//			int tid = t.triWithBeamid[k];
	//			if (tid != adjQ.quadWithBeamid[0] && tid != adjQ.quadWithBeamid[1])
	//			{
	//				//k-1和k+1为公共边
	//				//检查QuadFace的哪个三角形是直接和当前三角形相邻
	//				float3 tp1 = t.p[mod(k - 1, 3)];
	//				float3 tp2 = t.p[mod(k + 1, 3)];
	//				for (int v = 0; v < 2; v++)
	//				{
	//					Triangle quadT = adjQ.t[v];
	//					if (quadT.flag == 0)
	//					{
	//						float theta = acos(min(max(Dot(t.normal, quadT.normal),-1.0f),1.0f));
	//						costT.push_back(theta);
	//						costN1N2.push_back(theta);
	//						/*if (!(theta >=0&&theta<=PI))
	//							std::cout << Dot(t.normal, quadT.normal) <<" "<<theta<<" "<<"wrong angle!!!" << std::endl;*/
	//					}
	//					
	//					//int cnt = 0;
	//					//for (int x = 0; x < 3; x++)
	//					//{
	//					//	Eigen::Vector3d qtp = quadT.p[x];
	//					//	if (pow(tp1[0] - qtp[0], 2) + pow(tp1[1] - qtp[1], 2) + pow(tp1[2] - qtp[2], 2) < 1e-6)
	//					//		cnt++;
	//					//	if (pow(tp2[0] - qtp[0], 2) + pow(tp2[1] - qtp[1], 2) + pow(tp2[2] - qtp[2], 2) < 1e-6)
	//					//		cnt++;
	//					//}
	//					//if (cnt == 2)//即相邻三角形
	//					//{
	//					//	float theta = acos(t.normal.dot(quadT.normal));
	//					//	costT.push_back(theta);
	//					//	costN1N2.push_back(theta);
	//					//	break;
	//					//}
	//				}
	//			}
	//		}
	//	}

	//	t.cost = *std::max_element(costT.begin(), costT.end());
	//	
	//}
	//for (int i = 0; i < quadNum; i++)
	//{
	//	QuadFace qTmp = quadList[i];
	//	if (qTmp.t[0].flag == 0 && qTmp.t[1].flag == 0)
	//	{
	//		float theta = acos(min(max(Dot(qTmp.t[0].normal, qTmp.t[1].normal),-1.0f),1.0f));
	//		costN1N2.push_back(theta);
	//	}
	//	
	//}
	////float max_N1N2dis = *std::max_element(costN1N2.begin(), costN1N2.end());
	//float max_N1N2dis = 0;
	//int costNum = costN1N2.size();
	//for (int i = 0; i < costNum; i++)
	//	max_N1N2dis += pow(cos(costN1N2[i])-1,2);

	///*int costQNum = costQ.size();
	//for (int i = 0; i < costQNum; i++)
	//	max_N1N2dis += pow(cos(costQ[i]) - 1, 2);*/

	//for (int i = 0; i < nVar; i++)
	//{
	//	max_N1N2dis += pow(cos(abs(gws.rotationAngle[i]))-1, 2);
	//}

	//return max_N1N2dis;
}
void measureDistance(Triangle* &triList,QuadFace* &quadList, int triNum,int quadNum,int nObj, GreyWolves& gws, int nVar)
{
	std::vector<float> costTmp,costN,costN1N2,costQ;
	//float* cost = new float[nObj];
	//int triNum = triList.size();
	//bool* visited = new bool[triNum];
	/*std::vector<bool> visited;
	visited.resize(triNum);
	std::fill(visited.begin(),visited.end(),false);*/
	for (int i = 0; i < triNum; i++)
	{
		Triangle* t = &triList[i];
		//costN.push_back(acos(Dot(t.normal,t.originNormal)));
		//计算每个三角形和相邻四边面的法线角度差
		for (int j = 0; j < 3; j++)
		{
			QuadFace* adjQ = t->adjQuadFaces[j];
			//找公共边
			for (int k = 0; k < 3; k++)
			{
				int tid = t->triWithBeamid[k];
				if (tid != adjQ->quadWithBeamid[0] && tid != adjQ->quadWithBeamid[1])
				{
					//k-1和k+1为公共边
					//检查QuadFace的哪个三角形是直接和当前三角形相邻
					float3 tp1 = t->p[mod(k - 1, 3)];
					float3 tp2 = t->p[mod(k + 1, 3)];
					for (int v = 0; v < 2; v++)
					{
						Triangle* quadT = &(adjQ->t[v]);
						if (quadT->flag == 0)
						{
							float theta = acos(min(max(Dot(t->normal, quadT->normal), -1.0f), 1.0f));
							//costT.push_back(theta);
							costN1N2.push_back(theta);
							/*if (!(theta >=0&&theta<=PI))
								std::cout << Dot(t.normal, quadT.normal) <<" "<<theta<<" "<<"wrong angle!!!" << std::endl;*/
						}
						//int cnt = 0;
						//for (int x = 0; x < 3; x++)
						//{
						//	float3 qtp = quadT.p[x];
						//	if (pow(tp1.x - qtp.x, 2) + pow(tp1.y - qtp.y, 2) + pow(tp1.z - qtp.z, 2) < 1e-6)
						//		cnt++;
						//	if (pow(tp2.x - qtp.x, 2) + pow(tp2.y - qtp.y, 2) + pow(tp2.z - qtp.z, 2) < 1e-6)
						//		cnt++;
						//}
						//if (cnt == 2)//即相邻三角形
						//{
						//	costN1N2.push_back(acos(Dot(t.normal,quadT.normal)));
						//	break;
						//}
					}
				}
			}
		}
		//for (int j = 0; j < 3; j++)
		//{
		//	Triangle* adjT = t.adjTriangles[j];
		//	if (!visited[adjT.triId])
		//	{
		//		//计算两个三角形之间的距离
		//		float dis = computeDistance(t,adjT);
		//		//std::cout << dis << std::endl;
		//		costTmp.push_back(dis);
		//	}
		//}
		//visited[t.triId] = true;
	}
	//float min_value = *std::min_element(costTmp.begin(), costTmp.end());
	////cost.push_back(min_value);

	//float sum = std::accumulate(costTmp.begin(), costTmp.end(), 0.0);
	//float avg = sum / costTmp.size();
	////cost.push_back(avg);

	//int subNum = costTmp.size();
	//std::vector<float> sub;
	//for (int i = 0; i < subNum; i++)
	//{
	//	sub.push_back(pow(costTmp[i] - avg, 2));
	//}
	//sum = std::accumulate(sub.begin(),sub.end(),0.0);
	//float variance = sum / subNum;
	//cost.push_back(1/(variance));
	//std::cout << min_value << " " << avg << std::endl;
	/*float max_Ndis = *std::max_element(costN.begin(), costN.end());
	cost.push_back(1 / max_Ndis);*/

	//quadface cost
	//costTmp.clear();
	//int quadNum = quadList.size();
	//visited.resize(quadNum);
	//fill(visited.begin(),visited.end(),false);
	//for (int i = 0; i < quadNum; i++)
	//{
	//	QuadFace* q = quadList[i];
	//	for (int j = 0; j < 4; j++)
	//	{
	//		QuadFace* adjQ = q.adjQuadFaces[j];
	//		if (!visited[adjQ.quadId])
	//		{
	//			//计算两个四边面之间的距离
	//			std::vector<float> dis;
	//			for (int k = 0; k < 2; k++)
	//			{
	//				for (int v = 0; v < 2; v++)
	//				{
	//					Triangle* t1 = q.t[k];
	//					Triangle* t2 = adjQ.t[v];
	//					float d = computeDistance(t1,t2);
	//					dis.push_back(d);
	//				}
	//			}
	//			costTmp.push_back(*std::min_element(dis.begin(), dis.end()));
	//		}
	//	}
	//	visited[q.quadId] = true;
	//}
	//cost.push_back(*std::min_element(costTmp.begin(), costTmp.end()));
	for (int i = 0; i < quadNum; i++)
	{
		QuadFace* qTmp = &quadList[i];
		if (qTmp->t[0].flag == 0 && qTmp->t[1].flag == 0)
		{
			float theta = acos(min(max(Dot(qTmp->t[0].normal, qTmp->t[1].normal), -1.0f), 1.0f));
			costN1N2.push_back(theta);
		}

	}
	//float max_N1N2dis = *std::max_element(costN1N2.begin(), costN1N2.end());
	float max_N1N2dis = 0;
	int costNum = costN1N2.size();
	for (int i = 0; i < costNum; i++)
		max_N1N2dis += pow(cos(costN1N2[i]) - 1, 2);

	/*int costQNum = costQ.size();
	for (int i = 0; i < costQNum; i++)
		max_N1N2dis += pow(cos(costQ[i]) - 1, 2);*/

	for (int i = 0; i < nVar; i++)
	{
		max_N1N2dis += pow(cos(abs(gws.rotationAngle[i])) - 1, 2);
	}
	gws.Cost[0] = max_N1N2dis;
	//cost[0] = 1/max_N1N2dis;

	float variance = 0;
	for (int i = 0; i < quadNum; i++)
	{
		QuadFace* qTmp = &quadList[i];
		float avg = (qTmp->width[0] + qTmp->width[1]) / 2;
		float v = (pow(qTmp->width[0] - avg, 2) + pow(qTmp->width[1] - avg, 2)) / 2;
		variance += v;
	}
	gws.Cost[1] = variance;
	//int quadNum = quadList.size();
	/*for (int i = 0; i < quadNum; i++)
	{
		QuadFace q = quadList[i];
		costQ.push_back(acos(Dot(q.t[0].normal,q.t[1].normal)));
	}
	float max_Q = *std::max_element(costQ.begin(), costQ.end());
	cost[1] = 1 / max_Q;*/

	//return cost;
}
void findAdjTriangles(Triangle* triList,  int triNum)
{
	//bool tmpInverted = false;
	//inverted = false;
	//float3 t1Tot2[2], t2Tot1[2];
	//float3 projP1[2], projP2[2];
	////公共边的两个点
	//float3 edge1Points[2], edge2Points[2];
	//int id1[2], id2[2];

	//int triNum = triList.size();
	//cout << triNum << endl;
	/*for (int i = 0; i < triNum; i++)
		triList[i]->adjTriangles.clear();*/
	for (int i = 0; i < triNum-1; i++)
	{
		Triangle* t1 = triList+i;
		for (int j = i + 1; j < triNum; j++)
		{
			Triangle* t2 = triList+j;
			//只要有两个点相同就是相邻三角形
			int cnt = 0;
			for (int k = 0; k < 3; k++)
			{
				int p2 = t2->triWithBeamid[k];
				for (int v = 0; v < 3; v++)
				{
					int p1 = t1->triWithBeamid[v];
					if (p2 == p1)
					{
						/*t1Tot2[cnt] = make_float3(t2.p[k].x - t1.p[v].x, t2.p[k].y - t1.p[v].y, t2.p[k].z - t1.p[v].z);
						t2Tot1[cnt] = make_float3(-t1Tot2[cnt].x, -t1Tot2[cnt].y, -t1Tot2[cnt].z);

						edge1Points[cnt] = t1.p[v];
						id1[cnt] = v;
						edge2Points[cnt] = t2.p[k];
						id2[cnt] = k;*/
						//计算2个投影点，共4个投影点
						/*float theta = acos((t1.normal.dot(t1Tot2[cnt])) / (t1.normal.norm()*t1Tot2[cnt].norm()));
						projP1[cnt] = t1Tot2[cnt] * sin(theta) + t1.p[v];
						theta = acos((t2.normal.dot(t2Tot1[cnt])) / (t2.normal.norm()*t2Tot1[cnt].norm()));
						projP2[cnt] = t2Tot1[cnt] * sin(theta) + t2.p[k];*/

						cnt++;
					}
				}
			}
			if (cnt == 2)
			{
				/*if ((t1Tot2[0].x == 0 && t1Tot2[0].y == 0 && t1Tot2[0].z == 0) ||
					(t1Tot2[1].x == 0 && t1Tot2[1].y == 0 && t1Tot2[1].z == 0))
					cout << "co-point!!!" << endl;*/
				/*if(t1.adjTriNum==0)
					t1.adjTriangles = (Triangle*)malloc(sizeof(Triangle));
				else
					t1.adjTriangles = (Triangle*)realloc(t1.adjTriangles, (t1.adjTriNum + 1) * sizeof(Triangle));*/
				/*t1.adjTriangles[t1.adjTriNum].p[0] = t2.p[0];
				t1.adjTriangles[t1.adjTriNum].p[1] = t2.p[1];
				t1.adjTriangles[t1.adjTriNum].p[2] = t2.p[2];
				t1.adjTriangles[t1.adjTriNum].triId = t2.triId;
				t1.adjTriangles[t1.adjTriNum].triWithBeamid[0] = t2.triWithBeamid[0];
				t1.adjTriangles[t1.adjTriNum].triWithBeamid[1] = t2.triWithBeamid[1];
				t1.adjTriangles[t1.adjTriNum].triWithBeamid[2] = t2.triWithBeamid[2];
				t1.adjTriangles[t1.adjTriNum].normal = t2.normal;*/
				t1->adjTriangles[t1->adjTriNum] = t2;
				t1->adjTriNum++;
				//triList[i] = t1;
				/*if(t2.adjTriNum==0)
					t2.adjTriangles = (Triangle*)malloc(sizeof(Triangle));
				else
					t2.adjTriangles = (Triangle*)realloc(t2.adjTriangles, (t2.adjTriNum + 1) * sizeof(Triangle));*/
				/*t2.adjTriangles[t2.adjTriNum].p[0] = t1.p[0];
				t2.adjTriangles[t2.adjTriNum].p[1] = t1.p[1];
				t2.adjTriangles[t2.adjTriNum].p[2] = t1.p[2];
				t2.adjTriangles[t2.adjTriNum].triId = t1.triId;
				t2.adjTriangles[t2.adjTriNum].triWithBeamid[0] = t1.triWithBeamid[0];
				t2.adjTriangles[t2.adjTriNum].triWithBeamid[1] = t1.triWithBeamid[1];
				t2.adjTriangles[t2.adjTriNum].triWithBeamid[2] = t1.triWithBeamid[2];
				t2.adjTriangles[t2.adjTriNum].normal = t1.normal;*/
				t2->adjTriangles[t2->adjTriNum] = t1;
				t2->adjTriNum++;
				//triList[j] = t2;
				//t1.adjTriangles.push_back(*t2);
				//t2.adjTriangles.push_back(*t1);
				//判断是否翻转
				//判断投影点是否在三角形内部
				//if (!inverted)
				//{
				//	//float i1[2], i2[2], j1[2], j2[2], k1[2], k2[2];
				//	for (int k = 0; k < 2; k++)
				//	{
				//		float3 e1, e2;
				//		if (id2[0] == 0 && id2[1] == 2)
				//		{
				//			e1 = make_float3(edge1Points[1].x - edge1Points[0].x, edge1Points[1].y - edge1Points[0].y, edge1Points[1].z - edge1Points[0].z);
				//			e2 = make_float3(edge2Points[0].x - edge2Points[1].x, edge2Points[0].y - edge2Points[1].y, edge2Points[0].z - edge2Points[1].z);
				//		}
				//		else
				//		{
				//			e1 = make_float3(edge1Points[0].x - edge1Points[1].x, edge1Points[0].y - edge1Points[1].y, edge1Points[0].z - edge1Points[1].z);
				//			e2 = make_float3(edge2Points[1].x - edge2Points[0].x, edge2Points[1].y - edge2Points[0].y, edge2Points[1].z - edge2Points[0].z);
				//		}
				//		float3 edge1Normal = cross(e1,t1.normal);
				//		float3 edge2Normal = cross(e2,t2.normal);
				//		if (Dot(t1Tot2[0],edge1Normal) < 0 || Dot(t1Tot2[1],edge1Normal) < 0 ||
				//			Dot(t2Tot1[0],edge2Normal) < 0 || Dot(t2Tot1[1],edge2Normal) < 0)
				//			tmpInverted = true;
				//		/*i1[k] = (-(projP1[k][0] - t1->p[1][0])*(t1->p[2][1] - t1->p[1][1]) + (projP1[k][1] - t1->p[1][1])*(t1->p[2][0] - t1->p[1][0]))
				//			/ (-(t1->p[0][0] - t1->p[1][0])*(t1->p[2][1] - t1->p[1][1]) + (t1->p[0][1] - t1->p[1][1])*(t1->p[2][0] - t1->p[1][0]));
				//		j1[k] = (-(projP1[k][0] - t1->p[2][0])*(t1->p[0][1] - t1->p[2][1]) + (projP1[k][1] - t1->p[2][1])*(t1->p[0][0] - t1->p[2][0]))
				//			/ (-(t1->p[1][0] - t1->p[2][0])*(t1->p[0][1] - t1->p[2][1]) + (t1->p[1][1] - t1->p[2][1])*(t1->p[0][0] - t1->p[2][0]));
				//		k1[k] = 1 - i1[k] - j1[k];
				//		if (i1[k] > 0 && i1[k] < 1 && j1[k] > 0 && j1[k] < 1 && k1[k] > 0 && k1[k] < 1)tmpInverted = true;*/

				//		/*i2[k] = (-(projP2[k][0] - t2->p[1][0])*(t2->p[2][1] - t2->p[1][1]) + (projP2[k][1] - t2->p[1][1])*(t2->p[2][0] - t2->p[1][0]))
				//			/ (-(t2->p[0][0] - t2->p[1][0])*(t2->p[2][1] - t2->p[1][1]) + (t2->p[0][1] - t2->p[1][1])*(t2->p[2][0] - t2->p[1][0]));
				//		j2[k] = (-(projP2[k][0] - t2->p[2][0])*(t2->p[0][1] - t2->p[2][1]) + (projP2[k][1] - t2->p[2][1])*(t2->p[0][0] - t2->p[2][0]))
				//			/ (-(t2->p[1][0] - t2->p[2][0])*(t2->p[0][1] - t2->p[2][1]) + (t2->p[1][1] - t2->p[2][1])*(t2->p[0][0] - t2->p[2][0]));
				//		k2[k] = 1 - i2[k] - j2[k];
				//		if (i2[k] > 0 && i2[k] < 1 && j2[k] > 0 && j2[k] < 1 && k2[k] > 0 && k2[k] < 1)tmpInverted = true;*/

				//	}
				//	if (tmpInverted)
				//	{
				//		//若为凸，则不算翻转
				//		if (Dot(t1.normal,t1Tot2[0]) < 0 && Dot(t1.normal,t1Tot2[1]) < 0 && Dot(t2.normal,t2Tot1[0]) < 0 && Dot(t2.normal,t2Tot1[1]) < 0)
				//			tmpInverted = false;
				//	}
				//	if (tmpInverted)
				//		inverted = true;
				//}
				
				/*if (t1->normal.dot(t1Tot2[0]) > 0 || t1->normal.dot(t1Tot2[1]) > 0)
					inverted = true;
				if (t2->normal.dot(t2Tot1[0]) > 0 || t2->normal.dot(t2Tot1[1]) > 0)
					inverted = true;*/
			}
		}
	}
}
int mod(float a, float b)
{
	return a - (b*floor(a / b));
}
void findQuadFaces(Triangle* triList, QuadFace* quadList,int triNum, bool &inverted)
{
	short tmpInverted = 0;
	inverted = false;
	bool tmpFlipped = false;
	int quadId = 0;
	//int triNum = triList.size();
	bool* visited = (bool*)malloc(triNum*sizeof(bool));
	for (int i = 0; i < triNum; i++)visited[i] = false;
	//std::fill(visited, visited + triNum, false);
	/*for (int i = 0; i < triNum; i++)
		triList[i]->adjQuadFaces.clear();*/
	for (int i = 0; i < triNum; i++)
	{
		Triangle* t = triList+i;
		//shared_ptr<QuadFace> threeQuad[3];
		for (int j = 0; j < 3; j++)
		{
			Triangle* adjT = t->adjTriangles[j];
			if (!visited[adjT->triId])
			{
				//找公共边
				std::vector<int> adjPoints;
				for (int k = 0; k < 3; k++)
				{
					adjPoints.push_back(adjT->triWithBeamid[k]);
				}
				for (int k = 0; k < 3; k++)
				{
					int pId = t->triWithBeamid[k];
					if (find(adjPoints.begin(), adjPoints.end(), pId) == adjPoints.end())
					{
						//k-1和k+1就是公共边
						QuadFace* q = quadList+quadId;
						//q->t.resize(2);
						q->quadId = quadId++;
						q->inverted[0] = q->inverted[1] = q->inverted[2] = q->inverted[3] = q->inverted[4] = q->inverted[5] = 0;
						q->quadWithBeamid[0] = t->triWithBeamid[mod((k - 1) , 3)];
						q->quadWithBeamid[1] = t->triWithBeamid[mod((k + 1) , 3)];
						Triangle t0;
						Triangle t1;

						t0.p[0] = t->p[mod((k - 1) , 3)];
						t0.p[1] = t->p[mod((k + 1) , 3)];
						auto it = find(adjPoints.begin(), adjPoints.end(), t->triWithBeamid[(k+1)%3]);
						int k2 = it - adjPoints.begin();
						t0.p[2] = adjT->p[k2];
						//判断是否共点
						float3 line1 = make_float3(t0.p[2].x - t0.p[1].x, t0.p[2].y - t0.p[1].y, t0.p[2].z - t0.p[1].z);
						if (abs(line1.x) == 0 && abs(line1.y) == 0&& abs(line1.z) == 0)
						{
							q->t[0].flag = 1;
							q->width[0] = 0;
						}
						else
						{
							float3 e1 = make_float3(t0.p[1].x - t0.p[0].x, t0.p[1].y - t0.p[0].y, t0.p[1].z - t0.p[0].z);
							float3 e2 = make_float3(t0.p[2].x - t0.p[0].x, t0.p[2].y - t0.p[0].y, t0.p[2].z - t0.p[0].z);
							t0.normal = cross(e1, e2);
							t0.normal = make_norm(t0.normal);
							q->t[0] = t0;
							//float3 line1 = make_float3(t0.p[2].x - t0.p[1].x, t0.p[2].y - t0.p[1].y, t0.p[2].z - t0.p[1].z);
							q->width[0] = Norm(line1);
							//判断是否翻转
							float3 edge1Normal = cross(Subtraction(t0.p[0], t0.p[1]), t->normal);
							if (Dot(edge1Normal, line1) < 0)
								tmpInverted++; //tmpInverted = true;
							q->t[0].flag = 0;
							q->t[0].inverted[0] = 0;
							q->COS[0] = Dot(edge1Normal, line1) / (Norm(edge1Normal) * Norm(line1));
						}
						

						int k3 = mod((k2 - 1) , 3);
						t1.p[0] = t->p[mod((k - 1) , 3)];
						t1.p[1] = adjT->p[k2];
						t1.p[2] = adjT->p[k3];
						//判断是否共点
						float3 line2 = make_float3(t1.p[0].x - t1.p[2].x, t1.p[0].y - t1.p[2].y, t1.p[0].z - t1.p[2].z);
						if (abs(line2.x) == 0 && abs(line2.y) == 0 && abs(line2.z) == 0)
						{
							q->t[1].flag = 1;
							q->width[1] = 0;
						}
						else
						{
							float3 e1 = make_float3(t1.p[1].x - t1.p[0].x, t1.p[1].y - t1.p[0].y, t1.p[1].z - t1.p[0].z);
							float3 e2 = make_float3(t1.p[2].x - t1.p[0].x, t1.p[2].y - t1.p[0].y, t1.p[2].z - t1.p[0].z);
							t1.normal = cross(e1, e2);
							t1.normal = make_norm(t1.normal);
							q->t[1] = t1;
							//float3 line2 = make_float3(t1.p[0].x - t1.p[2].x, t1.p[0].y - t1.p[2].y, t1.p[0].z - t1.p[2].z);
							q->width[1] = Norm(line2);
							//判断是否翻转
							float3 edge2Normal = cross(Subtraction(t1.p[1], t1.p[2]), adjT->normal);
							if (Dot(edge2Normal, line2) < 0)
								tmpInverted++; //tmpInverted = true;
							q->t[1].flag = 0;
							q->t[1].inverted[0] = 0;
							q->COS[3] = Dot(edge2Normal, line2) / (Norm(edge2Normal) * Norm(line2));
						}
						

						/*if(t->adjQuadNum==0)
							t->adjQuadFaces = (QuadFace*)malloc(sizeof(QuadFace));
						else
                            t->adjQuadFaces = (QuadFace*)realloc(t->adjQuadFaces, (t->adjQuadNum + 1) * sizeof(QuadFace));*/
						t->adjQuadFaces[t->adjQuadNum] = q;
						t->adjQuadNum++;
						//triList[i] = t;
						adjT->adjQuadFaces[adjT->adjQuadNum] = q;
						adjT->adjQuadNum++;
						//Triangle Ttmp = triList[adjT.triId];
						/*if(Ttmp->adjQuadNum==0)
							Ttmp->adjQuadFaces = (QuadFace*)malloc(sizeof(QuadFace));
						else
                            Ttmp->adjQuadFaces = (QuadFace*)realloc(Ttmp->adjQuadFaces, (Ttmp->adjQuadNum + 1) * sizeof(QuadFace));*/
						//Ttmp.adjQuadFaces[Ttmp.adjQuadNum] = q;
						//Ttmp.adjQuadNum++;
						//triList[adjT.triId] = Ttmp;

						//quadList[quadId-1] = q;
						/*t->adjQuadFaces.push_back(*q);
						triList[adjT.triId]->adjQuadFaces.push_back(*q);*/

						//threeQuad[j] = q;
						//quadList.push_back(q);
						/*if(quadId==1)
							quadList = (QuadFace**)malloc(sizeof(QuadFace*));
						else
							quadList = (QuadFace**)realloc(quadList, (quadId) * sizeof(QuadFace*));
						quadList[quadId - 1] = q;*/
						if (Dot(q->t[0].normal, q->t[1].normal) < 0)
							tmpInverted++;
						//if (!inverted)
						//{
						//	if (tmpInverted>=2)
						//	{
						//		//若为凸，则不算翻转
						//		float3 nline1 = make_float3(-line1.x, -line1.y, -line1.z);
						//		float3 nline2 = make_float3(-line2.x, -line2.y, -line2.z);
						//		if (Dot(t->normal, line1) < 0 && Dot(t->normal, nline2) < 0 &&
						//			Dot(adjT->normal, line2) < 0 && Dot(adjT->normal, nline1) < 0);
						//		else
						//		{
						//			t->inverted[0] = 1; adjT->inverted[0] = 1;
						//			q->t[0].inverted[0] = q->t[1].inverted[0] = 1;
						//			inverted = true;
						//		}
						//			//tmpInverted = false;
						//	}
						//	/*if (tmpInverted)
						//		inverted = true;*/
						//}
						tmpInverted = 0;

						Triangle* t2 = &q->t[2];
						Triangle* t3 = &q->t[3];
						t2->p[0] = t->p[mod((k - 1), 3)];
						t2->p[1] = t->p[mod((k + 1), 3)];
						it = find(adjPoints.begin(),adjPoints.end(), t->triWithBeamid[mod((k - 1),3)]);
						k2 = it - adjPoints.begin();
						t2->p[2] = adjT->p[k2];
						//判断是否共点
						float3 line3 = make_float3(t2->p[2].x - t2->p[0].x, t2->p[2].y - t2->p[0].y, t2->p[2].z - t2->p[0].z);
						if (line3.x == 0 && line3.y == 0 && line3.z == 0)
						{
							q->t[2].flag = 1;
						}
						else
						{
							float3 e1 = make_float3(t2->p[1].x - t2->p[0].x, t2->p[1].y - t2->p[0].y, t2->p[1].z - t2->p[0].z);
							float3 e2 = make_float3(t2->p[2].x - t2->p[0].x, t2->p[2].y - t2->p[0].y, t2->p[2].z - t2->p[0].z);
							t2->normal = cross(e1, e2);
							t2->normal = make_norm(t2->normal);
							//判断是否翻转
							float3 edge3Normal = cross(Subtraction(t2->p[0], t2->p[1]), t->normal);
							if (Dot(edge3Normal, line3) < 0)
								tmpInverted++;
							q->t[2].flag = 0;
							q->t[2].inverted[0] = 0;
							q->COS[2] = Dot(edge3Normal, line3) / (Norm(edge3Normal) * Norm(line3));
						}

						k3 = mod((k2 + 1), 3);
						t3->p[0] = adjT->p[k2];
						t3->p[1] = t->p[mod((k + 1), 3)];
						t3->p[2] = adjT->p[k3];
						//判断是否共点
						float3 line4 = make_float3(t3->p[1].x - t3->p[2].x, t3->p[1].y - t3->p[2].y, t3->p[1].z - t3->p[2].z);
						if (abs(line4.x) == 0 && abs(line4.y) == 0 && abs(line4.z) == 0)
						{
							q->t[3].flag = 1;
						}
						else
						{
							float3 e1 = make_float3(t3->p[1].x - t3->p[0].x, t3->p[1].y - t3->p[0].y, t3->p[1].z - t3->p[0].z);
							float3 e2 = make_float3(t3->p[2].x - t3->p[0].x, t3->p[2].y - t3->p[0].y, t3->p[2].z - t3->p[0].z);
							t3->normal = cross(e1, e2);
							t3->normal = make_norm(t3->normal);
							//判断是否翻转
							float3 edge4Normal = cross(Subtraction(t3->p[2], t3->p[0]), adjT->normal);
							if (Dot(edge4Normal, line4) < 0)
								tmpInverted++;
							q->t[3].flag = 0;
							q->t[3].inverted[0] = 0;
							q->COS[1] = Dot(edge4Normal, line4) / (Norm(edge4Normal) * Norm(line4));
						}
						if (Dot(q->t[2].normal, q->t[3].normal) < 0)
							tmpInverted++;

						if ((q->COS[0] < 0 && q->COS[1] < 0) || (q->COS[2] < 0 && q->COS[3] < 0) || (q->COS[0] < 0 && q->COS[3] < 0) || (q->COS[1] < 0 && q->COS[2] < 0) ||
							(q->COS[0] < 0 && q->COS[2] < 0) || (q->COS[1] < 0 && q->COS[3] < 0))
						{
							float3 nline3 = make_float3(-line3.x, -line3.y, -line3.z);
							float3 nline4 = make_float3(-line4.x, -line4.y, -line4.z);
							float3 line5 = Subtraction(adjT->p[mod(k3 + 1, 3)], t2->p[0]);
							float3 line6 = Subtraction(t->p[k], t3->p[0]);

							if (q->COS[0] < 0 && q->COS[1] < 0)
							{
								q->inverted[0] = 1;
								tmpFlipped = true;
							}
							if (q->COS[2] < 0 && q->COS[3] < 0)
							{
								q->inverted[1] = 1;
								tmpFlipped = true;
							}
							if (q->COS[0] < 0 && q->COS[3] < 0)
							{
								q->inverted[2] = 1;
								tmpFlipped = true;
							}
							if (q->COS[1] < 0 && q->COS[2] < 0)
							{
								q->inverted[3] = 1;
								tmpFlipped = true;
							}
							if (q->COS[0] < 0 && q->COS[2] < 0)
							{
								q->inverted[4] = 1;
								tmpFlipped = true;
								if ((Dot(t->normal, line3) < 0 && Dot(t->normal, nline4) < 0 && Dot(t->normal, line5) < 0) ||
									(Dot(adjT->normal, line4) < 0 && Dot(adjT->normal, nline3) < 0 && Dot(adjT->normal, line6) < 0))
								{
									q->inverted[4] = 0;
									tmpFlipped = false;
								}
							}
							if (q->COS[1] < 0 && q->COS[3] < 0)
							{
								q->inverted[5] = 1;
								tmpFlipped = true;
								if ((Dot(t->normal, line3) < 0 && Dot(t->normal, nline4) < 0 && Dot(t->normal, line5) < 0) ||
									(Dot(adjT->normal, line4) < 0 && Dot(adjT->normal, nline3) < 0 && Dot(adjT->normal, line6) < 0))
								{
									q->inverted[5] = 0;
									tmpFlipped = false;
								}
							}
							if (Dot(t->normal, line3) < 0 && Dot(t->normal, nline4) < 0 && Dot(t->normal, line5) < 0 &&
								Dot(adjT->normal, line4) < 0 && Dot(adjT->normal, nline3) < 0 && Dot(adjT->normal, line6) < 0)
							{
								q->inverted[0] = q->inverted[1] = q->inverted[2] = q->inverted[3] = q->inverted[4] = q->inverted[5] = 0;
								tmpFlipped = false;
							}
							if (tmpFlipped)
								inverted = true;
						}
						//if (tmpInverted >= 2)
						//{
						//	//若为凸，则不算翻转
						//	float3 nline3 = make_float3(-line3.x, -line3.y, -line3.z);
						//	float3 nline4 = make_float3(-line4.x, -line4.y, -line4.z);
						//	if (Dot(t->normal, line3) < 0 && Dot(t->normal, nline4) < 0 &&
						//		Dot(adjT->normal, line4) < 0 && Dot(adjT->normal, nline3) < 0);
						//	else
						//	{
						//		t->inverted[1] = 1; adjT->inverted[1] = 1;
						//		q->t[2].inverted[0] = q->t[3].inverted[0] = 1;
						//		inverted = true;
						//	}
						//}
						tmpInverted = 0;
					}
				}
			}
			//else
			//{
			//	//若相邻三角形已访问过
			//	std::vector<int> tPoints;
			//	for (int k = 0; k < 3; k++)
			//	{
			//		tPoints.push_back(t->triWithBeamid[k]);
			//	}
			//	int adjNum = 3;
			//	for (int k = 0; k < adjNum; k++)
			//	{
			//		QuadFace qTmp = adjT.adjQuadFaces[k];
			//		int id1 = qTmp.quadWithBeamid[0];
			//		int id2 = qTmp.quadWithBeamid[1];
			//		if (find(tPoints.begin(), tPoints.end(), id1) != tPoints.end() && find(tPoints.begin(), tPoints.end(), id2) != tPoints.end())
			//		{
			//			threeQuad[j] = qTmp;
			//		}
			//	}
			//}
		}
		visited[t->triId] = true;
		//建立quadFace的邻接关系
		/*for (int j = 0; j < 3; j++)
		{
			threeQuad[j]->adjQuadFaces.push_back(threeQuad[(j + 1) % 3]);
			threeQuad[(j + 1) % 3]->adjQuadFaces.push_back(threeQuad[j]);
		}*/
	}
	free(visited);
	//quadNum = quadId;
}
void generateFaces(Triangle* &triList, int triNum, QuadFace* &quadList,int quadNum,Junction* &J)
{
	//int triNum = triList.size();
	//J->f = (Face*)malloc((triNum+ quadNum*2) * sizeof(Face));
	J->f = (Face*)malloc((triNum) * sizeof(Face));
	J->triNum = triNum;
	J->quadNum = quadNum;
	for (int i = 0; i < triNum; i++)
	{
		Triangle t = triList[i];
		Face ftmp;
		ftmp.p1 = t.p[0];
		ftmp.p2 = t.p[1];
		ftmp.p3 = t.p[2];
		ftmp.normal = t.normal;
		J->f[i] = ftmp;
	}
	//int quadNum = quadList.size();
	/*for (int i = triNum; i < triNum+2*quadNum; i+=2)
	{
		QuadFace q = quadList[(i-triNum)/2];
		Face ftmp;
		ftmp.p1 = q.t[0].p[0];
		ftmp.p2 = q.t[0].p[1];
		ftmp.p3 = q.t[0].p[2];
		ftmp.normal = q.t[0].normal;
		J->f[i] = (ftmp);

		ftmp.p1 = q.t[1].p[0];
		ftmp.p2 = q.t[1].p[1];
		ftmp.p3 = q.t[1].p[2];
		ftmp.normal = q.t[1].normal;
		J->f[i+1]=(ftmp);
	}*/
	
}
bool needExchange(std::vector<Eigen::Vector3d>& E, Eigen::Vector3d o1, Eigen::Vector3d diskNorm)
{
	int pointNum = E.size();
	for (int i = 0; i < pointNum; i++)
	{
		Eigen::Vector3d o1p1 = E[i]-o1;
		Eigen::Vector3d o1p2 = E[(i + 1)%pointNum] - o1;
		Eigen::Vector3d direct = o1p1.cross(o1p2);
		if (direct.dot(diskNorm) < 0)
		{
			o1p1.normalize(); o1p2.normalize();
			float cosVal = o1p1.dot(o1p2);
			float theta = acos(cosVal);
			if (theta < PI / 2)
			    return true;
		}
	}
	return false;
}
void exchangePoint(std::vector<Eigen::Vector3d>& E, Eigen::Vector3d o1, Eigen::Vector3d diskNorm)
{
	int pointNum = E.size();
	for (int i = 0; i < pointNum; i++)
	{
		Eigen::Vector3d o1p1 = E[i] - o1;
		Eigen::Vector3d o1p2 = E[(i + 1) % pointNum] - o1;
		Eigen::Vector3d direct = o1p1.cross(o1p2);
		direct.normalize(); diskNorm.normalize();
		if (direct.dot(diskNorm) < 0)
		{
			//do exchange
			o1p1.normalize(); o1p2.normalize();
			float cosVal = o1p1.dot(o1p2);
			float theta = acos(cosVal);
			if (theta < PI / 2)
			{
				Eigen::Vector3d tmp = E[i];
				E[i] = E[(i + 1) % pointNum];
				E[(i + 1) % pointNum] = tmp;
			}
			
		}
	}
}
float** getRotationMatrix(float3 diskNorm,float theta, float3 transl)
{
	float u = diskNorm.x, v = diskNorm.y, w = diskNorm.z;
	float** m = (float**)malloc(4*sizeof(float*));
	for (int i = 0; i < 4; i++)
	{
		m[i] = (float*)malloc(4 * sizeof(float));
	}
	m[0][0] = pow(u, 2) + (1 - pow(u, 2))*cos(theta);
	m[0][1] = u * v*(1 - cos(theta)) - w * sin(theta);
	m[0][2] = u * w*(1 - cos(theta)) + v * sin(theta);
	m[1][0] = u * v*(1 - cos(theta)) + w * sin(theta);
	m[1][1] = pow(v, 2) + (1 - pow(v, 2))*cos(theta);
	m[1][2] = v * w*(1 - cos(theta)) - u * sin(theta);
	m[2][0] = u * w*(1 - cos(theta)) - v * sin(theta);
	m[2][1] = v * w*(1 - cos(theta)) + u * sin(theta);
	m[2][2] = pow(w, 2) + (1 - pow(w, 2))*cos(theta);
	
	m[0][3] = transl.x;
	m[1][3] = transl.y;
	m[2][3] = transl.z;
	m[3][3] = 1;

	m[3][0] = 0; m[3][1] = 0; m[3][2] = 0;
	//m << 0;
	//m << 0, 0, 0, 1;

	//m << 1, 0, 0,0,1,0,0,0, 1, 0, 0, 0, 1, 0, 0, 0;
	/*Eigen::MatrixXd ll(1, 4);
	ll << 0, 0, 0, 1;

	Eigen::MatrixXd M(4, 4);
	M << m, transl,ll;*/
	//std::cout << M << std::endl;
	return m;
}
//void separatingRegions(std::vector<Eigen::Vector3i> triOnConvex, std::map<int, Eigen::Vector3d> mp,BeamPlugin b[], Face &ftmp, Junction* &J)
//{
//	std::cout<< J->nodeId << " "<< std::endl;
//	Eigen::Vector3d SA[3];//separatingAxis
//	Eigen::Vector3d diskNorm[3];
//	Eigen::Vector3d Jpoint = J->position;
//	Eigen::Vector3d C[3];//三个圆心
//	for (int i = 0; i < 3; i++)
//	{
//		if (pow(J->position[0] - b[i].axis.p1[0], 2) + pow(J->position[1] - b[i].axis.p1[1], 2) + pow(J->position[2] - b[i].axis.p1[2], 2)
//			< pow(J->position[0] - b[i].axis.p2[0], 2) + pow(J->position[1] - b[i].axis.p2[1], 2) + pow(J->position[2] - b[i].axis.p2[2], 2))
//		{
//			C[i] = b[i].axis.p1;
//			diskNorm[i] = b[i].axis.p1-b[i].axis.p2;
//		}
//		else
//		{
//			C[i] = b[i].axis.p2;
//			diskNorm[i] = b[i].axis.p2 - b[i].axis.p1;
//		}
//		diskNorm[i].normalize();
//	}
//	Eigen::Vector3d selfNormal;
//	selfNormal = (C[1] - C[0]).cross(C[2] - C[0]);
//	selfNormal.normalize();
//
//	//找出相邻三个三角形
//	Eigen::Vector3i triAdj[3];
//	Eigen::Vector3d triNormal[3];
//	for (int i = 0; i < 3; i++)
//	{
//		int id1 = b[i].beamId,id2 = b[(i+1)%3].beamId;
//		int triNum = triOnConvex.size();
//		for (int j = 0; j < triNum; j++)
//		{
//			Eigen::Vector3i triTmp = triOnConvex[j];
//			if (triTmp[1] == id2 && triTmp[2] == id1)
//			{
//				triAdj[i] = triTmp;
//				break;
//			}
//			else if (triTmp[0] == id2 && triTmp[1] == id1)
//			{
//				triAdj[i] = triTmp;
//				break;
//			}
//			else if (triTmp[0] == id1 && triTmp[2] == id2)
//			{
//				triAdj[i] = triTmp;
//				break;
//			}
//		}
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		Eigen::Vector3i triTmp = triAdj[i];
//		Eigen::Vector3d v1 = mp[triTmp[0]];
//		Eigen::Vector3d v2 = mp[triTmp[1]];
//		Eigen::Vector3d v3 = mp[triTmp[2]];
//		triNormal[i] = (v2 - v1).cross(v3 - v1);
//		triNormal[i].normalize();
//	}
//
//	//separatingAxis 0 to 1, 1 to 2, 2 to 0
//	for (int i = 0; i < 3; i++)
//	{
//		//Eigen::Vector3d CtoJ = Jpoint - C[i];
//		Eigen::Vector3d CtoC = C[(i + 1) % 3] - C[i];
//		Eigen::Vector3d evenSep = (triNormal[i] - selfNormal) / 2 + selfNormal;
//		evenSep.normalize();
//		SA[i] = evenSep.cross(CtoC);
//		//SA[i] = CtoJ.cross(CtoC);
//		SA[i].normalize();
//		//ensure SA point to inside
//		//计算两个三角形法线和SA的夹角
//		/*float selfAngle = acos(selfNormal.dot(SA[i]));
//		float adjAngle = acos(triNormal[i].dot(SA[i]));*/
//
//		if ((C[(i+2)%3]-C[i]).dot(SA[i])<=0)
//			SA[i] = -SA[i];
//		
//	}
//	//保存三角形面三个顶点
//	Eigen::Vector3d triVertex[3];
//	triVertex[0] = ftmp.p1;
//	triVertex[1] = ftmp.p2;
//	triVertex[2] = ftmp.p3;
//	//保存9个向量
//	Eigen::Vector3d triVector[3][3];
//	//i是第i个separatingAxis
//	for (int i = 0; i < 3; i++)
//	{
//		//j是第j个三角形顶点向量
//		for (int j = 0; j < 3; j++)
//		{
//			triVector[i][j] = triVertex[j] - C[i];
//		}
//	}
//
//	//调整三角形位置使其位于三个SA范围内
//	for (int j = 0; j < 3; j++)
//	{
//		float proj[3];
//		for (int i = 0; i < 3; i++)
//		{
//			//int flag = 1;//表示逆时针旋转
//			proj[i] = triVector[i][j].dot(SA[i]);
//		}
//		float selfProj;
//		float selfProj2;
//		/*float twoProj[2];
//		
//		if (j == 0)
//		{
//			twoProj[0] = proj[0];
//			twoProj[1] = proj[2];
//		}
//		else if (j == 1)
//		{
//			twoProj[0] = proj[0];
//			twoProj[1] = proj[1];
//		}
//		else if (j == 2)
//		{
//			twoProj[0] = proj[1];
//			twoProj[1] = proj[2];
//		}*/
//		selfProj = triVector[j][j].dot(selfNormal);
//		//selfProj2 = triVector[j][j].dot(selfNormal);
//		while (proj[0] < 0 || proj[1] < 0 ||proj[2]<0)
//		{
//			//std::cout << twoProj[0] << " " << twoProj[1] << std::endl;
//			Eigen::MatrixXd m;
//			m = getRotationMatrix(diskNorm[j], PI / 2880, C[j]);
//			Eigen::Vector3d rotateV = triVertex[j] - C[j];
//			Eigen::Vector4d RV(rotateV[0], rotateV[1], rotateV[2], 1);
//			RV = m * RV;
//
//			for (int i = 0; i < 3; i++)
//			{
//				proj[i] = (RV.head(3) - C[i]).dot(SA[i]);
//				triVertex[j] = RV.head(3);
//				triVector[i][j] = triVertex[j] - C[i];
//			}
//			/*if (j == 0)
//			{
//				twoProj[0] = proj[0];
//				twoProj[1] = proj[2];
//			}
//			else if (j == 1)
//			{
//				twoProj[0] = proj[0];
//				twoProj[1] = proj[1];
//			}
//			else if (j == 2)
//			{
//				twoProj[0] = proj[1];
//				twoProj[1] = proj[2];
//			}*/
//			selfProj = triVector[j][j].dot(selfNormal);
//
//		}
//		std::cout << proj[0] << " " << proj[1] << " " << proj[2] << " "<<selfProj<<std::endl;
//
//	}
//
//	ftmp.p1 = triVertex[0];
//	ftmp.p2 = triVertex[1];
//	ftmp.p3 = triVertex[2];
//	Eigen::Vector3d v1 = ftmp.p1;
//	Eigen::Vector3d v2 = ftmp.p2;
//	Eigen::Vector3d v3 = ftmp.p3;
//	ftmp.normal[0] = (v2[1] - v1[1])*(v3[2] - v1[2]) - (v2[2] - v1[2])*(v3[1] - v1[1]);
//	ftmp.normal[1] = (v2[2] - v1[2])*(v3[0] - v1[0]) - (v2[0] - v1[0])*(v3[2] - v1[2]);
//	ftmp.normal[2] = (v2[0] - v1[0])*(v3[1] - v1[1]) - (v2[1] - v1[1])*(v3[0] - v1[0]);
//}
//void makeFaces(std::vector<Eigen::Vector3i> triOnConvex, std::map<int, Eigen::Vector3d> mp, std::vector<BeamPlugin*> &beams, Junction* &J)
//{
//	int beamNum = beams.size();
//	std::set<std::string> triSet;
//	BeamPlugin b[3];
//	for (int i = 0; i < beamNum; i++)
//	{
//		b[0] = *beams[i];
//		std::vector<Eigen::Vector3i> tris[3];
//		std::vector<Eigen::Vector3d> ends[3];
//		if (pow(J->position[0] - b[0].axis.p1[0], 2) + pow(J->position[1] - b[0].axis.p1[1], 2) + pow(J->position[2] - b[0].axis.p1[2], 2)
//			< pow(J->position[0] - b[0].axis.p2[0], 2) + pow(J->position[1] - b[0].axis.p2[1], 2) + pow(J->position[2] - b[0].axis.p2[2], 2))
//		{
//			tris[0] = b[0].tris1Vec;
//			ends[0] = b[0].end1Vec;
//		}
//		else
//		{
//			tris[0] = b[0].tris2Vec;
//			ends[0] = b[0].end2Vec;
//		}
//		for (int j = 0; j < tris[0].size(); j++)
//		{
//			Eigen::Vector3d finalP[3];
//			Eigen::Vector3i triTmp = tris[0][j];
//			std::string tmpTri = std::to_string(triTmp[0]) + " " + std::to_string(triTmp[1])
//				+ " " + std::to_string(triTmp[2]);
//			if (triSet.find(tmpTri) == triSet.end())//未访问过的三角形
//			{
//				triSet.insert(tmpTri);
//				triSet.insert(std::to_string(triTmp[1]) + " " + std::to_string(triTmp[2])
//					+ " " + std::to_string(triTmp[0]));
//				triSet.insert(std::to_string(triTmp[2]) + " " + std::to_string(triTmp[0])
//					+ " " + std::to_string(triTmp[1]));
//
//				finalP[0] = ends[0][j];
//
//				for (int k = 0; k < beamNum; k++)
//				{
//					BeamPlugin btmp = *beams[k];
//					if (btmp.beamId == triTmp[1])
//					{
//						b[1] = btmp;
//					}
//					else if (btmp.beamId == triTmp[2])
//					{
//						b[2] = btmp;
//					}
//				}
//				if (pow(J->position[0] - b[1].axis.p1[0], 2) + pow(J->position[1] - b[1].axis.p1[1], 2) + pow(J->position[2] - b[1].axis.p1[2], 2)
//					< pow(J->position[0] - b[1].axis.p2[0], 2) + pow(J->position[1] - b[1].axis.p2[1], 2) + pow(J->position[2] - b[1].axis.p2[2], 2))
//				{
//					tris[1] = b[1].tris1Vec;
//					ends[1] = b[1].end1Vec;
//				}
//				else
//				{
//					tris[1] = b[1].tris2Vec;
//					ends[1] = b[1].end2Vec;
//				}
//				if (pow(J->position[0] - b[2].axis.p1[0], 2) + pow(J->position[1] - b[2].axis.p1[1], 2) + pow(J->position[2] - b[2].axis.p1[2], 2)
//					< pow(J->position[0] - b[2].axis.p2[0], 2) + pow(J->position[1] - b[2].axis.p2[1], 2) + pow(J->position[2] - b[2].axis.p2[2], 2))
//				{
//					tris[2] = b[2].tris1Vec;
//					ends[2] = b[2].end1Vec;
//				}
//				else
//				{
//					tris[2] = b[2].tris2Vec;
//					ends[2] = b[2].end2Vec;
//				}
//				for (int k = 0; k < tris[1].size(); k++)
//				{
//					Eigen::Vector3i triJudge = tris[1][k];
//					if (triJudge[0] == triTmp[1] && triJudge[1] == triTmp[2] && triJudge[2] == triTmp[0])
//					{
//						finalP[1] = ends[1][k];
//						break;
//					}
//				}
//				for (int k = 0; k < tris[2].size(); k++)
//				{
//					Eigen::Vector3i triJudge = tris[2][k];
//					if (triJudge[0] == triTmp[2] && triJudge[1] == triTmp[0] && triJudge[2] == triTmp[1])
//					{
//						finalP[2] = ends[2][k];
//						break;
//					}
//				}
//				Face ftmp;
//				ftmp.p1 = finalP[0];
//				ftmp.p2 = finalP[1];
//				ftmp.p3 = finalP[2];
//				Eigen::Vector3d v1 = ftmp.p1;
//				Eigen::Vector3d v2 = ftmp.p2;
//				Eigen::Vector3d v3 = ftmp.p3;
//				ftmp.normal[0] = (v2[1] - v1[1])*(v3[2] - v1[2]) - (v2[2] - v1[2])*(v3[1] - v1[1]);
//				ftmp.normal[1] = (v2[2] - v1[2])*(v3[0] - v1[0]) - (v2[0] - v1[0])*(v3[2] - v1[2]);
//				ftmp.normal[2] = (v2[0] - v1[0])*(v3[1] - v1[1]) - (v2[1] - v1[1])*(v3[0] - v1[0]);
//				/*if (mark == 1)
//				{
//					ftmp.p1 = finalP[0];
//					ftmp.p2 = finalP[2];
//					ftmp.p3 = finalP[1];
//					Eigen::Vector3d v1 = ftmp.p1;
//					Eigen::Vector3d v2 = ftmp.p2;
//					Eigen::Vector3d v3 = ftmp.p3;
//					ftmp.normal[0] = (v2[1] - v1[1])*(v3[2] - v1[2]) - (v2[2] - v1[2])*(v3[1] - v1[1]);
//					ftmp.normal[1] = (v2[2] - v1[2])*(v3[0] - v1[0]) - (v2[0] - v1[0])*(v3[2] - v1[2]);
//					ftmp.normal[2] = (v2[0] - v1[0])*(v3[1] - v1[1]) - (v2[1] - v1[1])*(v3[0] - v1[0]);
//				}*/
//				//separatingRegions(triOnConvex,mp,b,ftmp,J);
//				J->f.push_back(ftmp);
//			}
//		}
//
//	}
//}
//void checkIntersection_makeFaces(std::vector<Triangle*> triList,std::vector<Eigen::Vector3i> triOnConvex,std::map<int,Eigen::Vector3d> mp,std::vector<BeamPlugin*> &beams, Junction* &J)
//{
//	int mark = 0;
//	int beamNum = beams.size();
//	int triNum = triList.size();
//	for (int i = 0; i < beamNum; i++)
//	{
//		BeamPlugin b = *beams[i];
//		int flag = 0;
//		std::vector<Eigen::Vector3i> tris,T;
//		std::vector<Eigen::Vector3d> ends,E;
//		Eigen::Vector3d o1;
//		Eigen::Vector3d diskNorm;
//		if (pow(J->position[0] - b.axis.p1[0], 2) + pow(J->position[1] - b.axis.p1[1], 2) + pow(J->position[2] - b.axis.p1[2], 2)
//			< pow(J->position[0] - b.axis.p2[0], 2) + pow(J->position[1] - b.axis.p2[1], 2) + pow(J->position[2] - b.axis.p2[2], 2))
//		{
//			flag = 1;
//			tris = b.tris1Vec;
//			ends = b.end1Vec;
//			o1 = b.axis.p1;
//			diskNorm = b.axis.p1 - b.axis.p2;
//		}
//		else
//		{
//			tris = b.tris2Vec;
//			ends = b.end2Vec;
//			o1 = b.axis.p2;
//			diskNorm = b.axis.p2 - b.axis.p1;
//		}
//		diskNorm.normalize();
//		//get m
//		float rotateAngle = 2 * PI / tris.size();
//		float test = cos(PI);
//		Eigen::MatrixXd m = getRotationMatrix(diskNorm, rotateAngle,o1);
//
//		Eigen::Vector3i firstTri = tris[0]; T.push_back(firstTri); E.push_back(ends[0]);
//		int A = firstTri[0], B = firstTri[1];
//		//用作测试
//		Triangle* t;
//		for (int j = 0; j < triNum; j++)
//		{
//			t = triList[j];
//			if (t->triWithBeamid[0] == firstTri[0] && t->triWithBeamid[1] == firstTri[1] && t->triWithBeamid[2] == firstTri[2])
//			{
//				t->p[0] = ends[0]; break;
//			}
//			else if (t->triWithBeamid[1] == firstTri[0] && t->triWithBeamid[2] == firstTri[1] && t->triWithBeamid[0] == firstTri[2])
//			{
//				t->p[1] = ends[0]; break;
//			}
//			else if (t->triWithBeamid[2] == firstTri[0] && t->triWithBeamid[0] == firstTri[1] && t->triWithBeamid[1] == firstTri[2])
//			{
//				t->p[2] = ends[0]; break;
//			}
//		}
//
//		int findTriNum = tris.size() - 1;
//		while (findTriNum--)
//		{
//			for (int j = 1; j < tris.size(); j++)
//			{
//				Eigen::Vector3i triTmp = tris[j];
//				if (A == triTmp[0] && B == triTmp[2])
//				{
//					T.push_back(triTmp);
//					Eigen::Vector3d o1end = E.back() - o1;
//					Eigen::Vector4d endTmp(o1end[0], o1end[1], o1end[2],1);
//					endTmp = m * endTmp;
//					E.push_back(Eigen::Vector3d(endTmp[0],endTmp[1],endTmp[2]));
//					//用作测试
//					for (int k = 0; k < triNum; k++)
//					{
//						t = triList[k];
//						if (t->triWithBeamid[0] == triTmp[0] && t->triWithBeamid[1] == triTmp[1] && t->triWithBeamid[2] == triTmp[2])
//						{
//							t->p[0] = E.back(); break;
//						}
//						else if (t->triWithBeamid[1] == triTmp[0] && t->triWithBeamid[2] == triTmp[1] && t->triWithBeamid[0] == triTmp[2])
//						{
//							t->p[1] = E.back(); break;
//						}
//						else if (t->triWithBeamid[2] == triTmp[0] && t->triWithBeamid[0] == triTmp[1] && t->triWithBeamid[1] == triTmp[2])
//						{
//							t->p[2] = E.back(); break;
//						}
//					}
//
//					A = triTmp[0];
//					B = triTmp[1];
//					break;
//				}
//			}
//			
//		}
//		
//		//std::cout << m;
//		//while (needExchange(E, o1, diskNorm))
//		//{
//		//	//mark = 1;
//		//	std::cout << b.beamId << std::endl;
//		//	for (int j = 0; j < beamNum; j++)
//		//	{
//		//		std::cout << (*beams[j]).beamId << " ";
//		//	}
//		//	exchangePoint(E, o1, diskNorm);
//		//	//break;
//		//}
//
//		if (flag == 1)
//		{
//			beams[i]->tris1Vec = T;
//			beams[i]->end1Vec = E;
//		}
//		else
//		{
//			beams[i]->tris2Vec = T;
//			beams[i]->end2Vec = E;
//		}
//
//	}
//
//	makeFaces(triOnConvex,mp,beams,J);
//}
#ifdef cgal
int getTopology(std::vector<BeamPlugin*> &beams,Junction* &J)
{
	//std::ifstream in((argc > 1) ? argv[1] : CGAL::data_file_path("points_3/half[0]yz"));
	std::vector<Point_3> points;//构建邻接关系的点
	int beamNum = beams.size();
	for (int i = 0; i < beamNum; i++)
	{
		BeamPlugin b = *beams[i];
		Eigen::Vector3d st, ed;
		if (pow(J->position[0] - b.axis.p1[0], 2) + pow(J->position[1] - b.axis.p1[1], 2) + pow(J->position[2] - b.axis.p1[2], 2)
			< pow(J->position[0] - b.axis.p2[0], 2) + pow(J->position[1] - b.axis.p2[1], 2) + pow(J->position[2] - b.axis.p2[2], 2))
		{
			st = b.axis.p1; ed = b.axis.p2;
		}
		else
		{
			st = b.axis.p2; ed = b.axis.p1;
		}
		points.push_back(Point_3(st[0],st[1],st[2]));
	}
	Mesh m;
	CGAL::convex_hull_3(points.begin(), points.end(), m);
	std::vector<BeamPlugin*> beamsTmp;

	points.clear();
	for (auto it = m.vertices_begin(); it != m.vertices_end(); it++)
	{
		//std::cout << m.point(*it)[0] << " " << m.point(*it)[1] << " " << m.point(*it)[2] << std::endl;
		points.push_back(m.point(*it));
		Eigen::Vector3d pointNow(points.back()[0], points.back()[1], points.back()[2]);
		for (int i = 0; i < beamNum; i++)
		{
			BeamPlugin* b = beams[i];
			Eigen::Vector3d st, ed;
			if (pow(J->position[0] - b->axis.p1[0], 2) + pow(J->position[1] - b->axis.p1[1], 2) + pow(J->position[2] - b->axis.p1[2], 2)
				< pow(J->position[0] - b->axis.p2[0], 2) + pow(J->position[1] - b->axis.p2[1], 2) + pow(J->position[2] - b->axis.p2[2], 2))
			{
				st = b->axis.p1; ed = b->axis.p2;
			}
			else
			{
				st = b->axis.p2; ed = b->axis.p1;
			}
			if (sqrt(pow(pointNow[0] - st[0], 2) + pow(pointNow[1] - st[1], 2) + pow(pointNow[2] - st[2], 2) < 1e-6))
			{
				beamsTmp.push_back(b);
				break;
			}
		}
	}
	beams = beamsTmp;
	/*std::copy(std::istream_iterator<Point_3>(in),
		std::istream_iterator<Point_3>(),
		std::back_inserter(points));*/
	/*Construct construct(m, points.begin(), points.end());
	CGAL::advancing_front_surface_reconstruction(points.begin(),
		points.end(),
		construct);
	std::ofstream fout("D:\\MutiresolutionLattice\\Results\\m.off", std::ios::out);
	fout << m;
	fout.close();

	int faceNum = 4;
	if (points.size() > 4)
		faceNum += 2 * (points.size() - 4);

	std::vector<vertex_descriptor> v;
	for (auto it = m.vertices_begin(); it != m.vertices_end(); it++)
	{
		v.push_back(*it);
	}
	std::cout << m.number_of_faces() << std::endl;
	if(m.number_of_faces() < faceNum)
	{
		
		for (int i = 0; i < v.size() - 2; i++)
		{
			for (int j = i + 1; j < v.size() - 1; j++)
			{
				for (int k = j + 1; k < v.size(); k++)
				{
					m.add_face(v[i],v[j],v[k]);
					
					std::cout << m.number_of_faces() << std::endl;
				}
			}
		}		
		for (int i = v.size()-1; i >1 ; i--)
		{
			for (int j = i - 1; j > 0; j--)
			{
				for (int k = j - 1; k > -1; k--)
				{
					m.add_face(v[i], v[j], v[k]);

					std::cout << m.number_of_faces() << std::endl;
				}
			}
		}
	}*/

	//std::cout << points[3] << std::endl;
	Triangle* triList; int triId = 0,quadNum=0;
	QuadFace* quadList;
	std::vector<face_descriptor> f;
	std::vector<Eigen::Vector3i> triOnConvex;
	std::map<int, Eigen::Vector3d> mp;
	for (auto it = m.faces_begin();it!=m.faces_end();it++)
	{
		Eigen::Vector3i triWithId;
		std::vector<int> vIndex;
		Eigen::Vector3d v[3];
		BeamPlugin *b[3]; int cnt = 0;
		f.push_back(*it);
		//std::cout << "vertices around face " << *it << std::endl;
		CGAL::Vertex_around_face_iterator<Mesh> vbegin, vend;
		for (boost::tie(vbegin, vend) = vertices_around_face(m.halfedge(*it), m);
			vbegin != vend;
			++vbegin) {
			//std::cout <<*vbegin<< vbegin->idx() << " ";
			b[cnt] = beams[vbegin->idx()];
			triWithId[cnt] = b[cnt]->beamId;

			v[cnt][0] = points[vbegin->idx()][0];
			v[cnt][1] = points[vbegin->idx()][1];
			v[cnt][2] = points[vbegin->idx()][2];
			mp[triWithId[cnt]] = v[cnt];

			cnt++;
			vIndex.push_back(vbegin->idx());
		}
		triOnConvex.push_back(triWithId);
		//getTriangle(b,J);
		Face ftmp;
		ftmp.p1 = v[0];
		ftmp.p2 = v[1];
		ftmp.p3 = v[2];
		Eigen::Vector3d v1 = ftmp.p1;
		Eigen::Vector3d v2 = ftmp.p2;
		Eigen::Vector3d v3 = ftmp.p3;
		ftmp.normal[0] = (v2[1] - v1[1])*(v3[2] - v1[2]) - (v2[2] - v1[2])*(v3[1] - v1[1]);
		ftmp.normal[1] = (v2[2] - v1[2])*(v3[0] - v1[0]) - (v2[0] - v1[0])*(v3[2] - v1[2]);
		ftmp.normal[2] = (v2[0] - v1[0])*(v3[1] - v1[1]) - (v2[1] - v1[1])*(v3[0] - v1[0]);
		//int otherIndex;
		//for (int i = 0; i < points.size(); i++)
		//{
		//	if (find(vIndex.begin(), vIndex.end(), i) == vIndex.end())
		//	{
		//		otherIndex = i;
		//		break;
		//	}
		//}
		//Eigen::Vector3d otherVec(points[otherIndex][0]-points[vIndex[0]][0], 
		//	points[otherIndex][1] - points[vIndex[0]][1], 
		//	points[otherIndex][2] - points[vIndex[0]][2]);
		//
		//if (otherVec.dot(ftmp.normal) > 0)//法向朝内
		//{
		//	BeamPlugin *btmp = b[0];
		//	b[0] = b[1];
		//	b[1] = btmp;
		//	Eigen::Vector3d tmp = ftmp.p1;
		//	ftmp.p1 = ftmp.p2;
		//	ftmp.p2 = tmp;
		//	Eigen::Vector3d v1 = ftmp.p1;
		//	Eigen::Vector3d v2 = ftmp.p2;
		//	Eigen::Vector3d v3 = ftmp.p3;
		//	ftmp.normal[0] = (v2[1] - v1[1])*(v3[2] - v1[2]) - (v2[2] - v1[2])*(v3[1] - v1[1]);
		//	ftmp.normal[1] = (v2[2] - v1[2])*(v3[0] - v1[0]) - (v2[0] - v1[0])*(v3[2] - v1[2]);
		//	ftmp.normal[2] = (v2[0] - v1[0])*(v3[1] - v1[1]) - (v2[1] - v1[1])*(v3[0] - v1[0]);
		//}
		getTriangle(b, J,triList, triId,mp);
		triId++;
		//J->f.push_back(ftmp);
		//std::cout <<std::endl;
	}
	//findAdjTriangles(triList);
	//measureDistance(triList);
	//GreyWolfOptimizerTest(30);
	quadNum = 6 + 3 * ((triId - 4) / 2);
	quadList = (QuadFace*)malloc(quadNum * sizeof(QuadFace));
	GreyWolfOptimizer(triId * 3, triList, quadList,triId, quadNum);
	//MultiObjectiveGreyWolfOptimizer(triList.size()*3,2,triList,quadList);
	generateFaces(triList,triId,quadList,quadNum, J);
	//checkIntersection_makeFaces(triList,triOnConvex,mp,beams,J);
	/*vertex_descriptor V = *(m.vertices_begin());
	std::cout << "vertices around vertex " << V << std::endl;
	CGAL::Vertex_around_target_circulator<Mesh> vbegin(m.halfedge(V), m), done(vbegin);
	do {
		std::cout << *vbegin++ << std::endl;
	} while (vbegin != done);*/
	
	return 0;
}
#endif
using namespace dt;
void locateArcs(std::vector<BeamPlugin*> &beams, Triangle* triList,QuadFace* quadList, ArcType* arcList, short* flag,int nodeid)
{
	int beamNum = beams.size();
	int triNum = 8 + (beamNum - 6) * 2;
	int quadNum= 6 + 3 * ((triNum - 4) / 2);
	for (int i = 0; i < triNum; i++)
	{
		triList[i].adjTriNum = 0;
		triList[i].adjQuadNum = 0;
		triList[i].nodeid = nodeid;
	}
	for (int i = 0; i < quadNum; i++)
	{
		quadList[i].nodeid = nodeid;
	}
	findAdjTriangles(triList, triNum); bool tmpinverted = false;
	findQuadFaces(triList, quadList, triNum, tmpinverted);
	int arcId=0;
	bool f = false;
	for (int i = 0; i < beamNum; i++)
	{
		BeamPlugin* b = beams[i];
		int idx = flag[i] - 1;
		b->nodeid[idx] = nodeid;
		for (int j = 0; j < b->arcNum[idx]; j++)
		{
			Triangle* tri = triList + b->arcTriList[idx][j];
			Triangle* tri2 = triList + b->arcTriList[idx][mod(j + 1, b->arcNum[idx])];
			bool locateArc = false;
			//找到两个相邻三角形中间的quadFace
			for (int k = 0; k < 3; k++)
			{
				for (int v = 0; v < 3; v++)
				{
					if (tri->adjQuadFaces[k]->quadId == tri2->adjQuadFaces[v]->quadId)
					{
						locateArc = true;
						f = true;
						QuadFace* q = quadList + tri->adjQuadFaces[k]->quadId;
						ArcType* arc = &arcList[arcId++];
						if (q->quadWithBeamid[0] == i)
						{
							arc->st = q->t[1].p[0];
							arc->ed = q->t[1].p[2];
							q->arc[0] = arc;
						}
						else
						{
							arc->st = q->t[0].p[2];
							arc->ed = q->t[0].p[1];
							q->arc[1] = arc;
						}
						arc->o = b->axis.p[idx];
						float3 direct = Subtraction(b->axis.p[idx], b->axis.p[mod(idx + 1, 2)]);
						float3 e1 = Subtraction(arc->st, arc->o);
						e1 = make_norm(e1);
						float3 e2 = Subtraction(arc->ed, arc->o);
						e2 = make_norm(e2);
						float3 arcNorm = cross(e1, e2);
						arcNorm = make_norm(arcNorm);
						if (Dot(arcNorm, direct) > 0)
						{
							arc->theta = 2 * PI - acos(min(max(Dot(e1, e2), -1.0f), 1.0f));
							arc->diskNorm = make_float3(-arcNorm.x, -arcNorm.y, -arcNorm.z);
						}
						else
						{
							arc->theta = acos(min(max(Dot(e1, e2), -1.0f), 1.0f));
							arc->diskNorm = arcNorm;
						}
						b->arcArray[idx][j] = arc;
						break;
					}
				}
				if (f)
				{
					f = false;
					break;
				}
			}
			if (locateArc == false)
				printf("locate arc failing!\n");
		}
		
	}
}
void setSegNum(std::vector<BeamPlugin*> &beams,float chordError,short* flag,int& totalSampleNum,int& totalArcSampleNum)
{
	int beamNum = beams.size();
	for (int i = 0; i < beamNum; i++)
	{
		BeamPlugin* b = beams[i];
		float r = b->radius;
		float segTheta = 2 * acos(1 - chordError / r);
		short idx = flag[i] - 1;
		for (int j = 0; j < b->arcNum[idx]; j++)
		{
			ArcType* arc = b->arcArray[idx][j];
			arc->sampleNum = ceil(arc->theta / segTheta)+1;
			arc->segTheta = arc->theta / (arc->sampleNum-1);
			totalSampleNum += (arc->sampleNum-1);
			totalArcSampleNum += arc->sampleNum;
		}
		totalSampleNum++;
	}
}
int getTopology(std::vector<BeamPlugin*> &beams, Junction* J,Triangle* triList,QuadFace* quadList,short* flag,float* longestLength)
{
	int beamNum = beams.size();
	//vector<Vector3D*> dots;
	Point** dots = (Point**)malloc(beamNum * sizeof(Point*));
	float projRadius = 1;
	std::vector<Point_3> points;//构建邻接关系的点
	cout << J->nodeId << endl;
	//short* flag = (short*)malloc(beamNum * sizeof(short));
	bool* visitedDot = (bool*)malloc(beamNum * sizeof(bool));
	for (int i = 0; i < beamNum; i++)
	{
		visitedDot[i] = false;
		BeamPlugin b = *beams[i];
		float3 st, ed;
		if (pow(J->position.x - b.axis.p[0].x, 2) + pow(J->position.y - b.axis.p[0].y, 2) + pow(J->position.z - b.axis.p[0].z, 2)
			< pow(J->position.x - b.axis.p[1].x, 2) + pow(J->position.y - b.axis.p[1].y, 2) + pow(J->position.z - b.axis.p[1].z, 2))
		{
			st = b.axis.p[0]; ed = b.axis.p[1]; flag[i] = 1;
		}
		else
		{
			st = b.axis.p[1]; ed = b.axis.p[0]; flag[i] = 2;
		}
		points.push_back(Point_3(st.x, st.y, st.z));

		Point* dot = (Point*)malloc(sizeof(Point));
		dot->x = st.x - J->position.x; dot->y = st.y - J->position.y; dot->z = st.z - J->position.z;
		//float3 proj = make_float3(st.x-J->position.x, st.y- J->position.y, st.z- J->position.z);
		//projRadius += Norm(proj);
		//Vector3D* dot = new Vector3D(proj.x, proj.y, proj.z, (uint8_t)255, (uint8_t)248, (uint8_t)220);
		float length = sqrt(pow(dot->x, 2) + pow(dot->y, 2) + pow(dot->z, 2));
		float scaleFactor = projRadius / length;

		dot->x = scaleFactor * dot->x;
		dot->y = scaleFactor * dot->y;
		dot->z = scaleFactor * dot->z;

		dot->id = i;
		dot->isVisited = false;
		dots[i] = (dot);
	}
	//projRadius = projRadius / beamNum; 
	//projRadius = 1;
	/*DelaunayTriangulation triangulation = DelaunayTriangulation(projRadius);
	vector<tuple<int, int, int>*> mesh = triangulation.GetTriangulationResult(dots,projRadius);*/
	int nFaces = 8 + (beamNum - 6) * 2;
	Triangle** Mesh = (Triangle**)malloc(nFaces * sizeof(Triangle*));
	BuildInitialHull(dots, beamNum, Mesh);

	int currentMeshNum = 4;
	for (int i = 0; i < beamNum; i++)
	{
		Point* dot = dots[i];
		if (!dot->isVisited)
		{
			InsertDot(dot, Mesh, currentMeshNum);
			currentMeshNum += 2;
		}
		//free(dot);
	}
	for (int i = 0; i < beamNum; i++)
	{
		free(dots[i]);
	}
	free(dots);

	for (int i = 0; i < nFaces; i++)
	{
		Mesh[i]->triId = i;
	}
	//建立每个dot上的三角形id链
	for (int i = 0; i < nFaces; i++)
	{
		int startid = Mesh[i]->triId;
		
		for (int j = 0; j < 3; j++)
		{
			int beamid = Mesh[i]->triWithBeamid[j];
			if (visitedDot[beamid])
				continue;
			if (flag[beamid] == 1)
			{
				beams[beamid]->arcTriList[0][beams[beamid]->arcNum[0]] = Mesh[i]->triId;
				beams[beamid]->arcNum[0]++;
			}
			else
			{
				beams[beamid]->arcTriList[1][beams[beamid]->arcNum[1]] = Mesh[i]->triId;
				beams[beamid]->arcNum[1]++;
			}
			int nextid = Mesh[i]->adjTriangles[mod(j + 2, 3)]->triId;
			while (nextid != startid)
			{
				if (flag[beamid] == 1)
				{
					beams[beamid]->arcTriList[0][beams[beamid]->arcNum[0]] = nextid;
					beams[beamid]->arcNum[0]++;
				}
				else
				{
					beams[beamid]->arcTriList[1][beams[beamid]->arcNum[1]] = nextid;
					beams[beamid]->arcNum[1]++;
				}
				for (int v = 0; v < 3; v++)
				{
					if (Mesh[nextid]->triWithBeamid[v] == beamid)
					{
						nextid = Mesh[nextid]->adjTriangles[mod(v + 2, 3)]->triId;
						break;
					}
				}
			}
			visitedDot[beamid] = true;
		}
		
	}
	/*int* faceIndices = NULL;
	int nFaces;
	convhull_3d_build(vertices, beamNum, &faceIndices, &nFaces);
	char* OUTPUT_OBJ_FILE_NAME = (char*)("D:\\MutiresolutionLattice\\Results\\testConvexHull");
	convhull_3d_export_obj(vertices, beamNum, faceIndices, nFaces, 1, OUTPUT_OBJ_FILE_NAME);*/

	/*Mesh m;
	CGAL::convex_hull_3(points.begin(), points.end(), m);
	std::ofstream fout("D:\\MutiresolutionLattice\\Results\\m.off", std::ios::out);
	fout << m;
	fout.close();*/

	//int nFaces = 8 + (beamNum - 6) * 2;
	//Triangle* triList = (Triangle*)malloc(nFaces*sizeof(Triangle)); 
	int triId = 0, quadNum = 0;
	//QuadFace* quadList;
	//std::map<int, Eigen::Vector3d> mp;
	//J->f = (Face*)malloc(nFaces * sizeof(Face));
	J->triNum = nFaces;
	for (int i = 0; i < nFaces; i++)
	{
		Eigen::Vector3i triWithId;
		float3 v[3];
		BeamPlugin *b[3]; int cnt = 0;
		Triangle* mesh = Mesh[i];

		vector<int> vIndex;
		vIndex.resize(3);

		//tuple<int, int, int>* tri = mesh[i]; int id[3];
		//id[0] = std::get<0>(*tri), id[1] = std::get<1>(*tri), id[2] = std::get<2>(*tri);

		for (int j = 0; j < 3; j++)
		{
			triList[i].triWithBeamid[j] = mesh->triWithBeamid[j];
			/*int idx = faceIndices[i * 3 + j];
			vIndex[cnt] = idx;*/
			int idx = mesh->triWithBeamid[j];
			b[cnt] = beams[idx];
			triWithId[cnt] = b[cnt]->beamId;

			v[cnt].x = points[idx][0];
			v[cnt].y = points[idx][1];
			v[cnt].z = points[idx][2];
			//mp[triWithId[cnt]] = v[cnt];

			cnt++;
		}
		//int otheridx;
		/*for (int j = 0; j < beamNum; j++)
		{
			if (find(vIndex.begin(), vIndex.end(), j) == vIndex.end())
			{
				int otheridx = j; 
				int curridx = faceIndices[i * 3];
				float3 otherVec = make_float3(vertices[otheridx].x - vertices[curridx].x,
					vertices[otheridx].y - vertices[curridx].y, vertices[otheridx].z - vertices[curridx].z);

				float3 e1 = make_float3(v[1].x - v[0].x, v[1].y - v[0].y, v[1].z - v[0].z);
				float3 e2 = make_float3(v[2].x - v[0].x, v[2].y - v[0].y, v[2].z - v[0].z);
				float3 faceNormal = cross(e1, e2);
				if (Dot(otherVec, faceNormal) > 0)
				{
					BeamPlugin *btmp = b[0];
					b[0] = b[1];
					b[1] = btmp;
					float3 vtmp = v[0];
					v[0] = v[1];
					v[1] = vtmp;
				}
				else if (Dot(otherVec, faceNormal) == 0)
					continue;
				else
					break;
				break;
			}
		}*/
		/*Face ftmp;
		ftmp.p1 = v[0];
		ftmp.p2 = v[1];
		ftmp.p3 = v[2];
		ftmp.normal = cross(Subtraction(v[1], v[0]), Subtraction(v[2], v[0]));
		ftmp.normal = make_norm(ftmp.normal);
		J->f[triId] = ftmp;*/

		triList[i].o[0] = v[0];
		triList[i].o[1] = v[1];
		triList[i].o[2] = v[2];

		getTriangle(b, J, triList, triId);
		triId++;

	}
	for (int i = 0; i < nFaces; i++)
	{
		free(Mesh[i]);
	}
	free(Mesh);

	quadNum = 6 + 3 * ((triId - 4) / 2);
	//quadList = (QuadFace*)malloc(quadNum * sizeof(QuadFace));
	shapeOptimization(beams, J->position, triList, triId, flag,longestLength);
	//GreyWolfOptimizer(triId * 3, triList, quadList, triId, quadNum);
	//MultiObjectiveGreyWolfOptimizer(triId*3,2,triList,quadList,triId,quadNum);
	//Arc* arcList = (Arc*)malloc(2 * quadNum * sizeof(Arc));
	/*locateArcs(beams, triList,quadList,flag,arcList);
	setSegNum(beams, chordError, flag);*/
	//generateFaces(triList, triId, quadList, quadNum, J);

	//free(vertices);
	/*for (int i = 0; i < nFaces; i++)
	{
		free(triList[i].adjTriangles);
		free(triList[i].adjQuadFaces);
	}*/
	/*free(triList);
	free(quadList);*/

	return 0;
}