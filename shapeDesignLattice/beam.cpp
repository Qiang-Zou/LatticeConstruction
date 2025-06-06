#include<cstdio>
#include<iostream>
#include<vector>
#include<cmath>
#include<Eigen/Dense>
#include <Eigen/Geometry>
#include"beam.h"
#include "MOGWO.h"

using namespace Eigen;
extern vector<Face> faces;
extern vector<Face> allFaces;
//BeamPlugin::BeamPlugin():radius(1),length(6),samples(60)
//{
//	beamId = -1;
//}

//BeamPlugin::~BeamPlugin()
//{
//}
float** BeamPlugin::getMatrix()
{
	axis.middle.x = (axis.p[0].x + axis.p[1].x) / 2;
	axis.middle.y = (axis.p[0].y + axis.p[1].y) / 2;
	axis.middle.z = (axis.p[0].z + axis.p[1].z) / 2;

	float3 xAxis = make_float3(axis.p[0].x - axis.p[1].x, axis.p[0].y - axis.p[1].y, axis.p[0].z - axis.p[1].z);
	xAxis = make_norm(xAxis); //X轴
	double A, B, C, D; //定义平面
	A = xAxis.x; B = xAxis.y; C = xAxis.z;
	D = -(A*axis.middle.x + B * axis.middle.y + C * axis.middle.z);
	//std::cout<<A<<" "<<B<<" "<<C<<" "<<D<<std::endl;
	float3 ppla = make_float3(0, 0, 0);
	if (A != 0)
	{
		ppla.y = axis.middle.y + 1;
		ppla.z = axis.middle.z + 1;
		ppla.x = (-D-B*ppla.y-C*ppla.z) / A;
	}
	else if (B != 0)
	{
		ppla.x = axis.middle.x + 1;
		ppla.z = axis.middle.z + 1;
		ppla.y = (-D-A*ppla.x-C*ppla.z) / B;
	}
	else if (C != 0)
	{
		ppla.x = axis.middle.x + 1;
		ppla.y = axis.middle.y + 1;
		ppla.z = (-D-A*ppla.x-B*ppla.y) / C;
	}
	float3 yAxis = make_float3(ppla.x - axis.middle.x, ppla.y - axis.middle.y, ppla.z - axis.middle.z);
	yAxis = make_norm(yAxis);  //Y轴
	float3 zAxis = cross(xAxis,yAxis);
	zAxis = make_norm(zAxis);  //Z轴

	float** m = (float**)malloc(4 * sizeof(float*));
	for (int i = 0; i < 4; i++)
	{
		m[i] = (float*)malloc(4 * sizeof(float));
	}
	m[0][0] = xAxis.x; m[0][1] = yAxis.x; m[0][2] = zAxis.x; m[0][3] = axis.middle.x;
	m[1][0] = xAxis.y; m[1][1] = yAxis.y; m[1][2] = zAxis.y; m[1][3] = axis.middle.y;
	m[2][0] = xAxis.z; m[2][1] = yAxis.z; m[2][2] = zAxis.z; m[2][3] = axis.middle.z;
	m[3][0] = 0;       m[3][1] = 0;       m[3][2] = 0;       m[3][3] = 1;
	//Matrix3d rotate;
	//rotate << xAxis, yAxis, zAxis;
	//Vector3d transl(, , );
	//MatrixXd ll(1, 4);
	//ll << 0, 0, 0, 1;
	////仿射变换矩阵
	//MatrixXd m(4, 4);
	//m << rotate, transl, ll;

	return m;
}

void BeamPlugin::BeamTransform()
{
	double step = 2 * acos(-1) / samples;
	std::vector<float3> C1;
	std::vector<float3> C2;
	for (int i = 0; i < samples; i++)
	{
		float3 p;
		p.x = -length / 2;
		p.y = radius * sin(i*step);
		p.z = radius * cos(i*step);
		C1.push_back(p);
	}
	for (int i = 0; i < samples; i++)
	{
		float3 p;
		p.x = length / 2;
		p.y = radius * sin(i*step);
		p.z = radius * cos(i*step);
		C2.push_back(p);
	}
	//faces
	for (int i = 0; i < samples; i++)
	{
		Face tmp;
		tmp.p1 = C1[i];
		tmp.p2 = C2[i];
		tmp.p3 = C2[(i + 1) % samples];
		float3 v1 = tmp.p1;
		float3 v2 = tmp.p2;
		float3 v3 = tmp.p3;
		tmp.normal.x = (v2.y - v1.y)*(v3.z - v1.z) - (v2.z - v1.z)*(v3.y - v1.y);
		tmp.normal.y = (v2.z - v1.z)*(v3.x - v1.x) - (v2.x - v1.x)*(v3.z - v1.z);
		tmp.normal.z = (v2.x - v1.x)*(v3.y - v1.y) - (v2.y - v1.y)*(v3.x - v1.x);
		//f.push_back(tmp);
		tmp.inverted = 0;
		faces.push_back(tmp);

		tmp.p1 = C1[i];
		tmp.p2 = C2[(i + 1) % samples];
		tmp.p3 = C1[(i + 1) % samples];
		v1 = tmp.p1;
		v2 = tmp.p2;
		v3 = tmp.p3;
		tmp.normal.x = (v2.y - v1.y)*(v3.z - v1.z) - (v2.z - v1.z)*(v3.y - v1.y);
		tmp.normal.y = (v2.z - v1.z)*(v3.x - v1.x) - (v2.x - v1.x)*(v3.z - v1.z);
		tmp.normal.z = (v2.x - v1.x)*(v3.y - v1.y) - (v2.y - v1.y)*(v3.x - v1.x);
		//f.push_back(tmp);
		faces.push_back(tmp);
	}

	

	float** m = getMatrix();
	//std::cout << m << std::endl;
	//scaling
	//double s = (sqrt(pow(axis.p1.x - axis.p2.x, 2) + pow(axis.p1.y - axis.p2.y, 2) + pow(axis.p1.z - axis.p2.z, 2))) / length;

	//for (int i = 0; i < f.size(); i++)
	for (int i = 2 * samples; i >0 ; i--)
	{
		Face &f = *(faces.end() - i);
		float3 p1 = f.p1;
		float3 p2 = f.p2;
		float3 p3 = f.p3;

		float4 np1; np1 = make_float4(p1.x, p1.y, p1.z, 1);
		np1 = Multiply(m , np1);
		float4 np2; np2 = make_float4(p2.x, p2.y, p2.z, 1);
		np2 = Multiply(m , np2);
		float4 np3; np3 = make_float4(p3.x, p3.y, p3.z, 1);
		np3 = Multiply(m , np3);
		//new points np

		//scaling之后重新计算法向
		float3 n;
		n.x = (np2.y - np1.y)*(np3.z - np1.z) - (np2.z - np1.z)*(np3.y - np1.y);
		n.y = (np2.z - np1.z)*(np3.x - np1.x) - (np2.x - np1.x)*(np3.z - np1.z);
		n.z = (np2.x - np1.x)*(np3.y - np1.y) - (np2.y - np1.y)*(np3.x - np1.x);

		Face ftmp;
		ftmp.p1.x = np1.x; ftmp.p1.y = np1.y; ftmp.p1.z = np1.z;
		ftmp.p2.x = np2.x; ftmp.p2.y = np2.y; ftmp.p2.z = np2.z;
		ftmp.p3.x = np3.x; ftmp.p3.y = np3.y; ftmp.p3.z = np3.z;
		ftmp.normal = n;

		f = ftmp;
		//f[i] = ftmp;
		//faces.push_back(ftmp);
	}
	
}

bool BeamPlugin::BeamConstruct(Edge etmp,double r)
{
	samples = 60;
	axis = etmp;
	radius = r;
	length = sqrt(pow(axis.p[0].x - axis.p[1].x, 2) + pow(axis.p[0].y - axis.p[1].y, 2) + pow(axis.p[0].z - axis.p[1].z, 2));
	originLength = length;
    

    return true;
}

Junction::Junction()
{
	nodeId = -1;
}

Junction::~Junction()
{

}

//bool Junction::JunctionConstruct()
//{
//	return true;
//}
//__device__ Triangle::Triangle()
//{
//
//}
//__device__ Triangle::Triangle(const Triangle &T)
//{
//	triId = T.triId;
//	normal = T.normal;
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
//	cost = T.cost;
//}
//Triangle::~Triangle()
//{
//	/*int adjqNum = this->adjQuadFaces.size();
//	for (int i = 0; i < adjqNum; i++)
//	{
//		if (this->adjQuadFaces[i] != NULL)
//		{
//			delete this->adjQuadFaces[i];
//			this->adjQuadFaces[i] = NULL;
//		}
//	}*/
//	/*if(this->adjTriNum>0)
//		delete[] this->adjTriangles;
//	if(this->adjQuadNum>0)
//		delete[] this->adjQuadFaces;*/
//}
//QuadFace::QuadFace()
//{
//
//}
//QuadFace::~QuadFace()
//{
//
//}
