#include <regex>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include "../Header/Triangulation.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../MOGWO.h"
#include<stack>
using namespace std;
using namespace dt;

DelaunayTriangulation::DelaunayTriangulation(float projRadius)
{
    for (int i = 0; i < INIT_VERTICES_COUNT; i++)
    {
        _AuxiliaryDots[i] = new Vector3D(
            (i % 2 == 0 ? 1 : -1) * (i / 2 == 0 ? projRadius : 0),
            (i % 2 == 0 ? 1 : -1) * (i / 2 == 1 ? projRadius : 0),
            (i % 2 == 0 ? 1 : -1) * (i / 2 == 2 ? projRadius : 0),
            true, 0, 0, 0
        );
    }

    _ProjectedDots = new vector<Vector3D*>();
    _Mesh = new vector<triangle*>();

    for (int i = 0; i < sizeof(_Statistics) / sizeof(long); i++)
    {
        _Statistics[i] = 0;
    }
}

DelaunayTriangulation::~DelaunayTriangulation()
{
    for (int i = 0; i < INIT_VERTICES_COUNT; i++)
    {
        delete _AuxiliaryDots[i];
    }

    vector<Vector3D*>::iterator itDots;
    for (itDots = _ProjectedDots->begin(); itDots != _ProjectedDots->end(); itDots++)
    {
        delete *itDots;
    }

    vector<triangle*>::iterator itMesh;
    for (itMesh = _Mesh->begin(); itMesh != _Mesh->end(); itMesh++)
    {
        delete *itMesh;
    }

    delete _ProjectedDots;
    delete _Mesh;
}
void BuildInitialHull(Point** dots, int pointNum, Triangle** Mesh)
{
	Point* initialVertices[4];
	Triangle* initialHullFaces[4];
	float3 P[4];
	for (int i = 0; i < 3; i++)
	{
		initialVertices[i] = *(dots + i);
		P[i] = make_float3((*(dots + i))->x, (*(dots + i))->y, (*(dots + i))->z);
		(*(dots + i))->isVisited = true;
	}
	float3 n_circumcenter = ComputeNegativeCircumcenter(P);

	initialVertices[3] = *(dots + 3);
	P[3] = make_float3((*(dots + 3))->x, (*(dots + 3))->y, (*(dots + 3))->z);
	float dis = Norm(Subtraction(P[3], n_circumcenter));
	//找到距离-C0最近的点
	for (int i = 4; i < pointNum; i++)
	{
		float3 Pk = make_float3(dots[i]->x, dots[i]->y, dots[i]->z);
		//判断四点是否共面，若共面，continue
		float3 P0toPk = Subtraction(Pk, P[0]);
		float3 normal = cross(Subtraction(P[1], P[0]), Subtraction(P[2], P[0]));
		if (Dot(normal, P0toPk) == 0)
			continue;
		float tmpDis = Norm(Subtraction(Pk, n_circumcenter));
		if (tmpDis < dis)
		{
			initialVertices[3] = dots[i];
			P[3] = Pk;
			dis = tmpDis;
		}
	}
	//printf("%d\n", initialVertices[3]->id);
	//确定三角形法线朝向与半球位置一致
	float3 normal = cross(Subtraction(P[1], P[0]), Subtraction(P[2], P[0]));
	float3 CtoP = P[0];
	if (Dot(normal, CtoP) < 0)
	{
		Point* vtmp = initialVertices[0];
		initialVertices[0] = initialVertices[1];
		initialVertices[1] = vtmp;

		float3 ptmp = P[0];
		P[0] = P[1];
		P[1] = ptmp;
		normal = cross(Subtraction(P[1], P[0]), Subtraction(P[2], P[0]));
	}
	//根据Pk确定三角形法线朝向
	float3 P0toPk = Subtraction(P[3], P[0]);
	if (Dot(normal, P0toPk) > 0)
	{
		Point* vtmp = initialVertices[0];
		initialVertices[0] = initialVertices[1];
		initialVertices[1] = vtmp;

		float3 ptmp = P[0];
		P[0] = P[1];
		P[1] = ptmp;
		normal = cross(Subtraction(P[1], P[0]), Subtraction(P[2], P[0]));
	}

	//建立四面体
	int vertex0Index[] = { 0, 0, 0, 1 };
	int vertex1Index[] = { 1, 3, 2, 3 };
	int vertex2Index[] = { 2, 1, 3, 2 };
	//int triId[4][3];
	for (int i = 0; i < 4; i++)
	{
		Point* v0 = initialVertices[vertex0Index[i]];
		Point* v1 = initialVertices[vertex1Index[i]];
		Point* v2 = initialVertices[vertex2Index[i]];

		Triangle* tri = (Triangle*)malloc(sizeof(Triangle));
		tri->p[0] = make_float3(v0->x, v0->y, v0->z);
		tri->p[1] = make_float3(v1->x, v1->y, v1->z);
		tri->p[2] = make_float3(v2->x, v2->y, v2->z);
		tri->triWithBeamid[0] = v0->id;
		tri->triWithBeamid[1] = v1->id;
		tri->triWithBeamid[2] = v2->id;
		//tri->adjTriangles = (Triangle*)malloc(3 * sizeof(Triangle));

		initialHullFaces[i] = tri;
		/*for (int j = 0; j < 3; j++)
		{
			triId[i][j] = initialHullFaces[i]->Vertex[j]->Id;
		}*/
		//cnt++;
		//initialHullFaces[i] = tri;

		Mesh[i] = tri;
	}

	int neighbor0Index[] = { 1, 2, 0, 1 };
	int neighbor1Index[] = { 3, 3, 3, 2 };
	int neighbor2Index[] = { 2, 0, 1, 0 };
	for (int i = 0; i < 4; i++)
	{
		Triangle* n0 = initialHullFaces[neighbor0Index[i]];
		Triangle* n1 = initialHullFaces[neighbor1Index[i]];
		Triangle* n2 = initialHullFaces[neighbor2Index[i]];
		initialHullFaces[i]->adjTriangles[0] = n0;
		initialHullFaces[i]->adjTriangles[1] = n1;
		initialHullFaces[i]->adjTriangles[2] = n2;
		//initialHullFaces[i]->AssignNeighbors(n0, n1, n2);
	}

	// dot already in the mesh, avoid being visited by InsertDot() again
	for (int i = 0; i < 4; i++)
	{
		initialVertices[i]->isVisited = true;
	}
}
float GetDeterminant(float matrix[])
{
	// inversed for left handed coordinate system
	float determinant = matrix[2] * matrix[4] * matrix[6]
		+ matrix[0] * matrix[5] * matrix[7]
		+ matrix[1] * matrix[3] * matrix[8]
		- matrix[0] * matrix[4] * matrix[8]
		- matrix[1] * matrix[5] * matrix[6]
		- matrix[2] * matrix[3] * matrix[7];

	// adjust result based on float number accuracy, otherwise causing deadloop
	return abs(determinant) <= DBL_EPSILON ? 0 : determinant;
}
float GetDeterminant(float3 v0, float3 v1, float3 v2)
{
	float matrix[] = {
		v0.x, v0.y, v0.z,
		v1.x, v1.y, v1.z,
		v2.x, v2.y, v2.z
	};

	return GetDeterminant(matrix);
}
void FixNeighborhood(Triangle* target, Triangle* oldNeighbor, Triangle* newNeighbor)
{
	for (int i = 0; i < 3; i++)
	{
		if (target->adjTriangles[i] == oldNeighbor)
		{
			target->adjTriangles[i] = newNeighbor;
			break;
		}
	}
}
bool TrySwapDiagonal(Triangle* t0, Triangle* t1);
void DoLocalOptimization(Triangle* t0, Triangle* t1)
{
	//_Statistics[1]++;
	//stack<Triangle*>stk;
	stack<Triangle*>stk0;
	stack<Triangle*>stk1;

	stk0.push(t0); stk1.push(t1);
	bool flag = false;
	while ((!stk0.empty()) && (!stk1.empty()))
	{
		Triangle* t0 = stk0.top();
		Triangle* t1 = stk1.top();
		stk0.pop(); stk1.pop();

		for (int i = 0; i < 3; i++)
		{
			if (t1->triWithBeamid[i] == t0->triWithBeamid[0] ||
				t1->triWithBeamid[i] == t0->triWithBeamid[1] ||
				t1->triWithBeamid[i] == t0->triWithBeamid[2])
			{
				continue;
			}

			float matrix[] = {
				t1->p[i].x - t0->p[0].x,
				t1->p[i].y - t0->p[0].y,
				t1->p[i].z - t0->p[0].z,

				t1->p[i].x - t0->p[1].x,
				t1->p[i].y - t0->p[1].y,
				t1->p[i].z - t0->p[1].z,

				t1->p[i].x - t0->p[2].x,
				t1->p[i].y - t0->p[2].y,
				t1->p[i].z - t0->p[2].z
			};
			//std::cout << GetDeterminant(matrix) << std::endl;
			if (GetDeterminant(matrix) >= 0)
			{
				//flag = true;
				// terminate after optimized
				break;
			}

			/*if (TrySwapDiagonal(t0, t1))
			{
				return;
			}*/
			//bool flag = false;
			for (int j = 0; j < 3; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					if (t0->triWithBeamid[j] != t1->triWithBeamid[0] &&
						t0->triWithBeamid[j] != t1->triWithBeamid[1] &&
						t0->triWithBeamid[j] != t1->triWithBeamid[2] &&
						t1->triWithBeamid[k] != t0->triWithBeamid[0] &&
						t1->triWithBeamid[k] != t0->triWithBeamid[1] &&
						t1->triWithBeamid[k] != t0->triWithBeamid[2])
					{
						t0->triWithBeamid[(j + 2) % 3] = t1->triWithBeamid[k];
						t0->p[(j + 2) % 3] = t1->p[k];
						t1->triWithBeamid[(k + 2) % 3] = t0->triWithBeamid[j];
						t1->p[(k + 2) % 3] = t0->p[j];

						t0->adjTriangles[(j + 1) % 3] = t1->adjTriangles[(k + 2) % 3];
						t1->adjTriangles[(k + 1) % 3] = t0->adjTriangles[(j + 2) % 3];
						t0->adjTriangles[(j + 2) % 3] = t1;
						t1->adjTriangles[(k + 2) % 3] = t0;

						FixNeighborhood(t0->adjTriangles[(j + 1) % 3], t1, t0);
						FixNeighborhood(t1->adjTriangles[(k + 1) % 3], t0, t1);

						/*DoLocalOptimization(t0, t0->adjTriangles[j]);
						DoLocalOptimization(t0, t0->adjTriangles[(j + 1) % 3]);
						DoLocalOptimization(t1, t1->adjTriangles[k]);
						DoLocalOptimization(t1, t1->adjTriangles[(k + 1) % 3]);*/
						stk0.push(t1); stk1.push(t1->adjTriangles[(k + 1) % 3]);
						stk0.push(t1); stk1.push(t1->adjTriangles[k]);
						stk0.push(t0); stk1.push(t0->adjTriangles[(j + 1) % 3]);
						stk0.push(t0); stk1.push(t0->adjTriangles[j]);

						flag = true;
						break;
					}
				}
				if (flag)
				{
					//flag = false;
					break;
				}
			}

			if (flag)
			{
				flag = false;
				break;// return;
			}

		}
		/*if (flag)
		{
			stk0.pop();
			stk1.pop();
			flag = false;
		}*/

	}
	//for (int i = 0; i < 3; i++)
	//{
	//	if (t1->triWithBeamid[i] == t0->triWithBeamid[0] ||
	//		t1->triWithBeamid[i] == t0->triWithBeamid[1] ||
	//		t1->triWithBeamid[i] == t0->triWithBeamid[2])
	//	{
	//		continue;
	//	}

	//	double matrix[] = {
	//		t1->p[i].x - t0->p[0].x,
	//		t1->p[i].y - t0->p[0].y,
	//		t1->p[i].z - t0->p[0].z,

	//		t1->p[i].x - t0->p[1].x,
	//		t1->p[i].y - t0->p[1].y,
	//		t1->p[i].z - t0->p[1].z,

	//		t1->p[i].x - t0->p[2].x,
	//		t1->p[i].y - t0->p[2].y,
	//		t1->p[i].z - t0->p[2].z
	//	};
	//	std::cout << GetDeterminant(matrix) << std::endl;
	//	if (GetDeterminant(matrix) >= 0)
	//	{
	//		// terminate after optimized
	//		break;
	//	}

	//	if (TrySwapDiagonal(t0, t1))
	//	{
	//		return;
	//	}
	/*	bool flag = false;
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				if (t0->triWithBeamid[j] != t1->triWithBeamid[0] &&
					t0->triWithBeamid[j] != t1->triWithBeamid[1] &&
					t0->triWithBeamid[j] != t1->triWithBeamid[2] &&
					t1->triWithBeamid[k] != t0->triWithBeamid[0] &&
					t1->triWithBeamid[k] != t0->triWithBeamid[1] &&
					t1->triWithBeamid[k] != t0->triWithBeamid[2])
				{
					t0->triWithBeamid[(j + 2) % 3] = t1->triWithBeamid[k];
					t0->p[(j + 2) % 3] = t1->p[k];
					t1->triWithBeamid[(k + 2) % 3] = t0->triWithBeamid[j];
					t1->p[(k + 2) % 3] = t0->p[j];

					t0->adjTriangles[(j + 1) % 3] = t1->adjTriangles[(k + 2) % 3];
					t1->adjTriangles[(k + 1) % 3] = t0->adjTriangles[(j + 2) % 3];
					t0->adjTriangles[(j + 2) % 3] = t1;
					t1->adjTriangles[(k + 2) % 3] = t0;

					FixNeighborhood(t0->adjTriangles[(j + 1) % 3], t1, t0);
					FixNeighborhood(t1->adjTriangles[(k + 1) % 3], t0, t1);

					DoLocalOptimization(t0, t0->adjTriangles[j]);
					DoLocalOptimization(t0, t0->adjTriangles[(j + 1) % 3]);
					DoLocalOptimization(t1, t1->adjTriangles[k]);
					DoLocalOptimization(t1, t1->adjTriangles[(k + 1) % 3]);

					flag = true;
				}
			}
		}

		if (flag)
		{
			return;
		}

	}*/
}
bool TrySwapDiagonal(Triangle* t0, Triangle* t1)
{
	for (int j = 0; j < 3; j++)
	{
		for (int k = 0; k < 3; k++)
		{
			if (t0->triWithBeamid[j] != t1->triWithBeamid[0] &&
				t0->triWithBeamid[j] != t1->triWithBeamid[1] &&
				t0->triWithBeamid[j] != t1->triWithBeamid[2] &&
				t1->triWithBeamid[k] != t0->triWithBeamid[0] &&
				t1->triWithBeamid[k] != t0->triWithBeamid[1] &&
				t1->triWithBeamid[k] != t0->triWithBeamid[2])
			{
				t0->triWithBeamid[(j + 2) % 3] = t1->triWithBeamid[k];
				t0->p[(j + 2) % 3] = t1->p[k];
				t1->triWithBeamid[(k + 2) % 3] = t0->triWithBeamid[j];
				t1->p[(k + 2) % 3] = t0->p[j];

				t0->adjTriangles[(j + 1) % 3] = t1->adjTriangles[(k + 2) % 3];
				t1->adjTriangles[(k + 1) % 3] = t0->adjTriangles[(j + 2) % 3];
				t0->adjTriangles[(j + 2) % 3] = t1;
				t1->adjTriangles[(k + 2) % 3] = t0;

				FixNeighborhood(t0->adjTriangles[(j + 1) % 3], t1, t0);
				FixNeighborhood(t1->adjTriangles[(k + 1) % 3], t0, t1);

				DoLocalOptimization(t0, t0->adjTriangles[j]);
				DoLocalOptimization(t0, t0->adjTriangles[(j + 1) % 3]);
				DoLocalOptimization(t1, t1->adjTriangles[k]);
				DoLocalOptimization(t1, t1->adjTriangles[(k + 1) % 3]);

				return true;
			}
		}
	}

	return false;
}
void AssignNeighbors(Triangle* tri,Triangle* t0, Triangle* t1, Triangle* t2)
{
	tri->adjTriangles[0] = t0;
	tri->adjTriangles[1] = t1;
	tri->adjTriangles[2] = t2;
}
void SplitTriangle(Triangle** Mesh, int meshIdx, Triangle* tri, Point* dot)
{
	
	//Triangle* newTriangle1 = new triangle(dot, tri->Vertex[1], tri->Vertex[2]);
	//Triangle* newTriangle2 = new triangle(dot, tri->Vertex[2], tri->Vertex[0]);
	Mesh[meshIdx] = (Triangle*)malloc(sizeof(Triangle));
	Triangle* newTriangle1 = Mesh[meshIdx];
	newTriangle1->p[0] = make_float3(dot->x, dot->y, dot->z); newTriangle1->triWithBeamid[0] = dot->id;
	newTriangle1->p[1] = tri->p[1]; newTriangle1->triWithBeamid[1] = tri->triWithBeamid[1];
	newTriangle1->p[2] = tri->p[2]; newTriangle1->triWithBeamid[2] = tri->triWithBeamid[2];

	Mesh[meshIdx + 1] = (Triangle*)malloc(sizeof(Triangle));
	Triangle* newTriangle2 = Mesh[meshIdx + 1];
	newTriangle2->p[0] = make_float3(dot->x, dot->y, dot->z); newTriangle2->triWithBeamid[0] = dot->id;
	newTriangle2->p[1] = tri->p[2]; newTriangle2->triWithBeamid[1] = tri->triWithBeamid[2];
	newTriangle2->p[2] = tri->p[0]; newTriangle2->triWithBeamid[2] = tri->triWithBeamid[0];

	tri->p[2] = tri->p[1]; tri->triWithBeamid[2] = tri->triWithBeamid[1];
	tri->p[1] = tri->p[0]; tri->triWithBeamid[1] = tri->triWithBeamid[0];
	tri->p[0] = make_float3(dot->x, dot->y, dot->z); tri->triWithBeamid[0] = dot->id;

	//AssignNeighbors
	//newTriangle1->AssignNeighbors(tri, tri->Neighbor[1], newTriangle2);
	AssignNeighbors(newTriangle1,tri, tri->adjTriangles[1], newTriangle2);
	/*newTriangle1->adjTriangles[0] = tri;
	newTriangle1->adjTriangles[1] = tri->adjTriangles[1];
	newTriangle1->adjTriangles[2] = newTriangle2;*/
	//newTriangle2->AssignNeighbors(newTriangle1, tri->Neighbor[2], tri);
	AssignNeighbors(newTriangle2,newTriangle1, tri->adjTriangles[2], tri);
	/*newTriangle2->adjTriangles[0] = newTriangle1;
	newTriangle2->adjTriangles[1] = tri->adjTriangles[2];
	newTriangle2->adjTriangles[2] = tri;*/
	//tri->AssignNeighbors(newTriangle2, tri->Neighbor[0], newTriangle1);
	AssignNeighbors(tri,newTriangle2, tri->adjTriangles[0], newTriangle1);
	/*tri->adjTriangles[0] = newTriangle2;
	tri->adjTriangles[1] = tri->adjTriangles[0];
	tri->adjTriangles[2] = newTriangle1;*/

	FixNeighborhood(newTriangle1->adjTriangles[1], tri, newTriangle1);
	FixNeighborhood(newTriangle2->adjTriangles[1], tri, newTriangle2);

	/*_Mesh->push_back(newTriangle1);
	_Mesh->push_back(newTriangle2);*/

	// optimize triangles according to delaunay triangulation definition
	
	DoLocalOptimization(tri, tri->adjTriangles[1]);

	DoLocalOptimization(newTriangle1, newTriangle1->adjTriangles[1]);
	DoLocalOptimization(newTriangle2, newTriangle2->adjTriangles[1]);
}

void InsertDot(Point* dot, Triangle** Mesh, int currentMeshNum)
{
	double det[] = { 0, 0, 0 };

	/*vector<triangle*>::iterator it;
	it = _Mesh->begin();*/
	Triangle* tri = Mesh[0];
	Triangle* tri2 = Mesh[3];
	short flag = 0;
	int i = 0;
	//while (it != _Mesh->end())
	while (i < currentMeshNum)
	{
		//_Statistics[0]++;
		//tri = *it++;
		tri = Mesh[i++];
		float3 dotInsert = make_float3(dot->x, dot->y, dot->z);
		det[0] = GetDeterminant(tri->p[0], tri->p[1], dotInsert);
		det[1] = GetDeterminant(tri->p[1], tri->p[2], dotInsert);
		det[2] = GetDeterminant(tri->p[2], tri->p[0], dotInsert);

		if (flag)
		{
			float3 v[3];
			for (int j = 0; j < 3; j++)
			{
				v[j] = make_float3(tri->p[j].x, tri->p[j].y, tri->p[j].z);
				//v[j] = tri->p[j];
			}
			float3 triNormal = cross(Subtraction(v[1], v[0]), Subtraction(v[2], v[0]));
			float3 otherV = make_float3(dot->x, dot->y, dot->z);
			float3 otherVec = Subtraction(otherV, v[0]);
			if (Dot(triNormal, otherVec) > 0)
			{
				////判断和相邻三角形是否共面
				//bool coFace = false;
				//for (int i = 0; i < 3; i++)
				//{
				//	triangle* adj = tri->Neighbor[i];
				//	float3 adjv[3];
				//	for (int j = 0; j < 3; j++)
				//	{
				//		adjv[j] = make_float3(adj->Vertex[j]->X, adj->Vertex[j]->Y, adj->Vertex[j]->Z);
				//	}
				//	float3 adjNormal = cross(Subtraction(adjv[1], adjv[0]), Subtraction(adjv[2], adjv[0]));
				//	float3 adjOtherVec = Subtraction(otherV, adjv[0]);
				//	if (Dot(adjNormal, adjOtherVec) == 0)
				//	{
				//		coFace = true;
				//		break;
				//	}
				//}
				//if (coFace)
				//	continue;
				//if (!tri->HasVertexCoincidentWith(dot))
				SplitTriangle(Mesh, currentMeshNum, tri, dot);
				//printf("%d %d %d\n",flag, i, currentMeshNum);

				return;
			}
		}
		// if this dot projected into an existing triangle, split the existing triangle to 3 new ones
		if ((!flag) && det[0] <= 0 && det[1] <= 0 && det[2] <= 0)
		{
			//判断和当前三角形是否共面
			bool coFace = false;
			//for (int i = 0; i < 3; i++)
			{
				//triangle* adj = tri->Neighbor[i];
				float3 v[3];
				for (int j = 0; j < 3; j++)
				{
					v[j] = make_float3(tri->p[j].x, tri->p[j].y, tri->p[j].z);
					//v[j] = tri->p[j];
				}
				float3 triNormal = cross(Subtraction(v[1], v[0]), Subtraction(v[2], v[0]));
				float3 otherV = make_float3(dot->x, dot->y, dot->z);
				float3 otherVec = Subtraction(otherV, v[0]);
				if (Dot(triNormal, otherVec) == 0)
				{
					coFace = true;
					//break;
				}
			}
			/*if (coFace)
				break;*/
				//if (!tri->HasVertexCoincidentWith(dot) && !coFace)
			if (!coFace)
			{
				SplitTriangle(Mesh, currentMeshNum, tri, dot);
				//printf("%d %d %d\n",flag, i, currentMeshNum);

				return;
			}
		}
		//if (it == _Mesh->end() && !flag)
		if (i == currentMeshNum && !flag)
		{
			flag = 1;
			i = 0;
		}


		//// on one side, search neighbors
		//else if (det[1] >= 0 && det[2] >= 0)
		//    triangle = triangle->Neighbor[0];
		//else if (det[0] >= 0 && det[2] >= 0)
		//    triangle = triangle->Neighbor[1];
		//else if (det[0] >= 0 && det[1] >= 0)
		//    triangle = triangle->Neighbor[2];

		//// cannot determine effectively 
		//else if (det[0] >= 0)
		//    triangle = triangle->Neighbor[1];
		//else if (det[1] >= 0)
		//    triangle = triangle->Neighbor[2];
		//else if (det[2] >= 0)
		//    triangle = triangle->Neighbor[0];
		//else
		//    triangle = *it++;
	}
}


vector<tuple<int, int, int>*> DelaunayTriangulation::GetTriangulationResult(vector<Vector3D*> &dots,float projRadius)
{
    _Statistics[2] = clock();

    _ProjectedDots->reserve(dots.size());

    // N random dots can form 8+(N-6)*2 triangles based on the algorithm
    _Mesh->reserve(8 + (dots.size() - 6) * 2);

    // project dots to an unit shpere for triangulation
    vector<Vector3D*>::iterator itDots;
    for (itDots = dots.begin(); itDots != dots.end(); itDots++)
    {
        Vector3D* projectedDot = new Vector3D((*itDots), projRadius);
        _ProjectedDots->push_back(projectedDot);
    }
	/*int nFaces = 8 + (dots.size() - 6) * 2;
	Triangle** Mesh = (Triangle**)malloc(nFaces * sizeof(Triangle*));*/
    // prepare initial convex hull with 6 vertices and 8 triangle faces
    BuildInitialHull(_ProjectedDots);
	//BuildInitialHull()
    for (itDots = _ProjectedDots->begin(); itDots != _ProjectedDots->end(); itDots++)
    {
        Vector3D* dot = *itDots;
        if (!dot->IsVisited)
        {
            InsertDot(dot);
        }
    }

    // remove trianges connected with auxiliary dots
    RemoveExtraTriangles();

    // generate output
    vector<tuple<int, int, int>*> mesh = vector<tuple<int, int, int>*>();
    vector<triangle*>::iterator itMesh;
    for (itMesh = _Mesh->begin(); itMesh != _Mesh->end(); itMesh++)
    {
		triangle* triangle = *itMesh;
        mesh.push_back(new tuple<int, int, int>(
            triangle->Vertex[0]->Id,
            triangle->Vertex[1]->Id,
            triangle->Vertex[2]->Id
            ));
    }

    _Statistics[3] = clock();

    return mesh;
}
int findTriid(int* triId,int a)
{
	for (int i = 0; i < 3; i++)
	{
		if (triId[i] == a)
			return i;
	}
	return 3;
}
void DelaunayTriangulation::BuildInitialHull(vector<Vector3D*>* dots)
{
	Vector3D* initialVertices[4];
	triangle* initialHullFaces[4];
	float3 P[4];
	for (int i = 0; i < 3; i++)
	{
		initialVertices[i] = *(dots->begin() + i);
		P[i] = make_float3((*(dots->begin()+i))->X, (*(dots->begin() + i))->Y, (*(dots->begin() + i))->Z);
		(*(dots->begin() + i))->IsVisited = true;
	}
	float3 n_circumcenter = ComputeNegativeCircumcenter(P);

	initialVertices[3] = *(dots->begin() + 3);
	P[3] = make_float3((*(dots->begin() + 3))->X, (*(dots->begin() + 3))->Y, (*(dots->begin() + 3))->Z);
	float dis = Norm(Subtraction(P[3], n_circumcenter));
	//找到距离-C0最近的点
	for (auto it = dots->begin()+4; it != dots->end(); it++)
	{
		float3 Pk = make_float3((*it)->X, (*it)->Y, (*it)->Z);
		//判断四点是否共面，若共面，continue
		float3 P0toPk = Subtraction(Pk, P[0]);
		float3 normal = cross(Subtraction(P[1], P[0]), Subtraction(P[2], P[0]));
		if (Dot(normal, P0toPk) == 0)
			continue;
		float tmpDis = Norm(Subtraction(Pk, n_circumcenter));
		if (tmpDis < dis)
		{
			initialVertices[3] = *it;
			P[3] = Pk;
			dis = tmpDis;
		}
	}
	printf("%d\n", initialVertices[3]->Id);
	//确定三角形法线朝向与半球位置一致
	float3 normal = cross(Subtraction(P[1], P[0]), Subtraction(P[2], P[0]));
	float3 CtoP = P[0];
	if (Dot(normal, CtoP) < 0)
	{
		Vector3D* vtmp = initialVertices[0];
		initialVertices[0] = initialVertices[1];
		initialVertices[1] = vtmp;

		float3 ptmp = P[0];
		P[0] = P[1];
		P[1] = ptmp;
		normal = cross(Subtraction(P[1], P[0]), Subtraction(P[2], P[0]));
	}
	//根据Pk确定三角形法线朝向
	float3 P0toPk = Subtraction(P[3], P[0]);
	if (Dot(normal, P0toPk) > 0)
	{
		Vector3D* vtmp = initialVertices[0];
		initialVertices[0] = initialVertices[1];
		initialVertices[1] = vtmp;

		float3 ptmp = P[0];
		P[0] = P[1];
		P[1] = ptmp;
		normal = cross(Subtraction(P[1], P[0]), Subtraction(P[2], P[0]));
	}

	//建立四面体
	int vertex0Index[] = { 0, 0, 0, 1 };
	int vertex1Index[] = { 1, 3, 2, 3 };
	int vertex2Index[] = { 2, 1, 3, 2 };
	int triId[4][3];
	for (int i = 0; i < 4; i++)
	{
		Vector3D* v0 = initialVertices[vertex0Index[i]];
		Vector3D* v1 = initialVertices[vertex1Index[i]];
		Vector3D* v2 = initialVertices[vertex2Index[i]];

		triangle* tri = new triangle(v0, v1, v2);
		initialHullFaces[i] = tri;
		for (int j = 0; j < 3; j++)
		{
			triId[i][j] = initialHullFaces[i]->Vertex[j]->Id;
		}
		//cnt++;
		//initialHullFaces[i] = tri;

		_Mesh->push_back(tri);
	}

	int neighbor0Index[] = { 1, 2, 0, 1};
	int neighbor1Index[] = { 3, 3, 3, 2};
	int neighbor2Index[] = { 2, 0, 1, 0};
	for (int i = 0; i < 4; i++)
    {
		triangle* n0 = initialHullFaces[neighbor0Index[i]];
		triangle* n1 = initialHullFaces[neighbor1Index[i]];
		triangle* n2 = initialHullFaces[neighbor2Index[i]];
        initialHullFaces[i]->AssignNeighbors(n0, n1, n2);
    }

	// dot already in the mesh, avoid being visited by InsertDot() again
	for (int i = 0; i < 4; i++)
	{
		initialVertices[i]->IsVisited = true;
	}
}
void DelaunayTriangulation::BuildInitialHull(vector<Vector3D*>* dots,float projRadius)
{
    Vector3D* initialVertices[INIT_VERTICES_COUNT];
	triangle* initialHullFaces[INIT_FACES_COUNT];

    for (int i = 0; i < INIT_VERTICES_COUNT; i++)
    {
        //initialVertices[i] = _AuxiliaryDots[i];
		initialVertices[i] = (Vector3D*)malloc(sizeof(Vector3D));
    }
	initialVertices[0]->X = -projRadius;
	initialVertices[1]->X = projRadius;
	initialVertices[2]->Y = -projRadius;
	initialVertices[3]->Y = projRadius;
	initialVertices[4]->Z = -projRadius;
	initialVertices[5]->Z = projRadius;
    // if close enough, use input dots to replace auxiliary dots so won't be removed in the end
    double minDistance[INIT_VERTICES_COUNT] = { 0, 0, 0, 0, 0, 0 };
    vector<Vector3D*>::iterator it;
    for (it = dots->begin(); it != dots->end(); it++)
    {
		for (int i = 0; i < 6; i++)
		{
			if (i == 0)
			{
				if((*it)->X>initialVertices[i]->X)
					initialVertices[i] = *it;
			}
			else if (i == 1)
			{
				if((*it)->X < initialVertices[i]->X)
					initialVertices[i] = *it;
			}
			else if (i == 2)
			{
				if((*it)->Y> initialVertices[i]->Y)
					initialVertices[i] = *it;
			}
			else if (i == 3)
			{
				if ((*it)->Y < initialVertices[i]->Y)
					initialVertices[i] = *it;
			}
			else if (i == 4)
			{
				if((*it)->Z> initialVertices[i]->Z)
					initialVertices[i] = *it;
			}
			else if (i == 5)
			{
				if ((*it)->Z < initialVertices[i]->Z)
					initialVertices[i] = *it;
			}
		}
  //      double distance[INIT_VERTICES_COUNT];
  //      for (int i = 0; i < INIT_VERTICES_COUNT; i++)
  //      {
  //          distance[i] = GetDistance(_AuxiliaryDots[i], *it);
  //         /* if (minDistance[i] == 0 || distance[i] < minDistance[i])
  //          {
  //              minDistance[i] = distance[i];
  //          }*/
  //      }
		//for (int i = 0; i < INIT_VERTICES_COUNT; i++)
		//{
		//	if (IsMinimumValueInArray(distance, INIT_VERTICES_COUNT, i))
		//	{
		//		initialVertices[i] = *it;
		//	}
		//}
       /* for (int i = 0; i < INIT_VERTICES_COUNT; i++)
        {
            if (minDistance[i] == distance[i] && IsMinimumValueInArray(distance, INIT_VERTICES_COUNT, i))
            {
                initialVertices[i] = *it;
            }
        }*/
    }

    int vertex0Index[] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    int vertex1Index[] = { 4, 3, 5, 2, 2, 4, 3, 5 };
    int vertex2Index[] = { 2, 4, 3, 5, 4, 3, 5, 2 };
	/*int vertex0Index[] = { 0, 0, 0, 3, 1, 1, 1, 1 };
	int vertex1Index[] = { 4, 3, 2, 2, 2, 4, 3, 5 };
	int vertex2Index[] = { 2, 4, 3, 5, 4, 3, 5, 2 };*/
	int cnt = 0;
	int triId[INIT_FACES_COUNT][3];
    for (int i = 0; i < INIT_FACES_COUNT; i++)
    {
        Vector3D* v0 = initialVertices[vertex0Index[i]];
        Vector3D* v1 = initialVertices[vertex1Index[i]];
        Vector3D* v2 = initialVertices[vertex2Index[i]];
		if (v0->Id == v1->Id || v0->Id == v2->Id || v1->Id == v2->Id)
			continue;

		triangle* tri = new triangle(v0, v1, v2);
		initialHullFaces[cnt] = tri;
		for (int j = 0; j < 3; j++)
		{
			triId[cnt][j] = initialHullFaces[cnt]->Vertex[j]->Id;
		}
		cnt++;
        //initialHullFaces[i] = tri;

        _Mesh->push_back(tri);
    }
	int faces_cnt = _Mesh->size();

    int neighbor0Index[] = { 1, 2, 3, 0, 7, 4, 5, 6 };
    int neighbor1Index[] = { 4, 5, 6, 7, 0, 1, 2, 3 };
    int neighbor2Index[] = { 3, 0, 1, 2, 5, 6, 7, 4 };

	for (int i = 0; i < faces_cnt; i++)
	{
		triangle* n0 = (triangle*)malloc(sizeof(triangle));
		triangle* n1 = (triangle*)malloc(sizeof(triangle));
		triangle* n2 = (triangle*)malloc(sizeof(triangle));
		for (int j = 0; j < faces_cnt; j++)
		{
			if (j != i)
			{
				if (findTriid(triId[j], triId[i][0]) != 3 &&
					findTriid(triId[j], triId[i][1]) != 3)
				{
					n0 = initialHullFaces[j];
				}
				else if (findTriid(triId[j], triId[i][1]) != 3 &&
					findTriid(triId[j], triId[i][2]) != 3)
				{
					n1 = initialHullFaces[j];
				}
				else if (findTriid(triId[j], triId[i][2]) != 3 &&
					findTriid(triId[j], triId[i][0]) != 3)
				{
					n2 = initialHullFaces[j];
				}
			}
		}
		initialHullFaces[i]->AssignNeighbors(n0, n1, n2);
	}
    /*for (int i = 0; i < INIT_FACES_COUNT; i++)
    {
		triangle* n0 = initialHullFaces[neighbor0Index[i]];
		triangle* n1 = initialHullFaces[neighbor1Index[i]];
		triangle* n2 = initialHullFaces[neighbor2Index[i]];
        initialHullFaces[i]->AssignNeighbors(n0, n1, n2);
    }*/

    // dot already in the mesh, avoid being visited by InsertDot() again
    for (int i = 0; i < INIT_VERTICES_COUNT; i++)
    {
        initialVertices[i]->IsVisited = true;
    }
}

void DelaunayTriangulation::InsertDot(Vector3D* dot)
{
    double det[] = { 0, 0, 0 };

    vector<triangle*>::iterator it;
    it = _Mesh->begin();
	triangle* tri = *it;

	short flag = 0;
    while (it != _Mesh->end())
    {
        _Statistics[0]++;
		tri = *it++;

        det[0] = GetDeterminant(tri->Vertex[0], tri->Vertex[1], dot);
        det[1] = GetDeterminant(tri->Vertex[1], tri->Vertex[2], dot);
        det[2] = GetDeterminant(tri->Vertex[2], tri->Vertex[0], dot);

		if (flag)
		{
			float3 v[3];
			for (int i = 0; i < 3; i++)
			{
				v[i] = make_float3(tri->Vertex[i]->X, tri->Vertex[i]->Y, tri->Vertex[i]->Z);
			}
			float3 triNormal = cross(Subtraction(v[1], v[0]), Subtraction(v[2], v[0]));
			float3 otherV = make_float3(dot->X,dot->Y,dot->Z);
			float3 otherVec = Subtraction(otherV, v[0]);
			if (Dot(triNormal,otherVec)>0)
			{
				////判断和相邻三角形是否共面
				//bool coFace = false;
				//for (int i = 0; i < 3; i++)
				//{
				//	triangle* adj = tri->Neighbor[i];
				//	float3 adjv[3];
				//	for (int j = 0; j < 3; j++)
				//	{
				//		adjv[j] = make_float3(adj->Vertex[j]->X, adj->Vertex[j]->Y, adj->Vertex[j]->Z);
				//	}
				//	float3 adjNormal = cross(Subtraction(adjv[1], adjv[0]), Subtraction(adjv[2], adjv[0]));
				//	float3 adjOtherVec = Subtraction(otherV, adjv[0]);
				//	if (Dot(adjNormal, adjOtherVec) == 0)
				//	{
				//		coFace = true;
				//		break;
				//	}
				//}
				//if (coFace)
				//	continue;
				//if (!tri->HasVertexCoincidentWith(dot))
				{
					printf("%d %d %d\n", flag,it-_Mesh->begin(), _Mesh->size());
					SplitTriangle(tri, dot);
				}

				return;
			}
		}
        // if this dot projected into an existing triangle, split the existing triangle to 3 new ones
        if ((!flag)&&det[0] <= 0 && det[1] <= 0 && det[2] <= 0)
        {
			//判断和当前三角形是否共面
			bool coFace = false;
			//for (int i = 0; i < 3; i++)
			{
				//triangle* adj = tri->Neighbor[i];
				float3 v[3];
				for (int j = 0; j < 3; j++)
				{
					v[j] = make_float3(tri->Vertex[j]->X, tri->Vertex[j]->Y, tri->Vertex[j]->Z);
				}
				float3 triNormal = cross(Subtraction(v[1], v[0]), Subtraction(v[2], v[0]));
				float3 otherV = make_float3(dot->X, dot->Y, dot->Z);
				float3 otherVec = Subtraction(otherV, v[0]);
				if (Dot(triNormal,otherVec) == 0)
				{
					coFace = true;
					//break;
				}
			}
			/*if (coFace)
				break;*/
			if ( !coFace)
			{
				printf("%d %d %d\n",flag, it - _Mesh->begin(), _Mesh->size());
				SplitTriangle(tri, dot);


				return;
			}
        }
		if (it == _Mesh->end()&&!flag)
		{
			flag = 1;
			it = _Mesh->begin();
		}

		
        //// on one side, search neighbors
        //else if (det[1] >= 0 && det[2] >= 0)
        //    triangle = triangle->Neighbor[0];
        //else if (det[0] >= 0 && det[2] >= 0)
        //    triangle = triangle->Neighbor[1];
        //else if (det[0] >= 0 && det[1] >= 0)
        //    triangle = triangle->Neighbor[2];

        //// cannot determine effectively 
        //else if (det[0] >= 0)
        //    triangle = triangle->Neighbor[1];
        //else if (det[1] >= 0)
        //    triangle = triangle->Neighbor[2];
        //else if (det[2] >= 0)
        //    triangle = triangle->Neighbor[0];
        //else
        //    triangle = *it++;
    }
}

void DelaunayTriangulation::RemoveExtraTriangles()
{
    vector<triangle*>::iterator it;
    for (it = _Mesh->begin(); it != _Mesh->end();)
    {
		triangle* triangle = *it;
        bool isExtraTriangle = false;
        for (int i = 0; i < 3; i++)
        {
            if (triangle->Vertex[i]->IsAuxiliaryDot)
            {
                isExtraTriangle = true;
                break;
            }
        }

        if (isExtraTriangle)
        {
            delete *it;
            it = _Mesh->erase(it);
        }
        else
        {
            it++;
        }
    }
}

void DelaunayTriangulation::SplitTriangle(triangle* tri, Vector3D* dot)
{
	triangle* newTriangle1 = new triangle(dot, tri->Vertex[1], tri->Vertex[2]);
	triangle* newTriangle2 = new triangle(dot, tri->Vertex[2], tri->Vertex[0]);

	tri->Vertex[2] = tri->Vertex[1];
	tri->Vertex[1] = tri->Vertex[0];
	tri->Vertex[0] = dot;

    newTriangle1->AssignNeighbors(tri, tri->Neighbor[1], newTriangle2);
    newTriangle2->AssignNeighbors(newTriangle1, tri->Neighbor[2], tri);
	tri->AssignNeighbors(newTriangle2, tri->Neighbor[0], newTriangle1);

    FixNeighborhood(newTriangle1->Neighbor[1], tri, newTriangle1);
    FixNeighborhood(newTriangle2->Neighbor[1], tri, newTriangle2);

    _Mesh->push_back(newTriangle1);
    _Mesh->push_back(newTriangle2);

    // optimize triangles according to delaunay triangulation definition
    DoLocalOptimization(tri, tri->Neighbor[1]);
    DoLocalOptimization(newTriangle1, newTriangle1->Neighbor[1]);
    DoLocalOptimization(newTriangle2, newTriangle2->Neighbor[1]);
}

void DelaunayTriangulation::FixNeighborhood(triangle* target, triangle* oldNeighbor, triangle* newNeighbor)
{
    for (int i = 0; i < 3; i++)
    {
        if (target->Neighbor[i] == oldNeighbor)
        {
            target->Neighbor[i] = newNeighbor;
            break;
        }
    }
}

void DelaunayTriangulation::DoLocalOptimization(triangle* t0, triangle* t1)
{
    _Statistics[1]++;

    for (int i = 0; i < 3; i++)
    {
        if (t1->Vertex[i]->Id == t0->Vertex[0]->Id ||
            t1->Vertex[i]->Id == t0->Vertex[1]->Id ||
            t1->Vertex[i]->Id == t0->Vertex[2]->Id)
        {
            continue;
        }

        double matrix[] = {
            t1->Vertex[i]->X - t0->Vertex[0]->X,
            t1->Vertex[i]->Y - t0->Vertex[0]->Y,
            t1->Vertex[i]->Z - t0->Vertex[0]->Z,

            t1->Vertex[i]->X - t0->Vertex[1]->X,
            t1->Vertex[i]->Y - t0->Vertex[1]->Y,
            t1->Vertex[i]->Z - t0->Vertex[1]->Z,

            t1->Vertex[i]->X - t0->Vertex[2]->X,
            t1->Vertex[i]->Y - t0->Vertex[2]->Y,
            t1->Vertex[i]->Z - t0->Vertex[2]->Z
        };
		std::cout << GetDeterminant(matrix) << std::endl;
        if (GetDeterminant(matrix) >= 0)
        {
            // terminate after optimized
            break;
        }

        if (TrySwapDiagonal(t0, t1))
        {
            return;
        }
		else
		{
			printf("FALSE\n");
		}
    }
}

bool DelaunayTriangulation::TrySwapDiagonal(triangle* t0, triangle* t1)
{
    for (int j = 0; j < 3; j++)
    {
        for (int k = 0; k < 3; k++)
        {
            if (t0->Vertex[j]->Id != t1->Vertex[0]->Id &&
                t0->Vertex[j]->Id != t1->Vertex[1]->Id &&
                t0->Vertex[j]->Id != t1->Vertex[2]->Id &&
                t1->Vertex[k]->Id != t0->Vertex[0]->Id &&
                t1->Vertex[k]->Id != t0->Vertex[1]->Id &&
                t1->Vertex[k]->Id != t0->Vertex[2]->Id)
            {
                t0->Vertex[(j + 2) % 3] = t1->Vertex[k];
                t1->Vertex[(k + 2) % 3] = t0->Vertex[j];

                t0->Neighbor[(j + 1) % 3] = t1->Neighbor[(k + 2) % 3];
                t1->Neighbor[(k + 1) % 3] = t0->Neighbor[(j + 2) % 3];
                t0->Neighbor[(j + 2) % 3] = t1;
                t1->Neighbor[(k + 2) % 3] = t0;

                FixNeighborhood(t0->Neighbor[(j + 1) % 3], t1, t0);
                FixNeighborhood(t1->Neighbor[(k + 1) % 3], t0, t1);

                DoLocalOptimization(t0, t0->Neighbor[j]);
                DoLocalOptimization(t0, t0->Neighbor[(j + 1) % 3]);
                DoLocalOptimization(t1, t1->Neighbor[k]);
                DoLocalOptimization(t1, t1->Neighbor[(k + 1) % 3]);

                return true;
            }
        }
    }

    return false;
}

bool DelaunayTriangulation::IsMinimumValueInArray(double arr[], int length, int index)
{
    for (int i = 0; i < length; i++)
    {
        if (arr[i] < arr[index])
        {
            return false;
        }
    }

    return true;
}

double DelaunayTriangulation::GetDistance(Vector3D* v0, Vector3D* v1)
{
    return sqrt(pow((v0->X - v1->X), 2) +
        pow((v0->Y - v1->Y), 2) +
        pow((v0->Z - v1->Z), 2));
}

double DelaunayTriangulation::GetDeterminant(Vector3D* v0, Vector3D* v1, Vector3D* v2)
{
    double matrix[] = {
        v0->X, v0->Y, v0->Z,
        v1->X, v1->Y, v1->Z,
        v2->X, v2->Y, v2->Z
    };

    return GetDeterminant(matrix);
}

double DelaunayTriangulation::GetDeterminant(double matrix[])
{
    // inversed for left handed coordinate system
    double determinant = matrix[2] * matrix[4] * matrix[6]
        + matrix[0] * matrix[5] * matrix[7]
        + matrix[1] * matrix[3] * matrix[8]
        - matrix[0] * matrix[4] * matrix[8]
        - matrix[1] * matrix[5] * matrix[6]
        - matrix[2] * matrix[3] * matrix[7];

    // adjust result based on float number accuracy, otherwise causing deadloop
    return abs(determinant) <= DBL_EPSILON ? 0 : determinant;
}

string DelaunayTriangulation::GetStatistics()
{
    // display thousands separator
    regex regex("\\d{1,3}(?=(\\d{3})+$)");

    return "\nTriangle count: "
        + regex_replace(to_string(_Mesh->size()), regex, "$&,")
        + "\nTriangle search operations: "
        + regex_replace(to_string(_Statistics[0]), regex, "$&,")
        + "\nLocal optimizations: "
        + regex_replace(to_string(_Statistics[1]), regex, "$&,")
        + "\nTriangulation cost: "
        + to_string(_Statistics[3] - _Statistics[2])
        + "ms\n";
}