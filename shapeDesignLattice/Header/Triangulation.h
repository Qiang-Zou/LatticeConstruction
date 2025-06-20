#pragma once
#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#define INIT_VERTICES_COUNT 6 /* count of vertices in the initial hull */
#define INIT_FACES_COUNT 8 /* count of faces in the initial hull */
#define VECTOR_LENGTH 1 /* radius of unit sphere the dots projected into */

#include "DataStructure.h"
#include "../beam.h"
namespace dt
{
    class DelaunayTriangulation
    {
    private:
        Vector3D * _AuxiliaryDots[INIT_VERTICES_COUNT];
        std::vector<Vector3D*>* _ProjectedDots;
        std::vector<triangle*>* _Mesh;

        // 0: triangle search operations
        // 1: local optimizations
        // 2: start time; 3: end time;
        long _Statistics[4];
		void BuildInitialHull(std::vector<Vector3D*>* dots);
        void BuildInitialHull(std::vector<Vector3D*>* dots,float projRadius);
        void InsertDot(Vector3D* dot);
        void RemoveExtraTriangles();
        void SplitTriangle(triangle* triangle, Vector3D* dot);
        void FixNeighborhood(triangle* target, triangle* oldNeighbor, triangle* newNeighbor);
        void DoLocalOptimization(triangle* t0, triangle* t1);
        bool TrySwapDiagonal(triangle* t0, triangle* t1);
        bool IsMinimumValueInArray(double arr[], int length, int index);
        double GetDistance(Vector3D* v0, Vector3D* v1);
        double GetDeterminant(Vector3D* v0, Vector3D* v1, Vector3D* v2);
        double GetDeterminant(double matrix[]);

    public:
        DelaunayTriangulation(float projRadius);
        ~DelaunayTriangulation();

        std::vector<std::tuple<int, int, int>*> GetTriangulationResult(std::vector<Vector3D*> &dots, float projRadius);
        std::string GetStatistics();
    };
}

void BuildInitialHull(Point** dots, int pointNum, Triangle** Mesh);
void InsertDot(Point* dot, Triangle** Mesh, int currentMeshNum);

#endif