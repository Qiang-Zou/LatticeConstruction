#ifndef DATASTRUCTURE_H
#define DATASTRUCTURE_H

namespace dt
{
    class Vector3D
    {
    private:
        int GenerateRunningId();
    public:
        int Id = 0;

        // coordinate
        double X, Y, Z;

        // color
        uint8_t R, G, B;

        bool IsVisited = false;
        bool IsAuxiliaryDot = false;

        Vector3D(double x, double y, double z, uint8_t r = 255, uint8_t g = 248, uint8_t b = 220);
        Vector3D(double x, double y, double z, bool isAuxiliaryDot, uint8_t r = 255, uint8_t g = 248, uint8_t b = 220);
        Vector3D(Vector3D* dot, double lengthAfterProjection);
        ~Vector3D();

        bool IsCoincidentWith(Vector3D* dot);
        std::string ToString();
    };

    class triangle
    {
    private:
        int GenerateRunningId();
    public:
        int Id = 0;

        // pointers pointing to 3 vertices
        Vector3D* Vertex[3];

        // pointers pointing to 3 neighbors
        triangle* Neighbor[3];

		triangle(Vector3D* v0, Vector3D* v1, Vector3D* v2);
        ~triangle();

        bool HasVertexCoincidentWith(Vector3D* dot);
        void AssignNeighbors(triangle* n0, triangle* n1, triangle* n2);
        std::string ToString();
    };
}

#endif