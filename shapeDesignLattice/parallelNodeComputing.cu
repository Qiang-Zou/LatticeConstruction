#include "cuda.h"
#include "cuda_runtime.h"
#include"curand_kernel.h"
#include "device_launch_parameters.h"
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "MOGWO.h"
#include "beam.h"
#include "call_cuda.h"
//#include "convhull_3d.cuh"
#define THREADS_PER_BLOCK			96
#define BLOCKS_PER_GRID				4096
using namespace Eigen;
__device__ double Min(double a, double b)
{
	return (a < b ? a : b);
}
__device__ double Max(double a, double b)
{
	return (a > b ? a : b);
}
__device__  float3 cuda_cross(float3 a, float3 b)
{
	float3 result;
	result.x = a.y*b.z - b.y*a.z;
	result.y = -(a.x*b.z - b.x*a.z);
	result.z = a.x*b.y - b.x*a.y;
	return result;
}
__device__  float4 cuda_multiply(float* m, float4 a)
{
	float4 result;
	result.x = m[0*4+0] * a.x + m[0*4+1] * a.y + m[0*4+2] * a.z + m[0*4+3] * a.w;
	result.y = m[1*4+0] * a.x + m[1*4+1] * a.y + m[1*4+2] * a.z + m[1*4+3] * a.w;
	result.z = m[2*4+0] * a.x + m[2*4+1] * a.y + m[2*4+2] * a.z + m[2*4+3] * a.w;
	result.w = m[3*4+0] * a.x + m[3*4+1] * a.y + m[3*4+2] * a.z + m[3*4+3] * a.w;

	return result;
}
__device__  float3 cuda_make_norm(float3 a)
{
	float norm = sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2));
	float3 result = make_float3(a.x / norm, a.y / norm, a.z / norm);
	return result;
}
__device__  float cuda_Dot(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__  float cuda_Norm(float3 a)
{
	return sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2));
}
__device__ float3 cuda_Subtraction(float3 a, float3 b)
{
	float3 result = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	return result;
}
__device__ float3 cuda_ComputeNegativeCircumcenter(float3 P[])
{
	float D = (2 * pow(cuda_Norm(cuda_cross(cuda_Subtraction(P[0], P[1]), cuda_Subtraction(P[1], P[2]))), 2));

	float alpha = pow(cuda_Norm(cuda_Subtraction(P[1], P[2])), 2)*(cuda_Dot(cuda_Subtraction(P[0], P[1]), cuda_Subtraction(P[0], P[2])));
	alpha = alpha / D;

	float beta = pow(cuda_Norm(cuda_Subtraction(P[0], P[2])), 2)*(cuda_Dot(cuda_Subtraction(P[1], P[0]), cuda_Subtraction(P[1], P[2])));
	beta = beta / D;

	float gamma = pow(cuda_Norm(cuda_Subtraction(P[0], P[1])), 2)*(cuda_Dot(cuda_Subtraction(P[2], P[0]), cuda_Subtraction(P[2], P[1])));
	gamma = gamma / D;

	float3 result = make_float3(-(alpha*P[0].x + beta * P[1].x + gamma * P[2].x), -(alpha*P[0].y + beta * P[1].y + gamma * P[2].y),
		-(alpha*P[0].z + beta * P[1].z + gamma * P[2].z));

	return result;
}
//#ifndef CONVHULL_3D_INCLUDED
//#define CONVHULL_3D_INCLUDED
//
//#ifdef __cplusplus
//extern "C" {
//#endif
//
//#ifdef CONVHULL_3D_USE_SINGLE_PRECISION
//	typedef float CH_FLOAT;
//#else
//	typedef double CH_FLOAT;
//#endif
//	typedef struct _ch_vertex {
//		union {
//			CH_FLOAT v[3];
//			struct {
//				CH_FLOAT x, y, z;
//			};
//		};
//	} ch_vertex;
//	typedef ch_vertex ch_vec3;
//
//	/* builds the 3-D convexhull */
//	__device__  void convhull_3d_build(/* input arguments */
//		ch_vertex* const in_vertices,            /* vector of input vertices; nVert x 1 */
//		const int nVert,                         /* number of vertices */
//		/* output arguments */
//		int** out_faces,                         /* & of empty int*, output face indices; flat: nOut_faces x 3 */
//		int* nOut_faces);                        /* & of int, number of output face indices */
//
///* exports the vertices, face indices, and face normals, as an 'obj' file, ready for GPU (for 3d convexhulls only) */
//	void convhull_3d_export_obj(/* input arguments */
//		ch_vertex* const vertices,          /* vector of input vertices; nVert x 1 */
//		const int nVert,                    /* number of vertices */
//		int* const faces,                   /* face indices; flat: nFaces x 3 */
//		const int nFaces,                   /* number of faces in hull */
//		const int keepOnlyUsedVerticesFLAG, /* 0: exports in_vertices, 1: exports only used vertices  */
//		char* const obj_filename);          /* obj filename, WITHOUT extension */
//
///* exports the vertices, face indices, and face normals, as an 'm' file, for MatLab verification (for 3d convexhulls only) */
//	void convhull_3d_export_m(/* input arguments */
//		ch_vertex* const vertices,            /* vector of input vertices; nVert x 1 */
//		const int nVert,                      /* number of vertices */
//		int* const faces,                     /* face indices; flat: nFaces x 3 */
//		const int nFaces,                     /* number of faces in hull */
//		char* const m_filename);              /* m filename, WITHOUT extension */
//
///* reads an 'obj' file and extracts only the vertices (for 3d convexhulls only) */
//	void extract_vertices_from_obj_file(/* input arguments */
//		char* const obj_filename,       /* obj filename, WITHOUT extension */
//		/* output arguments */
//		ch_vertex** out_vertices,       /* & of empty ch_vertex*, output vertices; out_nVert x 1 */
//		int* out_nVert);                /* & of int, number of vertices */
//
///**** NEW! ****/
//
///* builds the N-Dimensional convexhull of a grid of points */
//	void convhull_nd_build(/* input arguments */
//		CH_FLOAT* const in_points,               /* Matrix of points in 'd' dimensions; FLAT: nPoints x d */
//		const int nPoints,                       /* number of points */
//		const int d,                             /* Number of dimensions */
//		/* output arguments */
//		int** out_faces,                         /* (&) output face indices; FLAT: nOut_faces x d */
//		CH_FLOAT** out_cf,                       /* (&) contains the coefficients of the planes (set to NULL if not wanted); FLAT: nOut_faces x d */
//		CH_FLOAT** out_df,                       /* (&) contains the constant terms of the planes (set to NULL if not wanted); nOut_faces x 1 */
//		int* nOut_faces);                        /* (&) number of output face indices */
//
///* Computes the Delaunay triangulation (mesh) of an arrangement of points in N-dimensional space */
//	void delaunay_nd_mesh(/* input Arguments */
//		const float* points,                      /* The input points; FLAT: nPoints x nd */
//		const int nPoints,                        /* Number of points */
//		const int nd,                             /* The number of dimensions */
//		/* output Arguments */
//		int** Mesh,                               /* (&) the indices defining the Delaunay triangulation of the points; FLAT: nMesh x (nd+1) */
//		int* nMesh);                              /* (&) Number of triangulations */
//
///**** CUSTOM ALLOCATOR VERSIONS ****/
//
///* builds the 3-D convexhull */
//	__device__  void convhull_3d_build_alloc(/* input arguments */
//		ch_vertex* const in_vertices,            /* vector of input vertices; nVert x 1 */
//		const int nVert,                         /* number of vertices */
//		/* output arguments */
//		int** out_faces,                         /* & of empty int*, output face indices; flat: nOut_faces x 3 */
//		int* nOut_faces,                         /* & of int, number of output face indices */
//		void* allocator);                        /* & of an allocator */
//
///* builds the N-Dimensional convexhull of a grid of points */
//	void convhull_nd_build_alloc(/* input arguments */
//		CH_FLOAT* const in_points,               /* Matrix of points in 'd' dimensions; FLAT: nPoints x d */
//		const int nPoints,                       /* number of points */
//		const int d,                             /* Number of dimensions */
//		/* output arguments */
//		int** out_faces,                         /* (&) output face indices; FLAT: nOut_faces x d */
//		CH_FLOAT** out_cf,                       /* (&) contains the coefficients of the planes (set to NULL if not wanted); FLAT: nOut_faces x d */
//		CH_FLOAT** out_df,                       /* (&) contains the constant terms of the planes (set to NULL if not wanted); nOut_faces x 1 */
//		int* nOut_faces,                         /* (&) number of output face indices */
//		void* allocator);                        /* & of an allocator */
//
///* Computes the Delaunay triangulation (mesh) of an arrangement of points in N-dimensional space */
//	void delaunay_nd_mesh_alloc(/* input Arguments */
//		const float* points,                      /* The input points; FLAT: nPoints x nd */
//		const int nPoints,                        /* Number of points */
//		const int nd,                             /* The number of dimensions */
//		/* output Arguments */
//		int** Mesh,                               /* (&) the indices defining the Delaunay triangulation of the points; FLAT: nMesh x (nd+1) */
//		int* nMesh,                               /* (&) Number of triangulations */
//		void* allocator);                         /* & of an allocator */
//
///* reads an 'obj' file and extracts only the vertices (for 3d convexhulls only) */
//	void extract_vertices_from_obj_file_alloc(/* input arguments */
//		char* const obj_filename,       /* obj filename, WITHOUT extension */
//		/* output arguments */
//		ch_vertex** out_vertices,       /* & of empty ch_vertex*, output vertices; out_nVert x 1 */
//		int* out_nVert,                 /* & of int, number of vertices */
//		void* allocator);               /* & of an allocator */
//
//#ifdef __cplusplus
//} /*extern "C"*/
//#endif
//
//#endif /* CONVHULL_3D_INCLUDED */
//
//
///************
// * INTERNAL:
// ***********/
//#define CONVHULL_3D_ENABLE
//#ifdef CONVHULL_3D_ENABLE
//
//#include <stdlib.h>
//#include <stdio.h>
//#include <math.h>
//#include <string.h>
//#include <float.h>
//#include <ctype.h>
//#include <string.h>
//#include <errno.h> 
//#include <assert.h>
//#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
//#define CV_STRNCPY(a,b,c) strncpy_s(a,c+1,b,c);
//#define CV_STRCAT(a,b) strcat_s(a,sizeof(b),b);
//#else
//#define CV_STRNCPY(a,b,c) strncpy(a,b,c);
//#define CV_STRCAT(a,b) strcat(a,b);
//#endif
//#ifdef CONVHULL_3D_USE_SINGLE_PRECISION
//#define CH_FLT_MIN FLT_MIN
//#define CH_FLT_MAX FLT_MAX
//#define CH_NOISE_VAL 0.00001f
//#define ch_pow powf
//#define ch_sqrt sqrtf
//#else
//#define CH_FLT_MIN DBL_MIN
//#define CH_FLT_MAX DBL_MAX
//#define CH_NOISE_VAL 0.0000001
//#define ch_pow pow
//#define ch_sqrt sqrt
//#endif
//#ifndef MIN
//#define MIN(a,b) (( (a) < (b) ) ? (a) : (b) )
//#endif
//#ifndef MAX
//#define MAX(a,b) (( (a) > (b) ) ? (a) : (b) )
//#endif
//#ifndef ch_malloc
//#define ch_malloc malloc
//#endif
//#ifndef ch_calloc
//#define ch_calloc calloc
//#endif
//#ifndef ch_realloc
//#define ch_realloc realloc
//#endif
//#ifndef ch_free
//#define ch_free free
//#endif
//#ifndef ch_stateful_malloc
//#define ch_stateful_malloc(allocator, size) ch_malloc(size)
//#endif
//#ifndef ch_stateful_calloc
//#define ch_stateful_calloc(allocator, num, size) ch_calloc(num, size)
//#endif
//#ifndef ch_stateful_realloc
//#define ch_stateful_realloc(allocator, ptr, size) ch_realloc(ptr, size)
//#endif
//#ifndef ch_stateful_free
//#define ch_stateful_free(allocator, ptr) ch_free(ptr)
//#endif
//#ifndef ch_stateful_resize
//#define ch_stateful_resize(allocator, ptr, size) default_memory_resize(allocator, ptr, size)
//#define CONVHULL_CREATE_DEFAULT_RESIZE 1
//#endif
//
//#define CH_MAX_NUM_FACES 50000
//#define CONVHULL_3D_MAX_DIMENSIONS 3
//#define CONVHULL_ND_MAX_DIMENSIONS 5
//
// /* structs for qsort */
//typedef struct float_w_idx {
//	CH_FLOAT val;
//	int idx;
//}float_w_idx;
//
///* internal functions prototypes: */
//static int cmp_asc_float(const void*, const void*);
//static int cmp_desc_float(const void*, const void*);
//static int cmp_asc_int(const void*, const void*);
//__device__ static void sort_float(CH_FLOAT*, CH_FLOAT*, int*, int, int, void*);
//__device__ static void sort_int(int*, int);
//static ch_vec3 cross(ch_vec3*, ch_vec3*);
//__device__ static CH_FLOAT det_4x4(CH_FLOAT*);
//__device__ static void plane_3d(CH_FLOAT*, CH_FLOAT*, CH_FLOAT*);
//__device__ static void ismember(int*, int*, int*, int, int);
//
///* internal functions definitions: */
//#ifdef CONVHULL_CREATE_DEFAULT_RESIZE
//__device__ static void* default_memory_resize(void* allocator, void* ptr, size_t size)
//{
//	if (ptr)
//		ch_stateful_free(allocator, ptr);
//	return ch_stateful_malloc(allocator, size);
//}
//#endif
//
//static int cmp_asc_float(const void *a, const void *b) {
//	struct float_w_idx *a1 = (struct float_w_idx*)a;
//	struct float_w_idx *a2 = (struct float_w_idx*)b;
//	if ((*a1).val < (*a2).val)return -1;
//	else if ((*a1).val > (*a2).val)return 1;
//	else return 0;
//}
//
//static int cmp_desc_float(const void *a, const void *b) {
//	struct float_w_idx *a1 = (struct float_w_idx*)a;
//	struct float_w_idx *a2 = (struct float_w_idx*)b;
//	if ((*a1).val > (*a2).val)return -1;
//	else if ((*a1).val < (*a2).val)return 1;
//	else return 0;
//}
//
//static int cmp_asc_int(const void *a, const void *b) {
//	int *a1 = (int*)a;
//	int *a2 = (int*)b;
//	if ((*a1) < (*a2))return -1;
//	else if ((*a1) > (*a2))return 1;
//	else return 0;
//}
//__device__ bool cmp_desc(float_w_idx a, float_w_idx b)
//{
//	return a.val > b.val;
//}
//__device__ bool cmp_asc(float_w_idx a, float_w_idx b)
//{
//	return a.val < b.val;
//}
//__device__ static void sort_float
//(
//	CH_FLOAT* in_vec,  /* vector[len] to be sorted */
//	CH_FLOAT* out_vec, /* if NULL, then in_vec is sorted "in-place" */
//	int* new_idices,   /* set to NULL if you don't need them */
//	int len,           /* number of elements in vectors, must be consistent with the input data */
//	int descendFLAG,   /* !1:ascending, 1:descending */
//	void* allocator    /* (stateful) allocator */
//)
//{
//	int i;
//	struct float_w_idx *data;
//
//	data = (float_w_idx*)ch_stateful_malloc(allocator, len * sizeof(float_w_idx));
//	for (i = 0; i < len; i++) {
//		data[i].val = in_vec[i];
//		data[i].idx = i;
//	}
//	thrust::device_ptr<float_w_idx> pointer(data);
//	if (descendFLAG)
//	{
//		//qsort(data, len, sizeof(data[0]), cmp_desc_float);
//		thrust::sort(pointer, pointer + len, cmp_desc);
//	}
//	else
//	{
//		//qsort(data, len, sizeof(data[0]), cmp_asc_float);
//		thrust::sort(pointer, pointer + len, cmp_asc);
//	}
//	for (i = 0; i < len; i++) {
//		if (out_vec != NULL)
//			out_vec[i] = data[i].val;
//		else
//			in_vec[i] = data[i].val; /* overwrite input vector */
//		if (new_idices != NULL)
//			new_idices[i] = data[i].idx;
//	}
//	ch_stateful_free(allocator, data);
//}
//__device__ bool cmp(int a, int b)
//{
//	return a < b;
//}
//__device__ static void sort_int
//(
//	int* io_vec,     /* vector[len] to be sorted */
//	int len          /* number of elements in vectors, must be consistent with the input data */
//)
//{
//	//qsort(io_vec,len,sizeof(io_vec[0]),cmp_asc_int);
//	thrust::device_ptr<int> pointer(io_vec);
//	thrust::sort(pointer, pointer + len, cmp);
//}
//
//static ch_vec3 cross(ch_vec3* v1, ch_vec3* v2)
//{
//	ch_vec3 cross;
//	cross.x = v1->y * v2->z - v1->z * v2->y;
//	cross.y = v1->z * v2->x - v1->x * v2->z;
//	cross.z = v1->x * v2->y - v1->y * v2->x;
//	return cross;
//}
//
///* calculates the determinent of a 4x4 matrix */
//__device__ static CH_FLOAT det_4x4(CH_FLOAT* m) {
//	return
//		m[3] * m[6] * m[9] * m[12] - m[2] * m[7] * m[9] * m[12] -
//		m[3] * m[5] * m[10] * m[12] + m[1] * m[7] * m[10] * m[12] +
//		m[2] * m[5] * m[11] * m[12] - m[1] * m[6] * m[11] * m[12] -
//		m[3] * m[6] * m[8] * m[13] + m[2] * m[7] * m[8] * m[13] +
//		m[3] * m[4] * m[10] * m[13] - m[0] * m[7] * m[10] * m[13] -
//		m[2] * m[4] * m[11] * m[13] + m[0] * m[6] * m[11] * m[13] +
//		m[3] * m[5] * m[8] * m[14] - m[1] * m[7] * m[8] * m[14] -
//		m[3] * m[4] * m[9] * m[14] + m[0] * m[7] * m[9] * m[14] +
//		m[1] * m[4] * m[11] * m[14] - m[0] * m[5] * m[11] * m[14] -
//		m[2] * m[5] * m[8] * m[15] + m[1] * m[6] * m[8] * m[15] +
//		m[2] * m[4] * m[9] * m[15] - m[0] * m[6] * m[9] * m[15] -
//		m[1] * m[4] * m[10] * m[15] + m[0] * m[5] * m[10] * m[15];
//}
//
///* Helper function for det_NxN()  */
//static void createSubMatrix
//(
//	CH_FLOAT* m,
//	int N,
//	int i,
//	CH_FLOAT* sub_m
//)
//{
//	int j, k;
//	for (j = N, k = 0; j < N * N; j++) {
//		if (j % N != i) { /* i is the index to remove */
//			sub_m[k] = m[j];
//			k++;
//		}
//	}
//}
//
//static CH_FLOAT det_NxN
//(
//	CH_FLOAT* m,
//	int d
//)
//{
//	CH_FLOAT sum;
//	CH_FLOAT sub_m[CONVHULL_ND_MAX_DIMENSIONS*CONVHULL_ND_MAX_DIMENSIONS];
//	int sign;
//
//	if (d == 0)
//		return 1.0;
//	sum = 0.0;
//	sign = 1;
//	for (int i = 0; i < d; i++) {
//		createSubMatrix(m, d, i, sub_m);
//		sum += sign * m[i] * det_NxN(sub_m, d - 1);
//		sign *= -1;
//	}
//	return sum;
//}
//
///* Calculates the coefficients of the equation of a PLANE in 3D.
// * Original Copyright (c) 2014, George Papazafeiropoulos
// * Distributed under the BSD (2-clause) license
// */
//__device__ static void plane_3d
//(
//	CH_FLOAT* p,
//	CH_FLOAT* c,
//	CH_FLOAT* d
//)
//{
//	int i, j, k, l;
//	int r[3];
//	CH_FLOAT sign, det, norm_c;
//	CH_FLOAT pdiff[2][3], pdiff_s[2][2];
//
//	for (i = 0; i < 2; i++)
//		for (j = 0; j < 3; j++)
//			pdiff[i][j] = p[(i + 1) * 3 + j] - p[i * 3 + j];
//	memset(c, 0, 3 * sizeof(CH_FLOAT));
//	sign = 1.0;
//	for (i = 0; i < 3; i++)
//		r[i] = i;
//	for (i = 0; i < 3; i++) {
//		for (j = 0; j < 2; j++) {
//			for (k = 0, l = 0; k < 3; k++) {
//				if (r[k] != i) {
//					pdiff_s[j][l] = pdiff[j][k];
//					l++;
//				}
//			}
//		}
//		det = pdiff_s[0][0] * pdiff_s[1][1] - pdiff_s[1][0] * pdiff_s[0][1];
//		c[i] = sign * det;
//		sign *= -1.0;
//	}
//	norm_c = (CH_FLOAT)0.0;
//	for (i = 0; i < 3; i++)
//		norm_c += (ch_pow(c[i], (CH_FLOAT)2.0));
//	norm_c = ch_sqrt(norm_c);
//	for (i = 0; i < 3; i++)
//		c[i] /= norm_c;
//	(*d) = (CH_FLOAT)0.0;
//	for (i = 0; i < 3; i++)
//		(*d) += -p[i] * c[i];
//}
//
///* Calculates the coefficients of the equation of a PLANE in ND.
// * Original Copyright (c) 2014, George Papazafeiropoulos
// * Distributed under the BSD (2-clause) license
// */
// //static void plane_nd
// //(
// //    const int Nd,
// //    CH_FLOAT* p,
// //    CH_FLOAT* c,
// //    CH_FLOAT* d
// //)
// //{
// //    int i, j, k, l;
// //    int r[CONVHULL_ND_MAX_DIMENSIONS];
// //    CH_FLOAT sign, det, norm_c;
// //    CH_FLOAT pdiff[CONVHULL_ND_MAX_DIMENSIONS-1][CONVHULL_ND_MAX_DIMENSIONS], pdiff_s[(CONVHULL_ND_MAX_DIMENSIONS-1)*(CONVHULL_ND_MAX_DIMENSIONS-1)];
// //
// //    if(Nd==3){
// //        plane_3d(p,c,d);
// //        return;
// //    }
// //
// //    for(i=0; i<Nd-1; i++)
// //        for(j=0; j<Nd; j++)
// //            pdiff[i][j] = p[(i+1)*Nd+j] - p[i*Nd+j];
// //    memset(c, 0, Nd*sizeof(CH_FLOAT));
// //    sign = 1.0;
// //    for(i=0; i<Nd; i++)
// //        r[i] = i;
// //    for(i=0; i<Nd; i++){
// //        for(j=0; j<Nd-1; j++){
// //            for(k=0, l=0; k<Nd; k++){
// //                if(r[k]!=i){
// //                    pdiff_s[j*(Nd-1)+l] = pdiff[j][k];
// //                    l++;
// //                }
// //            }
// //        }
// //        /* Determinant 1 dimension lower */
// //        if(Nd==3)
// //            det = pdiff_s[0*(Nd-1)+0]*pdiff_s[1*(Nd-1)+1] - pdiff_s[1*(Nd-1)+0]*pdiff_s[0*(Nd-1)+1];
// //        else if(Nd==5)
// //            det = det_4x4((CH_FLOAT*)pdiff_s);
// //        else{
// //            det = det_NxN((CH_FLOAT*)pdiff_s, Nd-1);
// //        }
// //        c[i] = sign * det;
// //        sign *= -1.0;
// //    }
// //    norm_c = (CH_FLOAT)0.0;
// //    for(i=0; i<Nd; i++)
// //        norm_c += (ch_pow(c[i], (CH_FLOAT)2.0));
// //    norm_c = ch_sqrt(norm_c);
// //    for(i=0; i<Nd; i++)
// //        c[i] /= norm_c;
// //    (*d) = (CH_FLOAT)0.0;
// //    for(i=0; i<Nd; i++)
// //        (*d) += -p[i] * c[i];
// //}
//
//__device__ static void ismember
//(
//	int* pLeft,          /* left vector; nLeftElements x 1 */
//	int* pRight,         /* right vector; nRightElements x 1 */
//	int* pOut,           /* 0, unless pRight elements are present in pLeft then 1; nLeftElements x 1 */
//	int nLeftElements,   /* number of elements in pLeft */
//	int nRightElements   /* number of elements in pRight */
//)
//{
//	int i, j;
//	memset(pOut, 0, nLeftElements * sizeof(int));
//	for (i = 0; i < nLeftElements; i++)
//		for (j = 0; j < nRightElements; j++)
//			if (pLeft[i] == pRight[j])
//				pOut[i] = 1;
//}
//
//__device__ __host__ static CH_FLOAT rnd(int x, int y)
//{
//	// Reference(s):
//	//
//	// - Improvements to the canonical one-liner GLSL rand() for OpenGL ES 2.0
//	//   http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
//	//
//	CH_FLOAT a = (CH_FLOAT) 12.9898;
//	CH_FLOAT b = (CH_FLOAT) 78.233;
//	CH_FLOAT c = (CH_FLOAT) 43758.5453;
//	CH_FLOAT dt = x * a + y * b;
//#ifdef CONVHULL_3D_USE_SINGLE_PRECISION
//	float sn = fmodf(dt, 3.14f);
//	float intpart;
//	return modff(sinf(sn) * c, &intpart);
//#else
//	double sn = fmod(dt, 3.14);
//	double intpart;
//	return modf(sin(sn) * c, &intpart);
//#endif // CONVHULL_3D_USE_SINGLE_PRECISION
//}
//
///* A C version of the 3D quickhull matlab implementation from here:
// * https://www.mathworks.com/matlabcentral/fileexchange/48509-computational-geometry-toolbox?focused=3851550&tab=example
// * (*out_faces) is returned as NULL, if triangulation fails *
// * Original Copyright (c) 2014, George Papazafeiropoulos
// * Distributed under the BSD (2-clause) license
// * Reference: "The Quickhull Algorithm for Convex Hull, C. Bradford Barber, David P. Dobkin
// *             and Hannu Huhdanpaa, Geometry Center Technical Report GCG53, July 30, 1993"
// */
//__device__ int* my_realloc(int oldsize, int newsize, int* old)
//{
//	int* newT = (int*)malloc(newsize * sizeof(int));
//
//	int size;
//	if (newsize < oldsize)
//	{
//		size = newsize;
//	}
//	else
//	{
//		size = oldsize;
//	}
//	for (int i = 0; i < size; i++)
//	{
//		newT[i] = old[i];
//	}
//
//	free(old);
//	return newT;
//}
//__device__ CH_FLOAT* my_realloc(int oldsize, int newsize, CH_FLOAT* old)
//{
//	CH_FLOAT* newT = (CH_FLOAT*)malloc(newsize * sizeof(CH_FLOAT));
//
//	int size;
//	if (newsize < oldsize)
//	{
//		size = newsize;
//	}
//	else
//	{
//		size = oldsize;
//	}
//	for (int i = 0; i < size; i++)
//	{
//		newT[i] = old[i];
//	}
//
//	free(old);
//	return newT;
//}
//__device__  void convhull_3d_build
//(
//	ch_vertex* const in_vertices,
//	const int nVert,
//	int** out_faces,
//	int* nOut_faces
//)
//{
//	convhull_3d_build_alloc(in_vertices, nVert, out_faces, nOut_faces, NULL);
//}
//
//__device__  void convhull_3d_build_alloc
//(
//	ch_vertex* const in_vertices,
//	const int nVert,
//	int** out_faces,
//	int* nOut_faces,
//	void* allocator
//)
//{
//	int i, j, k, l, h;
//	int nFaces, p, d;
//	int* aVec, *faces;
//	CH_FLOAT dfi, v, max_p, min_p;
//	CH_FLOAT span[CONVHULL_3D_MAX_DIMENSIONS];
//	CH_FLOAT cfi[CONVHULL_3D_MAX_DIMENSIONS];
//	CH_FLOAT p_s[CONVHULL_3D_MAX_DIMENSIONS*CONVHULL_3D_MAX_DIMENSIONS];
//	CH_FLOAT* points, *cf, *df;
//
//	if (nVert <= 3 || in_vertices == NULL) {
//		(*out_faces) = NULL;
//		(*nOut_faces) = 0;
//		return;
//	}
//
//	/* 3 dimensions. The code should theoretically work for >=2 dimensions, but "plane_3d" and "det_4x4" are hardcoded for 3,
//	 * so would need to be rewritten */
//	d = 3;
//
//	/* Add noise to the points */
//	points = (CH_FLOAT*)ch_stateful_malloc(allocator, nVert*(d + 1) * sizeof(CH_FLOAT));
//	for (i = 0; i < nVert; i++) {
//		for (j = 0; j < d; j++)
//			points[i*(d + 1) + j] = in_vertices[i].v[j] + CH_NOISE_VAL * rnd(i, j); /* noise mitigates duplicates */
//		points[i*(d + 1) + d] = 1.0f; /* add a last column of ones. Used only for determinant calculation */
//	}
//
//	/* Find the span */
//	for (j = 0; j < d; j++) {
//		max_p = (CH_FLOAT)-2.23e+13; min_p = (CH_FLOAT)2.23e+13;
//		for (i = 0; i < nVert; i++) {
//			max_p = MAX(max_p, points[i*(d + 1) + j]);
//			min_p = MIN(min_p, points[i*(d + 1) + j]);
//		}
//		span[j] = max_p - min_p;
//#ifndef CONVHULL_ALLOW_BUILD_IN_HIGHER_DIM
//		/* If you hit this assertion error, then the input vertices do not span all 3 dimensions. Therefore the convex hull could be built in less dimensions.
//		 * In these cases, consider reducing the dimensionality of the points and calling convhull_nd_build() instead with d<3
//		 * You can turn this assert off using CONVHULL_ALLOW_BUILD_IN_HIGHER_DIM if you still wish to build in a higher number of dimensions. */
//		assert(span[j] > 0.000000001);
//#endif
//	}
//
//	/* The initial convex hull is a simplex with (d+1) facets, where d is the number of dimensions */
//	nFaces = (d + 1);
//	//faces = (int*)ch_stateful_calloc(allocator, nFaces*d, sizeof(int));
//	faces = (int*)ch_stateful_malloc(allocator, nFaces*d * sizeof(int));
//	memset(faces, 0, sizeof(faces));
//	aVec = (int*)ch_stateful_malloc(allocator, nFaces * sizeof(int));
//	for (i = 0; i < nFaces; i++)
//		aVec[i] = i;
//
//	/* Each column of cf contains the coefficients of a plane */
//	cf = (CH_FLOAT*)ch_stateful_malloc(allocator, nFaces*d * sizeof(CH_FLOAT));
//	df = (CH_FLOAT*)ch_stateful_malloc(allocator, nFaces * sizeof(CH_FLOAT));
//	for (i = 0; i < nFaces; i++) {
//		/* Set the indices of the points defining the face  */
//		for (j = 0, k = 0; j < (d + 1); j++) {
//			if (aVec[j] != i) {
//				faces[i*d + k] = aVec[j];
//				k++;
//			}
//		}
//
//		/* Calculate and store the plane coefficients of the face */
//		for (j = 0; j < d; j++)
//			for (k = 0; k < d; k++)
//				p_s[j*d + k] = points[(faces[i*d + j])*(d + 1) + k];
//
//		/* Calculate and store the plane coefficients of the face */
//		plane_3d(p_s, cfi, &dfi);
//		for (j = 0; j < d; j++)
//			cf[i*d + j] = cfi[j];
//		df[i] = dfi;
//	}
//	CH_FLOAT A[(CONVHULL_3D_MAX_DIMENSIONS + 1)*(CONVHULL_3D_MAX_DIMENSIONS + 1)];
//	int fVec[CONVHULL_3D_MAX_DIMENSIONS + 1];
//	int face_tmp[2];
//
//	/* Check to make sure that faces are correctly oriented */
//	int bVec[CONVHULL_3D_MAX_DIMENSIONS + 1];
//	for (i = 0; i < d + 1; i++)
//		bVec[i] = i;
//
//	/* A contains the coordinates of the points forming a simplex */
//	memset(A, 0, sizeof(A));
//	for (k = 0; k < (d + 1); k++) {
//		/* Get the point that is not on the current face (point p) */
//		for (i = 0; i < d; i++)
//			fVec[i] = faces[k*d + i];
//		sort_int(fVec, d); /* sort ascending */
//		p = k;
//		for (i = 0; i < d; i++)
//			for (j = 0; j < (d + 1); j++)
//				A[i*(d + 1) + j] = points[(faces[k*d + i])*(d + 1) + j];
//		for (; i < (d + 1); i++)
//			for (j = 0; j < (d + 1); j++)
//				A[i*(d + 1) + j] = points[p*(d + 1) + j];
//
//		/* det(A) determines the orientation of the face */
//		v = det_4x4(A);
//
//		/* Orient so that each point on the original simplex can't see the opposite face */
//		if (v < 0) {
//			/* Reverse the order of the last two vertices to change the volume */
//			for (j = 0; j < 2; j++)
//				face_tmp[j] = faces[k*d + d - j - 1];
//			for (j = 0; j < 2; j++)
//				faces[k*d + d - j - 1] = face_tmp[1 - j];
//
//			/* Modify the plane coefficients of the properly oriented faces */
//			for (j = 0; j < d; j++)
//				cf[k*d + j] = -cf[k*d + j];
//			df[k] = -df[k];
//			for (i = 0; i < d; i++)
//				for (j = 0; j < (d + 1); j++)
//					A[i*(d + 1) + j] = points[(faces[k*d + i])*(d + 1) + j];
//			for (; i < (d + 1); i++)
//				for (j = 0; j < (d + 1); j++)
//					A[i*(d + 1) + j] = points[p*(d + 1) + j];
//		}
//	}
//
//	/* Coordinates of the center of the point set */
//	CH_FLOAT meanp[CONVHULL_3D_MAX_DIMENSIONS];
//	CH_FLOAT* absdist, *reldist, *desReldist;
//	memset(meanp, 0, sizeof(meanp));
//	for (i = d + 1; i < nVert; i++)
//		for (j = 0; j < d; j++)
//			meanp[j] += points[i*(d + 1) + j];
//	for (j = 0; j < d; j++)
//		meanp[j] = meanp[j] / (CH_FLOAT)(nVert - d - 1);
//
//	/* Absolute distance of points from the center */
//	absdist = (CH_FLOAT*)ch_stateful_malloc(allocator, (nVert - d - 1)*d * sizeof(CH_FLOAT));
//	for (i = d + 1, k = 0; i < nVert; i++, k++)
//		for (j = 0; j < d; j++)
//			absdist[k*d + j] = (points[i*(d + 1) + j] - meanp[j]) / span[j];
//
//	/* Relative distance of points from the center */
//	//reldist = (CH_FLOAT*)ch_stateful_calloc(allocator, (nVert-d-1), sizeof(CH_FLOAT));
//	reldist = (CH_FLOAT*)ch_stateful_malloc(allocator, (nVert - d - 1) * sizeof(CH_FLOAT));
//	memset(reldist, 0, sizeof(reldist));
//	desReldist = (CH_FLOAT*)ch_stateful_malloc(allocator, (nVert - d - 1) * sizeof(CH_FLOAT));
//	for (i = 0; i < (nVert - d - 1); i++)
//		for (j = 0; j < d; j++)
//			reldist[i] += ch_pow(absdist[i*d + j], (CH_FLOAT)2.0);
//
//	/* Sort from maximum to minimum relative distance */
//	int num_pleft, cnt;
//	int* ind, *pleft;
//	ind = (int*)ch_stateful_malloc(allocator, (nVert - d - 1) * sizeof(int));
//	pleft = (int*)ch_stateful_malloc(allocator, (nVert - d - 1) * sizeof(int));
//	sort_float(reldist, desReldist, ind, (nVert - d - 1), 1, allocator);
//
//	/* Initialize the vector of points left. The points with the larger relative
//	 distance from the center are scanned first. */
//	num_pleft = (nVert - d - 1);
//	for (i = 0; i < num_pleft; i++)
//		pleft[i] = ind[i] + d + 1;
//
//	/* Loop over all remaining points that are not deleted. Deletion of points
//	 occurs every #iter2del# iterations of this while loop */
//	memset(A, 0, sizeof(A));
//
//	/* cnt is equal to the points having been selected without deletion of
//	 nonvisible points (i.e. points inside the current convex hull) */
//	cnt = 0;
//
//	/* The main loop for the quickhull algorithm */
//	CH_FLOAT detA;
//	CH_FLOAT* points_cf;
//	CH_FLOAT points_s[CONVHULL_3D_MAX_DIMENSIONS];
//	int face_s[CONVHULL_3D_MAX_DIMENSIONS];
//	int gVec[CONVHULL_3D_MAX_DIMENSIONS];
//	int* visible_ind, *visible, *nonvisible_faces, *f0, *u, *horizon, *hVec, *pp, *hVec_mem_face;
//	int num_visible_ind, num_nonvisible_faces, n_newfaces, n_realloc_faces, count, vis;
//	int f0_sum, u_len, start, num_p, index, horizon_size1;
//	int FUCKED;
//	FUCKED = 0;
//	/* These pointers need to be assigned NULL as they only use realloc/resize (which act like malloc on a NULL pointer */
//	visible = nonvisible_faces = f0 = u = horizon = hVec = pp = hVec_mem_face = NULL;
//	nFaces = d + 1;
//	int old_nFaces = nFaces;
//	int old_n_realloc_faces = nFaces;
//	visible_ind = (int*)ch_stateful_malloc(allocator, nFaces * sizeof(int));
//	points_cf = (CH_FLOAT*)ch_stateful_malloc(allocator, nFaces * sizeof(CH_FLOAT));
//	while ((num_pleft > 0)) {
//		/* i is the first point of the points left */
//		i = pleft[0];
//
//		/* Delete the point selected */
//		for (j = 0; j < num_pleft - 1; j++)
//			pleft[j] = pleft[j + 1];
//		num_pleft--;
//		if (num_pleft == 0)
//			ch_stateful_free(allocator, pleft);
//		else
//		{
//			//pleft = (int*)ch_stateful_realloc(allocator, pleft, num_pleft * sizeof(int));
//			pleft = my_realloc(num_pleft + 1, num_pleft, pleft);
//		}
//
//		/* Update point selection counter */
//		cnt++;
//
//		/* find visible faces */
//		for (j = 0; j < d; j++)
//			points_s[j] = points[i*(d + 1) + j];
//		//points_cf = (CH_FLOAT*)ch_stateful_realloc(allocator, points_cf, nFaces*sizeof(CH_FLOAT));
//		points_cf = my_realloc(old_nFaces, nFaces, points_cf);
//		//visible_ind = (int*)ch_stateful_realloc(allocator, visible_ind, nFaces*sizeof(int));
//		visible_ind = my_realloc(old_nFaces, nFaces, visible_ind);
//		old_nFaces = nFaces;
//#ifdef CONVHULL_3D_USE_CBLAS
//#ifdef CONVHULL_3D_USE_SINGLE_PRECISION
//		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, nFaces, d, 1.0f,
//			points_s, d,
//			cf, d, 0.0f,
//			points_cf, nFaces);
//#else
//		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, nFaces, d, 1.0,
//			points_s, d,
//			cf, d, 0.0,
//			points_cf, nFaces);
//#endif
//#else
//		for (j = 0; j < nFaces; j++) {
//			points_cf[j] = 0;
//			for (k = 0; k < d; k++)
//				points_cf[j] += points_s[k] * cf[j*d + k];
//		}
//#endif
//		num_visible_ind = 0;
//		for (j = 0; j < nFaces; j++) {
//			if (points_cf[j] + df[j] > 0.0) {
//				num_visible_ind++; /* will sum to 0 if none are visible */
//				visible_ind[j] = 1;
//			}
//			else
//				visible_ind[j] = 0;
//		}
//		num_nonvisible_faces = nFaces - num_visible_ind;
//
//		/* proceed if there are any visible faces */
//		if (num_visible_ind != 0) {
//			/* Find visible face indices */
//			visible = (int*)ch_stateful_resize(allocator, visible, num_visible_ind * sizeof(int));
//			for (j = 0, k = 0; j < nFaces; j++) {
//				if (visible_ind[j] == 1) {
//					visible[k] = j;
//					k++;
//				}
//			}
//
//			/* Find nonvisible faces */
//			nonvisible_faces = (int*)ch_stateful_resize(allocator, nonvisible_faces, num_nonvisible_faces*d * sizeof(int));
//			f0 = (int*)ch_stateful_resize(allocator, f0, num_nonvisible_faces*d * sizeof(int));
//			for (j = 0, k = 0; j < nFaces; j++) {
//				if (visible_ind[j] == 0) {
//					for (l = 0; l < d; l++)
//						nonvisible_faces[k*d + l] = faces[j*d + l];
//					k++;
//				}
//			}
//
//			/* Create horizon (count is the number of the edges of the horizon) */
//			count = 0;
//			for (j = 0; j < num_visible_ind; j++) {
//				/* visible face */
//				vis = visible[j];
//				for (k = 0; k < d; k++)
//					face_s[k] = faces[vis*d + k];
//				sort_int(face_s, d);
//				ismember(nonvisible_faces, face_s, f0, num_nonvisible_faces*d, d);
//				u_len = 0;
//
//				/* u are the nonvisible faces connected to the face v, if any */
//				for (k = 0; k < num_nonvisible_faces; k++) {
//					f0_sum = 0;
//					for (l = 0; l < d; l++)
//						f0_sum += f0[k*d + l];
//					if (f0_sum == d - 1) {
//						u_len++;
//						if (u_len == 1)
//							u = (int*)ch_stateful_resize(allocator, u, u_len * sizeof(int));
//						else
//						{
//							//u = (int*)ch_stateful_realloc(allocator, u, u_len * sizeof(int));
//							u = my_realloc(u_len - 1, u_len, u);
//						}
//						u[u_len - 1] = k;
//					}
//				}
//				for (k = 0; k < u_len; k++) {
//					/* The boundary between the visible face v and the k(th) nonvisible face connected to the face v forms part of the horizon */
//					count++;
//					if (count == 1)
//						horizon = (int*)ch_stateful_resize(allocator, horizon, count*(d - 1) * sizeof(int));
//					else
//					{
//						//horizon = (int*)ch_stateful_realloc(allocator, horizon, count*(d - 1) * sizeof(int));
//						horizon = my_realloc((count - 1)*(d - 1), count*(d - 1), horizon);
//					}
//					for (l = 0; l < d; l++)
//						gVec[l] = nonvisible_faces[u[k] * d + l];
//					for (l = 0, h = 0; l < d; l++) {
//						if (f0[u[k] * d + l]) {
//							horizon[(count - 1)*(d - 1) + h] = gVec[l];
//							h++;
//						}
//					}
//				}
//			}
//			horizon_size1 = count;
//			for (j = 0, l = 0; j < nFaces; j++) {
//				if (!visible_ind[j]) {
//					/* Delete visible faces */
//					for (k = 0; k < d; k++)
//						faces[l*d + k] = faces[j*d + k];
//
//					/* Delete the corresponding plane coefficients of the faces */
//					for (k = 0; k < d; k++)
//						cf[l*d + k] = cf[j*d + k];
//					df[l] = df[j];
//					l++;
//				}
//			}
//
//			/* Update the number of faces */
//			nFaces = nFaces - num_visible_ind;
//
//			/* start is the first row of the new faces */
//			start = nFaces;
//
//			/* Add faces connecting horizon to the new point */
//			n_newfaces = horizon_size1;
//			n_realloc_faces = nFaces + n_newfaces;
//			if (n_realloc_faces > CH_MAX_NUM_FACES)
//				n_realloc_faces = CH_MAX_NUM_FACES + 1;
//			//faces = (int*)ch_stateful_realloc(allocator, faces, n_realloc_faces*d*sizeof(int));
//			faces = my_realloc(old_n_realloc_faces*d, n_realloc_faces*d, faces);
//			//cf = (CH_FLOAT*)ch_stateful_realloc(allocator, cf, n_realloc_faces*d*sizeof(CH_FLOAT));
//			cf = my_realloc(old_n_realloc_faces*d, n_realloc_faces*d, cf);
//			//df = (CH_FLOAT*)ch_stateful_realloc(allocator, df, n_realloc_faces*sizeof(CH_FLOAT));
//			df = my_realloc(old_n_realloc_faces, n_realloc_faces, df);
//
//			old_n_realloc_faces = n_realloc_faces;
//
//			for (j = 0; j < n_newfaces; j++) {
//				nFaces++;
//				for (k = 0; k < d - 1; k++)
//					faces[(nFaces - 1)*d + k] = horizon[j*(d - 1) + k];
//				faces[(nFaces - 1)*d + (d - 1)] = i;
//
//				/* Calculate and store appropriately the plane coefficients of the faces */
//				for (k = 0; k < d; k++)
//					for (l = 0; l < d; l++)
//						p_s[k*d + l] = points[(faces[(nFaces - 1)*d + k])*(d + 1) + l];
//				plane_3d(p_s, cfi, &dfi);
//				for (k = 0; k < d; k++)
//					cf[(nFaces - 1)*d + k] = cfi[k];
//				df[(nFaces - 1)] = dfi;
//				if (nFaces > CH_MAX_NUM_FACES) {
//					FUCKED = 1;
//					nFaces = 0;
//					break;
//				}
//			}
//
//			/* Orient each new face properly */
//			hVec = (int*)ch_stateful_resize(allocator, hVec, nFaces * sizeof(int));
//			hVec_mem_face = (int*)ch_stateful_resize(allocator, hVec_mem_face, nFaces * sizeof(int));
//			for (j = 0; j < nFaces; j++)
//				hVec[j] = j;
//			for (k = start; k < nFaces; k++) {
//				for (j = 0; j < d; j++)
//					face_s[j] = faces[k*d + j];
//				sort_int(face_s, d);
//				ismember(hVec, face_s, hVec_mem_face, nFaces, d);
//				num_p = 0;
//				for (j = 0; j < nFaces; j++)
//					if (!hVec_mem_face[j])
//						num_p++;
//				pp = (int*)ch_stateful_resize(allocator, pp, num_p * sizeof(int));
//				for (j = 0, l = 0; j < nFaces; j++) {
//					if (!hVec_mem_face[j]) {
//						pp[l] = hVec[j];
//						l++;
//					}
//				}
//				index = 0;
//				detA = 0.0;
//
//				/* While new point is coplanar, choose another point */
//				while (detA == 0.0) {
//					for (j = 0; j < d; j++)
//						for (l = 0; l < d + 1; l++)
//							A[j*(d + 1) + l] = points[(faces[k*d + j])*(d + 1) + l];
//					for (; j < d + 1; j++)
//						for (l = 0; l < d + 1; l++)
//							A[j*(d + 1) + l] = points[pp[index] * (d + 1) + l];
//					index++;
//					detA = det_4x4(A);
//				}
//
//				/* Orient faces so that each point on the original simplex can't see the opposite face */
//				if (detA < 0.0) {
//					/* If orientation is improper, reverse the order to change the volume sign */
//					for (j = 0; j < 2; j++)
//						face_tmp[j] = faces[k*d + d - j - 1];
//					for (j = 0; j < 2; j++)
//						faces[k*d + d - j - 1] = face_tmp[1 - j];
//
//					/* Modify the plane coefficients of the properly oriented faces */
//					for (j = 0; j < d; j++)
//						cf[k*d + j] = -cf[k*d + j];
//					df[k] = -df[k];
//					for (l = 0; l < d; l++)
//						for (j = 0; j < d + 1; j++)
//							A[l*(d + 1) + j] = points[(faces[k*d + l])*(d + 1) + j];
//					for (; l < d + 1; l++)
//						for (j = 0; j < d + 1; j++)
//							A[l*(d + 1) + j] = points[pp[index] * (d + 1) + j];
//#ifndef NDEBUG
//					/* Check */
//					detA = det_4x4(A);
//					/* If you hit this assertion error, then the face cannot be properly orientated */
//					assert(detA > 0.0);
//#endif
//				}
//			}
//		}
//		if (FUCKED) {
//			break;
//		}
//	}
//
//	/* output */
//	if (FUCKED) {
//		(*out_faces) = NULL;
//		(*nOut_faces) = 0;
//	}
//	else {
//		(*out_faces) = (int*)ch_stateful_malloc(allocator, nFaces*d * sizeof(int));
//		memcpy((*out_faces), faces, nFaces*d * sizeof(int));
//		(*nOut_faces) = nFaces;
//	}
//
//	/* clean-up */
//	ch_stateful_free(allocator, u);
//	ch_stateful_free(allocator, pp);
//	ch_stateful_free(allocator, horizon);
//	ch_stateful_free(allocator, f0);
//	ch_stateful_free(allocator, nonvisible_faces);
//	ch_stateful_free(allocator, visible);
//	ch_stateful_free(allocator, hVec);
//	ch_stateful_free(allocator, hVec_mem_face);
//	ch_stateful_free(allocator, visible_ind);
//	ch_stateful_free(allocator, points_cf);
//	ch_stateful_free(allocator, absdist);
//	ch_stateful_free(allocator, reldist);
//	ch_stateful_free(allocator, desReldist);
//	ch_stateful_free(allocator, ind);
//	ch_stateful_free(allocator, points);
//	ch_stateful_free(allocator, faces);
//	ch_stateful_free(allocator, aVec);
//	ch_stateful_free(allocator, cf);
//	ch_stateful_free(allocator, df);
//}
//
//void convhull_3d_export_obj
//(
//	ch_vertex* const vertices,
//	const int nVert,
//	int* const faces,
//	const int nFaces,
//	const int keepOnlyUsedVerticesFLAG,
//	char* const obj_filename
//)
//{
//	int i, j;
//	char path[256] = "\0";
//	CV_STRNCPY(path, obj_filename, strlen(obj_filename));
//	FILE* obj_file;
//#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
//	CV_STRCAT(path, ".obj");
//	fopen_s(&obj_file, path, "wt");
//#else
//	errno = 0;
//	obj_file = fopen(strcat(path, ".obj"), "wt");
//#endif
//	if (obj_file == NULL) {
//		printf("Error %d \n", errno);
//		printf("It's null");
//	}
//	fprintf(obj_file, "o\n");
//	CH_FLOAT scale;
//	ch_vec3 v1, v2, normal;
//
//	/* export vertices */
//	if (keepOnlyUsedVerticesFLAG) {
//		for (i = 0; i < nFaces; i++)
//			for (j = 0; j < 3; j++)
//				fprintf(obj_file, "v %f %f %f\n", vertices[faces[i * 3 + j]].x,
//					vertices[faces[i * 3 + j]].y, vertices[faces[i * 3 + j]].z);
//	}
//	else {
//		for (i = 0; i < nVert; i++)
//			fprintf(obj_file, "v %f %f %f\n", vertices[i].x,
//				vertices[i].y, vertices[i].z);
//	}
//
//	/* export the face normals */
//	for (i = 0; i < nFaces; i++) {
//		/* calculate cross product between v1-v0 and v2-v0 */
//		v1 = vertices[faces[i * 3 + 1]];
//		v2 = vertices[faces[i * 3 + 2]];
//		v1.x -= vertices[faces[i * 3]].x;
//		v1.y -= vertices[faces[i * 3]].y;
//		v1.z -= vertices[faces[i * 3]].z;
//		v2.x -= vertices[faces[i * 3]].x;
//		v2.y -= vertices[faces[i * 3]].y;
//		v2.z -= vertices[faces[i * 3]].z;
//		normal = cross(&v1, &v2);
//
//		/* normalise to unit length */
//		scale = ((CH_FLOAT)1.0) / (ch_sqrt(ch_pow(normal.x, (CH_FLOAT)2.0) + ch_pow(normal.y, (CH_FLOAT)2.0) + ch_pow(normal.z, (CH_FLOAT)2.0)) + (CH_FLOAT)2.23e-9);
//		normal.x *= scale;
//		normal.y *= scale;
//		normal.z *= scale;
//		fprintf(obj_file, "vn %f %f %f\n", normal.x, normal.y, normal.z);
//	}
//
//	/* export the face indices */
//	if (keepOnlyUsedVerticesFLAG) {
//		for (i = 0; i < nFaces; i++) {
//			/* vertices are in same order as the faces, and normals are in order */
//			fprintf(obj_file, "f %u//%u %u//%u %u//%u\n",
//				i * 3 + 1, i + 1,
//				i * 3 + 1 + 1, i + 1,
//				i * 3 + 2 + 1, i + 1);
//		}
//	}
//	else {
//		/* just normals are in order  */
//		for (i = 0; i < nFaces; i++) {
//			fprintf(obj_file, "f %u//%u %u//%u %u//%u\n",
//				faces[i * 3] + 1, i + 1,
//				faces[i * 3 + 1] + 1, i + 1,
//				faces[i * 3 + 2] + 1, i + 1);
//		}
//	}
//	fclose(obj_file);
//}
//
//void convhull_3d_export_m
//(
//	ch_vertex* const vertices,
//	const int nVert,
//	int* const faces,
//	const int nFaces,
//	char* const m_filename
//)
//{
//	int i;
//	char path[256] = { "\0" };
//	memcpy(path, m_filename, strlen(m_filename));
//	FILE* m_file;
//#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
//	CV_STRCAT(path, ".m");
//	fopen_s(&m_file, path, "wt");
//#else
//	m_file = fopen(strcat(path, ".m"), "wt");
//#endif
//
//	/* save face indices and vertices for verification in matlab: */
//	fprintf(m_file, "vertices = [\n");
//	for (i = 0; i < nVert; i++)
//		fprintf(m_file, "%f, %f, %f;\n", vertices[i].x, vertices[i].y, vertices[i].z);
//	fprintf(m_file, "];\n\n\n");
//	fprintf(m_file, "faces = [\n");
//	for (i = 0; i < nFaces; i++) {
//		fprintf(m_file, " %u, %u, %u;\n",
//			faces[3 * i + 0] + 1,
//			faces[3 * i + 1] + 1,
//			faces[3 * i + 2] + 1);
//	}
//	fprintf(m_file, "];\n\n\n");
//	fclose(m_file);
//}
//
//void extract_vertices_from_obj_file
//(
//	char* const obj_filename,
//	ch_vertex** out_vertices,
//	int* out_nVert)
//{
//	extract_vertices_from_obj_file_alloc(obj_filename, out_vertices, out_nVert, NULL);
//}
//
//void extract_vertices_from_obj_file_alloc
//(
//	char* const obj_filename,
//	ch_vertex** out_vertices,
//	int* out_nVert,
//	void* allocator
//)
//{
//	FILE* obj_file;
//#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
//	CV_STRCAT(obj_filename, ".obj");
//	fopen_s(&obj_file, obj_filename, "r");
//#else
//	obj_file = fopen(strcat(obj_filename, ".obj"), "r");
//#endif 
//
//	/* determine number of vertices */
//	unsigned int nVert = 0;
//	char line[256];
//	while (fgets(line, sizeof(line), obj_file)) {
//		char* vexists = strstr(line, "v ");
//		if (vexists != NULL)
//			nVert++;
//	}
//	(*out_nVert) = nVert;
//	(*out_vertices) = (ch_vertex*)ch_stateful_malloc(allocator, nVert * sizeof(ch_vertex));
//
//	/* extract the vertices */
//	rewind(obj_file);
//	int i = 0;
//	int vertID, prev_char_isDigit, current_char_isDigit;
//	char vert_char[256] = { 0 };
//	while (fgets(line, sizeof(line), obj_file)) {
//		char* vexists = strstr(line, "v ");
//		if (vexists != NULL) {
//			prev_char_isDigit = 0;
//			vertID = -1;
//			for (size_t j = 0; j < strlen(line) - 1; j++) {
//				if (isdigit(line[j]) || line[j] == '.' || line[j] == '-' || line[j] == '+' || line[j] == 'E' || line[j] == 'e') {
//					vert_char[strlen(vert_char)] = line[j];
//					current_char_isDigit = 1;
//				}
//				else
//					current_char_isDigit = 0;
//				if ((prev_char_isDigit && !current_char_isDigit) || j == strlen(line) - 2) {
//					vertID++;
//					if (vertID > 4) {
//						/* not a valid file */
//						ch_stateful_free(allocator, (*out_vertices));
//						(*out_vertices) = NULL;
//						(*out_nVert) = 0;
//						return;
//					}
//					(*out_vertices)[i].v[vertID] = (CH_FLOAT)atof(vert_char);
//					memset(vert_char, 0, 256 * sizeof(char));
//				}
//				prev_char_isDigit = current_char_isDigit;
//			}
//			i++;
//		}
//	}
//}
//
//#endif 



extern __global__ void NodeComputing(int plines, BeamPlugin** G, int* beamNumVec, float3* positions, 
	Triangle* AllMesh, QuadFace* AllQuad, Point* Alldots,
	 short* AllFlag, NodeStatistic* statis, int batchIdx, int batchSize, int batchNum);

__device__ void cuda_recursiveOptimalCut(BeamPlugin** beams, int i, int size, float3* direct, double* intersectionLength)
{
	//optimal cut
	//for (int i = cutId; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (j == i)continue;
			BeamPlugin* b1 = beams[i];
			BeamPlugin* b2 = beams[j];
			double r1 = b1->radius, r2 = b2->radius;

			double cosVal = cuda_Dot(direct[i], direct[j]) / (cuda_Norm(direct[i])*cuda_Norm(direct[j]));
			double angle = acos(cosVal);
			/*if (angle >= PI / 2)
				continue;*/

			double length1 = sqrt(pow(r1, 2) + pow(intersectionLength[i], 2));
			double itoj = angle - atan(r1 / intersectionLength[i]);
			double cut1 = length1 * cos(itoj);
			if (cut1 > intersectionLength[j])
			{
				intersectionLength[j] = cut1;
				cuda_recursiveOptimalCut(beams, j, size, direct, intersectionLength);
			}

			double length2 = sqrt(pow(r2, 2) + pow(intersectionLength[j], 2));
			double jtoi = angle - atan(r2 / intersectionLength[j]);
			double cut2 = length2 * cos(jtoi);
			if (cut2 > intersectionLength[i])
			{
				intersectionLength[i] = cut2;
				cuda_recursiveOptimalCut(beams, i, size, direct, intersectionLength);
			}
		}
	}
}
__device__ void HeapInit(Heap* heap,Compare compare) {
	if (heap == NULL) {
		return;//非法输入
	}
	heap->heapSize = 0;
	heap->cmp = compare;
	return;
}
__device__ void Swap(BeamPlugin* a, BeamPlugin* b) {
	BeamPlugin tmp;
	tmp = *a;
	*a = *b;
	*b = tmp;
	return;
}
__device__ bool cmp1(BeamPlugin a, BeamPlugin b)
{
	return a.cutLength > b.cutLength;
}
__device__ bool cmp2(BeamPlugin a, BeamPlugin b)
{
	return a.cutOtherLength > b.cutOtherLength;
}
__device__ void AdjustUp(BeamPlugin data[], size_t size, Compare cmp, size_t index) {
	if (index >= size) {
		return;
	}
	//1.先找到当前节点对应的父节点
	size_t child = index;
	size_t parent = (child - 1) / 2;
	while (child > 0) {
		//2.比较父节点和子节点的大小关系，如果子节点值比父节点小，交换父子节点的值，如果子节点的值比父节点的大，说明调整也完成了
		if (cmp(data[child], data[parent])) {
			Swap(&data[child], &data[parent]);
		}
		else {
			break;
		}

		//3.将当前父节点作为新的子节点，再去找子节点的父节点，循环进行比较和交换
		child = parent;
		parent = (child - 1) / 2;
	}
	//4.子节点下标等于0，循环结束
	return;
}
__device__ void HeapInsert(Heap* heap, BeamPlugin value) {
	if (heap == NULL) {
		return;
	}
	if (heap->heapSize >= 60) {
		return;//堆满了
	}
	heap->b[heap->heapSize++] = value;
	AdjustUp(heap->b, heap->heapSize,heap->cmp, heap->heapSize - 1);
}
__device__ BeamPlugin HeapRoot(Heap* heap) {
	if (heap == NULL) {
		return ;
	}
	//*value = heap->data[0];
	return heap->b[0];
}
__device__ void AdjustDown(BeamPlugin data[], size_t size,Compare cmp, size_t index) {
	//1.设定parent指向开始的位置，找到对应的子树节点
	size_t parent = index;
	//2.设定一个child指向parent的左子树
	size_t child = parent * 2 + 1;
	//3.判定child和child+1的大小关系，如果child+1的值比child小，就让child = child + 1
	while (child < size) {
		if (child + 1 < size && cmp(data[child + 1], data[child])) {
			child = child + 1;
		}
		//4.判定parent和child的值的大小关系，如果parent比child的值打，就进行交换
		if (cmp(data[child], data[parent])) {
			Swap(&data[child], &data[parent]);
		}
		else {
			//否则就说明调整已经完成
			break;
		}
		//5.parent赋值为child，child再重新复制成parent的做孩子节点
		parent = child;
		child = parent * 2 + 1;

	}
}
__device__ void HeapErase(Heap* heap) {
	if (heap == NULL) {
		return;//非法输入
	}
	if (heap->heapSize == 0) {
		return;//堆为空
	}
	Swap(&heap->b[0], &heap->b[heap->heapSize - 1]);
	--heap->heapSize;
	AdjustDown(heap->b, heap->heapSize,heap->cmp, 0);
}
__device__ int HeapEmpty(Heap* heap) {
	if (heap == NULL) {
		return 0;
	}
	return heap->heapSize == 0 ? 1 : 0;
}
__device__ void cuda_computeIntersectionLength(BeamPlugin** beams, int size,float3 startP, float& longestLength,int nodeid)
{
	//printf("hhh");
	double threshold = 4.0/4;//0.2*6;
	Heap *N[40]; 
	for (int i = 0; i < size; i++)
	{
		N[i] = (Heap*)malloc(sizeof(Heap));
		HeapInit(N[i], cmp2);
	}
	bool visited[40] = { false };
	BeamPlugin* tmpBeams = (BeamPlugin*)malloc(size * sizeof(BeamPlugin));
	for (int i = 0; i < size; i++)
	{
		tmpBeams[i] = *beams[i];
		tmpBeams[i].beamId = i;
		tmpBeams[i].cutLength = 0;
	}
	double* intersectionLength = (double*)malloc(size * sizeof(double));// = new double[size];
	for (int i = 0; i < size; i++)intersectionLength[i] = 0;
	//fill(intersectionLength, intersectionLength + size, 0);
	float3* direct = (float3*)malloc(size * sizeof(float3));//float3* direct = new float3[size];
	/*double* faceAngle = (double*)malloc((size*(size - 1) / 2) * sizeof(double)); int cnt = 0;*/
	for (int i = 0; i < size - 1; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			BeamPlugin* b1 = &tmpBeams[i];
			BeamPlugin* b2 = &tmpBeams[j];
			double r1 = b1->radius, r2 = b2->radius;
			Edge e1 = b1->axis; Edge e2 = b2->axis;
			float3 st, ed;
			if (sqrt(pow(startP.x - b1->axis.p[0].x, 2) + pow(startP.y - b1->axis.p[0].y, 2) + pow(startP.z - b1->axis.p[0].z, 2)) < 1e-6)
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
			direc1 = cuda_make_norm(direc1); direct[i] = direc1;
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
			direc2 = cuda_make_norm(direc2);  direct[j] = direc2;
			double cosVal = cuda_Dot(direc1, direc2) / (cuda_Norm(direc1)*cuda_Norm(direc2));
			if (abs(cosVal + 1) < 1e-1)
			{
				continue;
			}
			double angle = acos(Min(Max(cosVal,-1.0f),1.0f));//是弧度角Min(Max(cuda_Dot(t->normal, quadT->normal), -1.0f), 1.0f)
			//statis[nodeid].face_angle = Min(statis[nodeid].face_angle, angle);
			/*faceAngle[cnt++] = angle;
			statis[nodeid].face_angle_variance += angle;*/
			double beta = atan((r1 + r2 * cos(angle)) / (r2*sin(angle)));
			double alpha = beta + angle - PI / 2;
			double ltmp1 = r1 / tan(alpha), ltmp2 = tan(beta)*r2;
			if (ltmp1 < 0)
			{
				ltmp1 = 0; ltmp2 = tan(PI - angle)*b2->radius;
			}
			if (ltmp2 < 0)
			{
				ltmp2 = 0; ltmp1 = tan(PI - angle)*b1->radius;
			}
			if (ltmp1 > 0 || ltmp2 > 0)
			{
				BeamPlugin b2tmp = *b2;
				b2tmp.cutOtherLength = ltmp1;
				b2tmp.cutLength = ltmp2;
				//N[i].b[N[i].heapSize++]=(b2tmp);
				HeapInsert(N[i], b2tmp);
				BeamPlugin b1tmp = *b1;
				b1tmp.cutOtherLength = ltmp2;
				b1tmp.cutLength = ltmp1;
				//N[j].b[N[j].heapSize++]=(b1tmp);
				HeapInsert(N[j], b1tmp);
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
			/*if (ltmp1 > intersectionLength[i])
				intersectionLength[i] = ltmp1;
			if (ltmp2 > intersectionLength[j])
				intersectionLength[j] = ltmp2;*/
		}
	}
	/*statis[nodeid].face_angle_variance /= (size*(size - 1)) / 2; double v = 0;
	for (int i = 0; i < (size*(size - 1)) / 2; i++)
	{
		v += pow(faceAngle[i]-statis[nodeid].face_angle_variance, 2);

	}
	statis[nodeid].face_angle_variance = v / ((size*(size - 1)) / 2); printf("%f ", statis[nodeid].face_angle_variance);*/
	Heap *pq = (Heap*)malloc(sizeof(Heap)); 
	HeapInit(pq, cmp1);
	for (int i = 0; i < size; i++)
	{
		//pq.b[pq.heapSize++] = (tmpBeams[i]);
		longestLength = Max(tmpBeams[i].cutLength + threshold,longestLength);
		HeapInsert(pq, tmpBeams[i]);
		if (N[i]->heapSize>0)
		{
			//make_heap(N[i].begin(), N[i].end(), CMP2);

		}
	}
	while (pq->heapSize > 0)
	{
		BeamPlugin* self = &(pq->b[0]); //printf("%f ", self->cutLength);
		int beamid = self->beamId;

		while (N[beamid]->heapSize > 0)
		{
			BeamPlugin neighbor = N[beamid]->b[0];
			while (visited[neighbor.beamId])
			{
				HeapErase(N[beamid]);
				if (N[beamid]->heapSize>0)
					neighbor = N[beamid]->b[0];
				else
					break;
			}
			if (N[beamid]->heapSize==0)
			{
				visited[beamid] = true;
				HeapErase(pq);
				//pop_heap(pq.begin(), pq.end(), CMP);
				//pq.pop_back();
				break;
			}
			if (self->cutLength != N[beamid]->b[0].cutOtherLength)
			{
				self->cutLength = N[beamid]->b[0].cutOtherLength;
				AdjustDown(pq->b, pq->heapSize,pq->cmp, 0);
				//make_heap(pq.begin(), pq.end(), CMP);
				break;
			}

			if (self->cutLength > intersectionLength[beamid])
				intersectionLength[beamid] = self->cutLength;
			while (N[beamid]->heapSize>0)
			{
				double cosVal = cuda_Dot(direct[beamid], direct[neighbor.beamId]) / (cuda_Norm(direct[beamid])*cuda_Norm(direct[neighbor.beamId]));
				if (cosVal < 0)
				{
					//重新算neighbor要cut的长度，可能更短了
					double theta1 = acos(min(max(cosVal, -1.0), 1.0));
					double L1 = (self->radius + neighbor.radius*cosVal) / sin(theta1);
					double theta2 = acos(min(max(cosVal, -1.0), 1.0)) - PI / 2;
					double tmp = intersectionLength[beamid] / cos(theta2);
					double L2 = 0;
					if (tmp < neighbor.radius)
					{
						double tmp2 = neighbor.radius - tmp;
						L2 = tmp2 / tan(theta2);
					}
					double L = min(L1, L2);//取两者中较小值
					if (L > intersectionLength[neighbor.beamId])
						intersectionLength[neighbor.beamId] = L;
				}
				HeapErase(N[beamid]);
				if (N[beamid]->heapSize>0)
					neighbor = N[beamid]->b[0];
				else
					break;
				while (visited[neighbor.beamId])
				{
					HeapErase(N[beamid]);
					/*pop_heap(N[beamid].begin(), N[beamid].end(), CMP2);
					N[beamid].pop_back();*/
					if (N[beamid]->heapSize>0)
						neighbor = N[beamid]->b[0];
					else
						break;
				}
			}

			if (N[beamid]->heapSize==0)
			{
				visited[beamid] = true;
				HeapErase(pq);
				/*pop_heap(pq.begin(), pq.end(), CMP);
				pq.pop_back();*/
				break;
			}
		}
	}
	//printf("\n");
	//optimal cut
	//stack<int> stk; stack<int> stk2;
	//int stk[100]; int stk2[100];
	//int cnt = 0,cnt2=0;
	//for (int cutId = 0; cutId < size; cutId++)
	//{
	//	//recursiveOptimalCut(beams, i, size, direct, intersectionLength);
	//	//stk.push(cutId);
	//	stk[cnt] = cutId;
	//	cnt++;
	//	while (cnt>0)
	//	{
	//		int i = stk[cnt-1];
	//		int j = 0;
	//		if (cnt2 == cnt)
	//		{
	//			j = stk2[cnt2-1];
	//			j++;
	//			cnt2--;
	//			//stk2.pop();
	//		}
	//		for (; j < size; j++)
	//		{
	//			if (j == i)continue;
	//			BeamPlugin* b1 = beams[i];
	//			BeamPlugin* b2 = beams[j];
	//			double r1 = b1->radius, r2 = b2->radius;

	//			double cosVal = cuda_Dot(direct[i], direct[j]) / (cuda_Norm(direct[i])*cuda_Norm(direct[j]));
	//			double angle = acos(Min(Max(cosVal,-1.0f),1.0f));
	//			/*if (abs(cosVal+1)<0.31||angle>3*PI/4)
	//			{
	//				continue;
	//			}*/
	//			/*if (angle >= PI / 2)
	//				continue;*/

	//			double length1 = sqrt(pow(r1, 2) + pow(intersectionLength[i], 2));
	//			double itoj = angle - atan(r1 / intersectionLength[i]);
	//			double cut1 = length1 * cos(itoj);
	//			if (cut1 > intersectionLength[j])
	//			{
	//				intersectionLength[j] = cut1;
	//				stk[cnt] = j; cnt++;
	//				stk2[cnt2] = j; cnt2++;
	//				break;
	//				//cuda_recursiveOptimalCut(beams, j, size, direct, intersectionLength);
	//			}

	//			double length2 = sqrt(pow(r2, 2) + pow(intersectionLength[j], 2));
	//			double jtoi = angle - atan(r2 / intersectionLength[j]);
	//			double cut2 = length2 * cos(jtoi);
	//			if (cut2 > intersectionLength[i])
	//			{
	//				intersectionLength[i] = cut2;
	//				stk[cnt] = i; cnt++;
	//				stk2[cnt2] = j; cnt2++;
	//				break;
	//				//cuda_recursiveOptimalCut(beams, i, size, direct, intersectionLength);
	//			}
	//		}
	//		if (j == size)
	//		{
	//			cnt--;
	//			//stk.pop();
	//		}
	//	}
	//}
	//for (int i = 0; i < size; i++)
	//{
	//	for (int j = 0; j < size; j++)
	//	{
	//		if (j == i)continue;
	//		BeamPlugin* b1 = beams[i];
	//		BeamPlugin* b2 = beams[j];
	//		double r1 = b1->radius, r2 = b2->radius;

	//		double cosVal = cuda_Dot(direct[i], direct[j]) / (cuda_Norm(direct[i])*cuda_Norm(direct[j]));
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
	free(pq);
	free(tmpBeams);
	free(direct);
	//double maxLength = intersectionLength[0];
	for (int i = 0; i < size; i++)
	{
		free(N[i]);
		intersectionLength[i] += threshold;
		/*if (intersectionLength[i] > maxLength)
			maxLength = intersectionLength[i];*/
	}
	//maxLength += threshold;
	//找到推进距离之后更新axis的起点和length
	for (int i = 0; i < size; i++)
	{
		BeamPlugin* b = beams[i];
		float3 start, end;
		int flag = 0;
		if (sqrt(pow(startP.x - b->axis.p[0].x, 2) + pow(startP.y - b->axis.p[0].y, 2) + pow(startP.z - b->axis.p[0].z, 2)) < 1e-6)
		{
			start = b->axis.p[0];
			end = b->axis.p[1];
			//std::cout << b.axis.p1[0]<<" "<< b.axis.p1[1]<<" "<< b.axis.p1[2]<<" "<<b.axis.p2[0]<<" "<< b.axis.p2[1] << " "<<b.axis.p2[2];
		}
		else
		{
			start = b->axis.p[1];
			end = b->axis.p[0];
			flag = 1;
			//std::cout << b.axis.p2[0] << " " << b.axis.p2[1] << " " << b.axis.p2[2] << " " << b.axis.p1[0] << " " << b.axis.p1[1] << " " << b.axis.p1[2];
		}
		float3 direction = make_float3(end.x - start.x, end.y - start.y, end.z - start.z);
		direction = cuda_make_norm(direction);
		start.x += direction.x* intersectionLength[i];
		start.y += direction.y* intersectionLength[i];
		start.z += direction.z* intersectionLength[i];
		if (!flag)
		{
			b->axis.p[0] = start; b->longestLength[0] = longestLength;
			//std::cout<< " "<< maxLength<<std::endl;
		}
		else
		{
			b->axis.p[1] = start; b->longestLength[1] = longestLength;
			//std::cout<<" " << maxLength << std::endl;
		}
		atomicAdd(&(b->length), -intersectionLength[i]);
		//b->length -= intersectionLength[i];
		//printf("%f ", b->length);
		if (b->length <= 0)printf("cutting too much!-%f %f\n",b->radius,b->originLength);
		if (b->longestLength[0] + b->longestLength[1] >= b->originLength)printf("optimize too much! %f %f\n", b->radius,b->originLength);
		if(b->longestLength[0]+b->convexLength[1]>=b->originLength)printf("optimize too much! %f %f\n", b->radius, b->originLength);
		if(b->convexLength[0]+b->longestLength[1]>=b->originLength)printf("optimize too much! %f %f\n", b->radius, b->originLength);
	}
	//printf("\n");
	free(intersectionLength);
}
__device__ void cuda_convexCuttingLength(BeamPlugin** beams, int size, float3 startP, int nodeid)
{
	double threshold = 4.0/4;
	double* intersectionLength = (double*)malloc(size * sizeof(double));// = new double[size];
	for (int i = 0; i < size; i++)intersectionLength[i] = 0;
	//fill(intersectionLength, intersectionLength + size, 0);
	float3* direct = (float3*)malloc(size * sizeof(float3));//float3* direct = new float3[size];
	short* flag = (short*)malloc(size * sizeof(short));
	/*double* faceAngle = (double*)malloc((size*(size - 1) / 2) * sizeof(double)); int cnt = 0;*/
	for (int i = 0; i < size - 1; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			BeamPlugin* b1 = beams[i];
			BeamPlugin* b2 = beams[j];
			double r1 = b1->radius, r2 = b2->radius;
			Edge e1 = b1->axis; Edge e2 = b2->axis;
			float3 st, ed;
			if (sqrt(pow(startP.x - b1->axis.p[0].x, 2) + pow(startP.y - b1->axis.p[0].y, 2) + pow(startP.z - b1->axis.p[0].z, 2)) < 1e-6)
			{
				st = b1->axis.p[0];
				ed = b1->axis.p[1]; flag[i] = 0;
			}
			else
			{
				st = b1->axis.p[1];
				ed = b1->axis.p[0]; flag[i] = 1;
			}
			float3 direc1 = make_float3(ed.x - st.x, ed.y - st.y, ed.z - st.z);
			direc1 = cuda_make_norm(direc1); direct[i] = direc1;
			if (sqrt(pow(startP.x - b2->axis.p[0].x, 2) + pow(startP.y - b2->axis.p[0].y, 2) + pow(startP.z - b2->axis.p[0].z, 2)) < 1e-6)
			{
				st = b2->axis.p[0];
				ed = b2->axis.p[1]; flag[j] = 0;
			}
			else
			{
				st = b2->axis.p[1];
				ed = b2->axis.p[0]; flag[j] = 1;
			}
			float3 direc2 = make_float3(ed.x - st.x, ed.y - st.y, ed.z - st.z);
			direc2 = cuda_make_norm(direc2);  direct[j] = direc2;
			double cosVal = cuda_Dot(direc1, direc2) / (cuda_Norm(direc1)*cuda_Norm(direc2));
			if (abs(cosVal + 1) < 1e-1)
			{
				continue;
			}
			double angle = acos(Min(Max(cosVal, -1.0f), 1.0f));//是弧度角Min(Max(cuda_Dot(t->normal, quadT->normal), -1.0f), 1.0f)
			//statis[nodeid].face_angle = Min(statis[nodeid].face_angle, angle);
			/*faceAngle[cnt++] = angle;
			statis[nodeid].face_angle_variance += angle;*/
			double beta = atan((r1 + r2 * cos(angle)) / (r2*sin(angle)));
			double alpha = beta + angle - PI / 2;
			double ltmp1 = r1 / tan(alpha), ltmp2 = tan(beta)*r2;
			if (ltmp1 < 0)
			{
				ltmp1 = 0; ltmp2 = tan(PI - angle)*b2->radius;
			}
			if (ltmp2 < 0)
			{
				ltmp2 = 0; ltmp1 = tan(PI - angle)*b1->radius;
			}
			//if (ltmp1 > 0 || ltmp2 > 0)
			//{
			//	BeamPlugin b2tmp = *b2;
			//	b2tmp.cutOtherLength = ltmp1;
			//	b2tmp.cutLength = ltmp2;
			//	//N[i].b[N[i].heapSize++]=(b2tmp);
			//	HeapInsert(N[i], b2tmp);
			//	BeamPlugin b1tmp = *b1;
			//	b1tmp.cutOtherLength = ltmp2;
			//	b1tmp.cutLength = ltmp1;
			//	//N[j].b[N[j].heapSize++]=(b1tmp);
			//	HeapInsert(N[j], b1tmp);
			//}
			//if (ltmp1 > b1->cutLength)
			//{
			//	//intersectionLength[i] = ltmp1;
			//	b1->cutLength = ltmp1;
			//}
			//if (ltmp2 > b2->cutLength)
			//{
			//	//intersectionLength[j] = ltmp2;
			//	b2->cutLength = ltmp2;
			//}
			if (ltmp1 > intersectionLength[i])
				intersectionLength[i] = ltmp1;
			if (ltmp2 > intersectionLength[j])
				intersectionLength[j] = ltmp2;
		}
	}
	//optimal cut
	//stack<int> stk; stack<int> stk2;
	int stk[100]; int stk2[100];
	int cnt = 0,cnt2=0;
	for (int cutId = 0; cutId < size; cutId++)
	{
		//recursiveOptimalCut(beams, i, size, direct, intersectionLength);
		//stk.push(cutId);
		stk[cnt] = cutId;
		cnt++;
		while (cnt>0)
		{
			int i = stk[cnt-1];
			int j = 0;
			if (cnt2 == cnt)
			{
				j = stk2[cnt2-1];
				j++;
				cnt2--;
				//stk2.pop();
			}
			for (; j < size; j++)
			{
				if (j == i)continue;
				BeamPlugin* b1 = beams[i];
				BeamPlugin* b2 = beams[j];
				double r1 = b1->radius, r2 = b2->radius;

				double cosVal = cuda_Dot(direct[i], direct[j]) / (cuda_Norm(direct[i])*cuda_Norm(direct[j]));
				double angle = acos(Min(Max(cosVal,-1.0f),1.0f));
				/*if (abs(cosVal+1)<0.31||angle>3*PI/4)
				{
					continue;
				}*/
				/*if (angle >= PI / 2)
					continue;*/

				double length1 = sqrt(pow(r1, 2) + pow(intersectionLength[i], 2));
				double itoj = angle - atan(r1 / intersectionLength[i]);
				double cut1 = length1 * cos(itoj);
				if (cut1 > intersectionLength[j])
				{
					intersectionLength[j] = cut1;
					stk[cnt] = j; cnt++;
					stk2[cnt2] = j; cnt2++;
					break;
					//cuda_recursiveOptimalCut(beams, j, size, direct, intersectionLength);
				}

				double length2 = sqrt(pow(r2, 2) + pow(intersectionLength[j], 2));
				double jtoi = angle - atan(r2 / intersectionLength[j]);
				double cut2 = length2 * cos(jtoi);
				if (cut2 > intersectionLength[i])
				{
					intersectionLength[i] = cut2;
					stk[cnt] = i; cnt++;
					stk2[cnt2] = j; cnt2++;
					break;
					//cuda_recursiveOptimalCut(beams, i, size, direct, intersectionLength);
				}
			}
			if (j == size)
			{
				cnt--;
				//stk.pop();
			}
		}
	}
	free(direct);

	for (int i = 0; i < size; i++)
	{
		BeamPlugin* b = beams[i];
		b->convexLength[flag[i]] = intersectionLength[i]+threshold;
		if (b->convexLength[0] + b->convexLength[1] >= b->originLength)
			printf("radius too large! %f %f\n", b->radius,b->originLength);
	}
	free(intersectionLength);
	free(flag);
}
__device__ void cuda_getTriangle(BeamPlugin *b[], float3  position, Triangle* &triList, int triId)
{
	float3 finalP[3];
	int flag[3] = { 0 };

	for (int i = 0; i < 3; i++)
	{
		BeamPlugin *btmp = b[i];
		if (pow(position.x - btmp->axis.p[0].x, 2) + pow(position.y - btmp->axis.p[0].y, 2) + pow(position.z - btmp->axis.p[0].z, 2)
			< pow(position.x - btmp->axis.p[1].x, 2) + pow(position.y - btmp->axis.p[1].y, 2) + pow(position.z - btmp->axis.p[1].z, 2))
			flag[i] = 1;
		else
			flag[i] = 2;

		
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
		diskNorm[i] = cuda_make_norm(diskNorm[i]);

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

		normal = cuda_make_norm(normal);

		float cosVal = cuda_Dot(normal, diskNorm[i]);
		float theta = acos(Min(Max(cosVal, -1.0f), 1.0f));//acos(cosVal);
		float s = b[i]->radius / sin(theta);
		finalP[i] = make_float3(s * normal.x + o1.x - s * cos(theta)*diskNorm[i].x,
			s * normal.y + o1.y - s * cos(theta)*diskNorm[i].y, s * normal.z + o1.z - s * cos(theta)*diskNorm[i].z);
		

	}


	Triangle* triTmp = triList+triId;
	triTmp->p[0] = finalP[0];
	triTmp->p[1] = finalP[1];
	triTmp->p[2] = finalP[2];
	float3 e1 = make_float3(finalP[1].x - finalP[0].x, finalP[1].y - finalP[0].y, finalP[1].z - finalP[0].z);
	float3 e2 = make_float3(finalP[2].x - finalP[0].x, finalP[2].y - finalP[0].y, finalP[2].z - finalP[0].z);
	triTmp->normal = cuda_cross(e1, e2);
	triTmp->normal = cuda_make_norm(triTmp->normal);
	triTmp->triId = triId;
	for (int i = 0; i < 3; i++)
	{
		//triTmp->triWithBeamid[i] = b[i]->beamId;
		triTmp->diskNorm[i] = diskNorm[i];
	}
	/*triTmp.o[0] = mp[b[0]->beamId];
	triTmp.o[1] = mp[b[1]->beamId];
	triTmp.o[2] = mp[b[2]->beamId];*/
	e1 = make_float3(triTmp->o[1].x - triTmp->o[0].x, triTmp->o[1].y - triTmp->o[0].y, triTmp->o[1].z - triTmp->o[0].z);
	e2 = make_float3(triTmp->o[2].x - triTmp->o[0].x, triTmp->o[2].y - triTmp->o[0].y, triTmp->o[2].z - triTmp->o[0].z);
	triTmp->originNormal = cuda_cross(e1, e2);
	triTmp->originNormal = cuda_make_norm(triTmp->originNormal);

	//triTmp.adjTriangles = (Triangle*)malloc(3 * sizeof(Triangle));
	//triTmp.adjQuadFaces = (QuadFace*)malloc(3 * sizeof(QuadFace));
	triTmp->adjTriNum = 0;
	triTmp->adjQuadNum = 0;
	triTmp->flag = 0;
	triTmp->cost = 0;
	triTmp->projSeg = 0;
	triTmp->inverted[0] = triTmp->inverted[1]=0;
}
__device__ void cuda_findAdjTriangles(Triangle* triList, int triNum)
{
	//bool tmpInverted = false;
	//inverted[0] = false;
	//float3 t1Tot2[2], t2Tot1[2];
	//float3 projP1[2], projP2[2];
	////公共边的两个点
	//float3 edge1Points[2], edge2Points[2];
	//int id1[2], id2[2];

	//int triNum = triList.size();
	//cout << triNum << endl;
	/*for (int i = 0; i < triNum; i++)
		triList[i]->adjTriangles.clear();*/
	for (int i = 0; i < triNum - 1; i++)
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
						/*t1Tot2[cnt] = make_float3(t2->p[k].x - t1->p[v].x, t2->p[k].y - t1->p[v].y, t2->p[k].z - t1->p[v].z);
						t2Tot1[cnt] = make_float3(-t1Tot2[cnt].x, -t1Tot2[cnt].y, -t1Tot2[cnt].z);

						edge1Points[cnt] = t1->p[v];
						id1[cnt] = v;
						edge2Points[cnt] = t2->p[k];
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
				/*if(t1.adjTriNum==0)
					t1.adjTriangles = (Triangle*)malloc(sizeof(Triangle));
				else
					t1.adjTriangles = (Triangle*)realloc(t1.adjTriangles, (t1.adjTriNum + 1) * sizeof(Triangle));*/
				/*t1->adjTriangles[t1->adjTriNum].p[0] = t2->p[0];
				t1->adjTriangles[t1->adjTriNum].p[1] = t2->p[1];
				t1->adjTriangles[t1->adjTriNum].p[2] = t2->p[2];
				t1->adjTriangles[t1->adjTriNum].triId = t2->triId;
				t1->adjTriangles[t1->adjTriNum].triWithBeamid[0] = t2->triWithBeamid[0];
				t1->adjTriangles[t1->adjTriNum].triWithBeamid[1] = t2->triWithBeamid[1];
				t1->adjTriangles[t1->adjTriNum].triWithBeamid[2] = t2->triWithBeamid[2];*/
				t1->adjTriangles[t1->adjTriNum] = t2;
				t1->adjTriNum++;
				//triList[i] = t1;
				/*if(t2.adjTriNum==0)
					t2.adjTriangles = (Triangle*)malloc(sizeof(Triangle));
				else
					t2.adjTriangles = (Triangle*)realloc(t2.adjTriangles, (t2.adjTriNum + 1) * sizeof(Triangle));*/
				/*t2->adjTriangles[t2->adjTriNum].p[0] = t1->p[0];
				t2->adjTriangles[t2->adjTriNum].p[1] = t1->p[1];
				t2->adjTriangles[t2->adjTriNum].p[2] = t1->p[2];
				t2->adjTriangles[t2->adjTriNum].triId = t1->triId;
				t2->adjTriangles[t2->adjTriNum].triWithBeamid[0] = t1->triWithBeamid[0];
				t2->adjTriangles[t2->adjTriNum].triWithBeamid[1] = t1->triWithBeamid[1];
				t2->adjTriangles[t2->adjTriNum].triWithBeamid[2] = t1->triWithBeamid[2];*/
				t2->adjTriangles[t2->adjTriNum] = t1;
				t2->adjTriNum++;
				//triList[j] = t2;
				//t1.adjTriangles.push_back(*t2);
				//t2.adjTriangles.push_back(*t1);
				//判断是否翻转
				//判断投影点是否在三角形内部
				//if (!inverted[0])
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
				//		float3 edge1Normal = cuda_cross(e1, t1->normal);
				//		float3 edge2Normal = cuda_cross(e2, t2->normal);
				//		if (cuda_Dot(t1Tot2[0], edge1Normal) < 0 || cuda_Dot(t1Tot2[1], edge1Normal) < 0 ||
				//			cuda_Dot(t2Tot1[0], edge2Normal) < 0 || cuda_Dot(t2Tot1[1], edge2Normal) < 0)
				//			tmpInverted = true;
				//		

				//	}
				//	if (tmpInverted)
				//	{
				//		//若为凸，则不算翻转
				//		if (cuda_Dot(t1->normal, t1Tot2[0]) < 0 && cuda_Dot(t1->normal, t1Tot2[1]) < 0 && cuda_Dot(t2->normal, t2Tot1[0]) < 0 && cuda_Dot(t2->normal, t2Tot1[1]) < 0)
				//			tmpInverted = false;
				//	}
				//	if (tmpInverted)
				//		inverted[0] = true;
				//}

				/*if (t1->normal.dot(t1Tot2[0]) > 0 || t1->normal.dot(t1Tot2[1]) > 0)
					inverted = true;
				if (t2->normal.dot(t2Tot1[0]) > 0 || t2->normal.dot(t2Tot1[1]) > 0)
					inverted = true;*/
			}
		}
	}
}
__device__ int cuda_mod(float a, float b)
{
	return a - (b*floor(a / b));
}
__device__ int* find(int* adjPoints, int pId)
{
	for (int i = 0; i < 3; i++)
	{
		if (adjPoints[i] == pId)
		{
			return adjPoints + i;
		}
	}
	return adjPoints + 3;
}
__device__ void cuda_findQuadFaces(Triangle* triList, QuadFace* quadList, int triNum, bool &inverted)
{
	short tmpInverted = 0;
	inverted = false;
	bool tmpFlipped = false;
	int quadId = 0;
	//int triNum = triList.size();
	//bool* visited = (bool*)malloc(triNum * sizeof(bool));
	bool visited[100];
	for (int i = 0; i < triNum; i++)visited[i] = false;
	//std::fill(visited, visited + triNum, false);
	/*for (int i = 0; i < triNum; i++)
		triList[i]->adjQuadFaces.clear();*/
	for (int i = 0; i < triNum; i++)
	{
		Triangle *t = triList+i;
		//shared_ptr<QuadFace> threeQuad[3];
		for (int j = 0; j < 3; j++)
		{
			Triangle* adjT = t->adjTriangles[j];
			if (!visited[adjT->triId])
			{
				//找公共边
				int adjPoints[3];
				for (int k = 0; k < 3; k++)
				{
					adjPoints[k] = adjT->triWithBeamid[k];
				}
				for (int k = 0; k < 3; k++)
				{
					int pId = t->triWithBeamid[k];
					if (find(adjPoints, pId) - adjPoints == 3)
					{
						//k-1和k+1就是公共边
						QuadFace* q = quadList+quadId;
						//q->t->resize(2);
						q->quadId = quadId++;
						q->inverted[0] = q->inverted[1] = q->inverted[2] = q->inverted[3] = q->inverted[4] = q->inverted[5] = 0;
						q->quadWithBeamid[0] = t->triWithBeamid[cuda_mod((k - 1), 3)];
						q->quadWithBeamid[1] = t->triWithBeamid[cuda_mod((k + 1), 3)];
						Triangle* t0 = &q->t[0];
						Triangle* t1 = &q->t[1];

						t0->p[0] = t->p[cuda_mod((k - 1), 3)];
						t0->p[1] = t->p[cuda_mod((k + 1), 3)];
						int*  it = find(adjPoints, t->triWithBeamid[(k + 1) % 3]);
						int k2 = it - adjPoints;
						t0->p[2] = adjT->p[k2];
						//判断是否共点
						float3 line1 = make_float3(t0->p[2].x - t0->p[1].x, t0->p[2].y - t0->p[1].y, t0->p[2].z - t0->p[1].z);
						if (line1.x == 0 && line1.y == 0 && line1.z == 0)
						{
							q->t[0].flag = 1;
							q->width[0] = 0;
						}
						else
						{
							float3 e1 = make_float3(t0->p[1].x - t0->p[0].x, t0->p[1].y - t0->p[0].y, t0->p[1].z - t0->p[0].z);
							float3 e2 = make_float3(t0->p[2].x - t0->p[0].x, t0->p[2].y - t0->p[0].y, t0->p[2].z - t0->p[0].z);
							t0->normal = cuda_cross(e1, e2);
							t0->normal = cuda_make_norm(t0->normal);
							//q->t[0] = t0;
							//float3 line1 = make_float3(t0.p[2].x - t0.p[1].x, t0.p[2].y - t0.p[1].y, t0.p[2].z - t0.p[1].z);
							q->width[0] = cuda_Norm(line1);
							//判断是否翻转
							float3 edge1Normal = cuda_cross(cuda_Subtraction(t0->p[0], t0->p[1]), t->normal);
							if (cuda_Dot(edge1Normal, line1) < 0)
								tmpInverted++;
							q->t[0].flag = 0;
							q->t[0].inverted[0] = 0;
							q->COS[0] = cuda_Dot(edge1Normal, line1) / (cuda_Norm(edge1Normal) * cuda_Norm(line1));
						}
						////加入新判断，判断是否交叉
						//float3 ov1 = cuda_Subtraction(t->p[cuda_mod((k + 1), 3)], t->o[cuda_mod((k + 1), 3)]);
						//float3 ov2 = cuda_Subtraction(adjT->p[k2], adjT->o[k2]);
						//float3 direct = cuda_cross(ov1, ov2);
						//direct = cuda_make_norm(direct);
						//if (cuda_Dot(direct, t->diskNorm[cuda_mod((k + 1), 3)]) < 0)
						//	inverted = true;

						int k3 = cuda_mod((k2 - 1), 3);
						t1->p[0] = t->p[cuda_mod((k - 1), 3)];
						t1->p[1] = adjT->p[k2];
						t1->p[2] = adjT->p[k3];
						//判断是否共点
						float3 line2 = make_float3(t1->p[0].x - t1->p[2].x, t1->p[0].y - t1->p[2].y, t1->p[0].z - t1->p[2].z);
						if (abs(line2.x) == 0 && abs(line2.y) == 0 && abs(line2.z) == 0)
						{
							q->t[1].flag = 1;
							q->width[1] = 0;
						}
						else
						{
							float3 e1 = make_float3(t1->p[1].x - t1->p[0].x, t1->p[1].y - t1->p[0].y, t1->p[1].z - t1->p[0].z);
							float3 e2 = make_float3(t1->p[2].x - t1->p[0].x, t1->p[2].y - t1->p[0].y, t1->p[2].z - t1->p[0].z);
							t1->normal = cuda_cross(e1, e2);
							t1->normal = cuda_make_norm(t1->normal);
							//q->t[1] = t1;
							//float3 line2 = make_float3(t1.p[0].x - t1.p[2].x, t1.p[0].y - t1.p[2].y, t1.p[0].z - t1.p[2].z);
							q->width[1] = cuda_Norm(line2);
							//判断是否翻转
							float3 edge2Normal = cuda_cross(cuda_Subtraction(t1->p[1], t1->p[2]), adjT->normal);
							if (cuda_Dot(edge2Normal, line2) < 0)
								tmpInverted++;
							q->t[1].flag = 0;
							q->t[1].inverted[0] = 0;
							q->COS[3] = cuda_Dot(edge2Normal, line2) / (cuda_Norm(edge2Normal) * cuda_Norm(line2));
						}
						////加入新判断，判断是否交叉
						//ov1 = cuda_Subtraction(adjT->p[k3], adjT->o[k3]);
						//ov2 = cuda_Subtraction(t->p[cuda_mod((k - 1), 3)], t->o[cuda_mod((k - 1), 3)]);
						//direct = cuda_cross(ov1, ov2);
						//direct = cuda_make_norm(direct);
						//if (cuda_Dot(direct, t->diskNorm[cuda_mod((k - 1), 3)]) < 0)
						//	inverted = true;

						/*if(t->adjQuadNum==0)
							t->adjQuadFaces = (QuadFace*)malloc(sizeof(QuadFace));
						else
							t->adjQuadFaces = (QuadFace*)realloc(t->adjQuadFaces, (t->adjQuadNum + 1) * sizeof(QuadFace));*/
						q->F[0] = q->F[1] = 0;
						t->adjQuadFaces[t->adjQuadNum] = q;
						t->adjQuadNum++;
						//triList[i] = t;

						//Triangle *Ttmp = &triList[adjT->triId];
						/*if(Ttmp->adjQuadNum==0)
							Ttmp->adjQuadFaces = (QuadFace*)malloc(sizeof(QuadFace));
						else
							Ttmp->adjQuadFaces = (QuadFace*)realloc(Ttmp->adjQuadFaces, (Ttmp->adjQuadNum + 1) * sizeof(QuadFace));*/
						//Ttmp->adjQuadFaces[Ttmp->adjQuadNum] = *q;
						//Ttmp->adjQuadNum++;
						adjT->adjQuadFaces[adjT->adjQuadNum] = q;
						adjT->adjQuadNum++;
						//triList[adjT->triId] = Ttmp;

						//quadList[quadId - 1] = q;
						/*t->adjQuadFaces.push_back(*q);
						triList[adjT.triId]->adjQuadFaces.push_back(*q);*/

						//threeQuad[j] = q;
						//quadList.push_back(q);
						/*if(quadId==1)
							quadList = (QuadFace**)malloc(sizeof(QuadFace*));
						else
							quadList = (QuadFace**)realloc(quadList, (quadId) * sizeof(QuadFace*));
						quadList[quadId - 1] = q;*/
						if (cuda_Dot(q->t[0].normal, q->t[1].normal) < 0)
							tmpInverted++;
						//if (!inverted)
						//{
						//	if (tmpInverted>=2)
						//	{
						//		//若为凸，则不算翻转
						//		float3 nline1 = make_float3(-line1.x, -line1.y, -line1.z);
						//		float3 nline2 = make_float3(-line2.x, -line2.y, -line2.z);
						//		if (cuda_Dot(t->normal, line1) < 0 && cuda_Dot(t->normal, nline2) < 0 &&
						//			cuda_Dot(adjT->normal, line2) < 0 && cuda_Dot(adjT->normal, nline1) < 0 );
						//		else
						//		{
						//			t->inverted[0] = 1; adjT->inverted[0] = 1;
						//			q->t[0].inverted[0] = q->t[1].inverted[0] = 1;
						//			inverted = true;
						//		}
						//	}
						//	/*if (tmpInverted)
						//	{
						//		t->inverted = 1;
						//		inverted = true;
						//	}*/
						//}
						tmpInverted = 0;

						Triangle* t2 = &q->t[2];
						Triangle* t3 = &q->t[3];
						t2->p[0] = t->p[cuda_mod((k - 1), 3)];
						t2->p[1] = t->p[cuda_mod((k + 1), 3)];
						it = find(adjPoints, t->triWithBeamid[cuda_mod((k - 1),3)]);
						k2 = it - adjPoints;
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
							t2->normal = cuda_cross(e1, e2);
							t2->normal = cuda_make_norm(t2->normal);
							//判断是否翻转
							float3 edge3Normal = cuda_cross(cuda_Subtraction(t2->p[0], t2->p[1]), t->normal);
							if (cuda_Dot(edge3Normal, line3) < 0)
								tmpInverted++;
							q->t[2].flag = 0;
							q->t[2].inverted[0] = 0;
							q->COS[2] = cuda_Dot(edge3Normal, line3) / (cuda_Norm(edge3Normal) * cuda_Norm(line3));
						}

						k3 = cuda_mod((k2 + 1), 3);
						t3->p[0] = adjT->p[k2]; 
						t3->p[1] = t->p[cuda_mod((k + 1), 3)];
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
							t3->normal = cuda_cross(e1, e2);
							t3->normal = cuda_make_norm(t3->normal);
							//判断是否翻转
							float3 edge4Normal = cuda_cross(cuda_Subtraction(t3->p[2], t3->p[0]), adjT->normal);
							if (cuda_Dot(edge4Normal, line4) < 0)
								tmpInverted++;
							q->t[3].flag = 0;
							q->t[3].inverted[0] = 0;
							q->COS[1] = cuda_Dot(edge4Normal, line4) / (cuda_Norm(edge4Normal) * cuda_Norm(line4));
						}
						if (cuda_Dot(q->t[2].normal, q->t[3].normal) < 0)
							tmpInverted++;

						if ((q->COS[0] < 0 && q->COS[1] < 0) || (q->COS[2] < 0 && q->COS[3] < 0)||(q->COS[0]<0&&q->COS[3]<0)||(q->COS[1]<0&&q->COS[2]<0)||
							(q->COS[0]<0&&q->COS[2]<0)||(q->COS[1]<0&&q->COS[3]<0))
						{
							float3 nline3 = make_float3(-line3.x, -line3.y, -line3.z);
							float3 nline4 = make_float3(-line4.x, -line4.y, -line4.z);
							float3 line5 = cuda_Subtraction(adjT->p[cuda_mod(k3 + 1, 3)], t2->p[0]);
							float3 line6 = cuda_Subtraction(t->p[k], t3->p[0]);
							
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
								if ((cuda_Dot(t->normal, line3) < 0 && cuda_Dot(t->normal, nline4) < 0 && cuda_Dot(t->normal, line5) < 0) ||
									(cuda_Dot(adjT->normal, line4) < 0 && cuda_Dot(adjT->normal, nline3) < 0 && cuda_Dot(adjT->normal, line6) < 0))
								{
									q->inverted[4] = 0;
									tmpFlipped = false;
								}
							}
							if (q->COS[1] < 0 && q->COS[3] < 0)
							{
								q->inverted[5] = 1;
								tmpFlipped = true;
								if ((cuda_Dot(t->normal, line3) < 0 && cuda_Dot(t->normal, nline4) < 0 && cuda_Dot(t->normal, line5) < 0) ||
									(cuda_Dot(adjT->normal, line4) < 0 && cuda_Dot(adjT->normal, nline3) < 0 && cuda_Dot(adjT->normal, line6) < 0))
								{
									q->inverted[5] = 0;
									tmpFlipped = false;
								}
							}
							if (cuda_Dot(t->normal, line3) < 0 && cuda_Dot(t->normal, nline4) < 0 && cuda_Dot(t->normal, line5) < 0&& 
								cuda_Dot(adjT->normal, line4) < 0 && cuda_Dot(adjT->normal, nline3) < 0 && cuda_Dot(adjT->normal, line6) < 0)
							{
								q->inverted[0] = q->inverted[1] = q->inverted[2]=q->inverted[3]=q->inverted[4]=q->inverted[5]=0;
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
						//	if (cuda_Dot(t->normal, line3) < 0 && cuda_Dot(t->normal, nline4) < 0 &&
						//		cuda_Dot(adjT->normal, line4) < 0 && cuda_Dot(adjT->normal, nline3) < 0);
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
	//inverted = false;
	//free(visited);
	//quadNum = quadId;
}
__device__ void cuda_generateFaces(Triangle* triList, int triNum, QuadFace* quadList, int quadNum, Face** cudaf,int fid)
{
	//int triNum = triList.size();
	cudaf[fid] = (Face*)malloc(triNum * sizeof(Face));
	//single_triNum[0] = triNum;
	for (int i = 0; i < triNum; i++)
	{
		Triangle *t = &triList[i];
		Face *ftmp = &cudaf[fid][i];
		ftmp->p1 = t->p[0];
		ftmp->p2 = t->p[1];
		ftmp->p3 = t->p[2];
		ftmp->normal = t->normal;
		//cf[i] = ftmp;
	}
	//hf = cf;
	//int quadNum = quadList.size();
	/*for (int i = 0; i < quadNum; i++)
	{
		QuadFace* q = quadList[i];
		Face ftmp;
		ftmp.p1 = q->t[0].p[0];
		ftmp.p2 = q->t[0].p[1];
		ftmp.p3 = q->t[0].p[2];
		ftmp.normal = q->t[0].normal;
		J->f.push_back(ftmp);

		ftmp.p1 = q->t[1].p[0];
		ftmp.p2 = q->t[1].p[1];
		ftmp.p3 = q->t[1].p[2];
		ftmp.normal = q->t[1].normal;
		J->f.push_back(ftmp);
	}*/

}
__device__ void cuda_getRotationMatrix(float3 diskNorm, float theta, float3 transl,float* m)
{
	float u = diskNorm.x, v = diskNorm.y, w = diskNorm.z;
	/*float** m = (float**)malloc(4 * sizeof(float*));
	for (int i = 0; i < 4; i++)
	{
		m[i] = (float*)malloc(4 * sizeof(float));
	}*/
	m[0*4+0] = pow(u, 2) + (1 - pow(u, 2))*cos(theta);
	m[0*4+1] = u * v*(1 - cos(theta)) - w * sin(theta);
	m[0*4+2] = u * w*(1 - cos(theta)) + v * sin(theta);
	m[1*4+0] = u * v*(1 - cos(theta)) + w * sin(theta);
	m[1*4+1] = pow(v, 2) + (1 - pow(v, 2))*cos(theta);
	m[1*4+2] = v * w*(1 - cos(theta)) - u * sin(theta);
	m[2*4+0] = u * w*(1 - cos(theta)) - v * sin(theta);
	m[2*4+1] = v * w*(1 - cos(theta)) + u * sin(theta);
	m[2*4+2] = pow(w, 2) + (1 - pow(w, 2))*cos(theta);

	m[0*4+3] = transl.x;
	m[1*4+3] = transl.y;
	m[2*4+3] = transl.z;
	m[3*4+3] = 1;

	m[3*4+0] = 0; m[3*4+1] = 0; m[3*4+2] = 0;
	//m << 0;
	//m << 0, 0, 0, 1;

	//m << 1, 0, 0,0,1,0,0,0, 1, 0, 0, 0, 1, 0, 0, 0;
	/*Eigen::MatrixXd ll(1, 4);
	ll << 0, 0, 0, 1;

	Eigen::MatrixXd M(4, 4);
	M << m, transl,ll;*/
	//std::cout << M << std::endl;
	//return m;
}
__device__ void isCross(float3 O, float3 A,float3 B,float &proj)
{
	proj = 0;
	float3 AO = cuda_Subtraction(O, A);
	float3 AB = cuda_Subtraction(B, A);
	float r = cuda_Dot(AO, AB) / (cuda_Norm(AO) * cuda_Norm(AB));
	proj = r;

	/*float r = cuda_Dot(AO, AB) / (pow(cuda_Norm(AB), 2));
	if (r > 0 && r < 1)
	{
		float3 ab = cuda_make_norm(AB);
		proj = cuda_Dot(AO, ab) * 2;
		return true;
	}
	else
		return false;*/
}
__device__ float cuda_measureSingleCost(Triangle* triList, int triNum, QuadFace* quadList, int quadNum,
	GreyWolves gws,int nVar,bool& inverted)
{
	float centerDis = 0;
	float angleDis = 0;
	float max_N1N2dis = 0;
	int quadCnt = 0;
	//vector<float> costN1N2;
	//vector<float> costT;
	//int triNum = triList.size();
	float projSeg = 0,projSeg2=0;
	float triEdgeLength = 0;
	float BF1 = 0;
	float BF = 0;
	for (int i = 0; i < triNum; i++)
	{
		//costT.clear();
		Triangle* t = triList+i;
		float3 originCenter = make_float3((t->o[0].x+t->o[1].x+t->o[2].x)/3,(t->o[0].y+t->o[1].y+t->o[2].y)/3,
			(t->o[0].z+t->o[1].z+t->o[2].z)/3);
		float3 center = make_float3((t->p[0].x + t->p[1].x + t->p[2].x) / 3, (t->p[0].y + t->p[1].y + t->p[2].y) / 3,
			(t->p[0].z + t->p[1].z + t->p[2].z) / 3);
		centerDis += cuda_Norm(cuda_Subtraction(originCenter, center));
		float originAngle[3];
		for (int j = 0; j < 3; j++)
		{
			float3 e1 = cuda_Subtraction(t->o[cuda_mod(j + 1, 3)], t->o[j]);
			float3 e2 = cuda_Subtraction(t->o[cuda_mod(j + 2, 3)], t->o[j]);
			originAngle[j] =acos( cuda_Dot(e1, e2) / (cuda_Norm(e1)*cuda_Norm(e2)));
				
		}
		float triAngle[3];
		for (int j = 0; j < 3; j++)
		{
			float3 e1 = cuda_Subtraction(t->p[cuda_mod(j + 1, 3)], t->p[j]);
			float3 e2 = cuda_Subtraction(t->p[cuda_mod(j + 2, 3)], t->p[j]);
			triAngle[j] = acos(cuda_Dot(e1, e2) / (cuda_Norm(e1)*cuda_Norm(e2)));
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
			//		float3 tp1 = t->p[cuda_mod(k - 1, 3)];
			//		float3 tp2 = t->p[cuda_mod(k + 1, 3)];
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
			//					if (cuda_Norm(cuda_Subtraction(tp1, qtp)) < 1e-6)
			//						cnt++;
			//					if (cuda_Norm(cuda_Subtraction(tp2, qtp)) < 1e-6)
			//						cnt++;
			//				}
			//				if (cnt == 2)
			//				{
			//					float COS = cuda_Dot(t->normal, quadT->normal);
			//					//float theta = acos(Min(Max(cuda_Dot(t->normal, quadT->normal), -1.0f), 1.0f));//acos(cuda_Dot(t->normal,quadT->normal));
			//				//costT.push_back(theta);
			//				//costN1N2.push_back(theta);
			//					max_N1N2dis += pow(COS - 1, 2);
			//					if (COS<0&&t->inverted[v/2] == 1)//翻转
			//					{
			//						//float x = (theta - PI / 2) / (PI / 2);
			//						float x = -COS;
			//						BF1 += -(1000 / x)*log(1 - x);
			//					}
			//					//break;
			//				}
			//				//else
			//				//{
			//				//	float COS = cuda_Dot(t->normal, quadT->normal);
			//				//	//float theta = acos(Min(Max(cuda_Dot(t->normal, quadT->normal), -1.0f), 1.0f));
			//				//	max_N1N2dis += pow(COS - 1, 2);
			//				//}
			//				t->cost = Max(t->cost, acos(Min(Max(cuda_Dot(t->normal, quadT->normal), -1.0f), 1.0f)));
			//			}
			//			
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
			float3 line1 = cuda_Subtraction(t->p[cuda_mod(j + 1, 3)], t->p[j]);
			float3 line2 = cuda_Subtraction(t->p[cuda_mod(j + 2, 3)], t->p[j]);
			if (cuda_Dot(n_diskNorm, t->normal) > 0|| cuda_Dot(line1, n_diskNorm) > 0 || cuda_Dot(line2, n_diskNorm) > 0)
			{
				
				float cosTheta = cuda_Dot(t->diskNorm[j], line1) / cuda_Norm(line1);
				float length1 = cosTheta * cuda_Norm(line1);
				float3 newDiskNorm1 = make_float3(t->diskNorm[j].x*length1, t->diskNorm[j].y*length1, t->diskNorm[j].z*length1);

				/*float cosTheta = cuda_Dot(n_diskNorm, t->normal);
				float theta = acos(Min(Max(cosTheta, -1.0f), 1.0f));
				float length1 = sin(theta)*cuda_Norm(line1);
				float3 newDiskNorm1 = make_float3(t->diskNorm[j].x*length1, t->diskNorm[j].y*length1, t->diskNorm[j].z*length1);*/
				float3 sub1 = cuda_Subtraction(line1, newDiskNorm1);
				float3 projL1 = make_float3(sub1.x+t->p[j].x, sub1.y+t->p[j].y, sub1.z+t->p[j].z);
				
				cosTheta = cuda_Dot(t->diskNorm[j], line2) / cuda_Norm(line2);
				float length2 = cosTheta * cuda_Norm(line2);
				float3 newDiskNorm2 = make_float3(t->diskNorm[j].x*length2, t->diskNorm[j].y*length2, t->diskNorm[j].z*length2);
				/*float length2 = sin(theta)*cuda_Norm(line2);
				float3 newDiskNorm2 = make_float3(t->diskNorm[j].x*length2, t->diskNorm[j].y*length2, t->diskNorm[j].z*length2);*/
				float3 sub2 = cuda_Subtraction(line2, newDiskNorm2);
				float3 projL2 = make_float3(sub2.x + t->p[j].x, sub2.y + t->p[j].y, sub2.z + t->p[j].z);

				float projLength1, projLength2;
				isCross(t->o[j], t->p[j], projL1, projLength1);
				//projSeg += projLength1;
				isCross(t->o[j], t->p[j], projL2, projLength2);
				//projSeg += projLength2;
				if (projLength1 > 0 && projLength2 < 0)
					projSeg += projLength1;
				else if (projLength1 < 0 && projLength2>0)
					projSeg += projLength2;
				else
					projSeg += projLength1 + projLength2;
				/*if (isCross(t->o[j], t->p[j], projL1, projLength1))
				{
					projSeg += projLength1;
				}
				if (isCross(t->o[j], t->p[j], projL2, projLength2))
				{
					projSeg += projLength2;
				}*/
				//判断是否本来需要optimal cut
				//float3 vjtoj_1 = cuda_Subtraction(t->p[cuda_mod(j+1,3)],t->p[j]);
				if ((cuda_Dot(line1, n_diskNorm) > 0&& projLength1 > 0) ||(cuda_Dot(line2,n_diskNorm)>0&& projLength2 > 0))
				{
					//if (projLength1 > 0 || projLength2 > 0)
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
						//float r = cuda_Norm(cuda_Subtraction(t->p[j], t->o[j]));
						//bf = bf / (3 * r);//scale to (0,1)
						//if (bf >= 1)bf = 0.999;
						BF += -(1000/bf)*log(1 - bf);
						//BF = 10000;
					}
				}
			}
			else if (cuda_Dot(n_diskNorm, t->normal) < 0)
			{
				float cosTheta = cuda_Dot(t->diskNorm[j], line1) / cuda_Norm(line1);
				float length1 = cosTheta * cuda_Norm(line1);
				float3 newDiskNorm1 = make_float3(t->diskNorm[j].x * length1, t->diskNorm[j].y * length1, t->diskNorm[j].z * length1);

				/*float cosTheta = cuda_Dot(n_diskNorm, t->normal);
				float theta = acos(Min(Max(cosTheta, -1.0f), 1.0f));
				float length1 = sin(theta)*cuda_Norm(line1);
				float3 newDiskNorm1 = make_float3(t->diskNorm[j].x*length1, t->diskNorm[j].y*length1, t->diskNorm[j].z*length1);*/
				float3 sub1 = cuda_Subtraction(line1, newDiskNorm1);
				float3 projL1 = make_float3(sub1.x + t->p[j].x, sub1.y + t->p[j].y, sub1.z + t->p[j].z);

				cosTheta = cuda_Dot(t->diskNorm[j], line2) / cuda_Norm(line2);
				float length2 = cosTheta * cuda_Norm(line2);
				float3 newDiskNorm2 = make_float3(t->diskNorm[j].x * length2, t->diskNorm[j].y * length2, t->diskNorm[j].z * length2);
				/*float length2 = sin(theta)*cuda_Norm(line2);
				float3 newDiskNorm2 = make_float3(t->diskNorm[j].x*length2, t->diskNorm[j].y*length2, t->diskNorm[j].z*length2);*/
				float3 sub2 = cuda_Subtraction(line2, newDiskNorm2);
				float3 projL2 = make_float3(sub2.x + t->p[j].x, sub2.y + t->p[j].y, sub2.z + t->p[j].z);

				float projLength1, projLength2;
				isCross(t->o[j], t->p[j], projL1, projLength1);
				//projSeg += projLength1;
				isCross(t->o[j], t->p[j], projL2, projLength2);
				//projSeg += projLength2;
				if (projLength1 > 0 && projLength2 < 0)
					projSeg += -projLength2;
				else if (projLength1 < 0 && projLength2>0)
					projSeg += -projLength1;
				else
					projSeg += -(projLength1 + projLength2);
			}
			/*else
			{
				projSeg += -0.5;
			}*/
			


		}
		for (int j = 0; j < 3; j++)
		{
			float l = cuda_Norm(cuda_Subtraction(t->p[cuda_mod(j + 1, 3)], t->p[j]));
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
		//	float COS = cuda_Dot(qTmp->t[0].normal, qTmp->t[1].normal);
		//	//float theta = acos(Min(Max(cuda_Dot(qTmp->t[0].normal, qTmp->t[1].normal), -1.0f), 1.0f));//acos(cuda_Dot(qTmp->t[0].normal,qTmp->t[1].normal));
		//    //costN1N2.push_back(theta);
		//	max_N1N2dis += pow(COS - 1, 2);
		//	//if (COS < 0 && qTmp->t[0].inverted[0] == 1)//翻转
		//	//{
		//	//	//float x = (theta - PI / 2) / (PI / 2);
		//	//	float x = -COS;
		//	//	BF1 += -(1000 / x) * log(1 - x);
		//	//}
		//}
		//if (qTmp->t[2].flag == 0 && qTmp->t[3].flag == 0)
		//{
		//	float COS = cuda_Dot(qTmp->t[2].normal, qTmp->t[3].normal);
		//	//float theta = acos(Min(Max(cuda_Dot(qTmp->t[0].normal, qTmp->t[1].normal), -1.0f), 1.0f));//acos(cuda_Dot(qTmp->t[0].normal,qTmp->t[1].normal));
		//	//costN1N2.push_back(theta);
		//	max_N1N2dis += pow(COS - 1, 2);
		//	//if (COS < 0 && qTmp->t[2].inverted[0] == 1)//翻转
		//	//{
		//	//	//float x = (theta - PI / 2) / (PI / 2);
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
	//printf("%f %f %f %f\n", max_N1N2dis, projSeg, BF, BF1);
	return max_N1N2dis+projSeg+BF+BF1;
}
__device__ void cuda_computeX123(float* leaderRotationAngle, GreyWolves* gws, int nVar, float a, int it,  float* X)
{

	//% Eq.(3.4) in the paper
	//float* c = (float*)malloc(nVar * sizeof(float));

	/*if (it == 0)
	{
		gws.chaoticC.clear();
		gws.chaoticA.clear();
	}*/
	//srand((unsigned)time(NULL));
	float c[300];
	float D[300];
	float A[300];

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
			//curand_init((unsigned long long)(seed*1e6), 0, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution
			/*if(i==18)
			printf("i:%d %f ",i, RANDOM);*/
			rndC = abs(RANDOM); //if(rndC==1.0)cout << rndC << endl;
			c[i] = (2 * rndC);
			//gws->chaoticC[i] = rndC;
			//gws.chaoticC.push_back(rndC);
		}
		//else
		//{
		//	rndC = cos(0.5*acos(gws->chaoticC[i]));
		//	//rndC = 0.5*gws.chaoticC[i] * (1 - gws.chaoticC[i]);
		//	c[i] = (2 * rndC);
		//	gws->chaoticC[i] = rndC;
		//}

	}
	//float* D = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		D[i] = (abs(c[i] * leaderRotationAngle[i] - gws->rotationAngle[i]));
	}
	//float* A = (float*)malloc(nVar * sizeof(float));

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
			//curand_init((unsigned long long)(seed*1e6), 0, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution

			rndA = abs(RANDOM);
			A[i] = (2 * a*rndA - a);
			//gws->chaoticA[i] = rndA;
			//gws.chaoticA.push_back(rndA);
		}
		//else
		//{
		//	rndA = cos(0.5*acos(gws->chaoticA[i]));
		//	//rndA = 0.5*gws.chaoticA[i] * (1 - gws.chaoticA[i]);
		//	A[i]=(2 * a*rndA - a);
		//	gws->chaoticA[i] = rndA;
		//}

	}
	//float* X = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		X[i] = (leaderRotationAngle[i] - A[i] * abs(D[i]));
	}
	/*free(c);
	free(D);
	free(A);*/

	//return X;
}
__device__ void cuda_GreyWolfOptimizer(int nVar, Triangle* triangles, Triangle* TriList,
	QuadFace* quads, QuadFace* QuadList, int triNum, int quadNum, int nodeId, int lane_id);
__device__ void cuda_GWO(BeamPlugin** beams,int beamNum,short* flag,int nVar, Triangle* triangles,
	QuadFace* quads, int triNum, int quadNum, int nodeId, int lane_id, NodeStatistic* statis,float3 startP)
{
	float MaxIt = 100;

	__shared__ bool Alpha_inverted;
	Alpha_inverted = false;
	__shared__ float Alpha_pos[300];
	__shared__ float Alpha_score;
	Alpha_score = INF;

	__shared__ float Beta_pos[300];
	__shared__ float Beta_score;
	Beta_score = INF;

	__shared__ float Delta_pos[300];
	__shared__ float Delta_score;
	Delta_score = INF;

	for (int i = 0; i < 300; i++)
	{
		Alpha_pos[i] = 0;
		Beta_pos[i] = 0;
		Delta_pos[i] = 0;
	}

	__shared__ GreyWolves gws[96];
	__shared__ float fitness[96];

	Triangle triList[100];
	QuadFace quadList[150];

	gws[lane_id].rotationAngle = (float*)malloc((nVar+beamNum) * sizeof(float));
	/*gws[lane_id].chaoticA = (float*)malloc((nVar+beamNum) * sizeof(float));
	gws[lane_id].chaoticC = (float*)malloc((nVar+beamNum) * sizeof(float));*/
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
			curand_init((unsigned long long)clock(), cnt+k, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//RANDOM = RANDOM * 2 * PI;
			if (k != 0)
			{
				gws[lane_id].rotationAngle[cnt + k] = (RANDOM+0.5) *(- 2 * PI / rndThetaNum);
				//gws[lane_id].rotationAngle[cnt + k] = Min(Max((double)(gws[lane_id].rotationAngle[cnt + k]), -2*PI/rndThetaNum*2), 0);
			}	
			if (k != 0)
				thetaSum += gws[lane_id].rotationAngle[cnt + k];
			if (k == 0)
				gws[lane_id].rotationAngle[cnt + k] = (RANDOM - 0.5) * PI*2;
				//gws[lane_id].rotationAngle[cnt + k] = -RANDOM * 2 * PI;
		}
		for (int k = 1; k <= rndThetaNum; k++)
		{
			gws[lane_id].rotationAngle[cnt + k] /= thetaSum;
			gws[lane_id].rotationAngle[cnt + k] *= -2 * PI;
		}
		cnt += rndThetaNum+1;
	}

	float record[100];
	float tmpRecord[20];
	int tmp_iter = 0;

	int total_iter = -1;
	int force_iter = 0; bool converge = false;
	//record.resize(MaxIt*3);
	//main loop
	for (float it = 0; it < MaxIt; it++)
	{
		float a = 2 - it * ((2) / MaxIt);

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
					float theta = gws[lane_id].rotationAngle[cnt];
					float3 O = tri->o[v];
					float m[16];
					cuda_getRotationMatrix(diskNorm, theta, O, m);

					float3 rotateV = make_float3(tri->p[v].x - O.x, tri->p[v].y - O.y, tri->p[v].z - O.z);
					float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
					RV = cuda_multiply(m, RV);
					tri->p[v] = make_float3(RV.x, RV.y, RV.z);
					prevP = tri->p[v];
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
						float theta = gws[lane_id].rotationAngle[cnt + k];
						float3 O = tri2->o[v];
						float m[16];
						cuda_getRotationMatrix(diskNorm, theta, O, m);

						float3 rotateV = make_float3(prevP.x - O.x, prevP.y - O.y, prevP.z - O.z);
						float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
						RV = cuda_multiply(m, RV);
						tri2->p[v] = make_float3(RV.x, RV.y, RV.z);
						prevP = tri2->p[v];
						break;
					}
				}
				
				//tri = tri2;
	
			}
			cnt += b->arcNum[idx]+1;
		}
		//所有三角形的normal更新
		for (int j = 0; j < triNum; j++)
		{
			Triangle* t = &triList[j];
			float3 e1 = make_float3(t->p[1].x - t->p[0].x, t->p[1].y - t->p[0].y, t->p[1].z - t->p[0].z);
			float3 e2 = make_float3(t->p[2].x - t->p[0].x, t->p[2].y - t->p[0].y, t->p[2].z - t->p[0].z);
			t->normal = cuda_cross(e1, e2);
			t->normal = cuda_make_norm(t->normal);
		}
		//计算cost
		bool inverted = false;
		cuda_findAdjTriangles(triList, triNum);//否则找到的邻居三角形还是以前的
		cuda_findQuadFaces(triList, quadList, triNum, inverted);
		gws[lane_id].singleCost = cuda_measureSingleCost(triList, triNum, quadList, quadNum, gws[lane_id], nVar, inverted);
		gws[lane_id].inverted = inverted;

		fitness[lane_id] = gws[lane_id].singleCost;

		__syncthreads();
		if (lane_id == 0)
		{
			for (int i = 0; i < 96; i++)
			{
				//Update Alpha, Beta, and Delta
				if (fitness[i] < Alpha_score)
				{
					Alpha_score = fitness[i];
					//copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Alpha_pos); //Alpha_pos = gws[i].rotationAngle;
					for (int j = 0; j < nVar+beamNum; j++)
						Alpha_pos[j] = gws[i].rotationAngle[j];
					//memcpy(Alpha_pos, gws[i].rotationAngle, nVar * sizeof(float));
					//Alpha = gws[i];
					Alpha_inverted = gws[i].inverted;
				}
				if (fitness[i] > Alpha_score && fitness[i] < Beta_score)
				{
					Beta_score = fitness[i];
					//copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Beta_pos); //Beta_pos = gws[i].rotationAngle;
					for (int j = 0; j < nVar + beamNum; j++)
						Beta_pos[j] = gws[i].rotationAngle[j];
					//memcpy(Beta_pos, gws[i].rotationAngle, nVar * sizeof(float));
					//Beta = gws[i];
				}
				if (fitness[i] > Alpha_score && fitness[i] > Beta_score && fitness[i] < Delta_score)
				{
					Delta_score = fitness[i];
					//copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Delta_pos); //Delta_pos = gws[i].rotationAngle;
					for (int j = 0; j < nVar + beamNum; j++)
						Delta_pos[j] = gws[i].rotationAngle[j];
					//memcpy(Delta_pos, gws[i].rotationAngle, nVar * sizeof(float));
					//Delta = gws[i];
				}
			}
		}
		total_iter++;
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
		float X1[300];
		float X2[300];
		float X3[300];
		cuda_computeX123(Alpha_pos, &gws[lane_id], nVar+beamNum, a, 0, X1);
		cuda_computeX123(Beta_pos, &gws[lane_id], nVar + beamNum, a, 0, X2);
		cuda_computeX123(Delta_pos, &gws[lane_id], nVar + beamNum, a, 0, X3);

		cnt = 0;
		for (int j = 0; j < beamNum; j++)
		{
			BeamPlugin* b = beams[j];
			int idx = flag[j] - 1;
			int rndThetaNum = b->arcNum[idx];
			float thetaSum = 0;
			for (int k = 0; k <= rndThetaNum; k++)
			{
				gws[lane_id].rotationAngle[cnt+k] = (X1[cnt+k] + X2[cnt+k] + X3[cnt+k]) / 3;
				if(k>0)
				gws[lane_id].rotationAngle[cnt + k] = Min(Max((double)(gws[lane_id].rotationAngle[cnt + k]), -2*PI/rndThetaNum*1.4), -2 * PI / rndThetaNum*0.2);
				if(k==0)
					gws[lane_id].rotationAngle[cnt + k] = Min(Max((double)(gws[lane_id].rotationAngle[cnt + k]), - PI), PI);
				if(k>0)
					thetaSum+= gws[lane_id].rotationAngle[cnt + k];
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
					gws[lane_id].rotationAngle[cnt + k] /= thetaSum;
					gws[lane_id].rotationAngle[cnt + k] *= -2 * PI;
				}
			}
			cnt += rndThetaNum+1;
		}
		/*for (int k = 0; k < nVar; k++)
		{
			gws[lane_id].rotationAngle[k] = (X1[k] + X2[k] + X3[k]) / 3;
			gws[lane_id].rotationAngle[k] = Min(Max((double)(gws[lane_id].rotationAngle[k]), -PI), PI);
		}*/
		if (lane_id == 0)
		{
			//printf("%d %f %d\n", Alpha_inverted, Alpha_score, nodeId);
		}
		
		record[total_iter] = Alpha_score;
		//tmpRecord[tmp_iter] = Alpha_score;
		/*Y.push_back(Alpha_score);
		invertFlag.push_back(Alpha_inverted);*/
		//tmp_iter++;
		
		if(Alpha_inverted==true)
		force_iter++;
		else
		{
			if (lane_id == 0 && converge == false)
			{
				//printf("degree: %d phase 2 iternum: %d nodeid: %d\n", beamNum, force_iter, nodeId);
				converge = true;
			}
				
		}
		//if (it == MaxIt - 1)
		//force_iter = 3000; it = MaxIt - 1; Alpha_inverted = true; total_iter = 51;
		if(total_iter>50)
		{
			//判断是否翻转
			if (Alpha_inverted == false && (abs(record[(int)total_iter - 1] - record[(int)total_iter - 10]) < 1||force_iter>=3000))
			{
				if (lane_id == 0)
				{
					statis[nodeId].converge = true;
					statis[nodeId].iterNum = force_iter;
					//printf("iteration number: %d\n", force_iter);
					//for (int j = 0; j < nVar; j++)
					//	Alpha_pos[j] = 0.2;
					//	//printf("%f ", Alpha_pos[j]);
					//printf("\n");
					 cnt = 0;
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
								float m[16];
								cuda_getRotationMatrix(diskNorm, theta, O, m);

								float3 rotateV = make_float3(tri->p[v].x - O.x, tri->p[v].y - O.y, tri->p[v].z - O.z);
								float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
								RV = cuda_multiply(m, RV);
								tri->p[v] = make_float3(RV.x, RV.y, RV.z);
								prevP = tri->p[v];
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
									float m[16];
									cuda_getRotationMatrix(diskNorm, theta, O, m);

									float3 rotateV = make_float3(prevP.x - O.x, prevP.y - O.y, prevP.z - O.z);
									float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
									RV = cuda_multiply(m, RV);
									tri2->p[v] = make_float3(RV.x, RV.y, RV.z);
									prevP = tri2->p[v];
									break;
								}
							}

							//tri = tri2;

						}
						cnt += b->arcNum[idx]+1;
					}
					//所有三角形的normal更新
					for (int j = 0; j < triNum; j++)
					{
						Triangle* t = &triangles[j];
						float3 e1 = make_float3(t->p[1].x - t->p[0].x, t->p[1].y - t->p[0].y, t->p[1].z - t->p[0].z);
						float3 e2 = make_float3(t->p[2].x - t->p[0].x, t->p[2].y - t->p[0].y, t->p[2].z - t->p[0].z);
						t->normal = cuda_cross(e1, e2);
						t->normal = cuda_make_norm(t->normal);
						t->flag = 0;
					}
					/*for (int j = 0; j < quadNum; j++)
					{
						QuadFace* q = &quads[j];
						q->convergeF = 1;
					}*/
					//计算cost
					bool inverted = false;
					cuda_findAdjTriangles(triangles, triNum);//否则找到的邻居三角形还是以前的
					cuda_findQuadFaces(triangles, quads, triNum, inverted);
					cuda_measureSingleCost(triangles, triNum, quads, quadNum, gws[0], nVar, inverted);
					//printf("inverted: %d\n", inverted);
					/*triangles = triList;
					quads = quadList;*/
					
				}
				__syncthreads();
				break;
			}
			else if(force_iter<3000&& it == MaxIt - 1)
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
			else if (force_iter >= 3000 && it == MaxIt - 1)
			{
			if (lane_id == 0)
			{
				//printf("iteration number: %d\n", force_iter);
				//for (int j = 0; j < nVar; j++)
				//	Alpha_pos[j] = 0.2;
				//	//printf("%f ", Alpha_pos[j]);
				//printf("\n");
				for (int j = 0; j < beamNum; j++)
				{
					BeamPlugin* b = beams[j];
					short idx = flag[j] - 1;
					float currentLength = cuda_Norm(cuda_Subtraction(b->axis.p[idx], startP));
					if ( currentLength!= b->convexLength[idx])
					{
						float sub = b->convexLength[idx] - currentLength;
						float3 start = b->axis.p[idx];
						float3 end = b->axis.p[cuda_mod(idx + 1, 2)];
						float3 direction = make_float3(end.x - start.x, end.y - start.y, end.z - start.z);
						direction = cuda_make_norm(direction);
						start.x += direction.x* sub;
						start.y += direction.y* sub;
						start.z += direction.z* sub;
						b->axis.p[idx] = start;
						atomicAdd(&(b->length), -sub);
					}
				}
				for (int j = 0; j < triNum; j++)
				{
					BeamPlugin *b[3];
					Triangle* t = &triangles[j];
					
					for (int k = 0; k < 3; k++)
					{
						int idx = t->triWithBeamid[k];
						b[k] = beams[idx];
						t->o[k] = b[k]->axis.p[flag[idx] - 1];
					}
					cuda_getTriangle(b, startP, triangles, j);
					t->inverted[0] = 3;
				}
				
				//cnt = 0;
				//for (int j = 0; j < beamNum; j++)
				//{
				//	BeamPlugin* b = beams[j];
				//	int idx = flag[j] - 1;
				//	Triangle* tri = triangles + b->arcTriList[idx][0];
				//	float3 prevP;
				//	for (int v = 0; v < 3; v++)
				//	{
				//		if (tri->triWithBeamid[v] == j)
				//		{
				//			float3 diskNorm = tri->diskNorm[v];
				//			float theta = Alpha_pos[cnt];
				//			float3 O = tri->o[v];
				//			float m[16];
				//			cuda_getRotationMatrix(diskNorm, theta, O, m);

				//			float3 rotateV = make_float3(tri->p[v].x - O.x, tri->p[v].y - O.y, tri->p[v].z - O.z);
				//			float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
				//			RV = cuda_multiply(m, RV);
				//			tri->p[v] = make_float3(RV.x, RV.y, RV.z);
				//			prevP = tri->p[v];
				//			break;
				//		}

				//	}
				//	for (int k = 1; k < b->arcNum[idx]; k++)
				//	{
				//		//Triangle* tri = triList + b->arcTriList[idx][k];
				//		Triangle* tri2 = triangles + b->arcTriList[idx][k];
				//		for (int v = 0; v < 3; v++)
				//		{
				//			if (tri2->triWithBeamid[v] == j)
				//			{
				//				float3 diskNorm = tri2->diskNorm[v];
				//				float theta = Alpha_pos[cnt + k];
				//				float3 O = tri2->o[v];
				//				float m[16];
				//				cuda_getRotationMatrix(diskNorm, theta, O, m);

				//				float3 rotateV = make_float3(prevP.x - O.x, prevP.y - O.y, prevP.z - O.z);
				//				float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
				//				RV = cuda_multiply(m, RV);
				//				tri2->p[v] = make_float3(RV.x, RV.y, RV.z);
				//				prevP = tri2->p[v];
				//				break;
				//			}
				//		}

				//		//tri = tri2;

				//	}
				//	cnt += b->arcNum[idx] + 1;
				//}
				////所有三角形的normal更新
				//for (int j = 0; j < triNum; j++)
				//{
				//	Triangle* t = &triangles[j];
				//	float3 e1 = make_float3(t->p[1].x - t->p[0].x, t->p[1].y - t->p[0].y, t->p[1].z - t->p[0].z);
				//	float3 e2 = make_float3(t->p[2].x - t->p[0].x, t->p[2].y - t->p[0].y, t->p[2].z - t->p[0].z);
				//	t->normal = cuda_cross(e1, e2);
				//	t->normal = cuda_make_norm(t->normal);
				//	t->flag = 1;
				//	//t->inverted = true;
				//}
				///*for (int j = 0; j < quadNum; j++)
				//{
				//	QuadFace* q = &quads[j];
				//	q->convergeF = 0;
				//}*/
				////计算cost
				//bool inverted = false;
				//cuda_findAdjTriangles(triangles, triNum);//否则找到的邻居三角形还是以前的
				//cuda_findQuadFaces(triangles, quads, triNum, inverted);
				//cuda_measureSingleCost(triangles, triNum, quads, quadNum, gws[0], nVar, inverted);
				//printf("inverted: %d\n", inverted);
				/*triangles = triList;
				quads = quadList;*/
				//break;
			}
			__syncthreads();
			//cuda_GreyWolfOptimizer(nVar, triangles, TriList, quads, QuadList, triNum, quadNum, nodeId, lane_id);
			/*it = -1;
			MaxIt = 600;
			total_iter = -1;*/
			}
		}


	}
	free(gws[lane_id].rotationAngle);
	/*free(gws[lane_id].chaoticA);
	free(gws[lane_id].chaoticC);*/
}
__device__ void cuda_GreyWolfOptimizer(BeamPlugin** beams, int beamNum, short* flag,float3 startP,int nVar, Triangle* triangles,
	QuadFace* quads,  int triNum, int quadNum,int nodeId,int lane_id,NodeStatistic* statis)
{
	//int triNum = triangles.size();
	int GreyWolves_num = 100;
	float MaxIt = 100;

	__shared__ bool Alpha_inverted;
	Alpha_inverted = false;
	__shared__ float Alpha_pos[300];// = (float*)malloc(nVar * sizeof(float));// new float[nVar];
	/*if (lane_id < 150)
		Alpha_pos[lane_id] = 0;*/
	//memset(Alpha_pos, 0, nVar * sizeof(float));
	__shared__ float Alpha_score;
	Alpha_score = INF;
	//GreyWolves Alpha; Alpha.rotationAngle = (float*)malloc(nVar * sizeof(float));// new float[nVar];
	//for (int i = 0; i < nVar; i++)
	//	Alpha.rotationAngle[i] = 0;

	__shared__ float Beta_pos[300]; //= (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	/*if (lane_id < 150)
		Beta_pos[lane_id] = 0;*/
	//memset(Beta_pos, 0, nVar * sizeof(float));
	__shared__ float Beta_score;
	Beta_score = INF;
	//GreyWolves Beta; Beta.rotationAngle = (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	//for (int i = 0; i < nVar; i++)
	//	Beta.rotationAngle[i] = 0;

	__shared__ float Delta_pos[300]; //= (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	/*if (lane_id < 150)
		Delta_pos[lane_id] = 0;*/
	//memset(Delta_pos, 0, nVar * sizeof(float));
	__shared__ float Delta_score;
	Delta_score = INF;

	for (int i = 0; i < 300; i++)
	{
		Alpha_pos[i] = 0;
		Beta_pos[i] = 0;
		Delta_pos[i] = 0;
	}
	//GreyWolves Delta; Delta.rotationAngle = (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	//for (int i = 0; i < nVar; i++)
	//	Delta.rotationAngle[i] = 0;

	//std::vector<GreyWolves> gws;
	//gws.resize(GreyWolves_num);
	//GreyWolves* gws = (GreyWolves*)malloc(GreyWolves_num * sizeof(GreyWolves));
	__shared__ GreyWolves gws[128];
	__shared__ float fitness[128];
	//Initialization
	//std::vector<shared_ptr<Triangle>> triList;
	//Triangle* triList;
	//std::vector<shared_ptr<QuadFace>> quadList;
	//QuadFace* quadList;
	Triangle triList[100];
	QuadFace quadList[150];


	//for (int i = 0; i < GreyWolves_num; i++)
	{
		gws[lane_id].rotationAngle = (float*)malloc(nVar * sizeof(float));
		gws[lane_id].chaoticA = (float*)malloc(nVar * sizeof(float));
		gws[lane_id].chaoticC = (float*)malloc(nVar * sizeof(float));
		for (int j = 0; j < nVar; j++)
		{
			////初始化position即随机旋转角
			//std::default_random_engine e;
			//std::uniform_real_distribution<float> u(-PI, PI);
			//e.seed(time(0));
			curandState devStates;
			/*srand(time(0));
			int seed = rand();*/
			curand_init((unsigned long long)clock(), j, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//curand_init((unsigned long long)(seed*1e6), 0, 0, &devStates2);// initialize the state
			//float RANDOM = curand_uniform(&devStates2);// uniform distribution
			RANDOM = (RANDOM - 0.5) * 2 * PI;
			//cout << i * nVar + j << " " << host_rA[i*nVar + j] << endl;
			gws[lane_id].rotationAngle[j] = RANDOM;
		}
		//gws[i].rA = gws[i].rotationAngle.data();
	}

	//float* record = (float*)malloc(MaxIt * sizeof(float));
	//float* tmpRecord = (float*)malloc(20 * sizeof(float));
	float record[100];
	float tmpRecord[20];
	int tmp_iter = 0;
	/*vector<float> Y;
	vector<bool> invertFlag;*/

	int total_iter = 0;
	int force_iter = 0;
	//record.resize(MaxIt*3);
	//main loop
	for (float it = 0; it < MaxIt; it++)
	{
		float a = 2 - it * ((2) / MaxIt);




		//for (int i = 0; i < GreyWolves_num; i++)
		{
			
			/*quadList = (QuadFace*)malloc(quadNum * sizeof(QuadFace));
			triList = (Triangle*)malloc(triNum * sizeof(Triangle));*/
			/*triList[j] = make_shared<Triangle>();*/
			for (int j = 0; j < triNum; j++)
			{
				//Triangle* t = &triList[j];
				/*if (it >= 1)
				{
					if( triList[j].cost < 1.45&&triList[j].projSeg == 0)
				    {
				    for (int k = 0; k < 3; k++)
				    {
					    gws[lane_id].rotationAngle[j * 3 + k] = triList[j].rotationAngle[k];
				    }
				    continue;
				    }
				}*/
				triList[j] = triangles[j];
			}
			//计算在当前三角形位置上旋转随机角度后的位置
			for (int j = 0; j < triNum; j++)
			{
				
				Triangle* t = &triList[j];
				for (int k = 0; k < 3; k++)
				{
					float3 diskNorm = t->diskNorm[k];
					float theta = gws[lane_id].rotationAngle[j * 3 + k];
					t->rotationAngle[k] = theta;
					float3 O = t->o[k];
					float m[16]; 
					cuda_getRotationMatrix(diskNorm, theta, O,m);

					float3 rotateV = make_float3(t->p[k].x - O.x, t->p[k].y - O.y, t->p[k].z - O.z);
					float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
					RV = cuda_multiply(m, RV);
					t->p[k] = make_float3(RV.x, RV.y, RV.z);

					//for (int f = 0; f < 4; f++)
					//{
					//	free(m[f]);// = (float*)malloc(4 * sizeof(float));
					//}
					//free(m);
				}
				float3 e1 = make_float3(t->p[1].x - t->p[0].x, t->p[1].y - t->p[0].y, t->p[1].z - t->p[0].z);
				float3 e2 = make_float3(t->p[2].x - t->p[0].x, t->p[2].y - t->p[0].y, t->p[2].z - t->p[0].z);
				t->normal = cuda_cross(e1, e2);
				t->normal = cuda_make_norm(t->normal);
				//triList[j] = t;
			}
			//计算cost
			bool inverted = false;
			cuda_findAdjTriangles(triList, triNum);//否则找到的邻居三角形还是以前的
			cuda_findQuadFaces(triList, quadList, triNum,inverted);
			gws[lane_id].singleCost = cuda_measureSingleCost(triList, triNum, quadList, quadNum, gws[lane_id],nVar,inverted);
			gws[lane_id].inverted = inverted;

			fitness[lane_id] = gws[lane_id].singleCost;

			__syncthreads();
			if (lane_id == 0)
			{
				for (int i = 0; i < 128; i++)
				{
					//Update Alpha, Beta, and Delta
					if (fitness[i] < Alpha_score)
					{
						Alpha_score = fitness[i];
						//copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Alpha_pos); //Alpha_pos = gws[i].rotationAngle;
						for (int j = 0; j < nVar; j++)
							Alpha_pos[j] = gws[i].rotationAngle[j];
						//memcpy(Alpha_pos, gws[i].rotationAngle, nVar * sizeof(float));
						//Alpha = gws[i];
						Alpha_inverted = gws[i].inverted;
					}
					if (fitness[i] > Alpha_score && fitness[i] < Beta_score)
					{
						Beta_score = fitness[i];
						//copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Beta_pos); //Beta_pos = gws[i].rotationAngle;
						for (int j = 0; j < nVar; j++)
							Beta_pos[j] = gws[i].rotationAngle[j];
						//memcpy(Beta_pos, gws[i].rotationAngle, nVar * sizeof(float));
						//Beta = gws[i];
					}
					if (fitness[i] > Alpha_score && fitness[i] > Beta_score && fitness[i] < Delta_score)
					{
						Delta_score = fitness[i];
						//copy(gws[i].rotationAngle, gws[i].rotationAngle + nVar, Delta_pos); //Delta_pos = gws[i].rotationAngle;
						for (int j = 0; j < nVar; j++)
							Delta_pos[j] = gws[i].rotationAngle[j];
						//memcpy(Delta_pos, gws[i].rotationAngle, nVar * sizeof(float));
						//Delta = gws[i];
					}
				}
			}
			
			/*free(triList);
			free(quadList);*/
			//for (int v = 0; v < triNum; v++)
			//{
			//	
			//	free(triList[v].adjTriangles);
			//	free(triList[v].adjQuadFaces);
			//	//free(triList[v]);
			//}
			/*free(triList);
			free(quadList);*/
			//for (int v = 0; v < quadNum; v++)
			//{
			//	free(quadList[v]);
			//}

		}

		//for (int i = 0; i < GreyWolves_num; i++)
		{
			int It = 0;
			if (total_iter > 10)
			{
				//It = 0;
				//if (record.back() == *(record.end() - 20))
				if(abs(record[total_iter -1]-record[total_iter -10])<0.5)
				{
					It = it; //printf("chaotic!\n");
				}
				else
				{
					It = 0;
				}
			}
			float X1[300];
			float X2[300];
			float X3[300];
			cuda_computeX123(Alpha_pos, &gws[lane_id], nVar, a, 0,X1);
			cuda_computeX123(Beta_pos, &gws[lane_id], nVar, a, 0,X2);
			cuda_computeX123(Delta_pos, &gws[lane_id], nVar, a, 0,X3);

			for (int k = 0; k < nVar; k++)
			{
				gws[lane_id].rotationAngle[k] = (X1[k] + X2[k] + X3[k]) / 3;
				gws[lane_id].rotationAngle[k] = Min(Max((double)(gws[lane_id].rotationAngle[k]), -PI), PI);
			}
			//free(X1); free(X2); free(X3);
		}
		if (lane_id == 0)
		{
			printf("%d %f %d\n", Alpha_inverted, Alpha_score, nodeId);
		}
			record[total_iter] = Alpha_score;
			tmpRecord[tmp_iter] = Alpha_score;
			/*Y.push_back(Alpha_score);
			invertFlag.push_back(Alpha_inverted);*/
			tmp_iter++;
			total_iter++;
			force_iter++;
		

		//重新随机化，加速收敛
		//if (tmp_iter > 10 && tmp_iter <= 20)
		//{
		//	if (tmpRecord[tmp_iter - 1] == tmpRecord[tmp_iter - 10])
		//	{
		//		//for (int i = 0; i < GreyWolves_num; i++)
		//		{
		//			//gws[i].rotationAngle.clear();
		//			for (int j = 0; j < nVar; j++)
		//			{
		//				////初始化position即随机旋转角
		//				//std::default_random_engine e;
		//				//std::uniform_real_distribution<float> u(-PI, PI);
		//				//e.seed(time(0));
		//				curandState devStates;
		//				/*srand(time(0));
		//				int seed = rand();*/
		//				curand_init((unsigned long long)clock(), j, 0, &devStates);// initialize the state
		//				float RANDOM = curand_uniform(&devStates);// uniform distribution
		//				//curand_init((unsigned long long)(seed*1e6), 0, 0, &devStates2);// initialize the state
		//				//float RANDOM = curand_uniform(&devStates2);// uniform distribution
		//				RANDOM = (RANDOM - 0.5) * 2 * PI;

		//				gws[lane_id].rotationAngle[j] = RANDOM;
		//			}

		//		}
		//		//free(tmpRecord);
		//		//tmpRecord = (float*)malloc(20 * sizeof(float));
		//		tmp_iter = 0;
		//	}
		//}
		//if (tmp_iter >= 20)
		//{
		//	//free(tmpRecord);
		//	//tmpRecord = (float*)malloc(20 * sizeof(float));
		//	tmp_iter = 0;
		//}

		if (it == MaxIt - 1)
		{
			//判断是否翻转
			if (Alpha_inverted == false && abs(record[(int)MaxIt - 1] - record[(int)MaxIt - 11]) < 1)
			{
				if (lane_id == 0)
				{
					statis[nodeId].converge = true;
					statis[nodeId].iterNum = force_iter;
					//atomicAdd(&(convergeNum[0]), 1);
					for (int j = 0; j < triNum; j++)
					{
						Triangle *t = &triangles[j];
						for (int k = 0; k < 3; k++)
						{
							float3 diskNorm = t->diskNorm[k];
							float theta = Alpha_pos[j * 3 + k];
							float3 O = t->o[k];
							float m[16];
							cuda_getRotationMatrix(diskNorm, theta, O, m);

							float3 rotateV = make_float3(t->p[k].x - O.x, t->p[k].y - O.y, t->p[k].z - O.z);
							float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
							RV = cuda_multiply(m, RV);
							t->p[k] = make_float3(RV.x, RV.y, RV.z);

							//for (int f = 0; f < 4; f++)
							//{
							//	free(m[f]);// = (float*)malloc(4 * sizeof(float));
							//}
							//free(m);
						}
						float3 e1 = make_float3(t->p[1].x - t->p[0].x, t->p[1].y - t->p[0].y, t->p[1].z - t->p[0].z);
						float3 e2 = make_float3(t->p[2].x - t->p[0].x, t->p[2].y - t->p[0].y, t->p[2].z - t->p[0].z);
						t->normal = cuda_cross(e1, e2);
						t->normal = cuda_make_norm(t->normal);
						t->flag = 0;
						//triangles[j] = t;
					}
					//计算cost
					bool inverted = false;
					cuda_findAdjTriangles(triangles, triNum);//否则找到的邻居三角形还是以前的
					cuda_findQuadFaces(triangles, quads, triNum, inverted);

					/*triangles = triList;
					quads = quadList;*/
					//break;
				}
				
			}
			else if(force_iter<600)
			{

				/*triList.clear();
				quadList.clear();*/
				it = -1;
				MaxIt = 100;
				//if(record.size()==MaxIt)
				//free(record);
				//record = (float*)malloc(MaxIt * sizeof(float));
				total_iter = 0;
				//record.resize(MaxIt);
				//break;

			}
			else
			{
				if (lane_id == 0)
				{
					for (int j = 0; j < beamNum; j++)
					{
						BeamPlugin* b = beams[j];
						short idx = flag[j] - 1;
						float currentLength = cuda_Norm(cuda_Subtraction(b->axis.p[idx], startP));
						if (currentLength != b->convexLength[idx])
						{
							float sub = b->convexLength[idx] - currentLength;
							float3 start = b->axis.p[idx];
							float3 end = b->axis.p[cuda_mod(idx + 1, 2)];
							float3 direction = make_float3(end.x - start.x, end.y - start.y, end.z - start.z);
							direction = cuda_make_norm(direction);
							start.x += direction.x * sub;
							start.y += direction.y * sub;
							start.z += direction.z * sub;
							b->axis.p[idx] = start;
							atomicAdd(&(b->length), -sub);
						}
					}
					for (int j = 0; j < triNum; j++)
					{
						BeamPlugin* b[3];
						Triangle* t = &triangles[j];

						for (int k = 0; k < 3; k++)
						{
							int idx = t->triWithBeamid[k];
							b[k] = beams[idx];
							t->o[k] = b[k]->axis.p[flag[idx] - 1];
						}
						cuda_getTriangle(b, startP, triangles, j);
						t->inverted[0] = 3;
					}
					//for (int j = 0; j < triNum; j++)
					//{
					//	Triangle *t = &triangles[j];
					//	for (int k = 0; k < 3; k++)
					//	{
					//		float3 diskNorm = t->diskNorm[k];
					//		float theta = Alpha_pos[j * 3 + k];
					//		float3 O = t->o[k];
					//		float m[16];
					//		cuda_getRotationMatrix(diskNorm, theta, O, m);

					//		float3 rotateV = make_float3(t->p[k].x - O.x, t->p[k].y - O.y, t->p[k].z - O.z);
					//		float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
					//		RV = cuda_multiply(m, RV);
					//		t->p[k] = make_float3(RV.x, RV.y, RV.z);

					//		//for (int f = 0; f < 4; f++)
					//		//{
					//		//	free(m[f]);// = (float*)malloc(4 * sizeof(float));
					//		//}
					//		//free(m);
					//	}
					//	float3 e1 = make_float3(t->p[1].x - t->p[0].x, t->p[1].y - t->p[0].y, t->p[1].z - t->p[0].z);
					//	float3 e2 = make_float3(t->p[2].x - t->p[0].x, t->p[2].y - t->p[0].y, t->p[2].z - t->p[0].z);
					//	t->normal = cuda_cross(e1, e2);
					//	t->normal = cuda_make_norm(t->normal);
					//	t->flag = 0;
					//	//triangles[j] = t;
					//}
					////计算cost
					//bool inverted = false;
					//cuda_findAdjTriangles(triangles, triNum);//否则找到的邻居三角形还是以前的
					//cuda_findQuadFaces(triangles, quads, triNum, inverted);

					/*triangles = triList;
					quads = quadList;*/
					//break;
				}
				__syncthreads();
			}

		}
	}
	//for (int i = 0; i < GreyWolves_num; i++)
	{
		free(gws[lane_id].rotationAngle);
		free(gws[lane_id].chaoticA);
		free(gws[lane_id].chaoticC);
	}
	//free(gws);
	/*free(record);
	free(tmpRecord);*/
	//free(Alpha_pos);
	//free(Alpha.rotationAngle);
	//free(Beta_pos);
	//free(Beta.rotationAngle);
	//free(Delta_pos);
	//free(Delta.rotationAngle);

}
__device__ void cuda_BuildInitialHull(Point* dots,int pointNum,Triangle* Mesh)
{
	Point* initialVertices[4];
	Triangle* initialHullFaces[4];
	float3 P[4];
	for (int i = 0; i < 3; i++)
	{
		initialVertices[i] = dots + i;
		P[i] = make_float3((dots + i)->x, (dots + i)->y, (dots + i)->z);
		(dots + i)->isVisited = true;
	}
	float3 n_circumcenter = cuda_ComputeNegativeCircumcenter(P);

	initialVertices[3] = dots + 3;
	P[3] = make_float3((dots + 3)->x, (dots + 3)->y, (dots + 3)->z);
	float dis = cuda_Norm(cuda_Subtraction(P[3], n_circumcenter));
	//找到距离-C0最近的点
	for (int i = 4; i<pointNum; i++)
	{
		float3 Pk = make_float3((dots+i)->x, (dots + i)->y, (dots + i)->z);
		//判断四点是否共面，若共面，continue
		float3 P0toPk = cuda_Subtraction(Pk, P[0]);
		float3 normal = cuda_cross(cuda_Subtraction(P[1], P[0]), cuda_Subtraction(P[2], P[0]));
		if (cuda_Dot(normal, P0toPk) == 0)
			continue;
		float tmpDis = cuda_Norm(cuda_Subtraction(Pk, n_circumcenter));
		if (tmpDis < dis)
		{
			initialVertices[3] = dots+i;
			P[3] = Pk;
			dis = tmpDis;
		}
	}

	//确定三角形法线朝向与半球位置一致
	float3 normal = cuda_cross(cuda_Subtraction(P[1], P[0]), cuda_Subtraction(P[2], P[0]));
	float3 CtoP = P[0];
	if (cuda_Dot(normal, CtoP) < 0)
	{
		Point* vtmp = initialVertices[0];
		initialVertices[0] = initialVertices[1];
		initialVertices[1] = vtmp;

		float3 ptmp = P[0];
		P[0] = P[1];
		P[1] = ptmp;
		normal = cuda_cross(cuda_Subtraction(P[1], P[0]), cuda_Subtraction(P[2], P[0]));
	}
	//根据Pk确定三角形法线朝向
	float3 P0toPk = cuda_Subtraction(P[3], P[0]);
	if (cuda_Dot(normal, P0toPk) > 0)
	{
		Point* vtmp = initialVertices[0];
		initialVertices[0] = initialVertices[1];
		initialVertices[1] = vtmp;

		float3 ptmp = P[0];
		P[0] = P[1];
		P[1] = ptmp;
		normal = cuda_cross(cuda_Subtraction(P[1], P[0]), cuda_Subtraction(P[2], P[0]));
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

		Triangle* tri = Mesh+i;//(Triangle*)malloc(sizeof(Triangle));
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

		//Mesh[i] = tri;
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
__device__ float cuda_GetDeterminant(float matrix[])
{
	// inversed for left handed coordinate system
	float determinant = matrix[2] * matrix[4] * matrix[6]
		+ matrix[0] * matrix[5] * matrix[7]
		+ matrix[1] * matrix[3] * matrix[8]
		- matrix[0] * matrix[4] * matrix[8]
		- matrix[1] * matrix[5] * matrix[6]
		- matrix[2] * matrix[3] * matrix[7];

	// adjust result based on float number accuracy, otherwise causing deadloop
	return abs(determinant) <= FLT_EPSILON ? 0 : determinant;
}
__device__ float cuda_GetDeterminant(float3 v0, float3 v1, float3 v2)
{
	float matrix[] = {
		v0.x, v0.y, v0.z,
		v1.x, v1.y, v1.z,
		v2.x, v2.y, v2.z
	};

	return cuda_GetDeterminant(matrix);
}
__device__ void cuda_FixNeighborhood(Triangle* target, Triangle* oldNeighbor, Triangle* newNeighbor)
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
__device__ bool cuda_TrySwapDiagonal(Triangle* t0, Triangle* t1);
__device__ void cuda_DoLocalOptimization(Triangle* t0, Triangle* t1)
{
	//_Statistics[1]++;
	Triangle* stk0[100];
	Triangle* stk1[100];

	stk0[0] = t0; stk1[0] = t1;
	short cnt = 1;
	bool flag = false;
	while (cnt!=0)
	{
		Triangle* t0 = stk0[cnt-1];
		Triangle* t1 = stk1[cnt-1];
		cnt--;
		//stk0.erase(stk0.end()-1); stk1.erase(stk1.end()-1);

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
			if (cuda_GetDeterminant(matrix) >= 0)
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

						cuda_FixNeighborhood(t0->adjTriangles[(j + 1) % 3], t1, t0);
						cuda_FixNeighborhood(t1->adjTriangles[(k + 1) % 3], t0, t1);

						//DoLocalOptimization(t0, t0->adjTriangles[j]);
						//DoLocalOptimization(t0, t0->adjTriangles[(j + 1) % 3]);
						//DoLocalOptimization(t1, t1->adjTriangles[k]);
						//DoLocalOptimization(t1, t1->adjTriangles[(k + 1) % 3]);
						stk0[cnt] = t1; stk1[cnt] = t1->adjTriangles[(k + 1) % 3]; cnt++;
						stk0[cnt] = t1; stk1[cnt] = t1->adjTriangles[k]; cnt++;
						stk0[cnt] = t0; stk1[cnt] = t0->adjTriangles[(j + 1) % 3]; cnt++;
						stk0[cnt] = t0; stk1[cnt] = t0->adjTriangles[j]; cnt++;

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

	//	float matrix[] = {
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
	//	//std::cout << GetDeterminant(matrix) << std::endl;
	//	if (cuda_GetDeterminant(matrix) >= 0)
	//	{
	//		// terminate after optimized
	//		break;
	//	}

	//	if (cuda_TrySwapDiagonal(t0, t1))
	//	{
	//		return;
	//	}
	//}
}

__device__ bool cuda_TrySwapDiagonal(Triangle* t0, Triangle* t1)
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

				cuda_FixNeighborhood(t0->adjTriangles[(j + 1) % 3], t1, t0);
				cuda_FixNeighborhood(t1->adjTriangles[(k + 1) % 3], t0, t1);

				cuda_DoLocalOptimization(t0, t0->adjTriangles[j]);
				cuda_DoLocalOptimization(t0, t0->adjTriangles[(j + 1) % 3]);
				cuda_DoLocalOptimization(t1, t1->adjTriangles[k]);
				cuda_DoLocalOptimization(t1, t1->adjTriangles[(k + 1) % 3]);

				return true;
			}
		}
	}

	return false;
}
__device__ void cuda_AssignNeighbors(Triangle* tri, Triangle* t0, Triangle* t1, Triangle* t2)
{
	tri->adjTriangles[0] = t0;
	tri->adjTriangles[1] = t1;
	tri->adjTriangles[2] = t2;
}

__device__ void cuda_SplitTriangle(Triangle* Mesh,int meshIdx,Triangle* tri, Point* dot)
{
	//Triangle* newTriangle1 = new triangle(dot, tri->Vertex[1], tri->Vertex[2]);
	//Triangle* newTriangle2 = new triangle(dot, tri->Vertex[2], tri->Vertex[0]);
	Triangle* newTriangle1 = Mesh+meshIdx;// (Triangle*)malloc(sizeof(Triangle));
	//Mesh[meshIdx] = newTriangle1;
	newTriangle1->p[0] = make_float3(dot->x, dot->y, dot->z); newTriangle1->triWithBeamid[0] = dot->id;
	newTriangle1->p[1] = tri->p[1]; newTriangle1->triWithBeamid[1] = tri->triWithBeamid[1];
	newTriangle1->p[2] = tri->p[2]; newTriangle1->triWithBeamid[2] = tri->triWithBeamid[2];
	
	Triangle* newTriangle2 = Mesh+meshIdx + 1;// (Triangle*)malloc(sizeof(Triangle));
	//Mesh[meshIdx + 1] = newTriangle2;
	newTriangle2->p[0] = make_float3(dot->x, dot->y, dot->z); newTriangle2->triWithBeamid[0] = dot->id;
	newTriangle2->p[1] = tri->p[2]; newTriangle2->triWithBeamid[1] = tri->triWithBeamid[2];
	newTriangle2->p[2] = tri->p[0]; newTriangle2->triWithBeamid[2] = tri->triWithBeamid[0];

	tri->p[2] = tri->p[1]; tri->triWithBeamid[2] = tri->triWithBeamid[1];
	tri->p[1] = tri->p[0]; tri->triWithBeamid[1] = tri->triWithBeamid[0];
	tri->p[0] = make_float3(dot->x, dot->y, dot->z); tri->triWithBeamid[0] = dot->id;

	//AssignNeighbors
	//newTriangle1->AssignNeighbors(tri, tri->Neighbor[1], newTriangle2);
	cuda_AssignNeighbors(newTriangle1, tri, tri->adjTriangles[1], newTriangle2);
	/*newTriangle1->adjTriangles[0] = tri;
	newTriangle1->adjTriangles[1] = tri->adjTriangles[1];
	newTriangle1->adjTriangles[2] = newTriangle2;*/
	//newTriangle2->AssignNeighbors(newTriangle1, tri->Neighbor[2], tri);
	cuda_AssignNeighbors(newTriangle2, newTriangle1, tri->adjTriangles[2], tri);
	/*newTriangle2->adjTriangles[0] = newTriangle1;
	newTriangle2->adjTriangles[1] = tri->adjTriangles[2];
	newTriangle2->adjTriangles[2] = tri;*/
	//tri->AssignNeighbors(newTriangle2, tri->Neighbor[0], newTriangle1);
	cuda_AssignNeighbors(tri, newTriangle2, tri->adjTriangles[0], newTriangle1);
	/*tri->adjTriangles[0] = newTriangle2;
	tri->adjTriangles[1] = tri->adjTriangles[0];
	tri->adjTriangles[2] = newTriangle1;*/

	cuda_FixNeighborhood(newTriangle1->adjTriangles[1], tri, newTriangle1);
	cuda_FixNeighborhood(newTriangle2->adjTriangles[1], tri, newTriangle2);

	/*_Mesh->push_back(newTriangle1);
	_Mesh->push_back(newTriangle2);*/

	// optimize triangles according to delaunay triangulation definition
	cuda_DoLocalOptimization(tri, tri->adjTriangles[1]);
	cuda_DoLocalOptimization(newTriangle1, newTriangle1->adjTriangles[1]);
	cuda_DoLocalOptimization(newTriangle2, newTriangle2->adjTriangles[1]);
}

__device__ void cuda_InsertDot(Point* dot,Triangle* Mesh,int currentMeshNum)
{
	float det[] = { 0, 0, 0 };

	/*vector<triangle*>::iterator it;
	it = _Mesh->begin();*/
	Triangle* tri = Mesh;

	short flag = 0;
	int i = 0;
	//while (it != _Mesh->end())
	while (i< currentMeshNum)
	{
		//_Statistics[0]++;
		//tri = *it++;
		tri = Mesh + i; i++;
		float3 dotInsert = make_float3(dot->x, dot->y, dot->z);
		det[0] = cuda_GetDeterminant(tri->p[0], tri->p[1], dotInsert);
		det[1] = cuda_GetDeterminant(tri->p[1], tri->p[2], dotInsert);
		det[2] = cuda_GetDeterminant(tri->p[2], tri->p[0], dotInsert);

		if (flag)
		{
			float3 v[3];
			for (int j = 0; j < 3; j++)
			{
				//v[i] = make_float3(tri->Vertex[i]->X, tri->Vertex[i]->Y, tri->Vertex[i]->Z);
				v[j] = tri->p[j];
			}
			float3 triNormal = cuda_cross(cuda_Subtraction(v[1], v[0]), cuda_Subtraction(v[2], v[0]));
			float3 otherV = make_float3(dot->x, dot->y, dot->z);
			float3 otherVec = cuda_Subtraction(otherV, v[0]);
			if (cuda_Dot(triNormal, otherVec) > 0)
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
				cuda_SplitTriangle(Mesh, currentMeshNum,tri, dot);
				//printf("%d %d\n", dot->id,currentMeshNum);

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
					//v[j] = make_float3(tri->Vertex[j]->X, tri->Vertex[j]->Y, tri->Vertex[j]->Z);
					v[j] = tri->p[j];
				}
				float3 triNormal = cuda_cross(cuda_Subtraction(v[1], v[0]), cuda_Subtraction(v[2], v[0]));
				float3 otherV = make_float3(dot->x, dot->y, dot->z);
				float3 otherVec = cuda_Subtraction(otherV, v[0]);
				if (cuda_Dot(triNormal, otherVec) == 0)
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
				cuda_SplitTriangle(Mesh, currentMeshNum,tri, dot);
				//printf("%d %d\n", dot->id,currentMeshNum);

				return;
			}
		}
		//if (it == _Mesh->end() && !flag)
		if (i== currentMeshNum && !flag)
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
__device__ void preprocessing(BeamPlugin** beams, int beamNum,short* flag, Triangle* triList)
{
	int cnt = 0;
	for (int i = 0; i < beamNum; i++)
	{
		BeamPlugin* b = beams[i];
		int idx = flag[i] - 1;
		//int rndThetaNum = b->arcNum[idx];
		Triangle* t = triList + b->arcTriList[idx][cnt];
		float projSeg = 0;
		for (int v = 0; v < 3; v++)
		{
			if (t->triWithBeamid[v] == i)
			{
				float3 n_diskNorm = make_float3(-t->diskNorm[v].x, -t->diskNorm[v].y, -t->diskNorm[v].z);
				if (cuda_Dot(n_diskNorm, t->normal) > 0)
				{
					float3 line1 = cuda_Subtraction(t->p[cuda_mod(v + 1, 3)], t->p[v]);
					float3 line2 = cuda_Subtraction(t->p[cuda_mod(v + 2, 3)], t->p[v]);
					float cosTheta = cuda_Dot(n_diskNorm, t->normal);
					float theta = acos(Min(Max(cosTheta, -1.0f), 1.0f));
					float length1 = sin(theta)*cuda_Norm(line1);
					float3 newDiskNorm1 = make_float3(t->diskNorm[v].x*length1, t->diskNorm[v].y*length1, t->diskNorm[v].z*length1);
					float3 sub1 = cuda_Subtraction(line1, newDiskNorm1);
					float3 projL1 = make_float3(sub1.x + t->p[v].x, sub1.y + t->p[v].y, sub1.z + t->p[v].z);

					float length2 = sin(theta)*cuda_Norm(line2);
					float3 newDiskNorm2 = make_float3(t->diskNorm[v].x*length2, t->diskNorm[v].y*length2, t->diskNorm[v].z*length2);
					float3 sub2 = cuda_Subtraction(line2, newDiskNorm2);
					float3 projL2 = make_float3(sub2.x + t->p[v].x, sub2.y + t->p[v].y, sub2.z + t->p[v].z);

					float projLength1, projLength2;
					/*if (isCross(t->o[v], t->p[v], projL1, projLength1))
					{
						projSeg += projLength1;
					}
					if (isCross(t->o[v], t->p[v], projL2, projLength2))
					{
						projSeg += projLength2;
					}*/
					//判断是否本来需要optimal cut
					//float3 vjtoj_1 = cuda_Subtraction(t->p[cuda_mod(j+1,3)],t->p[j]);
					//if (cuda_Dot(line1, n_diskNorm) > 0 || cuda_Dot(line2, n_diskNorm) > 0)
					//{
					//	if (projLength1 > 0 || projLength2 > 0)
					//	{
					//		//inverted = true;
					//		float bf = projLength1 + projLength2;
					//		float r = cuda_Norm(cuda_Subtraction(t->p[v], t->o[v]));
					//		bf = bf / (3 * r);//scale to (0,1)
					//		if (bf >= 1)bf = 0.999;
					//		BF += -(1000 / bf)*log(1 - bf);
					//		BF = 10000;
					//	}
					//}
					if (projSeg > 0)
					{
						int count = 0; 
						float Tsum = 0;
						float Theta[2];
						while (count < 2)
						{
							float T = PI / 180; Tsum += T;
							float3 O = t->o[v];
							float m[16];
							cuda_getRotationMatrix(t->diskNorm[v], T, t->o[v], m);

							float3 rotateV = make_float3(t->p[v].x - O.x, t->p[v].y - O.y, t->p[v].z - O.z);
							float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
							RV = cuda_multiply(m, RV);
							t->p[v] = make_float3(RV.x, RV.y, RV.z);

							projSeg = 0;
							float3 line1 = cuda_Subtraction(t->p[cuda_mod(v + 1, 3)], t->p[v]);
							float3 line2 = cuda_Subtraction(t->p[cuda_mod(v + 2, 3)], t->p[v]);
							float cosTheta = cuda_Dot(n_diskNorm, t->normal);
							float theta = acos(Min(Max(cosTheta, -1.0f), 1.0f));
							float length1 = sin(theta)*cuda_Norm(line1);
							float3 newDiskNorm1 = make_float3(t->diskNorm[v].x*length1, t->diskNorm[v].y*length1, t->diskNorm[v].z*length1);
							float3 sub1 = cuda_Subtraction(line1, newDiskNorm1);
							float3 projL1 = make_float3(sub1.x + t->p[v].x, sub1.y + t->p[v].y, sub1.z + t->p[v].z);

							float length2 = sin(theta)*cuda_Norm(line2);
							float3 newDiskNorm2 = make_float3(t->diskNorm[v].x*length2, t->diskNorm[v].y*length2, t->diskNorm[v].z*length2);
							float3 sub2 = cuda_Subtraction(line2, newDiskNorm2);
							float3 projL2 = make_float3(sub2.x + t->p[v].x, sub2.y + t->p[v].y, sub2.z + t->p[v].z);

							float projLength1, projLength2;
							/*if (isCross(t->o[v], t->p[v], projL1, projLength1))
							{
								projSeg += projLength1;
							}
							if (isCross(t->o[v], t->p[v], projL2, projLength2))
							{
								projSeg += projLength2;
							}*/
							if (projSeg == 0&&count==0)
							{
								Theta[count] = Tsum;
								count++; break;
							}
							if (projSeg > 0 && count == 1)
							{
								Theta[count] = Tsum;
								count++;
							}
						}
						/*float T = (Theta[0] + Theta[1]) / 2 - Theta[0];
						float3 O = t->o[v];
						float m[16];
						cuda_getRotationMatrix(t->diskNorm[v], -T, t->o[v], m);

						float3 rotateV = make_float3(t->p[v].x - O.x, t->p[v].y - O.y, t->p[v].z - O.z);
						float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
						RV = cuda_multiply(m, RV);
						t->p[v] = make_float3(RV.x, RV.y, RV.z);*/
					}
					
				}
				break;
			}
		}

		cnt += b->arcNum[idx];
	}
}
__device__ float cuda_measureShapeCost(BeamPlugin** beams, float3 node, Triangle* triList, int triNum,
	GwsForShape gws, int nVar, short* flag,float& longestLength, float &d1,float &d2)
{
	float max1 = 0, max2 = 0;
	//计算到原始球面的距离
	float disToSphere = 0;
	for (int i = 0; i < nVar; i++)
	{
		BeamPlugin* b = beams[i];
		//int idx = flag[i] - 1;
		/*float offset = gws.offset[i];
		float dis = abs(Norm(Subtraction(b->axis.p[idx], node))+offset-originR);*/
		disToSphere += pow(gws.offset[i],2);

		//int idx = flag[i] - 1;
		//max1 += pow(longestLength - cuda_Norm(cuda_Subtraction(b->axis.p[idx], node)),2);
	}


	float circleDis = 0;
	for (int i = 0; i < triNum; i++)
	{
		Triangle* t = &triList[i];
		BeamPlugin b[3]; BeamPlugin tmpb[3];
		//float3 v[3];
		float3 diskNorm[3];
		for (int j = 0; j < 3; j++)
		{
			b[j] = *beams[t->triWithBeamid[j]]; tmpb[j] = b[j];
			diskNorm[j] = make_float3(-t->diskNorm[j].x, -t->diskNorm[j].y, -t->diskNorm[j].z);
			float offset = gws.offset[t->triWithBeamid[j]];
			int idx = flag[t->triWithBeamid[j]] - 1;
			b[j].axis.p[idx].x = b[j].axis.p[idx].x + diskNorm[j].x*offset;
			b[j].axis.p[idx].y = b[j].axis.p[idx].y + diskNorm[j].y*offset;
			b[j].axis.p[idx].z = b[j].axis.p[idx].z + diskNorm[j].z*offset;
		}
		for (int j = 0; j < 2; j++)
		{
			for (int k = j + 1; k < 3; k++)
			{
				int idx1 = flag[t->triWithBeamid[j]] - 1, idx2 = flag[t->triWithBeamid[k]] - 1;

				float3 direct1 = cuda_make_norm(cuda_Subtraction(b[j].axis.p[flag[idx1] - 1], node));
				float3 direct2 = cuda_make_norm(cuda_Subtraction(b[k].axis.p[flag[idx2] - 1], node));

				float3 cross1 = cuda_cross(direct1, direct2);
				if (cross1.x == 0 && cross1.y == 0 && cross1.z == 0)
				{
					circleDis += pow(cuda_Norm(cuda_Subtraction(b[j].axis.p[idx1], b[k].axis.p[idx2])),2);
					//max2+= cuda_Norm(cuda_Subtraction(tmpb[j].axis.p[idx1], tmpb[k].axis.p[idx2]));
					continue;
				}
				float3 v1 = cuda_make_norm(cuda_cross(cross1, direct1));
				//printf("%f %f %f\n", v1.x, v1.y, v1.z);
				v1 = make_float3(b[j].radius*v1.x, b[j].radius*v1.y, b[j].radius*v1.z);
				v1 = make_float3(b[j].axis.p[idx1].x + v1.x, b[j].axis.p[idx1].y + v1.y, b[j].axis.p[idx1].z + v1.z);
				//float3 tmpv1 = make_float3(tmpb[j].axis.p[idx1].x + v1.x, tmpb[j].axis.p[idx1].y + v1.y, tmpb[j].axis.p[idx1].z + v1.z);

				float3 cross2 = cuda_cross(direct2, direct1);
				float3 v2 = cuda_make_norm(cuda_cross(cross2, direct2));
				//printf("%f %f %f\n", v2.x, v2.y, v2.z);
				v2 = make_float3(b[k].radius*v2.x, b[k].radius*v2.y, b[k].radius*v2.z);
				v2 = make_float3(b[k].axis.p[idx2].x + v2.x, b[k].axis.p[idx2].y + v2.y, b[k].axis.p[idx2].z + v2.z);
				//float3 tmpv2 = make_float3(tmpb[k].axis.p[idx2].x + v2.x, tmpb[k].axis.p[idx2].y + v2.y, tmpb[k].axis.p[idx2].z + v2.z);

				float3 OtoV1 = cuda_make_norm(cuda_Subtraction(v1, b[j].axis.p[idx1]));
				float3 V1toV2 = cuda_make_norm(cuda_Subtraction(v2, v1));
				float3 OtoV2 = cuda_make_norm(cuda_Subtraction(v2, b[k].axis.p[idx2]));
				float3 V2toV1 = cuda_make_norm(cuda_Subtraction(v1, v2));

				//circleAngle += pow(Dot(OtoV1, V1toV2) - 1, 2) + pow(Dot(OtoV2, V2toV1) - 1, 2);
				if (cuda_Dot(t->diskNorm[j], V1toV2) < 0 || cuda_Dot(t->diskNorm[k], V2toV1) < 0)
					circleDis += pow(cuda_Norm(cuda_Subtraction(v1, v2)),2);

				/*float3 tmpV1toV2 = cuda_make_norm(cuda_Subtraction(tmpv2, tmpv1));
				float3 tmpV2toV1 = cuda_make_norm(cuda_Subtraction(tmpv1, tmpv2));
				if (cuda_Dot(t->diskNorm[j], tmpV1toV2) < 0 || cuda_Dot(t->diskNorm[k], tmpV2toV1) < 0)
				{
					max2 += pow(cuda_Norm(cuda_Subtraction(tmpv1, tmpv2)), 2);
				}*/
				//if (j == k)continue;
				//double r1 = b[j].radius, r2 = b[k].radius;

				//double cosVal = Dot(diskNorm[j], diskNorm[k]) / (Norm(diskNorm[j])*Norm(diskNorm[k]));
				//double angle = acos(cosVal);

				//double cutj = Norm(Subtraction(b[j].axis.p[flag[t->triWithBeamid[j]] - 1], node));
				//double cutk = Norm(Subtraction(b[k].axis.p[flag[t->triWithBeamid[k]] - 1], node));
				//double length1 = sqrt(pow(r1, 2) + pow(cutj, 2));
				//double itoj = angle - atan(r1 / cutj);
				//double cut1 = length1 * cos(itoj);
				//if (cut1 > cutk)
				//{
				//	needCut += cut1 - cutk;
				//	cutk = cut1;
				//	/*intersectionLength[j] = cut1;
				//	stk.push(j);
				//	stk2.push(j);
				//	break;*/

				//}

				//double length2 = sqrt(pow(r2, 2) + pow(cutk, 2));
				//double jtoi = angle - atan(r2 / cutk);
				//double cut2 = length2 * cos(jtoi);
				//if (cut2 > cutj)
				//{
				//	needCut += cut2 - cutj;
				//	cutj = cut2;
				//	/*intersectionLength[i] = cut2;
				//	stk.push(i);
				//	stk2.push(j);
				//	break;*/
				//}
			}
		}
	}
	//atomicAdd(&(d1), 1*disToSphere);
	//atomicAdd(&(d2), 18*circleDis);
	//printf("%f %f\n", (max2 / max1)*disToSphere, circleDis);
	return  disToSphere+4*circleDis;
}
__device__ void cuda_shapeComputeX123(float* leaderPos, GwsForShape gws, int nVar, float a,float* X)
{
	float c[40];
	float D[40];
	float A[40];
	//% Eq.(3.4) in the paper
	//float* c = (float*)malloc(nVar * sizeof(float));

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
	//float* D = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		D[i] = (abs(c[i] * leaderPos[i] - gws.offset[i]));
	}
	//float* A = (float*)malloc(nVar * sizeof(float));

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
	//float* X = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		X[i] = (leaderPos[i] - A[i] * abs(D[i]));
		//printf("%f ", X[i]);
	}
	/*free(c);
	free(D);
	free(A);

	return X;*/
}
__device__ void cuda_shapeOptimization(BeamPlugin** beams,int nVar, float3 node, Triangle* triList,
	int triNum, short* flag, float& longestLength,int lane_id, NodeStatistic* statis,int nodeid)
{
	//int nVar = beams.size();

	int GreyWolves_num = 96;
	float MaxIt = 100;

	__shared__ float Alpha_pos[40];// = (float*)malloc(nVar * sizeof(float));// new float[nVar];
	//memset(Alpha_pos, 0, nVar * sizeof(float));
	__shared__ float Alpha_score;
	Alpha_score = INF;

	__shared__ float Beta_pos[40];// = (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	//memset(Beta_pos, 0, nVar * sizeof(float));
	__shared__ float Beta_score;
	Beta_score = INF;

	__shared__ float Delta_pos[40];// = (float*)malloc(nVar * sizeof(float)); //new float[nVar];
	//memset(Delta_pos, 0, nVar * sizeof(float));
	__shared__ float Delta_score;
	Delta_score = INF;

	for (int i = 0; i < nVar; i++)
	{
		Alpha_pos[i] = 0;
		Beta_pos[i] = 0;
		Delta_pos[i] = 0;
	}

	__shared__ GwsForShape gws[96];// = (GwsForShape*)malloc(GreyWolves_num * sizeof(GwsForShape));
	__shared__ float fitness[96];
	//原始球半径计算
	float originR = 0;
	//for (int i = 0; i < GreyWolves_num; i++)
	{
		for (int j = 0; j < nVar; j++)
		{
			curandState devStates;
			curand_init((unsigned long long)clock(), j, 0, &devStates);// initialize the state
			float RANDOM = curand_uniform(&devStates);// uniform distribution
			//if (longestLength[j] - Norm(Subtraction(beams[j]->axis.p[flag[j] - 1], node)) < 0)printf("%f\n", longestLength[j] - Norm(Subtraction(beams[j]->axis.p[flag[j] - 1], node)));
			RANDOM *= longestLength - cuda_Norm(cuda_Subtraction(beams[j]->axis.p[flag[j] - 1], node));
			gws[lane_id].offset[j] = RANDOM;
			/*if (beams[j]->radius > originR)
				originR = beams[j]->radius;*/
		}
	}
	float record[100];// = (float*)malloc(MaxIt * sizeof(float));
	int total_iter = 0; bool converge = false; float minCut = 1000000; float maxCut=0,AvgR = 0;
	__shared__ float d1, d2; d1 = 0; d2 = 0;
	//main loop
	for (float it = 0; it < MaxIt; it++)
	{
		float a = 2 - it * ((2) / MaxIt);
		//for (int i = 0; i < GreyWolves_num; i++)
		//计算cost
		gws[lane_id].singleCost = cuda_measureShapeCost(beams, node, triList, triNum, gws[lane_id], nVar, flag,longestLength,d1,d2);
		fitness[lane_id] = gws[lane_id].singleCost;
		__syncthreads();

		if (lane_id == 0)
		{
			/*statis[nodeid].disToSphere += d1 / 128; d1 = 0;
			statis[nodeid].circleDis += d2 / 128; d2 = 0;*/
			//Update Alpha, Beta, and Delta
			for (int i = 0; i < 96; i++)
			{
				if (fitness[i] < Alpha_score)
				{
					Alpha_score = fitness[i];
					memcpy(Alpha_pos, gws[i].offset, nVar * sizeof(float));
				}
				if (fitness[i] > Alpha_score && fitness[i] < Beta_score)
				{
					Beta_score = fitness[i];
					memcpy(Beta_pos, gws[i].offset, nVar * sizeof(float));
				}
				if (fitness[i] > Alpha_score && fitness[i] > Beta_score && fitness[i] < Delta_score)
				{
					Delta_score = fitness[i];
					memcpy(Delta_pos, gws[i].offset, nVar * sizeof(float));
				}
			}
			
		}
		
			
		
		//for (int i = 0; i < GreyWolves_num; i++)
		{
			int It = 0;
			float X1[40];
			float X2[40];
			float X3[40];
			cuda_shapeComputeX123(Alpha_pos, gws[lane_id], nVar, a,X1);
			cuda_shapeComputeX123(Beta_pos, gws[lane_id], nVar, a,X2);
			cuda_shapeComputeX123(Delta_pos, gws[lane_id], nVar, a,X3);

			for (int k = 0; k < nVar; k++)
			{
				gws[lane_id].offset[k] = (X1[k] + X2[k] + X3[k]) / 3;
				gws[lane_id].offset[k] = Min(Max((double)(gws[lane_id].offset[k]), 0.0), (double)(longestLength - cuda_Norm(cuda_Subtraction(beams[k]->axis.p[flag[k] - 1], node))));
				//printf("%f ", gws[i].rotationAngle[k]);
			}
			//printf("\n");
			//free(X1); free(X2); free(X3);
		}
		//if(lane_id==0)
		    //printf("%f\n", Alpha_score);
		record[total_iter] = Alpha_score;
		if (total_iter >= 8 && lane_id == 0)
		{
			if (abs(record[total_iter] - record[total_iter - 8]) < 1&&converge==false)
			{
				//printf("degree: %d phase 1 iternum: %d nodeid: %d\n", nVar, total_iter, nodeid);
				converge = true;
			}
		}
		total_iter++;
		if (it == MaxIt - 1)
		{
			if (abs(record[(int)MaxIt - 1] - record[(int)MaxIt - 2]) < 1)
			{
				if (lane_id == 0)
				{
					/*statis[nodeid].disToSphere /= MaxIt;
					statis[nodeid].circleDis /= MaxIt;*/
					for (int j = 0; j < nVar; j++)
					{
						BeamPlugin* b = beams[j];
						float offset = Alpha_pos[j]; //printf("%f ", offset);
						float3 direction;
						if (flag[j] == 1)
						{
							direction = cuda_make_norm(cuda_Subtraction(b->axis.p[1], b->axis.p[0]));
							b->axis.p[0].x += direction.x*offset;
							b->axis.p[0].y += direction.y*offset;
							b->axis.p[0].z += direction.z*offset;
						}
						else
						{
							direction = cuda_make_norm(cuda_Subtraction(b->axis.p[0], b->axis.p[1]));
							b->axis.p[1].x += direction.x*offset;
							b->axis.p[1].y += direction.y*offset;
							b->axis.p[1].z += direction.z*offset;
						}
						atomicAdd(&(b->length), -offset);
						//b->length -= offset;
						float cut = cuda_Norm(cuda_Subtraction(beams[j]->axis.p[flag[j] - 1], node));
						AvgR += cut;
						if (cut < minCut)minCut = cut;
						if (cut > maxCut)maxCut = cut;
					}
					AvgR /= nVar;
					//printf("degree: %d originR: %f minR: %f %f maxR: %f %f AvgR: %f %f\n", nVar,originR, minCut, minCut / originR, maxCut, maxCut / originR, AvgR, AvgR / originR);
					//printf("\n");
					//break;
				}
				
			}
			else
			{

				/*triList.clear();
				quadList.clear();*/
				it = -1;
				MaxIt = 100;
				//if(record.size()==MaxIt)
				/*free(record);
				record = (float*)malloc(MaxIt * sizeof(float));*/
				total_iter = 0;
				//record.resize(MaxIt);
				//break;

			}
		}
	}
	//free(gws);
	//free(record);
	//free(Alpha_pos);
	////free(Alpha.rotationAngle);
	//free(Beta_pos);
	////free(Beta.rotationAngle);
	//free(Delta_pos);
}
__device__ void cuda_getTopology(BeamPlugin** beams, int beamNum, float3 position,
	Triangle* Mesh,QuadFace* quads, Point* dots,
	int nodeId,int lane_id,short* flag,float& longestLength, NodeStatistic* statis)
{
	//ch_vertex* vertices = (ch_vertex*)malloc(beamNum * sizeof(ch_vertex));
	if (lane_id == 0)
	{
		float projRadius = 1;
		//Point** dots = (Point**)malloc(beamNum * sizeof(Point*));
		float3 *points = (float3*)malloc(beamNum * sizeof(float3));
		bool* visitedDot = (bool*)malloc(beamNum * sizeof(bool));
		for (int i = 0; i < beamNum; i++)
		{
			visitedDot[i] = false;
			BeamPlugin* b = beams[i];
			float3 st, ed;
			if (pow(position.x - b->axis.p[0].x, 2) + pow(position.y - b->axis.p[0].y, 2) + pow(position.z - b->axis.p[0].z, 2)
				< pow(position.x - b->axis.p[1].x, 2) + pow(position.y - b->axis.p[1].y, 2) + pow(position.z - b->axis.p[1].z, 2))
			{
				st = b->axis.p[0]; ed = b->axis.p[1]; flag[i] = 1;
			}
			else
			{
				st = b->axis.p[1]; ed = b->axis.p[0]; flag[i] = 2;
			}
			points[i] = st;
			//points.push_back(Point_3(st.x, st.y, st.z));
			//float3 proj = make_float3(st.x - position.x, st.y - position.y, st.z - position.z);
			//projRadius += Norm(proj);
			Point* dot = dots + i;//(Point*)malloc(sizeof(Point));
			dot->x = st.x - position.x; dot->y = st.y - position.y; dot->z = st.z - position.z;

			float length = sqrt(pow(dot->x, 2) + pow(dot->y, 2) + pow(dot->z, 2));
			float scaleFactor = projRadius / length;

			dot->x = scaleFactor * dot->x;
			dot->y = scaleFactor * dot->y;
			dot->z = scaleFactor * dot->z;

			dot->id = i;
			dot->isVisited = false;
			//dots[i] = dot;

			/*vertices[i].x = st.x;
			vertices[i].y = st.y;
			vertices[i].z = st.z;*/
		}
		//projRadius = projRadius / beamNum; //projRadius = 1;

		/*int* faceIndices = NULL;
		int nFaces;
		convhull_3d_build(vertices, beamNum, &faceIndices, &nFaces);*/
		int nFaces = 8 + (beamNum - 6) * 2;
		//Triangle** Mesh = (Triangle**)malloc(nFaces * sizeof(Triangle*));
		cuda_BuildInitialHull(dots, beamNum, Mesh);

		int currentMeshNum = 4;
		//printf("%d %d\n", nFaces, meshNum);
		for (int i = 0; i < beamNum; i++)
		{
			Point* dot = dots + i;
			if (!dot->isVisited)
			{
				cuda_InsertDot(dot, Mesh, currentMeshNum);
				currentMeshNum += 2;
			}
			//free(dot);
		}
		//printf("%d %d\n", nFaces, currentMeshNum);
		/*for (int i = 0; i < beamNum; i++)
		{
			free(dots[i]);
		}*/
		//free(dots);
		for (int i = 0; i < nFaces; i++)
		{
			(Mesh+i)->triId = i;
		}
		//建立每个dot上的三角形id链
		for (int i = 0; i < nFaces; i++)
		{
			int startid = (Mesh + i)->triId;

			for (int j = 0; j < 3; j++)
			{
				int beamid = (Mesh + i)->triWithBeamid[j];
				if (visitedDot[beamid])
					continue;
				if (flag[beamid] == 1)
				{
					beams[beamid]->arcTriList[0][beams[beamid]->arcNum[0]] = (Mesh + i)->triId;
					beams[beamid]->arcNum[0]++;
				}
				else
				{
					beams[beamid]->arcTriList[1][beams[beamid]->arcNum[1]] = (Mesh + i)->triId;
					beams[beamid]->arcNum[1]++;
				}
				int nextid = (Mesh + i)->adjTriangles[cuda_mod(j + 2, 3)]->triId;
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
						if ((Mesh+nextid)->triWithBeamid[v] == beamid)
						{
							nextid = (Mesh+nextid)->adjTriangles[cuda_mod(v + 2, 3)]->triId;
							break;
						}
					}
				}
				visitedDot[beamid] = true;
			}

		}
		Triangle* triangles = Mesh;//(Triangle*)malloc(nFaces * sizeof(Triangle)); 
		int triId = 0;
		//QuadFace* quadList;
		//std::map<int, Eigen::Vector3d> mp;
		for (int i = 0; i < nFaces; i++)
		{
			//Eigen::Vector3i triWithId;
			float3 v[3];
			BeamPlugin *b[3]; int cnt = 0;
			Triangle* mesh = Mesh + i;

			for (int j = 0; j < 3; j++)
			{
				//int idx = faceIndices[i * 3 + j];
				int idx = mesh->triWithBeamid[j];
				b[cnt] = beams[idx];
				//triWithId[cnt] = b[cnt]->beamId;
				v[cnt] = points[idx];
				//v[cnt].x = mesh->p[j].x;//vertices[idx].x;
				//v[cnt].y = mesh->p[j].y;//vertices[idx].y;
				//v[cnt].z = mesh->p[j].z;//vertices[idx].z;
				//mp[triWithId[cnt]] = v[cnt];

				cnt++;
			}
			triangles[i].o[0] = v[0];
			triangles[i].o[1] = v[1];
			triangles[i].o[2] = v[2];

			cuda_getTriangle(b, position, triangles, triId);
			triId++;
			//free(mesh);
		}
		free(points);
		//preprocessing(beams, beamNum, flag, triangles);
		/*int triNum = 8 + (beamNum - 6) * 2;;
		cuda_shapeOptimization(beams, beamNum, position, Mesh, triNum, flag, longestLength);*/
	}
	
	__syncthreads();
	/*for (int i = 0; i < nFaces; i++)
	{
		free(Mesh[i]);
	}*/
	//free(Mesh);
	int triNum = 8 + (beamNum - 6) * 2;;
	int quadNum = 6 + 3 * ((triNum - 4) / 2);
	
	//quadList = (QuadFace*)malloc(quadNum * sizeof(QuadFace));
	//cuda_GreyWolfOptimizer(triNum * 3, Mesh,triList,quads, quadList, triNum, quadNum,c,D,A,nodeId, lane_id);
	cuda_shapeOptimization(beams, beamNum, position, Mesh, triNum, flag, longestLength,lane_id,statis,nodeId);
	__syncthreads();
	if (lane_id == 0)
	{
		int triId = 0;
		//QuadFace* quadList;
		//std::map<int, Eigen::Vector3d> mp;
		//float* faceAngle = (float*)malloc(3 * triNum * sizeof(float));
		//float faceAngle[300]; int faCnt = 0;
		for (int i = 0; i < triNum; i++)
		{
			//Eigen::Vector3i triWithId;
			//float3 v[3];
			BeamPlugin *b[3]; int cnt = 0;
			Triangle* mesh = Mesh + i;
			float3 direct[3];
			for (int j = 0; j < 3; j++)
			{
				//int idx = faceIndices[i * 3 + j];
				int idx = mesh->triWithBeamid[j];
				b[cnt] = beams[idx];
				//triWithId[cnt] = b[cnt]->beamId;
				//v[cnt] = points[idx];
				//v[cnt].x = mesh->p[j].x;//vertices[idx].x;
				//v[cnt].y = mesh->p[j].y;//vertices[idx].y;
				//v[cnt].z = mesh->p[j].z;//vertices[idx].z;
				//mp[triWithId[cnt]] = v[cnt];
				Mesh[i].o[cnt] = b[cnt]->axis.p[flag[idx] - 1];
				

				direct[j] = cuda_Subtraction(b[cnt]->axis.p[cuda_mod(flag[idx], 2)], b[cnt]->axis.p[flag[idx] - 1]);
				direct[j] = cuda_make_norm(direct[j]);
				cnt++;
			}
			/*for (int j = 0; j < 3; j++)
			{
				float fa = acos(Min(Max(cuda_Dot(direct[j], direct[cuda_mod(j + 1, 3)]),-1.0f),1.0f));
				statis[nodeId].face_angle_variance += fa;
				faceAngle[faCnt++] = fa;
			}*/

			cuda_getTriangle(b, position, Mesh, triId);
			triId++;
			//free(mesh);
		}
		/*statis[nodeId].face_angle_variance /= 3 * triNum;
		float v = 0;
		for (int i = 0; i < 3 * triNum; i++)
		{
			v += pow(faceAngle[i] - statis[nodeId].face_angle_variance, 2);

		}
		statis[nodeId].face_angle_variance = v / (3 * triNum);*/
		//free(faceAngle);
	}
	__syncthreads();
	cuda_GWO(beams, beamNum, flag, triNum * 3, Mesh, quads,  triNum, quadNum,  nodeId, lane_id,statis,position);
	//cuda_GreyWolfOptimizer(beams,beamNum,flag,position,triNum * 3, Mesh, quads, triNum, quadNum,nodeId, lane_id, statis);
	/*if(Mesh[0].inverted==3)
	cuda_GreyWolfOptimizer(triNum * 3, Mesh, triList, quads, quadList, triNum, quadNum, nodeId, lane_id);*/
	//MultiObjectiveGreyWolfOptimizer(triList.size()*3,2,triList,quadList);
	//cuda_generateFaces(triList, triId, quadList, quadNum, cuda_f,fid);

	//free(vertices);
	/*for (int i = 0; i < nFaces; i++)
	{
		free(triList[i].adjTriangles);
		free(triList[i].adjQuadFaces);
	}*/
	
	/*free(triList);
	free(quadList);*/
}

extern "C" void call_NodeComputing(int plines, BeamPlugin** G, int* beamNumVec, float3* positions,
	Triangle* AllMesh,QuadFace* AllQuad,Point* Alldots, 
	short* AllFlag,NodeStatistic* statis,int batchIdx,int batchSize,int batchNum)
{
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	NodeComputing << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (plines,G,beamNumVec, positions, 
		AllMesh,AllQuad,Alldots,AllFlag,statis,batchIdx,batchSize,batchNum);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error: %s\n", cudaGetErrorString(err));
		// Possibly: exit(-1) if program cannot continue....
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%f\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//printf("%d\n", convergeNum[0]);
}
//__global__ void computeLength(int plines, BeamPlugin** G, int* beamNumVec, float3* positions, 
//	BeamPlugin** beams, int size, float3 startP)
//{
//	int index = threadIdx.x + blockIdx.x*blockDim.x;
//	while (index < plines)
//	{
//		int i = index + 1;
//		int st = 0;
//		for (int j = 1; j < i; j++)
//		{
//			st += beamNumVec[j];
//		}
//
//		BeamPlugin** beams = G + st;
//		int size = beamNumVec[i];
//		float3 startP = positions[i];
//		if (size > 1)
//		{
//			cuda_computeIntersectionLength(beams, size, startP);
//		}
//		index += blockDim.x * gridDim.x;
//
//	}
//}
__global__ void NodeComputing(int plines,BeamPlugin** G, int* beamNumVec, float3* positions, 
	Triangle* AllMesh,  QuadFace* AllQuad,  Point* Alldots, 
	 short* AllFlag, NodeStatistic* statis, int batchIdx, int batchSize, int batchNum)
{
	int thread_id = threadIdx.x + blockIdx.x*blockDim.x;
	int warp_id = thread_id / 96;
	int lane_id = thread_id % 96;

	int index = warp_id+batchIdx*batchSize+1;
	int upper;
	if (batchIdx == batchNum - 1)
		upper = plines;
	else
		upper = (batchIdx + 1) * batchSize;
	
	while (index <= upper)
	{
		int i = index;
		/*if(lane_id==0)
		printf("Node: %d\n", i);*/
		//if (i == 7)
		{


			int st = 0;
			int meshSt = 0;
			int quadSt = 0;
			int cDAXSt = 0;
			//int arcSt = 0;
			int flagSt = 0;
			for (int j = 1; j < i; j++)
			{
				st += beamNumVec[j];
				int meshNumincre = 8 + (beamNumVec[j] - 6) * 2;
				if (beamNumVec[j] < 4)meshNumincre = 0;
				meshSt += meshNumincre;
				int quadNumincre = 6 + 3 * ((meshNumincre - 4) / 2);
				quadSt += quadNumincre;
				cDAXSt += 3 * meshNumincre;
				//arcSt += 2 * quadNumincre;
				if (beamNumVec[j] >= 4)flagSt += beamNumVec[j];
			}
			BeamPlugin** beams = G + st;
			Triangle* Mesh = AllMesh + meshSt;
			//Triangle* triList = AllTriList + meshSt;
			QuadFace* quads = AllQuad + quadSt;
			//QuadFace* quadList = AllQuadList + quadSt;
			Point* dots = Alldots + st;
			/*float* c = Allc + cDAXSt;
			float* D = AllD + cDAXSt;
			float* A = AllA + cDAXSt;
			float* X1 = AllX1 + cDAXSt;
			float* X2 = AllX2 + cDAXSt;
			float* X3 = AllX3 + cDAXSt;*/
			//ArcType* arcList = AllArc + arcSt;
			short* flag = AllFlag + flagSt;
			//float* longestLength = AllLength + flagSt;

			int size = beamNumVec[i];
			//int meshNum = meshNumVec[i];
			float3 startP = positions[i];
			//Face* cf = cuda_f[i];
			//Face* hf = host_f[i];
			if (size > 1)
			{
				__shared__ float longestLength;
				longestLength = 0;
				if (lane_id == 0)
				{
					statis[i].face_angle = PI;
					statis[i].face_angle_variance = 0;
					cuda_convexCuttingLength(beams, size, startP, i);
					cuda_computeIntersectionLength(beams, size, startP, longestLength, i);
					statis[i].degree = beamNumVec[i];
					statis[i].converge = false;
					statis[i].disToSphere = 0;
					statis[i].circleDis = 0;
				}
				cuda_getTopology(beams, size, startP, Mesh, quads, dots, i, lane_id, flag, longestLength, statis);


				//printf("%f %f %f\n", cuda_f[1][5].normal[0], cuda_f[1][5].normal[1], cuda_f[1][5].normal[2]);
			}
			/*if (lane_id == 0)
			printf("The end of Node: %d\n", i);*/
		}
		thread_id += blockDim.x * gridDim.x;
		warp_id = thread_id / 96;
		lane_id = thread_id % 96;

		index = warp_id + batchIdx * batchSize + 1;
		//index = warp_id;
	}
	return;
}

extern __global__ void locateArcs_setSegNum(int plines, BeamPlugin** G, int* beamNumVec, Triangle* AllTriangle,
	QuadFace* AllQuad, ArcType* AllArc, short* AllFlag, int* totalSampleNum, int* totalArcSampleNum,float ce);

extern "C" void call_locateArcs_setSegNum(int plines, BeamPlugin** G, int* beamNumVec, Triangle* AllTriangle, 
	QuadFace* AllQuad, ArcType* AllArc, short* AllFlag, int* totalSampleNum, int* totalArcSampleNum,float ce)
{
	locateArcs_setSegNum << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (plines,G,beamNumVec,AllTriangle,AllQuad,AllArc,AllFlag,
		totalSampleNum,totalArcSampleNum,ce);
}

__device__ void cuda_locateArcs(BeamPlugin** beams,int beamNum, Triangle* triList, QuadFace* quadList, ArcType* arcList, short* flag)
{
	int triNum = 8 + (beamNum - 6) * 2;
	for (int i = 0; i < triNum; i++)
	{
		triList[i].adjTriNum = 0;
		triList[i].adjQuadNum = 0;
	}
	cuda_findAdjTriangles(triList,triNum); bool tmpinverted = false;
	cuda_findQuadFaces(triList, quadList, triNum, tmpinverted);
	int arcId = 0;
	bool f = false;
	for (int i = 0; i < beamNum; i++)
	{
		BeamPlugin* b = beams[i];
		int idx = flag[i] - 1;
		for (int j = 0; j < b->arcNum[idx]; j++)
		{
			Triangle* tri = triList + b->arcTriList[idx][j];
			Triangle* tri2 = triList + b->arcTriList[idx][cuda_mod(j + 1, b->arcNum[idx])];
			//找到两个相邻三角形中间的quadFace
			for (int k = 0; k < 3; k++)
			{
				for (int v = 0; v < 3; v++)
				{
					if (tri->adjQuadFaces[k]->quadId == tri2->adjQuadFaces[v]->quadId)
					{
						f = true;
						QuadFace* q = quadList + tri->adjQuadFaces[k]->quadId;
						ArcType* arc = &arcList[arcId++];
						if (q->quadWithBeamid[0] == i)
						{
							arc->st = q->t[1].p[0];
							arc->ed = q->t[1].p[2];
							q->arc[0] = arc;
							q->F[0] = 1;
						}
						else
						{
							arc->st = q->t[0].p[2];
							arc->ed = q->t[0].p[1];
							q->arc[1] = arc;
							q->F[1] = 1;
						}
						arc->o = b->axis.p[idx];
						float3 direct = cuda_Subtraction(b->axis.p[idx], b->axis.p[cuda_mod(idx + 1, 2)]);
						float3 e1 = cuda_Subtraction(arc->st, arc->o);
						e1 = cuda_make_norm(e1);
						float3 e2 = cuda_Subtraction(arc->ed, arc->o);
						e2 = cuda_make_norm(e2);
						float3 arcNorm = cuda_cross(e1, e2);
						arcNorm = cuda_make_norm(arcNorm);
						if (cuda_Dot(arcNorm, direct) > 0)
						{
							arc->theta = 2 * PI - acos(Min(Max(cuda_Dot(e1, e2), -1.0f), 1.0f));
							arc->diskNorm = make_float3(-arcNorm.x, -arcNorm.y, -arcNorm.z);
						}
						else
						{
							arc->theta = acos(Min(Max(cuda_Dot(e1, e2), -1.0f), 1.0f));
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
		}

	}
}

__device__ void cuda_setSegNum(BeamPlugin** beams,int beamNum, float chordError, short* flag, int* totalSampleNum, int* totalArcSampleNum)
{
	for (int i = 0; i < beamNum; i++)
	{
		BeamPlugin* b = beams[i];
		float r = b->radius;
		float segTheta = 2 * acos(Min(Max(1 - chordError / r,-1.0f),1.0f)); 
		
		short idx = flag[i] - 1;
		for (int j = 0; j < b->arcNum[idx]; j++)
		{
			ArcType* arc = b->arcArray[idx][j];
			if (abs(segTheta - 0.0f) < 1e-6)
			{
				segTheta = 0.01;
			}
			arc->sampleNum = ceil(arc->theta / segTheta) + 1;
			//printf("%d\n", arc->sampleNum);
			if (arc->sampleNum == 1)
			{
				arc->segTheta = 0;
				arc->sampleNum = 2; printf("correcting sample num!\n");
			}
			else
				arc->segTheta = arc->theta / (arc->sampleNum - 1);
			
			/*totalSampleNum += (arc->sampleNum - 1);
			totalArcSampleNum += arc->sampleNum;*/
			atomicAdd(&(totalSampleNum[0]), (arc->sampleNum - 1));
			atomicAdd(&(totalArcSampleNum[0]), arc->sampleNum);
		}
		//totalSampleNum++;
		atomicAdd(&(totalSampleNum[0]), 1);
	}
}

__global__ void locateArcs_setSegNum(int plines, BeamPlugin** G, int* beamNumVec, Triangle* AllTriangle,
	QuadFace* AllQuad, ArcType* AllArc, short* AllFlag, int* totalSampleNum, int* totalArcSampleNum,float ce)
{
	int thread_id = threadIdx.x + blockIdx.x*blockDim.x;

	while (thread_id < plines)
	{
		int i = thread_id + 1;

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

		BeamPlugin** beams = G + st;
		int size = beamNumVec[i];
		if (size > 1)
		{
			cuda_locateArcs(beams,size, AllTriangle + meshSt, AllQuad + quadSt, AllArc + arcSt, AllFlag + flagSt);
			cuda_setSegNum(beams,size, ce, AllFlag + flagSt, totalSampleNum, totalArcSampleNum);
		}

		thread_id += blockDim.x * gridDim.x;
	}
	return;
}
__device__ bool cuda_pickPos(int* pos, int n, int m)
{
	int cnt = 0;
	int max_value;
	int min_value;
	int Pos = 0;
	for (int i = 0; i < m - 1; i++)
	{
		curandState devStates;
		curand_init((unsigned long long)clock(), i, 0, &devStates);// initialize the state
		double RANDOM = curand_uniform(&devStates);// uniform distribution

		max_value = Min(n - (m - 1 - i), n - 1);
		min_value = Max(i, Pos);
		if (max_value == min_value)
		{
			Pos = max_value;
			pos[i] = Pos;
			continue;
		}
		RANDOM *= (max_value - min_value + 0.999999);
		RANDOM += min_value;
		for (int t = 0; t < n; t++)
		{
			if (abs(t - RANDOM) < 1)
			{
				Pos = t;
				cnt++;
				break;
			}
			
		}
		//Pos = __double2int_rd(RANDOM);
		pos[i] = Pos;
	}
	if (cnt == m - 1)
		return true;
	else
		return false;
	
}
__device__ float cuda_measureVariance(int n, int m, Triangle* resultTri, int triNum, int* pos)
{
	
	float variance = 0, v0 = 0, v1 = 0;
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
		while (pos_idx + 1 < m - 1)
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
		remain_n--;
	}
	for (int i = 0; i < remain_n; i++)
	{
		v1 += pow(numfor_n, 2);
	}
	v0 = v0 / m;
	v1 = v1 / n;
	variance = v0 + v1;

	float cost = 0;
	for (int i = 0; i < triNum; i++)
	{
		float3 e1 = cuda_Subtraction(resultTri[i].p[1], resultTri[i].p[0]);
		float3 e2 = cuda_Subtraction(resultTri[i].p[2], resultTri[i].p[0]);
		e1 = cuda_make_norm(e1);
		e2 = cuda_make_norm(e2);
		float theta1 = acos(Min(Max(cuda_Dot(e1, e2), -1.0f), 1.0f));
		e1 = cuda_Subtraction(resultTri[i].p[2], resultTri[i].p[1]);
		e2 = cuda_Subtraction(resultTri[i].p[2], resultTri[i].p[0]);
		e1 = cuda_make_norm(e1);
		e2 = cuda_make_norm(e2);
		float theta2 = acos(Min(Max(cuda_Dot(e1, e2), -1.0f), 1.0f));
		cost += pow(theta1 - theta2, 2);
	}
	return variance;
}
__device__ void cuda_computeX123(int* leaderPos, GwsForTriangulation &gws, int nVar, double a, double* c, double* D, double* A, double* X)
{

	//% Eq.(3.4) in the paper
	//float* c = (float*)malloc(nVar * sizeof(float));

	/*if (it == 0)
	{
		gws.chaoticC.clear();
		gws.chaoticA.clear();
	}*/
	//srand((unsigned)time(NULL));

	for (int i = 0; i < nVar; i++)
	{
		double rndC;
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
			double RANDOM = curand_uniform(&devStates);// uniform distribution
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
	//float* D = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		D[i] = (abs(c[i] * leaderPos[i] - gws.separatePos[i]));
	}
	//float* A = (float*)malloc(nVar * sizeof(float));

	for (int i = 0; i < nVar; i++)
	{
		double rndA;
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
			double RANDOM = curand_uniform(&devStates);// uniform distribution
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
	//float* X = (float*)malloc(nVar * sizeof(float));
	for (int i = 0; i < nVar; i++)
	{
		X[i] = (leaderPos[i] - A[i] * abs(D[i]));
		//printf("%f ", X[i]);
	}
	/*free(c);
	free(D);
	free(A);

	return X;*/
}
__device__ void cuda_GWOforQuadTriangulation(int n, int m, float3* sample_n, float3* sample_m, Triangle* resultTri, int triNum, bool isCircle)
{
	if (n > 1000 || m > 1000)printf("n is wrong!\n");
	if (m == 2 && n == 2 && cuda_Norm(cuda_Subtraction(sample_m[0], sample_m[1])) < 1e-6&&cuda_Norm(cuda_Subtraction(sample_n[0], sample_n[1])) < 1e-6)
	{
		printf("All coincident!\n");
		for (int i = 0; i < triNum; i++)
		{
			Triangle* tri = &resultTri[i];
			tri->flag = 1;
		}
		return;
	}
	if (m == 2 && cuda_Norm(cuda_Subtraction(sample_m[0], sample_m[1])) < 1e-5)
	{
		printf("two m coincident!\n");
		int i = 0;
		for (; i < n - 1; i++)//n-1 triangles
		{
			Triangle* tri = &resultTri[i];
			tri->flag = 0;
			tri->p[0] = sample_n[i];
			tri->p[1] = sample_m[1];
			tri->p[2] = sample_n[i + 1];
			tri->normal = cuda_cross(cuda_Subtraction(tri->p[1], tri->p[0]), cuda_Subtraction(tri->p[2], tri->p[0]));
		}
		for (; i < triNum; i++)
		{
			Triangle* tri = &resultTri[i];
			tri->flag = 1;
		}
		return;
	}
	if (n == 2 && cuda_Norm(cuda_Subtraction(sample_n[0], sample_n[1])) < 1e-5)
	{
		printf("two n coincident!\n");
		int i = 0;
		for (; i < m - 1; i++)//m-1 triangles
		{
			Triangle* tri = &resultTri[i];
			tri->flag = 0;
			tri->p[0] = sample_m[i];
			tri->p[1] = sample_n[1];
			tri->p[2] = sample_m[i + 1];
			tri->normal = cuda_cross(cuda_Subtraction(tri->p[1], tri->p[0]), cuda_Subtraction(tri->p[2], tri->p[0]));
		}
		for (; i < triNum; i++)
		{
			Triangle* tri = &resultTri[i];
			tri->flag = 1;
		}
		return;
	}
	/*if (!isCircle)
		printf("%d %d\n", n, m);*/
	int GreyWolves_num = 100;
	double MaxIt = 100;

	//int* Alpha_pos = (int*)malloc((m - 1) * sizeof(int));
	int Alpha_pos[180];
	if(cuda_pickPos(Alpha_pos, n, m)==false)printf("float wrong!\n");
	
	float Alpha_score = INF;

	//int* Beta_pos = (int*)malloc((m - 1) * sizeof(int));
	int Beta_pos[180];
	if(cuda_pickPos(Beta_pos, n, m)==false)printf("float wrong!\n");
	float Beta_score = INF;

	//int* Delta_pos = (int*)malloc((m - 1) * sizeof(int));
	int Delta_pos[180];
	if(cuda_pickPos(Delta_pos, n, m)==false)printf("float wrong!\n");
	float Delta_score = INF;

	GwsForTriangulation gws[100];// = (GwsForTriangulation*)malloc(GreyWolves_num * sizeof(GwsForTriangulation));

	double c[180];// = (double*)malloc((m - 1) * sizeof(double));
	double D[180];// = (double*)malloc((m - 1) * sizeof(double));
	double A[180];// = (double*)malloc((m - 1) * sizeof(double));
	double X1[180];// = (double*)malloc((m - 1) * sizeof(double));
	double X2[180];// = (double*)malloc((m - 1) * sizeof(double));
	double X3[180];// = (double*)malloc((m - 1) * sizeof(double));

	for (int i = 0; i < GreyWolves_num; i++)
	{
		//gws[i].separatePos = (int*)malloc((m - 1) * sizeof(int));
		if(cuda_pickPos(gws[i].separatePos, n, m)==false)printf("float wrong!\n");
	}
	int m_pos = 0;
	if (isCircle)
	{
		float3 point = sample_n[0];
		float dis = cuda_Norm(cuda_Subtraction(point, sample_m[m - 1]));
		for (int i = 1; i < m; i++)
		{
			float tmp = cuda_Norm(cuda_Subtraction(point, sample_m[m - 1 - i]));
			if (tmp < dis)
			{
				m_pos = i;
				dis = tmp;
			}
		}
	}
	//main loop
	for (double it = 0; it < MaxIt; it++)
	{
		double a = 2 - it * (2 / MaxIt);
		for (int i = 0; i < GreyWolves_num; i++)
		{
			bool skip = false;
			int* pos = gws[i].separatePos;
			//if (!isCircle)
			//{
			//	for (int w = 0; w < m - 1; w++)
			//		printf("%d ", gws[i].separatePos[w]);
			//	printf("\n");
			//	/*printf("%d\n", tmp_triNum);
			//	printf("%f %f %f\n", sample_n[triSt_idx].x, sample_n[triSt_idx].y, sample_n[triSt_idx].z);
			//	printf("%d %f %f %f\n", m - 1 - idx, sample_m[m - 1 - idx].x, sample_m[m - 1 - idx].y, sample_m[m - 1 - idx].z);
			//	printf("%f %f %f\n", sample_n[triSt_idx + 1].x, sample_n[triSt_idx + 1].y, sample_n[triSt_idx + 1].z);*/
			//	printf("_______\n");
			//}
			int triSt_idx = 0, triEd_idx = pos[0];
			int tri_cnt = 0;
			int pos_idx = 0;
			for (int j = m_pos; j < m + m_pos; j++)
			{
				int idx;
				if (isCircle)
					idx = cuda_mod(j, m - 1);
				else
					idx = cuda_mod(j, m);
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
					tri->p[1] = sample_m[m - 1 - idx];
					tri->p[2] = sample_n[triSt_idx + 1];
					//if (!isCircle)
					{
						/*for (int w = 0; w < m - 1; w++)
							printf("%d ", pos[w]);
						printf("\n");*/
						
						//printf("%d %d %d \n", tmp_triNum, triEd_idx, triSt_idx);
						///*printf("%f %f %f\n", sample_n[triSt_idx].x, sample_n[triSt_idx].y, sample_n[triSt_idx].z);
						//printf("%d %f %f %f\n", m - 1 - idx, sample_m[m - 1 - idx].x, sample_m[m - 1 - idx].y, sample_m[m - 1 - idx].z);
						//printf("%f %f %f\n", sample_n[triSt_idx + 1].x, sample_n[triSt_idx + 1].y, sample_n[triSt_idx + 1].z);*/
						//printf("_______\n");
					}
					tri->normal = cuda_cross(cuda_Subtraction(tri->p[1], tri->p[0]), cuda_Subtraction(tri->p[2], tri->p[0]));
					triSt_idx++;
				}
				//if (skip)
				//{
				//	//skip = false;
				//	break;
				//}
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
			if (skip)
			{
				//skip = false;
				continue;
			}
			pos_idx = 0;
			for (int j = m_pos; j < m - 1 + m_pos; j++)
			{
				if (pos[pos_idx] > n||pos[pos_idx]<0)
				{
					skip = true;
					break;
				}
				int idx = cuda_mod(j, m - 1);
				Triangle* tri = &resultTri[tri_cnt++];
				tri->p[0] = sample_m[m - 1 - (idx + 1)];
				tri->p[1] = sample_n[pos[pos_idx]];
				tri->p[2] = sample_m[m - 1 - idx];
				tri->normal = cuda_cross(cuda_Subtraction(tri->p[1], tri->p[0]), cuda_Subtraction(tri->p[2], tri->p[0]));

				pos_idx++;
			}
			if (skip)
			{
				//skip = false;
				continue;
			}
			gws[i].singleCost = cuda_measureVariance(n, m, resultTri, (n - m) + (m - 1) * 2, pos);
			float fitness = gws[i].singleCost;
			if (fitness < Alpha_score)
			{
				Alpha_score = fitness;
				memcpy(Alpha_pos, gws[i].separatePos, 180 * sizeof(int));
			}
			if (fitness > Alpha_score && fitness < Beta_score)
			{
				Beta_score = fitness;
				memcpy(Beta_pos, gws[i].separatePos, 180 * sizeof(int));
			}
			if (fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score)
			{
				Delta_score = fitness;
				memcpy(Delta_pos, gws[i].separatePos, 180 * sizeof(int));
			}
		}

		for (int i = 0; i < GreyWolves_num; i++)
		{
			cuda_computeX123(Alpha_pos, gws[i], m - 1, a,c,D,A,X1);
			cuda_computeX123(Beta_pos, gws[i], m - 1, a,c,D,A,X2);
			cuda_computeX123(Delta_pos, gws[i], m - 1, a,c,D,A,X3);

			int max_value;
			int min_value;
			int Pos = 0;
			for (int k = 0; k < m - 1; k++)
			{
				//gws[i].separatePos[k] = __double2int_rd((X1[k] + X2[k] + X3[k]) / 3.0);
				double tmp = (X1[k] + X2[k] + X3[k]) / 3.0;
				int t = 0;
				for (; t < n; t++)
				{
					if (tmp < 0)
					{
						gws[i].separatePos[k] = 0;
						break;
					}
					else if (tmp >= n)
					{
						gws[i].separatePos[k] = n - 1;
						break;
					}
					if (abs(t - tmp) < 1)
					{
						gws[i].separatePos[k] = t;
						break;
					}
				}
				if (t == n)printf("Down float wrong!\n");
				//if (!isCircle)
				//{
				//	//for (int w = 0; w < m - 1; w++)
				//	if (k >= 1)
				//	{
				//		int tmpV = gws[i].separatePos[k]- gws[i].separatePos[k-1];
				//		if (tmpV > n+100)
				//			printf("%d %f ", gws[i].separatePos[k], gws[i].separatePos[k - 1]);
				//	}
				//	

				//}
				//if (!isCircle)
				//{
				//	//for (int w = 0; w < m - 1; w++)
				//	if(gws[i].separatePos[k]>n-1)
				//	printf("%d %f ", gws[i].separatePos[k], (X1[k] + X2[k] + X3[k]) / 3);

				//}
				max_value = min(n - (m - 1 - k), n - 1);
				min_value = max(k, Pos);
				if (max_value == min_value)
				{
					Pos = max_value;
					gws[i].separatePos[k] = Pos;
					continue;
				}
				gws[i].separatePos[k] = Min(Max(gws[i].separatePos[k], min_value), max_value);
				Pos = gws[i].separatePos[k];
				
			}
			//if (!isCircle)
			//{
			//	printf("\n");
			//	/*printf("%d\n", tmp_triNum);
			//	printf("%f %f %f\n", sample_n[triSt_idx].x, sample_n[triSt_idx].y, sample_n[triSt_idx].z);
			//	printf("%d %f %f %f\n", m - 1 - idx, sample_m[m - 1 - idx].x, sample_m[m - 1 - idx].y, sample_m[m - 1 - idx].z);
			//	printf("%f %f %f\n", sample_n[triSt_idx + 1].x, sample_n[triSt_idx + 1].y, sample_n[triSt_idx + 1].z);*/
			//	printf("_______\n");
			//}
			
			//free(X1); free(X2); free(X3);
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
					idx = cuda_mod(j, m - 1);
				else
					idx = cuda_mod(j, m);
				int tmp_triNum = triEd_idx - triSt_idx;
				for (int k = 0; k < tmp_triNum; k++)
				{
					Triangle* tri = &resultTri[tri_cnt++];
					tri->flag = 0;
					tri->p[0] = sample_n[triSt_idx];
					tri->p[1] = sample_m[m - 1 - idx];
					tri->p[2] = sample_n[triSt_idx + 1];
					if (cuda_Norm(cuda_Subtraction(tri->p[0], tri->p[2])) < 1e-5)
					{
						printf("beam two p coincident!\n");
						tri->flag = 1;
					}
					/*if (!isCircle)
					{
						printf("%d\n", tmp_triNum);
						printf("%f %f %f\n", sample_n[triSt_idx].x, sample_n[triSt_idx].y, sample_n[triSt_idx].z);
						printf("%d %f %f %f\n", m - 1 - idx, sample_m[m - 1 - idx].x, sample_m[m - 1 - idx].y, sample_m[m - 1 - idx].z);
						printf("%f %f %f\n", sample_n[triSt_idx + 1].x, sample_n[triSt_idx + 1].y, sample_n[triSt_idx + 1].z);
						printf("_______\n");
					}*/
						
					tri->normal = cuda_cross(cuda_Subtraction(tri->p[1], tri->p[0]), cuda_Subtraction(tri->p[2], tri->p[0]));
					tri->normal = cuda_make_norm(tri->normal);
					if (isCircle)
					{
						float3 tmp = tri->p[0];
						tri->p[0] = tri->p[2];
						tri->p[2] = tmp;
						tri->normal = make_float3(-tri->normal.x, -tri->normal.y, -tri->normal.z);
					}
					else
					{
						//tri->flag = 1;
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
				int idx = cuda_mod(j, m - 1);
				Triangle* tri = &resultTri[tri_cnt++];
				tri->flag = 0;
				tri->p[0] = sample_m[m - 1 - (idx + 1)];
				tri->p[1] = sample_n[pos[pos_idx]];
				tri->p[2] = sample_m[m - 1 - idx];
				if (cuda_Norm(cuda_Subtraction(tri->p[0], tri->p[2])) < 1e-5)
				{
					printf("beam two p coincident!\n");
					tri->flag = 1;
				}
				tri->normal = cuda_cross(cuda_Subtraction(tri->p[1], tri->p[0]), cuda_Subtraction(tri->p[2], tri->p[0]));
				tri->normal = cuda_make_norm(tri->normal);

				if (isCircle)
				{
					float3 tmp = tri->p[0];
					tri->p[0] = tri->p[2];
					tri->p[2] = tmp;
					tri->normal = make_float3(-tri->normal.x, -tri->normal.y, -tri->normal.z);

				}
				else
				{
					//tri->flag = 1;
				}
				pos_idx++;
			}
		}
	}
	/*for (int i = 0; i < GreyWolves_num; i++)
	{
		free(gws[i].separatePos);
	}*/
	//free(gws);
	/*free(Alpha_pos);
	free(Beta_pos);
	free(Delta_pos);*/
	//free(c); free(D); free(A); free(X1); free(X2); free(X3);
}
__global__ void beamTriangulation(int elines, BeamPlugin** beamVector, float3* Allsample, ArcType* AllbeamArc, Triangle* AllresultTri)
{
	int thread_id = threadIdx.x + blockIdx.x*blockDim.x;

	while (thread_id < elines)
	{
		int i = thread_id + 1;

		BeamPlugin* b = beamVector[i];
		int beamArcSt = 0;
		int beamSampleSt = 0;
		int beamTriSt = 0;
		for (int k = 1; k < i; k++)
		{
			/*beamSampleSt += AllbeamArc[beamArcSt].sampleNum + AllbeamArc[beamArcSt + 1].sampleNum;
			beamTriSt += AllbeamArc[beamArcSt].sampleNum + AllbeamArc[beamArcSt + 1].sampleNum - 2;*/
			BeamPlugin* btmp = beamVector[k];
			for (int idx = 0; idx < 2; idx++)
			{
				for (int j = 0; j < btmp->arcNum[idx]; j++)
				{
					beamSampleSt+= btmp->arcArray[idx][j]->sampleNum - 1;
					beamTriSt+= btmp->arcArray[idx][j]->sampleNum - 1;
				}
				beamSampleSt++;
				beamTriSt++;
			}
			beamTriSt -= 2;
			beamArcSt += 2;

		}
		ArcType* beamArc[2];
		for (int idx = 0; idx < 2; idx++)
		{
			beamArc[idx] = AllbeamArc + beamArcSt + idx;
			beamArc[idx]->sample = Allsample + beamSampleSt;
			beamArc[idx]->sampleNum = 0;
			int smallArcSt = 0;
			for (int j = 0; j < b->arcNum[idx]; j++)
			{
				beamArc[idx]->sampleNum += b->arcArray[idx][j]->sampleNum - 1;
				//计算旋转后的采样点位置
				b->arcArray[idx][j]->sample = &Allsample[beamSampleSt + smallArcSt];
				Allsample[beamSampleSt + smallArcSt] = b->arcArray[idx][j]->st;
				int k = 1;
				for (; k <= b->arcArray[idx][j]->sampleNum - 2; k++)
				{
					float m[16];
					cuda_getRotationMatrix(b->arcArray[idx][j]->diskNorm, b->arcArray[idx][j]->segTheta*k, b->arcArray[idx][j]->o,m);
					float3 rotateV = make_float3(Allsample[beamSampleSt + smallArcSt].x - b->arcArray[idx][j]->o.x,
						Allsample[beamSampleSt + smallArcSt].y - b->arcArray[idx][j]->o.y, Allsample[beamSampleSt + smallArcSt].z - b->arcArray[idx][j]->o.z);
					float4 RV = make_float4(rotateV.x, rotateV.y, rotateV.z, 1);
					RV = cuda_multiply(m, RV);
					Allsample[beamSampleSt + smallArcSt + k] = make_float3(RV.x, RV.y, RV.z);

					
				}
				if (j == b->arcNum[idx] - 1)
				{
					Allsample[beamSampleSt + smallArcSt + k] = b->arcArray[idx][j]->ed;
				}
				smallArcSt += b->arcArray[idx][j]->sampleNum - 1;
			}
			beamArc[idx]->sampleNum++;
			beamSampleSt += beamArc[idx]->sampleNum;
		}
		int resultTriNum = beamArc[0]->sampleNum + beamArc[1]->sampleNum - 2;
		//if (b->convergeF[0] == 0 && b->convergeF[1] == 0)
		//{
		//	Triangle* resultTri = AllresultTri + beamTriSt;
		//	for (int j=0; j < resultTriNum; j++)
		//	{
		//		Triangle* tri = &resultTri[j];
		//		tri->flag = 1;
		//	}
		//	thread_id += blockDim.x * gridDim.x;
		//	continue;
		//}
		//else if (b->convergeF[0] == 1 && b->convergeF[1] == 0)
		//{
		//	Triangle* resultTri = AllresultTri + beamTriSt;
		//	int j = 0;
		//	for (; j < beamArc[0]->sampleNum-1; j++)//n triangles
		//	{
		//		Triangle* tri = &resultTri[j];
		//		tri->flag = 0;
		//		tri->p[0] = beamArc[0]->sample[j];
		//		tri->p[2] = b->axis.p[0];
		//		tri->p[1] = beamArc[0]->sample[j + 1];
		//		tri->normal = cuda_cross(cuda_Subtraction(tri->p[1], tri->p[0]), cuda_Subtraction(tri->p[2], tri->p[0]));
		//	}
		//	for (; j < resultTriNum; j++)
		//	{
		//		Triangle* tri = &resultTri[j];
		//		tri->flag = 1;
		//	}
		//	thread_id += blockDim.x * gridDim.x;
		//	continue;
		//}
		//else if (b->convergeF[0] == 0 && b->convergeF[1] == 1)
		//{
		//	Triangle* resultTri = AllresultTri + beamTriSt;
		//	int j = 0;
		//	for (; j < beamArc[1]->sampleNum-1; j++)//n triangles
		//	{
		//		Triangle* tri = &resultTri[j];
		//		tri->flag = 0;
		//		tri->p[0] = beamArc[1]->sample[j];
		//		tri->p[2] = b->axis.p[1];
		//		tri->p[1] = beamArc[1]->sample[j + 1];
		//		tri->normal = cuda_cross(cuda_Subtraction(tri->p[1], tri->p[0]), cuda_Subtraction(tri->p[2], tri->p[0]));
		//	}
		//	for (; j < resultTriNum; j++)
		//	{
		//		Triangle* tri = &resultTri[j];
		//		tri->flag = 1;
		//	}
		//	thread_id += blockDim.x * gridDim.x;
		//	continue;
		//}
		//printf("%d %d\n", beamArc[0]->sampleNum, beamArc[1]->sampleNum);
		if (beamArc[0]->sampleNum > beamArc[1]->sampleNum)
		{
			cuda_GWOforQuadTriangulation(beamArc[0]->sampleNum, beamArc[1]->sampleNum, beamArc[0]->sample, beamArc[1]->sample, AllresultTri + beamTriSt, resultTriNum, true);
		}
		else
		{
			cuda_GWOforQuadTriangulation(beamArc[1]->sampleNum, beamArc[0]->sampleNum, beamArc[1]->sample, beamArc[0]->sample, AllresultTri + beamTriSt, resultTriNum, true);
		}
		thread_id += blockDim.x * gridDim.x;
	}
	return;
}


extern "C" void call_beamTriangulation(int elines,BeamPlugin** beamVector, float3* Allsample, ArcType* AllbeamArc, Triangle* AllresultTri)
{
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	beamTriangulation << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (elines, beamVector, Allsample, AllbeamArc, AllresultTri);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%f\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}




__global__ void quadTriangulation(int totalQuadNum,QuadFace* AllQuad, Triangle* AllQuadresultTri)
{
	int thread_id = threadIdx.x + blockIdx.x*blockDim.x;

	while (thread_id < totalQuadNum)
	{
		int i = thread_id;

		QuadFace* q = AllQuad + i;
		int sampleSt = 0;
		for (int j = 0; j < i; j++)
		{
			QuadFace* qtmp = AllQuad + j;
			sampleSt += qtmp->arc[0]->sampleNum + qtmp->arc[1]->sampleNum - 2;
		}
		int resultTriNum = q->arc[0]->sampleNum + q->arc[1]->sampleNum - 2;
		/*if (q->convergeF == 0)
		{
			Triangle* resultTri = AllQuadresultTri + sampleSt;
			for (int j=0; j < resultTriNum; j++)
			{
				Triangle* tri = &resultTri[j];
				tri->flag = 1;
			}
			thread_id += blockDim.x * gridDim.x;
			continue;
		}*/
		if (q->F[0] == 0)printf("FFFFF0\n");
		if (q->F[1] == 0)printf("FFFFF1\n");
		if (q->arc[0]->sampleNum > q->arc[1]->sampleNum)
		{
			cuda_GWOforQuadTriangulation(q->arc[0]->sampleNum, q->arc[1]->sampleNum, q->arc[0]->sample, q->arc[1]->sample, AllQuadresultTri + sampleSt, resultTriNum, false);
		}
		else
		{
			cuda_GWOforQuadTriangulation(q->arc[1]->sampleNum, q->arc[0]->sampleNum, q->arc[1]->sample, q->arc[0]->sample, AllQuadresultTri + sampleSt, resultTriNum, false);
		}
		thread_id += blockDim.x * gridDim.x;
	}
	return;
}

extern "C" void call_quadTriangulation(int totalQuadNum, QuadFace* AllQuad, Triangle* AllQuadresultTri)
{
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	quadTriangulation << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (totalQuadNum, AllQuad, AllQuadresultTri);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%f\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
__global__ void getMinAngle(int plines, BeamPlugin** G, int* beamNumVec, float3* positions);
__global__ void adjustRadius(int elines, BeamPlugin** beamVector, float threshold);
extern "C" void call_latticePreprocessing(int plines, BeamPlugin** G, int* beamNumVec, float3* positions,int elines, BeamPlugin** beamVector,float threshold)
{
	//for (;;)
	{
		getMinAngle << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (plines,G,beamNumVec,positions);
		
		adjustRadius << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (elines, beamVector, threshold);
	}
}
__device__ void getStrutAngle(BeamPlugin** beams, int size, float3 startP)
{
	short* flag = (short*)malloc(size * sizeof(short));
	for (int i = 0; i < size - 1; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			BeamPlugin* b1 = beams[i];
			BeamPlugin* b2 = beams[j];
			
			
			float3 st, ed;
			if (sqrt(pow(startP.x - b1->axis.p[0].x, 2) + pow(startP.y - b1->axis.p[0].y, 2) + pow(startP.z - b1->axis.p[0].z, 2)) < 1e-6)
			{
				st = b1->axis.p[0];
				ed = b1->axis.p[1]; flag[i] = 0;
			}
			else
			{
				st = b1->axis.p[1];
				ed = b1->axis.p[0]; flag[i] = 1;
			}
			float3 direc1 = make_float3(ed.x - st.x, ed.y - st.y, ed.z - st.z);
			direc1 = cuda_make_norm(direc1); 
			if (sqrt(pow(startP.x - b2->axis.p[0].x, 2) + pow(startP.y - b2->axis.p[0].y, 2) + pow(startP.z - b2->axis.p[0].z, 2)) < 1e-6)
			{
				st = b2->axis.p[0];
				ed = b2->axis.p[1]; flag[j] = 0;
			}
			else
			{
				st = b2->axis.p[1];
				ed = b2->axis.p[0]; flag[j] = 1;
			}
			float3 direc2 = make_float3(ed.x - st.x, ed.y - st.y, ed.z - st.z);
			direc2 = cuda_make_norm(direc2);  
			double cosVal = cuda_Dot(direc1, direc2) / (cuda_Norm(direc1)*cuda_Norm(direc2));
			if (abs(cosVal + 1) < 1e-1)
			{
				continue;
			}
			double angle = acos(Min(Max(cosVal, -1.0f), 1.0f));//是弧度角Min(Max(cuda_Dot(t->normal, quadT->normal), -1.0f), 1.0f)
			b1->minAngle[flag[i]] = Min(b1->minAngle[flag[i]], angle);
			b2->minAngle[flag[j]] = Min(b2->minAngle[flag[j]], angle);
		}
	}
	free(flag);
}
__global__ void getMinAngle(int plines, BeamPlugin** G, int* beamNumVec, float3* positions)
{
	int thread_id = threadIdx.x + blockIdx.x*blockDim.x;

	while (thread_id < plines)
	{
		int i = thread_id + 1;
		int st = 0, flagSt = 0;
		for (int j = 1; j < i; j++)
		{
			st += beamNumVec[j];
		}
		BeamPlugin** beams = G + st;
		int size = beamNumVec[i];
		float3 startP = positions[i];

		if (size > 1)
		{
			getStrutAngle(beams, size, startP);
		}
		thread_id += blockDim.x * gridDim.x;
	}
	return;
}
__global__ void adjustRadius(int elines, BeamPlugin** beamVector, float threshold)
{
	int thread_id = threadIdx.x + blockIdx.x*blockDim.x;

	while (thread_id < elines)
	{
		int i = thread_id + 1;
		BeamPlugin* b = beamVector[i];
		float angle = Min(b->minAngle[0], b->minAngle[1])/2;
		float dis = 2 * sin(angle)*threshold;
		if (dis < b->radius*1.5)
		{
			b->radius = dis/1.5;
		}
		thread_id += blockDim.x * gridDim.x;
	}
	return;
}