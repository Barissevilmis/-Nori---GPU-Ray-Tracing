
   
#pragma once

#include <nori/bboxgpu.h>
#include <nori/block.h>
#include <nori/camera.h>
#include <nori/mesh.h>
#include <nori/integrator.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define GPU_INNER_BLOCK_SIZE 16
#define GPU_GRID_SIZE 12

namespace nori{


    struct Node{
        Node* childA;
        Node* childB;
        Node* parent;
        int flag;
        bool isLeaf;
        BoundingBoxGPU3f BBox;
        int object_id;

        CUDA_DEV 
        Node() : isLeaf(false) , flag(0), parent(nullptr) {}
    };
    struct LeafNode : public Node {
        CUDA_DEV
        LeafNode() {
            this->isLeaf = true;
        }
    };
    struct InternalNode : public Node {
        CUDA_DEV
        InternalNode() {
            this->isLeaf = false;
        }
    };

    //Device BVH
    class BVH_d {
        friend class SceneKernel;
        public:
            uint32_t* mortonCodes;
            uint32_t* object_ids;

            LeafNode*       leafNodes; //numTriangles
            InternalNode*   internalNodes; //numTriangles - 1

            // These are stored in the scene
            uint32_t *numTriangles_h;
            uint32_t *numTriangles;
            uint32_t *numVertices_h;
            //uint32_t *numEmitmesh;
            BoundingBoxGPU3f* Bboxs;
            //BoundingBoxGPU3f* Bboxs_emitmesh;
            uint32_t* m_indices;
            uint32_t* t_indices;
            //MeshKernel* meshGPUs;
            VectorGPU3f mMin;
            VectorGPU3f mMax;

            /* MESH DATA */
            VectorGPU3f      *v_V;
            VectorGPU3f      *v_N;
            VectorGPU2f      *v_UV;
            VectorGPU3i      *v_F;
            uint32_t         *N_exists;
            uint32_t         *UV_exists;
            /* MESH DATA */
            


        public:
            ~BVH_d(){
                cudaFree(numTriangles);
                cudaFree(mortonCodes);
                cudaFree(object_ids);
                cudaFree(leafNodes);
                cudaFree(internalNodes);
                
            }

            void findMinMax(VectorGPU3f& mMin, VectorGPU3f& mMax);
            void setUp(BoundingBoxGPU3f* bboxes_mesh_d,  uint32_t* triangle_ids_d, uint32_t* mesh_ids_d, uint32_t* triangle_count_host, VectorGPU3f* v_V_dev, VectorGPU3f* v_N_dev, VectorGPU2f* v_UV_dev, VectorGPU3i* v_F_dev, uint32_t* N_exists_dev, uint32_t* UV_exists_dev, uint32_t* vertex_count_host);

            void computeMortonCodes(VectorGPU3f& mMin, VectorGPU3f& mMax); //Also Generates the objectIds
            void sortMortonCodes();

            void setupLeafNodes();
            void buildTree();

            CUDA_DEV
            bool rayIntersect(RayGPU3f *ray_, IntersectionGPU *its, uint32_t* t_indices, uint32_t* m_indices, bool shadowRay,uint32_t * nT, LeafNode*  leafNodes,InternalNode* internalNodes,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize, uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const;
            CUDA_DEV
            bool meshIntersect(uint32_t index, RayGPU3f *ray, IntersectionGPU *its,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize,uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const;
            CUDA_DEV 
            bool boundingboxIntersect(BoundingBoxGPU3f bb, RayGPU3f *ray, const VectorGPU2i *resSize,uint32_t* x_render_offset_dev, uint32_t* y_render_offset_dev) const;

            
    };

    class SceneGPU;
    void bvh(SceneGPU & scene, ImageBlock & result, const VectorGPU2i * resSize, uint32_t maxThreadAmount);

}




