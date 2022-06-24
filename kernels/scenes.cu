/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include <nori/scenegpu.h>
#include <nori/common.h>
#include <nori/bitmap.h>
#include <nori/sampler.h>
#include <nori/camera.h>
#include <nori/integrator.h>
#include <nori/emitter.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>

namespace nori{



    SceneKernel::SceneKernel(){}

    SceneKernel::~SceneKernel(){
        free(vertex_count_h);
        free(m_meshes_size_h);
        free(triangle_count_h);
        cudaFree(mesh_ids_dev);
        cudaFree(bboxes_mesh_dev);
        cudaFree(triangle_ids_dev);
        cudaFree(resSize_d);
        cudaFree(v_V_dev);
        cudaFree(v_F_dev);
        if(existing_N_ids.size()){
            cudaFree(v_N_dev);
            cudaFree(N_exists_dev);
        }
        if(existing_N_ids.size()){
            cudaFree(v_UV_dev);
            cudaFree(UV_exists_dev); 
        }
             
    }
    CUDA_DEV 
    bool SceneKernel::rayIntersect(RayGPU3f *ray, IntersectionGPU *its, uint32_t* t_indices, uint32_t* m_indices,uint32_t * nT, LeafNode*  leafNodes,InternalNode* internalNodes,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize, uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const {
        return m_bvh_dev.rayIntersect(ray, its, t_indices, m_indices, false, nT, leafNodes, internalNodes, v_V, v_N, v_UV, v_F, N_exists, UV_exists, resSize,x_render_offset_dev, y_render_offset_dev);
    }

    /**
     * \brief Intersect a ray against all triangles stored in the scene
     * and \a only determine whether or not there is an intersection.
     *
     * This method much faster than the other ray tracing function,
     * but the performance comes at the cost of not providing any
     * additional information about the detected intersection
     * (not even its position).
     *
     * \param ray
     *    A 3-dimensional ray data structure with minimum/maximum
     *    extent information
     *
     * \return \c true if an intersection was found
     */
    /*
    CUDA_DEV 
    bool SceneKernel::rayIntersect(RayGPU3f *ray) const {
        IntersectionGPU *its;
        return m_bvh_dev.rayIntersect(ray, its, true);
    }*/
    /// Construct a new scene object
    SceneGPU::SceneGPU(const PropertyList &){
        m_meshes_size_h = (uint32_t*)(malloc(sizeof(uint32_t)));
        //m_emitmesh_size_h = (uint32_t*)(malloc(sizeof(uint32_t)));
        triangle_count_h = (uint32_t*)(malloc(sizeof(uint32_t)));
        vertex_count_h = (uint32_t*)(malloc(sizeof(uint32_t)));
        *m_meshes_size_h = 0;
        //*m_emitmesh_size_h = 0;
        *triangle_count_h = 0;
        *vertex_count_h = 0;
        offsets.push_back(0);
    }
    SceneGPU::~SceneGPU() {
    
        offsets.clear();
        triangle_ids.clear();
        mesh_ids.clear();
        bboxes_mesh.clear();
        
        v_V_h.clear();
        v_N_h.clear();
        v_UV_h.clear();
        v_F_h.clear();
        N_exists_h.clear();
        UV_exists_h.clear();

        //m_emitmesh.clear(); 
        //bboxes_emitmesh.clear();
        
        free(m_meshes_size_h);
        free(triangle_count_h);
        free(vertex_count_h);
    
    }
    
    /**
     * \brief Inherited from \ref NoriObject::activate()
     *
     * Initializes the internal data structures (kd-tree,
     * emitter sampling data structures, etc.)
     */
    void SceneGPU::activate(){
        if (!m_integrator)
            throw NoriException("No integrator was specified!");
        if (!m_camera)
            throw NoriException("No camera was specified!");
        
        if (!m_sampler) {
            /* Create a default (independent) sampler */
            m_sampler = static_cast<IndependentSampler *>(
                NoriObjectFactory::createInstance("independent", PropertyList()));
        }
        triangle_ids.reserve(*triangle_count_h);
        mesh_ids.reserve(*triangle_count_h);
        uint32_t sum_i = 0;
        for(uint32_t i = 0; i < offsets.size()-1; i++){
            sum_i += offsets[i];
            for(uint32_t j = sum_i; j < offsets[i+1] + sum_i; j++){
                triangle_ids[j] = j - sum_i;
                mesh_ids[j] = i;
            }
        }
        
    }

    SceneKernel& SceneKernel::operator=(const SceneGPU & scene){ 
        triangle_count_h = (uint32_t*)malloc(sizeof(uint32_t));
        m_meshes_size_h = (uint32_t*)malloc(sizeof(uint32_t));
        vertex_count_h = (uint32_t*)malloc(sizeof(uint32_t));
        *m_meshes_size_h = *(scene.m_meshes_size_h);
        *triangle_count_h = *(scene.triangle_count_h);
        *vertex_count_h = *(scene.vertex_count_h);
        
        /*Check if normal and/or texture data exists if so fow which meshes? 1 yes : 0 no*/
        printf("Loading Triangle and Boundingbox data of size %u\n",*triangle_count_h );
        checkCuda(cudaMalloc((void**)&mesh_ids_dev, *triangle_count_h * sizeof(uint32_t)));
        checkCuda(cudaMalloc((void**)&triangle_ids_dev, *triangle_count_h  * sizeof(uint32_t)));
        checkCuda(cudaMalloc((void**)&bboxes_mesh_dev,*triangle_count_h  * sizeof(BoundingBoxGPU3f)));

        printf("Loading Vertex data of size %u\n",*vertex_count_h );

        /*Allocate data for mesh information*/
        checkCuda(cudaMalloc((void**)&v_V_dev,*vertex_count_h  * sizeof(VectorGPU3f)));
        checkCuda(cudaMalloc((void**)&v_F_dev,*triangle_count_h  * sizeof(VectorGPU3i)));      
       
        checkCuda(cudaMemcpy(mesh_ids_dev,scene.mesh_ids.data(),*triangle_count_h  * sizeof(uint32_t), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(triangle_ids_dev, scene.triangle_ids.data(),*triangle_count_h  * sizeof(uint32_t), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(bboxes_mesh_dev, scene.bboxes_mesh.data(),*triangle_count_h * sizeof(BoundingBoxGPU3f), cudaMemcpyHostToDevice));

        /*Copy mesh data into scene and then bvh tree*/
        checkCuda(cudaMemcpy(v_V_dev,scene.v_V_h.data(),*vertex_count_h  * sizeof(VectorGPU3f), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(v_F_dev,scene.v_F_h.data(),*triangle_count_h  * sizeof(VectorGPU3i), cudaMemcpyHostToDevice));

        checkCuda(cudaMalloc((void**)&N_exists_dev,*m_meshes_size_h  * sizeof(uint32_t)));
        checkCuda(cudaMemcpy(N_exists_dev,scene.N_exists_h.data(),*m_meshes_size_h  * sizeof(uint32_t),cudaMemcpyHostToDevice));    
        checkCuda(cudaMalloc((void**)&UV_exists_dev,*m_meshes_size_h  * sizeof(uint32_t)));
        checkCuda(cudaMemcpy(UV_exists_dev,scene.UV_exists_h.data(),*m_meshes_size_h  * sizeof(uint32_t),cudaMemcpyHostToDevice));       
        /*Check if normal and/or texture data exists if so fow which meshes? 1 yes : 0 no
         Save the corresponding ids: Size will be used for allocatin on GPU*/
        for(uint32_t i = 0; i < *m_meshes_size_h; i++){
            if(scene.N_exists_h[i]){
                existing_N_ids.push_back(i);
            }
            if(scene.UV_exists_h[i]){
                existing_UV_ids.push_back(i);
            }
        }

        if(!existing_N_ids.empty()){
            printf("Loading Normals data of size %u\n",*vertex_count_h );
            checkCuda(cudaMalloc((void**)&v_N_dev, *vertex_count_h  * sizeof(VectorGPU3f)));
            checkCuda(cudaMemcpy(v_N_dev,scene.v_N_h.data(),*vertex_count_h  * sizeof(VectorGPU3f), cudaMemcpyHostToDevice));
        }

        if(!existing_UV_ids.empty()){
            printf("Loading Texture data of size %u\n",*vertex_count_h );
            checkCuda(cudaMalloc((void**)&v_UV_dev,*vertex_count_h  * sizeof(VectorGPU2f)));
            checkCuda(cudaMemcpy(v_UV_dev,scene.v_UV_h.data(),*vertex_count_h  * sizeof(VectorGPU2f), cudaMemcpyHostToDevice));     
        }
        
        printf("Data loaded onto GPU! Switching to BVH tree!\n");
        m_bvh_dev.setUp(bboxes_mesh_dev, triangle_ids_dev, mesh_ids_dev, triangle_count_h, v_V_dev, v_N_dev,v_UV_dev,v_F_dev,N_exists_dev,UV_exists_dev, vertex_count_h);
        printf("BVH tree setup completed!\n");
        return *this;
    }


    /// Add a child object to the scene (meshes, integrators etc.)
    void SceneGPU::addChild(NoriObject *obj){
        switch (obj->getClassType()) {
        case EMesh: {
                Mesh* mesh = static_cast<Mesh*>(obj);
                
                /*
                if(mesh->isEmitter())
                {
                    m_emitmesh.push_back(*mesh);
                    for(int i = 0; i < mesh->getTriangleCount(); i++)
                        bboxes_emitmesh.push_back(mesh->getBoundingBox(i));
                    *m_emitmesh_size_h +=1;
                }
                */
                //m_meshes.push_back(*mesh);
                *m_meshes_size_h +=1;
                *triangle_count_h += mesh->getTriangleCount();
                *vertex_count_h += mesh->getVertexCount();
                offsets.push_back(mesh->getTriangleCount());
                N_exists_h.push_back(mesh->N_exists);
                UV_exists_h.push_back(mesh->UV_exists);
                for(uint32_t i = 0; i < mesh->getTriangleCount(); i++){
                    bboxes_mesh.push_back(mesh->getBoundingBox(i));
                    //printf("%f min %f min % f\n", mesh->getBoundingBox(i).min[0], mesh->getBoundingBox(i).min[1], mesh->getBoundingBox(i).min[2]); 
                    //printf("%f max %f max % f\n", mesh->getBoundingBox(i).max[0], mesh->getBoundingBox(i).max[1], mesh->getBoundingBox(i).max[2]);   
                    v_F_h.push_back(mesh->v_F[i]);               
                }
                for(uint32_t i = 0; i < mesh->getVertexCount(); i++){
                    v_V_h.push_back(mesh->v_V[i]);
                    if(mesh->v_N.size())
                        v_N_h.push_back(mesh->v_N[i]);
                    if(mesh->v_UV.size())
                        v_UV_h.push_back(mesh->v_UV[i]);
                }
                printf("Loading of mesh is completed!\n");
            }
            break;
        
        case EEmitter: 
                /* TBD */
            break;

        case ESampler:
            if (m_sampler)
                throw NoriException("There can only be one sampler per scene!");
            m_sampler = static_cast<IndependentSampler *>(obj);
            break;

        case ECamera:
            if (m_camera)
                throw NoriException("There can only be one camera per scene!");
            m_camera = static_cast<PerspectiveCamera *>(obj);
            break;
        
        case EIntegrator:
            if (m_integrator)
                throw NoriException("There can only be one integrator per scene!");
            m_integrator = static_cast<NormalIntegrator *>(obj);
            break;
        
        default:
            throw NoriException("Scene::addChild(<%s>) is not supported!",classTypeName(obj->getClassType()));
        }
    }
    
    NORI_REGISTER_CLASS(SceneGPU, "scene");
}
