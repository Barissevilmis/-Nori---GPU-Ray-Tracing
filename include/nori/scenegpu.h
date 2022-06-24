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

#pragma once
#include <nori/block.h>
#include <nori/sampler.h>
#include <nori/integrator.h>
#include <nori/mesh.h>
#include <nori/camera.h>
#include <nori/accelbvh.h>

namespace nori{

    /**
     * \brief Main scene data structure
     *
     * This class holds information on scene objects and is responsible for
     * coordinating rendering jobs. It also provides useful query routines that
     * are mostly used by the \ref Integrator implementations.
     */
    class BVH_d;
    class SceneKernel{
        friend class NormalKernel;
        public:
            SceneKernel();
            ~SceneKernel();
            
            CUDA_DEV 
            bool rayIntersect(RayGPU3f *ray, IntersectionGPU * its, uint32_t* t_indices, uint32_t* m_indices ,uint32_t * nT, LeafNode*  leafNodes,InternalNode* internalNodes,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize, uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const;

            /*
            CUDA_DEV 
            bool rayIntersect(RayGPU3f *ray) const;
            */

            SceneKernel& operator=(const SceneGPU & scene);
            

        public:
            uint32_t *m_meshes_size_h;
            uint32_t *triangle_count_h;
            uint32_t *vertex_count_h;

            uint32_t* triangle_ids_dev;
            uint32_t* mesh_ids_dev;
            BoundingBoxGPU3f*  bboxes_mesh_dev;

            BVH_d m_bvh_dev;

            /* MESH DATA */
            VectorGPU3f      *v_V_dev;
            VectorGPU3f      *v_N_dev;
            VectorGPU2f      *v_UV_dev;
            VectorGPU3i      *v_F_dev;
            uint32_t         *N_exists_dev;
            uint32_t         *UV_exists_dev;
            std::vector<uint32_t> existing_N_ids;
            std::vector<uint32_t> existing_UV_ids;
            /* MESH DATA */


            ColorGPU4f *res_d;   
            ColorGPU4f *res_h;
            VectorGPU2i *resSize_d;
            VectorGPU2i *resSize_h;
    };

    class SceneGPU: public NoriObject {
    friend class SceneKernel;
    public:


        /// Construct a new scene object
        SceneGPU(const PropertyList &);
        
        ~SceneGPU();
        

        /// Return a pointer to the scene's integrator
        const NormalIntegrator *getIntegrator() const { return m_integrator; }

        /// Return a pointer to the scene's integrator
        NormalIntegrator *getIntegrator() { return m_integrator; }

        /// Return a pointer to the scene's camera
        const PerspectiveCamera *getCamera() const { return m_camera; }

        /// Return a pointer to the scene's sample generator
        IndependentSampler * getSampler() { return m_sampler; }

        /// Return a reference to an array containing all meshes
        //const std::vector<Mesh> &getMeshes() const { return  m_meshes; }

        /// Return a reference to an array containing all meshes with emitters
        //const std::vector<Mesh> &getMeshesWithEmitters() const { return m_emitmesh; }

        /**
         * \brief Inherited from \ref NoriObject::activate()
         *
         * Initializes the internal data structures (kd-tree,
         * emitter sampling data structures, etc.)
         */
        void activate();

        /// Add a child object to the scene (meshes, integrators etc.)
        void addChild(NoriObject *obj);

        std::string toString() const {
            return "GPU-Scene[]";
        }

        EClassType getClassType() const { return EScene; }
   


        uint32_t *m_meshes_size_h;
        //uint32_t *m_emitmesh_size_h;
        uint32_t *triangle_count_h;
        uint32_t *vertex_count_h;
       
        //std::vector<Mesh> m_meshes;
        //std::vector<Mesh> m_emitmesh;
        std::vector<uint32_t> offsets;
        std::vector<uint32_t> triangle_ids;
        std::vector<uint32_t> mesh_ids;
        std::vector<BoundingBoxGPU3f> bboxes_mesh;

        std::vector<VectorGPU3f> v_V_h;
        std::vector<VectorGPU3f> v_N_h;
        std::vector<VectorGPU2f> v_UV_h;
        std::vector<VectorGPU3i> v_F_h;
        std::vector<uint32_t>  N_exists_h;
        std::vector<uint32_t>  UV_exists_h;
        //std::vector<BoundingBoxGPU3f> bboxes_emitmesh;


        //BVH_d m_bvh;
        IndependentSampler* m_sampler = nullptr;
        PerspectiveCamera *m_camera = nullptr;
        NormalIntegrator *m_integrator = nullptr;


    };
}
