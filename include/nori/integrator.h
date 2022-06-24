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
#include <nori/object.h>
#include <cuda_runtime.h>
#include <nori/sampler.h>
#include <nori/accelbvh.h>
#include <nori/mesh.h>

namespace nori{
    /**
     * \brief Abstract integrator (i.e. a rendering technique)
     *
     * In Nori, the different rendering techniques are collectively referred to as 
     * integrators, since they perform integration over a high-dimensional
     * space. Each integrator represents a specific approach for solving
     * the light transport equation---usually favored in certain scenarios, but
     * at the same time affected by its own set of intrinsic limitations.
     */
    struct LeafNode;
    struct InternalNode;
    class NormalIntegrator: public NoriObject {
    public:

        NormalIntegrator(const PropertyList &props){}
        /// Release all memory
        ~NormalIntegrator() { }


        /**
         * \brief Sample the incident radiance along a ray
         *
         * \param scene
         *    A pointer to the underlying scene
         * \param sampler
         *    A pointer to a sample generator
         * \param ray
         *    The ray in question
         * \return
         *    A (usually) unbiased estimate of the radiance in this direction
         */
        //CUDA_DEV
        //ColorGPU3f Li(const SceneKernel *scene, IndependentSampler* sampler, const RayGPU3f &ray) const;


        std::string toString() const {
            return "GPU-Integrator(Normal)[]";
        }   
        /**
         * \brief Return the type of object (i.e. Mesh/BSDF/etc.) 
         * provided by this instance
         * */
        EClassType getClassType() const { return EIntegrator; }
    };

    class SimpleIntegrator: public NoriObject {
    public:
        PointGPU3f position;
        ColorGPU3f energy;

        SimpleIntegrator(const PropertyList &props){
            position = props.getPoint("position");
            energy = props.getColor("energy");
        }
        /// Release all memory
        ~SimpleIntegrator() { }


        /**
         * \brief Sample the incident radiance along a ray
         *
         * \param scene
         *    A pointer to the underlying scene
         * \param sampler
         *    A pointer to a sample generator
         * \param ray
         *    The ray in question
         * \return
         *    A (usually) unbiased estimate of the radiance in this direction
         */
        //CUDA_DEV
        //ColorGPU3f Li(const SceneKernel *scene, IndependentSampler* sampler, const RayGPU3f &ray) const;


        std::string toString() const {
            return "GPU-Integrator(Simple)[]";
        }   
        /**
         * \brief Return the type of object (i.e. Mesh/BSDF/etc.) 
         * provided by this instance
         * */
        EClassType getClassType() const { return EIntegrator; }
    };

    class NormalKernel {
    public:
        NormalKernel(){}
        /// Release all memory
        ~NormalKernel() {}

        CUDA_DEV
        ColorGPU3f Li(SceneKernel *scene, IndependentKernel* sampler, RayGPU3f* ray, IntersectionGPU* its, uint32_t* t_indices, uint32_t* m_indices, uint32_t* nT,  LeafNode*  leafNodes,InternalNode* internalNodes,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize, uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const;   
        
    };

    class SimpleKernel {
    public:
        PointGPU3f position;
        ColorGPU3f energy;

        SimpleKernel(PointGPU3f p, ColorGPU3f e){
            position = p;
            energy = e;
        }
        ~SimpleKernel(){}

        CUDA_DEV
        ColorGPU3f Li(SceneKernel *scene, IndependentKernel* sampler, RayGPU3f* ray, IntersectionGPU* its, uint32_t* t_indices, uint32_t* m_indices, uint32_t* nT,  LeafNode*  leafNodes,InternalNode* internalNodes,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize, uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const;
    };



}


