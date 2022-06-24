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
#include <nori/rfilter.h>
#include <nori/transform.h>
#include <Eigen/Geometry>

namespace nori{

/**
 * \brief Generic camera interface
 * 
 * This class provides an abstract interface to cameras in Nori and
 * exposes the ability to sample their response function. By default, only
 * a perspective camera implementation exists, but you may choose to
 * implement other types (e.g. an environment camera, or a physically-based 
 * camera model that simulates the behavior actual lenses)
 */
class PerspectiveCamera : public NoriObject{
    friend class PerspectiveKernel;
    public:
        /**
         * \brief Importance sample a ray according to the camera's response function
         *
         * \param ray
         *    A ray data structure to be filled with a position 
         *    and direction value
         *
         * \param samplePosition
         *    Denotes the desired sample position on the film
         *    expressed in fractional pixel coordinates
         *
         * \param apertureSample
         *    A uniformly distributed 2D vector that is used to sample
         *    a position on the aperture of the sensor if necessary.
         *
         * \return
         *    An importance weight associated with the sampled ray.
         *    This accounts for the difference in the camera response
         *    function and the sampling density.
         */

        PerspectiveCamera(const PropertyList &propList);
        ~PerspectiveCamera(){
            free(m_outputSize);
            free(m_invOutputSize);
            free(m_fov);
            free(m_nearClip);
            free(m_farClip);
        }

        void activate();
        /*
        CUDA_DEV
        ColorGPU3f sampleRay(RayGPU3f &ray,
            const PointGPU2f &samplePosition,
            const PointGPU2f &apertureSample) const{}
        */
        /// Return the size of the output image in pixels
        VectorGPU2i *getOutputSize() const { return m_outputSize; }

        /// Return the camera's reconstruction filter in image space
        /*
        const GaussianFilter *getReconstructionFilter() const { return m_rfilter_gg; }
        const MitchellNetravaliFilter *getReconstructionFilterM() const { return m_rfilter_mm; }
        const TentFilter *getReconstructionFilterT() const { return m_rfilter_tt; }
        const BoxFilter *getReconstructionFilterB() const { return m_rfilter_bb; }
        */
        /**
         * \brief Return the type of object (i.e. Mesh/Camera/etc.) 
         * provided by this instance
         * */
        void addChild(NoriObject *obj);
        EClassType getClassType() const { return ECamera; }

        std::string toString() const {
            return "GPU-PerspectiveCamera[]";
        }
        VectorGPU2i *m_outputSize;
        GaussianFilter *m_rfilter_gg;
        /*
        MitchellNetravaliFilter *m_rfilter_mm;
        TentFilter *m_rfilter_tt;
        BoxFilter *m_rfilter_bb;
        */
        VectorGPU2f* m_invOutputSize;
        Transform m_sampleToCamera;
        Transform m_cameraToWorld;
        float* m_fov;
        float* m_nearClip;
        float* m_farClip;
    };

    class PerspectiveKernel{
    public:
        PerspectiveKernel(){}
        

        CUDA_DEV
        ColorGPU3f sampleRay(RayGPU3f *ray,  PointGPU2f samplePosition,  PointGPU2f apertureSample,  VectorGPU2i* m_outputSize_dev, VectorGPU2f* m_invOutputSize_dev, float* m_sampleToCamera_dev_tr, float* m_cameraToWorld_dev_tr,float* m_sampleToCamera_dev_inv, float* m_cameraToWorld_dev_inv, float* m_nearClip_dev, float* m_farClip_dev,const VectorGPU2i * resSize, uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const;


        /*
        GaussianFilter *m_rfilter_gg;
        MitchellNetravaliFilter *m_rfilter_mm;
        TentFilter *m_rfilter_tt;
        BoxFilter *m_rfilter_bb;
        */
        bool hasFilter =false;


    };


}
