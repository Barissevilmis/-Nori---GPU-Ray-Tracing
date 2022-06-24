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

#include <nori/camera.h>

namespace nori
{
    CUDA_DEV
    ColorGPU3f PerspectiveKernel::sampleRay(RayGPU3f *ray,  PointGPU2f samplePosition,  PointGPU2f apertureSample,  VectorGPU2i* m_outputSize_dev, VectorGPU2f* m_invOutputSize_dev, float* m_sampleToCamera_dev_tr, float* m_cameraToWorld_dev_tr,float* m_sampleToCamera_dev_inv, float* m_cameraToWorld_dev_inv, float* m_nearClip_dev, float* m_farClip_dev,const VectorGPU2i * resSize,uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const
    {
        /* Compute the corresponding position on the
        near plane (in local camera space) */
        int indexX = *x_render_offset_dev + blockIdx.x * blockDim.x + threadIdx.x;
        int indexY = *y_render_offset_dev + blockIdx.y * blockDim.y + threadIdx.y;
        
        float val1 = m_sampleToCamera_dev_tr[0] * samplePosition.x() * m_invOutputSize_dev->x() + m_sampleToCamera_dev_tr[4] * samplePosition.y() * m_invOutputSize_dev->y() + m_sampleToCamera_dev_tr[8] * 0.f + m_sampleToCamera_dev_tr[12];
        float val2 = m_sampleToCamera_dev_tr[1] * samplePosition.x() * m_invOutputSize_dev->x() + m_sampleToCamera_dev_tr[5] * samplePosition.y() * m_invOutputSize_dev->y() + m_sampleToCamera_dev_tr[9] * 0.f + m_sampleToCamera_dev_tr[13];
        float val3 = m_sampleToCamera_dev_tr[2] * samplePosition.x() * m_invOutputSize_dev->x() + m_sampleToCamera_dev_tr[6] * samplePosition.y() * m_invOutputSize_dev->y() + m_sampleToCamera_dev_tr[10] * 0.f + m_sampleToCamera_dev_tr[14];
        float div = m_sampleToCamera_dev_tr[3] * samplePosition.x() * m_invOutputSize_dev->x() + m_sampleToCamera_dev_tr[7] * samplePosition.y() * m_invOutputSize_dev->y() + m_sampleToCamera_dev_tr[11] * 0.f + m_sampleToCamera_dev_tr[15];
        PointGPU3f nearP = PointGPU3f(val1/div, val2/div, val3/div);

        /* Turn into a normalized ray direction, and
        adjust the ray interval accordingly */
        VectorGPU3f d = nearP.normalized();
        //float len_d = sqrtf(nearP.x() * nearP.x() +  nearP.y() * nearP.y() + nearP.z() * nearP.z());
        //VectorGPU3f d = nearP/len_d;
        float invZ = 1.0f / d.z();

        float o1 = m_cameraToWorld_dev_tr[12];
        float o2 = m_cameraToWorld_dev_tr[13];
        float o3 = m_cameraToWorld_dev_tr[14];
        float divo = m_cameraToWorld_dev_tr[15];
        PointGPU3f op = PointGPU3f(o1/divo, o2/divo, o3/divo);
        
        float d1 = m_cameraToWorld_dev_tr[0] * d.x() + m_cameraToWorld_dev_tr[4] * d.y() + m_cameraToWorld_dev_tr[8] * d.z();
        float d2 = m_cameraToWorld_dev_tr[1] * d.x() + m_cameraToWorld_dev_tr[5] * d.y() + m_cameraToWorld_dev_tr[9] * d.z();
        float d3 = m_cameraToWorld_dev_tr[2] * d.x() + m_cameraToWorld_dev_tr[6] * d.y() + m_cameraToWorld_dev_tr[10] * d.z();

        //printf("%f :kek: %f :kek: %f :kek: %f\n", samplePosition.x(), samplePosition.y(), samplePosition.x()* m_invOutputSize_dev->x(), samplePosition.y()*m_invOutputSize_dev->y());
        //printf("%f :tt: %f :tt: %f\n", d.x(), d.y(), d.z());

        ray[resSize->y() * indexX + indexY].o = op;
        ray[resSize->y() * indexX + indexY].d = VectorGPU3f(d1, d2, d3);
        ray[resSize->y() * indexX + indexY].mint = *m_nearClip_dev * invZ;
        ray[resSize->y() * indexX + indexY].maxt = *m_farClip_dev * invZ;
        ray[resSize->y() * indexX + indexY].update();
    
        return ColorGPU3f(1.0f);
    }

    // Default filter => Gaussian
    void PerspectiveCamera::addChild(NoriObject *obj)
    {
        switch (obj->getClassType())
        {
        case EReconstructionFilter:
            if (m_rfilter_gg)
                throw NoriException("Camera: tried to register multiple reconstruction filters!");
            m_rfilter_gg = static_cast<GaussianFilter *>(obj);
            /*
            if (m_rfilter_mm)
                throw NoriException("Camera: tried to register multiple reconstruction filters!");
            m_rfilter_mm = static_cast<MitchellNetravaliFilter *>(obj);
            if (m_rfilter_tt)
                throw NoriException("Camera: tried to register multiple reconstruction filters!");
            m_rfilter_tt = static_cast<TentFilter *>(obj);
            if (m_rfilter_bb)
                throw NoriException("Camera: tried to register multiple reconstruction filters!");
            m_rfilter_bb = static_cast<BoxFilter *>(obj);
            */
            break;

        default:
            throw NoriException("Camera::addChild(<%s>) is not supported!",
                                classTypeName(obj->getClassType()));
        }
    }

    NORI_REGISTER_CLASS(PerspectiveCamera, "perspective");
}
