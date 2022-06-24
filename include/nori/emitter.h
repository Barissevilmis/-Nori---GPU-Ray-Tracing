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

#include <nori/vectorgpu.h>
#include <nori/colorgpu.h>
#include <nori/common.h>
#include <nori/object.h>

namespace nori{
    /**
     * \brief Simple struct to hold query records
     * points[0]: Shading Point
     * points[1]: Light Point
     * normals[0]: Shading Normal
     * normals[1]: Light Normal
     */
  /**
     * \brief Simple struct to hold query records
     * points[0]: Shading Point
     * points[1]: Light Point
     * normals[0]: Shading Normal
     * normals[1]: Light Normal
     */
    struct EmitterQueryRecordGPU{

        PointGPU3f s_pnt;
        PointGPU3f l_pnt;
        NormalGPU3f s_nrm;
        NormalGPU3f l_nrm;
        EmitterQueryRecordGPU(PointGPU3f pnt1, PointGPU3f pnt2, NormalGPU3f nrm1, NormalGPU3f nrm2)
        : s_pnt(pnt1), l_pnt(pnt2), s_nrm(nrm1), l_nrm(nrm2) {}
    };

    /**
     * \brief Superclass of all emitters
     */
    class AreaLightEmitter : public NoriObject{
    friend class AreaLightKernel;
    public:

        /**
         * \brief Return the type of object (i.e. Mesh/Emitter/etc.) 
         * provided by this instance
         * */
        AreaLightEmitter(const PropertyList &props);
        EClassType getClassType() const { return EEmitter; }
        CUDA_DEV
        ColorGPU3f lightEmmitPerRecord(const EmitterQueryRecordGPU & rec) const;
        CUDA_DEV
        ColorGPU3f getRadiance() const;

        std::string toString() const {
            return "GPU-AreaLight[]";
        }


    private:
        ColorGPU3f radiance;
    };



    class AreaLightKernel : public NoriObject{
    public:

        /**
         * \brief Return the type of object (i.e. Mesh/Emitter/etc.) 
         * provided by this instance
         * */
        AreaLightKernel(){}
        ~AreaLightKernel(){}
        
        CUDA_DEV
        void setLightSource(const AreaLightEmitter* emit);
        CUDA_DEV
        ColorGPU3f lightEmmitPerRecord(const EmitterQueryRecordGPU & rec) const;
        CUDA_DEV
        ColorGPU3f getRadiance() const;

        ColorGPU3f radiance;
    };

}
