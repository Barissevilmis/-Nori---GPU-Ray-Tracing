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

#include <nori/emitter.h>


namespace nori{


    /**
     * \brief AreaLight class
     * getRadiance(): Return radiance(Color3f)
     * lightEmmitPerRecord(): Return evaluated result of EmitterQueryRecord(Color3f)
     */
 
    AreaLightEmitter::AreaLightEmitter(const PropertyList &props){
        radiance = props.getColor("radiance");
    }
    CUDA_DEV
    void AreaLightKernel::setLightSource(const AreaLightEmitter* emit){
        radiance = emit->getRadiance();
    }
    CUDA_DEV 
    ColorGPU3f AreaLightKernel::lightEmmitPerRecord(const EmitterQueryRecordGPU & rec) const{
        VectorGPU3f l_dist = rec.l_pnt - rec.s_pnt;
        l_dist.normalize();
        float cos_tht = l_dist.dot(rec.s_nrm);
        if(cos_tht > 0)
            return ColorGPU3f(radiance * cos_tht);
        else
            return ColorGPU3f(0.0f);
    }
    CUDA_DEV 
    ColorGPU3f AreaLightKernel::getRadiance() const{
        return radiance;
    }   
    
    NORI_REGISTER_CLASS(AreaLightEmitter, "area");
}
