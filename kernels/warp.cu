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

#include <nori/warpgpu.h>

namespace nori {


    WarpGPU::WarpGPU(){};
    /// Dummy warping function: takes uniformly distributed points in a square and just returns them
    CUDA_DEV
    PointGPU2f WarpGPU::squareToUniformSquare(const PointGPU2f &sample)const{
        return sample;
    }

    /// Probability density of \ref squareToUniformSquare()
    CUDA_DEV 
    float WarpGPU::squareToUniformSquarePdf(const PointGPU2f &p)const{
        return ((p.array() >= 0).all() && (p.array() <= 1).all()) ? 1.0f : 0.0f;
    }
    /// Sample a 2D tent distribution
    CUDA_DEV 
    PointGPU2f WarpGPU::squareToTent(const PointGPU2f &sample)const{
        PointGPU2f b = PointGPU2f(-1.f, 0.f);
        PointGPU2f vec_u, vec_v;
        float u, v;
        u = sqrtf(1.f - sample.x());
        v = sample.y() * (1.f - (1.f - u));
        vec_u = PointGPU2f(2.f, 0.f) * (1.f - u);
        vec_v = PointGPU2f(1.f, 1.f) * v;
        return b + vec_u + vec_v;
    }

    /// Probability density of \ref squareToTent()
    CUDA_DEV 
    float WarpGPU::squareToTentPdf(const PointGPU2f &p)const{
        PointGPU2f v_0 = PointGPU2f(-1.f, 0.f);
        PointGPU2f v_1 = PointGPU2f(1.f, 0.f);
        PointGPU2f v_2 = PointGPU2f(0.f, 1.f);
        float s1 = (p.x() - v_1.x()) * (v_0.y() - v_1.y())
                        - ((v_0.x() - v_1.x())) * (p.y() - v_1.y());
        float s2 = (p.x() - v_2.x()) * (v_1.y() - v_2.y())
                        - ((v_1.x() - v_2.x())) * (p.y() - v_2.y());
        float s3 = (p.x() - v_0.x()) * (v_2.y() - v_0.y())
                        - ((v_2.x() - v_0.x())) * (p.y() - v_0.y());
        
        if((s1 < 0 || s2 < 0 || s3 < 0) && (s1 >0 || s2 > 0 || s3 > 0))
            return 0.f;
        else
            return 1.f;
    }

    /// Uniformly sample a vector on a 2D disk with radius 1, centered around the origin
    CUDA_DEV 
    PointGPU2f WarpGPU::squareToUniformDisk(const PointGPU2f &sample)const{
        float param_0 = sqrtf(sample.y());
        float param_1 = 2.f * M_PI * sample.x();
        return PointGPU2f(param_0*cosf(param_1), param_0 * sinf(param_1));
    }

    /// Probability density of \ref squareToUniformDisk()
    CUDA_DEV 
    float WarpGPU::squareToUniformDiskPdf(const PointGPU2f &p)const{
        if((p.x() * p.x() + p.y() * p.y()) <= 1)
            return M_1_PI;
        else
            return 0.f;
    }

    /// Uniformly sample a vector on the unit sphere with respect to solid angles
    CUDA_DEV 
    VectorGPU3f WarpGPU::squareToUniformSphere(const PointGPU2f &sample)const{
        float param_0 = acosf(1 - 2*sample.x());
        float param_1 = 2 * M_PI * sample.y();
        return VectorGPU3f(sinf(param_0) * cosf(param_1), sinf(param_0) * sinf(param_1), cosf(param_0));
    }

    /// Probability density of \ref squareToUniformSphere()
    CUDA_DEV 
    float WarpGPU::squareToUniformSpherePdf(const VectorGPU3f &v)const{
        if(fabsf(v.x() * v.x() + v.y() * v.y() + v.z() * v.z() - 1.f) < Epsilon)
            return M_1_PI / 4.f;
        else
            return 0.f;
}

    /// Uniformly sample a vector on the unit hemisphere around the pole (0,0,1) with respect to solid angles
    CUDA_DEV 
    VectorGPU3f WarpGPU::squareToUniformHemisphere(const PointGPU2f &sample)const{
        float param_0 = acosf(1 - 2*sample.x());
        float param_1 = 2 * M_PI * sample.y();
        return VectorGPU3f(sinf(param_0) * cosf(param_1), sinf(param_0) * sinf(param_1), cosf(param_0));
    }

    /// Probability density of \ref squareToUniformHemisphere()
    CUDA_DEV 
    float WarpGPU::squareToUniformHemispherePdf(const VectorGPU3f &v)const{
        if(fabsf(v.x() * v.x() + v.y() * v.y() + v.z() * v.z() - 1.f) < Epsilon && v.z() > 0.f)
            return M_1_PI / 2.f;
        else
            return 0.f;
    }

    /// Uniformly sample a vector on the unit hemisphere around the pole (0,0,1) with respect to projected solid angles
    CUDA_DEV 
    VectorGPU3f WarpGPU::squareToCosineHemisphere(const PointGPU2f &sample)const{
        float param_0 = acosf(sqrtf(1 - sample.x()));
        float param_1 = 2 * M_PI * sample.y();
        return VectorGPU3f(sinf(param_0) * cosf(param_1), sinf(param_0) * sinf(param_1), cosf(param_0));
    }

    /// Probability density of \ref squareToCosineHemisphere()
    CUDA_DEV 
    float WarpGPU::squareToCosineHemispherePdf(const VectorGPU3f &v)const{
        if(abs(v.x() * v.x() + v.y() * v.y() + v.z() * v.z()) < Epsilon && v.z() > 0.f)
            return M_1_PI * v.z();
        else
            return 0.f;
}

    /// Warp a uniformly distributed square sample to a Beckmann distribution * cosine for the given 'alpha' parameter
    CUDA_DEV 
    VectorGPU3f WarpGPU::squareToBeckmann(const PointGPU2f &sample, float alpha)const{
        float param_0 = atanf(sqrtf(-alpha * alpha * logf(1 - sample.y())));
        float param_1 = 2.f * M_PI * sample.x();
        return VectorGPU3f(sinf(param_0) * cosf(param_1), sinf(param_0) * sinf(param_1), cosf(param_0)).normalized();
    }

    /// Probability density of \ref squareToBeckmann()
    CUDA_DEV 
    float WarpGPU::squareToBeckmannPdf(const VectorGPU3f &m, float alpha)const{  
        if(m.z() > 0)
        {
            float num = 2 * expf(tanf(acosf(m.z())) * tanf(acosf(m.z()))/(alpha * alpha));
            float denum = alpha * alpha * m.z() * m.z() * m.z();
            float res =  (M_1_PI / 2) * (num/denum);
            return res;
        }
        else
            return 0.f; 
    }
}
