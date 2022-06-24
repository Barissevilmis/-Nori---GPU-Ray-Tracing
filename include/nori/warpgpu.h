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

#include <nori/common.h>
#include <nori/sampler.h>
#include <nori/vectorgpu.h>
#include <nori/framegpu.h>


namespace nori {

    /// A collection of useful warping functions for importance sampling
    class WarpGPU {
    public:

        WarpGPU();
        /// Dummy warping function: takes uniformly distributed points in a square and just returns them
        CUDA_DEV
        PointGPU2f squareToUniformSquare(const PointGPU2f &sample)const;

        /// Probability density of \ref squareToUniformSquare()
        CUDA_DEV 
        float squareToUniformSquarePdf(const PointGPU2f &p)const;
        /// Sample a 2D tent distribution
        CUDA_DEV 
        PointGPU2f squareToTent(const PointGPU2f &sample)const;

        /// Probability density of \ref squareToTent()
        CUDA_DEV 
        float squareToTentPdf(const PointGPU2f &p)const;

        /// Uniformly sample a vector on a 2D disk with radius 1, centered around the origin
        CUDA_DEV 
        PointGPU2f squareToUniformDisk(const PointGPU2f &sample)const;

        /// Probability density of \ref squareToUniformDisk()
        CUDA_DEV 
        float squareToUniformDiskPdf(const PointGPU2f &p)const;

        /// Uniformly sample a vector on the unit sphere with respect to solid angles
        CUDA_DEV 
        VectorGPU3f squareToUniformSphere(const PointGPU2f &sample)const;

        /// Probability density of \ref squareToUniformSphere()
        CUDA_DEV 
        float squareToUniformSpherePdf(const VectorGPU3f &v)const;

        /// Uniformly sample a vector on the unit hemisphere around the pole (0,0,1) with respect to solid angles
        CUDA_DEV 
        VectorGPU3f squareToUniformHemisphere(const PointGPU2f &sample)const;

        /// Probability density of \ref squareToUniformHemisphere()
        CUDA_DEV 
        float squareToUniformHemispherePdf(const VectorGPU3f &v)const;

        /// Uniformly sample a vector on the unit hemisphere around the pole (0,0,1) with respect to projected solid angles
        CUDA_DEV 
        VectorGPU3f squareToCosineHemisphere(const PointGPU2f &sample)const;

        /// Probability density of \ref squareToCosineHemisphere()
        CUDA_DEV 
        float squareToCosineHemispherePdf(const VectorGPU3f &v)const;

        /// Warp a uniformly distributed square sample to a Beckmann distribution * cosine for the given 'alpha' parameter
        CUDA_DEV 
        VectorGPU3f squareToBeckmann(const PointGPU2f &sample, float alpha)const;

        /// Probability density of \ref squareToBeckmann()
        CUDA_DEV 
        float squareToBeckmannPdf(const VectorGPU3f &m, float alpha)const;
    };

}
