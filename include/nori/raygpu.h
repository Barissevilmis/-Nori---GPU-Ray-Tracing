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

namespace nori{

    /**
     * \brief Simple n-dimensional ray segment data structure
     * 
     * Along with the ray origin and direction, this data structure additionally
     * stores a ray segment [mint, maxt] (whose entries may include positive/negative
     * infinity), as well as the componentwise reciprocals of the ray direction.
     * That is just done for convenience, as these values are frequently required.
     *
     * \remark Important: be careful when changing the ray direction. You must
     * call \ref update() to compute the componentwise reciprocals as well, or Nori's
     * ray-triangle intersection code will go haywire.
     */
    template <typename _PointType, typename _VectorType> struct TRayGPU {
        typedef _PointType                  PointType;
        typedef _VectorType                 VectorType;
        typedef typename PointType::Scalar  Scalar;

        PointType o;     ///< Ray origin
        VectorType d;    ///< Ray direction
        VectorType dRcp; ///< Componentwise reciprocals of the ray direction
        Scalar mint;     ///< Minimum position on the ray segment
        Scalar maxt;     ///< Maximum position on the ray segment

        /// Construct a new ray 
        CUDA_DEV
        TRayGPU() : mint(1e-4f), 
            maxt(1.0e308) { }
        
        /// Construct a new ray 
        CUDA_DEV
        TRayGPU(const PointType &o, const VectorType &d) : o(o), d(d), 
                mint(1e-4f), maxt(1.0e308) {
            update();
        }

        /// Construct a new ray 
        CUDA_DEV
        TRayGPU(const PointType &o, const VectorType &d, 
            Scalar mint, Scalar maxt) : o(o), d(d), mint(mint), maxt(maxt) {
            update();
        }

        /// Copy constructor
        CUDA_DEV
        TRayGPU(const TRayGPU &ray) 
        : o(ray.o), d(ray.d), dRcp(ray.dRcp),
        mint(ray.mint), maxt(ray.maxt) { }

        /// Copy a ray, but change the covered segment of the copy 
        CUDA_DEV
        TRayGPU(const TRayGPU &ray, Scalar mint, Scalar maxt) 
        : o(ray.o), d(ray.d), dRcp(ray.dRcp), mint(mint), maxt(maxt) { }

        /// Update the reciprocal ray directions after changing 'd'
        CUDA_DEV
        void update() {
            dRcp = d.cwiseInverse();
        }

        /// Return the position of a point along the ray
        CUDA_DEV
        PointType operator() (Scalar t) const { return o + t * d; }

        /// Return a ray that points into the opposite direction
        CUDA_DEV
        TRayGPU reverse() const {
            TRayGPU result;
            result.o = o; result.d = -d; result.dRcp = -dRcp;
            result.mint = mint; result.maxt = maxt;
            return result;

        }

    };
}
