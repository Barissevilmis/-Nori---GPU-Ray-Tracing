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

namespace nori {

    /* ===================================================================
        This file contains a few templates and specializations, which
        provide 2/3D points, vectors, and normals over different
        underlying data types. This implementation will have purpose to be the GPU version of Vector class.
        Points, vectors, and normals are distinct in Nori, because they transform differently under homogeneous
        coordinate transformations.
    * =================================================================== */

    /**
     * \brief Generic N-dimensional vector data structure based on Eigen::Matrix
     */
    template <typename _Scalar, int _Dimension> struct TVectorGPU: public Eigen::Matrix<_Scalar, _Dimension, 1> {
    public:
        enum {
            Dimension = _Dimension
        };

        typedef _Scalar                             Scalar;
        typedef Eigen::Matrix<Scalar, Dimension, 1> Base;
        typedef TVectorGPU<Scalar, Dimension>          VectorType;
        typedef TPointGPU<Scalar, Dimension>           PointType;

        /// Create a new vector with constant component values
        CUDA_HOSTDEV 
        TVectorGPU(Scalar value = (Scalar) 0) { Base::setConstant(value); }

        /// Create a new 2D vector (type error if \c Dimension != 2)
        CUDA_HOSTDEV 
        TVectorGPU(Scalar x, Scalar y) : Base(x, y) { }

        /// Create a new 3D vector (type error if \c Dimension != 3)
        CUDA_HOSTDEV 
        TVectorGPU(Scalar x, Scalar y, Scalar z) : Base(x, y, z) { }

        /// Create a new 4D vector (type error if \c Dimension != 4)
        CUDA_HOSTDEV 
        TVectorGPU(Scalar x, Scalar y, Scalar z, Scalar w) : Base(x, y, z, w) { }

        /// Construct a vector from MatrixBase (needed to play nice with Eigen)
        template <typename Derived> 
        CUDA_HOSTDEV
        TVectorGPU(const Eigen::MatrixBase<Derived>& p)
            : Base(p) { }

        /// Assign a vector from MatrixBase (needed to play nice with Eigen) 
        template <typename Derived> 
        CUDA_HOSTDEV
        TVectorGPU &operator=(const Eigen::MatrixBase<Derived>& p) {
            this->Base::operator=(p);
            return *this;
        }
    };

    /**
     * \brief Generic N-dimensional point data structure based on Eigen::Matrix
     */
    template <typename _Scalar, int _Dimension> struct TPointGPU : public Eigen::Matrix<_Scalar, _Dimension, 1> {
    public:
        enum {
            Dimension = _Dimension
        };

        typedef _Scalar                             Scalar;
        typedef Eigen::Matrix<Scalar, Dimension, 1> Base;
        typedef TVectorGPU<Scalar, Dimension>          VectorType;
        typedef TPointGPU<Scalar, Dimension>           PointType;

        /// Create a new point with constant component vlaues
        CUDA_HOSTDEV 
        TPointGPU(Scalar value = (Scalar) 0) { Base::setConstant(value); }

        /// Create a new 2D point (type error if \c Dimension != 2)
        CUDA_HOSTDEV 
        TPointGPU(Scalar x, Scalar y) : Base(x, y) { }

        /// Create a new 3D point (type error if \c Dimension != 3)
        CUDA_HOSTDEV 
        TPointGPU(Scalar x, Scalar y, Scalar z) : Base(x, y, z) { }

        /// Create a new 4D point (type error if \c Dimension != 4)
        CUDA_HOSTDEV 
        TPointGPU(Scalar x, Scalar y, Scalar z, Scalar w) : Base(x, y, z, w) { }

        /// Construct a point from MatrixBase (needed to play nice with Eigen)
        
        template <typename Derived> 
        CUDA_HOSTDEV 
        TPointGPU(const Eigen::MatrixBase<Derived>& p)
            : Base(p) { }

        /// Assign a point from MatrixBase (needed to play nice with Eigen)
        template <typename Derived> 
        CUDA_HOSTDEV
        TPointGPU &operator=(const Eigen::MatrixBase<Derived>& p) {
            this->Base::operator=(p);
            return *this;
        }
    };

    /**
     * \brief 3-dimensional surface normal representation
     */
    struct NormalGPU3f : public Eigen::Matrix<float, 3, 1> {
    public:
        enum {
            Dimension = 3
        };

        typedef float                               Scalar;
        typedef Eigen::Matrix<Scalar, Dimension, 1> Base;
        typedef TVectorGPU<Scalar, Dimension>          VectorType;
        typedef TPointGPU<Scalar, Dimension>           PointType;


        /// Create a new normal with constant component vlaues
        CUDA_HOSTDEV 
        NormalGPU3f(Scalar value = 0.0f) { Base::setConstant(value); }

        /// Create a new 3D normal
        CUDA_HOSTDEV 
        NormalGPU3f(Scalar x, Scalar y, Scalar z) : Base(x, y, z) { }

        /// Construct a normal from MatrixBase (needed to play nice with Eigen) 
        template <typename Derived> 
        CUDA_HOSTDEV
        NormalGPU3f(const Eigen::MatrixBase<Derived>& p)
            : Base(p) { }

        /// Assign a normal from MatrixBase (needed to play nice with Eigen)
        template <typename Derived> 
        CUDA_HOSTDEV
        NormalGPU3f &operator=(const Eigen::MatrixBase<Derived>& p) {
            this->Base::operator=(p);
            return *this;
            }

    // /// Complete the set {a} to an orthonormal base
    // extern void coordinateSystem(const VectorGPU3f &a, VectorGPU3f &b, VectorGPU3f &c);
    };
}
