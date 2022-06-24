#pragma once

#include <nori/vectorgpu.h>
#include <cuda_runtime.h>

namespace nori{

    /**
     * \brief Stores a three-dimensional orthonormal coordinate frame
     *
     * This class is mostly used to quickly convert between different
     * cartesian coordinate systems and to efficiently compute certain
     * quantities (e.g. \ref cosTheta(), \ref tanTheta, ..).
     */
    struct FrameGPU {
        VectorGPU3f s, t;
        NormalGPU3f n;

        /// Default constructor -- performs no initialization!
        CUDA_DEV 
        FrameGPU() { }

        /// Given a normal and tangent vectors, construct a new coordinate frame
        CUDA_DEV 
        FrameGPU(const VectorGPU3f &s, const VectorGPU3f &t, const NormalGPU3f &n)
        : s(s), t(t), n(n) { }

        /// Construct a frame from the given orthonormal vectors
        CUDA_DEV 
        FrameGPU(const VectorGPU3f &x, const VectorGPU3f &y, const VectorGPU3f &z)
        : s(x), t(y), n(z) { }

        /// Construct a new coordinate frame from a single vector
        
        CUDA_DEV 
        FrameGPU(const VectorGPU3f &n) : n(n) {
            coordinateSystem(n, s, t);
        }
    
        /// Convert from world coordinates to local coordinates
        CUDA_DEV 
        VectorGPU3f toLocal(const VectorGPU3f &v) const {
            return VectorGPU3f(
                v.dot(s), v.dot(t), v.dot(n)
            );
        }

        /// Convert from local coordinates to world coordinates
        CUDA_DEV 
        VectorGPU3f toWorld(const VectorGPU3f &v) const {
            return s * v.x() + t * v.y() + n * v.z();
        }

        CUDA_DEV
        void coordinateSystem(const VectorGPU3f &a, VectorGPU3f &b, VectorGPU3f &c) {
            if (fabsf(a.x()) > fabsf(a.y())) {
                float invLen = 1.0f / sqrtf(a.x() * a.x() + a.z() * a.z());
                c = VectorGPU3f(a.z() * invLen, 0.0f, -a.x() * invLen);
            } else {
                float invLen = 1.0f / sqrtf(a.y() * a.y() + a.z() * a.z());
                c = VectorGPU3f(0.0f, a.z() * invLen, -a.y() * invLen);
            }
            b = c.cross(a);
        }

        /** \brief Assuming that the given direction is in the local coordinate 
         * system, return the cosine of the angle between the normal and v */
        CUDA_DEV 
        float cosTheta(const VectorGPU3f &v) const {
            return v.z();
        }

        /** \brief Assuming that the given direction is in the local coordinate
         * system, return the sine of the angle between the normal and v */
        CUDA_DEV 
        float sinTheta(const VectorGPU3f &v) const {
            float temp = sinTheta2(v);
            if (temp <= 0.0f)
                return 0.0f;
            return sqrtf(temp);
        }

        /** \brief Assuming that the given direction is in the local coordinate
         * system, return the tangent of the angle between the normal and v */
        CUDA_DEV 
        float tanTheta(const VectorGPU3f &v) const {
            float temp = 1 - v.z()*v.z();
            if (temp <= 0.0f)
                return 0.0f;
            return sqrtf(temp) / v.z();
        }

        /** \brief Assuming that the given direction is in the local coordinate
         * system, return the squared sine of the angle between the normal and v */
        CUDA_DEV 
        float sinTheta2(const VectorGPU3f &v) const  {
            return 1.0f - v.z() * v.z();
        }

        /** \brief Assuming that the given direction is in the local coordinate 
         * system, return the sine of the phi parameter in spherical coordinates */
        CUDA_DEV 
        float sinPhi(const VectorGPU3f &v) const {
            float sinTheta = FrameGPU::sinTheta(v);
            if (sinTheta == 0.0f)
                return 1.0f;
            return clamp(v.y() / sinTheta, -1.0f, 1.0f);
        }

        /** \brief Assuming that the given direction is in the local coordinate 
         * system, return the cosine of the phi parameter in spherical coordinates */
        CUDA_DEV 
        float cosPhi(const VectorGPU3f &v) const{
            float sinTheta = FrameGPU::sinTheta(v);
            if (sinTheta == 0.0f)
                return 1.0f;
            return clamp(v.x() / sinTheta, -1.0f, 1.0f);
        }

        /** \brief Assuming that the given direction is in the local coordinate
         * system, return the squared sine of the phi parameter in  spherical
         * coordinates */
        CUDA_DEV 
        float sinPhi2(const VectorGPU3f &v) const {
            return clamp(v.y() * v.y() / sinTheta2(v), 0.0f, 1.0f);
        }

        /** \brief Assuming that the given direction is in the local coordinate
         * system, return the squared cosine of the phi parameter in  spherical
         * coordinates */
        CUDA_DEV 
        float cosPhi2(const VectorGPU3f &v)const {
            return clamp(v.x() * v.x() / sinTheta2(v), 0.0f, 1.0f);
        }

        /// Equality test
        CUDA_DEV
        bool operator==(const FrameGPU &frame) const {
            return frame.s == s && frame.t == t && frame.n == n;
        }

        /// Inequality test 
        CUDA_DEV
        bool operator!=(const FrameGPU &frame) const {
            return !operator==(frame);
        }
    };

}