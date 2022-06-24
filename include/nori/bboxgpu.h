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

#include <nori/raygpu.h>

namespace nori{

    
    /**
     * \brief Generic n-dimensional bounding box data structure
     *
     * Maintains a minimum and maximum position along each dimension and provides
     * various convenience functions for querying and modifying them.
     *
     * This class is parameterized by the underlying point data structure,
     * which permits the use of different scalar types and dimensionalities, e.g.
     * \code
     * TBoundingBox<Vector3i> integerBBox(Point3i(0, 1, 3), Point3i(4, 5, 6));
     * TBoundingBox<Vector2d> floatBBox(Point2d(0.0, 1.0), Point2d(4.0, 5.0));
     * \endcode
     *
     * \tparam T The underlying point data type (e.g. \c Point2d)
     * \ingroup libcore
     */
    template <typename _PointType>
    struct TBoundingBoxGPU {
        enum {
            Dimension = _PointType::Dimension
        };

        typedef _PointType                             PointType;
        typedef typename PointType::Scalar             Scalar;
        typedef typename PointType::VectorType         VectorType;

        /** 
         * \brief Create a new invalid bounding box
         * 
         * Initializes the components of the minimum 
         * and maximum position to \f$\infty\f$ and \f$-\infty\f$,
         * respectively.
         */
        CUDA_HOSTDEV
        TBoundingBoxGPU() {
            reset();
        }

        /// Create a collapsed bounding box from a single point
        CUDA_HOSTDEV
        TBoundingBoxGPU(const PointType &p) 
            : min(p), max(p) {}

        /// Create a bounding box from two positions
        CUDA_HOSTDEV
        TBoundingBoxGPU(const PointType &min, const PointType &max)
            : min(min), max(max) {}

        CUDA_HOSTDEV
        bool operator==(const TBoundingBoxGPU &bbox) const {
            return min == bbox.min && max == bbox.max;
        }

        /// Test for inequality against another bounding box
        CUDA_HOSTDEV
        bool operator!=(const TBoundingBoxGPU &bbox) const {
            return min != bbox.min || max != bbox.max;
        }

        /// Calculate the n-dimensional volume of the bounding box
        CUDA_HOSTDEV Scalar getVolume() const {
            return (max - min).prod();
        }

        /// Calculate the n-1 dimensional volume of the boundary
        CUDA_HOSTDEV 
        float getSurfaceArea() const {
            VectorType d = max - min;
            float result = 0.0f;
            for (int i=0; i<Dimension; ++i) {
                float term = 1.0f;
                for (int j=0; j<Dimension; ++j) {
                    if (i == j)
                        continue;
                    term *= d[j];
                }
                result += term;
            }
            return 2.0f * result;
        }

        /// Return the center point
        CUDA_HOSTDEV 
        PointType getCenter() const {
            return (max + min) * (Scalar) 0.5f;
        }
        /**
         * \brief Check whether a point lies \a on or \a inside the bounding box
         *
         * \param p The point to be tested
         *
         * \param strict Set this parameter to \c true if the bounding
         *               box boundary should be excluded in the test
         */
        CUDA_HOSTDEV 
        bool contains(const PointType &p, bool strict = false) const {
            if (strict) {
                return (p.array() > min.array()).all() 
                    && (p.array() < max.array()).all();
            } else {
                return (p.array() >= min.array()).all() 
                    && (p.array() <= max.array()).all();
            }
        }

        /**
         * \brief Check whether a specified bounding box lies \a on or \a within 
         * the current bounding box
         *
         * Note that by definition, an 'invalid' bounding box (where min=\f$\infty\f$
         * and max=\f$-\infty\f$) does not cover any space. Hence, this method will always 
         * return \a true when given such an argument.
         *
         * \param strict Set this parameter to \c true if the bounding
         *               box boundary should be excluded in the test
         */
        CUDA_HOSTDEV 
        bool contains(const TBoundingBoxGPU &bbox, bool strict = false) const {
            if (strict) {
                return (bbox.min.array() > min.array()).all() 
                    && (bbox.max.array() < max.array()).all();
            } else {
                return (bbox.min.array() >= min.array()).all() 
                    && (bbox.max.array() <= max.array()).all();
            }
        }

        /**
         * \brief Check two axis-aligned bounding boxes for possible overlap.
         *
         * \param strict Set this parameter to \c true if the bounding
         *               box boundary should be excluded in the test
         *
         * \return \c true If overlap was detected.
         */
        CUDA_HOSTDEV 
        bool overlaps(const TBoundingBoxGPU &bbox, bool strict = false) const {
            if (strict) {
                return (bbox.min.array() < max.array()).all() 
                    && (bbox.max.array() > min.array()).all();
            } else {
                return (bbox.min.array() <= max.array()).all() 
                    && (bbox.max.array() >= min.array()).all();
            }
        }

        /**
         * \brief Calculate the smallest squared distance between
         * the axis-aligned bounding box and the point \c p.
         */
        CUDA_HOSTDEV 
        Scalar squaredDistanceTo(const PointType &p) const {
            Scalar result = 0;

            for (int i=0; i<Dimension; ++i) {
                Scalar value = 0;
                if (p[i] < min[i])
                    value = min[i] - p[i];
                else if (p[i] > max[i])
                    value = p[i] - max[i];
                result += value*value;
            }

            return result;
        }

        /**
         * \brief Calculate the smallest distance between
         * the axis-aligned bounding box and the point \c p.
         */
        CUDA_HOSTDEV 
        Scalar distanceTo(const PointType &p) const {
            return sqrtf(squaredDistanceTo(p));
        }
    
        /**
         * \brief Calculate the smallest square distance between
         * the axis-aligned bounding box and \c bbox.
         */
        CUDA_HOSTDEV 
        Scalar squaredDistanceTo(const TBoundingBoxGPU &bbox) const {
            Scalar result = 0;

            for (int i=0; i<Dimension; ++i) {
                Scalar value = 0;
                if (bbox.max[i] < min[i])
                    value = min[i] - bbox.max[i];
                else if (bbox.min[i] > max[i])
                    value = bbox.min[i] - max[i];
                result += value*value;
            }

            return result;
        }

        /**
         * \brief Calculate the smallest distance between
         * the axis-aligned bounding box and \c bbox.
         */
        CUDA_HOSTDEV 
        Scalar distanceTo(const TBoundingBoxGPU &bbox) const {
            return sqrtf(squaredDistanceTo(bbox));
        }

        /**
         * \brief Check whether this is a valid bounding box
         *
         * A bounding box \c bbox is valid when
         * \code
         * bbox.min[dim] <= bbox.max[dim]
         * \endcode
         * holds along each dimension \c dim.
         */
        CUDA_HOSTDEV 
        bool isValid() const {
            return (max.array() >= min.array()).all();
        }

        /// Check whether this bounding box has collapsed to a single point
        CUDA_HOSTDEV 
        bool isPoint() const {
            return (max.array() == min.array()).all();
        }

        /// Check whether this bounding box has any associated volume
        CUDA_HOSTDEV 
        bool hasVolume() const {
            return (max.array() > min.array()).all();
        }

        CUDA_HOSTDEV
        int getMajorAxis() const {
            VectorType d = max - min;
            int largest = 0;
            for (int i=1; i<Dimension; ++i)
                if (d[i] > d[largest])
                    largest = i;
            return largest;
        }
        

        /// Return the dimension index with the shortest associated side length
        CUDA_HOSTDEV 
        int getMinorAxis() const {
            VectorType d = max - min;
            int shortest = 0;
            for (int i=1; i<Dimension; ++i)
                if (d[i] < d[shortest])
                    shortest = i;
            return shortest;
        }

        /**
         * \brief Calculate the bounding box extents
         * \return max-min
         */
        CUDA_HOSTDEV 
        VectorType getExtents() const {
            return max - min;
        }

        /// Clip to another bounding box
        CUDA_HOSTDEV 
        void clip(const TBoundingBoxGPU &bbox) {
            min = min.cwiseMax(bbox.min);
            max = max.cwiseMin(bbox.max);
        }

        /** 
         * \brief Mark the bounding box as invalid.
         * 
         * This operation sets the components of the minimum 
         * and maximum position to \f$\infty\f$ and \f$-\infty\f$,
         * respectively.
         */
        CUDA_HOSTDEV 
        void reset() {
            min.setConstant(1.0e308);
            max.setConstant(-1.0e308);
        }

        /// Expand the bounding box to contain another point
        CUDA_HOSTDEV 
        void expandBy(const PointType &p) {
            min = min.cwiseMin(p);
            max = max.cwiseMax(p);
        }

        /// Expand the bounding box to contain another bounding box
        CUDA_DEV 
        void expandBy(const TBoundingBoxGPU &bbox) {
            //printf("%f b:b %f b:b %f\n",min.x(), min.y(), min.z());
            //printf("%f d:d %f d:d %f\n",bbox.min.x(), bbox.min.y(), bbox.min.z());
            min = min.cwiseMin(bbox.min);
            //printf("%f ::: %f ::: %f\n",min.x(), min.y(), min.z());
            max = max.cwiseMax(bbox.max);
        }

        /// Merge two bounding boxes
        CUDA_HOSTDEV 
        TBoundingBoxGPU merge(const TBoundingBoxGPU &bbox1, const TBoundingBoxGPU &bbox2) {
            return TBoundingBoxGPU(
                bbox1.min.cwiseMin(bbox2.min),
                bbox1.max.cwiseMax(bbox2.max)
            );
        }

        /// Return the index of the largest axis
        CUDA_HOSTDEV
        int getLargestAxis() const {
            VectorType extents = max-min;

            if (extents[0] >= extents[1] && extents[0] >= extents[2])
                return 0;
            else if (extents[1] >= extents[0] && extents[1] >= extents[2])
                return 1;
            else
                return 2;
        }

        /// Return the position of a bounding box corner
        CUDA_HOSTDEV 
        PointType getCorner(int index) const {
            PointType result;
            for (int i=0; i<Dimension; ++i)
                result[i] = (index & (1 << i)) ? max[i] : min[i];
            return result;
        }

        /// Check if a ray intersects a bounding box
        CUDA_DEV 
        bool rayIntersect(RayGPU3f *ray, const VectorGPU2i *resSize, uint32_t indexX, uint32_t indexY) const {
            float nearT = -1.0e308;
            float farT = 1.0e308;
            //printf("%f ::: %f ::: %f\n",min[0],min[1], min[2]);
            //printf("%f ttt %f ttt %f\n",max[0],max[1], max[2]);
            for (int i=0; i<3; i++) {
                float origin = ray[resSize->y() * indexX + indexY].o[i];
                float minVal = min[i], maxVal = max[i];
                //printf("%f :: %f\n",minVal, maxVal);
                if (ray[resSize->y() * indexX + indexY].d[i] == 0) {
                    if (origin < minVal || origin > maxVal)
                        return false;
                } else {
                    float t1 = (minVal - origin) * ray[resSize->y() * indexX + indexY].dRcp[i];
                    float t2 = (maxVal - origin) * ray[resSize->y() * indexX + indexY].dRcp[i];

                    if (t1 > t2)
                    {
                        //swap(t1, t2);
                        float tmp = t1;
                        t1 = t2;
                        t2 = tmp;
                    }

                    nearT = fmaxf(t1, nearT);
                    farT = fminf(t2, farT);
                    if (!(nearT <= farT))
                        return false;
                }
            }
            bool res = ray[resSize->y() * indexX + indexY].mint <= farT && nearT <= ray[resSize->y() * indexX + indexY].maxt;
            //printf("%d\n",res);
            return res;
        }
        CUDA_HOSTDEV
        PointType getMin() const{
            return min;
        }

        CUDA_HOSTDEV
        PointType getMax() const{
            return max;
        }
        /// Return the overlapping region of the bounding box and an unbounded ray
      

        PointType min; ///< Component-wise minimum 
        PointType max; ///< Component-wise maximum 
    };

}
