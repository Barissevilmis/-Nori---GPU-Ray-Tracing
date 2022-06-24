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

/* =======================================================================
     This file contains classes for parallel rendering of "image blocks".
 * ======================================================================= */

#pragma once

#include <nori/colorgpu.h>
#include <nori/vectorgpu.h>
#include <tbb/mutex.h>
#include <nori/rfilter.h>

#define NORI_BLOCK_SIZE 32 /* Block size used for parallelization */
#define NORI_FILTER_RESOLUTION 32 

namespace nori{

    /**
     * \brief Weighted pixel storage for a rectangular subregion of an image
     *
     * This class implements storage for a rectangular subregion of a
     * larger image that is being rendered. For each pixel, it records color
     * values along with a weight that specifies the accumulated influence of
     * nearby samples on the pixel (according to the used reconstruction filter).
     *
     * When rendering with filters, the samples in a rectangular
     * region will generally also contribute to pixels just outside of 
     * this region. For that reason, this class also stores information about
     * a small border region around the rectangle, whose size depends on the
     * properties of the reconstruction filter.
     */
    class ImageBlock : public Eigen::Array<ColorGPU4f, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> {
    public:
        /**
         * Create a new image block of the specified maximum size
         * \param size
         *     Desired maximum size of the block
         * \param filter
         *     Samples will be convolved with the image reconstruction
         *     filter provided here.
         */
        ImageBlock(const VectorGPU2i &size);
        void AddGaussian(const GaussianFilter *filter);
        void AddMitchellNetravali(const MitchellNetravaliFilter *filter);
        void AddTent(const TentFilter *filter);
        void AddBox(const BoxFilter *filter);
        
        /// Release all memory
        ~ImageBlock();
        
        /// Configure the offset of the block within the main image
        void setOffset(const PointGPU2i &offset) { m_offset = offset; }

        /// Return the offset of the block within the main image
        inline const PointGPU2i &getOffset() const { return m_offset; }
        
        /// Configure the size of the block within the main image
        void setSize(const PointGPU2i &size) { *m_size = size; }

        /// Return the size of the block within the main image
        inline const VectorGPU2i &getSize() const { return *m_size; }

        /// Return the border size in pixels
        inline int getBorderSize() const { return m_borderSize; }
        

        /**
         * \brief Turn the block into a proper bitmap
         * 
         * This entails normalizing all pixels and discarding
         * the border region.
         */
        Bitmap *toBitmap() const;

        /// Convert a bitmap into an image block
        void fromBitmap(const Bitmap &bitmap);

        /// Clear all contents
        void clear() { setConstant(ColorGPU4f()); }

        ColorGPU4f* getImage();

        /// Record a sample with the given position and radiance value
        void putResult(ColorGPU4f* res_h);

        /**
         * \brief Merge another image block into this one
         *
         * During the merge operation, this function locks 
         * the destination block using a mutex.
         */
        void putHost(ImageBlock &b);

        float* getWeightsX() const{
            return m_weightsX;
        }
        float* getWeightsY() const{
            return m_weightsY;
        }

        VectorGPU2i *getFilterSize() const{
            VectorGPU2i *res = (VectorGPU2i*)malloc(sizeof(VectorGPU2i));
            *res = VectorGPU2i(weightSize, weightSize);
            return res;
        }

        float* getLookUpFactor() const{
            return m_lookupFactor;
        }

        bool hasFilter() const{
            return has_filter;
        }

        /// Lock the image block (using an internal mutex)
        inline void lock() const { m_mutex.lock(); }
        
        /// Unlock the image block
        inline void unlock() const { m_mutex.unlock(); }

        /// Return a human-readable string summary
        std::string toString() const;
    protected:
        PointGPU2i m_offset;
        VectorGPU2i *m_size;
        int m_borderSize = 0;
        int weightSize = 0;
        float *m_filter = nullptr;
        float m_filterRadius = 0;
        float *m_weightsX = nullptr;
        float *m_weightsY = nullptr;
        float *m_lookupFactor = nullptr;
        bool has_filter = false;
        mutable tbb::mutex m_mutex;
    };

    /**
     * \brief Spiraling block generator
     *
     * This class can be used to chop up an image into many small
     * rectangular blocks suitable for parallel rendering. The blocks
     * are ordered in spiraling pattern so that the center is
     * rendered first.
     */
    class BlockGenerator {
    public:
        /**
         * \brief Create a block generator with
         * \param size
         *      Size of the image that should be split into blocks
         * \param blockSize
         *      Maximum size of the individual blocks
         */
        BlockGenerator(const VectorGPU2i &size, int blockSize);
        
        /**
         * \brief Return the next block to be rendered
         *
         * This function is thread-safe
         *
         * \return \c false if there were no more blocks
         */
        bool next(ImageBlock &block);

        /// Return the total number of blocks
        int getBlockCount() const { return m_blocksLeft; }
    protected:
        enum EDirection { ERight = 0, EDown, ELeft, EUp };

        PointGPU2i m_block;
        VectorGPU2i m_numBlocks;
        VectorGPU2i m_size;
        int m_blockSize;
        int m_numSteps;
        int m_blocksLeft;
        int m_stepsLeft;
        int m_direction;
        tbb::mutex m_mutex;
    };

}
