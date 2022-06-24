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

#include <nori/block.h>
#include <nori/bitmap.h>
#include <nori/rfilter.h>
#include <tbb/tbb.h>

namespace nori{

    ImageBlock::ImageBlock(const VectorGPU2i &size) 
            : m_offset(0, 0) {
                m_size = (VectorGPU2i*)malloc(sizeof(VectorGPU2i));
                *m_size = size;
            /* Allocate space for pixels and border regions */
            resize(m_size->y() + 2*m_borderSize, m_size->x() + 2*m_borderSize);
            }
       


    void ImageBlock::AddGaussian(const GaussianFilter *filter){
        if (filter) {
            /* Tabulate the image reconstruction filter for performance reasons */
            has_filter = true;
            m_filterRadius = filter->getRadius();
            m_borderSize = (int) std::ceil(m_filterRadius - 0.5f);
            m_filter = new float[NORI_FILTER_RESOLUTION + 1];
            m_lookupFactor = new float();
            for (int i=0; i<NORI_FILTER_RESOLUTION; ++i) {
                float pos = (m_filterRadius * i) / NORI_FILTER_RESOLUTION;
                m_filter[i] = filter->eval(pos);
            }
            m_filter[NORI_FILTER_RESOLUTION] = 0.0f;
            *m_lookupFactor = NORI_FILTER_RESOLUTION / m_filterRadius;
            weightSize = (int) std::ceil(2*m_filterRadius) + 1;
            m_weightsX = new float[weightSize];
            m_weightsY = new float[weightSize];
            memset(m_weightsX, 0, sizeof(float) * weightSize);
            memset(m_weightsY, 0, sizeof(float) * weightSize);
        }

        /* Allocate space for pixels and border regions */
        resize(m_size->y() + 2*m_borderSize, m_size->x() + 2*m_borderSize);

    }

    void ImageBlock::AddMitchellNetravali(const MitchellNetravaliFilter *filter){
        if (filter) {
            /* Tabulate the image reconstruction filter for performance reasons */
            has_filter = true;
            m_filterRadius = filter->getRadius();
            m_borderSize = (int) std::ceil(m_filterRadius - 0.5f);
            m_filter = new float[NORI_FILTER_RESOLUTION + 1];
            m_lookupFactor = new float();
            for (int i=0; i<NORI_FILTER_RESOLUTION; ++i) {
                float pos = (m_filterRadius * i) / NORI_FILTER_RESOLUTION;
                m_filter[i] = filter->eval(pos);
            }
            m_filter[NORI_FILTER_RESOLUTION] = 0.0f;
            *m_lookupFactor = NORI_FILTER_RESOLUTION / m_filterRadius;
            weightSize = (int) std::ceil(2*m_filterRadius) + 1;
            m_weightsX = new float[weightSize];
            m_weightsY = new float[weightSize];
            memset(m_weightsX, 0, sizeof(float) * weightSize);
            memset(m_weightsY, 0, sizeof(float) * weightSize);
        }
    }

    void ImageBlock::AddTent(const TentFilter *filter){
        if (filter) {
            /* Tabulate the image reconstruction filter for performance reasons */
            has_filter = true;
            m_filterRadius = filter->getRadius();
            m_borderSize = (int) std::ceil(m_filterRadius - 0.5f);
            m_filter = new float[NORI_FILTER_RESOLUTION + 1];
            m_lookupFactor = new float();
            for (int i=0; i<NORI_FILTER_RESOLUTION; ++i) {
                float pos = (m_filterRadius * i) / NORI_FILTER_RESOLUTION;
                m_filter[i] = filter->eval(pos);
            }
            m_filter[NORI_FILTER_RESOLUTION] = 0.0f;
            *m_lookupFactor = NORI_FILTER_RESOLUTION / m_filterRadius;
            weightSize = (int) std::ceil(2*m_filterRadius) + 1;
            m_weightsX = new float[weightSize];
            m_weightsY = new float[weightSize];;
            memset(m_weightsX, 0, sizeof(float) * weightSize);
            memset(m_weightsY, 0, sizeof(float) * weightSize);
        }

    }

    void ImageBlock::AddBox(const BoxFilter *filter){
        if (filter) {
            /* Tabulate the image reconstruction filter for performance reasons */
            has_filter = true;
            m_filterRadius = filter->getRadius();
            m_borderSize = (int) std::ceil(m_filterRadius - 0.5f);
            m_filter = new float[NORI_FILTER_RESOLUTION + 1];
            m_lookupFactor = new float();
            for (int i=0; i<NORI_FILTER_RESOLUTION; ++i) {
                float pos = (m_filterRadius * i) / NORI_FILTER_RESOLUTION;
                m_filter[i] = filter->eval(pos);
            }
            m_filter[NORI_FILTER_RESOLUTION] = 0.0f;
            *m_lookupFactor = NORI_FILTER_RESOLUTION / m_filterRadius;
            weightSize = (int) std::ceil(2*m_filterRadius) + 1;
            m_weightsX = new float[weightSize];
            m_weightsY = new float[weightSize];
            memset(m_weightsX, 0, sizeof(float) * weightSize);
            memset(m_weightsY, 0, sizeof(float) * weightSize);
        }

    }

    ImageBlock::~ImageBlock() {
        delete m_size;
        if(hasFilter()){
            delete[] m_filter;
            delete[] m_weightsX;
            delete[] m_weightsY;
            delete m_lookupFactor;
        }
    }

    Bitmap *ImageBlock::toBitmap() const {
        Bitmap *result = new Bitmap(*m_size);
        for (int y=0; y<m_size->y(); ++y)
            for (int x=0; x<m_size->x(); ++x)
                result->coeffRef(y, x) = coeff(y + m_borderSize, x + m_borderSize).divideByFilterWeight();
        return result;
    }

    void ImageBlock::fromBitmap(const Bitmap &bitmap) {
        if (bitmap.cols() != cols() || bitmap.rows() != rows())
            throw NoriException("Invalid bitmap dimensions!");

        for (int y=0; y<m_size->y(); ++y)
            for (int x=0; x<m_size->x(); ++x)
                coeffRef(y, x) << bitmap.coeff(y, x), 1;
    }
    ColorGPU4f* ImageBlock::getImage(){
        ColorGPU4f * value=(ColorGPU4f*)malloc(m_size->x()*m_size->y()*sizeof(ColorGPU4f)); 
        for (int y=0; y < m_size->y(); y++) 
            for (int x=0; x < m_size->x(); x++) 
                value[m_size->y() * x + y] = coeffRef(y, x);
        return value;
    }

    void ImageBlock::putResult(ColorGPU4f* res_h) {
        //printf("%f ------- %f\n", res_h[1].x(), res_h[1].y());
        for (int y=0; y <= m_size->y(); y++){ 
            for (int x=0; x <= m_size->x(); x++){
                coeffRef(y, x) += res_h[m_size->y() * x + y];
            }
        }
    }
        
    void ImageBlock::putHost(ImageBlock &b) {
        VectorGPU2i offset = b.getOffset() - m_offset +
            VectorGPU2i::Constant(m_borderSize - b.getBorderSize());
        VectorGPU2i size   = b.getSize()   + VectorGPU2i(2*b.getBorderSize());

        tbb::mutex::scoped_lock lock(m_mutex);

        block(offset.y(), offset.x(), size.y(), size.x()) 
            += b.topLeftCorner(size.y(), size.x());
    }

    std::string ImageBlock::toString() const {
        return "ImageBlock";
    }

    BlockGenerator::BlockGenerator(const VectorGPU2i &size, int blockSize)
            : m_size(size), m_blockSize(blockSize) {
        m_numBlocks = VectorGPU2i(
            (int) std::ceil(size.x() / (float) blockSize),
            (int) std::ceil(size.y() / (float) blockSize));
        m_blocksLeft = m_numBlocks.x() * m_numBlocks.y();
        m_direction = ERight;
        m_block = PointGPU2i(m_numBlocks / 2);
        m_stepsLeft = 1;
        m_numSteps = 1;
    }

    bool BlockGenerator::next(ImageBlock &block) {
        tbb::mutex::scoped_lock lock(m_mutex);

        if (m_blocksLeft == 0)
            return false;

        PointGPU2i pos = m_block * m_blockSize;
        block.setOffset(pos);
        block.setSize((m_size - pos).cwiseMin(VectorGPU2i::Constant(m_blockSize)));

        if (--m_blocksLeft == 0)
            return true;

        do {
            switch (m_direction) {
                case ERight: ++m_block.x(); break;
                case EDown:  ++m_block.y(); break;
                case ELeft:  --m_block.x(); break;
                case EUp:    --m_block.y(); break;
            }

            if (--m_stepsLeft == 0) {
                m_direction = (m_direction + 1) % 4;
                if (m_direction == ELeft || m_direction == ERight) 
                    ++m_numSteps;
                m_stepsLeft = m_numSteps;
            }
        } while ((m_block.array() < 0).any() ||
                (m_block.array() >= m_numBlocks.array()).any());

        return true;
    }

}
