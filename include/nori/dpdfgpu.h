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
#include <cuda_runtime.h>
#include <cuda.h>

namespace nori{
    /**
     * \brief Discrete probability distribution
     * 
     * This data structure can be used to transform uniformly distributed
     * samples to a stored discrete probability distribution.
     * 
     * \ingroup libcore
     */
    struct DiscretePDFGPU {
    public:
        /// Allocate memory for a distribution with the given number of entries
        explicit DiscretePDFGPU(size_t nEntries = 0) {
            cudaMalloc((void **)&m_cdf, (nEntries+1)*sizeof(float));
            m_cdf[0] = 0.f;
            curr_size = 1;
        }
        ~DiscretePDFGPU(){
        cudaFree((void **)&m_cdf);
        }

        /// Append an entry with the specified discrete probability
        void append(float pdfValue) {
            m_cdf[curr_size] =  m_cdf[curr_size-1] + pdfValue;
            curr_size++;   
        }

        /// Return the number of entries so far
        CUDA_DEV
        size_t size() const {
            return curr_size;
        }

        /// Access an entry by its index
        CUDA_DEV
        float operator[](size_t entry) const {
            return m_cdf[entry+1] - m_cdf[entry];
        }

        /// Have the probability densities been normalized?
        CUDA_DEV
        bool isNormalized() const {
            return m_normalized;
        }

        /**
         * \brief Return the original (unnormalized) sum of all PDF entries
         *
         * This assumes that \ref normalize() has previously been called
         */
        CUDA_DEV 
        float getSum() const {
            return m_sum;
        }

        /**
         * \brief Return the normalization factor (i.e. the inverse of \ref getSum())
         *
         * This assumes that \ref normalize() has previously been called
         */
        CUDA_DEV 
        float getNormalization() const {
            return m_normalization;
        }

        /**
         * \brief Normalize the distribution
         *
         * \return Sum of the (previously unnormalized) entries
         */
        float normalize() {
            m_sum = m_cdf[curr_size-1];
            if (m_sum > 0) {
                m_normalization = 1.0f / m_sum;
                for (size_t i=1; i<curr_size; ++i) 
                    m_cdf[i] *= m_normalization;
                m_cdf[curr_size-1] = 1.0f;
                m_normalized = true;
            } else {
                m_normalization = 0.0f;
            }
            return m_sum;
        }

        /**
         * \brief %Transform a uniformly distributed sample to the stored distribution
         * 
         * \param[in] sampleValue
         *     An uniformly distributed sample on [0,1]
         * \return
         *     The discrete index associated with the sample
         */
        CUDA_DEV
        uint64_t sample(float sampleValue) const {
            uint64_t entry = 0;
            for(uint64_t i = 0; curr_size; i++){
                if(sampleValue > entry){
                    entry = i;
                }
            }
            if(!entry)
                entry = curr_size - 1;

            uint64_t index = (uint64_t) fmaxf(0, entry - 1);
            return (uint64_t) fminf(index, curr_size-2);
        }

        /**
         * \brief %Transform a uniformly distributed sample to the stored distribution
         * 
         * \param[in] sampleValue
         *     An uniformly distributed sample on [0,1]
         * \param[out] pdf
         *     Probability value of the sample
         * \return
         *     The discrete index associated with the sample
         */
        CUDA_DEV 
        uint64_t sample(float sampleValue, float &pdf) const {
            uint64_t index = sample(sampleValue);
            pdf = operator[](index);
            return index;
        }

        /**
         * \brief %Transform a uniformly distributed sample to the stored distribution
         * 
         * The original sample is value adjusted so that it can be "reused".
         *
         * \param[in, out] sampleValue
         *     An uniformly distributed sample on [0,1]
         * \return
         *     The discrete index associated with the sample
         */
        CUDA_DEV 
        size_t sampleReuse(float &sampleValue) const {
            size_t index = sample(sampleValue);
            sampleValue = (sampleValue - m_cdf[index])
                / (m_cdf[index + 1] - m_cdf[index]);
            return index;
        }

        /**
         * \brief %Transform a uniformly distributed sample. 
         * 
         * The original sample is value adjusted so that it can be "reused".
         *
         * \param[in,out]
         *     An uniformly distributed sample on [0,1]
         * \param[out] pdf
         *     Probability value of the sample
         * \return
         *     The discrete index associated with the sample
         */
        CUDA_DEV 
        size_t sampleReuse(float &sampleValue, float &pdf) const {
            size_t index = sample(sampleValue, pdf);
            sampleValue = (sampleValue - m_cdf[index])
                / (m_cdf[index + 1] - m_cdf[index]);
            return index;
        }

    private:
        float* m_cdf;
        float m_sum, m_normalization;
        uint64_t curr_size;
        bool m_normalized;
    };

}
