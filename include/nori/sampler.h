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

#include <nori/object.h>
#include <memory>
#include <cuda.h>
#include <curand_kernel.h>




namespace nori{

    //class ImageBlockGPU;

    /**
     * \brief Abstract sample generator
     *
     * A sample generator is responsible for generating the random number stream
     * that will be passed an \ref Integrator implementation as it computes the
     * radiance incident along a specified ray.
     *
     * The most simple conceivable sample generator is just a wrapper around the
     * Mersenne-Twister random number generator and is implemented in
     * <tt>independent.cpp</tt> (it is named this way because it generates 
     * statistically independent random numbers).
     *
     * Fancier samplers might use stratification or low-discrepancy sequences
     * (e.g. Halton, Hammersley, or Sobol point sets) for improved convergence.
     * Another use of this class is in producing intentionally correlated
     * random numbers, e.g. as part of a Metropolis-Hastings integration scheme.
     *
     * The general interface between a sampler and a rendering algorithm is as 
     * follows: Before beginning to render a pixel, the rendering algorithm calls 
     * \ref generate(). The first pixel sample can now be computed, after which
     * \ref advance() needs to be invoked. This repeats until all pixel samples have
     * been exhausted.  While computing a pixel sample, the rendering 
     * algorithm requests (pseudo-) random numbers using the \ref next1D() and
     * \ref next2D() functions.
     *
     * Conceptually, the right way of thinking of this goes as follows:
     * For each sample in a pixel, a sample generator produces a (hypothetical)
     * point in an infinite dimensional random number hypercube. A rendering 
     * algorithm can then request subsequent 1D or 2D components of this point 
     * using the \ref next1D() and \ref next2D() functions. Fancy implementations
     * of this class make certain guarantees about the stratification of the 
     * first n components with respect to the other points that are sampled 
     * within a pixel.
     */
    inline cudaError_t checkCuda(cudaError_t result) {
        if (result != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
            assert(result == cudaSuccess);
        }
        return result;
    }
    class IndependentSampler: public NoriObject{
    friend class IndependentKernel;
    public:

        IndependentSampler(const PropertyList &propList) {
            m_sampleCount = (size_t*)(malloc(sizeof(size_t)));
            *m_sampleCount = (size_t) propList.getInteger("sampleCount", 1);
        }
        /// Release all memory
        ~IndependentSampler() { 
            free(m_sampleCount);
        }

        /**
         * \brief Prepare to render a new image block
         * 
         * This function is called when the sampler begins rendering
         * a new image block. This can be used to deterministically
         * initialize the sampler so that repeated program runs
         * always create the same image.
         */

        void prepare(){}

        size_t getSampleCount() { return *m_sampleCount; }
        

        /**
         * \brief Return the type of object (i.e. Mesh/Sampler/etc.) 
         * provided by this instance
         * */

        std::string toString() const{
            return "GPU-Sampler[Independent]";
        }

        EClassType getClassType() const { return ESampler; }
    protected:
        size_t *m_sampleCount;
    };

    class IndependentKernel{
    public:
        
        IndependentKernel(){}

        ~IndependentKernel(){}


        CUDA_DEV
        float next1D(const VectorGPU2i * resSize, uint32_t* x_render_offset_dev, uint32_t* y_render_offset_dev);

        /// Retrieve the next two component values from the current sample
        CUDA_DEV
        PointGPU2f next2D(const VectorGPU2i * resSize, uint32_t* x_render_offset_dev, uint32_t* y_render_offset_dev);



    };

}
