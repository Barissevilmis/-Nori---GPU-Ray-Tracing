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

#include <nori/sampler.h>

namespace nori{
   
    CUDA_DEV 
    float IndependentKernel::next1D(const VectorGPU2i * resSize, uint32_t* x_render_offset_dev, uint32_t* y_render_offset_dev) {
        curandStateMtgp32_t m_state;
        int indexX = *x_render_offset_dev + blockIdx.x * blockDim.x + threadIdx.x;
        int indexY = *y_render_offset_dev + blockIdx.y * blockDim.y + threadIdx.y;
        int tid = indexY + (resSize->y() * indexX);
        return curand_uniform(&m_state);
    }
        
    CUDA_DEV 
    PointGPU2f IndependentKernel::next2D(const VectorGPU2i * resSize, uint32_t* x_render_offset_dev, uint32_t* y_render_offset_dev) {
        curandStateMtgp32_t m_state;
        int indexX = *x_render_offset_dev + blockIdx.x * blockDim.x + threadIdx.x;
        int indexY = *y_render_offset_dev + blockIdx.y * blockDim.y + threadIdx.y;
        int tid = indexY + (resSize->y() * indexX);
        return PointGPU2f(curand_uniform(&m_state), curand_uniform(&m_state));
    }


    NORI_REGISTER_CLASS(IndependentSampler, "independent");
}
