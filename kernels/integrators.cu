#include <nori/integrator.h>
#include <nori/scenegpu.h>

namespace nori {

        CUDA_DEV 
        ColorGPU3f NormalKernel::Li(SceneKernel *scene, IndependentKernel* sampler, RayGPU3f* ray, IntersectionGPU *its, uint32_t* t_indices, uint32_t* m_indices, uint32_t * nT,  LeafNode*  leafNodes,InternalNode* internalNodes,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize, uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const{
            /* Find the surface that is visible in the requested direction */
            int indexX = *x_render_offset_dev + blockIdx.x * blockDim.x + threadIdx.x;
            int indexY = *y_render_offset_dev + blockIdx.y * blockDim.y + threadIdx.y;
            //printf("%f ::: %f\n", ray[resSize->y() * indexX + indexY].maxt, ray[resSize->y() * indexX + indexY].mint);
            bool isIntersect = scene->rayIntersect(ray, its, t_indices, m_indices, nT, leafNodes, internalNodes, v_V, v_N, v_UV, v_F, N_exists, UV_exists, resSize, x_render_offset_dev, y_render_offset_dev);
            if (!isIntersect){
                //printf("%d ------%d\n", indexX, indexY);
                return ColorGPU3f(0.f);
            }
            else
            {
                NormalGPU3f n = its[resSize->y() * indexX + indexY].shFrame.n.cwiseAbs();
                //printf("%d ------%d\n", indexX, indexY);
                return ColorGPU3f(n.x(), n.y(), n.z());
            }
        }

        CUDA_DEV 
        ColorGPU3f SimpleKernel::Li(SceneKernel *scene, IndependentKernel* sampler, RayGPU3f* ray, IntersectionGPU *its, uint32_t* t_indices, uint32_t* m_indices, uint32_t * nT,  LeafNode*  leafNodes,InternalNode* internalNodes,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize, uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const{
            /* Find the surface that is visible in the requested direction */
            int indexX = *x_render_offset_dev + blockIdx.x * blockDim.x + threadIdx.x;
            int indexY = *y_render_offset_dev + blockIdx.y * blockDim.y + threadIdx.y;
            //printf("%f ::: %f\n", ray[resSize->y() * indexX + indexY].maxt, ray[resSize->y() * indexX + indexY].mint);
            bool isIntersect = scene->rayIntersect(ray, its, t_indices, m_indices, nT, leafNodes, internalNodes, v_V, v_N, v_UV, v_F, N_exists, UV_exists, resSize, x_render_offset_dev, y_render_offset_dev);
            if (!isIntersect){
                //printf("%d ------%d\n", indexX, indexY);
                return ColorGPU3f(0.f);
            }
            else
            {
                VectorGPU3f diff_vec = position - its[resSize->y() * indexX + indexY].p;
                float diff_dist = diff_vec.squaredNorm();
                diff_vec.normalize();
                float cos_tht = diff_vec.dot(its[resSize->y() * indexX + indexY].shFrame.n);
                //RayGPU3f sray = RayGPU3f(its[resSize->y() * indexX + indexY].p, diff_vec);
              
                ColorGPU3f L = ColorGPU3f((energy* fmaxf(0, cos_tht))/ (4 * M_PI * M_PI *diff_dist));
                return L;
                
                
            }
        }


    //EClassType getClassType() const { return EIntegrator; }    
    NORI_REGISTER_CLASS(NormalIntegrator, "normals");
    NORI_REGISTER_CLASS(SimpleIntegrator, "simple");
}