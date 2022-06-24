#include <nori/accelbvh.h>
#include <nori/scenegpu.h>
#include <nori/colorgpu.h>

namespace nori {
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess) 
        {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
    class SceneKernel;
    struct minAccessor{
            
            CUDA_HOSTDEV
            VectorGPU3f operator () (const BoundingBoxGPU3f& a){
                return a.min;
            }
        };

        struct minFunctor{
            CUDA_HOSTDEV
            VectorGPU3f operator () (const VectorGPU3f& a, const VectorGPU3f& b){
                return VectorGPU3f(fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z()));
            }
        };
        struct maxAccessor{
            
            CUDA_HOSTDEV
            VectorGPU3f operator () (const BoundingBoxGPU3f& a){
                return a.max;
            }
        };

        struct maxFunctor{
            CUDA_HOSTDEV
            VectorGPU3f operator () (const VectorGPU3f& a, const VectorGPU3f& b){
                return VectorGPU3f(fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z()));
            }
        };


    void BVH_d::findMinMax(VectorGPU3f& mMin, VectorGPU3f& mMax){
        thrust::device_ptr<BoundingBoxGPU3f> dvp(Bboxs);
        mMin =  thrust::transform_reduce(dvp, 
                dvp + *numTriangles_h, 
                minAccessor(),
                VectorGPU3f(1e9, 1e9, 1e9), 
                minFunctor());
        mMax =  thrust::transform_reduce(dvp, 
                dvp + *numTriangles_h,
                maxAccessor(), 
                VectorGPU3f(-1e9, -1e9, -1e9),
                maxFunctor());
    }

    //class BVH_d;
    // Expands a 10-bit integer into 30 bits
    // // by inserting 2 zeros after each bit.
    CUDA_DEV
    uint32_t expandBits(uint32_t v);

    // Calculates a 30-bit Morton code for the
    // given 3D point located within the unit cube [0,1].
    CUDA_DEV
    uint32_t morton3D(float x, float y, float z);


    
    CUDA_DEV
    int findSplit(uint32_t* sortedMortonCodes,
            int first,
            int last);

    CUDA_DEV
    int2 determineRange(uint32_t* sortedMortonCodes, uint32_t* numTriangles, int idx);

    __global__ 
    void computeMortonCodesKernel(uint32_t* mortonCodes, uint32_t* object_ids, 
            BoundingBoxGPU3f* BBoxs, uint32_t* numTriangles, const VectorGPU3f & mMin, const VectorGPU3f &mMax);
    __global__ 
    void setupLeafNodesKernel(uint32_t* sorted_object_ids, 
            LeafNode* leafNodes, BoundingBoxGPU3f* bboxes, uint32_t* numTriangles);

    __global__ 
    void computeBBoxesKernel( LeafNode* leafNodes,
            InternalNode* internalNodes,
            uint32_t *numTriangles);

    __global__ 
    void generateHierarchyKernel(uint32_t* mortonCodes,
            uint32_t* sorted_object_ids, 
            InternalNode* internalNodes,
            LeafNode* leafNodes, uint32_t *numTriangles, BoundingBoxGPU3f* BBoxs);


    //===========Begin KERNELS=============================
    // This kernel just computes the object id and morton code for the centroid of each bounding box
    __global__ 
    void computeMortonCodesKernel(uint32_t* mortonCodes, uint32_t* object_ids, 
            BoundingBoxGPU3f* BBoxs, uint32_t *numTriangles,  VectorGPU3f mMin , VectorGPU3f mMax, uint32_t *t_indices, uint32_t *m_indices){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= *numTriangles)
            return;
        object_ids[idx] = idx;
        PointGPU3f centroid = (BBoxs[idx].min + BBoxs[idx].max)/2.f;
        //printf("%f-%f-%f\n",centroid.x(), centroid.y(), centroid.z());
        centroid.x() = (centroid.x() - mMin.x())/(mMax.x() - mMin.x());
        centroid.y() = (centroid.y() - mMin.y())/(mMax.y() - mMin.y());
        centroid.z() = (centroid.z() - mMin.z())/(mMax.z() - mMin.z());
        //map this centroid to unit cube,
        mortonCodes[idx] = morton3D(centroid.x(), centroid.y(), centroid.z());
        //printf("in computeMortonCodesKernel: idx->%d , mortonCode->%d, centroid(%0.6f,%0.6f,%0.6f)\n", idx, mortonCodes[idx], centroid.x(), centroid.y(), centroid.z());

    };

    __global__ 
    void setupLeafNodesKernel(uint32_t* sorted_object_ids, 
            LeafNode* leafNodes, BoundingBoxGPU3f* bboxes, uint32_t *numTriangles){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= *numTriangles)
            return;
        leafNodes[idx].isLeaf = true;
        leafNodes[idx].object_id = sorted_object_ids[idx];
        leafNodes[idx].childA = nullptr;
        leafNodes[idx].childB = nullptr;
        leafNodes[idx].BBox = BoundingBoxGPU3f(bboxes[sorted_object_ids[idx]].min,bboxes[sorted_object_ids[idx]].max);
    }

    __global__ 
    void computeBBoxesKernel( LeafNode* leafNodes, InternalNode* internalNodes, uint32_t *numTriangles)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= *numTriangles)
            return;

        Node* Parent = leafNodes[idx].parent;
        while(Parent)
        {
            if(atomicCAS(&(Parent->flag), 0 , 1))
            {
                Parent->BBox.expandBy(Parent->childA->BBox);
                Parent->BBox.expandBy(Parent->childB->BBox);
                //Parent->BBox = Parent->BBox.merge(Parent->childA->BBox, Parent->childB->BBox);   
                Parent = Parent->parent;
            }
            else{
                return;
            }
        }
    }

    __global__ 
    void generateHierarchyKernel(uint32_t* sortedMortonCodes,
            uint32_t* sorted_object_ids, 
            InternalNode* internalNodes,
            LeafNode* leafNodes, uint32_t* numTriangles, BoundingBoxGPU3f* BBoxs){

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx > *numTriangles - 2 ) //there are n - 1 internal nodes
            return;

        internalNodes[idx].isLeaf = false;
        internalNodes[idx].object_id = -1;
        internalNodes[idx].BBox.reset();

        int2 range = determineRange(sortedMortonCodes, numTriangles, idx);
        int first = range.x;
        int last = range.y;

        //Determine where to split the range.

        int split = findSplit(sortedMortonCodes, first, last);

        // Select childA.

        Node* childA;
        if (split == first)
        {
            childA = &leafNodes[split];
            //childA->BBox = BoundingBoxGPU3f(BBoxs[split].min, BBoxs[split].max);

        }
        else
            childA = &internalNodes[split];

        // Select childB.

        Node* childB;
        if (split + 1 == last)
        {
            childB = &leafNodes[split + 1];
            //childB->BBox = BoundingBoxGPU3f(BBoxs[split + 1].min, BBoxs[split+1].max);
        }
        else
            childB = &internalNodes[split + 1];

        // Record parent-child relationships.
        //printf("%d\n", idx);
        internalNodes[idx].childA = childA;
        internalNodes[idx].childB = childB;
        childA->parent = &internalNodes[idx];
        childB->parent = &internalNodes[idx];

    }
    //===========END KERNELS=============================

    CUDA_DEV
    int findSplit( uint32_t* sortedMortonCodes,
            int first,
            int last)
    {
        // Identical Morton codes => split the range in the middle.
        uint32_t firstCode = sortedMortonCodes[first];
        uint32_t lastCode = sortedMortonCodes[last];

        if (firstCode == lastCode)
            return (first + last) >> 1;

        // Calculate the number of highest bits that are the same
        // for all objects, using the count-leading-zeros intrinsic.

        int commonPrefix = __clz(firstCode ^ lastCode);

        // Use binary search to find where the next bit differs.
        // Specifically, we are looking for the highest object that
        // shares more than commonPrefix bits with the first one.

        int split = first; // initial guess
        int step = last - first;

        do
        {
            step = (step + 1) >> 1; // exponential decrease
            int newSplit = split + step; // proposed new position

            if (newSplit < last)
            {
                uint32_t splitCode = sortedMortonCodes[newSplit];
                int splitPrefix = __clz(firstCode ^ splitCode);
                if (splitPrefix > commonPrefix)
                    split = newSplit; // accept proposal
            }
        }
        while (step > 1);

        return split;
    }

    CUDA_DEV
    int2 determineRange(uint32_t* sortedMortonCodes, uint32_t *numTriangles, int idx)
    {
    //determine the range of keys covered by each internal node (as well as its children)
        //direction is found by looking at the neighboring keys ki-1 , ki , ki+1
        //the index is either the beginning of the range or the end of the range
        int direction = 0;
        int common_prefix_with_left = 0;
        int common_prefix_with_right = 0;

        common_prefix_with_right = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx + 1]);
        if(idx == 0){
            common_prefix_with_left = -1;
        }
        else
        {
            common_prefix_with_left = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - 1]);

        }

        direction = ( (common_prefix_with_right - common_prefix_with_left) > 0 ) ? 1 : -1;
        int min_prefix_range = 0;

        if(idx == 0)
        {
            min_prefix_range = -1;

        }
        else
        {
            min_prefix_range = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - direction]); 
        }

        int lmax = 2;
        int next_key = idx + lmax*direction;

        while((next_key >= 0) && (next_key <  *numTriangles) && (__clz(sortedMortonCodes[idx] ^ sortedMortonCodes[next_key]) > min_prefix_range))
        {
            lmax *= 2;
            next_key = idx + lmax*direction;
        }
        //find the other end using binary search
        uint32_t l = 0;

        do
        {
            lmax = (lmax + 1) >> 1; // exponential decrease
            int new_val = idx + (l + lmax)*direction ; 

            if(new_val >= 0 && new_val < *numTriangles )
            {
                uint32_t Code = sortedMortonCodes[new_val];
                int Prefix = __clz(sortedMortonCodes[idx] ^ Code);
                if (Prefix > min_prefix_range)
                    l = l + lmax;
            }
        }
        while (lmax > 1);

        int j = idx + l*direction;

        int left = 0 ; 
        int right = 0;
        
        if(idx < j){
            left = idx;
            right = j;
        }
        else
        {
            left = j;
            right = idx;
        }

        //printf("idx : (%d) returning range (%d, %d) \n" , idx , left, right);

        return make_int2(left,right);
    }
    CUDA_DEV
    uint32_t expandBits(uint32_t v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

    // Calculates a 30-bit Morton code for the
    // given 3D point located within the unit cube [0,1].
    CUDA_DEV
    uint32_t morton3D(float x, float y, float z)
    {
        x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
        y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
        z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
        uint32_t xx = expandBits((uint32_t)x);
        uint32_t yy = expandBits((uint32_t)y);
        uint32_t zz = expandBits((uint32_t)z);
        return xx * 4 + yy * 2 + zz;
    }

    __global__ 
    void sampleRayGPU(PerspectiveKernel* camera, IndependentKernel * sampler, RayGPU3f* ray, ColorGPU3f* value, const VectorGPU2i * resSize, VectorGPU2i* m_outputSize_dev, VectorGPU2f* m_invOutputSize_dev, float* m_sampleToCamera_dev_tr, float* m_cameraToWorld_dev_tr,float* m_sampleToCamera_dev_inv, float* m_cameraToWorld_dev_inv, float* m_nearClip_dev, float* m_farClip_dev, uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev){
        int indexX = *x_render_offset_dev + blockIdx.x * blockDim.x + threadIdx.x;
        int indexY = *y_render_offset_dev + blockIdx.y * blockDim.y + threadIdx.y;
        if(indexX < resSize->x() && indexY < resSize->y()){
            PointGPU2f pixelSample = PointGPU2f((float) indexX, (float) indexY) + sampler->next2D(resSize,x_render_offset_dev,y_render_offset_dev);
            PointGPU2f apertureSample = sampler->next2D(resSize,x_render_offset_dev,y_render_offset_dev);
            value[resSize->y() * indexX + indexY] = camera->sampleRay(ray, pixelSample, apertureSample, m_outputSize_dev, m_invOutputSize_dev, m_sampleToCamera_dev_tr, m_cameraToWorld_dev_tr,m_sampleToCamera_dev_inv, m_cameraToWorld_dev_inv, m_nearClip_dev, m_farClip_dev, resSize, x_render_offset_dev, y_render_offset_dev);
            if (ray[resSize->y() * indexX + indexY].mint == 1e-4)
                ray[resSize->y() * indexX + indexY].mint = fmaxf(ray[resSize->y() * indexX + indexY].mint, ray[resSize->y() * indexX + indexY].mint * ray[resSize->y() * indexX + indexY].o.array().abs().maxCoeff());
        }
    }

    __global__ 
    void renderGPU(SceneKernel *scene ,NormalKernel* integrator, IndependentKernel* sampler,  RayGPU3f* ray, ColorGPU3f* value, ColorGPU4f* res_d, const VectorGPU2i * resSize, IntersectionGPU* its, uint32_t* t_indices, uint32_t* m_indices, uint32_t * nT, LeafNode*  leafNodes,InternalNode* internalNodes,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists, uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) {
        int indexX = *x_render_offset_dev + blockIdx.x * blockDim.x + threadIdx.x;
        int indexY = *y_render_offset_dev + blockIdx.y * blockDim.y + threadIdx.y;
        if(indexX < resSize->x() && indexY < resSize->y()){
 
            //float m_lookupFactor = NORI_FILTER_RESOLUTION / m_filter_d.getRadius();
            value[resSize->y() * indexX + indexY]  *= integrator->Li(scene, sampler, ray,  its, t_indices, m_indices, nT,  leafNodes, internalNodes, v_V, v_N, v_UV, v_F, N_exists, UV_exists, resSize,x_render_offset_dev, y_render_offset_dev);
            res_d[resSize->y() * indexX + indexY]  += ColorGPU4f(value[resSize->y() * indexX + indexY]);
            //printf("%f - %f - %f\n",value[resSize->y() * indexX + indexY].x(),value[resSize->y() * indexX + indexY].y(),value[resSize->y() * indexX + indexY].z());
        }
    }


    void bvh(SceneGPU & scene, ImageBlock & result, const VectorGPU2i * resSize, uint32_t maxThreadAmount){
        //Build tree and allocate all necessary objects
        SceneKernel sk;
        sk = scene;

        /* Modify according to the maximum thread amount: 1536 * 30 in our case*/
        /* Block of size 16*16 */
        /* GRID structure if needed : 12x12 => 16x16*/
        
        uint32_t* x_render_offset = (uint32_t*)malloc(sizeof(uint32_t));
        uint32_t* y_render_offset = (uint32_t*)malloc(sizeof(uint32_t));
        uint32_t* x_render_offset_dev;
        uint32_t* y_render_offset_dev;
        cudaMalloc((void**)&x_render_offset_dev, sizeof(uint32_t));
        cudaMalloc((void**)&y_render_offset_dev, sizeof(uint32_t));
        dim3 GRID(GPU_GRID_SIZE,GPU_GRID_SIZE);
        dim3 BLOCK(GPU_INNER_BLOCK_SIZE,GPU_INNER_BLOCK_SIZE);
    
        //Result size and arrays
        VectorGPU2i *resSize_d;
        ColorGPU4f *res_d;
        ColorGPU4f *res_h;
    
        res_h = (ColorGPU4f*)malloc(resSize->y()* resSize->x() *sizeof(ColorGPU4f));
        cudaMalloc((void**)& resSize_d, sizeof(VectorGPU2i));
        cudaMalloc((void**)& res_d, resSize->x()* resSize->y() * sizeof(ColorGPU4f));
        cudaMemcpy(resSize_d, resSize, sizeof(VectorGPU2i), cudaMemcpyHostToDevice);


        NormalKernel integrator;
        IndependentKernel sampler;
        
        /*RAY & INTERSECTION*/
        RayGPU3f * ray;
        IntersectionGPU *its;
        uint32_t *nT;
        cudaMalloc((void**)&nT, sizeof(uint32_t));
        cudaMemcpy(nT, scene.triangle_count_h,sizeof(uint32_t),cudaMemcpyHostToDevice);
        cudaMalloc((void**)&ray, resSize->x() * resSize->y() * sizeof(RayGPU3f));
        cudaMalloc((void**)&its, resSize->x() * resSize->y() * sizeof(IntersectionGPU));
        /*RAY & INTERSECTION*/
 
        /* TREE PARAMETERS */
        uint32_t         *t_indices = sk.m_bvh_dev.t_indices;
        uint32_t         *m_indices = sk.m_bvh_dev.m_indices;
        LeafNode         *leafNodes = sk.m_bvh_dev.leafNodes;
        InternalNode     *internalNodes = sk.m_bvh_dev.internalNodes;
        VectorGPU3f      *v_V = sk.m_bvh_dev.v_V;
        VectorGPU3f      *v_N = sk.m_bvh_dev.v_N;
        VectorGPU2f      *v_UV = sk.m_bvh_dev.v_UV;
        VectorGPU3i      *v_F = sk.m_bvh_dev.v_F;
        uint32_t         *N_exists = sk.m_bvh_dev.N_exists;
        uint32_t         *UV_exists = sk.m_bvh_dev.UV_exists;
        /* TREE PARAMETERS */

        /* CAMERA */
        PerspectiveKernel camera;
        VectorGPU2i* m_outputSize_dev;
        VectorGPU2f* m_invOutputSize_dev;
        float* m_sampleToCamera_dev_tr;
        float* m_cameraToWorld_dev_tr;
        float* m_sampleToCamera_dev_inv;
        float* m_cameraToWorld_dev_inv;
        float* m_nearClip_dev;
        float* m_farClip_dev; 
        const PerspectiveCamera* cam = scene.getCamera();

        checkCuda(cudaMalloc((void**)&m_outputSize_dev, sizeof(VectorGPU2i)));
        checkCuda(cudaMalloc((void**)&m_invOutputSize_dev, sizeof(VectorGPU2f)));
        checkCuda(cudaMalloc((void**)&m_sampleToCamera_dev_tr, 16*sizeof(float)));
        checkCuda(cudaMalloc((void**)&m_cameraToWorld_dev_tr, 16*sizeof(float)));
        checkCuda(cudaMalloc((void**)&m_sampleToCamera_dev_inv, 16*sizeof(float)));
        checkCuda(cudaMalloc((void**)&m_cameraToWorld_dev_inv,16*sizeof(float)));
        checkCuda(cudaMalloc((void**)&m_nearClip_dev, sizeof(float)));
        checkCuda(cudaMalloc((void**)&m_farClip_dev, sizeof(float)));
        checkCuda(cudaMemcpy(m_outputSize_dev, cam->m_outputSize, sizeof(VectorGPU2i), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(m_invOutputSize_dev, cam->m_invOutputSize, sizeof(VectorGPU2f), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(m_sampleToCamera_dev_tr, cam->m_sampleToCamera.m_transform.data(),16* sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(m_cameraToWorld_dev_tr, cam->m_cameraToWorld.m_transform.data(), 16*sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(m_sampleToCamera_dev_inv, cam->m_sampleToCamera.m_inverse.data(),16* sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(m_cameraToWorld_dev_inv, cam->m_sampleToCamera.m_inverse.data(), 16*sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(m_nearClip_dev, cam->m_nearClip, sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(m_farClip_dev, cam->m_farClip, sizeof(float), cudaMemcpyHostToDevice));
        /* CAMERA */

        IndependentSampler* samp = scene.getSampler();
        ColorGPU3f * value;
        checkCuda(cudaMalloc((void**)&value, resSize->y()*resSize->x()*sizeof(ColorGPU3f)));
        for(uint32_t i = 0; i < samp->getSampleCount(); i++){

            for(uint32_t ji = 0; ji < 4; ji++){
                *x_render_offset = ji * (GPU_GRID_SIZE * GPU_INNER_BLOCK_SIZE);
                checkCuda(cudaMemcpy(x_render_offset_dev, x_render_offset, sizeof(uint32_t) ,cudaMemcpyHostToDevice));
                for(uint32_t jj = 0; jj < 4; jj++){
                    /*Set offset*/
                    *y_render_offset = jj * (GPU_GRID_SIZE * GPU_INNER_BLOCK_SIZE);
                    checkCuda(cudaMemcpy(y_render_offset_dev, y_render_offset, sizeof(uint32_t) ,cudaMemcpyHostToDevice));
            
                    sampleRayGPU<<<GRID, BLOCK>>>(&camera, &sampler, ray, value, resSize_d, m_outputSize_dev, m_invOutputSize_dev, m_sampleToCamera_dev_tr, m_cameraToWorld_dev_tr, m_sampleToCamera_dev_inv, m_cameraToWorld_dev_inv, m_nearClip_dev,  m_farClip_dev, x_render_offset_dev, y_render_offset_dev);
                    cudaDeviceSynchronize();

                    renderGPU<<<GRID, BLOCK>>>(&sk, &integrator, &sampler, ray, value, res_d, resSize_d, its, t_indices, m_indices, nT, leafNodes, internalNodes, v_V, v_N, v_UV, v_F, N_exists, UV_exists,x_render_offset_dev, y_render_offset_dev);
                    cudaDeviceSynchronize();
                }
            }
        }
        

        checkCuda(cudaMemcpy(res_h, res_d, resSize->x()* resSize->y() *sizeof(ColorGPU4f), cudaMemcpyDeviceToHost));
        /*for(int i = 0; i < resSize->y()* resSize->x(); i++ )
            std::cout << res_h[i].x() << " " << res_h[i].y() << " " << res_h[i].z()<<endl;*/
        result.putResult(res_h);

        cudaFree(m_outputSize_dev);
        cudaFree(m_invOutputSize_dev);
        cudaFree(m_sampleToCamera_dev_tr);
        cudaFree(m_cameraToWorld_dev_tr);
        cudaFree(m_sampleToCamera_dev_inv);
        cudaFree(m_cameraToWorld_dev_inv);
        cudaFree(m_nearClip_dev);
        cudaFree(m_farClip_dev);
        cudaFree(x_render_offset_dev);
        cudaFree(y_render_offset_dev);
        cudaFree(value);
        cudaFree(its);
        cudaFree(nT);
        free(y_render_offset);
        free(x_render_offset);
        free(res_h);
        cudaFree(res_d);
    }

    void BVH_d::computeMortonCodes(VectorGPU3f& mMin, VectorGPU3f& mMax){
        int threadsPerBlock = 256;
        int blocksPerGrid = (*numTriangles_h + threadsPerBlock - 1) / threadsPerBlock;
        computeMortonCodesKernel<<<blocksPerGrid, threadsPerBlock>>>(mortonCodes, object_ids, Bboxs, numTriangles, mMin , mMax, t_indices, m_indices);
        cudaDeviceSynchronize();
    }
    
   
    //Check if sync after each kernel is needed: Not used in original tree implementation!
    void BVH_d::buildTree(){
        printf("BVH build started!\n");
        int threadsPerBlock = 256;
        int blocksPerGrid =
            (*numTriangles_h - 1 + threadsPerBlock - 1) / threadsPerBlock;
        setupLeafNodesKernel<<<blocksPerGrid, threadsPerBlock>>>(object_ids, leafNodes, Bboxs, numTriangles);
        cudaDeviceSynchronize();
        generateHierarchyKernel<<<blocksPerGrid, threadsPerBlock>>>(mortonCodes, object_ids, internalNodes , leafNodes , numTriangles, Bboxs);
        cudaDeviceSynchronize();
        computeBBoxesKernel<<<blocksPerGrid, threadsPerBlock>>>(leafNodes, internalNodes, numTriangles);
        cudaDeviceSynchronize();
        printf("BVH build finished!\n");
    }

    //  bool BVH_d::meshIntersect(uint32_t index, RayGPU3f* ray, IntersectionGPU* its,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize) const
    CUDA_DEV 
    bool BVH_d::rayIntersect(RayGPU3f *ray_, IntersectionGPU *its, uint32_t* t_indices, uint32_t* m_indices, bool  shadowRay, uint32_t * nT,  LeafNode*  leafNodes,InternalNode* internalNodes,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize, uint32_t* x_render_offset_dev, uint32_t* y_render_offset_dev) const {
        int indexX = *x_render_offset_dev + blockIdx.x * blockDim.x + threadIdx.x;
        int indexY = *y_render_offset_dev + blockIdx.y * blockDim.y + threadIdx.y;
        bool foundIntersection = false;
        int obj_id = -1;
        int depth_limit = 100;

   
        /*
        //BRUTE FORCE SEARCH CODE
        for(uint32_t i = 0; i < *nT - 1; i++ ){
            
            int j = leafNodes[i].object_id;

            if(boundingboxIntersect(leafNodes[i].BBox,ray_, resSize,x_render_offset_dev,y_render_offset_dev)){
                if(meshIntersect(j, ray_, its, v_V, v_N, v_UV, v_F,N_exists, UV_exists, resSize, x_render_offset_dev, y_render_offset_dev)){
                    foundIntersection = true;
                    obj_id = j;
                    //obj_id = node->object_id;
                    //printf("1 --- %d --- %d\n",obj_id, resSize->y()*indexX + indexY);
                    break;
                }   
            }
              
        }*/
        
        // Allocate traversal stack from thread-local memory,
        // and push NULL to indicate that there are no postponed nodes.
        Node* stack[64];
        stack[0] = nullptr;
        int stackPtr = 1;
        int curr_depth = 0;

        // Traverse nodes starting from the root.
        Node* node = &internalNodes[0];
        do
        {
            // Check each child node for overlap.
            Node* childL = node->childA;
            Node* childR = node->childB;
            bool overlapL = false;
            bool overlapR = false;
            if(childL){
                overlapL = boundingboxIntersect(childL->BBox, ray_, resSize,x_render_offset_dev,y_render_offset_dev);
                //overlapL = childL->BBox.rayIntersect(ray_, resSize, indexX, indexY);
            }
            if(childR){
                overlapR = boundingboxIntersect(childR->BBox, ray_, resSize,x_render_offset_dev,y_render_offset_dev);
                //overlapR = childR->BBox.rayIntersect(ray_, resSize, indexX, indexY);
            }

            bool traverseL = false;
            bool traverseR = false;

            // Query overlaps a leaf node => report collision.
            
            if (overlapL){
                if(childL->isLeaf){
                    //obj_id = childL->object_id;
                    foundIntersection = meshIntersect(childL->object_id, ray_, its, v_V, v_N, v_UV, v_F,N_exists, UV_exists, resSize, x_render_offset_dev, y_render_offset_dev);
                    if(foundIntersection){
                        obj_id = childL->object_id;
                        //printf("%d --- %d --- %d\n",foundIntersection,obj_id, resSize->y()*indexX + indexY);
                        break;         
                    }
                }
                else
                    traverseL = true;
            }
            
           
            if (overlapR){
                if(childR->isLeaf){
                    foundIntersection = meshIntersect(childR->object_id, ray_, its, v_V, v_N, v_UV, v_F,N_exists, UV_exists, resSize, x_render_offset_dev, y_render_offset_dev);
                    if(foundIntersection){
                        obj_id = childR->object_id;
                        //printf("2 --- %d --- %d\n",obj_id, resSize->y()*indexX + indexY);
                        break;
                    }
                }
                else
                    traverseR = true;
            }
            

            if (!traverseL && !traverseR){
                node = stack[--stackPtr]; // pop
            }
            else
            {
                node = (traverseL) ? childL : childR;
                if (traverseL && traverseR){
                    stack[stackPtr++] = childR; // push
                }
            }
            curr_depth++;
        }
        while (node != nullptr && curr_depth < depth_limit);
     
        if (foundIntersection) {
            uint32_t tt = t_indices[obj_id];
            uint32_t mm = m_indices[obj_id];
            /* At this point, we now know that there is an intersection,
            and we know the triangle index of the closest such intersection.

            The following computes a number of additional properties which
            characterize the intersection (normals, texture coordinates, etc..)
            */
        
            /* Find the barycentric coordinates */
            VectorGPU3f bary;
            bary << 1-its[resSize->y() * indexX + indexY].uv.sum(), its[resSize->y() * indexX + indexY].uv;

            /* Vertex indices of the triangle */
            uint32_t idx0 = v_F[tt].x(), idx1 = v_F[tt].y(), idx2 = v_F[tt].z();

            PointGPU3f p0 = v_V[idx0], p1 = v_V[idx1], p2 = v_V[idx2];
            /* Compute the intersection positon accurately
            using barycentric coordinates */
            its[resSize->y() * indexX + indexY].p = bary.x() * p0 + bary.y() * p1 + bary.z() * p2;
            /* Compute proper texture coordinates if provided by the mesh */
            if (UV_exists[mm])
                its[resSize->y() * indexX + indexY].uv = bary.x() * v_UV[idx0] +
                    bary.y() * v_UV[idx1] +
                    bary.z() * v_UV[idx2];
            /* Compute the geometry frame */
            its[resSize->y() * indexX + indexY].geoFrame = FrameGPU((p1-p0).cross(p2-p0).normalized());
            if (N_exists[mm]) {
                /* Compute the shading frame. Note that for simplicity,
                the current implementation doesn't attempt to provide
                tangents that are continuous across the surface. That
                means that this code will need to be modified to be able
                use anisotropic BRDFs, which need tangent continuity */
                its[resSize->y() * indexX + indexY].shFrame = FrameGPU(
                    (bary.x() * v_N[idx0] +
                    bary.y() * v_N[idx1] +
                    bary.z() * v_N[idx2]).normalized());
            } else {
                its[resSize->y() * indexX + indexY].shFrame = its[resSize->y() * indexX + indexY].geoFrame;
            }
        }
        
        return foundIntersection;
    }

    void BVH_d::setUp(BoundingBoxGPU3f* bboxes_mesh_d,  uint32_t* triangle_ids_d, uint32_t* mesh_ids_d, uint32_t* triangle_count_host,  VectorGPU3f* v_V_dev, VectorGPU3f* v_N_dev, VectorGPU2f* v_UV_dev, VectorGPU3i* v_F_dev, uint32_t* N_exists_dev, uint32_t* UV_exists_dev, uint32_t* vertex_count_host){
        Bboxs = bboxes_mesh_d;
        t_indices = triangle_ids_d;
        m_indices = mesh_ids_d;
        numTriangles_h = triangle_count_host;
        numVertices_h = vertex_count_host;
        v_N = v_N_dev;
        v_V = v_V_dev;
        v_F = v_F_dev;
        v_UV = v_UV_dev;
        N_exists = N_exists_dev;
        UV_exists = UV_exists_dev;

        checkCuda(cudaMalloc((void**)&numTriangles, sizeof(uint32_t)));
        checkCuda(cudaMalloc((void**)&mortonCodes, *triangle_count_host*sizeof(uint32_t)));
        checkCuda(cudaMalloc((void**)&object_ids, *triangle_count_host*sizeof(uint32_t)));
        checkCuda(cudaMalloc((void**)&leafNodes, *triangle_count_host*sizeof(LeafNode)));
        checkCuda(cudaMalloc((void**)&internalNodes, (*triangle_count_host - 1)*sizeof(InternalNode)));
        
        checkCuda(cudaMemcpy(numTriangles, triangle_count_host, sizeof(uint32_t), cudaMemcpyHostToDevice));
        mMin = VectorGPU3f(1e9, 1e9, 1e9);
        mMax = VectorGPU3f(-1e9, -1e9, -1e9);

        // Set up for the BVH Build
        findMinMax(mMin, mMax);
        computeMortonCodes(mMin, mMax);

        thrust::device_ptr<uint32_t> dev_mortonCodes(mortonCodes);
        thrust::device_ptr<uint32_t> dev_object_ids(object_ids);
        thrust::sort_by_key(dev_mortonCodes, dev_mortonCodes + *numTriangles_h, dev_object_ids);
        
        cudaDeviceSynchronize();
        buildTree();
    }

    /// Check if a ray intersects a bounding box
    CUDA_DEV 
    bool BVH_d::boundingboxIntersect(BoundingBoxGPU3f bb, RayGPU3f *ray, const VectorGPU2i *resSize, uint32_t* x_render_offset_dev, uint32_t* y_render_offset_dev) const {
        int indexX = *x_render_offset_dev + blockIdx.x * blockDim.x + threadIdx.x;
        int indexY = *y_render_offset_dev + blockIdx.y * blockDim.y + threadIdx.y;
        float nearT = -1.0e308f;
        float farT = 1.0e308f;
        float tmp;
        //printf("%f ::: %f ::: %f\n",min[0],min[1], min[2]);
        for (int i=0; i<3; i++) {
            float origin = ray[resSize->y() * indexX + indexY].o[i];
            float minVal = bb.min[i], maxVal = bb.max[i];
           
            if (ray[resSize->y() * indexX + indexY].d[i] == 0) {
                if (origin < minVal || origin > maxVal){
                    return false;
                }
            } else {
                float t1 = (minVal - origin) * ray[resSize->y() * indexX + indexY].dRcp[i];
                float t2 = (maxVal - origin) * ray[resSize->y() * indexX + indexY].dRcp[i];
                //printf("%f ttt %f ttt %f ttt %f ttt %f\n", minVal, maxVal ,origin, t1, t2);

                if (t1 > t2)
                {
                    tmp = t1;
                    t1 = t2;
                    t2 = tmp;
                }
                //printf("%f ttt %f ttt %f ttt %f\n", nearT, farT, t1 ,t2);
                nearT = fmaxf(t1, nearT);
                farT = fminf(t2, farT);           
                if (!(nearT <= farT)){
                    //printf("%f ttt %f\n", nearT, farT);
                    return false;
                }
            }
        }
        bool res = ray[resSize->y() * indexX + indexY].mint <= farT && nearT <= ray[resSize->y() * indexX + indexY].maxt;
        //printf("%d ------------ %f --------- %f\n",res,farT, nearT);
        return res;
    }

    CUDA_DEV 
    bool BVH_d::meshIntersect(uint32_t index, RayGPU3f* ray, IntersectionGPU* its,VectorGPU3f * v_V,VectorGPU3f * v_N,  VectorGPU2f* v_UV,  VectorGPU3i* v_F,uint32_t* N_exists,uint32_t* UV_exists,const VectorGPU2i * resSize,uint32_t* x_render_offset_dev,uint32_t* y_render_offset_dev) const{
        int indexX = *x_render_offset_dev + blockIdx.x * blockDim.x + threadIdx.x;
        int indexY = *y_render_offset_dev + blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t i0 = v_F[index].x(), i1 = v_F[index].y(), i2 = v_F[index].z();
        const VectorGPU3f p0 = v_V[i0], p1 = v_V[i1], p2 = v_V[i2];
        /* Find vectors for two edges sharing v[0] */
        VectorGPU3f edge1 = p1 - p0, edge2 = p2 - p0;

        /* Begin calculating determinant - also used to calculate U parameter */
        VectorGPU3f pvec = ray[resSize->y() * indexX + indexY].d.cross(edge2);
      
        /* If determinant is near zero, ray lies in plane of triangle */
        float det = edge1.dot(pvec);
    
        if (det > -1e-8f && det < 1e-8f){
            return false;
        }
        float inv_det = 1.0f / det;

        /* Calculate distance from v[0] to ray origin */
        VectorGPU3f tvec = ray[resSize->y() * indexX + indexY].o - p0;

        /* Calculate U parameter and test bounds */
        float u = tvec.dot(pvec) * inv_det;

        if (u < 0.0f || u > 1.0f)
            return false;

        /* Prepare to test V parameter */
        VectorGPU3f qvec = tvec.cross(edge1);

        /* Calculate V parameter and test bounds */
        float v = ray[resSize->y() * indexX + indexY].d.dot(qvec) * inv_det;
        if (v < 0.0 || u + v > 1.0)
            return false;

        /* Ray intersects triangle -> compute t */
        float t = edge2.dot(qvec) * inv_det;


        bool res = (t >= ray[resSize->y() * indexX + indexY].mint) && (t <= ray[resSize->y() * indexX + indexY].maxt);
        if(res){
            //printf("%d ------%d\n", indexX, indexY);
            ray[resSize->y() * indexX + indexY].maxt = t;
            its[resSize->y() * indexX + indexY].t = t;
            its[resSize->y() * indexX + indexY].uv = PointGPU2f(u,v);
        }
        //printf("dsgajdgjhsahgjs\n");
        return res;
    }

        /*
        CUDA_DEV
        void BVH_d::sampleUniformSurface(IndependentSampler * sampler, PointGPU3f & samplePnt, NormalGPU3f & surfNormal, float & dpdfConst) const{
            float alpha, beta;
            uint32_t tri_index = dpdf.sample(sampler->next1D());
            uint32_t idx1 = (*m_F_dev)(0, tri_index), idx2 = (*m_F_dev)(1, tri_index), idx3 = (*m_F_dev)(2, tri_index);
            PointGPU3f v_V1 = m_V_dev->col(idx1), v_V2 = m_V_dev->col(idx2), v_V3 =m_V_dev->col(idx3);

            PointGPU2f sample_pnt2d = sampler->next2D();
            alpha = 1 - sqrtf(1 - sample_pnt2d.x());
            beta = sample_pnt2d.y() * sqrtf(1 - sample_pnt2d.x());
            samplePnt = (alpha * v_V1 + beta * v_V2 + (1 - (alpha + beta)) * v_V3);

            if(!m_N.size())
            {
                PointGPU3f dir1 = v_V2 - v_V1;
                PointGPU3f dir2 = v_V3 - v_V1;
                surfNormal = dir1.cross(dir2);
            }
            else
            {
                NormalGPU3f v_N1 =m_N_dev->col(idx1), v_N2 = m_N_dev->col(idx2), v_N3 = m_N_dev->col(idx3);
                surfNormal = (alpha * v_N1 + beta * v_N2 + (1 - (alpha + beta)) * v_N3);
            }
            surfNormal.normalize();
            dpdfConst = dpdf.getNormalization();

        }  

        CUDA_DEV
        float getDPDFconst() const{return dpdf.getNormalization();}
        */

            /*
        bool notA = false;
        bool notB = false;
        int processedNode = 0;
        int iter_lim = 5000;
        int iter = 0;

        Node* node = &internalNodes[0];   
        while(iter < iter_lim){
            //printf("hallo bilbo -- %d \n", node->isLeaf);
        
            if(node->childA && node->childA->BBox.rayIntersect(ray_, resSize, indexX, indexY)){
                //printf("111 %d ----- %d\n",iter,iter_lim);
                if(node->childA->isLeaf){
                    obj_id = node->childA->object_id;
                    processedNode++;
                    //printf("1-%d\n",obj_id);
                    foundIntersection = meshIntersect(obj_id, ray_, its, v_V, v_N, v_UV, v_F,N_exists, UV_exists, resSize, x_render_offset_dev, y_render_offset_dev);
                    //printf("natzi bilbo -- %d -- %d \n", foundIntersection,node->object_id);
                    if(foundIntersection){
                        printf("GIRDI AMINA KOYIM 1 AQ\n");
                        break;
                    }
                    else{
                        notA = true;
                    }

                }
                else{
                    node = node->childA;
                    continue;
                }
            }
            if(node->childB && node->childB->BBox.rayIntersect(ray_, resSize, indexX, indexY)){
                //printf("111 %d ----- %d\n",iter,iter_lim);
                if(node->childB->isLeaf){
                    obj_id = node->childB->object_id;
                    processedNode++;
                    //printf("1-%d\n",obj_id);
                    foundIntersection = meshIntersect(obj_id, ray_, its, v_V, v_N, v_UV, v_F,N_exists, UV_exists, resSize, x_render_offset_dev, y_render_offset_dev);
                    //printf("natzi bilbo -- %d -- %d \n", foundIntersection,node->object_id);
                    if(foundIntersection){
                        printf("GIRDI AMINA KOYIM 2 AQ\n");
                        break;
                    }
                    else{
                        notB = true;
                    }
                } 
                else{
                    node = node->childB;
                    continue;
                }
            }
             
            //Traverse back through parents
            if(notA && notB){
                for(int i = 0; i < (processedNode + 1); i++){
                    if(node->parent)
                        node = node->parent;
                    else
                        return false;
                }
                notA = false;
                notB = false;
            }
            iter++;
        }*/
        /*
        */

}

