#include <nori/parser.h>
#include <nori/block.h>
#include <nori/timer.h>
#include <nori/bitmap.h>
#include <nori/scenegpu.h>
#include <nori/accelbvh.h>
#include <nori/gui.h>
#include <nori/camera.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <cuda_runtime.h>
#include <tbb/task_scheduler_init.h>
#include <filesystem/resolver.h>
#include <thread>

using namespace nori;
static int threadCount = -1;
static bool gui = true;
/*
inline cudaError_t checkCuda(cudaError_t result) {
        if (result != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
            assert(result == cudaSuccess);
        }
        return result;
    }
*/
//Fix

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Syntax: " << argv[0] << " <scene.xml> [--no-gui] [--threads N]" <<  endl;
        return -1;
    }
    std::string sceneName = "";
    std::string exrName = "";

    for (int i = 1; i < argc; ++i) {
        std::string token(argv[i]);
        if (token == "-t" || token == "--threads") {
            if (i+1 >= argc) {
                cerr << "\"--threads\" argument expects a positive integer following it." << endl;
                return -1;
            }
            threadCount = atoi(argv[i+1]);
            i++;
            if (threadCount <= 0) {
                cerr << "\"--threads\" argument expects a positive integer following it." << endl;
                return -1;
            }

            continue;
        }
        else if (token == "--no-gui") {
            gui = false;
            continue;
        }

        filesystem::path path(argv[i]);

        try {
            if (path.extension() == "xml") {
                sceneName = argv[i];

                /* Add the parent directory of the scene file to the
                   file resolver. That way, the XML file can reference
                   resources (OBJ files, textures) using relative paths */
                getFileResolver()->prepend(path.parent_path());
            } else if (path.extension() == "exr") {
                /* Alternatively, provide a basic OpenEXR image viewer */
                exrName = argv[i];
            } else {
                cerr << "Fatal error: unknown file \"" << argv[i]
                     << "\", expected an extension of type .xml or .exr" << endl;
            }
        } catch (const std::exception &e) {
            cerr << "Fatal error: " << e.what() << endl;
            return -1;
        }
    }
    uint32_t maxThreadAmount = 1;
    int deviceCount, device;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
        deviceCount = 0;
    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
    if (properties.major != 9999) /* 9999 means emulation only */
    if (device==0)
    {
        maxThreadAmount = properties.multiProcessorCount * properties.maxThreadsPerMultiProcessor;
        printf("multiProcessorCount %d\n",properties.multiProcessorCount);
        printf("maxThreadsPerMultiProcessor %d\n",properties.maxThreadsPerMultiProcessor);
    }
}

    if (exrName !="" && sceneName !="") {
        cerr << "Both .xml and .exr files were provided. Please only provide one of them." << endl;
        return -1;
    }
    else if (exrName == "" && sceneName == "") {
        cerr << "Please provide the path to a .xml (or .exr) file." << endl;
        return -1;
    }
    else if (exrName != "") {
        if (!gui) {
            cerr << "Flag --no-gui was set. Please remove it to display the EXR file." << endl;
            return -1;
        }
        try {
           
            Bitmap bitmap(exrName);
            ImageBlock block(VectorGPU2i((int) bitmap.cols(), (int) bitmap.rows()));
            block.fromBitmap(bitmap);
            
            nanogui::init();
            NoriScreen *screen = new NoriScreen(block);
            nanogui::mainloop(50.f);
            delete screen;
            nanogui::shutdown();
            
        } catch (const std::exception &e) {
            cerr << e.what() << endl;
            return -1;
        }
    }
    else { // sceneName != ""
        try {

            std::unique_ptr<NoriObject> root(loadFromXML(sceneName));
            if (root->getClassType() == NoriObject::EScene){
                SceneGPU* scene = static_cast<SceneGPU *>(root.get());
                const PerspectiveCamera* camera = scene->getCamera();
                VectorGPU2i* outputSize = camera->getOutputSize();
                ImageBlock result(*outputSize);
                ImageBlock tmp_block(GPU_INNER_BLOCK_SIZE);
                BlockGenerator blockGen(*outputSize, GPU_INNER_BLOCK_SIZE);
               
                result.clear();
                /*
                while(blockGen.next(tmp_block)){
                    offsets_abstract[ii].x() = tmp_block.getOffset().x();
                    offsets_abstract[ii].y() = tmp_block.getOffset().y();
                    ii+=1;
                    //printf("%d ++++++ %d --- %d\n", tmp_block.getOffset().x(), tmp_block.getOffset().y(),ii);
                }*/
                /*
                int k = 0;
                for(uint32_t i = 0; i < outputSize->x(); i+=16){
                    for(uint32_t j = 0; j < outputSize->y(); j+=16){

                        offsets[j + i * outputSize->y()] = offsets_abstract[k];
                    }
                }*/

                bvh(*scene, result, outputSize, maxThreadAmount);
                printf("Starting visualization...\n");
                
                NoriScreen *screen = nullptr;
                if (gui) {
                    nanogui::init();
                    screen = new NoriScreen(result);
                }

                //Put into the main
                if (gui){
                    nanogui::mainloop(50.f);
                }

                if (gui) {
                    delete screen;
                    nanogui::shutdown();
                }
                
                /* Now turn the rendered image block into
                a properly normalized bitmap */
                
                std::unique_ptr<Bitmap> bitmap(result.toBitmap());

                /* Determine the filename of the output bitmap */
                std::string outputName = sceneName;
                size_t lastdot = outputName.find_last_of(".");
                if (lastdot != std::string::npos)
                    outputName.erase(lastdot, std::string::npos);

                /* Save using the OpenEXR format */
                bitmap->saveEXR(outputName);

                /* Save tonemapped (sRGB) output using the PNG format */
                bitmap->savePNG(outputName);
                
            }           
            }catch (const std::exception &e) {
                cerr << e.what() << endl;
                return -1;
            }
        }
           
    return 0;
}