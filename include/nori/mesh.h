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

#include <nori/bboxgpu.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <Eigen/Geometry>
#include <nori/framegpu.h>
#include <cuda_runtime.h>
#include <nori/warpgpu.h>
#include <nori/vectorgpu.h>
#include <Eigen/Geometry>
//#include <nori/dpdfgpu.h>

namespace nori{

    /**
     * \brief Intersection data structure
     *
     * This data structure records local information about a ray-triangle intersection.
     * This includes the position, traveled ray distance, uv coordinates, as well
     * as well as two local coordinate frames (one that corresponds to the true
     * geometry, and one that is used for shading computations).
     * 
     * \brief Mesh Abstract Class
     * 
     * This class is the abstract mesh class -> Base new objects from here
     * If not abstract, results in seg fault with nvcc compiler.
     */struct IntersectionGPU {
        /// Position of the surface intersection
        PointGPU3f p;
        /// Unoccluded distance along the ray
        float t;
        /// UV coordinates, if any
        PointGPU2f uv;
        /// Shading frame (based on the shading normal)
        FrameGPU shFrame;
        /// Geometric frame (based on the true geometry)
        FrameGPU geoFrame;
        /// Pointer to the associated mesh
        //const MeshKernel *mesh;
        uint32_t mesh_id;


        /// Create an uninitialized intersection record 
        CUDA_DEV
        IntersectionGPU() : mesh_id(0) { }

        /// Transform a direction vector into the local shading frame
        CUDA_DEV 
        VectorGPU3f toLocal(const VectorGPU3f &d) const {
            return shFrame.toLocal(d);
        }

        /// Transform a direction vector from local to world coordinates
        CUDA_DEV 
        VectorGPU3f toWorld(const VectorGPU3f &d) const {
            return shFrame.toWorld(d);
        }
    };

    class Mesh : public NoriObject {
    friend class SceneGPU;
    public:

        Mesh(const PropertyList &propList); 

        ~Mesh(){
            delete m_bsdf;
            if(isEmitter())
                delete m_emitter;
        }

        /// Initialize internal data structures (called once by the XML parser)
        void activate(){
            if (!m_bsdf) {
                /* If no material was assigned, instantiate a diffuse BRDF */
                m_bsdf = static_cast<DiffuseBSDF *>(
                    NoriObjectFactory::createInstance("diffuse", PropertyList()));
            }
            /*
            tri_count = getTriangleCount();
            if(isEmitter()){
                dpdf = DiscretePDFGPU(tri_count);
                for(uint64_t i = 0; i < tri_count; i++){
                    dpdf.append(surfaceArea(i));
                }
                dpdf.normalize();
            }
            */
            
        }
    
        float surfaceArea(uint32_t index) const {
            uint32_t i0 = m_F(0, index), i1 = m_F(1, index), i2 = m_F(2, index);
            const PointGPU3f p0 = m_V.col(i0), p1 = m_V.col(i1), p2 = m_V.col(i2);
            return 0.5f * VectorGPU3f((p1 - p0).cross(p2 - p0)).norm();
        }

        /// Return a pointer to an attached area emitter instance
        AreaLightEmitter *getEmitter() { return m_emitter; }

        /// Return a pointer to an attached area emitter instance (const version)
        const AreaLightEmitter *getEmitter() const { return m_emitter; }

        /// Return a pointer to the BSDF associated with this mesh
        const DiffuseBSDF *getBSDF() const { return m_bsdf; }

        
        /// Is this mesh an area emitter?
        bool isEmitter() const { return m_emitter != nullptr; }

        BoundingBoxGPU3f getBoundingBox(uint32_t index) const{
            BoundingBoxGPU3f result(m_V.col(m_F(0,index)));
            result.expandBy(m_V.col(m_F(1,index)));
            result.expandBy(m_V.col(m_F(2,index)));
            return result;
        }

          /// Return the total number of triangles in this shape
        uint32_t getTriangleCount() const { return (uint32_t) m_F.cols(); }

        /// Return the total number of vertices in this shape
        uint32_t getVertexCount() const { return (uint32_t) m_V.cols();}

        std::string toString() const {
            return "GPU-Mesh[Wavefront]";
        }    

        /// Register a child object (e.g. a BSDF) with the mesh
        void addChild(NoriObject *obj){
            switch (obj->getClassType()) {
            case EBSDF:
                if (m_bsdf)
                    throw NoriException(
                        "Mesh: tried to register multiple BSDF instances!");
                m_bsdf = static_cast<DiffuseBSDF *>(obj);
                break;

            case EEmitter: { 
                    if (m_emitter)
                        throw NoriException(
                            "Mesh: tried to register multiple Emitter instances!");
                    m_emitter = static_cast<AreaLightEmitter *>(obj);
                }
                break;

            default:
                throw NoriException("Mesh::addChild(<%s>) is not supported!",
                                    classTypeName(obj->getClassType()));
            }
        }
       
        EClassType getClassType() const { return EMesh; }    
   
    protected:

        struct OBJVertex {
            uint32_t p = (uint32_t) -1;
            uint32_t n = (uint32_t) -1;
            uint32_t uv = (uint32_t) -1;

            inline OBJVertex() { }

            inline OBJVertex(const std::string &string) {
                std::vector<std::string> tokens = tokenize(string, "/", true);

                if (tokens.size() < 1 || tokens.size() > 3)
                    throw NoriException("Invalid vertex data: \"%s\"", string);

                p = toUInt(tokens[0]);

                if (tokens.size() >= 2 && !tokens[1].empty())
                    uv = toUInt(tokens[1]);

                if (tokens.size() >= 3 && !tokens[2].empty())
                    n = toUInt(tokens[2]);
            }

            inline bool operator==(const OBJVertex &v) const {
                return v.p == p && v.n == n && v.uv == uv;
            }
        };

        /// Hash function for OBJVertex
        struct OBJVertexHash {
            std::size_t operator()(const OBJVertex &v) const {
                size_t hash = std::hash<uint32_t>()(v.p);
                hash = hash * 37 + std::hash<uint32_t>()(v.uv);
                hash = hash * 37 + std::hash<uint32_t>()(v.n);
                return hash;
            }
        };
        /// Create an empty mesh
        //DiscretePDFGPU dpdf;                    ///< Discrete probability density function corresponding to the mesh object            
        /*CPU data is as before, therefore can be used regularly in CPU*/
        DiffuseBSDF       *m_bsdf = nullptr;      ///< BSDF of the surface
        AreaLightEmitter  *m_emitter = nullptr;     ///< Associated emitter, if any
        std::string m_name;
        BoundingBoxGPU3f m_bbox;                ///< Bounding box of the mesh
        uint32_t tri_count;                  ///< Triangle count of the mesh object    
        MatrixXf      m_V;                    ///< Vertex positions
        MatrixXf      m_N;                   ///< Vertex normals
        MatrixXf      m_UV;                  ///< Vertex texture coordinates
        MatrixXu      m_F;                    ///< Faces
        std::vector<VectorGPU3f> v_V;
        std::vector<VectorGPU3f> v_N;
        std::vector<VectorGPU2f> v_UV;
        std::vector<VectorGPU3i> v_F;
        uint32_t             N_exists;
        uint32_t            UV_exists;
    };
}