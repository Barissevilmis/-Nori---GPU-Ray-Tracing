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
#include <nori/object.h>


namespace nori{

    /**
     * Windowed Gaussian filter with configurable extent
     * and standard deviation. Often produces pleasing 
     * results, but may introduce too much blurring.
     */
    class GaussianFilter: public NoriObject{
    public:
        GaussianFilter(const PropertyList &propList);
        
        float eval(float x) const;
        float getRadius() const { return m_radius;}
        EClassType getClassType() const { return EReconstructionFilter; }
        std::string toString() const {
            return tfm::format("GaussianFilter[radius=%f, stddev=%f]", m_radius, m_stddev);
        }
    protected:
        float m_stddev, m_radius;
    };

    /**
     * Separable reconstruction filter by Mitchell and Netravali
     * 
     * D. Mitchell, A. Netravali, Reconstruction filters for computer graphics, 
     * Proceedings of SIGGRAPH 88, Computer Graphics 22(4), pp. 221-228, 1988.
     */
    class MitchellNetravaliFilter: public NoriObject{
    public:
        
        MitchellNetravaliFilter(const PropertyList &propList);
        float getRadius() const { return m_radius;}
        float eval(float x) const;
        EClassType getClassType() const { return EReconstructionFilter; }
        std::string toString() const {
            return tfm::format("MitchellNetravaliFilter[radius=%f, B=%f, C=%f]", m_radius, m_B, m_C);
        }
    protected:
        float m_B, m_C, m_radius;
    };

    /// Tent filter 
    class TentFilter: public NoriObject{
    public: 
        TentFilter(const PropertyList &);
        float getRadius() const { return m_radius;}
        float eval(float x) const;
        EClassType getClassType() const { return EReconstructionFilter; }
        std::string toString() const {
            return "TentFilter[]";
        }
    protected:
        float m_radius;
    };

    /// Box filter -- fastest, but prone to aliasing
    class BoxFilter: public NoriObject{
    public:
        
        BoxFilter(const PropertyList &);
        float getRadius() const { return m_radius;}
        float eval(float x) const;
        EClassType getClassType() const { return EReconstructionFilter; }
        std::string toString() const {
            return "BoxFilter[]";
        }
    protected:
        float m_radius;
    };
}
