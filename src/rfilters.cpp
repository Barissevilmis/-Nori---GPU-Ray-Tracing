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

#include <nori/rfilter.h>


namespace nori{

    /**
     * Windowed Gaussian filter with configurable extent
     * and standard deviation. Often produces pleasing 
     * results, but may introduce too much blurring.
     */
   
        GaussianFilter::GaussianFilter(const PropertyList &propList) {
            /* Half filter size */
            m_radius = propList.getFloat("radius", 2.0f);
            /* Standard deviation of the Gaussian */
            m_stddev = propList.getFloat("stddev", 0.5f);
        }
        
        float GaussianFilter::eval(float x) const {
            float alpha = -1.0f / (2.0f * m_stddev*m_stddev);
            return fmaxf(0.0f, expf(alpha * x * x) - expf(alpha * m_radius * m_radius));
        }
        
  

    /**
     * Separable reconstruction filter by Mitchell and Netravali
     * 
     * D. Mitchell, A. Netravali, Reconstruction filters for computer graphics, 
     * Proceedings of SIGGRAPH 88, Computer Graphics 22(4), pp. 221-228, 1988.
     */
        
    MitchellNetravaliFilter::MitchellNetravaliFilter(const PropertyList &propList) {
        /* Filter size in pixels */
        m_radius = propList.getFloat("radius", 2.0f);
        /* B parameter from the paper */
        m_B = propList.getFloat("B", 1.0f / 3.0f);
        /* C parameter from the paper */
        m_C = propList.getFloat("C", 1.0f / 3.0f);
    }
    
    float MitchellNetravaliFilter::eval(float x) const {
        x = fabsf(2.0f * x / m_radius);
        float x2 = x*x, x3 = x2*x;

        if (x < 1) {
            return 1.0f/6.0f * ((12-9*m_B-6*m_C)*x3 
                    + (-18+12*m_B+6*m_C) * x2 + (6-2*m_B));
        } else if (x < 2) {
            return 1.0f/6.0f * ((-m_B-6*m_C)*x3 + (6*m_B+30*m_C) * x2
                    + (-12*m_B-48*m_C)*x + (8*m_B + 24*m_C));
        } else {
            return 0.0f;
        }
    }
 
  
    TentFilter::TentFilter(const PropertyList &) {
        m_radius = 1.0f;
    }
    
    float TentFilter::eval(float x) const {
        return fmaxf(0.0f, 1.0f - fabsf(x));
    }
        
    
  

    /// Box filter -- fastest, but prone to aliasing

    
    BoxFilter::BoxFilter(const PropertyList &) {
        m_radius = 0.5f;
    }
    
    float BoxFilter::eval(float x) const {
        return 1.0f;
    }


    NORI_REGISTER_CLASS(GaussianFilter, "gaussian");
    NORI_REGISTER_CLASS(MitchellNetravaliFilter, "mitchell");
    NORI_REGISTER_CLASS(TentFilter, "tent");
    NORI_REGISTER_CLASS(BoxFilter, "box");

}
