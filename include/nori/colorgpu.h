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
namespace nori {

    /**
     * \brief Represents a linear RGB color value
     */
    struct ColorGPU3f : public Eigen::Array3f {
    public:
        typedef Eigen::Array3f Base;

        /// Initialize the color vector with a uniform value
        CUDA_HOSTDEV
        ColorGPU3f(float value = 0.f) : Base(value, value, value) { }

        /// Initialize the color vector with specific per-channel values
        CUDA_HOSTDEV
        ColorGPU3f(float r, float g, float b) : Base(r, g, b) { }

        /// Construct a color vector from ArrayBase (needed to play nice with Eigen)
        template <typename Derived> 
        CUDA_HOSTDEV
        ColorGPU3f(const Eigen::ArrayBase<Derived>& p) 
            : Base(p) { }

        /// Assign a color vector from ArrayBase (needed to play nice with Eigen)
        template <typename Derived> 
        CUDA_HOSTDEV
        ColorGPU3f &operator=(const Eigen::ArrayBase<Derived>& p) {
            this->Base::operator=(p);
            return *this;
        }

        /// Return a reference to the red channel
        CUDA_HOSTDEV
        float &r() { return x(); }
        /// Return a reference to the red channel (const version)
        CUDA_HOSTDEV
        const float &r() const { return x(); }
        /// Return a reference to the green channel
        CUDA_HOSTDEV
        float &g() { return y(); }
        /// Return a reference to the green channel (const version)
        CUDA_HOSTDEV
        const float &g() const { return y(); }
        /// Return a reference to the blue channel
        CUDA_HOSTDEV 
        float &b() { return z(); }
        /// Return a reference to the blue channel (const version)
        CUDA_HOSTDEV
        const float &b() const { return z(); }

        /// Clamp to the positive range
        // CUDA_HOSTDEV
        // ColorGPU3f clamp() const { return ColorGPU3f(max(r(), 0.0f),
        //     max(g(), 0.0f), max(b(), 0.0f)); }

        /// Check if the color vector contains a NaN/Inf/negative value
        CUDA_HOSTDEV
        bool isValid() const{
            for (int i=0; i<3; ++i) {
                float value = coeff(i);
                if (value < 0 || !isfinite(value))
                    return false;
            }
            return true;
        }

        /// Convert from sRGB to linear RGB
        CUDA_HOSTDEV
        ColorGPU3f toLinearRGB() const{
            ColorGPU3f result;

            for (int i=0; i<3; ++i) {
                float value = coeff(i);

                if (value <= 0.04045f)
                    result[i] = value * (1.0f / 12.92f);
                else
                    result[i] = powf((value + 0.055f)
                        * (1.0f / 1.055f), 2.4f);
            }

            return ColorGPU3f(result);
        }

        /// Convert from linear RGB to sRGB
        CUDA_HOSTDEV 
        ColorGPU3f toSRGB() const{
            ColorGPU3f result;

            for (int i=0; i<3; ++i) {
                float value = coeff(i);

                if (value <= 0.0031308f)
                    result[i] = 12.92f * value;
                else
                    result[i] = (1.0f + 0.055f)
                        * powf(value, 1.0f/2.4f) -  0.055f;
            }

            return result;
        }

        /// Return the associated luminance
        CUDA_HOSTDEV 
        float getLuminance() const {
            return coeff(0) * 0.212671f + coeff(1) * 0.715160f + coeff(2) * 0.072169f;
        }

    };

    /**
     * \brief Represents a linear RGB color and a weight
     *
     * This is used by Nori's image reconstruction filter code
     */
    struct ColorGPU4f : public Eigen::Array4f {
    public:
        typedef Eigen::Array4f Base;

        /// Create an zero value
        CUDA_HOSTDEV
        ColorGPU4f() : Base(0.0f, 0.0f, 0.0f, 0.0f) { }

        /// Create from a 3-channel color
        CUDA_HOSTDEV
        ColorGPU4f(const ColorGPU3f &c) : Base(c.r(), c.g(), c.b(), 1.0f) { }

        /// Initialize the color vector with specific per-channel values
        CUDA_HOSTDEV
        ColorGPU4f(float r, float g, float b, float w) : Base(r, g, b, w) { }

        /// Construct a color vector from ArrayBase (needed to play nice with Eigen)
        template <typename Derived> 
        CUDA_HOSTDEV
        ColorGPU4f(const Eigen::ArrayBase<Derived>& p) 
            : Base(p) { }

        /// Assign a color vector from ArrayBase (needed to play nice with Eigen)
        
        template <typename Derived> 
        CUDA_HOSTDEV
        ColorGPU4f &operator=(const Eigen::ArrayBase<Derived>& p) {
            this->Base::operator=(p);
            return *this;
        }

        
        /// Divide by the filter weight and convert into a \ref Color3f value
        CUDA_HOSTDEV
        ColorGPU3f divideByFilterWeight() const {
            if (w() != 0)
                return head<3>() / w();
            else
                return ColorGPU3f(0.0f);
        }
        
    };

}
