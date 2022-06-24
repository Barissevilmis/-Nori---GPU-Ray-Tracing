
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
#include <nori/bsdf.h>


namespace nori{
    /// Evaluate the BRDF model
    CUDA_DEV
    ColorGPU3f DiffuseKernel::eval(const BSDFQueryRecordGPU &bRec) const {
        /* This is a smooth BRDF -- return zero if the measure
        is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle
            || bRec.wi.z() <= 0
            || bRec.wo.z() <= 0)
            return ColorGPU3f(0.0f);

        /* The BRDF is simply the albedo / pi */
        return m_albedo_dev * INV_PI;
    }

    /// Compute the density of \ref sample() wrt. solid angles
    CUDA_DEV
    float DiffuseKernel::pdf(const BSDFQueryRecordGPU &bRec) const {
        /* This is a smooth BRDF -- return zero if the measure
        is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle
            || bRec.wi.z() <= 0
            || bRec.wo.z() <= 0)
            return 0.0f;


        /* Importance sampling density wrt. solid angles:
        cos(theta) / pi.

        Note that the directions in 'bRec' are in local coordinates,
        so Frame::cosTheta() actually just returns the 'z' component.
        */
        return INV_PI * bRec.wo.z();
    }

    /// Draw a a sample from the BRDF model
    CUDA_DEV
    ColorGPU3f DiffuseKernel::sample(BSDFQueryRecordGPU &bRec, const PointGPU2f &sample) const {
        if (bRec.wi.z() <= 0)
            return ColorGPU3f(0.0f);

        bRec.measure = ESolidAngle;

        /* Warp a uniformly distributed sample on [0,1]^2
        to a direction on a cosine-weighted hemisphere */

        float param_0 = acosf(sqrtf(1 - sample.x()));
        float param_1 = 2 * M_PI * sample.y();
        bRec.wo = VectorGPU3f(sinf(param_0) * cosf(param_1), sinf(param_0) * sinf(param_1), cosf(param_0));

        /* Relative index of refraction: no change */
        bRec.eta = 1.0f;

        /* eval() / pdf() * cos(theta) = albedo. There
        is no need to call these functions. */
        return ColorGPU3f(m_albedo_dev);
    }

    NORI_REGISTER_CLASS(DiffuseBSDF, "diffuse");
}
