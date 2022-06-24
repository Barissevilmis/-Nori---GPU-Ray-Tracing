#include <nori/camera.h>

namespace nori{

    /**
     * \brief Generic camera interface
     *
     * This class provides an abstract interface to cameras in Nori and
     * exposes the ability to sample their response function. By default, only
     * a perspective camera implementation exists, but you may choose to
     * implement other types (e.g. an environment camera, or a physically-based
     * camera model that simulates the behavior actual lenses)
     */

    PerspectiveCamera::PerspectiveCamera(const PropertyList &propList)
    {
        /* Width and height in pixels. Default: 720p */
        m_outputSize = (VectorGPU2i *)(malloc(sizeof(VectorGPU2i)));
        m_invOutputSize = (VectorGPU2f *)(malloc(sizeof(VectorGPU2f)));
        m_outputSize->x() = propList.getInteger("width", 1280);
        m_outputSize->y() = propList.getInteger("height", 720);
        *m_invOutputSize = m_outputSize->cast<float>().cwiseInverse();

        m_cameraToWorld = propList.getTransform("toWorld", Transform());
        /* Specifies an optional camera-to-world transformation. Default: none */

        m_fov = (float *)(malloc(sizeof(float)));
        /* Horizontal field of view in degrees */
        *m_fov = propList.getFloat("fov", 30.0f);

        /* Near and far clipping planes in world-space units */
        m_nearClip = (float *)(malloc(sizeof(float)));
        m_farClip = (float *)(malloc(sizeof(float)));
        *m_nearClip = propList.getFloat("nearClip", 1e-4f);
        *m_farClip = propList.getFloat("farClip", 1e4f);

        m_rfilter_gg = NULL;
    }

    void PerspectiveCamera::activate()
    {
        float aspect = m_outputSize->x() / (float)m_outputSize->y();
        /* Project vectors in camera space onto a plane at z=1:
         *
         *  xProj = cot * x / z
         *  yProj = cot * y / z
         *  zProj = (far * (z - near)) / (z * (far-near))
         *  The cotangent factor ensures that the field of view is
         *  mapped to the interval [-1, 1].
         */
        float recip = 1.0f / (*m_farClip - *m_nearClip),
              cot = 1.0f / std::tan(degToRad(*m_fov / 2.0f));
        Eigen::Matrix4f perspective;
        perspective << 
            cot, 0, 0, 0,
            0, cot, 0, 0,
            0, 0, *m_farClip * recip, (*m_nearClip * -1) * *m_farClip * recip,
            0, 0, 1, 0;
        /**
         * Translation and scaling to shift the clip coordinates into the
         * range from zero to one. Also takes the aspect ratio into account.
         */
        m_sampleToCamera = Transform(
                                Eigen::DiagonalMatrix<float, 3>(VectorGPU3f(-0.5f, -0.5f * aspect, 1.0f)) *
                                Eigen::Translation<float, 3>(-1.0f, -1.0f / aspect, 0.0f) * perspective).inverse();
        /* If no reconstruction filter was assigned, instantiate a Gaussian filter */
        if (!m_rfilter_gg)
            m_rfilter_gg = static_cast<GaussianFilter *>(NoriObjectFactory::createInstance("gaussian", PropertyList()));
    }
}