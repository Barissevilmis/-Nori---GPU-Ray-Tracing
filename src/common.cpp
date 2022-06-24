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

#include <nori/common.h>
#include <nori/object.h>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <filesystem/resolver.h>
#include <iomanip>

#if defined(PLATFORM_LINUX)
#include <malloc.h>
#endif

#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#endif

#if defined(PLATFORM_MACOS)
#include <sys/sysctl.h>
#endif

namespace nori{

    std::string indent(const std::string &string, int amount) {
        /* This could probably be done faster (it's not
        really speed-critical though) */
        std::istringstream iss(string);
        std::ostringstream oss;
        std::string spacer(amount, ' ');
        bool firstLine = true;
        for (std::string line; std::getline(iss, line); ) {
            if (!firstLine)
                oss << spacer;
            oss << line;
            if (!iss.eof())
                oss << endl;
            firstLine = false;
        }
        return oss.str();
    }

    bool endsWith(const std::string &value, const std::string &ending) {
        if (ending.size() > value.size())
            return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    }

    std::string toLower(const std::string &value) {
        std::string result;
        result.resize(value.size());
        std::transform(value.begin(), value.end(), result.begin(), ::tolower);
        return result;
    }

    bool toBool(const std::string &str) {
        std::string value = toLower(str);
        if (value == "false")
            return false;
        else if (value == "true")
            return true;
        else
            throw NoriException("Could not parse boolean value \"%s\"", str);
    }

    int toInt(const std::string &str) {
        char *end_ptr = nullptr;
        int result = (int) strtol(str.c_str(), &end_ptr, 10);
        if (*end_ptr != '\0')
            throw NoriException("Could not parse integer value \"%s\"", str);
        return result;
    }

    unsigned int toUInt(const std::string &str) {
        char *end_ptr = nullptr;
        unsigned int result = (int) strtoul(str.c_str(), &end_ptr, 10);
        if (*end_ptr != '\0')
            throw NoriException("Could not parse integer value \"%s\"", str);
        return result;
    }

    float toFloat(const std::string &str) {
        char *end_ptr = nullptr;
        float result = (float) strtof(str.c_str(), &end_ptr);
        if (*end_ptr != '\0')
            throw NoriException("Could not parse floating point value \"%s\"", str);
        return result;
    }

    Eigen::Vector3f toVector3f(const std::string &str) {
        std::vector<std::string> tokens = tokenize(str);
        if (tokens.size() != 3)
            throw NoriException("Expected 3 values");
        Eigen::Vector3f result;
        for (int i=0; i<3; ++i)
            result[i] = toFloat(tokens[i]);
        return result;
    }

    std::vector<std::string> tokenize(const std::string &string, const std::string &delim, bool includeEmpty) {
        std::string::size_type lastPos = 0, pos = string.find_first_of(delim, lastPos);
        std::vector<std::string> tokens;

        while (lastPos != std::string::npos) {
            if (pos != lastPos || includeEmpty)
                tokens.push_back(string.substr(lastPos, pos - lastPos));
            lastPos = pos;
            if (lastPos != std::string::npos) {
                lastPos += 1;
                pos = string.find_first_of(delim, lastPos);
            }
        }

        return tokens;
    }

    std::string timeString(double time, bool precise) {
        if (std::isnan(time) || std::isinf(time))
            return "inf";

        std::string suffix = "ms";
        if (time > 1000) {
            time /= 1000; suffix = "s";
            if (time > 60) {
                time /= 60; suffix = "m";
                if (time > 60) {
                    time /= 60; suffix = "h";
                    if (time > 12) {
                        time /= 12; suffix = "d";
                    }
                }
            }
        }

        std::ostringstream os;
        os << std::setprecision(precise ? 4 : 1)
        << std::fixed << time << suffix;

        return os.str();
    }

    std::string memString(size_t size, bool precise) {
        double value = (double) size;
        const char *suffixes[] = {
            "B", "KiB", "MiB", "GiB", "TiB", "PiB"
        };
        int suffix = 0;
        while (suffix < 5 && value > 1024.0f) {
            value /= 1024.0f; ++suffix;
        }

        std::ostringstream os;
        os << std::setprecision(suffix == 0 ? 0 : (precise ? 4 : 1))
        << std::fixed << value << " " << suffixes[suffix];

        return os.str();
    }

    filesystem::resolver *getFileResolver() {
        static filesystem::resolver *resolver = new filesystem::resolver();
        return resolver;
    }

   
    Transform::Transform(const Eigen::Matrix4f &trafo)
        : m_transform(trafo), m_inverse(trafo.inverse()) { }

    std::string Transform::toString() const {
        std::ostringstream oss;
        oss << m_transform.format(Eigen::IOFormat(4, 0, ", ", ";\n", "", "", "[", "]"));
        return oss.str();
    }

    Transform Transform::operator*(const Transform &t) const {
        return Transform(m_transform * t.m_transform,
            t.m_inverse * m_inverse);
    }

    VectorGPU3f sphericalDirection(float theta, float phi) {
        float sinTheta, cosTheta, sinPhi, cosPhi;

        sincosf(theta, &sinTheta, &cosTheta);
        sincosf(phi, &sinPhi, &cosPhi);

        return VectorGPU3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta );
    }

    PointGPU2f sphericalCoordinates(const VectorGPU3f &v) {
        PointGPU2f result(
            std::acos(v.z()),
            std::atan2(v.y(), v.x())
        );
        if (result.y() < 0)
            result.y() += 2*M_PI;
        return result;
    }


    float fresnel(float cosThetaI, float extIOR, float intIOR) {
        float etaI = extIOR, etaT = intIOR;

        if (extIOR == intIOR)
            return 0.0f;

        /* Swap the indices of refraction if the interaction starts
        at the inside of the object */
        if (cosThetaI < 0.0f) {
            std::swap(etaI, etaT);
            cosThetaI = -cosThetaI;
        }

        /* Using Snell's law, calculate the squared sine of the
        angle between the normal and the transmitted ray */
        float eta = etaI / etaT,
            sinThetaTSqr = eta*eta * (1-cosThetaI*cosThetaI);

        if (sinThetaTSqr > 1.0f)
            return 1.0f;  /* Total internal reflection! */

        float cosThetaT = std::sqrt(1.0f - sinThetaTSqr);

        float Rs = (etaI * cosThetaI - etaT * cosThetaT)
                / (etaI * cosThetaI + etaT * cosThetaT);
        float Rp = (etaT * cosThetaI - etaI * cosThetaT)
                / (etaT * cosThetaI + etaI * cosThetaT);

        return (Rs * Rs + Rp * Rp) / 2.0f;
    }

}
