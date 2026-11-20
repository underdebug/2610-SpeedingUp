#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace nurbs {
struct Vec3 {
    double x, y, z;
};

struct Vec4 {
    double x, y, z, w;
};

struct Surface {
    int nu = 0, nv = 0, pu = 0, pv = 0;
    std::vector<double> u, v;
    std::vector<Vec4> cp;  // homogeneous coordinates: (w*x,w*y,w*z,w)

    const Vec4& at(int i, int j) const {
        return cp.at(j * nu + i);
    }
};

inline Vec4 mix(Vec4 a, Vec4 b, double t) {
    return {(1 - t) * a.x + t * b.x, (1 - t) * a.y + t * b.y,
            (1 - t) * a.z + t * b.z, (1 - t) * a.w + t * b.w};
}

inline Vec3 project(Vec4 q) {
    if (std::abs(q.w) < 1e-14) {
        throw std::runtime_error("NURBS zero weight");
    }
    return {q.x / q.w, q.y / q.w, q.z / q.w};
}

inline int span(int n, int p, double t, const std::vector<double>& knot) {
    if (t >= knot[n + 1]) {
        return n;
    }
    if (t <= knot[p]) {
        return p;
    }
    int low = p, high = n + 1;
    while (high - low > 1) {
        int mid = (low + high) / 2;
        if (t < knot[mid]) {
            high = mid;
        } else {
            low = mid;
        }
    }
    return low;
}

inline std::vector<double> basis(int p, int k, double t,
                                 const std::vector<double>& U) {
    // Cox-de Boor stable, nonrecursive basis-function table (The NURBS Book
    // A2.2).
    std::vector<double> N(p + 1), left(p + 1), right(p + 1);
    N[0] = 1;
    for (int j = 1; j <= p; ++j) {
        left[j] = t - U[k + 1 - j];
        right[j] = U[k + j] - t;
        double saved = 0;
        for (int r = 0; r < j; ++r) {
            double den = right[r + 1] + left[j - r];
            double temp = (std::abs(den) > 1e-14) ? N[r] / den : 0;
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
    return N;
}

inline Vec3 coxDeBoor(const Surface& s, double u, double v) {
    int ku = span(s.nu - 1, s.pu, u, s.u), kv = span(s.nv - 1, s.pv, v, s.v);
    auto Nu = basis(s.pu, ku, u, s.u), Nv = basis(s.pv, kv, v, s.v);
    Vec4 out{0, 0, 0, 0};
    for (int j = 0; j <= s.pv; ++j) {
        for (int i = 0; i <= s.pu; ++i) {
            const auto& c = s.at(ku - s.pu + i, kv - s.pv + j);
            double w = Nu[i] * Nv[j];
            out.x += w * c.x;
            out.y += w * c.y;
            out.z += w * c.z;
            out.w += w * c.w;
        }
    }
    return project(out);
}

inline Vec4 deBoor1D(const std::vector<Vec4>& points, int p,
                     const std::vector<double>& U, double t) {
    int k = span(static_cast<int>(points.size()) - 1, p, t, U);
    std::vector<Vec4> d(p + 1);
    for (int j = 0; j <= p; ++j) {
        d[j] = points[k - p + j];
    }
    for (int r = 1; r <= p; ++r) {
        for (int j = p; j >= r; --j) {
            int i = k - p + j;
            double den = U[i + p - r + 1] - U[i]; // MINGJIE i + p - r + 1
            double alpha = (std::abs(den) > 1e-14) ? (t - U[i]) / den : 0.0;
            d[j] = mix(d[j - 1], d[j], alpha);
        }
    }
    return d[p];
}

inline Vec3 deBoor(const Surface& s, double u, double v) {
    // First interpolate every V-row along U, then interpolate results along V.
    std::vector<Vec4> column(s.nv);
    std::vector<Vec4> row(s.nu);
    for (int j = 0; j < s.nv; ++j) {
        for (int i = 0; i < s.nu; ++i) {
            row[i] = s.at(i, j);
        }
        column[j] = deBoor1D(row, s.pu, s.u, u);
    }
    return project(deBoor1D(column, s.pv, s.v, v));
}

inline std::vector<double> clampedKnots(int n, int p) {
    // n control points, p degree, uniform clamped vector, length n+p+1.
    if (n <= p) {
        throw std::runtime_error("Need more control points than degree");
    }
    std::vector<double> U(n + p + 1);
    for (int i = 0; i <= p; ++i) {
        U[i] = 0;
    }
    for (int i = p + 1; i < n; ++i) {
        U[i] = double(i - p) / double(n - p);
    }
    for (int i = n; i < n + p + 1; ++i) {
        U[i] = 1;
    }
    return U;
}

inline Surface exampleSurface() {
    Surface s;
    s.nu = 7;
    s.nv = 7;
    s.pu = 3;
    s.pv = 3;
    s.u = clampedKnots(s.nu, s.pu);
    s.v = clampedKnots(s.nv, s.pv);
    for (int j = 0; j < s.nv; ++j) {
        for (int i = 0; i < s.nu; ++i) {
            double x = (i - 3) * 0.65, y = (j - 3) * 0.65;
            double z = 0.8 * std::exp(-0.22 * (x * x + y * y)) +
                       0.25 * std::sin(1.5 * x) * std::cos(1.8 * y);
            double w = (i == 3 && j == 3) ? 1.75 : 1.0;
            s.cp.push_back({w * x, w * y, w * z, w});
        }
    }
    return s;
}
}  // namespace nurbs
