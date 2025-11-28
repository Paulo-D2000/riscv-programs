#ifndef CORDIC_H
#define CORDIC_H

#include <stdint.h>

#define CORDIC_ITER 24
#define PI 3.14159265f
#define HALF_PI (PI / 2)
#define TWO_PI (2 * PI)

// Precomputed arctangent values (atan(2^-i)) in radians
static const float cordic_atan[] = {
    0.78539816f, 0.46364761f, 0.24497866f, 0.12435499f,
    0.06241881f, 0.03123983f, 0.01562373f, 0.00781234f,
    0.00390623f, 0.00195312f, 0.00097656f, 0.00048828f,
    0.00024414f, 0.00012207f, 0.00006104f, 0.00003052f,
    0.00001526f, 0.00000763f, 0.00000381f, 0.00000191f,
    0.00000095f, 0.00000048f, 0.00000024f, 0.00000012f
};

// Normalize to [-π, π]
static inline float cordic_wrap_pi(float x) {
    while (x > PI) x -= TWO_PI;
    while (x < -PI) x += TWO_PI;
    return x;
}

// Core CORDIC rotation kernel for sin/cos
static inline void cordic_rotate(float theta, float* out_cos, float* out_sin) {
    theta = cordic_wrap_pi(theta);

    float x = 0.60725293f;  // CORDIC gain correction factor
    float y = 0.0f;
    float angle = 0.0f;

    for (int i = 0; i < CORDIC_ITER; i++) {
        float dx, dy;
        float shift = 1.0f / (1 << i);
        if (theta > angle) {
            dx = x - y * shift;
            dy = y + x * shift;
            angle += cordic_atan[i];
        } else {
            dx = x + y * shift;
            dy = y - x * shift;
            angle -= cordic_atan[i];
        }
        x = dx;
        y = dy;
    }

    if (out_cos) *out_cos = x;
    if (out_sin) *out_sin = y;
}

static inline float sin(float theta) {
    float s;
    cordic_rotate(theta, 0, &s);
    return s;
}

static inline float cos(float theta) {
    float c;
    cordic_rotate(theta, &c, 0);
    return c;
}

static inline float tan(float theta) {
    theta = cordic_wrap_pi(theta);
    if (theta > HALF_PI - 0.01f || theta < -HALF_PI + 0.01f) {
        return 1e9f; // Approaching asymptote
    }
    float s, c;
    cordic_rotate(theta, &c, &s);
    return s / c;
}

#endif // CORDIC_H