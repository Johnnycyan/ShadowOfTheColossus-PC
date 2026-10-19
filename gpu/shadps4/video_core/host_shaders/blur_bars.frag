// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: one dual Kawase step of the blurred side bars (BlurBarsPass). up = 0 halves the
// image, up = 1 doubles it back; delta is one texel of the source.

#version 450

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D src;

layout (push_constant) uniform settings {
    vec2 delta;
    uint up;
} pp;

void main() {
    const vec2 d = pp.delta;
    vec3 c;
    if (pp.up == 0u) {
        c = texture(src, uv).rgb * 4.0;
        c += texture(src, uv + vec2(0.5, 0.5) * d).rgb;
        c += texture(src, uv + vec2(-0.5, 0.5) * d).rgb;
        c += texture(src, uv + vec2(-0.5, -0.5) * d).rgb;
        c += texture(src, uv + vec2(0.5, -0.5) * d).rgb;
        c /= 8.0;
    } else {
        c = (texture(src, uv + vec2(0.5, 0.5) * d).rgb + texture(src, uv + vec2(-0.5, 0.5) * d).rgb +
             texture(src, uv + vec2(-0.5, -0.5) * d).rgb + texture(src, uv + vec2(0.5, -0.5) * d).rgb) * 2.0;
        c += texture(src, uv + vec2(-1.0, 0.0) * d).rgb + texture(src, uv + vec2(1.0, 0.0) * d).rgb;
        c += texture(src, uv + vec2(0.0, 1.0) * d).rgb + texture(src, uv + vec2(0.0, -1.0) * d).rgb;
        c /= 12.0;
    }
    color = vec4(c, 1.0);
}
