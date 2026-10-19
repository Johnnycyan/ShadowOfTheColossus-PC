// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: NVIDIA DLSS (DLAA) in place of Shadow of the Colossus's TAA, compute shader 0xa402ddac.
// Its images (cs_img0..6): 0 scene colour (linear HDR, RGBA16F), 1/2 depth and stencil of the
// D32S8 depth buffer, 3 velocity (RG16F, UV units), 4 history, 5 mask, 6 output. With
// SOTC_DLSS=1, the bridge DLLs and the "dlss" eboot mode (camera jitter copied to image offset
// 0x395f000, scripts/sotc_modes.py), the dispatch becomes a DLSS evaluation that writes the
// output image; the game's post-processing reads it as before. Conventions the capture could not
// settle can be overridden: SOTC_DLSS_MV=x,y and SOTC_DLSS_JITTER=x,y (scale and sign, applied
// to the render size), SOTC_DLSS_DEPTH_INV=0/1, SOTC_DLSS_JITTER_LAG=n (use the jitter read n
// TAA passes earlier: the game's CPU can be ahead of the GPU thread).

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <utility>

#include "../../../bbgpu.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {
constexpr u64 TaaShader = 0xa402ddacull;
constexpr uintptr_t JitterOffset = 0x395f000;

struct DlssTaaSettings {
    bool enabled = false;
    // Checked in game (2026-10-19): the velocity is current minus previous in UV units, and the
    // projection jitter's x and y point the other way from DLSS's sample offset in pixels.
    float mv_x = -1.0f, mv_y = -1.0f;
    float jitter_x = -0.5f, jitter_y = 0.5f;
    bool depth_inverted = true;
    u32 jitter_lag = 0;
};

void ReadPair(const char* name, float& x, float& y) {
    if (const char* v = std::getenv(name)) {
        std::sscanf(v, "%f,%f", &x, &y);
    }
}

const DlssTaaSettings& Settings() {
    static const DlssTaaSettings settings = [] {
        DlssTaaSettings s;
        const char* enabled = std::getenv("SOTC_DLSS");
        s.enabled = enabled && enabled[0] == '1';
        ReadPair("SOTC_DLSS_MV", s.mv_x, s.mv_y);
        ReadPair("SOTC_DLSS_JITTER", s.jitter_x, s.jitter_y);
        if (const char* v = std::getenv("SOTC_DLSS_DEPTH_INV")) {
            s.depth_inverted = v[0] != '0';
        }
        if (const char* v = std::getenv("SOTC_DLSS_JITTER_LAG")) {
            s.jitter_lag = static_cast<u32>(std::clamp(std::atoi(v), 0, 7));
        }
        if (s.enabled) {
            std::fprintf(stderr,
                         "DLSS TAA: mv %g,%g jitter %g,%g depth inverted %d jitter lag %u\n",
                         s.mv_x, s.mv_y, s.jitter_x, s.jitter_y, int(s.depth_inverted),
                         s.jitter_lag);
        }
        return s;
    }();
    return settings;
}
// SOTC_DLSS_TUNE=<file>: "mv_x mv_y jitter_x jitter_y lag depth_inverted", re-read every 30
// TAA passes, to find the conventions while the game runs.
const DlssTaaSettings& TunedSettings() {
    static DlssTaaSettings tuned = Settings();
    static const char* file = std::getenv("SOTC_DLSS_TUNE");
    static u32 calls = 0;
    if (file && calls++ % 30 == 0) {
        if (FILE* f = std::fopen(file, "r")) {
            DlssTaaSettings t = tuned;
            int lag = int(t.jitter_lag), inverted = int(t.depth_inverted);
            if (std::fscanf(f, "%f %f %f %f %d %d", &t.mv_x, &t.mv_y, &t.jitter_x, &t.jitter_y,
                            &lag, &inverted) >= 4) {
                t.jitter_lag = static_cast<u32>(std::clamp(lag, 0, 7));
                t.depth_inverted = inverted != 0;
                if (std::memcmp(&t, &tuned, sizeof(t)) != 0) {
                    std::fprintf(stderr, "DLSS TAA: tuned mv %g,%g jitter %g,%g lag %u depth inverted %d\n",
                                 t.mv_x, t.mv_y, t.jitter_x, t.jitter_y, t.jitter_lag,
                                 int(t.depth_inverted));
                }
                tuned = t;
            }
            std::fclose(f);
        }
    }
    return tuned;
}
} // namespace

bool Rasterizer::DispatchDlssTaa(const Shader::Info& cs) {
    if (cs.pgm_hash != TaaShader) {
        return false;
    }
    const auto& settings = TunedSettings();
    if (!settings.enabled || image_bindings.size() < 7 || !bbgpu_guest_image_base) {
        return false;
    }
    Dlss* dlss = Dlss::Get();
    if (!dlss || !dlss->Available()) {
        return false;
    }
    const auto resource = [&](size_t index, vk::ImageAspectFlags aspect, Dlss::Resource& out) {
        const auto& [id, desc] = image_bindings[index];
        if (!id) {
            return false;
        }
        auto& image = texture_cache.GetImage(id);
        auto& view = texture_cache.FindTexture(id, desc);
        out = Dlss::Resource{image.GetImage(),         *view.image_view,
                             image.info.pixel_format,  aspect,
                             image.info.size.width,    image.info.size.height};
        return true;
    };
    Dlss::Frame frame{};
    if (!resource(0, vk::ImageAspectFlagBits::eColor, frame.color) ||
        !resource(1, vk::ImageAspectFlagBits::eDepth, frame.depth) ||
        !resource(3, vk::ImageAspectFlagBits::eColor, frame.motion) ||
        !resource(6, vk::ImageAspectFlagBits::eColor, frame.output)) {
        return false;
    }

    const Dlss::FeatureDesc desc{frame.color.width,  frame.color.height, frame.output.width,
                                 frame.output.height, Dlss::QualityForScale(1.0f), true,
                                 settings.depth_inverted};
    // Feature creation and evaluation record through the driver (NGX): with the recording
    // thread they run there, on the real command buffer (Scheduler::RecordNative).
    static std::optional<Dlss::FeatureDesc> created;
    if (created != desc) {
        // A new size: the GPU must be done with the previous feature. That submits the command
        // buffer DispatchDirect holds, so from here on the game's dispatch cannot be the fallback.
        if (created) {
            scheduler.Finish();
        }
        created = desc;
        scheduler.RecordNative([dlss, desc](vk::CommandBuffer cmdbuf) {
            const bool ok = dlss->CreateFeature(cmdbuf, desc);
            std::fprintf(stderr, "DLSS TAA: feature %ux%u -> %ux%u %s\n", desc.input_width,
                         desc.input_height, desc.output_width, desc.output_height,
                         ok ? "created" : "FAILED");
        });
    }

    // The game's jitter in projection units, (2 * Halton - 1) * (s / width, s / height).
    static std::array<std::array<float, 2>, 8> history{};
    static u32 count = 0;
    std::memcpy(history[count % history.size()].data(),
                reinterpret_cast<const void*>(bbgpu_guest_image_base + JitterOffset),
                sizeof(float) * 2);
    const auto& jitter = history[(count + history.size() - settings.jitter_lag) % history.size()];
    ++count;
    const float width = static_cast<float>(frame.color.width);
    const float height = static_cast<float>(frame.color.height);
    frame.jitter_x = jitter[0] * width * settings.jitter_x;
    frame.jitter_y = jitter[1] * height * settings.jitter_y;
    frame.mv_scale_x = width * settings.mv_x;
    frame.mv_scale_y = height * settings.mv_y;
    frame.reset = count == 1;
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    frame.frame_ms = std::chrono::duration<float, std::milli>(now - last).count();
    last = now;
    frame.sharpness = 0.0f;
    if (count <= 3) {
        std::fprintf(stderr, "DLSS TAA: jitter %g,%g px\n", frame.jitter_x, frame.jitter_y);
    }
    scheduler.RecordNative([dlss, frame](vk::CommandBuffer cmdbuf) {
        static bool failed = false;
        if (!dlss->Evaluate(cmdbuf, frame) && !std::exchange(failed, true)) {
            std::fprintf(stderr, "DLSS TAA: evaluation failed\n");
        }
    });
    return true;
}

} // namespace Vulkan
