// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: NVIDIA DLSS for Shadow of the Colossus, through the game's TAA (compute shader 0xa402ddac)
// and its final composite (fragment shader 0xdfd767b6).
// TAA images (cs_img0..6): 0 scene colour (linear HDR, RGBA16F), 1/2 depth and stencil of the
// D32S8 depth buffer, 3 velocity (RG16F, UV units), 4 history, 5 mask, 6 output. The composite
// reads the UI (RGBA8 sRGB, display size) and the post-processed scene (RGBA16F, render size) and
// writes the display buffer.
// SOTC_DLSS=1 (DLAA): the TAA dispatch becomes a DLSS evaluation writing the output image at the
// render size; the game's post-processing reads it as before.
// SOTC_DLSS=2 (upscaling, render size below the display size): the TAA passes the jittered colour
// through, the post-processing runs on it, and at the composite DLSS upscales the scene to the
// display size into an image of its own, which the composite samples instead.
// Both need the bridge DLLs and the "dlss" eboot mode (camera jitter copied to image offset
// 0x395f000, scripts/sotc_modes.py). Overrides: SOTC_DLSS_MV=x,y and SOTC_DLSS_JITTER=x,y (scale
// and sign, applied to the render size), SOTC_DLSS_DEPTH_INV=0/1, SOTC_DLSS_JITTER_LAG=n (the
// jitter read n TAA passes earlier), SOTC_DLSS_TUNE=<file> (the same, re-read while running).

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <utility>

#include "../../../bbgpu.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {
constexpr uintptr_t JitterOffset = 0x395f000;

struct DlssTaaSettings {
    int mode = 0; // 0 off, 1 DLAA at the TAA, 2 upscaling at the composite
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
        if (const char* v = std::getenv("SOTC_DLSS")) {
            s.mode = std::clamp(std::atoi(v), 0, 2);
        }
        ReadPair("SOTC_DLSS_MV", s.mv_x, s.mv_y);
        ReadPair("SOTC_DLSS_JITTER", s.jitter_x, s.jitter_y);
        if (const char* v = std::getenv("SOTC_DLSS_DEPTH_INV")) {
            s.depth_inverted = v[0] != '0';
        }
        if (const char* v = std::getenv("SOTC_DLSS_JITTER_LAG")) {
            s.jitter_lag = static_cast<u32>(std::clamp(std::atoi(v), 0, 7));
        }
        if (s.mode) {
            std::fprintf(stderr,
                         "DLSS: %s, mv %g,%g jitter %g,%g depth inverted %d jitter lag %u\n",
                         s.mode == 1 ? "DLAA at the TAA" : "upscaling at the composite", s.mv_x,
                         s.mv_y, s.jitter_x, s.jitter_y, int(s.depth_inverted), s.jitter_lag);
        }
        return s;
    }();
    return settings;
}

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
                    std::fprintf(stderr, "DLSS: tuned mv %g,%g jitter %g,%g lag %u depth inverted %d\n",
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

/// The camera jitter of the frame whose TAA runs now, in projection units.
std::array<float, 2> ReadJitter(u32 lag) {
    static std::array<std::array<float, 2>, 8> history{};
    static u32 count = 0;
    std::memcpy(history[count % history.size()].data(),
                reinterpret_cast<const void*>(bbgpu_guest_image_base + JitterOffset),
                sizeof(float) * 2);
    const auto jitter = history[(count + history.size() - lag) % history.size()];
    ++count;
    return jitter;
}

float FrameMs() {
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(now - last).count();
    last = now;
    return ms;
}

/// What the upscaling path keeps from the TAA pass until the composite of the same frame.
struct UpscaleFrame {
    bool valid = false;
    VideoCore::ImageId depth_id, motion_id;
    VideoCore::TextureCache::ImageDesc depth_desc, motion_desc;
    float jitter_x = 0, jitter_y = 0, mv_scale_x = 1, mv_scale_y = 1;
    float frame_ms = 0;
    bool reset = true;
    bool depth_inverted = true;
    u32 render_width = 0, render_height = 0;
};
UpscaleFrame upscale;

/// The display-size image DLSS writes in upscaling mode.
struct UpscaleTarget {
    std::optional<VideoCore::UniqueImage> image;
    vk::UniqueImageView view;
    u32 width = 0, height = 0;
};
UpscaleTarget target;
} // namespace

bool Rasterizer::DispatchDlssTaa(const Shader::Info& cs) {
    const auto& settings = TunedSettings();
    if (!settings.mode || image_bindings.size() != 7 || !bbgpu_guest_image_base) {
        return false;
    }
    // The TAA by its images (0xa402ddac when the render and display sizes match; another hash
    // with a smaller render size): RGBA16F colour, D32S8 depth and stencil, RG16F velocity,
    // RGBA16F history, RGBA8 mask, RGBA16F output, all of one size.
    static constexpr std::array<vk::Format, 7> TaaFormats{
        vk::Format::eR16G16B16A16Sfloat, vk::Format::eD32SfloatS8Uint,
        vk::Format::eD32SfloatS8Uint,    vk::Format::eR16G16Sfloat,
        vk::Format::eR16G16B16A16Sfloat, vk::Format::eR8G8B8A8Unorm,
        vk::Format::eR16G16B16A16Sfloat};
    u32 taa_width = 0, taa_height = 0;
    for (size_t i = 0; i < TaaFormats.size(); ++i) {
        if (!image_bindings[i].first) {
            return false;
        }
        const auto& info = texture_cache.GetImage(image_bindings[i].first).info;
        if (info.pixel_format != TaaFormats[i] ||
            (i && (info.size.width != taa_width || info.size.height != taa_height))) {
            return false;
        }
        taa_width = info.size.width;
        taa_height = info.size.height;
    }
    static u64 logged = 0;
    if (std::exchange(logged, cs.pgm_hash) != cs.pgm_hash) {
        std::fprintf(stderr, "DLSS: TAA cs %#llx at %ux%u\n",
                     static_cast<unsigned long long>(cs.pgm_hash), taa_width, taa_height);
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

    // The game's jitter, (2 * Halton - 1) * (s / width, s / height), in render pixels.
    const auto jitter = ReadJitter(settings.jitter_lag);
    const float width = static_cast<float>(frame.color.width);
    const float height = static_cast<float>(frame.color.height);
    frame.jitter_x = jitter[0] * width * settings.jitter_x;
    frame.jitter_y = jitter[1] * height * settings.jitter_y;
    frame.mv_scale_x = width * settings.mv_x;
    frame.mv_scale_y = height * settings.mv_y;
    frame.frame_ms = FrameMs();
    frame.sharpness = 0.0f;

    if (settings.mode == 2) {
        // Pass the jittered colour through; the composite upscales the post-processed frame.
        runtime.CopyImage(&texture_cache.GetImage(image_bindings[0].first),
                          &texture_cache.GetImage(image_bindings[6].first));
        upscale.valid = true;
        upscale.depth_id = image_bindings[1].first;
        upscale.depth_desc = image_bindings[1].second;
        upscale.motion_id = image_bindings[3].first;
        upscale.motion_desc = image_bindings[3].second;
        upscale.jitter_x = frame.jitter_x;
        upscale.jitter_y = frame.jitter_y;
        upscale.mv_scale_x = frame.mv_scale_x;
        upscale.mv_scale_y = frame.mv_scale_y;
        upscale.frame_ms = frame.frame_ms;
        upscale.depth_inverted = settings.depth_inverted;
        upscale.render_width = frame.color.width;
        upscale.render_height = frame.color.height;
        static bool shown = false;
        if (!std::exchange(shown, true)) {
            std::fprintf(stderr, "DLSS: TAA passes %ux%u through for upscaling\n", frame.color.width,
                         frame.color.height);
        }
        return true;
    }

    // DLAA. Feature creation and evaluation record through the driver (NGX): with the recording
    // thread they run there, on the real command buffer (Scheduler::RecordNative).
    const Dlss::FeatureDesc desc{frame.color.width,  frame.color.height, frame.output.width,
                                 frame.output.height, Dlss::QualityForScale(1.0f), true,
                                 settings.depth_inverted};
    static std::optional<Dlss::FeatureDesc> created;
    if (created != desc) {
        // A new size: the GPU must be done with the previous feature. That submits the command
        // buffer DispatchDirect holds, so from here on the game's dispatch cannot be the fallback.
        if (created) {
            scheduler.Finish();
        }
        created = desc;
        frame.reset = true;
        scheduler.RecordNative([dlss, desc](vk::CommandBuffer cmdbuf) {
            const bool ok = dlss->CreateFeature(cmdbuf, desc);
            std::fprintf(stderr, "DLSS: feature %ux%u -> %ux%u %s\n", desc.input_width,
                         desc.input_height, desc.output_width, desc.output_height,
                         ok ? "created" : "FAILED");
        });
    }
    scheduler.RecordNative([dlss, frame](vk::CommandBuffer cmdbuf) {
        static bool failed = false;
        if (!dlss->Evaluate(cmdbuf, frame) && !std::exchange(failed, true)) {
            std::fprintf(stderr, "DLSS: evaluation failed\n");
        }
    });
    return true;
}

void Rasterizer::DlssComposite(const GraphicsPipeline* pipeline) {
    if (!upscale.valid || TunedSettings().mode != 2) {
        return;
    }
    const auto stages = pipeline->GetStages();
    const auto* fragment = stages[static_cast<u32>(Shader::SwStage::Fragment)];
    if (!fragment) {
        return; // a depth-only draw
    }
    // The composite (fragment shader 0xdfd767b6 when the render and display sizes match, another
    // variant when they do not) is the first draw after the TAA that fills a target larger than
    // the render size while sampling a render-size RGBA16F image: the post-processed scene.
    if (liverpool->regs.screen_scissor.GetWidth() <= upscale.render_width) {
        return;
    }
    const ImageBindingInfo* scene = nullptr;
    for (const auto& binding : fragment_image_bindings) {
        if (!binding.first) {
            continue;
        }
        const auto& info = texture_cache.GetImage(binding.first).info;
        if (info.pixel_format == vk::Format::eR16G16B16A16Sfloat &&
            info.size.width == upscale.render_width && info.size.height == upscale.render_height) {
            scene = &binding;
        }
    }
    if (!scene) {
        return;
    }
    upscale.valid = false;
    static u64 logged = 0;
    if (std::exchange(logged, fragment->pgm_hash) != fragment->pgm_hash) {
        std::fprintf(stderr, "DLSS: composite fs %#llx\n",
                     static_cast<unsigned long long>(fragment->pgm_hash));
    }
    Dlss* dlss = Dlss::Get();
    if (!dlss || !dlss->Available() || !upscale.depth_id || !upscale.motion_id) {
        return;
    }
    auto& color_image = texture_cache.GetImage(scene->first);
    auto& color_view = texture_cache.FindTexture(scene->first, scene->second);
    auto& depth_image = texture_cache.GetImage(upscale.depth_id);
    auto& motion_image = texture_cache.GetImage(upscale.motion_id);
    // The composite fills the display buffer: its screen scissor is the DLSS output size.
    const u32 out_width = liverpool->regs.screen_scissor.GetWidth();
    const u32 out_height = liverpool->regs.screen_scissor.GetHeight();
    static u32 shown = 0;
    if (shown++ < 3) {
        std::fprintf(stderr, "DLSS: composite scene %ux%u, display %ux%u\n",
                     color_image.info.size.width, color_image.info.size.height, out_width, out_height);
    }
    if (out_width <= color_image.info.size.width) {
        return; // nothing to upscale: the composite samples the scene as it is
    }

    if (!target.image || target.width != out_width || target.height != out_height) {
        if (target.image) {
            scheduler.Finish();
        }
        target.view.reset();
        target.image.emplace(instance.GetDevice(), instance.GetAllocator());
        target.image->Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .extent = {out_width, out_height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        target.view = Check<"create DLSS output view">(instance.GetDevice().createImageViewUnique(
            vk::ImageViewCreateInfo{
                .image = *target.image,
                .viewType = vk::ImageViewType::e2D,
                .format = vk::Format::eR16G16B16A16Sfloat,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            }));
        target.width = out_width;
        target.height = out_height;
        upscale.reset = true;
    }

    // Depth and velocity were written earlier in the frame; DLSS samples them.
    runtime.Transit(&depth_image, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead,
                    upscale.depth_desc.view_info.range);
    runtime.Transit(&motion_image, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead,
                    upscale.motion_desc.view_info.range);
    runtime.FlushBarriers();
    scheduler.EndRendering();

    Dlss::Frame frame{};
    frame.color = {color_image.GetImage(), *color_view.image_view, color_image.info.pixel_format,
                   vk::ImageAspectFlagBits::eColor, color_image.info.size.width,
                   color_image.info.size.height};
    frame.depth = {depth_image.GetImage(),
                   *texture_cache.FindTexture(upscale.depth_id, upscale.depth_desc).image_view,
                   depth_image.info.pixel_format, vk::ImageAspectFlagBits::eDepth,
                   depth_image.info.size.width, depth_image.info.size.height};
    frame.motion = {motion_image.GetImage(),
                    *texture_cache.FindTexture(upscale.motion_id, upscale.motion_desc).image_view,
                    motion_image.info.pixel_format, vk::ImageAspectFlagBits::eColor,
                    motion_image.info.size.width, motion_image.info.size.height};
    frame.output = {*target.image, *target.view, vk::Format::eR16G16B16A16Sfloat,
                    vk::ImageAspectFlagBits::eColor, out_width, out_height};
    frame.jitter_x = upscale.jitter_x;
    frame.jitter_y = upscale.jitter_y;
    frame.mv_scale_x = upscale.mv_scale_x;
    frame.mv_scale_y = upscale.mv_scale_y;
    frame.frame_ms = upscale.frame_ms;
    frame.reset = std::exchange(upscale.reset, false);
    frame.sharpness = 0.0f;

    const Dlss::FeatureDesc desc{
        frame.color.width, frame.color.height, out_width, out_height,
        Dlss::QualityForScale(static_cast<float>(out_width) / frame.color.width), true,
        upscale.depth_inverted};
    static std::optional<Dlss::FeatureDesc> created;
    const bool create = created != desc;
    if (create && created) {
        scheduler.Finish();
    }
    created = desc;
    const vk::Image output = *target.image;
    scheduler.RecordNative([dlss, desc, frame, create, output](vk::CommandBuffer cmdbuf) {
        if (create) {
            const bool ok = dlss->CreateFeature(cmdbuf, desc);
            std::fprintf(stderr, "DLSS: upscaling feature %ux%u -> %ux%u %s\n", desc.input_width,
                         desc.input_height, desc.output_width, desc.output_height,
                         ok ? "created" : "FAILED");
        }
        const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        const vk::ImageMemoryBarrier2 to_general{
            .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = output,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                   .pImageMemoryBarriers = &to_general});
        static bool failed = false;
        if (!dlss->Evaluate(cmdbuf, frame) && !std::exchange(failed, true)) {
            std::fprintf(stderr, "DLSS: upscaling evaluation failed\n");
        }
        const vk::ImageMemoryBarrier2 to_read{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = output,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                   .pImageMemoryBarriers = &to_read});
    });

    // The composite samples the upscaled scene.
    const vk::ImageView scene_view = *color_view.image_view;
    for (auto& info : image_infos) {
        if (info.imageView == scene_view) {
            info.imageView = *target.view;
            info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        }
    }
}

} // namespace Vulkan
