// SPDX-License-Identifier: GPL-2.0-or-later
// sotc: blurred side bars. When the game image does not fill the swapchain (a 16:9 game on a
// 21:9 screen), the bars show a blurred, stretched copy of the frame instead of black, like the
// ultrawide-overlay app: the frame is halved by blits to at most 512 pixels, blurred with dual
// Kawase passes and stretched over the whole swapchain image.

#pragma once

#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan::HostPasses {

class BlurBarsPass {
public:
    void Create(vk::Device device, VmaAllocator allocator);

    // Dual Kawase passes, SOTC_BAR_BLUR=0..10 (default 2); 0 keeps black bars.
    static u32 Passes();

    // src is in TransferSrcOptimal and dst in TransferDstOptimal; both stay so. Afterwards the
    // whole of dst holds the blurred src and later transfer writes to dst are ordered after it.
    void Render(vk::CommandBuffer cmdbuf, vk::Image src, vk::Extent2D src_size, vk::Image dst,
                vk::Extent2D dst_size);

private:
    struct Level {
        VideoCore::UniqueImage image;
        vk::UniqueImageView view;
        vk::Extent2D size;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };

    void Resize(vk::Extent2D src_size);
    void Transition(vk::CommandBuffer cmdbuf, Level& level, vk::ImageLayout layout);
    void Draw(vk::CommandBuffer cmdbuf, Level& from, Level& to, bool up);

    vk::Device device{};
    VmaAllocator allocator{};
    vk::UniquePipeline pipeline{};
    vk::UniquePipelineLayout pipeline_layout{};
    vk::UniqueDescriptorSetLayout desc_set_layout{};
    vk::UniqueSampler sampler{};

    vk::Extent2D cur_src_size{};
    // levels[i] is the frame halved i + 1 times; levels[reduced] is the blur input, the levels
    // after it are the Kawase chain.
    std::vector<Level> levels;
    u32 reduced = 0;
    u32 iterations = 0;
};

} // namespace Vulkan::HostPasses
