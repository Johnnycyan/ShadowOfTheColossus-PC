// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/host_passes/blur_bars_pass.h"

#include <algorithm>
#include <cstdlib>

#include "common/assert.h"
#include "video_core/host_shaders/blur_bars_frag.h"
#include "video_core/host_shaders/fs_tri_vert.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan::HostPasses {

namespace {
constexpr vk::Format LevelFormat = vk::Format::eR16G16B16A16Sfloat;
constexpr u32 MaxBlurInput = 512; // largest side of the blurred image, as in ultrawide-overlay

struct PushConstants {
    float delta[2];
    u32 up;
};

constexpr vk::ImageSubresourceRange ColorRange{
    .aspectMask = vk::ImageAspectFlagBits::eColor,
    .levelCount = 1,
    .layerCount = 1,
};

vk::ImageBlit FullBlit(vk::Extent2D from, vk::Extent2D to) {
    return vk::ImageBlit{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{static_cast<s32>(from.width),
                                              static_cast<s32>(from.height), 1}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{static_cast<s32>(to.width),
                                              static_cast<s32>(to.height), 1}},
    };
}
} // namespace

u32 BlurBarsPass::Passes() {
    static const u32 passes = [] {
        const char* v = std::getenv("SOTC_BAR_BLUR");
        return v && *v ? static_cast<u32>(std::clamp(std::atoi(v), 0, 10)) : 2u;
    }();
    return passes;
}

void BlurBarsPass::Create(vk::Device device_, VmaAllocator allocator_) {
    device = device_;
    allocator = allocator_;

    const vk::DescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };
    desc_set_layout = Check<"create blur bars descriptor set layout">(
        device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = 1,
            .pBindings = &binding,
        }));

    const vk::PushConstantRange push_constants{
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    pipeline_layout = Check<"create blur bars pipeline layout">(
        device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*desc_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constants,
        }));

    const auto vs_module = CompileSPV(FS_TRI_VERT, device);
    ASSERT(vs_module);
    const auto fs_module = CompileSPV(BLUR_BARS_FRAG, device);
    ASSERT(fs_module);
    SetObjectName(device, fs_module, "blur_bars.frag");
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vs_module,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fs_module,
            .pName = "main",
        },
    };

    const vk::PipelineRenderingCreateInfo rendering_ci{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &LevelFormat,
    };
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };
    const vk::PipelineViewportStateCreateInfo viewport_info{
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const vk::PipelineRasterizationStateCreateInfo raster_state{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eClockwise,
        .lineWidth = 1.0f,
    };
    const vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };
    const vk::PipelineColorBlendAttachmentState attachment{
        .blendEnable = false,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo color_blending{
        .attachmentCount = 1,
        .pAttachments = &attachment,
    };
    const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamic_info{
        .dynamicStateCount = dynamic_states.size(),
        .pDynamicStates = dynamic_states.data(),
    };
    pipeline = Check<"create blur bars pipeline">(device.createGraphicsPipelineUnique(
        {}, vk::GraphicsPipelineCreateInfo{
                .pNext = &rendering_ci,
                .stageCount = stages.size(),
                .pStages = stages.data(),
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &input_assembly,
                .pViewportState = &viewport_info,
                .pRasterizationState = &raster_state,
                .pMultisampleState = &multisampling,
                .pColorBlendState = &color_blending,
                .pDynamicState = &dynamic_info,
                .layout = *pipeline_layout,
            }));
    device.destroyShaderModule(vs_module);
    device.destroyShaderModule(fs_module);

    sampler = Check<"create blur bars sampler">(device.createSamplerUnique(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    }));
}

void BlurBarsPass::Resize(vk::Extent2D src_size) {
    cur_src_size = src_size;
    levels.clear();
    // Halve until the blur input fits MaxBlurInput (at least once), then one level per Kawase
    // pass while both sides stay at least 2 pixels.
    std::vector<vk::Extent2D> sizes;
    vk::Extent2D size = src_size;
    do {
        size = {std::max(size.width / 2, 1u), std::max(size.height / 2, 1u)};
        sizes.push_back(size);
    } while (std::max(size.width, size.height) > MaxBlurInput);
    reduced = static_cast<u32>(sizes.size() - 1);
    iterations = 0;
    while (iterations < Passes() && size.width / 2 >= 2 && size.height / 2 >= 2) {
        size = {size.width / 2, size.height / 2};
        sizes.push_back(size);
        ++iterations;
    }

    levels.resize(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto& level = levels[i];
        level.size = sizes[i];
        level.image = VideoCore::UniqueImage(device, allocator);
        level.image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = LevelFormat,
            .extent = {level.size.width, level.size.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                     vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        SetObjectName(device, static_cast<vk::Image>(level.image), "Blur bars level #{}", i);
        level.view = Check<"create blur bars image view">(
            device.createImageViewUnique(vk::ImageViewCreateInfo{
                .image = level.image,
                .viewType = vk::ImageViewType::e2D,
                .format = LevelFormat,
                .subresourceRange = ColorRange,
            }));
    }
}

void BlurBarsPass::Transition(vk::CommandBuffer cmdbuf, Level& level, vk::ImageLayout layout) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = level.layout,
        .newLayout = layout,
        .image = level.image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
    level.layout = layout;
}

void BlurBarsPass::Draw(vk::CommandBuffer cmdbuf, Level& from, Level& to, bool up) {
    Transition(cmdbuf, from, vk::ImageLayout::eShaderReadOnlyOptimal);
    Transition(cmdbuf, to, vk::ImageLayout::eColorAttachmentOptimal);

    const vk::RenderingAttachmentInfo attachment{
        .imageView = *to.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::DescriptorImageInfo image_info{
        .sampler = *sampler,
        .imageView = *from.view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    const vk::WriteDescriptorSet write{
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &image_info,
    };
    const PushConstants constants{
        .delta = {1.0f / static_cast<float>(from.size.width),
                  1.0f / static_cast<float>(from.size.height)},
        .up = up ? 1u : 0u,
    };

    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
    cmdbuf.setViewport(0, vk::Viewport{
                              .width = static_cast<float>(to.size.width),
                              .height = static_cast<float>(to.size.height),
                              .maxDepth = 1.0f,
                          });
    cmdbuf.setScissor(0, vk::Rect2D{.extent = to.size});
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, write);
    cmdbuf.pushConstants(*pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0,
                         sizeof(constants), &constants);
    cmdbuf.beginRendering(vk::RenderingInfo{
        .renderArea = {.extent = to.size},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment,
    });
    cmdbuf.draw(3, 1, 0, 0);
    cmdbuf.endRendering();
}

void BlurBarsPass::Render(vk::CommandBuffer cmdbuf, vk::Image src, vk::Extent2D src_size,
                          vk::Image dst, vk::Extent2D dst_size) {
    if (src_size != cur_src_size) {
        Resize(src_size);
    }
    // The previous frame's contents are not needed.
    for (auto& level : levels) {
        level.layout = vk::ImageLayout::eUndefined;
    }

    // Box-filtered halvings: each blit reads 2x2 texels, so the small image does not shimmer.
    for (u32 i = 0; i <= reduced; ++i) {
        Transition(cmdbuf, levels[i], vk::ImageLayout::eTransferDstOptimal);
        if (i == 0) {
            cmdbuf.blitImage(src, vk::ImageLayout::eTransferSrcOptimal, levels[0].image,
                             vk::ImageLayout::eTransferDstOptimal,
                             FullBlit(src_size, levels[0].size), vk::Filter::eLinear);
        } else {
            Transition(cmdbuf, levels[i - 1], vk::ImageLayout::eTransferSrcOptimal);
            cmdbuf.blitImage(levels[i - 1].image, vk::ImageLayout::eTransferSrcOptimal,
                             levels[i].image, vk::ImageLayout::eTransferDstOptimal,
                             FullBlit(levels[i - 1].size, levels[i].size), vk::Filter::eLinear);
        }
    }

    // Dual Kawase: down the chain, then back up into the blur input.
    const u32 last = reduced + iterations;
    for (u32 i = reduced; i < last; ++i) {
        Draw(cmdbuf, levels[i], levels[i + 1], false);
    }
    for (u32 i = last; i > reduced; --i) {
        Draw(cmdbuf, levels[i], levels[i - 1], true);
    }

    auto& blurred = levels[reduced];
    Transition(cmdbuf, blurred, vk::ImageLayout::eTransferSrcOptimal);
    cmdbuf.blitImage(blurred.image, vk::ImageLayout::eTransferSrcOptimal, dst,
                     vk::ImageLayout::eTransferDstOptimal, FullBlit(blurred.size, dst_size),
                     vk::Filter::eLinear);

    // The caller's blit of the game image overwrites the middle of dst.
    const vk::ImageMemoryBarrier2 dst_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eBlit,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = dst,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &dst_barrier,
    });
}

} // namespace Vulkan::HostPasses
