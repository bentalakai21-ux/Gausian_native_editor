//! WGPU integration for zero-copy IOSurface rendering
//!
//! This module provides integration between IOSurface frames and WGPU external textures
//! for zero-copy video rendering on macOS.

#[cfg(target_os = "macos")]
use super::IOSurfaceFrame;
use anyhow::Result;
use std::sync::Arc;

#[cfg(target_os = "macos")]
use core_foundation::base::TCFType;
#[cfg(target_os = "macos")]
use std::num::NonZeroU32;
#[cfg(target_os = "macos")]
use std::slice;
#[cfg(target_os = "macos")]
use wgpu::*;

#[cfg(target_os = "macos")]
extern "C" {
    fn avf_iosurface_lock_readonly(s: *mut std::ffi::c_void);
    fn avf_iosurface_unlock(s: *mut std::ffi::c_void);
    fn avf_iosurface_width_of_plane(s: *mut std::ffi::c_void, plane: usize) -> usize;
    fn avf_iosurface_height_of_plane(s: *mut std::ffi::c_void, plane: usize) -> usize;
    fn avf_iosurface_bytes_per_row_of_plane(s: *mut std::ffi::c_void, plane: usize) -> usize;
    fn avf_iosurface_base_address_of_plane(
        s: *mut std::ffi::c_void,
        plane: usize,
    ) -> *const std::ffi::c_void;
}

#[cfg(target_os = "macos")]
#[inline]
fn align_up(x: u32, align: u32) -> u32 {
    (x + align - 1) & !(align - 1)
}
#[cfg(target_os = "macos")]
use tracing::{debug, info};

/// WGPU external texture for IOSurface integration
#[cfg(target_os = "macos")]
pub struct IOSurfaceTexture {
    pub texture: Texture,
    pub view: TextureView,
    pub surface: io_surface::IOSurface,
    pub width: u32,
    pub height: u32,
}

#[cfg(target_os = "macos")]
impl IOSurfaceTexture {
    /// Create a WGPU texture from an IOSurface frame
    pub fn from_iosurface_frame(
        device: &Device,
        _queue: &Queue,
        frame: &IOSurfaceFrame,
    ) -> Result<Self> {
        // Create external texture descriptor
        let texture_descriptor = TextureDescriptor {
            label: Some("IOSurface External Texture"),
            size: Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        };

        // Create texture from IOSurface
        let texture = device.create_texture(&texture_descriptor);
        let view = texture.create_view(&TextureViewDescriptor::default());

        // Copy IOSurface data to texture
        // In a real implementation, we would use copy_external_texture_to_texture
        // or similar WGPU methods for zero-copy operations

        Ok(Self {
            texture,
            view,
            surface: frame.surface.clone(),
            width: frame.width,
            height: frame.height,
        })
    }

    /// Update texture with new IOSurface data
    pub fn update_from_iosurface(&self, _queue: &Queue, _frame: &IOSurfaceFrame) -> Result<()> {
        // In a real implementation, we would:
        // 1. Lock the IOSurface
        // 2. Copy data directly to GPU memory
        // 3. Unlock the IOSurface

        // For now, this is a placeholder
        info!("Updating IOSurface texture with new frame data");
        Ok(())
    }

    /// Get the texture view for rendering
    pub fn get_view(&self) -> &TextureView {
        &self.view
    }

    /// Get the underlying texture
    pub fn get_texture(&self) -> &Texture {
        &self.texture
    }
}

/// Two-plane NV12 GPU textures used by the preview renderer
#[cfg(target_os = "macos")]
pub struct GpuYuv {
    pub y_tex: std::sync::Arc<Texture>,  // R8Unorm WxH
    pub uv_tex: std::sync::Arc<Texture>, // Rg8Unorm (W/2)x(H/2)
}

#[cfg(target_os = "macos")]
impl GpuYuv {
    /// Import NV12 planes from an IOSurface into two textures.
    /// Assumes textures are created with COPY_DST | TEXTURE_BINDING usages,
    /// with formats R8Unorm for Y and Rg8Unorm for UV.
    pub fn import_from_iosurface(
        &self,
        queue: &Queue,
        frame: &IOSurfaceFrame,
    ) -> anyhow::Result<()> {
        // Obtain raw IOSurfaceRef
        let s_ref = frame.surface.as_concrete_TypeRef() as *mut std::ffi::c_void;
        if s_ref.is_null() {
            return Ok(());
        }

        unsafe { avf_iosurface_lock_readonly(s_ref) };

        // Plane 0 (Y)
        let w0 = unsafe { avf_iosurface_width_of_plane(s_ref, 0) } as u32;
        let h0 = unsafe { avf_iosurface_height_of_plane(s_ref, 0) } as u32;
        let src_bpr0 = unsafe { avf_iosurface_bytes_per_row_of_plane(s_ref, 0) } as u32;
        let base0 = unsafe { avf_iosurface_base_address_of_plane(s_ref, 0) } as *const u8;

        // Plane 1 (interleaved UV)
        let w1 = unsafe { avf_iosurface_width_of_plane(s_ref, 1) } as u32;
        let h1 = unsafe { avf_iosurface_height_of_plane(s_ref, 1) } as u32;
        let src_bpr1 = unsafe { avf_iosurface_bytes_per_row_of_plane(s_ref, 1) } as u32;
        let base1 = unsafe { avf_iosurface_base_address_of_plane(s_ref, 1) } as *const u8;

        debug!(
            "IOSurface planes: Y {}x{} bpr={} | UV {}x{} bpr={}",
            w0, h0, src_bpr0, w1, h1, src_bpr1
        );

        // Convert base addresses to slices
        let y_src = unsafe { slice::from_raw_parts(base0, (src_bpr0 * h0) as usize) };
        let uv_src = unsafe { slice::from_raw_parts(base1, (src_bpr1 * h1) as usize) };

        // Try direct upload using IOSurface BPR; fall back to repack if alignment is invalid.
        const ALIGN: u32 = 256;

        let (y_owned, y_bpr_use) = if src_bpr0 % ALIGN == 0 {
            (None, src_bpr0)
        } else {
            let dst_bpr0 = ((w0 + ALIGN - 1) / ALIGN) * ALIGN;
            let mut packed = vec![0u8; (dst_bpr0 * h0) as usize];
            for row in 0..h0 {
                let src_off = (row * src_bpr0) as usize;
                let dst_off = (row * dst_bpr0) as usize;
                packed[dst_off..dst_off + (w0 as usize)]
                    .copy_from_slice(&y_src[src_off..src_off + (w0 as usize)]);
            }
            debug!(
                "NV12 Y repack due to alignment: {} -> {}",
                src_bpr0, dst_bpr0
            );
            (Some(packed), dst_bpr0)
        };
        let y_bytes: &[u8] = y_owned.as_deref().unwrap_or(y_src);

        let row_bytes_uv = w1 * 2;
        let (uv_owned, uv_bpr_use) = if src_bpr1 % ALIGN == 0 && src_bpr1 >= row_bytes_uv {
            (None, src_bpr1)
        } else {
            let dst_bpr1 = ((row_bytes_uv + ALIGN - 1) / ALIGN) * ALIGN;
            let mut packed = vec![0u8; (dst_bpr1 * h1) as usize];
            for row in 0..h1 {
                let src_off = (row * src_bpr1) as usize;
                let dst_off = (row * dst_bpr1) as usize;
                packed[dst_off..dst_off + (row_bytes_uv as usize)]
                    .copy_from_slice(&uv_src[src_off..src_off + (row_bytes_uv as usize)]);
            }
            debug!(
                "NV12 UV repack due to alignment: {} -> {}",
                src_bpr1, dst_bpr1
            );
            (Some(packed), dst_bpr1)
        };
        let uv_bytes: &[u8] = uv_owned.as_deref().unwrap_or(uv_src);

        // Upload Y
        queue.write_texture(
            ImageCopyTexture {
                texture: &self.y_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            y_bytes,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(y_bpr_use),
                rows_per_image: Some(h0),
            },
            Extent3d {
                width: w0,
                height: h0,
                depth_or_array_layers: 1,
            },
        );

        // Upload UV
        queue.write_texture(
            ImageCopyTexture {
                texture: &self.uv_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            uv_bytes,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(uv_bpr_use),
                rows_per_image: Some(h1),
            },
            Extent3d {
                width: w1,
                height: h1,
                depth_or_array_layers: 1,
            },
        );

        debug!(
            "NV12 wrote bytes: Y={} UV={}",
            (y_bpr_use as usize) * (h0 as usize),
            (uv_bpr_use as usize) * (h1 as usize)
        );

        unsafe { avf_iosurface_unlock(s_ref) };
        Ok(())
    }
}

/// WGPU render pipeline for IOSurface textures
#[cfg(target_os = "macos")]
pub struct IOSurfaceRenderPipeline {
    pub pipeline: RenderPipeline,
    pub bind_group_layout: BindGroupLayout,
}

#[cfg(target_os = "macos")]
impl IOSurfaceRenderPipeline {
    /// Create a render pipeline for IOSurface textures
    pub fn new(device: &Device) -> Result<Self> {
        // Create shader module
        let shader_source = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
                var pos = array<vec2<f32>, 3>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>( 3.0, -1.0),
                    vec2<f32>(-1.0,  3.0)
                );
                return vec4<f32>(pos[vertex_index], 0.0, 1.0);
            }

            @fragment
            fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                // Sample from IOSurface texture
                return vec4<f32>(uv, 0.0, 1.0);
            }
        "#;

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("IOSurface Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("IOSurface Bind Group Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: TextureViewDimension::D2,
                    sample_type: TextureSampleType::Float { filterable: true },
                },
                count: None,
            }],
        });

        // Create render pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("IOSurface Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("IOSurface Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Bgra8Unorm,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    /// Create a bind group for an IOSurface texture
    pub fn create_bind_group(&self, device: &Device, texture_view: &TextureView) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("IOSurface Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(texture_view),
            }],
        })
    }
}

/// Zero-copy video renderer using IOSurface and WGPU
#[cfg(target_os = "macos")]
pub struct ZeroCopyVideoRenderer {
    pub pipeline: IOSurfaceRenderPipeline,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

#[cfg(target_os = "macos")]
impl ZeroCopyVideoRenderer {
    /// Create a new zero-copy video renderer
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self> {
        let pipeline = IOSurfaceRenderPipeline::new(&device)?;

        Ok(Self {
            pipeline,
            device,
            queue,
        })
    }

    /// Render an IOSurface frame to the screen
    pub fn render_frame(
        &self,
        encoder: &mut CommandEncoder,
        frame: &IOSurfaceFrame,
        output_texture: &TextureView,
    ) -> Result<()> {
        // Create IOSurface texture
        let iosurface_texture =
            IOSurfaceTexture::from_iosurface_frame(&self.device, &self.queue, frame)?;

        // Create bind group
        let bind_group = self
            .pipeline
            .create_bind_group(&self.device, &iosurface_texture.view);

        // Create render pass
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("IOSurface Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: output_texture,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Set pipeline and bind group
        render_pass.set_pipeline(&self.pipeline.pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);

        // Draw triangle
        render_pass.draw(0..3, 0..1);

        drop(render_pass);

        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
pub struct ZeroCopyVideoRenderer;

#[cfg(not(target_os = "macos"))]
impl ZeroCopyVideoRenderer {
    pub fn new(_device: Arc<wgpu::Device>, _queue: Arc<wgpu::Queue>) -> Result<Self> {
        Ok(Self)
    }

    pub fn render_frame(
        &self,
        _encoder: &mut wgpu::CommandEncoder,
        _frame: &super::VideoFrame,
        _output_texture: &wgpu::TextureView,
    ) -> Result<()> {
        // No-op on non-macOS platforms
        Ok(())
    }
}
