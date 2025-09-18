use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use eframe::egui::TextureHandle;
use eframe::egui_wgpu;
use eframe::{egui, wgpu};
use media_io::YuvPixFmt;
use native_decoder::{
    self, create_decoder, is_native_decoding_available, DecoderConfig, NativeVideoDecoder,
    VideoFrame, YuvPixFmt as NativeYuvPixFmt,
};

use crate::preview::visual_source_at;
use crate::VisualSource;
use crate::PRESENT_SIZE_MISMATCH_LOGGED;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PreviewShaderMode {
    Solid,
    ShowY,
    UvDebug,
    Nv12,
}

impl Default for PreviewShaderMode {
    fn default() -> Self {
        PreviewShaderMode::Solid
    }
}

pub(crate) struct StreamSlot {
    pub(crate) stream_id: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) fmt: YuvPixFmt,
    pub(crate) clear_color: egui::Color32,
    pub(crate) y_tex: Option<Arc<eframe::wgpu::Texture>>,
    pub(crate) uv_tex: Option<Arc<eframe::wgpu::Texture>>,
    pub(crate) out_tex: Option<Arc<eframe::wgpu::Texture>>,
    pub(crate) out_view: Option<eframe::wgpu::TextureView>,
    pub(crate) egui_tex_id: Option<egui::TextureId>,
}

pub(crate) struct StreamMetadata {
    pub(crate) stream_id: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) fmt: YuvPixFmt,
    pub(crate) clear_color: egui::Color32,
}

pub(crate) struct PreviewState {
    pub(crate) texture: Option<TextureHandle>,
    pub(crate) stream: Option<StreamSlot>,
    pub(crate) last_pts: Option<f64>,
    pub(crate) frame_cache: Arc<Mutex<HashMap<FrameCacheKey, CachedFrame>>>,
    pub(crate) cache_worker: Option<JoinHandle<()>>,
    pub(crate) cache_stop: Option<Arc<AtomicBool>>,
    pub(crate) current_source: Option<VisualSource>,
    pub(crate) last_frame_time: f64,
    pub(crate) last_size: (u32, u32),
    pub(crate) gpu_tex_a: Option<Arc<eframe::wgpu::Texture>>,
    pub(crate) gpu_view_a: Option<eframe::wgpu::TextureView>,
    pub(crate) gpu_tex_b: Option<Arc<eframe::wgpu::Texture>>,
    pub(crate) gpu_view_b: Option<eframe::wgpu::TextureView>,
    pub(crate) gpu_use_b: bool,
    pub(crate) gpu_tex_id: Option<egui::TextureId>,
    pub(crate) gpu_size: (u32, u32),
    pub(crate) y_tex: [Option<Arc<eframe::wgpu::Texture>>; 3],
    pub(crate) uv_tex: [Option<Arc<eframe::wgpu::Texture>>; 3],
    pub(crate) y_size: (u32, u32),
    pub(crate) uv_size: (u32, u32),
    pub(crate) ring_write: usize,
    pub(crate) ring_present: usize,
    nv12_cache: HashMap<FrameCacheKey, Nv12Frame>,
    nv12_keys: VecDeque<FrameCacheKey>,
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) decode_time_ms: f64,
    pub(crate) last_fmt: Option<YuvPixFmt>,
    pub(crate) last_cpu_tick: u64,
    pub(crate) last_present_tick: u64,
    pub(crate) shader_mode: PreviewShaderMode,
    #[cfg(target_os = "macos")]
    pub(crate) gpu_yuv: Option<native_decoder::GpuYuv>,
    #[cfg(target_os = "macos")]
    pub(crate) last_zc: Option<(
        YuvPixFmt,
        Arc<eframe::wgpu::Texture>,
        Arc<eframe::wgpu::Texture>,
        (u32, u32),
    )>,
    #[cfg(target_os = "macos")]
    pub(crate) last_zc_tick: u64,
    #[cfg(target_os = "macos")]
    pub(crate) zc_logged: bool,
}

impl PreviewState {
    pub(crate) fn new() -> Self {
        Self {
            texture: None,
            stream: None,
            last_pts: None,
            frame_cache: Arc::new(Mutex::new(HashMap::new())),
            cache_worker: None,
            cache_stop: None,
            current_source: None,
            last_frame_time: -1.0,
            last_size: (0, 0),
            gpu_tex_a: None,
            gpu_view_a: None,
            gpu_tex_b: None,
            gpu_view_b: None,
            gpu_use_b: false,
            gpu_tex_id: None,
            gpu_size: (0, 0),
            y_tex: [None, None, None],
            uv_tex: [None, None, None],
            y_size: (0, 0),
            uv_size: (0, 0),
            ring_write: 0,
            ring_present: 0,
            nv12_cache: HashMap::new(),
            nv12_keys: VecDeque::new(),
            cache_hits: 0,
            cache_misses: 0,
            decode_time_ms: 0.0,
            last_fmt: None,
            last_cpu_tick: 0,
            last_present_tick: 0,
            shader_mode: PreviewShaderMode::Nv12,
            #[cfg(target_os = "macos")]
            gpu_yuv: None,
            #[cfg(target_os = "macos")]
            last_zc: None,
            #[cfg(target_os = "macos")]
            last_zc_tick: 0,
            #[cfg(target_os = "macos")]
            zc_logged: false,
        }
    }

    pub(crate) fn ensure_stream_slot<'a>(
        &'a mut self,
        device: &eframe::wgpu::Device,
        renderer: &mut eframe::egui_wgpu::Renderer,
        meta: StreamMetadata,
    ) -> &'a mut StreamSlot {
        let StreamMetadata {
            stream_id,
            width,
            height,
            fmt,
            clear_color,
        } = meta;

        let ready = matches!(
            self.stream.as_ref(),
            Some(slot)
                if slot.stream_id == stream_id
                    && slot.width == width
                    && slot.height == height
                    && slot.fmt == fmt
                    && slot.y_tex.is_some()
                    && slot.uv_tex.is_some()
                    && slot.out_tex.is_some()
                    && slot.out_view.is_some()
                    && slot.egui_tex_id.is_some()
        );

        if ready {
            return self.stream.as_mut().unwrap();
        }

        if let Some(slot) = self.stream.take() {
            if let Some(id) = slot.egui_tex_id {
                renderer.free_texture(&id);
            }
        }

        let (y_format, uv_format) = match fmt {
            YuvPixFmt::Nv12 => (
                eframe::wgpu::TextureFormat::R8Unorm,
                eframe::wgpu::TextureFormat::Rg8Unorm,
            ),
            YuvPixFmt::P010 => (
                eframe::wgpu::TextureFormat::R16Unorm,
                eframe::wgpu::TextureFormat::Rg16Unorm,
            ),
        };

        let y_tex = Arc::new(device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("preview_stream_y"),
            size: eframe::wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: eframe::wgpu::TextureDimension::D2,
            format: y_format,
            usage: eframe::wgpu::TextureUsages::COPY_DST
                | eframe::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));

        let uv_tex = Arc::new(device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("preview_stream_uv"),
            size: eframe::wgpu::Extent3d {
                width: (width + 1) / 2,
                height: (height + 1) / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: eframe::wgpu::TextureDimension::D2,
            format: uv_format,
            usage: eframe::wgpu::TextureUsages::COPY_DST
                | eframe::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));

        let out_tex = Arc::new(device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("preview_stream_out"),
            size: eframe::wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: eframe::wgpu::TextureDimension::D2,
            format: eframe::wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: eframe::wgpu::TextureUsages::COPY_DST
                | eframe::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));

        let out_view = out_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default());
        let tex_id =
            renderer.register_native_texture(device, &out_view, eframe::wgpu::FilterMode::Linear);

        self.stream = Some(StreamSlot {
            stream_id,
            width,
            height,
            fmt,
            clear_color,
            y_tex: Some(y_tex),
            uv_tex: Some(uv_tex),
            out_tex: Some(out_tex),
            out_view: Some(out_view),
            egui_tex_id: Some(tex_id),
        });

        self.stream.as_mut().unwrap()
    }

    // Ensure triple-buffer NV12 plane textures at native size
    pub(crate) fn ensure_yuv_textures(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        w: u32,
        h: u32,
        fmt: YuvPixFmt,
    ) {
        let y_sz = (w, h);
        let uv_sz = ((w + 1) / 2, (h + 1) / 2);
        if self.y_size == y_sz
            && self.uv_size == uv_sz
            && self.y_tex[0].is_some()
            && self.uv_tex[0].is_some()
        {
            return;
        }
        let device = &*rs.device;
        let supports16 = device_supports_16bit_norm(rs);
        let (y_format, uv_format) = match fmt {
            YuvPixFmt::Nv12 => (
                eframe::wgpu::TextureFormat::R8Unorm,
                eframe::wgpu::TextureFormat::Rg8Unorm,
            ),
            YuvPixFmt::P010 => {
                if supports16 {
                    (
                        eframe::wgpu::TextureFormat::R16Unorm,
                        eframe::wgpu::TextureFormat::Rg16Unorm,
                    )
                } else {
                    (
                        eframe::wgpu::TextureFormat::R16Uint,
                        eframe::wgpu::TextureFormat::Rg16Uint,
                    )
                }
            }
        };
        let make_y = || {
            device.create_texture(&eframe::wgpu::TextureDescriptor {
                label: Some("preview_nv12_y"),
                size: eframe::wgpu::Extent3d {
                    width: y_sz.0,
                    height: y_sz.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: eframe::wgpu::TextureDimension::D2,
                format: y_format,
                usage: eframe::wgpu::TextureUsages::COPY_DST
                    | eframe::wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let make_uv = || {
            device.create_texture(&eframe::wgpu::TextureDescriptor {
                label: Some("preview_nv12_uv"),
                size: eframe::wgpu::Extent3d {
                    width: uv_sz.0,
                    height: uv_sz.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: eframe::wgpu::TextureDimension::D2,
                format: uv_format,
                usage: eframe::wgpu::TextureUsages::COPY_DST
                    | eframe::wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        for i in 0..3 {
            self.y_tex[i] = Some(std::sync::Arc::new(make_y()));
            self.uv_tex[i] = Some(std::sync::Arc::new(make_uv()));
        }
        self.ring_write = 0;
        self.ring_present = 0;
        self.y_size = y_sz;
        self.uv_size = uv_sz;
    }

    pub(crate) fn upload_yuv_planes(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        fmt: YuvPixFmt,
        y: &[u8],
        uv: &[u8],
        w: u32,
        h: u32,
    ) {
        self.ensure_yuv_textures(rs, w, h, fmt);
        let queue = &*rs.queue;
        let next_idx = (self.ring_write + 1) % 3;
        if next_idx == self.ring_present {
            eprintln!(
                "[RING DROP] write={} present={} (dropping frame to avoid stall)",
                self.ring_write, self.ring_present
            );
            return;
        }
        let idx = self.ring_write % 3;
        let y_tex = self.y_tex[idx].as_ref().map(|a| &**a).unwrap();
        let uv_tex = self.uv_tex[idx].as_ref().map(|a| &**a).unwrap();

        let uv_w = (w + 1) / 2;
        let uv_h = (h + 1) / 2;
        let (y_bpp, uv_bpp_per_texel) = match fmt {
            YuvPixFmt::Nv12 => (1usize, 2usize),
            YuvPixFmt::P010 => (2usize, 4usize),
        };
        upload_plane(queue, y_tex, y, w, h, (w as usize) * y_bpp, y_bpp);
        upload_plane(
            queue,
            uv_tex,
            uv,
            uv_w,
            uv_h,
            (uv_w as usize) * uv_bpp_per_texel,
            uv_bpp_per_texel,
        );

        self.ring_present = idx;
        self.ring_write = next_idx;
        self.last_fmt = Some(fmt);
    }

    pub(crate) fn current_plane_textures(
        &self,
    ) -> Option<(
        YuvPixFmt,
        std::sync::Arc<eframe::wgpu::Texture>,
        std::sync::Arc<eframe::wgpu::Texture>,
    )> {
        let mut best: Option<(
            u64,
            YuvPixFmt,
            std::sync::Arc<eframe::wgpu::Texture>,
            std::sync::Arc<eframe::wgpu::Texture>,
        )> = None;
        if let Some(fmt) = self.last_fmt {
            let idx = self.ring_present % 3;
            if let (Some(y), Some(uv)) = (self.y_tex[idx].as_ref(), self.uv_tex[idx].as_ref()) {
                best = Some((self.last_cpu_tick, fmt, y.clone(), uv.clone()));
            }
        }
        #[cfg(target_os = "macos")]
        if let Some((fmt, y, uv, _sz)) = self.last_zc.as_ref() {
            match best {
                Some((tick, ..)) if self.last_zc_tick <= tick => {}
                _ => {
                    best = Some((self.last_zc_tick, *fmt, y.clone(), uv.clone()));
                }
            }
        }
        best.map(|(_, fmt, y, uv)| (fmt, y, uv))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn ensure_zero_copy_nv12_textures(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        w: u32,
        h: u32,
    ) {
        let target_y = (w, h);
        let target_uv = ((w + 1) / 2, (h + 1) / 2);
        let needs_new = match &self.gpu_yuv {
            Some(_) if self.y_size == target_y && self.uv_size == target_uv => false,
            _ => true,
        };
        if !needs_new {
            return;
        }

        let device = &*rs.device;
        let make_tex = |label: &str, size: (u32, u32), format: eframe::wgpu::TextureFormat| {
            Arc::new(device.create_texture(&eframe::wgpu::TextureDescriptor {
                label: Some(label),
                size: eframe::wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: eframe::wgpu::TextureDimension::D2,
                format,
                usage: eframe::wgpu::TextureUsages::COPY_DST
                    | eframe::wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }))
        };

        let y_tex = make_tex(
            "preview_zc_nv12_y",
            target_y,
            eframe::wgpu::TextureFormat::R8Unorm,
        );
        let uv_tex = make_tex(
            "preview_zc_nv12_uv",
            target_uv,
            eframe::wgpu::TextureFormat::Rg8Unorm,
        );
        self.gpu_yuv = Some(native_decoder::GpuYuv {
            y_tex: y_tex.clone(),
            uv_tex: uv_tex.clone(),
        });
        self.y_size = target_y;
        self.uv_size = target_uv;
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn set_last_zc_present(
        &mut self,
        fmt: YuvPixFmt,
        y_tex: std::sync::Arc<eframe::wgpu::Texture>,
        uv_tex: std::sync::Arc<eframe::wgpu::Texture>,
        w: u32,
        h: u32,
    ) {
        self.last_zc = Some((fmt, y_tex, uv_tex, (w, h)));
        self.last_fmt = Some(fmt);
        self.y_size = (w, h);
        self.uv_size = ((w + 1) / 2, (h + 1) / 2);
        self.last_present_tick = self.last_present_tick.wrapping_add(1);
        self.last_zc_tick = self.last_present_tick;
    }

    pub(crate) fn present_yuv(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        path: &str,
        t_sec: f64,
    ) -> Option<(
        YuvPixFmt,
        Arc<eframe::wgpu::Texture>,
        Arc<eframe::wgpu::Texture>,
    )> {
        let key = FrameCacheKey::new(path, t_sec, 0, 0);
        let mut fmt;
        let mut y;
        let mut uv;
        let mut w;
        let mut h;
        if let Some(hit) = self.nv12_cache.get(&key) {
            fmt = hit.fmt;
            y = hit.y.clone();
            uv = hit.uv.clone();
            w = hit.w;
            h = hit.h;
            if let Some(pos) = self.nv12_keys.iter().position(|k| k == &key) {
                self.nv12_keys.remove(pos);
            }
            self.nv12_keys.push_back(key.clone());
        } else {
            if let Ok(frame) = media_io::decode_yuv_at(std::path::Path::new(path), t_sec) {
                fmt = frame.fmt;
                y = frame.y;
                uv = frame.uv;
                w = frame.width;
                h = frame.height;
                if fmt == YuvPixFmt::P010 && !device_supports_16bit_norm(rs) {
                    if let Some((_f, ny, nuv, nw, nh)) = decode_video_frame_nv12_only(path, t_sec) {
                        fmt = YuvPixFmt::Nv12;
                        y = ny;
                        uv = nuv;
                        w = nw;
                        h = nh;
                    }
                }
                self.nv12_cache.insert(
                    key.clone(),
                    Nv12Frame {
                        fmt,
                        y: y.clone(),
                        uv: uv.clone(),
                        w,
                        h,
                    },
                );
                self.nv12_keys.push_back(key.clone());
                if self.nv12_keys.len() > 64 {
                    if let Some(old) = self.nv12_keys.pop_front() {
                        self.nv12_cache.remove(&old);
                    }
                }
            } else {
                return None;
            }
        }
        self.upload_yuv_planes(rs, fmt, &y, &uv, w, h);
        let idx = self.ring_present;
        Some((
            fmt,
            self.y_tex[idx].as_ref().unwrap().clone(),
            self.uv_tex[idx].as_ref().unwrap().clone(),
        ))
    }

    // Ensure double-buffered GPU textures and a registered TextureId
    pub(crate) fn ensure_gpu_textures(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        w: u32,
        h: u32,
    ) {
        if self.gpu_size == (w, h)
            && self.gpu_tex_id.is_some()
            && (self.gpu_view_a.is_some() || self.gpu_view_b.is_some())
        {
            return;
        }
        let device = &*rs.device;
        let make_tex = || {
            device.create_texture(&eframe::wgpu::TextureDescriptor {
                label: Some("preview_native_tex"),
                size: eframe::wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: eframe::wgpu::TextureDimension::D2,
                format: eframe::wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: eframe::wgpu::TextureUsages::COPY_DST
                    | eframe::wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let tex_a = std::sync::Arc::new(make_tex());
        let view_a = tex_a.create_view(&eframe::wgpu::TextureViewDescriptor::default());
        let tex_b = std::sync::Arc::new(make_tex());
        let view_b = tex_b.create_view(&eframe::wgpu::TextureViewDescriptor::default());

        // Register a TextureId if needed, otherwise update it to A initially
        let mut renderer = rs.renderer.write();
        if let Some(id) = self.gpu_tex_id {
            renderer.update_egui_texture_from_wgpu_texture(
                device,
                &view_a,
                eframe::wgpu::FilterMode::Linear,
                id,
            );
        } else {
            let id =
                renderer.register_native_texture(device, &view_a, eframe::wgpu::FilterMode::Linear);
            self.gpu_tex_id = Some(id);
        }

        self.gpu_tex_a = Some(tex_a);
        self.gpu_view_a = Some(view_a);
        self.gpu_tex_b = Some(tex_b);
        self.gpu_view_b = Some(view_b);
        self.gpu_use_b = false;
        self.gpu_size = (w, h);
    }

    // Upload RGBA bytes into the next back buffer and retarget the TextureId to it
    pub(crate) fn upload_gpu_frame(&mut self, rs: &eframe::egui_wgpu::RenderState, rgba: &[u8]) {
        let (w, h) = self.gpu_size;
        let queue = &*rs.queue;
        // swap buffer
        self.gpu_use_b = !self.gpu_use_b;
        let (tex, view) = if self.gpu_use_b {
            (
                self.gpu_tex_b.as_ref().map(|a| &**a),
                self.gpu_view_b.as_ref(),
            )
        } else {
            (
                self.gpu_tex_a.as_ref().map(|a| &**a),
                self.gpu_view_a.as_ref(),
            )
        };
        if let (Some(tex), Some(view)) = (tex, view) {
            let bytes_per_row = (w * 4) as usize;
            let align = eframe::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize; // 256
            let padded_bpr = ((bytes_per_row + align - 1) / align) * align;
            if padded_bpr == bytes_per_row {
                queue.write_texture(
                    eframe::wgpu::ImageCopyTexture {
                        texture: tex,
                        mip_level: 0,
                        origin: eframe::wgpu::Origin3d::ZERO,
                        aspect: eframe::wgpu::TextureAspect::All,
                    },
                    rgba,
                    eframe::wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some((bytes_per_row) as u32),
                        rows_per_image: Some(h),
                    },
                    eframe::wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                );
            } else {
                // build a padded buffer per row to satisfy alignment
                let mut padded = vec![0u8; padded_bpr * (h as usize)];
                for row in 0..(h as usize) {
                    let src_off = row * bytes_per_row;
                    let dst_off = row * padded_bpr;
                    padded[dst_off..dst_off + bytes_per_row]
                        .copy_from_slice(&rgba[src_off..src_off + bytes_per_row]);
                }
                queue.write_texture(
                    eframe::wgpu::ImageCopyTexture {
                        texture: tex,
                        mip_level: 0,
                        origin: eframe::wgpu::Origin3d::ZERO,
                        aspect: eframe::wgpu::TextureAspect::All,
                    },
                    &padded,
                    eframe::wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bpr as u32),
                        rows_per_image: Some(h),
                    },
                    eframe::wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                );
            }
            if let Some(id) = self.gpu_tex_id {
                let device = &*rs.device;
                let mut renderer = rs.renderer.write();
                renderer.update_egui_texture_from_wgpu_texture(
                    device,
                    view,
                    eframe::wgpu::FilterMode::Linear,
                    id,
                );
            }
        }
    }

    // Present a GPU-cached frame for a source/time. If absent, decode one and upload.
    pub(crate) fn present_gpu_cached(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        path: &str,
        t_sec: f64,
        desired: (u32, u32),
    ) -> Option<egui::TextureId> {
        self.ensure_gpu_textures(rs, desired.0, desired.1);
        // Try cache first
        let key = FrameCacheKey::new(path, t_sec, desired.0, desired.1);
        if let Some(cached) = self.get_cached_frame(&key) {
            let mut bytes = Vec::with_capacity(cached.image.pixels.len() * 4);
            for p in &cached.image.pixels {
                bytes.extend_from_slice(&p.to_array());
            }
            self.upload_gpu_frame(rs, &bytes);
            return self.gpu_tex_id; // ignored in wgpu path; retained for compatibility
        }
        // Decode one frame on demand
        let decoded = if path.to_lowercase().ends_with(".png")
            || path.to_lowercase().ends_with(".jpg")
            || path.to_lowercase().ends_with(".jpeg")
        {
            decode_image_optimized(path, desired.0, desired.1)
        } else {
            decode_video_frame_optimized(path, t_sec, desired.0, desired.1)
        };
        if let Some(img) = decoded {
            let mut bytes = Vec::with_capacity(img.pixels.len() * 4);
            for p in &img.pixels {
                bytes.extend_from_slice(&p.to_array());
            }
            self.upload_gpu_frame(rs, &bytes);
            return self.gpu_tex_id; // ignored in wgpu path; retained for compatibility
        }
        None
    }

    pub(crate) fn update(
        &mut self,
        ctx: &egui::Context,
        size: (u32, u32),
        source: Option<&VisualSource>,
        _playing: bool,
        t_sec: f64,
    ) {
        // Check if we need to update the frame
        let need_update = match source {
            Some(src) => {
                self.current_source.as_ref().map_or(true, |current| {
                    current.path != src.path ||
                    (t_sec - self.last_frame_time).abs() > 0.05 || // Update every 50ms for smooth scrubbing
                    self.last_size != size
                })
            }
            None => self.current_source.is_some(),
        };

        if need_update {
            self.current_source = source.cloned();
            self.last_frame_time = t_sec;
            self.last_size = size;

            if let Some(src) = source {
                // Try to get frame from cache first
                let cache_key = FrameCacheKey::new(&src.path, t_sec, size.0, size.1);

                if let Some(_cached_frame) = self.get_cached_frame(&cache_key) {
                    // Cache hit - let present_gpu_cached upload to native WGPU on paint
                    self.cache_hits += 1;
                    ctx.request_repaint();
                } else {
                    // Cache miss - decode frame asynchronously
                    self.cache_misses += 1;
                    self.decode_frame_async(ctx, src.clone(), cache_key, t_sec);
                }
            } else {
                // no source
            }
        }
    }

    pub(crate) fn get_cached_frame(&self, key: &FrameCacheKey) -> Option<CachedFrame> {
        if let Ok(cache) = self.frame_cache.lock() {
            if let Some(mut frame) = cache.get(key).cloned() {
                frame.access_count += 1;
                frame.last_access = std::time::Instant::now();
                return Some(frame);
            }
        }
        None
    }

    pub(crate) fn decode_frame_async(
        &mut self,
        ctx: &egui::Context,
        source: VisualSource,
        cache_key: FrameCacheKey,
        t_sec: f64,
    ) {
        // If native decoding is available and this is a video, do not spawn RGBA decoding.
        // The persistent native decoder will feed frames via the ring buffer.
        if !source.is_image && is_native_decoding_available() {
            return;
        }
        let cache = self.frame_cache.clone();
        let ctx = ctx.clone();

        // Stop any existing cache worker
        if let Some(stop) = &self.cache_stop {
            stop.store(true, Ordering::Relaxed);
        }
        if let Some(worker) = self.cache_worker.take() {
            let _ = worker.join();
        }

        let stop_flag = Arc::new(AtomicBool::new(false));
        self.cache_stop = Some(stop_flag.clone());

        let worker = thread::spawn(move || {
            if stop_flag.load(Ordering::Relaxed) {
                return;
            }

            let start_time = std::time::Instant::now();

            // Decode frame efficiently
            let frame_result = if source.is_image {
                decode_image_optimized(&source.path, cache_key.width, cache_key.height)
            } else {
                // Use native decoder if available, fallback to FFmpeg
                if is_native_decoding_available() {
                    decode_video_frame_native(
                        &source.path,
                        t_sec,
                        cache_key.width,
                        cache_key.height,
                    )
                } else {
                    decode_video_frame_optimized(
                        &source.path,
                        t_sec,
                        cache_key.width,
                        cache_key.height,
                    )
                }
            };

            if stop_flag.load(Ordering::Relaxed) {
                return;
            }

            if let Some(image) = frame_result {
                let _decode_time = start_time.elapsed();

                // Cache the frame
                let cached_frame = CachedFrame {
                    image: image.clone(),
                    decoded_at: std::time::Instant::now(),
                    access_count: 1,
                    last_access: std::time::Instant::now(),
                };

                if let Ok(mut cache) = cache.lock() {
                    // Implement LRU eviction if cache is too large
                    if cache.len() > 50 {
                        // Max 50 cached frames
                        evict_lru_frames(&mut cache, 10); // Remove oldest 10 frames
                    }

                    cache.insert(cache_key, cached_frame);
                }

                // Update texture on main thread
                ctx.request_repaint();
            }
        });

        self.cache_worker = Some(worker);
    }

    pub(crate) fn stop_cache_worker(&mut self) {
        if let Some(stop) = &self.cache_stop {
            stop.store(true, Ordering::Relaxed);
        }
        if let Some(worker) = self.cache_worker.take() {
            let _ = worker.join();
        }
        self.cache_stop = None;
    }

    pub(crate) fn print_cache_stats(&self) {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests > 0 {
            let hit_rate = (self.cache_hits as f64 / total_requests as f64) * 100.0;
            println!(
                "Preview Cache Stats: {:.1}% hit rate ({}/{} requests), avg decode: {:.1}ms",
                hit_rate, self.cache_hits, total_requests, self.decode_time_ms
            );
        }
    }

    pub(crate) fn preload_nearby_frames(
        &self,
        source: &VisualSource,
        current_time: f64,
        size: (u32, u32),
    ) {
        if source.is_image {
            return;
        } // No need to preload for images

        let cache = self.frame_cache.clone();
        let source = source.clone();
        let (w, h) = size;

        // Preload frames around current time (±2 seconds)
        thread::spawn(move || {
            let _preload_range = 2.0; // seconds
            let _step = 0.2; // every 200ms

            for offset in [0.2, 0.4, 0.6, 0.8, 1.0, -0.2, -0.4, -0.6, -0.8, -1.0] {
                let preload_time = current_time + offset;
                if preload_time < 0.0 {
                    continue;
                }

                let cache_key = FrameCacheKey::new(&source.path, preload_time, w, h);

                // Check if frame is already cached
                if let Ok(cache) = cache.lock() {
                    if cache.contains_key(&cache_key) {
                        continue; // Already cached
                    }
                }

                // Decode frame in background
                if let Some(image) = decode_video_frame_optimized(&source.path, preload_time, w, h)
                {
                    let cached_frame = CachedFrame {
                        image,
                        decoded_at: std::time::Instant::now(),
                        access_count: 0,
                        last_access: std::time::Instant::now(),
                    };

                    if let Ok(mut cache) = cache.lock() {
                        // Only cache if we're not over the limit
                        if cache.len() < 50 {
                            cache.insert(cache_key, cached_frame);
                        }
                    }
                }

                // Small delay to avoid overwhelming the system
                thread::sleep(Duration::from_millis(10));
            }
        });
    }

    pub(crate) fn present_yuv_with_frame(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        path: &str,
        t_sec: f64,
        vf_opt: Option<&native_decoder::VideoFrame>,
    ) -> Option<(
        YuvPixFmt,
        Arc<eframe::wgpu::Texture>,
        Arc<eframe::wgpu::Texture>,
    )> {
        if let Some(vf) = vf_opt {
            // Map NativeYuvPixFmt to local YuvPixFmt and handle P010->NV12 fallback
            let mut fmt = match vf.format {
                native_decoder::YuvPixFmt::Nv12 => YuvPixFmt::Nv12,
                native_decoder::YuvPixFmt::P010 => YuvPixFmt::P010,
            };
            let mut y: Vec<u8> = vf.y_plane.clone();
            let mut uv: Vec<u8> = vf.uv_plane.clone();
            let w = vf.width;
            let h = vf.height;
            if fmt == YuvPixFmt::P010 && !device_supports_16bit_norm(rs) {
                if let Some((_f, ny, nuv, nw, nh)) = decode_video_frame_nv12_only(path, t_sec) {
                    fmt = YuvPixFmt::Nv12;
                    y = ny;
                    uv = nuv;
                    let _ = (nw, nh);
                }
            }
            let key = FrameCacheKey::new(path, t_sec, 0, 0);
            self.nv12_cache.insert(
                key.clone(),
                Nv12Frame {
                    fmt,
                    y: y.clone(),
                    uv: uv.clone(),
                    w,
                    h,
                },
            );
            self.nv12_keys.push_back(key);
            while self.nv12_keys.len() > 64 {
                if let Some(old) = self.nv12_keys.pop_front() {
                    self.nv12_cache.remove(&old);
                }
            }
            self.upload_yuv_planes(rs, fmt, &y, &uv, w, h);
            let idx = self.ring_present;
            return Some((
                fmt,
                self.y_tex[idx].as_ref().unwrap().clone(),
                self.uv_tex[idx].as_ref().unwrap().clone(),
            ));
        }
        // Fallback to old path
        self.present_yuv(rs, path, t_sec)
    }

    pub(crate) fn present_yuv_from_bytes(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        fmt: YuvPixFmt,
        y_bytes: &[u8],
        uv_bytes: &[u8],
        w: u32,
        h: u32,
    ) -> Option<(
        YuvPixFmt,
        Arc<eframe::wgpu::Texture>,
        Arc<eframe::wgpu::Texture>,
    )> {
        // Ensure textures/buffers exist at this decoded size/format
        self.ensure_yuv_textures(rs, w, h, fmt);

        // Write into current ring slot
        let wi = self.ring_write % 3;

        let (y_bpp, uv_bpp_per_texel) = match fmt {
            YuvPixFmt::Nv12 => (1usize, 2usize),
            YuvPixFmt::P010 => (2usize, 4usize),
        };
        let y_w = w as usize;
        let y_h = h as usize;
        let uv_w = ((w + 1) / 2) as usize;
        let uv_h = ((h + 1) / 2) as usize;

        // Guard: verify plane lengths once; early out if mismatched
        let expected_y = y_w * y_bpp * y_h;
        let expected_uv = uv_w * uv_bpp_per_texel * uv_h;
        debug_assert_eq!(y_bytes.len(), expected_y, "Y plane size mismatch");
        debug_assert_eq!(uv_bytes.len(), expected_uv, "UV plane size mismatch");
        if y_bytes.len() != expected_y || uv_bytes.len() != expected_uv {
            let flag = PRESENT_SIZE_MISMATCH_LOGGED.get_or_init(|| AtomicBool::new(false));
            if !flag.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[present] size mismatch: got Y={} UV={}, expected Y={} UV={}",
                    y_bytes.len(),
                    uv_bytes.len(),
                    expected_y,
                    expected_uv
                );
            }
            return None;
        }

        let queue = &*rs.queue;

        if let Some(y_tex) = self.y_tex[wi].as_ref() {
            upload_plane(queue, &**y_tex, y_bytes, w, h, y_w * y_bpp, y_bpp);
        }

        if let Some(uv_tex) = self.uv_tex[wi].as_ref() {
            upload_plane(
                queue,
                &**uv_tex,
                uv_bytes,
                (w + 1) / 2,
                (h + 1) / 2,
                uv_w * uv_bpp_per_texel,
                uv_bpp_per_texel,
            );
        }

        // Persist last-good so fallback can reuse
        self.last_fmt = Some(fmt);
        self.y_size = (w, h);
        self.uv_size = ((w + 1) / 2, (h + 1) / 2);
        self.ring_present = wi;
        self.ring_write = (wi + 1) % 3;
        self.last_present_tick = self.last_present_tick.wrapping_add(1);
        self.last_cpu_tick = self.last_present_tick;

        let y_tex = self.y_tex[wi].as_ref()?.clone();
        let uv_tex = self.uv_tex[wi].as_ref()?.clone();
        Some((fmt, y_tex, uv_tex))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn present_nv12_zero_copy(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        zc: &native_decoder::IOSurfaceFrame,
    ) -> Option<(
        YuvPixFmt,
        Arc<eframe::wgpu::Texture>,
        Arc<eframe::wgpu::Texture>,
    )> {
        self.ensure_zero_copy_nv12_textures(rs, zc.width, zc.height);
        if let Some((y_arc, uv_arc)) = self
            .gpu_yuv
            .as_ref()
            .map(|g| (g.y_tex.clone(), g.uv_tex.clone()))
        {
            let queue = &*rs.queue;
            if let Err(e) = self
                .gpu_yuv
                .as_ref()
                .unwrap()
                .import_from_iosurface(queue, zc)
            {
                eprintln!("[zc] import_from_iosurface error: {}", e);
                return None;
            }
            #[cfg(target_os = "macos")]
            if !self.zc_logged {
                tracing::info!(
                    "[preview] imported NV12 planes: Y={}x{}  UV={}x{}",
                    zc.width,
                    zc.height,
                    (zc.width + 1) / 2,
                    (zc.height + 1) / 2
                );
                self.zc_logged = true;
            }
            // Persist last ZC for reuse
            self.set_last_zc_present(
                YuvPixFmt::Nv12,
                y_arc.clone(),
                uv_arc.clone(),
                zc.width,
                zc.height,
            );
            return Some((YuvPixFmt::Nv12, y_arc, uv_arc));
        }
        None
    }
}

fn find_jpeg_frame(buf: &[u8]) -> Option<(usize, usize)> {
    // SOI 0xFFD8, EOI 0xFFD9
    let mut start = None;
    for i in 0..buf.len().saturating_sub(1) {
        if start.is_none() && buf[i] == 0xFF && buf[i + 1] == 0xD8 {
            start = Some(i);
        }
        if let Some(s) = start {
            if buf[i] == 0xFF && buf[i + 1] == 0xD9 {
                return Some((s, i + 2));
            }
        }
    }
    None
}

fn decode_to_color_image(bytes: &[u8]) -> Option<egui::ColorImage> {
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    let data = img.into_raw();
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [w as usize, h as usize],
        &data,
    ))
}

// Optimized video frame decode at native size (no scaling; GPU handles fit)
fn decode_video_frame_optimized(
    path: &str,
    t_sec: f64,
    w: u32,
    h: u32,
) -> Option<egui::ColorImage> {
    // Decode one frame at requested size to match GPU upload
    let frame_bytes = (w as usize) * (h as usize) * 4;
    let out = std::process::Command::new("ffmpeg")
        .arg("-ss")
        .arg(format!("{:.3}", t_sec.max(0.0)))
        .arg("-i")
        .arg(path)
        .arg("-frames:v")
        .arg("1")
        .arg("-vf")
        .arg(format!("scale={}x{}:flags=fast_bilinear", w, h))
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("rgba")
        .arg("-threads")
        .arg("1")
        .arg("-")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !out.status.success() {
        return None;
    }
    if out.stdout.len() < frame_bytes {
        return None;
    }
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [w as usize, h as usize],
        &out.stdout[..frame_bytes],
    ))
}

// Decode video frame using native decoder
fn decode_video_frame_native(path: &str, t_sec: f64, w: u32, h: u32) -> Option<egui::ColorImage> {
    let config = DecoderConfig {
        hardware_acceleration: true,
        preferred_format: Some(native_decoder::YuvPixFmt::Nv12),
        zero_copy: false, // Phase 1 only
    };

    match create_decoder(path, config) {
        Ok(mut decoder) => {
            match decoder.decode_frame(t_sec) {
                Ok(Some(video_frame)) => {
                    // Convert YUV to RGBA for egui::ColorImage
                    let rgba = yuv_to_rgba(
                        &video_frame.y_plane,
                        &video_frame.uv_plane,
                        video_frame.width,
                        video_frame.height,
                        video_frame.format,
                    );

                    // Scale to requested size if needed
                    if video_frame.width == w && video_frame.height == h {
                        Some(egui::ColorImage::from_rgba_unmultiplied(
                            [w as usize, h as usize],
                            &rgba,
                        ))
                    } else {
                        // Simple nearest-neighbor scaling for now
                        let scaled =
                            scale_rgba_nearest(&rgba, video_frame.width, video_frame.height, w, h);
                        Some(egui::ColorImage::from_rgba_unmultiplied(
                            [w as usize, h as usize],
                            &scaled,
                        ))
                    }
                }
                Ok(None) => {
                    eprintln!("Native decoder: No frame at timestamp {:.3}s", t_sec);
                    None
                }
                Err(e) => {
                    eprintln!("Native decoder error: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to create native decoder: {}", e);
            None
        }
    }
}

// Convert YUV to RGBA (simple implementation)
fn yuv_to_rgba(
    y_plane: &[u8],
    uv_plane: &[u8],
    width: u32,
    height: u32,
    format: native_decoder::YuvPixFmt,
) -> Vec<u8> {
    let mut rgba = vec![0u8; (width * height * 4) as usize];

    match format {
        native_decoder::YuvPixFmt::Nv12 => {
            // NV12: Y plane + interleaved UV plane
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let y_idx = y * width as usize + x;
                    let uv_idx = (y / 2) * width as usize + (x / 2) * 2;

                    let y_val = y_plane[y_idx] as f32;
                    let u_val = uv_plane[uv_idx] as f32 - 128.0;
                    let v_val = uv_plane[uv_idx + 1] as f32 - 128.0;

                    // YUV to RGB conversion (ITU-R BT.601)
                    let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

                    let rgba_idx = (y * width as usize + x) * 4;
                    rgba[rgba_idx] = r;
                    rgba[rgba_idx + 1] = g;
                    rgba[rgba_idx + 2] = b;
                    rgba[rgba_idx + 3] = 255; // Alpha
                }
            }
        }
        native_decoder::YuvPixFmt::P010 => {
            // P010: 10-bit YUV (simplified to 8-bit for now)
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let y_idx = y * width as usize + x;
                    let uv_idx = (y / 2) * width as usize + (x / 2) * 2;

                    // Convert 10-bit to 8-bit (shift right by 2)
                    let y_val = (y_plane[y_idx] as f32) * 4.0;
                    let u_val = (uv_plane[uv_idx] as f32) * 4.0 - 128.0;
                    let v_val = (uv_plane[uv_idx + 1] as f32) * 4.0 - 128.0;

                    // YUV to RGB conversion
                    let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

                    let rgba_idx = (y * width as usize + x) * 4;
                    rgba[rgba_idx] = r;
                    rgba[rgba_idx + 1] = g;
                    rgba[rgba_idx + 2] = b;
                    rgba[rgba_idx + 3] = 255; // Alpha
                }
            }
        }
    }

    rgba
}

// Simple nearest-neighbor scaling
fn scale_rgba_nearest(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];

    for y in 0..dst_h as usize {
        for x in 0..dst_w as usize {
            let src_x = (x as f32 * src_w as f32 / dst_w as f32) as usize;
            let src_y = (y as f32 * src_h as f32 / dst_h as f32) as usize;

            let src_idx = (src_y * src_w as usize + src_x) * 4;
            let dst_idx = (y * dst_w as usize + x) * 4;

            if src_idx + 3 < src.len() && dst_idx + 3 < dst.len() {
                dst[dst_idx] = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
                dst[dst_idx + 2] = src[src_idx + 2];
                dst[dst_idx + 3] = src[src_idx + 3];
            }
        }
    }

    dst
}

// Decode a single frame to NV12 or P010 at native size.
fn decode_video_frame_yuv(
    path: &str,
    t_sec: f64,
) -> Option<(YuvPixFmt, Vec<u8>, Vec<u8>, u32, u32)> {
    let info = media_io::probe_media(std::path::Path::new(path)).ok()?;
    let w = info.width?;
    let h = info.height?;
    // Try P010 first
    let out10 = std::process::Command::new("ffmpeg")
        .arg("-ss")
        .arg(format!("{:.3}", t_sec.max(0.0)))
        .arg("-i")
        .arg(path)
        .arg("-frames:v")
        .arg("1")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("p010le")
        .arg("-threads")
        .arg("1")
        .arg("-")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if out10.status.success() {
        let exp10 = (w as usize) * (h as usize) * 3; // Y:2 bytes * w*h ; UV: w*h bytes (2x16-bit at half res)
        if out10.stdout.len() >= exp10 {
            let y_bytes = (w as usize) * (h as usize) * 2;
            let y = out10.stdout[..y_bytes].to_vec();
            let uv = out10.stdout[y_bytes..y_bytes + (exp10 - y_bytes)].to_vec();
            return Some((YuvPixFmt::P010, y, uv, w, h));
        }
    }
    // Fallback NV12
    let expected = (w as usize) * (h as usize) + (w as usize) * (h as usize) / 2;
    let out = std::process::Command::new("ffmpeg")
        .arg("-ss")
        .arg(format!("{:.3}", t_sec.max(0.0)))
        .arg("-i")
        .arg(path)
        .arg("-frames:v")
        .arg("1")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("nv12")
        .arg("-threads")
        .arg("1")
        .arg("-")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.len() < expected {
        return None;
    }
    let y_size = (w as usize) * (h as usize);
    let y = out.stdout[..y_size].to_vec();
    let uv = out.stdout[y_size..y_size + (expected - y_size)].to_vec();
    Some((YuvPixFmt::Nv12, y, uv, w, h))
}

fn decode_video_frame_nv12_only(
    path: &str,
    t_sec: f64,
) -> Option<(YuvPixFmt, Vec<u8>, Vec<u8>, u32, u32)> {
    let info = media_io::probe_media(std::path::Path::new(path)).ok()?;
    let w = info.width?;
    let h = info.height?;
    let expected = (w as usize) * (h as usize) + (w as usize) * (h as usize) / 2;
    let out = std::process::Command::new("ffmpeg")
        .arg("-ss")
        .arg(format!("{:.3}", t_sec.max(0.0)))
        .arg("-i")
        .arg(path)
        .arg("-frames:v")
        .arg("1")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("nv12")
        .arg("-threads")
        .arg("1")
        .arg("-")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.len() < expected {
        return None;
    }
    let y_size = (w as usize) * (h as usize);
    let y = out.stdout[..y_size].to_vec();
    let uv = out.stdout[y_size..y_size + (expected - y_size)].to_vec();
    Some((YuvPixFmt::Nv12, y, uv, w, h))
}

pub(crate) fn device_supports_16bit_norm(rs: &eframe::egui_wgpu::RenderState) -> bool {
    rs.device
        .features()
        .contains(eframe::wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
}

pub(crate) fn upload_plane(
    queue: &eframe::wgpu::Queue,
    texture: &eframe::wgpu::Texture,
    src: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    bytes_per_pixel: usize,
) {
    if width == 0 || height == 0 {
        return;
    }
    let required = (width as usize) * bytes_per_pixel;
    assert!(stride >= required, "stride too small for upload");
    assert!(
        src.len() >= stride * height as usize,
        "plane buffer too small"
    );

    let align = eframe::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded = ((required + align - 1) / align) * align;

    if stride == padded {
        queue.write_texture(
            eframe::wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: eframe::wgpu::Origin3d::ZERO,
                aspect: eframe::wgpu::TextureAspect::All,
            },
            src,
            eframe::wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(height),
            },
            eframe::wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    } else {
        let mut repacked = vec![0u8; padded * height as usize];
        for row in 0..height as usize {
            let src_off = row * stride;
            let dst_off = row * padded;
            repacked[dst_off..dst_off + required]
                .copy_from_slice(&src[src_off..src_off + required]);
        }
        queue.write_texture(
            eframe::wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: eframe::wgpu::Origin3d::ZERO,
                aspect: eframe::wgpu::TextureAspect::All,
            },
            &repacked,
            eframe::wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(height),
            },
            eframe::wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}

#[derive(Clone)]
struct Nv12Frame {
    fmt: YuvPixFmt,
    y: Vec<u8>,
    uv: Vec<u8>,
    w: u32,
    h: u32,
}

// Using media_io::YuvPixFmt

// Optimized image decoding
fn decode_image_optimized(path: &str, w: u32, h: u32) -> Option<egui::ColorImage> {
    // For images, use the image crate directly for better performance
    let img = image::open(path).ok()?;
    let resized = img.resize(w, h, image::imageops::FilterType::Lanczos3);
    let rgba = resized.to_rgba8();
    let (width, height) = rgba.dimensions();

    Some(egui::ColorImage::from_rgba_unmultiplied(
        [width as usize, height as usize],
        &rgba.into_raw(),
    ))
}

// LRU eviction for frame cache
fn evict_lru_frames(cache: &mut HashMap<FrameCacheKey, CachedFrame>, count: usize) {
    if cache.len() <= count {
        return;
    }

    // Collect frames with their last access times
    let mut frames_with_time: Vec<(FrameCacheKey, std::time::Instant)> = cache
        .iter()
        .map(|(key, frame)| (key.clone(), frame.last_access))
        .collect();

    // Sort by last access time (oldest first)
    frames_with_time.sort_by_key(|(_, time)| *time);

    // Remove the oldest frames
    for (key, _) in frames_with_time.into_iter().take(count) {
        cache.remove(&key);
    }
}

fn grab_frame_at(path: &str, size: (u32, u32), t_sec: f64) -> Option<egui::ColorImage> {
    let (w, h) = size;
    decode_video_frame_optimized(path, t_sec, w, h)
}

// Efficient frame cache key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FrameCacheKey {
    pub(crate) path: String,
    pub(crate) time_sec: u32, // Rounded to nearest 0.1 second for cache efficiency
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl FrameCacheKey {
    pub(crate) fn new(path: &str, time_sec: f64, width: u32, height: u32) -> Self {
        Self {
            path: path.to_string(),
            time_sec: (time_sec * 10.0).round() as u32, // 0.1 second precision
            width,
            height,
        }
    }
}

// Cached frame with metadata
#[derive(Clone)]
struct CachedFrame {
    pub(crate) image: egui::ColorImage,
    pub(crate) decoded_at: std::time::Instant,
    pub(crate) access_count: u32,
    pub(crate) last_access: std::time::Instant,
}

// Frame buffer used by the preview scheduler (kept for compatibility)
struct FrameBuffer {
    pub(crate) pts: f64,
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) bytes: Vec<u8>,
}

// (removed legacy standalone WGPU context to avoid mixed versions)
