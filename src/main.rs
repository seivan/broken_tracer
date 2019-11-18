// #[macro_use]
// extern crate glsl_to_spirv_macros_impl;

#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]

use log::info;
// use wgpu::winit;
use sdl2;

//use wgpu_native;
//use sdl2_sys;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use std::io::Cursor;
use std::os::raw::*;
use std::time::Duration;

// use euclid::*;

/// Alias for ```euclid::default::Point2D<f32>```.
pub type Point = euclid::default::Point2D<f32>;
pub type Angle = euclid::Angle<f32>;

/// Alias for ```euclid::default::Point2D<f32>```.
pub type Vector = euclid::default::Vector2D<f32>;

/// Alias for ```euclid::default::Size2D<f32>```.
pub type Size = euclid::default::Size2D<f32>;

/// Alias for ```euclid::default::Rect<f32>```
pub type Rect = euclid::default::Rect<f32>;

/// Alias for ```euclid::default::Transform2D<f32>```
pub type Transform2D = euclid::default::Transform2D<f32>;

pub type Transform3D = euclid::default::Transform3D<f32>;

//use sdl2::event::Event;
// use wgpu::*;
//
use rand::{thread_rng, Rng};
static TEXDESC: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
    bindings: &[wgpu::BindGroupLayoutBinding {
        binding: 0,
        visibility: wgpu::ShaderStage::FRAGMENT,
        ty: wgpu::BindingType::SampledTexture {
            multisampled: false,
            dimension: wgpu::TextureViewDimension::D2,
        },
    }],
};

pub struct Shader {}
trait Something {
    fn set_sub_data<T: 'static>(
        &self,
        ttype: &str,
        device: &mut wgpu::Device,
        queue: &mut wgpu::Queue,
        offset: i32,
        new_data: &[T],
    ) where
        T: std::marker::Copy,
        T: std::fmt::Debug;

    fn set_sub_data_with_encoder<T: 'static>(
        &self,
        ttype: &str,
        device: &mut wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        offset: i32,
        new_data: &[T],
    ) where
        T: std::marker::Copy,
        T: std::fmt::Debug;
}

impl Something for wgpu::Buffer {
    fn set_sub_data<T: 'static>(
        &self,
        ttype: &str,
        device: &mut wgpu::Device,
        queue: &mut wgpu::Queue,
        offset: i32,
        new_data: &[T],
    ) where
        T: std::marker::Copy,
        T: std::fmt::Debug,
    {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        self.set_sub_data_with_encoder(ttype, device, &mut encoder, offset, new_data);
        queue.submit(&[encoder.finish()]);
    }

    fn set_sub_data_with_encoder<T: 'static>(
        &self,
        ttype: &str,
        device: &mut wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        offset: i32,
        new_data: &[T],
    ) where
        T: std::marker::Copy,
        T: std::fmt::Debug,
    {
        let count = new_data.len();
        if count == 0 {
            return;
        }
        let temp_buf = device
            .create_buffer_mapped(count, wgpu::BufferUsage::COPY_SRC)
            .fill_from_slice(new_data);

        let encoder = encoder;

        encoder.copy_buffer_to_buffer(
            &temp_buf,
            0 as u64,
            self,
            offset as u64,
            std::mem::size_of_val(new_data) as u64,
        );
    }
}
mod framework {

    pub fn cast_slice<T>(data: &[T]) -> &[u8] {
        use std::mem::size_of;
        use std::slice::from_raw_parts;

        unsafe { from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<T>()) }
    }

    #[allow(dead_code)]
    pub enum ShaderStage {
        Vertex,
        Fragment,
    }

    pub fn load_glsl(name: &str, stage: ShaderStage) -> Vec<u32> {
        use std::fs::read_to_string;
        use std::io::Read;
        use std::path::PathBuf;

        let mut compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_include_callback(|name, typez, anem, size| {
            println!("name = {:?}", name);
            println!("types = {:?}", typez);
            println!("size = {:?}", size);
            let x = include_str!("../data/cube.vert").to_owned();
            println!("x = {:?}", x);
            Ok(shaderc::ResolvedInclude {
                resolved_name: "cube.vert".to_owned(),
                content: x,
            })
        });
        let vbinary_result = compiler
            .compile_into_spirv(
                include_str!("../data/some_shader.glsl"),
                shaderc::ShaderKind::Vertex,
                "shader",
                "main",
                Some(&options),
            )
            .unwrap();

        let fbinary_result = compiler
            .compile_into_spirv(
                include_str!("../data/cube.frag"),
                shaderc::ShaderKind::Fragment,
                "shaderz",
                "main",
                None,
            )
            .unwrap();

        let ty = match stage {
            self::ShaderStage::Vertex => vbinary_result.as_binary(),
            self::ShaderStage::Fragment => fbinary_result.as_binary(),
        };

        let output = ty;

        ty.to_vec()
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    vert_pos: [f32; 2],
    tex: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct QuadInstance {
    color: [f32; 4],
    pos: [[f32; 2]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Wrapper<T>(T)
where
    T: Clone + Copy;

const fn vertex(x: f32, y: f32, tx_x: f32, tx_y: f32) -> Vertex {
    Vertex {
        vert_pos: [x, y],
        tex: [tx_x, tx_y],
    }
}

// coord: [100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 200.0, 200.0, 0.0, 1.0],
static VERTS: [Vertex; 4] = create_vertices();
const fn create_vertices() -> [Vertex; 4] {
    let vertex_data = [
        //ORIGIN TOP/LEFT - TriangleStrip
        // front_face: wgpu::FrontFace::Ccw,
        // vertex(1.0, 1.0, 1.0, 1.0),
        // vertex(1.0, 0.0, 1.0, 0.0),
        // vertex(0.0, 1.0, 0.0, 1.0),
        // vertex(0.0, 0.0, 0.0, 0.0),

        //ORIGIN TOP/LEFT -
        // TriangleStrip
        // front_face:  Cw,
        // Vulkan coordinate system, texture moves down - original
        // vertex(0.0, 0.0, 1.0, 1.0),
        // vertex(0.0, 1.0, 1.0, 0.0),
        // vertex(1.0, 0.0, 0.0, 1.0),
        // vertex(1.0, 1.0, 0.0, 0.0),
        ///// NO IDEA WHAT THE FUCK THIS IS, but it works when rendering to texture.
        vertex(0.0, 0.0, 0.0, 1.0),
        vertex(0.0, 1.0, 0.0, 0.0),
        vertex(1.0, 0.0, 1.0, 1.0),
        vertex(1.0, 1.0, 1.0, 0.0),
        // //ORIGIN TOP/LEFT - TriangleStrip
        // // front_face: wgpu::FrontFace::Cw,
        // //OpenGL, texture moves up.
        // vertex(0.0, 0.0, 0.0, 0.0),
        // vertex(0.0, 1.0, 0.0, 1.0),
        // vertex(1.0, 0.0, 1.0, 0.0),
        // vertex(1.0, 1.0, 1.0, 1.0),
    ];
    vertex_data

    //    vertex_data.to_vec()
}

struct Cube {
    add_one: bool,
    pressed: bool,
    instances: Vec<QuadInstance>,
    verts: Vec<Vertex>,
    scale_factor: f32,
    size: (f32, f32),
    instance_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    //fragment_uniform_buf: wgpu::Buffer,
    texture_layout: wgpu::BindGroupLayout,

    bind_group: wgpu::BindGroup,
    sprite: Texture,
    scene_bind_group: wgpu::BindGroup,
    scene_view: wgpu::TextureView,
    scene_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
}

struct Texture {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    extent: wgpu::Extent3d, //110 97
}
impl Cube {
    fn make_texture(
        device: &mut wgpu::Device,
        queue: &mut wgpu::Queue,
        texture_group_layout: &wgpu::BindGroupLayout,
        image: &'static [u8],
    ) -> Texture {
        let mut image = image::load_from_memory_with_format(&image[..], image::PNG)
            .expect("NO IMAGE?")
            .to_rgba();

        for pixel in image.pixels_mut() {
            let alpha = pixel[3] as f32 / 255.0;
            pixel[0] = (((pixel[0] as f32 / 255.0) * alpha) * 255.0) as u8;
            pixel[1] = (((pixel[1] as f32 / 255.0) * alpha) * 255.0) as u8;
            pixel[2] = (((pixel[2] as f32 / 255.0) * alpha) * 255.0) as u8;
        }

        let texture_extent = wgpu::Extent3d {
            width: image.dimensions().0,
            height: image.dimensions().1,
            depth: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });
        let texture_view = texture.create_default_view();
        let temp_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: image.len() as u64,
            usage: wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::COPY_DST,
        });

        temp_buf.set_sub_data("Create_texture", device, queue, 0, &image);
        let mut init_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        init_encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &temp_buf,
                offset: 0,
                row_pitch: 4 * image.dimensions().0,
                image_height: image.dimensions().1,
            },
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
            texture_extent,
        );

        queue.submit(&[init_encoder.finish()]);

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_group_layout,
            bindings: &[wgpu::Binding {
                binding: TEXDESC.bindings.first().unwrap().binding,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            }],
        });
        Texture {
            bind_group: texture_bind_group,
            extent: texture_extent,
            view: texture_view,
        }
    }

    fn init(
        device: &mut wgpu::Device,
        queue: &mut wgpu::Queue,
        sc_desc: &wgpu::SwapChainDescriptor,
        scale_factor: f32,
    ) -> Self {
        use std::mem;

        let (width, height) = (sc_desc.width as f32, sc_desc.height as f32);
        println!("width: {:?} - height: {:?}", width, height);
        let texture_group_layout = device.create_bind_group_layout(&TEXDESC);

        let texture_extent = wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth: 1,
        };

        let desc = wgpu::TextureDescriptor {
            size: texture_extent,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        };
        let texture = device.create_texture(&desc);

        let scene_view = texture.create_default_view();
        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_group_layout,
            bindings: &[wgpu::Binding {
                binding: TEXDESC.bindings.first().unwrap().binding,
                resource: wgpu::BindingResource::TextureView(&scene_view),
            }],
        });

        let quad_size = mem::size_of::<QuadInstance>();

        let scene_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: (1 * quad_size) as u64, //(7800000 * quad_size) as u32,
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });

        {
            let projection = Transform2D::identity().pre_translate((0.0, height).into());

            let bb = projection.transform_point(Point::new(0.0, 0.0 * -1.0));

            let affine: Transform2D = Transform2D::identity()
                .pre_translate((0.0, 0.0).into())
                .pre_scale(800.0, 600.0);

            let mut pos: [[f32; 2]; 4] = [[0.0; 2]; 4];
            for (i, v) in VERTS.iter().enumerate() {
                //Vectors aren't **translated**. So we gotta use points for translations.
                // Vectors will however **scale**

                let gl_pos = euclid::Point2D::new(v.vert_pos[0], v.vert_pos[1]);
                let actual = affine.transform_point(gl_pos);
                pos[i] = [actual.x, actual.y];
            }

            let q = QuadInstance {
                color: [1.0, 1.0, 1.0, 1.0],
                pos: pos,
            };

            // quad.coord.copy_from_slice(&model.data.as_ref()[0..16]);
            scene_buf.set_sub_data("scren", device, queue, 0, &[q]);
        }

        let vertex_size = mem::size_of::<[f32; 2]>();

        //        let vertices = Vec::from(vertex_data);

        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: (4 * vertex_size) as u64,
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });

        vertex_buf.set_sub_data(
            "verts",
            device,
            queue,
            0,
            VERTS.iter().map(|x| x.tex).collect::<Vec<_>>().as_ref(),
        );

        let quad_size = mem::size_of::<QuadInstance>();

        let instances = Vec::new();

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: (1000 * quad_size) as u64, //(7800000 * quad_size) as u32,
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
        });

        //        instance_buf.set_sub_data(0, framework::cast_slice(&instances[..]));

        // Create pipeline layout

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler,
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout, &texture_group_layout],
        });

        let sprite = Cube::make_texture(
            device,
            queue,
            &texture_group_layout,
            include_bytes!("../data/wabbit_alpha.png"),
        );

        // Create other resources
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: i32::max_value() as f32,

            compare_function: wgpu::CompareFunction::Never,
        });
        let uniform_buf;
        {
            let projection = Transform3D::ortho(0.0, width, height, 0.0, -1.0, 1.0);

            uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                size: 64,
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

            uniform_buf.set_sub_data(
                "uniform_buf; projection",
                device,
                queue,
                0,
                projection.to_row_major_array().as_ref(),
            );
        }

        let x: [[f32; 4]; 2] = [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]];
        let size = mem::size_of_val(&x) as u64;

        let fragment_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: size,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        fragment_uniform_buf.set_sub_data("fragment buf, color", device, queue, 0, x.as_ref());
        println!("size = {:?}", size);
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buf,
                        range: 0..size,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &fragment_uniform_buf,
                        range: 0..size,
                    },
                },
            ],
        });

        // Create the render pipeline
        let vs_bytes = framework::load_glsl("cube.vert", framework::ShaderStage::Vertex);
        let fs_bytes = framework::load_glsl("cube.frag", framework::ShaderStage::Fragment);
        let vs_module = device.create_shader_module(&vs_bytes);
        let fs_module = device.create_shader_module(&fs_bytes);
        let number = "hi";

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Cw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleStrip,
            color_states: &[wgpu::ColorStateDescriptor {
                format: sc_desc.format,
                color_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None, //Some(wgpu::DepthStencilStateDescriptor {
            //     format: wgpu::TextureFormat::Bgra8Unorm,
            //     depth_write_enabled: false,
            //     stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
            //     stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
            //     stencil_read_mask: 0u32,
            //     stencil_write_mask: 0u32,
            //     depth_compare: wgpu::CompareFunction::Never,
            // }),
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[
                wgpu::VertexBufferDescriptor {
                    stride: mem::size_of::<[f32; 2]>() as u64,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[wgpu::VertexAttributeDescriptor {
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float2,
                        offset: 0,
                    }],
                },
                wgpu::VertexBufferDescriptor {
                    //QuadInstance size.
                    //mem::size_of::<[f32; 16]>() +
                    stride: mem::size_of::<QuadInstance>() as u64,
                    step_mode: wgpu::InputStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float4,
                            offset: 0, //4*4 //mem::size_of::<Vertex>() as u32,
                        },
                        wgpu::VertexAttributeDescriptor {
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float2,
                            offset: 16,
                        },
                        wgpu::VertexAttributeDescriptor {
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float2,
                            offset: 16 + 4 * 2,
                        },
                        wgpu::VertexAttributeDescriptor {
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float2,
                            offset: 16 + 4 * 4, //4*4 //mem::size_of::<Vertex>() as u32,
                        },
                        wgpu::VertexAttributeDescriptor {
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float2,
                            offset: 16 + 4 * 6, //4*4 //mem::size_of::<Vertex>() as u32,
                        },
                    ],
                },
            ],
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        // Done
        Cube {
            add_one: false,
            pressed: false,
            instances,
            verts: Vec::new(),
            scale_factor,
            size: (width, height),
            uniform_buf,
            // fragment_uniform_buf,
            instance_buf,
            vertex_buf,
            bind_group,
            texture_layout: texture_group_layout,
            sprite,
            scene_bind_group,
            scene_view,
            scene_buf,
            pipeline,
        }
    }

    fn update(
        &mut self,
        device: &mut wgpu::Device,
        queue: &mut wgpu::Queue,
        event: sdl2::event::Event,
    ) {
        let sprite = &self.sprite;
        let instances = &mut self.instances;
        let verts = &mut self.verts;
        let length = instances.len();
        // if length >= 1 {
        //     return;
        // }
        let uniform_buf = &self.uniform_buf;
        // let instance_buf = &self.instance_buf;
        let vertex_buf = &self.vertex_buf;

        let (width, height) = (100.0, 100.0);
        let scale_factor = self.scale_factor;
        let screen_height = self.size.1;
        let screen_width = self.size.0;

        let projection = Transform2D::identity().pre_translate((0.0, screen_height).into());

        let mut add_sprite = |x: f32, y: f32| {
            let affine: Transform2D = Transform2D::identity()
                .pre_translate((x, y).into())
                .pre_scale(sprite.extent.width as f32, sprite.extent.height as f32);

            let mut pos: [[f32; 2]; 4] = [[0.0; 2]; 4];
            for (i, v) in VERTS.iter().enumerate() {
                let gl_pos = euclid::Point2D::new(v.vert_pos[0], v.vert_pos[1]);
                let actual = affine.transform_point(gl_pos);
                pos[i] = [actual.x, actual.y];
            }

            let quad = QuadInstance {
                color: [1.0, 1.0, 1.0, 1.0],
                pos: pos,
            };
            instances.push(quad);
        };

        let mut rng = thread_rng();

        match event {
            Event::MouseMotion { x, y, .. } => {
                let x = x as f32 * scale_factor; // * 1000.0;
                let y = y as f32 * scale_factor; // * 1000.0;
                let bb = projection.transform_point(Point::new(x, y * -1.0));

                if let Some(quad) = instances.last_mut() {
                    let rect: Rect = Rect::new(
                        Point::new(x, y),
                        Size::new(sprite.extent.width as f32, sprite.extent.height as f32),
                    );

                    let tf = rect.center().to_vector();

                    let affine: Transform2D = Transform2D::identity()
                        .pre_translate((bb.x, bb.y).into())
                        .pre_scale(sprite.extent.width as f32, sprite.extent.height as f32);

                    let mut pos: [[f32; 2]; 4] = [[0.0; 2]; 4];
                    for (i, v) in VERTS.iter().enumerate() {
                        let gl_pos = Point::new(v.vert_pos[0], v.vert_pos[1]);
                        let actual = affine.transform_point(gl_pos);

                        pos[i] = [actual.x, actual.y];
                    }
                    // println!("pos = {:?}", pos);

                    *quad = QuadInstance {
                        color: quad.color,
                        pos: pos,
                    };
                }
            }
            Event::MouseButtonDown { x, y, .. } => {
                // self.pressed = true;
            }

            Event::MouseButtonUp { x, y, .. } => {
                for _ in 0..1 {
                    let bb = projection.transform_point(Point::new(x as f32, y as f32 * -1.0));
                    add_sprite(bb.x, bb.y);
                }

                println!("instance count = {:?}", self.instances.len());
            }

            //This gives logical size on the touchpad/touch-screen - not cursor directory.
            Event::FingerUp { x, y, .. } => {
                // self.pressed = false;
                // self.add_one = true;
                let x = (x as f32 * self.scale_factor) * screen_width; // / 2.0;
                let y = (y as f32 * self.scale_factor) * screen_height; // / 2.0;
                                                                        //                add_sprite(x, y);
            }

            _ => {}
        }

        // self.instance_buf.set_sub_data(
        //     "instances, quad",
        //     device,
        //     queue,
        //     0,
        //     self.instances.as_ref(),
        // );

        self.instance_buf = device
            .create_buffer_mapped(self.instances.len(), wgpu::BufferUsage::VERTEX)
            .fill_from_slice(self.instances.as_ref());
    }

    // fn render(&mut self, encoder: &mut wgpu::CommandEncoder, frame: &wgpu::TextureView) {
    fn render(&mut self, encoder: &mut wgpu::CommandEncoder) {
        {
            let frame = &self.scene_view;
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    resolve_target: None,
                    attachment: frame,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    },
                }],
                depth_stencil_attachment: None, //Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                                                //     depth_load_op: wgpu::LoadOp::Clear,
                                                //     depth_store_op: wgpu::StoreOp::Clear,
                                                //     stencil_load_op: wgpu::LoadOp::Clear,
                                                //     stencil_store_op: wgpu::StoreOp::Clear,
                                                //     clear_stencil: 0,
                                                //     clear_depth: 0.0,
                                                //     attachment: &self.scene_view, //empty texture
                                                // }),
            });

            // rpass.set_viewport(0.0, 0.0, 800.0, 600.0, -10.0, 10.0);
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_bind_group(1, &self.sprite.bind_group, &[]);

            rpass.set_vertex_buffers(0, &[(&self.vertex_buf, 0), (&self.instance_buf, 0)]);

            rpass.draw(0..VERTS.len() as u32, 0..self.instances.len() as u32);
        }
    }
}

pub struct GodObject {
    recreate_swap_chain: bool,
    cube: Cube,

    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface,
    swap_chain: wgpu::SwapChain,
    sdl_context: sdl2::Sdl,
    canvas: sdl2::render::WindowCanvas,
    event_pump: sdl2::EventPump,
}

#[no_mangle]
pub extern "C" fn setup() -> *mut GodObject {
    unsafe { sdl2::sys::SDL_SetMainReady() };

    env_logger::init();

    //Should be a flag to window creation like Vulkan, make PR to SDL2.
    // #[cfg(feature = "metal")]
    sdl2::hint::set("SDL_RENDER_DRIVER", "metal");
    // sdl2::hint::set("SDL_RENDER_VSYNC", "1");
    // sdl2::hint::set("SDL_HINT_RENDER_VSYNC", "1");

    //Not needed on macOS(?) - seems wgpu can drive this?
    // sdl2::hint::set("SDL_VIDEO_HIGHDPI_DISABLED", "1");

    fn with_graphics_api() -> Option<u32> {
        for (index, item) in sdl2::render::drivers().enumerate() {
            println!("{:?}", item);
            println!("{:?}", index);
            if item.name == "metal" {
                return Some(index as u32);
            }
        }
        None
    }
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("rust-sdl2 demo", 800, 600)
        .position_centered()
        .allow_highdpi()
        // .vulkan()
        .resizable()
        .build()
        .unwrap();

    let size = &window.size();
    let x = with_graphics_api();
    println!("{:?}", x);
    // let mut canvas = window.into_canvas().index(with_graphics_api().unwrap()).build().unwrap();

    let mut sys_wm_info: sdl2::sys::SDL_SysWMinfo;

    unsafe {
        use std::mem;

        sys_wm_info = mem::uninitialized();
    }

    // sys_wm_info.version = sdl2::version::version();

    // println!("sdl version: {}.{}.{}", sys_wm_info.version.major, sys_wm_info.version.minor, sys_wm_info.version.patch);

    // #[cfg(feature = "vulkan")]
    unsafe {
        sdl2::sys::SDL_GetWindowWMInfo(window.raw(), &mut sys_wm_info);
        println!("{:?}", sys_wm_info.version.major);
        println!("{:?}", sys_wm_info.subsystem);
    }

    let canvas = window.into_canvas().build().unwrap();
    println!("{:?}", canvas.scale());

    //Not needed on macOS(?)
    //Not available for macOS.
    // unsafe { sdl2::sys::SDL_iPhoneSetEventPump(false); }
    // #[cfg(feature = "metal")]
    let metal = unsafe { sdl2::sys::SDL_RenderGetMetalLayer(canvas.raw()) };
    // #[cfg(feature = "metal")]
    println!("Metal Layer Found: {:?}", metal);

    let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        backends: wgpu::BackendBit::PRIMARY,
    })
    .unwrap();
    let (mut device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });
    //    #[cfg(feature = "metal")]
    let surface = wgpu::Surface::create_surface_from_core_animation_layer(metal);

    let width = size.0 as u32;
    let height = size.1 as u32;
    let scale = 1.0;

    let sc_desc = make_swapchain(width, height);
    let swap_chain = device.create_swap_chain(&surface, &sc_desc);

    let cube = Cube::init(&mut device, &mut queue, &sc_desc, scale);

    let event_pump = sdl_context.event_pump().unwrap();

    let x = GodObject {
        recreate_swap_chain: false,
        cube: cube,
        device,
        queue,
        surface,
        swap_chain,
        sdl_context,
        canvas,
        event_pump,
    };
    let x = Box::new(x);
    Box::into_raw(x)
}

fn make_swapchain(width: u32, height: u32) -> wgpu::SwapChainDescriptor {
    wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: width,
        height: height,
        present_mode: wgpu::PresentMode::NoVsync,
    }
}

#[no_mangle]
pub extern "C" fn start_event_block(god_object: *mut GodObject) {
    let god_object = unsafe { &mut *god_object };
    let sdl_context = &mut god_object.sdl_context;
    let canvas = &mut god_object.canvas;
    let event_pump = &mut god_object.event_pump;
    let surface = &mut god_object.surface;
    let device = &mut god_object.device;
    let queue = &mut god_object.queue;

    // let mut swap_chain = &mut god_object.swap_chain;
    // println!("FIRST {:?}", swap_chain);
    for event in event_pump.poll_iter() {
        match event {
            sdl2::event::Event::Quit { .. }
            | sdl2::event::Event::KeyDown {
                keycode: Some(Keycode::Escape),
                ..
            } => std::process::exit(0),
            sdl2::event::Event::Window { win_event, .. } => {
                println!("WINDOW EVENT {:?}", win_event);
                use sdl2::event::WindowEvent::*;
                match win_event {
                    Resized(_, _) | SizeChanged(_, _) | Maximized | Hidden | Minimized => {
                        println!("Recreating swapchain");
                        god_object.recreate_swap_chain = true;
                    }
                    _ => {}
                }
            }
            _ => {
                let example = &mut god_object.cube;

                example.update(device, queue, event);
            }
        }
    }

    if god_object.recreate_swap_chain {
        let (width, height) = canvas.window().size();
        println!("RECREATED SWAP CHAIN: {:?} - {:?}", width, height);

        let sc_desc = make_swapchain(width, height);
        let mut example = &mut god_object.cube;

        example.size = (width as f32, height as f32);
        god_object.swap_chain = device.create_swap_chain(&surface, &sc_desc);
        god_object.recreate_swap_chain = false;

        let projection = Transform3D::ortho(0.0, example.size.0, example.size.1, 0.0, -1.0, 1.0);

        example.uniform_buf.set_sub_data(
            "uniform buf, just projection",
            device,
            queue,
            0,
            projection.to_row_major_array().as_ref(),
        );
    }

    // The rest of the game loop goes here...
    // ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    // let frame = god_object.swap_chain.get_next_texture();
    //

    render_test(god_object);
}
use std::time::Instant;

fn render_test(god_object: &mut GodObject) {
    let device = &mut god_object.device;
    let queue = &mut god_object.queue;
    let cube = &mut god_object.cube;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

    // let tex = god_object.swap_chain.get_next_texture().unwrap();
    // cube.render(&mut encoder, &tex.view);

    cube.render(&mut encoder);
    queue.submit(&[encoder.finish()]);
    // let now = Instant::now();
    // println!("passed: {}", now.elapsed().as_millis());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    let tex = god_object.swap_chain.get_next_texture().unwrap();

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                resolve_target: None,
                attachment: &tex.view,
                load_op: wgpu::LoadOp::Clear,
                store_op: wgpu::StoreOp::Store,
                clear_color: wgpu::Color {
                    r: 0.33,
                    g: 0.33,
                    b: 0.33,
                    a: 1.0,
                },
            }],
            depth_stencil_attachment: None,
        });

        rpass.set_pipeline(&god_object.cube.pipeline);
        rpass.set_bind_group(0, &god_object.cube.bind_group, &[]);
        rpass.set_bind_group(1, &god_object.cube.scene_bind_group, &[]);

        rpass.set_vertex_buffers(
            0,
            &[
                (&god_object.cube.vertex_buf, 0),
                (&god_object.cube.scene_buf, 0),
            ],
        );

        rpass.draw(0..VERTS.len() as u32, 0..1 as u32);
    }
    //

    queue.submit(&[encoder.finish()]);
}

//#[cfg(feature = "metal")]
fn main() {
    let obj = setup();
    loop {
        // for mouse lag
        // std::thread::sleep(std::time::Duration::from_millis(8));
        start_event_block(obj);
    }
}
