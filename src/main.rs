use crate::prelude::*;
use std::env;
use std::process::exit;
use std::sync::LazyLock;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod prelude {
    pub use crate::app::State;
    pub use crate::geom::WeightedPoint;
    pub use crate::voronoi::{Hovered, Voronoi};
    pub use anyhow::Error;
    pub use notan::draw::*;
    pub use notan::math::{DVec2, Vec2};
    pub use notan::prelude::*;
    use std::ops::RangeInclusive;
    pub use tracing::{debug, error, info, instrument, trace};

    pub const PT_LBL_OFFSET: f32 = 5.;
    pub const PT_LBL_SIZE: f32 = 20.;
    pub const MIN_DIM: f64 = 1.;
    pub const SCROLL_INCREMENT: f64 = 100.;
    pub const ZOOM_FACTOR: f64 = 1.2;
    pub const ZOOM_RANGE: RangeInclusive<f64> = 0.001..=100000.;
    pub const BG_COLOR: u32 = 0x282B30FF;
    pub const FG_COLOR: u32 = 0xFFFFFFFF;
    pub const INACTIVE_COLOR: u32 = 0x82909880;
    /// Maximum distance, in screen-space pixels, to consider an element hovered by the cursor
    pub const HOVER_MAX_DIST: f64 = 15.;
}

mod app;
mod geom;
mod voronoi;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
const TGT_WASM: bool = false;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
const TGT_WASM: bool = true;
#[allow(dead_code)]
const CRATE_NAME: &str = env!("CARGO_CRATE_NAME");

pub static DEBUG_FIX_DISPLAY_SIZE: LazyLock<bool, fn() -> bool> = LazyLock::new(|| {
    env::var_os("DEBUG_FIX_DISPLAY_SIZE")
        .map(|var| var.to_string_lossy().to_string())
        .map(|var| &var == "true" || &var == "1")
        .unwrap_or(false)
});

#[notan_main]
fn main() -> Result<(), String> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let norender = env::var_os("DEBUG_NORENDER")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let win = WindowConfig::new()
        .set_size(1000, 800)
        .set_resizable(true)
        .set_vsync(true)
        .set_multisampling(4)
        .set_lazy_loop(true)
        .set_high_dpi(true)
        .set_visible(!norender);

    let app = notan::init_with(setup)
        .add_config(win)
        .add_config(DrawConfig);
    if !norender {
        app.draw(|gfx: &mut Graphics, state: &mut State| state.renderer(gfx))
            .event(|app: &mut App, state: &mut State, event: Event| state.listener(app, event))
            .build()?;
    } else {
        app.event(|_app: &mut App, _: Event| exit(0)).build()?;
    }
    Ok(())
}

fn create_voronoi() -> Result<Voronoi, Error> {
    let points = vec![
        // // Equilateral triangle
        // WeightedPoint::new("A", Vec2::new(0., 1.), 1.),
        // WeightedPoint::new("B", Vec2::new(f64::sqrt(2.), -f64::sqrt(2.)), 1.),
        // WeightedPoint::new("C", Vec2::new(-f64::sqrt(2.), -f64::sqrt(2.)), 1.),

        // WeightedPoint::new("A", Vec2::new(0., 0.), 1.),
        // WeightedPoint::new("B", Vec2::new(0., 1.), 2.),
        // WeightedPoint::new("C", Vec2::new(1., 1.), 1.),
        // WeightedPoint::new("D", Vec2::new(1., 0.), 2.),
        WeightedPoint::new("A", DVec2::new(0., 0.), 100),
        WeightedPoint::new("B", DVec2::new(1., 0.), 150),
        WeightedPoint::new("C", DVec2::new(2., 0.), 200),
    ];
    Voronoi::new(points)
}

fn setup(_app: &mut App, gfx: &mut Graphics) -> State {
    let font = gfx
        .create_font(include_bytes!("../assets/Stardate81316-aolE.ttf"))
        .unwrap();
    State::new(create_voronoi().unwrap(), font)
}
//
// fn parse_log_level(var: Option<String>) -> Result<LogConfig, Error> {
//     let mut set = vec![];
//     let mut config = LogConfig::debug().use_colors(!TGT_WASM);
//     if let Some(var) = var {
//         for part in var.split(',') {
//             let key = if let Some((key, value)) = part.split_once('=') {
//                 let key = key.trim();
//                 let value = value.trim();
//
//                 if key == "deps" {
//                     config = config.verbose(value.parse()?);
//                 } else {
//                     config = config.level_for(
//                         &[CRATE_NAME, "::", key].concat(),
//                         LevelFilter::from_str(value)?,
//                     );
//                 }
//                 key
//             } else {
//                 config = config.level(LevelFilter::from_str(part)?);
//                 CRATE_NAME
//             }
//             .to_string();
//             if set.contains(&key) {
//                 panic!("Debug level set multiple times for key {key}");
//             }
//             set.push(key)
//         }
//     }
//     Ok(config)
// }
