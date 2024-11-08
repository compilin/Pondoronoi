use crate::geom::WeightedPoint;
use std::env;
use std::str::FromStr;

pub use notan::draw::*;
pub use notan::log;
use notan::math::Vec2;
pub use notan::prelude::*;
use voronoi::Voronoi;

mod geom;
mod voronoi;

#[derive(AppState)]
struct State {
    voronoi: Voronoi,
    view: ViewPort,
    mouse_drag: Option<Vec2>,
    mouse_pos: Vec2,
    font: Font,
    display_size: (u32, u32),
    rendered: bool,
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
const TGT_WASM: bool = false;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
const TGT_WASM: bool = true;

#[notan_main]
fn main() -> Result<(), String> {
    let win = WindowConfig::new()
        .set_size(1000, 800)
        .set_vsync(true)
        .set_lazy_loop(true)
        .set_high_dpi(true);

    let log_level = env::var_os("LOG_LEVEL")
        .map(|ll| log::LevelFilter::from_str(&ll.into_string().unwrap()))
        .transpose()
        .map_err(|err| err.to_string())?
        .unwrap_or(log::LevelFilter::Debug);

    notan::init_with(setup)
        .add_config(win)
        .add_config(DrawConfig)
        .add_config(log::LogConfig::new(log_level).use_colors(!TGT_WASM))
        .draw(|gfx: &mut Graphics, state: &mut State| state.renderer(gfx))
        .event(|app: &mut App, state: &mut State, event: Event| state.listener(app, event))
        .build()
}

fn setup(_app: &mut App, gfx: &mut Graphics) -> State {
    let font = gfx
        .create_font(include_bytes!("../assets/Stardate81316-aolE.ttf"))
        .unwrap();
    let points = vec![
        WeightedPoint::new("A", Vec2::new(0., 0.), 1.),
        WeightedPoint::new("B", Vec2::new(1., 0.), 2.),
        WeightedPoint::new("C", Vec2::new(1., 1.), 3.),
        WeightedPoint::new("D", Vec2::new(0., 1.), 1.),
    ];
    State {
        voronoi: Voronoi::new(points),
        view: ViewPort::default(),
        font,
        mouse_pos: Vec2::ZERO,
        mouse_drag: None,
        display_size: (0, 0),
        rendered: false,
    }
}

const PT_LBL_OFFSET: f32 = 5.;
const PT_LBL_SIZE: f32 = 20.;
const MIN_DIM: f32 = 1.;
const SCROLL_INCREMENT: f32 = 100.;
const ZOOM_FACTOR: f32 = 1.2;
const BG_COLOR: u32 = 0x282B30FF;
const FG_COLOR: u32 = 0xFFFFFFFF;
const INACTIVE_TEXT_COLOR: u32 = 0x829098FF;
const INACTIVE_COLOR: u32 = 0x42454980;
/// Maximum screen distance, in pixels, to consider an element hovered by the cursor
const HOVER_MAX_DIST: f32 = 15.;

impl State {
    pub fn renderer(&mut self, gfx: &mut Graphics) {
        log::trace!("State::renderer");

        if !self.rendered {
            let (width, height) = gfx.size();
            self.view = ViewPort::enclosing(
                self.voronoi.vertices(),
                Vec2::new(width as f32, height as f32),
                2.,
            );
        }

        let mut draw = gfx.create_draw();
        let bg: Color = Color::from_hex(BG_COLOR);
        draw.clear(bg);

        Voronoi::render(self, gfx, &mut draw);

        gfx.render(&draw);
        self.rendered = true;
    }

    pub fn listener(&mut self, app: &mut App, event: Event) {
        log::trace!("Event: {event:?}");
        match &event {
            Event::MouseDown {
                button: MouseButton::Left,
                x,
                y,
            } if self.mouse_drag.is_none() => {
                self.mouse_drag = Some(self.view.to_math(Vec2::new(*x as f32, *y as f32)))
            }
            Event::MouseUp { .. } if self.mouse_drag.is_some() => {
                self.mouse_drag = None;
            }
            Event::MouseEnter { x, y } | Event::MouseMove { x, y } => {
                self.mouse_pos = Vec2::new(*x as f32, *y as f32);
                if let Some(pos) = self.mouse_drag {
                    let o = self.view.to_screen(pos);
                    self.view.move_screen(self.mouse_pos - o);
                }
                self.voronoi.hover(Some((
                    self.view.to_math(self.mouse_pos),
                    HOVER_MAX_DIST * self.view.to_math_ratio(),
                )));
                return;
            }
            Event::MouseLeft { .. } => {
                self.voronoi.hover(None);
            }
            Event::MouseWheel { delta_y, .. } => {
                let scroll = (delta_y / SCROLL_INCREMENT).abs().clamp(0.25, 2.0) * delta_y.signum();
                let factor = ZOOM_FACTOR.powf(scroll);
                self.view.zoom_around(factor, self.mouse_pos);
                log::debug!("Zoom: {delta_y} => {scroll} => {}", factor);
            }
            Event::WindowResize { width, height } => {
                if self.rendered {
                    self.view.move_screen(Vec2::new(
                        ((self.display_size.0 - width) / 2) as f32,
                        ((self.display_size.1 - height) / 2) as f32,
                    ));
                }
                self.display_size = (*width, *height);
            }
            Event::KeyUp { key } => match key {
                KeyCode::Space => {
                    use std::fmt::Write;
                    let mut buffer = String::new();
                    write!(&mut buffer, "Matrix: {:?}\nPoints: \n", self.view).unwrap();
                    for p in self.voronoi.vertices() {
                        writeln!(
                            &mut buffer,
                            "\t - {:?} => {:?}",
                            p,
                            self.view.to_screen(p.pos)
                        )
                        .unwrap();
                    }
                    print!("{buffer}");
                }
                KeyCode::Q => {
                    app.exit();
                }
                _ => return,
            },
            _ => return,
        }
        log::debug!("Processed event {:?}", event);
    }
}

#[derive(Debug, Clone)]
struct ViewPort {
    r: f32,
    off: Vec2,
}

#[allow(dead_code)]
impl ViewPort {
    pub fn new(r: f32, off: Vec2) -> ViewPort {
        Self { r, off }
    }

    pub fn enclosing<'a>(
        pts: impl IntoIterator<Item = &'a WeightedPoint>,
        display: Vec2,
        margin: f32,
    ) -> Self {
        let (min, max) = pts.into_iter().map(|p| p.pos).fold(
            (Vec2::NAN, Vec2::NAN),
            |acc: (Vec2, Vec2), v: Vec2| {
                if acc.0.x.is_finite() {
                    (acc.0.min(v), acc.1.max(v))
                } else {
                    (v, v)
                }
            },
        );

        let r = (display.x / (max.x - min.x).max(MIN_DIM))
            .min(display.y / (max.y - min.y).max(MIN_DIM))
            / margin;
        let mut ret = Self { r, off: Vec2::ZERO };
        let mid = ret.to_screen(((max.x - min.x) / 2., (max.y - min.y) / 2.));
        let viewmid = display / 2.;
        ret.move_screen(viewmid - mid);
        log::debug!(
            "Enclosing ({}..{}/{}..{}) into {display:?}: r={r},\n\
            \t{mid:?} -> {viewmid:?} => {:?}\n\tmatrix: {ret:?}",
            min.x,
            max.x,
            min.y,
            max.y,
            ret.to_screen(mid)
        );
        ret
    }

    pub fn zoom(&mut self, factor: f32) {
        self.r *= factor;
    }

    pub fn zoomed(mut self, factor: f32) -> Self {
        self.zoom(factor);
        self
    }

    pub fn zoom_around(&mut self, factor: f32, center: Vec2) {
        let center_prj = self.to_math(center);
        self.zoom(factor);
        let new_center = self.to_screen(center_prj);
        self.move_screen(center - new_center);
    }

    /// Returns the scaling ratio from math to screen
    pub fn to_screen_ratio(&self) -> f32 {
        self.r
    }

    /// Projects a point from the mathematical space to the screen
    pub fn to_screen(&self, p: impl Into<Vec2>) -> Vec2 {
        (p.into() * Vec2::new(self.r, -self.r)) + self.off
    }

    /// Returns the scalar ratio from screen to math
    pub fn to_math_ratio(&self) -> f32 {
        1. / self.r
    }

    /// Projects a point from the screen to the mathematical space
    pub fn to_math(&self, v: Vec2) -> Vec2 {
        let v1 = v - self.off;
        v1 / Vec2::new(self.r, -self.r)
    }

    /// Moves the viewport by the given delta in screen space
    pub fn move_screen(&mut self, delta: Vec2) {
        let old = self.off;
        self.off += delta;
        log::trace!("move_screen: {old} + {delta} = {:?}", self.off);
    }

    /// Moves the viewport by the given delta in mathematical space
    pub fn move_math(&mut self, delta: Vec2) {
        self.move_screen(delta * self.r);
    }
}

impl Default for ViewPort {
    fn default() -> Self {
        Self {
            r: 100.,
            off: Vec2::new(0., 0.),
        }
    }
}
