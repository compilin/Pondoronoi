use crate::geom::WeightedPoint;
use crate::log::{debug, trace};
use crate::voronoi::Voronoi;
use crate::{
    App, AppState, Color, CreateDraw, Event, Font, Graphics, KeyCode, MouseButton, BG_COLOR,
    DEBUG_FIX_DISPLAY_SIZE, HOVER_MAX_DIST, MIN_DIM, SCROLL_INCREMENT, ZOOM_FACTOR, ZOOM_RANGE,
};
use notan::math::{Affine2, UVec2, Vec2};

#[derive(AppState)]
pub struct State {
    voronoi: Voronoi,
    view: ViewPort,
    mouse_drag: Option<Vec2>,
    mouse_pos: Vec2,
    font: Font,
    render_size: UVec2,
    rendered: bool,
}

impl State {
    pub fn new(voronoi: Voronoi, font: Font) -> Self {
        State {
            voronoi,
            view: ViewPort::default(),
            font,
            mouse_pos: Vec2::ZERO,
            mouse_drag: None,
            render_size: (0, 0).into(),
            rendered: false,
        }
    }

    pub fn renderer(&mut self, gfx: &mut Graphics) {
        trace!("State::renderer");

        if !self.rendered {
            self.render_size = gfx.size().into();
            self.view = ViewPort::enclosing(
                self.voronoi.vertices().iter().map(|(_, v)| v),
                self.render_size.as_vec2(),
                2.,
            );
        } else {
            let size: UVec2 = gfx.size().into();
            if size != self.render_size {
                let prev = self.render_size.as_vec2();
                let new = size.as_vec2();
                let delta = (new - prev) / 2.;
                self.view.move_screen(delta);
                let zoom = new.length() / prev.length();
                self.view.zoom_around(zoom, new / 2.);
                debug!("Resizing: {prev} to {new}. Moving viewport by {delta}, zooming by {zoom} around {}",
                        new / 2.);
                self.render_size = size;
            }
        }

        let mut draw = gfx.create_draw();
        let bg: Color = Color::from_hex(BG_COLOR);
        draw.clear(bg);
        if *DEBUG_FIX_DISPLAY_SIZE {
            let offset = (UVec2::from(gfx.size()) - self.render_size).as_vec2();
            draw.transform()
                .push(Affine2::from_translation(offset / 2.).into());
        }

        Voronoi::render(self, gfx, &mut draw);

        gfx.render(&draw);
        self.rendered = true;
    }

    pub fn listener(&mut self, app: &mut App, event: Event) {
        trace!("Event: {event:?}");
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
                debug!("Zoom: {delta_y} => {scroll} => {}", factor);
            }
            Event::KeyUp { key } => match key {
                KeyCode::Space => {
                    use std::fmt::Write;
                    let mut buffer = String::new();
                    write!(&mut buffer, "Matrix: {:?}\nPoints: \n", self.view).unwrap();
                    for (i, p) in self.voronoi.vertices() {
                        writeln!(
                            &mut buffer,
                            "\t - {i: >3}: {:?} => {:?}",
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
        debug!("Processed event {:?}", event);
    }

    /// Returns a value big enough to ensure that any ray starting from inside the rendered area will
    /// be rendered to the edge of the screen
    pub fn render_edge_distance(&self) -> f32 {
        u32::max(self.render_size.x, self.render_size.y) as f32 * 2. * self.view.to_math_ratio()
    }

    pub fn view(&self) -> &ViewPort {
        &self.view
    }

    pub fn voronoi(&self) -> &Voronoi {
        &self.voronoi
    }

    pub fn font(&self) -> Font {
        self.font
    }
}

#[derive(Debug, Clone)]
pub struct ViewPort {
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
        debug!(
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
        assert!(factor.abs() > f32::EPSILON);
        let old = self.r;
        self.r = (self.r * factor).clamp(*ZOOM_RANGE.start(), *ZOOM_RANGE.end());
        debug!("zoom: {old} * {factor} = {:?}", self.r);
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
        debug!("move_screen: {old} + {delta} = {:?}", self.off);
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
