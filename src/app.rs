use crate::geom::WeightedPoint;
use crate::log::{debug, error, info, trace};
use crate::voronoi::{Hovered, Voronoi};
use crate::*;
use notan::math::{Affine2, DVec2, UVec2};
use notan::random::rand::distributions::uniform::SampleRange;
use notan::random::rand::distributions::WeightedIndex;
use notan::random::rand::prelude::*;
use std::collections::HashSet;
use std::iter::zip;
use std::ops::{Range, RangeInclusive};

#[derive(AppState)]
pub struct State {
    voronoi: Voronoi,
    view: ViewPort,
    mouse_drag: Drag,
    mouse_pos: Option<DVec2>,
    mod_keys: HashSet<Modifier>,
    font: Font,
    render_size: UVec2,
    rendered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Drag {
    None,
    Screen(DVec2),
    Point { id: usize, delta: DVec2 },
}

impl State {
    pub fn new(voronoi: Voronoi, font: Font) -> Self {
        State {
            voronoi,
            view: ViewPort::default(),
            font,
            mouse_pos: None,
            mouse_drag: Drag::None,
            mod_keys: HashSet::new(),
            render_size: (0, 0).into(),
            rendered: false,
        }
    }

    pub fn renderer(&mut self, gfx: &mut Graphics) {
        trace!("State::renderer");

        if !self.rendered {
            self.render_size = gfx.size().into();
            self.view = ViewPort::enclosing(
                self.voronoi.points().iter().map(|(_, v)| v),
                self.render_size.as_dvec2(),
                2.,
            );
        } else {
            let size: UVec2 = gfx.size().into();
            if size != self.render_size {
                let prev = self.render_size.as_dvec2();
                let new = size.as_dvec2();
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
        match event {
            Event::MouseDown {
                button: MouseButton::Left,
                x,
                y,
            } if self.mouse_drag == Drag::None => {
                let pos = self.view.to_math(DVec2::new(x as f64, y as f64));
                if self.mod_keys.contains(&Modifier::Control) {
                    if let Hovered::Point(i) = self.voronoi.hovered() {
                        let p = self.voronoi.points().get(&i).unwrap();
                        self.mouse_drag = Drag::Point {
                            id: i,
                            delta: pos - p.pos,
                        };
                    }
                } else {
                    self.mouse_drag = Drag::Screen(pos)
                }
            }
            Event::MouseUp { .. } if self.mouse_drag != Drag::None => {
                self.mouse_drag = Drag::None;
            }
            Event::MouseEnter { x, y } | Event::MouseMove { x, y } => {
                let mouse_pos = DVec2::new(x as f64, y as f64);
                self.mouse_pos = Some(mouse_pos);
                match self.mouse_drag {
                    Drag::Screen(pos) => {
                        let o = self.view.to_screen(pos);
                        self.view.move_screen(mouse_pos - o);
                    }
                    Drag::Point { id, delta } => {
                        let pos = self.view.to_math(mouse_pos) + delta;
                        if let Err(e) = self.voronoi.modify(id, |p| p.pos = pos) {
                            error!("Couldn't move point: {e}");
                        }
                    }
                    _ => {}
                }
                self.update_hover();
                return;
            }
            Event::MouseLeft { .. } => {
                self.mouse_pos = None;
                self.update_hover();
            }
            Event::MouseWheel { delta_y, .. } => {
                if let Some(mouse_pos) = self.mouse_pos {
                    if self.mod_keys.contains(&Modifier::Control) {
                        self.change_weight(delta_y > 0.);
                    } else {
                        let scroll = (delta_y as f64 / SCROLL_INCREMENT).abs().clamp(0.25, 2.0)
                            * delta_y.signum() as f64;
                        let factor = ZOOM_FACTOR.powf(scroll);
                        self.view.zoom_around(factor, mouse_pos);
                        debug!("Zoom: {delta_y} => {scroll} => {}", factor);
                    }
                }
            }
            Event::KeyDown { key } => {
                if let Some(modif) = Modifier::from_key(key) {
                    if !self.mod_keys.insert(modif) {
                        return;
                    }
                }
            }
            Event::KeyUp { key } => match key {
                KeyCode::R => self.randomize_point(),
                KeyCode::A => self.add_point(),
                KeyCode::D => self.delete_point(),
                KeyCode::T => self.toggle_point(),
                KeyCode::PageUp => {
                    self.change_weight(true);
                }
                KeyCode::PageDown => {
                    self.change_weight(false);
                }
                KeyCode::Space => {
                    use std::fmt::Write;
                    let mut buffer = String::new();
                    write!(&mut buffer, "Matrix: {:?}\nPoints: \n", self.view).unwrap();
                    for (i, p) in self.voronoi.points() {
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
                KeyCode::Q if !TGT_WASM => {
                    app.exit();
                }
                _ => {
                    if let Some(modif) = Modifier::from_key(key) {
                        self.mod_keys.remove(&modif);
                    } else {
                        return;
                    }
                }
            },
            _ => return,
        }
        debug!("Processed event {:?}", event);
    }

    /// Returns a value big enough to ensure that any ray starting from inside the rendered area will
    /// be rendered to the edge of the screen
    pub fn render_edge_distance(&self) -> f64 {
        u32::max(self.render_size.x, self.render_size.y) as f64 * 2. * self.view.to_math_ratio()
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

    pub fn render_size(&self) -> UVec2 {
        self.render_size
    }

    pub fn is_within_render(&self, p: DVec2) -> bool {
        (0. ..self.render_size.x as f64).contains(&p.x)
            && (0. ..self.render_size.y as f64).contains(&p.y)
    }

    fn update_hover(&mut self) {
        let prev = self.voronoi.hovered();
        let new = self.voronoi.hover(self.mouse_pos.map(|mouse_pos| {
            (
                self.view.to_math(mouse_pos),
                HOVER_MAX_DIST * self.view.to_math_ratio(),
            )
        }));
        if prev != new {
            debug!("Hovered: {prev:?} to {new:?}");
        }
    }

    fn randomize_point(&mut self) {
        info!("Generating random diagram");
        const RANDOM_COUNT: RangeInclusive<u8> = 3..=4;
        const RANDOM_SPACE: Range<i32> = -10..10;
        const RANDOM_SPACE_FACTOR: f64 = 10.;
        const RANDOM_WEIGHT: [u32; 9] = [100, 50, 70, 90, 110, 120, 150, 200, 300];
        const RANDOM_WEIGHT_WEIGHTS: [u32; 9] = [10, 1, 2, 2, 2, 2, 1, 1, 1];
        const NAMES: RangeInclusive<char> = 'A'..='Z';
        let rng = &mut thread_rng();
        let weight_index = WeightedIndex::new(RANDOM_WEIGHT_WEIGHTS).unwrap();

        self.voronoi.clear();
        for (_, name) in zip(0..RANDOM_COUNT.sample_single(rng), NAMES) {
            loop {
                let p = WeightedPoint::new(
                    name.to_string(),
                    DVec2::new(
                        RANDOM_SPACE.sample_single(rng) as f64 / RANDOM_SPACE_FACTOR,
                        RANDOM_SPACE.sample_single(rng) as f64 / RANDOM_SPACE_FACTOR,
                    ),
                    RANDOM_WEIGHT[weight_index.sample(rng)],
                );
                if self.voronoi.add(p).is_ok() {
                    break;
                }
            }
        }
        self.view = ViewPort::enclosing(
            self.voronoi.points().iter().map(|(_, v)| v),
            self.render_size.as_dvec2(),
            2.,
        );
        self.update_hover();
    }

    fn add_point(&mut self) {
        if let Some(pos) = self.mouse_pos {
            let name = ('A'..='Z')
                .chain('α'..='ω')
                .find(|n| {
                    !self
                        .voronoi
                        .points()
                        .iter()
                        .any(|(_, v)| v.name.as_bytes() == [*n as u8])
                })
                .unwrap()
                .to_string();
            self.voronoi
                .add(WeightedPoint::new(name, self.view.to_math(pos), 100))
                .unwrap()
        }
        self.update_hover();
    }

    fn delete_point(&mut self) {
        if let Hovered::Point(i) = self.voronoi.hovered() {
            if let Err(e) = self.voronoi.remove(i) {
                error!("{:#}", e);
            }
        }
        self.update_hover();
    }

    fn toggle_point(&mut self) {
        if let Hovered::Point(i) = self.voronoi.hovered() {
            if let Err(e) = self.voronoi.modify(i, |p| p.enabled ^= true) {
                error!("{e:#}");
            }
        }
        self.update_hover();
    }

    fn change_weight(&mut self, incr: bool) {
        let hovered = self.voronoi.hovered();
        debug!("Changing weight of: {hovered:?}");
        if let Hovered::Point(i) = hovered {
            const SCROLL_WEIGHT_DIFF: i32 = 5;
            const SCROLL_WEIGHT_RANGE: Range<i32> = 10..500;
            let diff = if incr { 1 } else { -1 } * SCROLL_WEIGHT_DIFF;
            if let Err(e) = self.voronoi.modify(i, |p| {
                p.weight = (p.weight as i32 + diff)
                    .clamp(SCROLL_WEIGHT_RANGE.start, SCROLL_WEIGHT_RANGE.end)
                    as u32
            }) {
                error!("Couldn't adjust point weight: {e:#}");
            }
            self.update_hover();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Modifier {
    Shift,
    Control,
    Alt,
}

impl Modifier {
    fn from_key(key: KeyCode) -> Option<Self> {
        Some(match key {
            KeyCode::LShift | KeyCode::RShift => Modifier::Shift,
            KeyCode::LControl | KeyCode::RControl => Modifier::Control,
            KeyCode::LAlt | KeyCode::RAlt => Modifier::Alt,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ViewPort {
    r: f64,
    off: DVec2,
}

#[allow(dead_code)]
impl ViewPort {
    pub fn new(r: f64, off: DVec2) -> ViewPort {
        Self { r, off }
    }

    pub fn enclosing<'a>(
        pts: impl IntoIterator<Item = &'a WeightedPoint>,
        display: DVec2,
        margin: f64,
    ) -> Self {
        let (min, max) = pts.into_iter().map(|p| p.pos).fold(
            (DVec2::NAN, DVec2::NAN),
            |acc: (DVec2, DVec2), v: DVec2| {
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
        let mut ret = Self {
            r,
            off: DVec2::ZERO,
        };
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

    pub fn zoom(&mut self, factor: f64) {
        assert!(factor.abs() > f64::EPSILON);
        let old = self.r;
        self.r = (self.r * factor).clamp(*ZOOM_RANGE.start(), *ZOOM_RANGE.end());
        debug!("zoom: {old} * {factor} = {:?}", self.r);
    }

    pub fn zoomed(mut self, factor: f64) -> Self {
        self.zoom(factor);
        self
    }

    pub fn zoom_around(&mut self, factor: f64, center: DVec2) {
        let center_prj = self.to_math(center);
        self.zoom(factor);
        let new_center = self.to_screen(center_prj);
        self.move_screen(center - new_center);
    }

    /// Returns the scaling ratio from math to screen
    pub fn to_screen_ratio(&self) -> f64 {
        self.r
    }

    /// Projects a point from the mathematical space to the screen
    pub fn to_screen(&self, p: impl Into<DVec2>) -> DVec2 {
        (p.into() * DVec2::new(self.r, -self.r)) + self.off
    }

    /// Returns the scalar ratio from screen to math
    pub fn to_math_ratio(&self) -> f64 {
        1. / self.r
    }

    /// Projects a point from the screen to the mathematical space
    pub fn to_math(&self, v: DVec2) -> DVec2 {
        let v1 = v - self.off;
        v1 / DVec2::new(self.r, -self.r)
    }

    /// Moves the viewport by the given delta in screen space
    pub fn move_screen(&mut self, delta: DVec2) {
        let old = self.off;
        self.off += delta;
        debug!("move_screen: {old} + {delta} = {:?}", self.off);
    }

    /// Moves the viewport by the given delta in mathematical space
    pub fn move_math(&mut self, delta: DVec2) {
        self.move_screen(delta * self.r);
    }
}

impl Default for ViewPort {
    fn default() -> Self {
        Self {
            r: 100.,
            off: DVec2::new(0., 0.),
        }
    }
}
