use crate::geom::{BisectMap, Bisector, WeightedPoint};
use crate::*;
use std::collections::HashMap;

pub struct Voronoi {
    vertices: Vec<WeightedPoint>,
    edges: BisectMap,
    hovered: Option<usize>,
}

impl Voronoi {
    pub(crate) fn new(pts: impl Into<Vec<WeightedPoint>>) -> Self {
        let vertices = pts.into();
        Self {
            edges: Self::bisect_all(&vertices),
            vertices,
            hovered: None,
        }
    }

    fn bisect_all<'a>(pts: &Vec<WeightedPoint>) -> BisectMap {
        let mut map = HashMap::new();
        let iter = pts.iter().enumerate();
        for (i1, p1) in iter.clone() {
            for (i2, p2) in iter.clone().skip(i1 + 1) {
                map.insert((i1, i2), p1.bisect(p2));
            }
        }

        map
    }

    pub(crate) fn render(state: &mut State, gfx: &mut Graphics, draw: &mut Draw) {
        let fg = Color::from_hex(FG_COLOR);
        let inact = Color::from_hex(INACTIVE_COLOR);
        let inact_txt = Color::from_hex(INACTIVE_TEXT_COLOR);

        for (i, p) in state.voronoi.vertices().iter().enumerate() {
            let Vec2 { x, y } = state.view.to_screen(p.pos);
            let (starcol, textcol) = if state.voronoi.hovered.map(|h| h == i).unwrap_or(true) {
                (fg, fg)
            } else {
                (inact, inact_txt)
            };
            draw.star(4, 5., 2.5).position(x, y).color(starcol);
            draw.text(&state.font, &p.label)
                .position(x + PT_LBL_OFFSET, y + PT_LBL_OFFSET)
                .size(PT_LBL_SIZE)
                .color(textcol)
                .h_align_left()
                .v_align_top();
        }

        let linemul = gfx.size().0.max(gfx.size().1) as f32 / state.view.r;
        for (ids, b) in state.voronoi.edges() {
            let col = if state
                .voronoi
                .hovered
                .map(|h| h == ids.0 || h == ids.1)
                .unwrap_or(true)
            {
                fg
            } else {
                inact
            };
            match b {
                Bisector::Line { orig, vec } => {
                    let p1 = state.view.to_screen(*orig + *vec * linemul);
                    let p2 = state.view.to_screen(*orig - *vec * linemul);
                    draw.line((p1.x, p1.y), (p2.x, p2.y)).color(col);
                }
                Bisector::Circle { center, radius } => {
                    let center = state.view.to_screen(*center);
                    draw.circle(*radius * state.view.r)
                        .position(center.x, center.y)
                        .stroke(1.)
                        .stroke_color(col);
                }
            }
        }
    }

    pub fn vertices(&self) -> &Vec<WeightedPoint> {
        &self.vertices
    }

    pub fn edges(&self) -> &BisectMap {
        &self.edges
    }

    pub fn hover(&mut self, pos: Option<(Vec2, f32)>) {
        self.hovered = pos
            .and_then(|(pos, max_dist)| {
                self.vertices
                    .iter()
                    .map(|v| v.pos.distance(pos))
                    .enumerate()
                    .min_by(|va, vb| f32::total_cmp(&va.1, &vb.1))
                    .take_if(|m| m.1 <= max_dist)
            })
            .map(|m| m.0);
    }
}
