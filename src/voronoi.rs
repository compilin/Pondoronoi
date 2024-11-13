use crate::geom::*;
use crate::*;
use std::cmp::Ordering;
use std::collections::HashMap;

pub type EdgeMap = HashMap<(usize, usize), BisectorSegments>;

pub struct Voronoi {
    points: HashMap<usize, WeightedPoint>,
    edges: EdgeMap,
    next_id: usize,
    hovered: Hovered,
}

impl Voronoi {
    pub fn new(pts: impl Into<Vec<WeightedPoint>>) -> Result<Self, Error> {
        let mut v = Self {
            points: Default::default(),
            edges: Default::default(),
            next_id: 0,
            hovered: Hovered::None,
        };

        for pt in pts.into() {
            v.add(pt)?;
        }

        Ok(v)
    }

    pub fn add(&mut self, point: WeightedPoint) -> Result<(), Error> {
        debug!("Adding point {point:?}");
        if let Some((_, p)) = self
            .points
            .iter()
            .find(|(_, p)| p.name == point.name || p.pos.distance(point.pos) < f32::EPSILON)
        {
            return if p.name == point.name {
                Err(Error::msg("A point with that name already exists"))
            } else {
                Err(Error::msg("A point at this position already exists"))
            };
        }
        let id = self.next_id;
        match self.points.len() {
            0 => {}
            1 => {
                let (i, pt) = self.points.iter().next().unwrap();
                self.edges.insert(
                    Self::edge_id(id, *i),
                    BisectorSegments::new(pt.bisect(&point)),
                );
            }
            _ => {
                let mut edges = HashMap::new();
                for ((i1, i2), e12) in &mut self.edges {
                    let get = |i| {
                        let p = self.points.get(i).unwrap();
                        let e = edges.get(i).cloned().unwrap_or_else(|| {
                            let bisect = BisectorSegments::new(point.bisect(p));
                            trace!(
                                "Calculating bisector of {}{}: {:?}",
                                point.name,
                                p.name,
                                bisect.base()
                            );
                            bisect
                        });
                        (p, e)
                    };
                    let (p1, mut e1) = get(i1);
                    let (p2, mut e2) = get(i2);
                    BisectorSegments::intersect(&point, p1, p2, &mut e1, &mut e2);
                    BisectorSegments::intersect(p1, &point, p2, &mut e1, e12);
                    // BisectorSegments::intersect(&p2, &point, &p1, &mut e2, e12);
                    edges.insert(*i1, e1);
                    edges.insert(*i2, e2);
                }
                self.edges
                    .extend(edges.into_iter().map(|(i, e)| (Self::edge_id(id, i), e)));
            }
        }
        self.points.insert(id, point);

        self.next_id += 1;
        while self.points.contains_key(&self.next_id) {
            self.next_id += 1;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn remove(&mut self, id: usize) {
        self.points.remove(&id);
        self.edges.retain(|(i1, i2), _| i1 != &id && i2 != &id)
    }

    pub fn render(state: &State, gfx: &mut Graphics, draw: &mut Draw) {
        let fg = Color::from_hex(FG_COLOR);
        let inact = Color::from_hex(INACTIVE_COLOR);

        for (i, p) in state.voronoi().vertices().iter() {
            let Vec2 { x, y } = state.view().to_screen(p.pos);
            let (starcol, textcol) = if (i, p).is_active(&state.voronoi().hovered) {
                (fg, fg)
            } else {
                (inact, inact.with_alpha(1.))
            };
            draw.star(4, 5., 2.5).position(x, y).color(starcol);
            draw.text(&state.font(), &p.label)
                .position(x + PT_LBL_OFFSET, y + PT_LBL_OFFSET)
                .size(PT_LBL_SIZE)
                .color(textcol)
                .h_align_left()
                .v_align_top();
        }

        for (ids, bs) in state.voronoi().edges() {
            let col = if (ids, bs).is_active(&state.voronoi().hovered) {
                fg
            } else {
                inact
            };

            bs.render(state, gfx, draw, col);
        }
    }

    pub fn vertices(&self) -> &HashMap<usize, WeightedPoint> {
        &self.points
    }

    pub fn edges(&self) -> &EdgeMap {
        &self.edges
    }

    pub fn hover(&mut self, pos: Option<(Vec2, f32)>) {
        fn get_hovers<H: Hoverable>(
            it: impl IntoIterator<Item = H>,
            pos: Vec2,
        ) -> impl Iterator<Item = (Hovered, u16, f32)> {
            it.into_iter()
                .map(move |pt| (pt.hover(), H::priority(), pt.distance(pos)))
        }
        let prev = self.hovered;
        self.hovered = pos
            .and_then(|(pos, max_dist)| {
                get_hovers(&self.points, pos)
                    .chain(get_hovers(&self.edges, pos))
                    .filter(|m| m.2 <= max_dist)
                    .min_by(|va, vb| u16::cmp(&va.1, &vb.1).then(f32::total_cmp(&va.2, &vb.2)))
            })
            .map(|m| m.0)
            .unwrap_or(Hovered::None);
        if self.hovered != prev {
            debug!("Hovered: {:?} to {:?}", prev, self.hovered);
        }
    }

    fn edge_id(i1: usize, i2: usize) -> (usize, usize) {
        match i1.cmp(&i2) {
            Ordering::Less => (i1, i2),
            Ordering::Equal => panic!(),
            Ordering::Greater => (i2, i1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hovered {
    None,
    Point(usize),
    Edge(usize, usize),
}
