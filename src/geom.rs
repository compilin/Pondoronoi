use crate::app::State;
use crate::voronoi::Hovered;
use crate::{Color, Draw, Graphics};
use anyhow::Error;
use notan::draw::DrawShapes;
use notan::log::*;
use notan::math::Vec2;
use std::f32::consts::{PI, TAU};
use std::ops::{DerefMut, Mul};

pub trait Hoverable {
    fn distance(&self, point: Vec2) -> f32;
    fn is_hovered(&self, hov: &Hovered) -> bool;

    fn is_active(&self, hov: &Hovered) -> bool {
        hov == &Hovered::None || self.is_hovered(hov)
    }

    fn hover(&self) -> Hovered;

    fn priority() -> u16;
}

pub trait Intersectable {
    type Point;
    fn intersect(
        p1: &Self::Point,
        p2: &Self::Point,
        p3: &Self::Point,
        edge1_2: impl DerefMut<Target = Self>,
        edge1_3: impl DerefMut<Target = Self>,
    );
}

#[derive(Debug, Clone)]
pub struct WeightedPoint {
    pub name: String,
    pub label: String,
    pub pos: Vec2,
    pub weight: f32,
}

impl WeightedPoint {
    pub fn new(name: impl ToString, p: Vec2, weight: f32) -> WeightedPoint {
        let name = name.to_string();
        Self {
            label: format!("{name}, {weight}"),
            name,
            pos: p,
            weight,
        }
    }

    pub fn bisect(&self, other: &Self) -> Bisector {
        let a = self.weight;
        let b = other.weight;
        if a == b {
            Bisector::Line {
                orig: (self.pos + other.pos) / 2.,
                vec: (-self.pos + other.pos).perp().normalize(),
            }
        } else {
            let a2 = a.powi(2);
            let b2 = b.powi(2);

            Bisector::Circle {
                center: self.pos + (other.pos - self.pos) * -a2 / (b2 - a2),
                radius: self.pos.distance(other.pos) * (a * b) / (b2 - a2).abs(),
            }
        }
    }
}

impl Hoverable for (&usize, &WeightedPoint) {
    fn distance(&self, point: Vec2) -> f32 {
        self.1.pos.distance(point)
    }

    fn is_hovered(&self, hov: &Hovered) -> bool {
        match hov {
            Hovered::None => false,
            Hovered::Point(i) => i == self.0,
            Hovered::Edge(i1, i2) => i1 == self.0 || i2 == self.0,
        }
    }

    fn hover(&self) -> Hovered {
        Hovered::Point(*self.0)
    }

    fn priority() -> u16 {
        1
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Bisector {
    Line { orig: Vec2, vec: Vec2 },
    Circle { center: Vec2, radius: f32 },
}

impl Mul<f32> for Bisector {
    type Output = Vec2;

    /// Allows multiplying a bisector by a float to get a point corresponding to `rhs` as a multiplier if
    /// `self` is a `Line`, or an angle if `self` is a `Circle`
    fn mul(self, rhs: f32) -> Self::Output {
        match self {
            Bisector::Line { orig, vec } => orig + vec * rhs,
            Bisector::Circle { center, radius } => center + Vec2::from_angle(rhs) * radius,
        }
    }
}

type Segment = (Option<f32>, Option<f32>);

// Describes a segment of a line, or an arc of a circle
#[derive(Debug, Clone)]
pub struct BisectorSegments {
    /// Vector of min-max angles of an arc, or boundary positions for a segment/ray.
    /// For circles, min and max must both be either None (for a full circle) or Some (for an arc),
    /// min is the starting angle within (0, 2pi), and max is the ending angle within (min, min+2pi)
    segments: Vec<Segment>,
    base: Bisector,
}

impl BisectorSegments {
    pub fn new(base: Bisector) -> Self {
        Self {
            segments: vec![(None, None)],
            base,
        }
    }

    pub fn render(&self, state: &State, _gfx: &mut Graphics, draw: &mut Draw, col: Color) {
        let point = |draw: &mut Draw, p: &Vec2| {
            draw.point(p.x, p.y).width(5.).color(col);
        };
        let stroke = 2.;
        let skel_col = col.with_alpha(0.15);
        let skel_stroke_width = stroke / 2.;
        match self.base {
            Bisector::Line { orig, vec } => {
                if !self.is_full() {
                    let p1 = orig + vec * -state.render_edge_distance();
                    let p2 = orig + vec * state.render_edge_distance();
                    draw.line(
                        state.view().to_screen(p1).into(),
                        state.view().to_screen(p2).into(),
                    )
                    .width(stroke)
                    .color(skel_col)
                    .width(skel_stroke_width);
                }

                for (min, max) in &self.segments {
                    let p1 = orig + vec * min.unwrap_or_else(|| -state.render_edge_distance());
                    let p1 = state.view().to_screen(p1);
                    let p2 = orig + vec * max.unwrap_or_else(|| state.render_edge_distance());
                    let p2 = state.view().to_screen(p2);

                    point(draw, &p1);
                    point(draw, &p2);
                    draw.line(p1.into(), p2.into()).width(stroke).color(col);
                }
            }
            base @ Bisector::Circle { center, radius, .. } => {
                let center = state.view().to_screen(center);
                let mut full_circle = draw.circle(radius * state.view().to_screen_ratio());
                full_circle.position(center.x, center.y);
                if self.is_full() {
                    full_circle.stroke(stroke).stroke_color(col);
                } else {
                    full_circle.stroke(skel_stroke_width).stroke_color(skel_col);
                    drop(full_circle);
                    for (min, max) in &self.segments {
                        if let (Some(min), Some(max)) = (*min, *max) {
                            // Approximate angle to increment to advance ~1px on screen
                            let pix_agl = state.view().to_math_ratio() / radius;

                            let mut agl = min;
                            let mut p = state.view().to_screen(base * agl);
                            point(draw, &p);
                            let mut path = draw.path();
                            path.stroke(stroke).color(col).move_to(p.x, p.y);
                            while max - agl > -f32::EPSILON {
                                p = state.view().to_screen(base * agl);
                                path.line_to(p.x, p.y);
                                agl += pix_agl;
                            }
                            drop(path);
                            point(draw, &p);
                        } else {
                            panic!()
                        }
                    }
                }
            }
        }
    }

    /// Returns the given arguments modulo'd so that min is within 0..2PI and max is within min..min+2PI
    fn mod_arc(min: f32, max: f32) -> (f32, f32) {
        let min = min.rem_euclid(TAU);
        (min, min + (max - min).rem_euclid(TAU))
    }

    // Removes a segment of arc from self
    fn remove(&mut self, (min, max): Segment) {
        if self.segments.is_empty() {
        } else if let (None, None) = (min, max) {
            // Removes the whole bisector
            self.segments.clear();
        } else {
            let fmin = min.unwrap_or(f32::NEG_INFINITY);
            let fmax = max.unwrap_or(f32::INFINITY);
            assert!(fmin <= fmax);
            match self.base {
                Bisector::Circle { .. } => {
                    self.check_seg_valid((min, max)).unwrap();
                    let mut i = 0;
                    while i < self.segments.len() {
                        let (smin, smax) = &mut self.segments[i];
                        // (min, max) can't be (None, None) nor half-open, so we know it's fully-closed
                        i += if smin.is_some() && smax.is_some() {
                            let (mut sfmin, mut sfmax) = (smin.unwrap(), smax.unwrap());
                            #[inline]
                            fn is_within(it: f32, (min, max): (f32, f32)) -> bool {
                                (it - min).rem_euclid(TAU) < (max - min)
                            }

                            let smin_in_rem = is_within(sfmin, (fmin, fmax));
                            let smax_in_rem = is_within(sfmax, (fmin, fmax));
                            let min_in_segment = is_within(fmin, (sfmin, sfmax));
                            println!("min: {fmin}, max: {fmax}, sfmin: {sfmin}, sfmax: {sfmax}, smin_in_rem: {smin_in_rem}, smax_in_rem: {smax_in_rem}, {min_in_segment}");
                            let i2 = if smin_in_rem {
                                if smax_in_rem {
                                    if min_in_segment {
                                        // Removal and segment are negatively included within each other:
                                        // trimming both ends
                                        (sfmin, sfmax) = Self::mod_arc(fmax, fmin);
                                        1
                                    } else {
                                        // Arc is fully within removal: removing
                                        self.segments.remove(i);
                                        0
                                    }
                                } else {
                                    // Start of arc is within removal
                                    (sfmin, sfmax) = Self::mod_arc(fmax, sfmax);
                                    1
                                }
                            } else if smax_in_rem {
                                // End of arc is within removal
                                sfmax = Self::mod_arc(sfmin, fmin).1;
                                1
                            } else if min_in_segment {
                                // Removal is fully within the arc: splitting
                                let (newmin, newmax) = Self::mod_arc(fmax, sfmax);
                                sfmax = Self::mod_arc(sfmin, fmin).1;
                                println!("seg: ({sfmin}, {sfmax}), newseg: ({newmin}, {newmax})");
                                self.segments.insert(i + 1, (Some(newmin), Some(newmax)));
                                2
                            } else {
                                // No intersection
                                1
                            };
                            debug_assert!(sfmin <= sfmax);
                            debug_assert!((sfmin, sfmax) == Self::mod_arc(sfmin, sfmax));
                            self.segments[i] = (Some(sfmin), Some(sfmax));
                            i2
                        } else if smin.is_none() && smax.is_none() {
                            // Removing an arc from a full circle: making it the inverse arc
                            let (sfmin, sfmax) = Self::mod_arc(fmax, fmin + TAU);
                            *smin = Some(sfmin);
                            *smax = Some(sfmax);
                            1
                        } else {
                            panic!("Invalid bisector: half-open arc");
                        };
                    }
                }
                Bisector::Line { .. } => {
                    let mut i = 0;
                    while i < self.segments.len() {
                        let (smin, smax) = &mut self.segments[i];
                        let sfmin = smin.unwrap_or(f32::NEG_INFINITY);
                        let sfmax = smax.unwrap_or(f32::INFINITY);

                        i += if sfmin >= fmax || sfmax <= fmin {
                            // No intersection
                            1
                        } else if sfmin < fmin {
                            if sfmax <= fmax {
                                // End of segment is within removal
                                *smax = min;
                                1
                            } else {
                                // Removal is fully within segment: splitting
                                let newseg = (max, *smax);
                                *smax = min;
                                self.segments.insert(i + 1, newseg);
                                2
                            }
                        } else {
                            // sfmin < fmax
                            if sfmax <= fmax {
                                // Segment fully within removal: deleting
                                self.segments.remove(i);
                                0
                            } else {
                                // Start of segment is within removal
                                *smin = max;
                                1
                            }
                        };
                    }
                }
            }
            debug_assert!(self
                .segments
                .iter()
                .all(|(min, max)| min.map_or(true, |min| max.map_or(true, |max| min <= max))));
        }
    }

    /// Checks whether the bounds are valid for `self`. Returns `Err` if `self.base` is a Circle
    /// and the arc is half-open
    fn check_seg_valid(&self, (min, max): Segment) -> Result<(), Error> {
        match self.base {
            Bisector::Circle { .. } if min.is_some() != max.is_some() => {
                Err(Error::msg("Invalid bisector: half-open arc"))
            }
            _ => Ok(()),
        }
    }

    /// Returns true if this only contains one segment which is the full bisector (None, None)
    fn is_full(&self) -> bool {
        self.segments == [(None, None)]
    }

    #[allow(dead_code)]
    pub fn segments(&self) -> &Vec<Segment> {
        &self.segments
    }

    pub fn base(&self) -> &Bisector {
        &self.base
    }
}

impl Hoverable for (&(usize, usize), &BisectorSegments) {
    fn distance(&self, point: Vec2) -> f32 {
        // TODO do the math for actual segments
        match self.1.base {
            Bisector::Line { orig, vec } => {
                let diff = point - orig;
                if diff.length() == 0. {
                    0.
                } else {
                    vec.perp_dot(diff).abs()
                }
            }
            Bisector::Circle { center, radius } => (center.distance(point) - radius).abs(),
        }
    }

    fn is_hovered(&self, hov: &Hovered) -> bool {
        match hov {
            Hovered::None => false,
            Hovered::Point(i) => *i == self.0 .0 || *i == self.0 .1,
            Hovered::Edge(i1, i2) => *i1 == self.0 .0 && *i2 == self.0 .1,
        }
    }

    fn hover(&self) -> Hovered {
        Hovered::Edge(self.0 .0, self.0 .1)
    }

    fn priority() -> u16 {
        2
    }
}

// #[cfg(none)]
impl Intersectable for BisectorSegments {
    type Point = WeightedPoint;

    fn intersect(
        p1: &Self::Point,
        p2: &Self::Point,
        p3: &Self::Point,
        edge1_2: impl DerefMut<Target = Self>,
        edge1_3: impl DerefMut<Target = Self>,
    ) {
        let mut seg1_2_rem = Vec::new();
        let mut seg1_3_rem = Vec::new();
        match (edge1_2.base, edge1_3.base) {
            (
                b1 @ Bisector::Line { orig: o1, vec: v1 },
                b2 @ Bisector::Line { orig: o2, vec: v2 },
            ) => {
                let denom = v1.perp_dot(v2);
                if denom.abs() <= f32::EPSILON {
                    // Parallel
                    trace!(
                        "intersect: Line({}{})/Line({}{}) = Parallel",
                        p1.name,
                        p2.name,
                        p1.name,
                        p3.name
                    );
                } else {
                    let mul =
                        |v: Vec2, odif: Vec2, denom: f32| (v.y * odif.x + v.x * -odif.y) / denom;
                    #[inline]
                    fn get_seg(v1: Vec2, mul: f32, pdif: Vec2) -> Segment {
                        // Intersection point, expressed as a factor of v1 from o1
                        if pdif.dot(v1) < 0. {
                            (Some(mul), None)
                        } else {
                            (None, Some(mul))
                        }
                    }
                    let mul_v1 = mul(v2, o2 - o1, denom);
                    let mul_v2 = mul(v1, o1 - o2, -denom);
                    let p = b1 * mul_v1;
                    debug_assert!(p.distance(b2 * mul_v2) <= f32::EPSILON);
                    trace!(
                        "Intersection Line({}{})/Line({}{}) => {p}",
                        p1.name,
                        p2.name,
                        p1.name,
                        p3.name
                    );
                    seg1_2_rem.push(get_seg(v1, mul_v1, -p3.pos + p1.pos));
                    seg1_3_rem.push(get_seg(v2, mul_v2, -p2.pos + p1.pos));
                }
            }
            (Bisector::Line { .. }, Bisector::Circle { .. }) => {
                return Self::intersect(p1, p3, p2, edge1_3, edge1_2);
            }
            (circle @ Bisector::Circle { center, radius }, Bisector::Line { orig, vec }) => {
                let cent_proj = orig + (center - orig).project_onto_normalized(vec);
                let dist = cent_proj.distance(center);
                if (dist / radius - 1.).abs() <= f32::EPSILON {
                    // Tangent TODO
                    // Some((cent_proj, None));
                } else if dist < radius {
                    let proj_agl = (cent_proj.y - center.y).atan2(cent_proj.x - center.x);
                    let agl = (dist / radius).acos();
                    let agl = (proj_agl + agl, proj_agl - agl);
                    let inter = (circle * agl.0, circle * agl.1);
                    {
                        assert!((-inter.0 + inter.1).dot(-center + cent_proj) <= f32::EPSILON);
                        let mut agl = Self::mod_arc(agl.0, agl.1);
                        // the center of the circle and p1 are on the same side of the line
                        let cp_same_side =
                            vec.perp_dot(-orig + center) * vec.perp_dot(-orig + p1.pos) >= 0.;
                        // the arg from agl.0 to agl.1 is bigger than agl.1 to agl.0
                        let is_big_arc = (agl.1 - agl.0) > PI;

                        if is_big_arc == cp_same_side {
                            // If we have the big arc and need the little one, or vice versa
                            agl = Self::mod_arc(agl.1, agl.0);
                        };
                        seg1_2_rem.push((Some(agl.0), Some(agl.1)));
                    }
                    {
                        let inter = ((inter.0 - orig).dot(vec), (inter.1 - orig).dot(vec));
                        let inter = (Some(inter.0.min(inter.1)), Some(inter.0.max(inter.1)));
                        if p1.weight > p2.weight {
                            // Remove the inner circle
                            seg1_3_rem.push(inter);
                        } else {
                            seg1_3_rem.push((None, inter.0));
                            seg1_3_rem.push((inter.1, None));
                        }
                    }
                }
            }
            (
                Bisector::Circle {
                    center: c1,
                    radius: r1,
                },
                Bisector::Circle {
                    center: c2,
                    radius: r2,
                },
            ) => {
                let dist = c1.distance(c2);
                let rs = r1 + r2;
                if ((dist - r1).abs() - r2).abs() / rs <= f32::EPSILON {
                    // Tengent
                    // if r1 < r2 {
                    //     // let vec = c1 - c2;
                    //     // Some((c2 + vec * (r2 / vec.length()), None));
                    // } else {
                    //     // let vec = c2 - c1;
                    //     // Some((c1 + vec * (r1 / vec.length()), None));
                    // }
                } else if dist < rs {
                    let x = dist / (2. * r1) + (r1.powi(2) - r2.powi(2)) / (2. * r1 * dist);
                    let vec = (-c1 + c2) * r1 / dist;
                    let p = c1 + vec * x;
                    let y = (1. - x.powi(2)).sqrt();
                    assert!(y.is_finite());
                    let points = (p + vec.perp() * y, p - vec.perp() * y);
                    let same_side = (-c1 + p).dot(-c2 + p) > 0.;
                    fn make_arc(
                        (p1, p2): (Vec2, Vec2),
                        (w1, w2): (f32, f32),
                        center: Vec2,
                        same_side: bool,
                    ) -> Segment {
                        let agl = (-center + p1, -center + p2);
                        let mut agl = BisectorSegments::mod_arc(
                            agl.0.y.atan2(agl.0.x),
                            agl.1.y.atan2(agl.1.x),
                        );
                        let small_arc = w2 <= w1 || !same_side;
                        if (agl.1 - agl.0 > PI) == small_arc {
                            agl = BisectorSegments::mod_arc(agl.1, agl.0);
                        }
                        (Some(agl.0), Some(agl.1))
                    }
                    seg1_2_rem.push(make_arc(points, (p1.weight, p2.weight), c1, same_side));
                    seg1_3_rem.push(make_arc(points, (p1.weight, p3.weight), c2, same_side));
                    // Some((p + vec.perp() * y, Some(p - vec.perp() * y)));
                }
            }
        };

        fn remove_segs(
            mut tgt: impl DerefMut<Target = BisectorSegments>,
            segs: Vec<Segment>,
            p1: &WeightedPoint,
            p2: &WeightedPoint,
        ) {
            let segtype = match &tgt.base {
                Bisector::Line { .. } => "Line",
                Bisector::Circle { .. } => "Circle",
            };
            if segs.is_empty() {
                trace!("Not removing any part of {segtype} {}{}", p1.name, p2.name)
            }
            for seg in segs {
                let segs = tgt.segments.clone();
                tgt.remove(seg);
                debug!(
                    "Removing {seg:?} from {segtype} {}{}: {:?} => {:?}",
                    p1.name, p2.name, segs, tgt.segments
                );
            }
        }
        remove_segs(edge1_2, seg1_2_rem, p1, p2);
        remove_segs(edge1_3, seg1_3_rem, p1, p3);
    }
}

#[cfg(test)]
mod tests {
    use crate::geom::{Bisector, BisectorSegments, Segment};
    use std::borrow::BorrowMut;
    use std::cmp::Ordering;
    use std::f32::consts::{PI, TAU};

    #[inline]
    fn to_deg(f: f32) -> i32 {
        (f * 180. / PI).round() as i32
    }

    #[inline]
    fn iter_to_deg<'a, A, B>(it: A) -> B
    where
        A: Iterator<Item = &'a Segment>,
        B: FromIterator<(Option<i32>, Option<i32>)>,
    {
        it.map(|seg| (seg.0.map(to_deg), seg.1.map(to_deg)))
            .collect()
    }

    fn cmp_segs(a: &Segment, b: &Segment) -> Ordering {
        a.0.unwrap_or(f32::NEG_INFINITY)
            .total_cmp(&b.0.unwrap_or(f32::NEG_INFINITY))
            .then(
                a.1.unwrap_or(f32::INFINITY)
                    .total_cmp(&b.1.unwrap_or(f32::INFINITY)),
            )
    }

    fn check_remove(mut seg: impl BorrowMut<BisectorSegments>, rem: Segment, expect: &[Segment]) {
        _check_remove(seg.borrow_mut(), rem, expect, false);
        // let seg2 = seg.borrow().clone();
        // _check_remove(seg2, rem, &seg.borrow().segments, true); TODO find a way to make it stable
    }

    fn _check_remove(
        mut seg: impl BorrowMut<BisectorSegments>,
        rem: Segment,
        expect: &[Segment],
        second: bool,
    ) {
        let seg = seg.borrow_mut();
        let segs = seg.segments.clone();
        seg.remove(rem);
        let mut res = seg.segments.clone();
        res.sort_by(cmp_segs);
        let mut exp = Vec::from(expect);
        exp.sort_by(cmp_segs);
        let msg = match seg.base {
            Bisector::Line { .. } => {
                format!("Removing {rem:?}\n from {segs:?}\n expected {exp:?}\n got {res:?} (second: {second})")
            }
            Bisector::Circle { .. } => {
                let rem = (rem.0.map(to_deg), rem.1.map(to_deg));
                let segs: Vec<_> = iter_to_deg(segs.iter());
                let exp: Vec<_> = iter_to_deg(exp.iter());
                let res: Vec<_> = iter_to_deg(res.iter());
                format!("Removing {rem:?}\n from {segs:?}\n expected {exp:?}\n got {res:?} (second: {second})")
            }
        };
        assert_eq!(exp, res, "{msg}");
    }

    #[test]
    fn check_segment_remove() {
        let seg = BisectorSegments::new(Bisector::Line {
            orig: (0., 0.).into(),
            vec: (1., 0.).into(),
        });

        assert_eq!(seg.segments, &[(None, None)]);
        check_remove(seg.clone(), (None, None), &[]);

        {
            let mut seg = seg.clone();
            check_remove(&mut seg, (None, Some(-1.)), &[(Some(-1.), None)]);
            check_remove(&mut seg, (Some(1.), None), &[(Some(-1.), Some(1.))]);
            check_remove(&mut seg, (Some(-10.), Some(-5.)), &[(Some(-1.), Some(1.))]);
            check_remove(&mut seg, (Some(5.), Some(10.)), &[(Some(-1.), Some(1.))]);
        }

        check_remove(
            seg.clone(),
            (Some(-1.), Some(1.)),
            &[(None, Some(-1.)), (Some(1.), None)],
        );
    }

    #[test]
    fn check_circle_remove() {
        let seg = BisectorSegments::new(Bisector::Circle {
            center: (0., 0.).into(),
            radius: 1.,
        });

        assert_eq!(seg.segments, &[(None, None)]);
        check_remove(seg.clone(), (None, None), &[]);

        for i in 0..4 {
            let mut seg = seg.clone();

            let min = (i as f32 + 0.5) * PI / 2.;
            let max = min + PI / 2.;
            let res = BisectorSegments::mod_arc(max, min + TAU);
            check_remove(
                &mut seg,
                (Some(min), Some(max)),
                &[(Some(res.0), Some(res.1))],
            );

            let (min, max) = BisectorSegments::mod_arc(min + PI, max + PI);
            let (res, res2) = (
                BisectorSegments::mod_arc(res.0, min),
                BisectorSegments::mod_arc(max, res.1),
            );
            check_remove(
                &mut seg,
                (Some(min), Some(max)),
                &[(Some(res.0), Some(res.1)), (Some(res2.0), Some(res2.1))],
            );
        }
    }
}
