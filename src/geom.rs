use crate::app::State;
use crate::voronoi::Hovered;
use crate::{Color, Draw, Graphics};
use anyhow::Error;
use notan::draw::DrawShapes;
use notan::log::*;
use notan::math::{DVec2, Vec2};
use std::borrow::Cow;
use std::f64::consts::{PI, TAU};
use std::ops::{DerefMut, Div, Mul};

pub trait Hoverable {
    fn distance(&self, point: DVec2) -> f64;
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
    pub pos: DVec2,
    pub weight: u32,
    pub enabled: bool,
}

impl WeightedPoint {
    pub fn new(name: impl ToString, p: DVec2, weight: u32) -> WeightedPoint {
        let name = name.to_string();
        Self {
            name,
            pos: p,
            weight,
            enabled: true,
        }
    }

    pub fn label(&self) -> String {
        format!("{}, {:.2}", self.name, self.weight)
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
            let a2 = a.pow(2) as f64;
            let b2 = b.pow(2) as f64;

            Bisector::Circle {
                center: self.pos + (other.pos - self.pos) * -a2 / (b2 - a2),
                radius: self.pos.distance(other.pos) * (a * b) as f64 / (b2 - a2).abs(),
            }
        }
    }
}

impl Hoverable for (&usize, &WeightedPoint) {
    fn distance(&self, point: DVec2) -> f64 {
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
    Line { orig: DVec2, vec: DVec2 },
    Circle { center: DVec2, radius: f64 },
}

impl Bisector {
    /// Calculates intersection between two bisectors. Regardless of combination of types there can
    /// only be zero, one or two bisectors, hence the return type. The returned points are expressed
    /// as (f64, f64), corresponding to the respective multipliers to apply to self and other to
    /// obtain the actual points
    #[allow(clippy::type_complexity)]
    pub fn intersect(self, other: Bisector) -> Option<((f64, f64), Option<(f64, f64)>)> {
        match (self, other) {
            (Bisector::Line { orig: o1, vec: v1 }, Bisector::Line { orig: o2, vec: v2 }) => {
                let denom = v1.perp_dot(v2);
                if denom.abs() <= f64::EPSILON {
                    // Parallel
                    None
                } else {
                    let mul =
                        |v: DVec2, odif: DVec2, denom: f64| (v.y * odif.x + v.x * -odif.y) / denom;
                    let mul_v1 = mul(v2, o2 - o1, denom);
                    let mul_v2 = mul(v1, o1 - o2, -denom);
                    Some(((mul_v1, mul_v2), None))
                }
            }
            (Bisector::Line { .. }, Bisector::Circle { .. }) => other
                .intersect(self)
                .map(|(seg, other)| ((seg.1, seg.0), other.map(|seg| (seg.1, seg.0)))),
            (circle @ Bisector::Circle { center, radius }, line @ Bisector::Line { orig, vec }) => {
                let cent_proj = orig + (center - orig).project_onto_normalized(vec);
                let dist = cent_proj.distance(center);
                #[allow(clippy::float_equality_without_abs)]
                if dist - radius < f64::EPSILON {
                    let proj_agl = self / cent_proj;
                    if (dist - radius).abs() < f64::EPSILON {
                        // Tangent
                        let agl = circle / cent_proj;
                        let seg = line / cent_proj;
                        Some(((agl, seg), None))
                    } else {
                        let agl = (dist / radius).acos();
                        let agl = (proj_agl - agl, proj_agl + agl);
                        let seg = (line / (circle * agl.0), line / (circle * agl.1));
                        Some(((agl.0, seg.0), Some((agl.1, seg.1))))
                    }
                } else {
                    None
                }
            }
            (
                circle1 @ Bisector::Circle {
                    center: c1,
                    radius: r1,
                },
                circle2 @ Bisector::Circle {
                    center: c2,
                    radius: r2,
                },
            ) => {
                let dist = c1.distance(c2);
                let diff = (dist - r1.max(r2)).abs() - r1.min(r2);

                if diff < f64::EPSILON {
                    let x = dist / (2. * r1) + (r1.powi(2) - r2.powi(2)) / (2. * r1 * dist);
                    let vec = (-c1 + c2) * r1 / dist;
                    let p = c1 + vec * x;
                    if diff > -f64::EPSILON {
                        Some(((circle1 / p, circle2 / p), None))
                    } else {
                        let y = (1. - x.powi(2)).sqrt();
                        assert!(y.is_finite());
                        let points = (p + vec.perp() * y, p - vec.perp() * y);
                        Some((
                            (circle1 / points.0, circle2 / points.0),
                            Some((circle1 / points.1, circle2 / points.1)),
                        ))
                    }
                } else {
                    None
                }
            }
        }
    }
}

impl Mul<f64> for Bisector {
    type Output = DVec2;

    /// Allows multiplying a bisector by a float to get a point corresponding to `rhs` as a multiplier if
    /// `self` is a `Line`, or an angle if `self` is a `Circle`
    fn mul(self, rhs: f64) -> Self::Output {
        match self {
            Bisector::Line { orig, vec } => orig + vec * rhs,
            Bisector::Circle { center, radius } => center + DVec2::from_angle(rhs) * radius,
        }
    }
}

impl Div<DVec2> for Bisector {
    type Output = f64;

    // Returns the f64 that, multiplied by self, would produce the closest point to rhs
    fn div(self, rhs: DVec2) -> Self::Output {
        match self {
            Bisector::Line { orig, vec } => (rhs - orig).dot(vec),
            Bisector::Circle { center, .. } => {
                let vec = rhs - center;
                f64::atan2(vec.y, vec.x)
            }
        }
    }
}

type Segment = (Option<f64>, Option<f64>);

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
                        state.view().to_screen(p1).as_vec2().into(),
                        state.view().to_screen(p2).as_vec2().into(),
                    )
                    .width(stroke)
                    .color(skel_col)
                    .width(skel_stroke_width);
                }

                for (min, max) in &self.segments {
                    let p1 = orig + vec * min.unwrap_or_else(|| -state.render_edge_distance());
                    let p1 = state.view().to_screen(p1).as_vec2();
                    let p2 = orig + vec * max.unwrap_or_else(|| state.render_edge_distance());
                    let p2 = state.view().to_screen(p2).as_vec2();

                    point(draw, &p1);
                    point(draw, &p2);
                    draw.line(p1.into(), p2.into()).width(stroke).color(col);
                }
            }
            base @ Bisector::Circle { center, radius, .. } => {
                let center = state.view().to_screen(center);
                let mut full_circle = draw.circle((radius * state.view().to_screen_ratio()) as f32);
                full_circle.position(center.x as f32, center.y as f32);
                if self.is_full() {
                    full_circle.stroke(stroke).stroke_color(col);
                } else {
                    full_circle.stroke(skel_stroke_width).stroke_color(skel_col);
                    drop(full_circle);
                    let segments = if !state.is_within_render(center) {
                        // Circle is significantly outside rendered area. Trim min-max angle
                        let mid = state.view().to_math(state.render_size().as_dvec2() / 2.);
                        let mid_agl = self.base / (mid);
                        let half_arc = ((mid.length() + 10.) / radius).min(PI / 2.);
                        let arc = Self::mod_arc(mid_agl + half_arc, mid_agl - half_arc);
                        trace!("Only rendering circle outside of arc {arc:?}");
                        let mut clipped = self.clone();
                        clipped.remove((Some(arc.0), Some(arc.1)));
                        Cow::from(clipped.segments)
                    } else {
                        Cow::from(&self.segments)
                    };

                    for (min, max) in segments.iter() {
                        // TODO limit if circle goes outside screen edges
                        if let (Some(min), Some(max)) = (*min, *max) {
                            // Approximate angle to increment to advance ~1px on screen
                            let pix_agl = state.view().to_math_ratio() / radius;

                            let mut agl = min;
                            let mut p = state.view().to_screen(base * agl).as_vec2();
                            point(draw, &p);
                            let mut path = draw.path();
                            path.stroke(stroke).color(col).move_to(p.x, p.y);
                            while max - agl > -f64::EPSILON {
                                p = state.view().to_screen(base * agl).as_vec2();
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
    fn mod_arc(min: f64, max: f64) -> (f64, f64) {
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
            let fmin = min.unwrap_or(f64::NEG_INFINITY);
            let fmax = max.unwrap_or(f64::INFINITY);
            assert!(
                fmin <= fmax,
                "Remove received invalid arc: ({min:?},{max:?})"
            );
            match self.base {
                Bisector::Circle { .. } => {
                    self.check_seg_valid((min, max)).unwrap();
                    let mut i = 0;
                    while i < self.segments.len() {
                        let (smin, smax) = &mut self.segments[i];
                        // (min, max) can't be (None, None) nor half-open, so we know it's fully-closed
                        i += if smin.is_some() && smax.is_some() {
                            let (sfmin, sfmax) = (smin.as_mut().unwrap(), smax.as_mut().unwrap());
                            #[inline]
                            fn is_within(it: f64, (min, max): (f64, f64)) -> bool {
                                (it - min).rem_euclid(TAU) < (max - min)
                            }

                            let smin_in_rem = is_within(*sfmin, (fmin, fmax));
                            let smax_in_rem = is_within(*sfmax, (fmin, fmax));
                            let min_in_segment = is_within(fmin, (*sfmin, *sfmax));
                            if smin_in_rem {
                                if smax_in_rem {
                                    if min_in_segment {
                                        trace!("Removal and segment are negatively included within each other: trimming both ends");
                                        (*sfmin, *sfmax) = Self::mod_arc(fmax, fmin);
                                        1
                                    } else {
                                        trace!("Arc is fully within removal: removing");
                                        self.segments.remove(i);
                                        0
                                    }
                                } else {
                                    trace!("Start of arc is within removal");
                                    (*sfmin, *sfmax) = Self::mod_arc(fmax, *sfmax);
                                    1
                                }
                            } else if smax_in_rem {
                                trace!("End of arc is within removal");
                                *sfmax = Self::mod_arc(*sfmin, fmin).1;
                                1
                            } else if min_in_segment {
                                trace!("Removal is fully within the arc: splitting");
                                let (newmin, newmax) = Self::mod_arc(fmax, *sfmax);
                                *sfmax = Self::mod_arc(*sfmin, fmin).1;
                                self.segments.insert(i + 1, (Some(newmin), Some(newmax)));
                                2
                            } else {
                                trace!("No intersection");
                                1
                            }
                        } else if smin.is_none() && smax.is_none() {
                            trace!("Removing an arc from a full circle: making it the inverse arc");
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
                        let sfmin = smin.unwrap_or(f64::NEG_INFINITY);
                        let sfmax = smax.unwrap_or(f64::INFINITY);

                        i += if sfmin >= fmax || sfmax <= fmin {
                            trace!("No intersection");
                            1
                        } else if sfmin < fmin {
                            if sfmax <= fmax {
                                trace!("End of segment is within removal");
                                *smax = min;
                                1
                            } else {
                                trace!("Removal is fully within segment: splitting");
                                let newseg = (max, *smax);
                                *smax = min;
                                self.segments.insert(i + 1, newseg);
                                2
                            }
                        } else {
                            // sfmin < fmax
                            if sfmax <= fmax {
                                trace!("Segment fully within removal: deleting");
                                self.segments.remove(i);
                                0
                            } else {
                                trace!("Start of segment is within removal");
                                *smin = max;
                                1
                            }
                        };
                    }
                }
            }
            self.segments.iter().for_each(|(min, max)| {
                min.map(|min| {
                    max.map(|max| {
                        debug_assert!(
                            min <= max,
                            "Segment doesn't satisfy min<=max: ({min}, {max})"
                        )
                    })
                });
            });
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

    fn segname(a: &str, b: &str) -> String {
        let mut v = [a, b];
        v.sort();
        v.concat()
    }
}

impl Hoverable for (&(usize, usize), &BisectorSegments) {
    fn distance(&self, point: DVec2) -> f64 {
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
        // Lambda for lazy computation, as we have one branch where it is not used
        let inter = || edge1_2.base.intersect(edge1_3.base);
        match (edge1_2.base, edge1_3.base) {
            (
                b1 @ Bisector::Line { orig: o1, vec: v1 },
                b2 @ Bisector::Line { orig: o2, vec: v2 },
            ) => {
                match inter() {
                    Some((_, Some(_))) => panic!(),
                    Some(((mul_e1, mul_e2), None)) => {
                        #[inline]
                        fn get_seg(v1: DVec2, mul: f64, pdif: DVec2) -> Segment {
                            // Intersection point, expressed as a factor of v1 from o1
                            if pdif.dot(v1) < 0. {
                                (Some(mul), None)
                            } else {
                                (None, Some(mul))
                            }
                        }
                        let p = b1 * mul_e1;
                        let dist_check = p.distance(b2 * mul_e2);
                        debug_assert!(
                            dist_check <= f64::EPSILON,
                            "dist_check ({dist_check}) should be <= f64::EPSILON"
                        );
                        trace!(
                            "Intersection Line({}{})/Line({}{}) => {p}",
                            p1.name,
                            p2.name,
                            p1.name,
                            p3.name
                        );
                        seg1_2_rem.push(get_seg(v1, mul_e1, -p3.pos + p1.pos));
                        seg1_3_rem.push(get_seg(v2, mul_e2, -p2.pos + p1.pos));
                    }
                    None => {
                        // Parallel
                        let dot1 = v1.perp_dot(-o2 + p1.pos);
                        let dot2 = v1.perp_dot(-o1 + p1.pos);
                        if dot1 * dot2 >= 0. {
                            // One line is behind the other relatively to p1
                            if dot1.abs() > dot2.abs() {
                                seg1_2_rem.push((None, None));
                            } else {
                                seg1_3_rem.push((None, None));
                            }
                        } // else p1 is between the two lines: n
                        trace!(
                            "intersect: Line({})/Line({}) = Parallel",
                            Self::segname(&p1.name, &p2.name),
                            Self::segname(&p1.name, &p3.name)
                        );
                    }
                }
            }
            (Bisector::Line { .. }, Bisector::Circle { .. }) => {
                return Self::intersect(p1, p3, p2, edge1_3, edge1_2);
            }
            (circle @ Bisector::Circle { center, .. }, Bisector::Line { orig, vec }) => {
                let cent_proj = orig + (center - orig).project_onto_normalized(vec);
                // the center of the circle and p1 are on the same side of the line
                let cp_same_side =
                    vec.perp_dot(-orig + center) * vec.perp_dot(-orig + p1.pos) >= 0.;
                match inter() {
                    Some(((mul1_e1, mul1_e2), Some((mul2_e1, mul2_e2)))) => {
                        let agl = (mul1_e1, mul2_e1);
                        let inter = (circle * agl.0, circle * agl.1);
                        {
                            let agl_check = (-inter.0 + inter.1)
                                .angle_between(-center + cent_proj)
                                .abs()
                                - PI / 2.;
                            debug_assert!(
                                agl_check / TAU <= f64::EPSILON,
                                "Angle between (inter.0, inter.1) and (center, cent_proj) = {agl_check} (should be ~=0)");
                            let mut agl = Self::mod_arc(agl.0, agl.1);
                            // the arg from agl.0 to agl.1 is bigger than agl.1 to agl.0
                            let is_big_arc = (agl.1 - agl.0) > PI;
                            if is_big_arc == cp_same_side {
                                // If we have the big arc and need the little one, or vice versa
                                agl = Self::mod_arc(agl.1, agl.0);
                            };
                            seg1_2_rem.push((Some(agl.0), Some(agl.1)));
                        }
                        {
                            let inter = (mul1_e2.min(mul2_e2), mul1_e2.max(mul2_e2));
                            let inter = (Some(inter.0), Some(inter.1));
                            if p1.weight > p2.weight {
                                // Remove the inner circle
                                seg1_3_rem.push(inter);
                            } else {
                                seg1_3_rem.push((None, inter.0));
                                seg1_3_rem.push((inter.1, None));
                            }
                        }
                    }
                    Some(((_mul_e1, _mul_e2), None)) => {
                        // TODO tangent
                    }
                    None => {
                        // No intersection or tangent
                        if p1.weight < p2.weight {
                            // Keeping the inside of the circle = removing the whole other line
                            seg1_3_rem.push((None, None));
                        } else if !cp_same_side {
                            // The circle is on the other side of the line : removing it
                            seg1_2_rem.push((None, None))
                        } // else p1 is between the line and the circle: do nothing
                    }
                }
            }
            (
                cir1 @ Bisector::Circle {
                    center: c1,
                    radius: r1,
                },
                cir2 @ Bisector::Circle {
                    center: c2,
                    radius: r2,
                },
            ) => {
                match inter() {
                    Some(((mul1_e1, mul1_e2), Some((mul2_e1, mul2_e2)))) => {
                        fn make_arc(
                            inside: bool,
                            (m1, m2): (f64, f64),
                            cir: Bisector,
                            cnt: DVec2,
                            rad: f64,
                        ) -> Segment {
                            let (m1, m2) = BisectorSegments::mod_arc(m1, m2);
                            let arc = if inside == ((cir * ((m1 + m2) / 2.)).distance(cnt) <= rad) {
                                // if (p1 is inside e1_2) == (e2 arc is inside e1), flip it
                                BisectorSegments::mod_arc(m2, m1)
                            } else {
                                (m1, m2)
                            };
                            (Some(arc.0), Some(arc.1))
                        }
                        seg1_2_rem.push(make_arc(
                            p1.weight < p3.weight,
                            (mul1_e1, mul2_e1),
                            cir1,
                            c2,
                            r2,
                        ));
                        seg1_3_rem.push(make_arc(
                            p1.weight < p2.weight,
                            (mul1_e2, mul2_e2),
                            cir2,
                            c1,
                            r1,
                        ));
                    }
                    Some(((_mul_e1, _mul_e2), None)) => {
                        // TODO tangent
                    }
                    None => {}
                }
                let dist = c1.distance(c2);
                let diff = (dist - r1.max(r2)).abs() - r1.min(r2);

                if diff < 0. {
                } else {
                    // No intersection
                    let nested = dist < r1.max(r2);
                    match (p1.weight < p2.weight, p1.weight < p3.weight) {
                        (true, true) => {
                            // p1 is inside both circles. Remove the outer one
                            assert!(nested);
                            if p2.weight > p3.weight {
                                &mut seg1_2_rem
                            } else {
                                &mut seg1_3_rem
                            }
                            .push((None, None))
                        }
                        (false, false) if nested => {
                            // Remove the smaller circle
                            if p2.weight < p3.weight {
                                &mut seg1_2_rem
                            } else {
                                &mut seg1_3_rem
                            }
                            .push((None, None))
                        } // Else do nothing
                        _ => {} // p1 is between one circle and outside the other. Do nothing
                    }
                    if dist < r1.max(r2) {
                        // One of the circle is inside the other
                    }
                }
            }
        };
        let seg1name = Self::segname(&p1.name, &p2.name);
        let seg2name = Self::segname(&p1.name, &p3.name);
        debug!(
            "Intersecting around {} of bisectors {seg1name} and {seg2name}",
            p1.name
        );

        fn remove_segs(
            mut tgt: impl DerefMut<Target = BisectorSegments>,
            segs: Vec<Segment>,
            segname: &String,
        ) {
            let segtype = match &tgt.base {
                Bisector::Line { .. } => "Line",
                Bisector::Circle { .. } => "Circle",
            };
            if segs.is_empty() {
                trace!("Not removing any part of {segtype} {segname}")
            } else {
                let prev = tgt.segments.clone();
                for seg in &segs {
                    tgt.remove(*seg);
                }
                debug!(
                    "Removing {segs:?} from {segtype} {segname}: {:?} => {:?}",
                    prev, tgt.segments
                );
            }
        }
        remove_segs(edge1_2, seg1_2_rem, &seg1name);
        remove_segs(edge1_3, seg1_3_rem, &seg2name);
    }
}

#[cfg(test)]
mod tests {
    use crate::geom::{Bisector, BisectorSegments, Segment};
    use std::borrow::BorrowMut;
    use std::cmp::Ordering;
    use std::f64::consts::{PI, TAU};

    #[inline]
    fn to_deg(f: f64) -> i32 {
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
        a.0.unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&b.0.unwrap_or(f64::NEG_INFINITY))
            .then(
                a.1.unwrap_or(f64::INFINITY)
                    .total_cmp(&b.1.unwrap_or(f64::INFINITY)),
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

            let min = (i as f64 + 0.5) * PI / 2.;
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
