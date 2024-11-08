use notan::math::Vec2;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct WeightedPoint {
    pub(crate) label: String,
    pub(crate) pos: Vec2,
    pub(crate) weight: f32,
}

impl WeightedPoint {
    pub fn new(name: impl ToString, p: Vec2, weight: f32) -> WeightedPoint {
        let name = name.to_string();
        Self {
            label: format!("{name}, {weight}"),
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
                vec: (other.pos - self.pos).rotate(Vec2::Y).normalize(),
            }
        } else {
            let a2 = a.powi(2);
            let b2 = b.powi(2);

            Bisector::Circle {
                center: self.pos + (other.pos - self.pos) * -a2 / (b2 - a2),
                radius: self.pos.distance(other.pos) * (a * b) / (b2 - a2),
            }
        }
    }
}

pub type BisectMap = HashMap<(usize, usize), Bisector>;

#[derive(Debug, Clone)]
pub enum Bisector {
    Line { orig: Vec2, vec: Vec2 },
    Circle { center: Vec2, radius: f32 },
}
