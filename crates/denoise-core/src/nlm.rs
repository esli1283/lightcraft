//! The built-in denoiser's filter: variance stabilisation and a non-local-means average whose weights come from all
//! four colour planes of a tile at once, so the planes are cleaned together instead of each on its own.
//!
//! Noise in a raw file has variance `a · signal + b`. The generalised Anscombe transform ([`Stabiliser`]) turns that
//! into noise of variance 1 everywhere, so one threshold fits every brightness. The four planes are then rotated
//! (a Hadamard transform, which keeps unit noise) into a brightness plane and three colour-difference planes.
//! Each cell becomes the average of the cells around it whose patch, over all four planes, looks like its own
//! within the noise, in two passes: the second finds matches on the first pass's much cleaner picture, so it can
//! look further and keep finer detail.

use crate::bayer::reflect;

/// What the filter may be tuned by (the defaults are what the built-in denoiser uses).
///
/// How strongly a photo is cleaned follows its noise, measured as the standard deviation at mid-grey (0.18, white = 1):
/// below `light` it is cleaned gently and in one pass, above `heavy` as hard as the filter goes, and in between smoothly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tuning {
    /// Search window radius, in cells, of the first pass.
    pub search: usize,
    /// Search window radius of the second pass, which finds matches on the first pass's cleaner picture; 0 skips it.
    pub search2: usize,
    /// How far above the noise two patches may differ and still be averaged in the first pass (larger blurs more),
    /// for a light and for a heavily noisy photo.
    pub h_light: f32,
    pub h_heavy: f32,
    /// The same for the second pass, which is blended in as the noise grows.
    pub h2: f32,
    /// Noise at mid-grey where the stronger settings start and where they are in full.
    pub light: f32,
    pub heavy: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning { search: 3, search2: 5, h_light: 0.3, h_heavy: 0.9, h2: 0.12, light: 0.03, heavy: 0.14 }
    }
}

impl Tuning {
    /// 0 for a lightly noisy photo up to 1 for a heavily noisy one.
    pub fn strength(&self, noise: f32) -> f32 {
        let span = (self.heavy - self.light).max(1e-6);
        let x = ((noise - self.light) / span).clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    }
}

/// The generalised Anscombe transform for noise of variance `a · v + b`.
#[derive(Clone, Copy, Debug)]
pub struct Stabiliser {
    a: f32,
    b: f32,
}

/// Below this the noise has no signal-dependent part worth modelling.
const MIN_GAIN: f32 = 1e-6;

impl Stabiliser {
    pub fn new(a: f32, b: f32) -> Self {
        Stabiliser { a: if a.is_finite() { a.max(0.0) } else { 0.0 }, b: if b.is_finite() { b.max(1e-12) } else { 1e-12 } }
    }

    pub fn forward(self, v: f32) -> f32 {
        if self.a < MIN_GAIN {
            return v / self.b.sqrt();
        }
        let x = (v + self.b / self.a) / self.a;
        2.0 * (x + 0.375).max(0.0).sqrt()
    }

    /// The inverse, with the correction for the bias a plain inverse has on averaged data.
    pub fn inverse(self, y: f32) -> f32 {
        if self.a < MIN_GAIN {
            return y * self.b.sqrt();
        }
        let x = if y >= 1.5 {
            let (i1, i2) = (1.0 / y, 1.0 / (y * y));
            y * y / 4.0 + 0.25 * 1.5f32.sqrt() * i1 - 11.0 / 8.0 * i2 + 5.0 / 8.0 * 1.5f32.sqrt() * i2 * i1 - 0.125
        } else {
            y * y / 4.0 - 0.375
        };
        self.a * x - self.b / self.a
    }
}

/// The orthonormal 4 × 4 Hadamard transform across four planes (its own inverse).
pub fn hadamard(p: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = p.first().map_or(0, Vec::len);
    let mut out = vec![vec![0f32; n]; 4];
    for i in 0..n {
        let g = |c: usize| p.get(c).and_then(|v| v.get(i)).copied().unwrap_or(0.0);
        let (r, g1, g2, b) = (g(0), g(1), g(2), g(3));
        let vals = [(r + g1 + g2 + b) / 2.0, (r - g1 + g2 - b) / 2.0, (r + g1 - g2 - b) / 2.0, (r - g1 - g2 + b) / 2.0];
        for (c, v) in vals.into_iter().enumerate() {
            if let Some(o) = out.get_mut(c).and_then(|o| o.get_mut(i)) {
                *o = v;
            }
        }
    }
    out
}

/// A plane with `pad` cells mirrored round every edge (`t + 2 · pad` on a side).
fn padded(src: &[f32], t: usize, pad: usize) -> Vec<f32> {
    let wd = t + 2 * pad;
    let mut out = vec![0f32; wd * wd];
    for y in 0..wd {
        let sy = reflect(y as isize - pad as isize, t);
        for x in 0..wd {
            let sx = reflect(x as isize - pad as isize, t);
            if let (Some(o), Some(v)) = (out.get_mut(y * wd + x), src.get(sy * t + sx)) {
                *o = *v;
            }
        }
    }
    out
}

/// One pass: every cell becomes the weighted average of the cells within `radius` whose 3 × 3 patch, over all planes,
/// is like its own. `guide` holds the planes the patches are compared on, `data` the planes that are averaged
/// (the same ones in the first pass). Weights are `exp(-(excess) / h)`, `excess` being how far the mean squared
/// difference of the patches is above `floor` (what two noisy views of one signal give).
fn pass(guide: &[Vec<f32>], data: &[Vec<f32>], t: usize, radius: usize, h: f32, floor: f32) -> Vec<Vec<f32>> {
    let pad = radius + 1;
    let wd = t + 2 * pad;
    let guide: Vec<Vec<f32>> = guide.iter().map(|p| padded(p, t, pad)).collect();
    let data: Vec<Vec<f32>> = data.iter().map(|p| padded(p, t, pad)).collect();
    let (n, region) = (t * t, t + 2);
    let planes = data.len().max(1);
    let mut num = vec![vec![0f32; n]; planes];
    let mut den = vec![0f32; n];
    let (mut d, mut rows) = (vec![0f32; region * region], vec![0f32; region * t]);
    let norm = 1.0 / (9.0 * guide.len().max(1) as f32);
    let r = radius as isize;
    for oy in -r..=r {
        for ox in -r..=r {
            if ox == 0 && oy == 0 {
                continue;
            }
            let shift = oy * wd as isize + ox;
            // squared differences over the planes, for the cells and one ring round them
            for j in 0..region {
                let base = (j + pad - 1) * wd + pad - 1;
                for i in 0..region {
                    let pos = base + i;
                    let other = (pos as isize + shift) as usize;
                    let mut s = 0f32;
                    for p in &guide {
                        let (a, b) = (p.get(pos).copied().unwrap_or(0.0), p.get(other).copied().unwrap_or(0.0));
                        s += (a - b) * (a - b);
                    }
                    if let Some(v) = d.get_mut(j * region + i) {
                        *v = s;
                    }
                }
            }
            // the sum over each cell's 3 × 3 patch: along rows, then down columns
            for j in 0..region {
                for x in 0..t {
                    let k = j * region + x;
                    let s = d.get(k).copied().unwrap_or(0.0) + d.get(k + 1).copied().unwrap_or(0.0) + d.get(k + 2).copied().unwrap_or(0.0);
                    if let Some(v) = rows.get_mut(j * t + x) {
                        *v = s;
                    }
                }
            }
            for y in 0..t {
                for x in 0..t {
                    let k = y * t + x;
                    let patch =
                        (rows.get(k).copied().unwrap_or(0.0) + rows.get(k + t).copied().unwrap_or(0.0) + rows.get(k + 2 * t).copied().unwrap_or(0.0))
                            * norm;
                    let w = (-((patch - floor).max(0.0)) / h).exp();
                    let pos = (y + pad) * wd + x + pad;
                    let other = (pos as isize + shift) as usize;
                    if let Some(dn) = den.get_mut(k) {
                        *dn += w;
                    }
                    for (nu, p) in num.iter_mut().zip(&data) {
                        if let (Some(nu), Some(v)) = (nu.get_mut(k), p.get(other)) {
                            *nu += w * v;
                        }
                    }
                }
            }
        }
    }
    // the cell itself counts as much as a perfect match
    let mut out = vec![vec![0f32; n]; planes];
    for (o, (nu, p)) in out.iter_mut().zip(num.iter().zip(&data)) {
        for k in 0..n {
            let own = p.get((k / t + pad) * wd + k % t + pad).copied().unwrap_or(0.0);
            if let (Some(o), Some(nu), Some(dn)) = (o.get_mut(k), nu.get(k), den.get(k)) {
                *o = (nu + own) / (dn + 1.0);
            }
        }
    }
    out
}

/// Denoise four unit-noise planes of `t × t` cells (see the module description); `noise` is the photo's noise at
/// mid-grey, which sets how strongly (see [`Tuning`]).
pub fn filter(planes: &[Vec<f32>], t: usize, tune: &Tuning, noise: f32) -> Vec<Vec<f32>> {
    let y = hadamard(planes);
    let k = tune.strength(if noise.is_finite() { noise } else { 0.0 });
    // two noisy views of one signal differ by 2 in mean square: the floor the excess is counted from
    let first = pass(&y, &y, t, tune.search, tune.h_light + (tune.h_heavy - tune.h_light) * k, 2.0);
    if tune.search2 == 0 || k <= 0.0 {
        return hadamard(&first);
    }
    // the first pass's picture still carries a little noise (of unknown size), so the floor is 0 and the threshold
    // alone decides
    let second = pass(&first, &y, t, tune.search2, tune.h2, 0.0);
    let blended: Vec<Vec<f32>> = first.iter().zip(&second).map(|(f, s)| f.iter().zip(s).map(|(f, s)| f + (s - f) * k).collect()).collect();
    hadamard(&blended)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stabiliser_round_trips_and_gives_unit_noise() {
        let s = Stabiliser::new(0.01, 0.0005);
        for v in [0.0f32, 0.01, 0.1, 0.5, 0.9] {
            let back = s.inverse(s.forward(v));
            assert!((back - v).abs() < 0.005 + 0.03 * v, "{v} -> {back}");
        }
        // a step of one standard deviation is a step of about 1 whatever the brightness
        for v in [0.05f32, 0.2, 0.6] {
            let sigma = (0.01 * v + 0.0005).sqrt();
            let step = s.forward(v + sigma) - s.forward(v);
            assert!((step - 1.0).abs() < 0.15, "at {v}: {step}");
        }
        let flat = Stabiliser::new(0.0, 4e-4);
        assert!((flat.forward(0.1) - 5.0).abs() < 1e-4 && (flat.inverse(5.0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn the_hadamard_transform_is_its_own_inverse_and_keeps_energy() {
        let planes: Vec<Vec<f32>> = (0..4).map(|c| (0..9).map(|i| (c * 9 + i) as f32 * 0.37 - 3.0).collect()).collect();
        let back = hadamard(&hadamard(&planes));
        for (a, b) in planes.iter().flatten().zip(back.iter().flatten()) {
            assert!((a - b).abs() < 1e-4);
        }
        let energy = |p: &[Vec<f32>]| p.iter().flatten().map(|v| v * v).sum::<f32>();
        assert!((energy(&planes) - energy(&hadamard(&planes))).abs() < 1e-2);
    }
}
