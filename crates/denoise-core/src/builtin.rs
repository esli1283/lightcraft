//! The built-in denoiser: what AI Denoise runs when no model has been installed, so it works out of the box.
//!
//! It is a [`TileRunner`] with the same contract as a model (four Bayer planes in, camera RGB at twice the plane
//! size out), written from scratch and needing no file: the noise level of the tile is measured from the tile itself
//! (Poisson-Gaussian: variance = a · signal + b), each colour plane is cleaned with a signal-adaptive non-local-means
//! filter, and the cleaned mosaic is demosaiced with edge-directed green and colour-difference red and blue.

use crate::bayer::reflect;
use crate::licence::{Commercial, Licence};
use crate::manifest::{DenoiserManifest, Domain, Gain};
use crate::run::{Error, TileRunner};

/// Id of the built-in denoiser; no installed model may use it (the models folder is never searched for it).
pub const ID: &str = "builtin";
/// Changes when the algorithm does, so that cached pictures made by an older one are made again.
pub const VERSION: u32 = 1;
/// Side of its tile, in cells.
pub const TILE: u32 = 256;
/// Cells neighbouring tiles share.
pub const OVERLAP: u32 = 16;

/// Search window radius of the non-local means, in cells.
const SEARCH: isize = 2;
/// How far above the noise two patches may differ and still be averaged (larger blurs more).
const STRENGTH: f32 = 1.0;
/// Cells on a side of the blocks the noise level is measured over.
const BLOCK: usize = 16;
/// The smallest variance used, so that a clean tile is left alone instead of divided by zero.
const MIN_VARIANCE: f32 = 1e-10;
/// Blocks with a mean above this (white = 1) may be clipped and say nothing about the noise.
const CLIPPED: f32 = 0.9;

/// The manifest of the built-in denoiser, as the engine treats an installed model.
pub fn manifest() -> DenoiserManifest {
    DenoiserManifest {
        id: ID.into(),
        name: "Built-in".into(),
        version: VERSION.to_string(),
        licence: Licence {
            name: "MIT OR Apache-2.0".into(),
            commercial: Commercial::Yes,
            url: None,
            notice: "Part of LightCraft: no download, no weights.".into(),
        },
        source: None,
        sha256: None,
        size_bytes: None,
        provenance: "Non-local-means filtering with a noise level measured from each photo, written for LightCraft.".into(),
        domain: Domain::BayerToRgb,
        tile: TILE,
        overlap: OVERLAP,
        gain: Gain::None,
    }
}

/// The built-in denoiser as a tile runner (it holds no state).
#[derive(Clone, Copy, Debug, Default)]
pub struct Builtin;

impl TileRunner for Builtin {
    fn run(&self, input: &[f32]) -> Result<Vec<f32>, Error> {
        let t = side_of(input.len()).ok_or_else(|| Error::Input("the tile is not four square planes".into()))?;
        let clean: Vec<f32> = input.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
        let n = t * t;
        let planes: Vec<&[f32]> = clean.chunks_exact(n).collect();
        let (a, b) = noise_model(&planes, t);
        let denoised: Vec<Vec<f32>> = planes.iter().map(|p| non_local_means(p, t, a, b)).collect();
        Ok(demosaic(&denoised, t))
    }
}

/// The side of the planes of a tile of `len` values, if it is four squares.
fn side_of(len: usize) -> Option<usize> {
    let per = len / 4;
    let t = (per as f64).sqrt().round() as usize;
    (len.is_multiple_of(4) && (8..=2048).contains(&t) && t * t == per).then_some(t)
}

/// Mean of each 3 × 3 neighbourhood (edge cells repeated).
fn box3(src: &[f32], t: usize) -> Vec<f32> {
    let at = |x: isize, y: isize| src.get(y.clamp(0, t as isize - 1) as usize * t + x.clamp(0, t as isize - 1) as usize).copied().unwrap_or(0.0);
    let mut rows = vec![0f32; t * t];
    for y in 0..t {
        for x in 0..t {
            let (xi, yi) = (x as isize, y as isize);
            if let Some(r) = rows.get_mut(y * t + x) {
                *r = at(xi - 1, yi) + at(xi, yi) + at(xi + 1, yi);
            }
        }
    }
    let row = |x: usize, y: isize| rows.get(y.clamp(0, t as isize - 1) as usize * t + x).copied().unwrap_or(0.0);
    let mut out = vec![0f32; t * t];
    for y in 0..t {
        for x in 0..t {
            if let Some(o) = out.get_mut(y * t + x) {
                *o = (row(x, y as isize - 1) + row(x, y as isize) + row(x, y as isize + 1)) / 9.0;
            }
        }
    }
    out
}

/// Measure the noise of a tile: `(a, b)` with variance = `a · signal + b`. Each block of a plane gives the signal
/// (its mean) and the variance of its cells' difference from their four neighbours (which texture and edges only
/// raise), the lower part of the variances at each brightness is the noise, and a line is fitted through them.
fn noise_model(planes: &[&[f32]], t: usize) -> (f32, f32) {
    let mut blocks: Vec<(f32, f32)> = Vec::new();
    for p in planes {
        let at = |x: usize, y: usize| p.get(y * t + x).copied().unwrap_or(0.0);
        let mut by = 1;
        while by + BLOCK < t {
            let mut bx = 1;
            while bx + BLOCK < t {
                let (mut sum, mut sq) = (0f64, 0f64);
                for y in by..by + BLOCK {
                    for x in bx..bx + BLOCK {
                        let c = at(x, y);
                        let r = c - 0.25 * (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1));
                        sum += f64::from(c);
                        sq += f64::from(r * r);
                    }
                }
                let cells = (BLOCK * BLOCK) as f64;
                let mean = (sum / cells) as f32;
                // a cell minus the mean of four neighbours has 1.25 times the variance of one
                let variance = (sq / cells / 1.25) as f32;
                if mean < CLIPPED && mean.is_finite() && variance.is_finite() {
                    blocks.push((mean, variance));
                }
                bx += BLOCK;
            }
            by += BLOCK;
        }
    }
    if blocks.is_empty() {
        return (0.0, MIN_VARIANCE);
    }
    blocks.sort_by(|x, y| x.0.total_cmp(&y.0));
    let groups = (blocks.len() / 8).clamp(1, 6);
    let mut points: Vec<(f64, f64)> = Vec::new();
    for g in 0..groups {
        let slice = blocks.get(g * blocks.len() / groups..(g + 1) * blocks.len() / groups).unwrap_or(&[]);
        if slice.is_empty() {
            continue;
        }
        let mean = slice.iter().map(|b| f64::from(b.0)).sum::<f64>() / slice.len() as f64;
        let mut variances: Vec<f32> = slice.iter().map(|b| b.1).collect();
        variances.sort_by(f32::total_cmp);
        let quiet = variances.get(variances.len() / 4).copied().unwrap_or(0.0);
        points.push((mean, f64::from(quiet)));
    }
    let n = points.len() as f64;
    let (sx, sy) = points.iter().fold((0.0, 0.0), |s, p| (s.0 + p.0, s.1 + p.1));
    let (sxx, sxy) = points.iter().fold((0.0, 0.0), |s, p| (s.0 + p.0 * p.0, s.1 + p.0 * p.1));
    let det = n * sxx - sx * sx;
    let (mut a, mut b) =
        if points.len() >= 2 && det.abs() > 1e-12 { ((n * sxy - sx * sy) / det, (sy - (n * sxy - sx * sy) / det * sx) / n) } else { (0.0, sy / n) };
    if a < 0.0 {
        (a, b) = (0.0, sy / n);
    } else if b < 0.0 {
        (a, b) = (if sxx > 0.0 { sxy / sxx } else { 0.0 }, 0.0);
    }
    (a.max(0.0) as f32, (b.max(0.0) as f32).max(MIN_VARIANCE))
}

/// Clean one plane: every cell becomes the average of the cells around it whose 3 × 3 patch looks like its own,
/// judged against the noise its brightness has.
fn non_local_means(src: &[f32], t: usize, a: f32, b: f32) -> Vec<f32> {
    let n = t * t;
    let sigma2: Vec<f32> = box3(src, t).into_iter().map(|m| (a * m.max(0.0) + b).max(MIN_VARIANCE)).collect();
    let (mut num, mut den, mut peak) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    let mut diff = vec![0f32; n];
    let last = t as isize - 1;
    for oy in -SEARCH..=SEARCH {
        for ox in -SEARCH..=SEARCH {
            if ox == 0 && oy == 0 {
                continue;
            }
            let other = |i: usize| -> usize {
                let (x, y) = ((i % t) as isize, (i / t) as isize);
                (y + oy).clamp(0, last) as usize * t + (x + ox).clamp(0, last) as usize
            };
            for (i, d) in diff.iter_mut().enumerate() {
                let v = src.get(i).copied().unwrap_or(0.0) - src.get(other(i)).copied().unwrap_or(0.0);
                *d = v * v;
            }
            let patch = box3(&diff, t);
            for i in 0..n {
                let (Some(&d2), Some(&s2)) = (patch.get(i), sigma2.get(i)) else { continue };
                let w = (-((d2 - 2.0 * s2).max(0.0)) / (STRENGTH * s2)).exp();
                let value = src.get(other(i)).copied().unwrap_or(0.0);
                if let (Some(nu), Some(de), Some(pk)) = (num.get_mut(i), den.get_mut(i), peak.get_mut(i)) {
                    *nu += w * value;
                    *de += w;
                    *pk = pk.max(w);
                }
            }
        }
    }
    // the cell itself counts as much as its best match
    (0..n)
        .map(|i| {
            let own = src.get(i).copied().unwrap_or(0.0);
            let (nu, de, pk) = (num.get(i).copied().unwrap_or(0.0), den.get(i).copied().unwrap_or(0.0), peak.get(i).copied().unwrap_or(0.0));
            let w = pk.max(1e-3);
            (nu + w * own) / (de + w)
        })
        .collect()
}

/// Demosaic the four denoised planes of an RGGB tile into camera RGB planes `[R, G, B]` of `2t × 2t`.
fn demosaic(planes: &[Vec<f32>], t: usize) -> Vec<f32> {
    let s = 2 * t;
    let si = s as isize;
    // the mosaic: R at (even, even), G1 at (odd, even), G2 at (even, odd), B at (odd, odd)
    let mut m = vec![0f32; s * s];
    for (p, plane) in planes.iter().enumerate() {
        let (dx, dy) = (p % 2, p / 2);
        for cy in 0..t {
            for cx in 0..t {
                if let (Some(dst), Some(v)) = (m.get_mut((2 * cy + dy) * s + 2 * cx + dx), plane.get(cy * t + cx)) {
                    *dst = *v;
                }
            }
        }
    }
    let at = |x: isize, y: isize| m.get(reflect(y, s) * s + reflect(x, s)).copied().unwrap_or(0.0);
    // green everywhere: measured where there is a green sample, else interpolated along the smoother direction
    let mut g = vec![0f32; s * s];
    for y in 0..si {
        for x in 0..si {
            let v = if (x + y) % 2 == 1 {
                at(x, y)
            } else {
                let c = at(x, y);
                let (l, r, u, d) = (at(x - 1, y), at(x + 1, y), at(x, y - 1), at(x, y + 1));
                let (gh, gv) =
                    ((l + r) / 2.0 + (2.0 * c - at(x - 2, y) - at(x + 2, y)) / 4.0, (u + d) / 2.0 + (2.0 * c - at(x, y - 2) - at(x, y + 2)) / 4.0);
                let (dh, dv) =
                    ((l - r).abs() + (2.0 * c - at(x - 2, y) - at(x + 2, y)).abs(), (u - d).abs() + (2.0 * c - at(x, y - 2) - at(x, y + 2)).abs());
                let est = if dh < dv {
                    gh
                } else if dv < dh {
                    gv
                } else {
                    (gh + gv) / 2.0
                };
                est.clamp(l.min(r).min(u).min(d), l.max(r).max(u).max(d))
            };
            if let Some(dst) = g.get_mut(y as usize * s + x as usize) {
                *dst = v;
            }
        }
    }
    let gat = |x: isize, y: isize| g.get(reflect(y, s) * s + reflect(x, s)).copied().unwrap_or(0.0);
    // red and blue: the difference from green is smooth, so interpolate it from the nearest samples of the colour
    let chroma = |px: isize, py: isize| -> Vec<f32> {
        let diff = |x: isize, y: isize| at(x, y) - gat(x, y);
        let mut out = vec![0f32; s * s];
        for y in 0..si {
            for x in 0..si {
                let (col, row) = (x % 2 == px, y % 2 == py);
                let d = match (col, row) {
                    (true, true) => diff(x, y),
                    (false, true) => (diff(x - 1, y) + diff(x + 1, y)) / 2.0,
                    (true, false) => (diff(x, y - 1) + diff(x, y + 1)) / 2.0,
                    (false, false) => (diff(x - 1, y - 1) + diff(x + 1, y - 1) + diff(x - 1, y + 1) + diff(x + 1, y + 1)) / 4.0,
                };
                if let Some(dst) = out.get_mut(y as usize * s + x as usize) {
                    *dst = d + gat(x, y);
                }
            }
        }
        out
    };
    let mut out = chroma(0, 0);
    out.extend_from_slice(&g);
    out.extend(chroma(1, 1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic standard normal numbers.
    struct Noise(u64);

    impl Noise {
        fn uniform(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        }

        fn normal(&mut self) -> f32 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
        }
    }

    /// A scene of camera RGB with an edge and a gradient, `s × s`.
    fn scene(s: usize) -> [Vec<f32>; 3] {
        let mut out = [vec![0f32; s * s], vec![0f32; s * s], vec![0f32; s * s]];
        for y in 0..s {
            for x in 0..s {
                let base = 0.15 + 0.5 * x as f32 / s as f32;
                let step = if y > s / 2 { 0.2 } else { 0.0 };
                for (c, plane) in out.iter_mut().enumerate() {
                    plane[y * s + x] = base + step * (1.0 + c as f32 * 0.3) / 1.6;
                }
            }
        }
        out
    }

    /// The scene's RGGB mosaic with noise of variance `a · signal + b`, as the four planes a tile has.
    fn mosaic_planes(truth: &[Vec<f32>; 3], t: usize, a: f32, b: f32, seed: u64) -> Vec<f32> {
        let s = 2 * t;
        let mut rng = Noise(seed);
        let mut planes = vec![0f32; 4 * t * t];
        for (p, &(dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].iter().enumerate() {
            let c = [0, 1, 1, 2][p];
            for cy in 0..t {
                for cx in 0..t {
                    let v = truth[c][(2 * cy + dy) * s + 2 * cx + dx];
                    planes[p * t * t + cy * t + cx] = v + rng.normal() * (a * v + b).sqrt();
                }
            }
        }
        planes
    }

    /// Mean squared error of an RGB result against the scene, away from the tile's edges.
    fn error(out: &[f32], truth: &[Vec<f32>; 3], s: usize) -> f64 {
        let (mut sum, mut n) = (0f64, 0f64);
        for (c, plane) in truth.iter().enumerate() {
            for y in 12..s - 12 {
                for x in 12..s - 12 {
                    let d = f64::from(out[c * s * s + y * s + x] - plane[y * s + x]);
                    sum += d * d;
                    n += 1.0;
                }
            }
        }
        sum / n
    }

    #[test]
    fn a_noisy_tile_comes_out_much_closer_to_the_scene_than_a_plain_demosaic() {
        let (t, a, b) = (64, 0.004, 0.0004);
        let truth = scene(2 * t);
        let input = mosaic_planes(&truth, t, a, b, 7);
        let planes: Vec<Vec<f32>> = input.chunks_exact(t * t).map(<[f32]>::to_vec).collect();
        let plain = error(&demosaic(&planes, t), &truth, 2 * t);
        let cleaned = error(&Builtin.run(&input).unwrap(), &truth, 2 * t);
        assert!(cleaned < plain / 5.0, "plain {plain}, cleaned {cleaned}");
    }

    #[test]
    fn a_noise_free_tile_is_demosaiced_without_blurring_its_edge() {
        let t = 64;
        let truth = scene(2 * t);
        let input = mosaic_planes(&truth, t, 0.0, 0.0, 1);
        let out = Builtin.run(&input).unwrap();
        assert!(error(&out, &truth, 2 * t) < 2e-5);
        // the step between rows 63 and 65 survives
        let s = 2 * t;
        let (above, below) = (out[s * (s / 2 - 2) + s / 2], out[s * (s / 2 + 2) + s / 2]);
        assert!(below - above > 0.9 * 0.2 / 1.6, "{above} {below}");
    }

    #[test]
    fn the_noise_level_is_found_from_the_tile() {
        let (t, a, b) = (128, 0.004, 0.0004);
        let truth = scene(2 * t);
        let input = mosaic_planes(&truth, t, a, b, 3);
        let planes: Vec<&[f32]> = input.chunks_exact(t * t).collect();
        let (ea, eb) = noise_model(&planes, t);
        for signal in [0.2f32, 0.5] {
            let (want, got) = (a * signal + b, ea * signal + eb);
            assert!((got / want - 1.0).abs() < 0.3, "at {signal}: {got} against {want}");
        }
    }

    #[test]
    fn flat_dark_and_clipped_tiles_do_not_break_it() {
        let t = 32;
        for v in [0.0f32, 0.5, 1.0] {
            let out = Builtin.run(&vec![v; 4 * t * t]).unwrap();
            assert_eq!(out.len(), 3 * 4 * t * t);
            assert!(out.iter().all(|x| x.is_finite() && (x - v).abs() < 1e-4), "{v}");
        }
    }

    #[test]
    fn bad_tiles_are_errors_and_bad_numbers_are_cleaned_up() {
        assert!(Builtin.run(&[]).is_err());
        assert!(Builtin.run(&[0.0; 30]).is_err());
        assert!(Builtin.run(&vec![0.0; 4 * 9 * 10]).is_err());
        let mut input = vec![0.3f32; 4 * 16 * 16];
        input[5] = f32::NAN;
        input[77] = f32::INFINITY;
        assert!(Builtin.run(&input).unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn its_manifest_is_valid() {
        let m = manifest();
        crate::manifest::validate(&m).unwrap();
        assert_eq!((m.tile, m.overlap), (TILE, OVERLAP));
    }
}
