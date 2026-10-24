//! Block-matching collaborative filtering (after Dabov, Foi, Katkovnik and Egiazarian, "Image denoising by sparse 3-D
//! transform-domain collaborative filtering", 2007; the colour variant is their CBM3D), written from the paper.
//!
//! It works on the same planes as [`crate::nlm`]: four Bayer planes of unit noise (see [`crate::nlm::Stabiliser`]),
//! rotated by a Hadamard transform into a brightness plane and three colour-difference planes. Blocks of the
//! brightness plane that look alike are stacked into a group; the stack is transformed (2-D DCT of each block, Haar
//! across the stack) where the picture is sparse and the noise is not, shrunk, transformed back and put back with
//! weights. The first step shrinks by a hard threshold and gives a basic estimate; the second matches blocks on that
//! cleaner estimate and shrinks by a Wiener filter built from it. Every plane of a group follows the brightness
//! plane's matches, so the colour differences are only averaged across like brightness.

use crate::nlm::hadamard;
use std::f32::consts::FRAC_1_SQRT_2;

/// What the filter may be tuned by (the defaults are what the built-in denoiser uses).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Side of a block, in cells.
    pub block: usize,
    /// Distance, in cells, between the blocks that start a group (1 = every block; larger is faster).
    pub step: usize,
    /// How far, in cells, a match is looked for round a block.
    pub search: usize,
    /// The most blocks in a group (rounded down to a power of two).
    pub group: usize,
    /// How far above the noise two blocks may differ (mean square, in units of the noise variance) and still be
    /// grouped, in the first step (on the noisy picture) and in the second (on the basic estimate).
    pub tau1: f32,
    pub tau2: f32,
    /// Hard-threshold level of the first step, in noise deviations, for a light and for a heavily noisy photo (see
    /// [`crate::nlm::Tuning::strength`]): a lightly noisy photo has more fine detail to keep per unit of noise.
    pub lambda: f32,
    pub lambda_heavy: f32,
    /// The noise's deviation as the filter takes it (1 = as measured; more cleans harder).
    pub sigma: f32,
    /// Whether the second (Wiener) step runs.
    pub wiener: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params { block: 4, step: 3, search: 8, group: 16, tau1: 4.0, tau2: 1.0, lambda: 1.9, lambda_heavy: 3.0, sigma: 1.0, wiener: true }
    }
}

/// The orthonormal DCT-II of a block and the window the filtered blocks are put back with.
struct Transform {
    n: usize,
    c: Vec<f32>,
    window: Vec<f32>,
}

/// The zeroth-order modified Bessel function of the first kind (a series; the Kaiser window needs it).
fn bessel_i0(x: f32) -> f32 {
    let (mut sum, mut term) = (1.0f32, 1.0f32);
    for k in 1..30 {
        term *= x * x / (4.0 * (k * k) as f32);
        sum += term;
    }
    sum
}

impl Transform {
    fn new(n: usize) -> Self {
        let mut c = vec![0f32; n * n];
        for (k, row) in c.chunks_exact_mut(n).enumerate() {
            let scale = if k == 0 { (1.0 / n as f32).sqrt() } else { (2.0 / n as f32).sqrt() };
            for (i, v) in row.iter_mut().enumerate() {
                *v = scale * (std::f32::consts::PI * (2 * i + 1) as f32 * k as f32 / (2 * n) as f32).cos();
            }
        }
        // a Kaiser window (beta 2) down each side, so a block's edge counts less than its middle
        let beta = 2.0f32;
        let side: Vec<f32> = (0..n)
            .map(|i| {
                let r = if n > 1 { 2.0 * i as f32 / (n - 1) as f32 - 1.0 } else { 0.0 };
                bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / bessel_i0(beta)
            })
            .collect();
        let window = side.iter().flat_map(|y| side.iter().map(move |x| x * y)).collect();
        Transform { n, c, window }
    }

    /// `blk` (`n × n`) becomes its 2-D DCT.
    fn forward(&self, blk: &mut [f32], tmp: &mut [f32]) {
        let n = self.n;
        for (trow, crow) in tmp.chunks_exact_mut(n).zip(self.c.chunks_exact(n)) {
            trow.fill(0.0);
            for (&ck, brow) in crow.iter().zip(blk.chunks_exact(n)) {
                for (t, &b) in trow.iter_mut().zip(brow) {
                    *t += ck * b;
                }
            }
        }
        for (orow, trow) in blk.chunks_exact_mut(n).zip(tmp.chunks_exact(n)) {
            for (o, crow) in orow.iter_mut().zip(self.c.chunks_exact(n)) {
                *o = trow.iter().zip(crow).map(|(a, b)| a * b).sum();
            }
        }
    }

    /// The inverse of [`Transform::forward`].
    fn inverse(&self, blk: &mut [f32], tmp: &mut [f32]) {
        let n = self.n;
        tmp.fill(0.0);
        for (crow, xrow) in self.c.chunks_exact(n).zip(blk.chunks_exact(n)) {
            for (trow, &cki) in tmp.chunks_exact_mut(n).zip(crow) {
                for (t, &x) in trow.iter_mut().zip(xrow) {
                    *t += cki * x;
                }
            }
        }
        for (brow, trow) in blk.chunks_exact_mut(n).zip(tmp.chunks_exact(n)) {
            brow.fill(0.0);
            for (&t, crow) in trow.iter().zip(self.c.chunks_exact(n)) {
                for (b, &c) in brow.iter_mut().zip(crow) {
                    *b += t * c;
                }
            }
        }
    }
}

/// The orthonormal Haar transform of a vector whose length is a power of two.
fn haar_forward(v: &mut [f32], tmp: &mut [f32]) {
    let mut len = v.len();
    while len > 1 {
        let half = len / 2;
        for i in 0..half {
            let (a, b) = (v.get(2 * i).copied().unwrap_or(0.0), v.get(2 * i + 1).copied().unwrap_or(0.0));
            if let Some(x) = tmp.get_mut(i) {
                *x = (a + b) * FRAC_1_SQRT_2;
            }
            if let Some(x) = tmp.get_mut(half + i) {
                *x = (a - b) * FRAC_1_SQRT_2;
            }
        }
        if let (Some(dst), Some(src)) = (v.get_mut(..len), tmp.get(..len)) {
            dst.copy_from_slice(src);
        }
        len = half;
    }
}

fn haar_inverse(v: &mut [f32], tmp: &mut [f32]) {
    let mut len = 2;
    while len <= v.len() {
        let half = len / 2;
        for i in 0..half {
            let (avg, det) = (v.get(i).copied().unwrap_or(0.0), v.get(half + i).copied().unwrap_or(0.0));
            if let Some(x) = tmp.get_mut(2 * i) {
                *x = (avg + det) * FRAC_1_SQRT_2;
            }
            if let Some(x) = tmp.get_mut(2 * i + 1) {
                *x = (avg - det) * FRAC_1_SQRT_2;
            }
        }
        if let (Some(dst), Some(src)) = (v.get_mut(..len), tmp.get(..len)) {
            dst.copy_from_slice(src);
        }
        len *= 2;
    }
}

/// Reusable buffers of the group transforms.
#[derive(Default)]
struct Scratch {
    tmp: Vec<f32>,
    col: Vec<f32>,
    col_tmp: Vec<f32>,
}

/// A group (`m` blocks of `n × n`, one after the other in `buf`) to the transform domain, or back.
fn group_transform(tf: &Transform, buf: &mut [f32], m: usize, s: &mut Scratch, forward: bool) {
    let nn = tf.n * tf.n;
    s.tmp.resize(nn, 0.0);
    if !forward {
        haar_columns(buf, m, nn, s, false);
    }
    for blk in buf.chunks_exact_mut(nn) {
        if forward {
            tf.forward(blk, &mut s.tmp);
        } else {
            tf.inverse(blk, &mut s.tmp);
        }
    }
    if forward {
        haar_columns(buf, m, nn, s, true);
    }
}

/// The Haar transform across the blocks of a group, at each of the `nn` positions of a block.
fn haar_columns(buf: &mut [f32], m: usize, nn: usize, s: &mut Scratch, forward: bool) {
    if m < 2 {
        return;
    }
    s.col.resize(m, 0.0);
    s.col_tmp.resize(m, 0.0);
    for k in 0..nn {
        for (i, c) in s.col.iter_mut().enumerate() {
            *c = buf.get(i * nn + k).copied().unwrap_or(0.0);
        }
        if forward {
            haar_forward(&mut s.col, &mut s.col_tmp);
        } else {
            haar_inverse(&mut s.col, &mut s.col_tmp);
        }
        for (i, c) in s.col.iter().enumerate() {
            if let Some(b) = buf.get_mut(i * nn + k) {
                *b = *c;
            }
        }
    }
}

/// Copy the blocks of a group out of a plane (`members` are the blocks' top-left cells).
pub(crate) fn gather(plane: &[f32], t: usize, n: usize, members: &[u32], out: &mut Vec<f32>) {
    out.clear();
    for &pos in members {
        for j in 0..n {
            let at = pos as usize + j * t;
            match plane.get(at..at + n) {
                Some(row) => out.extend_from_slice(row),
                None => out.extend(std::iter::repeat_n(0.0, n)),
            }
        }
    }
}

/// Where a plane's blocks start along one side: every `step` cells, and the last one too, so the edge is covered.
pub fn starts(blocks: usize, step: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..blocks).step_by(step.max(1)).collect();
    if let Some(last) = blocks.checked_sub(1)
        && v.last() != Some(&last)
    {
        v.push(last);
    }
    v
}

/// Insert `(distance, block)` into a list ordered by distance, keeping at most `cap`.
fn insert(list: &mut Vec<(f32, u32)>, cap: usize, d: f32, pos: u32) {
    if list.len() >= cap && list.last().is_some_and(|l| d >= l.0) {
        return;
    }
    let at = list.partition_point(|e| e.0 <= d);
    list.insert(at, (d, pos));
    list.truncate(cap);
}

/// For every block that starts a group, the blocks of `plane` within `search` cells (itself first) whose mean squared
/// difference from it is at most `tau`, nearest first and at most `cap`. Blocks (of `n × n` cells) are named by their
/// top-left cell, and a block starts a group every `step` cells.
pub(crate) fn match_groups(plane: &[f32], t: usize, n: usize, step: usize, search: usize, cap: usize, tau: f32) -> Vec<Vec<u32>> {
    let nb = t + 1 - n;
    let refs = starts(nb, step);
    let nr = refs.len();
    let cap = cap.max(1);
    let mut lists: Vec<Vec<(f32, u32)>> = Vec::with_capacity(nr * nr);
    for &ry in &refs {
        for &rx in &refs {
            let mut l = Vec::with_capacity(cap + 1);
            l.push((0.0, (ry * t + rx) as u32));
            lists.push(l);
        }
    }
    let norm = 1.0 / (n * n) as f32;
    let r = search as isize;
    let (mut d, mut hsum) = (vec![0f32; t * t], vec![0f32; t * nr]);
    let valid = |v: usize, o: isize| (v as isize + o) >= 0 && ((v as isize + o) as usize) < nb;
    for oy in -r..=r {
        for ox in -r..=r {
            if ox == 0 && oy == 0 {
                continue;
            }
            // squared differences between the plane and itself shifted by the offset
            let (x0, x1) = (0.max(-ox) as usize, (t as isize).min(t as isize - ox) as usize);
            for y in 0..t {
                let sy = y as isize + oy;
                if sy < 0 || sy >= t as isize {
                    continue;
                }
                let (Some(a), Some(b), Some(out)) = (
                    plane.get(y * t + x0..y * t + x1),
                    plane.get(sy as usize * t + (x0 as isize + ox) as usize..sy as usize * t + (x1 as isize + ox) as usize),
                    d.get_mut(y * t + x0..y * t + x1),
                ) else {
                    continue;
                };
                for ((o, &a), &b) in out.iter_mut().zip(a).zip(b) {
                    *o = (a - b) * (a - b);
                }
            }
            // block sums: along rows at the starting columns, then down at the starting rows
            for y in 0..t {
                for (ri, &rx) in refs.iter().enumerate() {
                    if !valid(rx, ox) {
                        continue;
                    }
                    let s: f32 = d.get(y * t + rx..y * t + rx + n).map_or(0.0, |v| v.iter().sum());
                    if let Some(h) = hsum.get_mut(y * nr + ri) {
                        *h = s;
                    }
                }
            }
            for (rj, &ry) in refs.iter().enumerate() {
                if !valid(ry, oy) {
                    continue;
                }
                for (ri, &rx) in refs.iter().enumerate() {
                    if !valid(rx, ox) {
                        continue;
                    }
                    let s: f32 = (0..n).map(|j| hsum.get((ry + j) * nr + ri).copied().unwrap_or(0.0)).sum::<f32>() * norm;
                    if s <= tau {
                        let other = ((ry as isize + oy) as usize * t) as isize + (rx as isize + ox);
                        if let Some(l) = lists.get_mut(rj * nr + ri) {
                            insert(l, cap, s, other as u32);
                        }
                    }
                }
            }
        }
    }
    lists.into_iter().map(|l| l.into_iter().map(|(_, pos)| pos).collect()).collect()
}

/// [`match_groups`] with each group cut to a power of two, so the Haar transform across it is complete.
fn match_blocks(plane: &[f32], t: usize, p: &Params, tau: f32) -> Vec<Vec<u32>> {
    let mut groups = match_groups(plane, t, p.block, p.step, p.search, p.group, tau);
    for g in &mut groups {
        let m = if g.is_empty() { 0 } else { 1usize << g.len().ilog2() };
        g.truncate(m);
    }
    groups
}

/// Filter every plane of `y` over the groups: hard threshold when there is no `pilot`, Wiener shrinkage by it when
/// there is.
fn collaborate(y: &[Vec<f32>], pilot: Option<&[Vec<f32>]>, groups: &[Vec<u32>], t: usize, p: &Params, lambda: f32, tf: &Transform) -> Vec<Vec<f32>> {
    let n = tf.n;
    let sigma = p.sigma.max(1e-3);
    let lambda = lambda * sigma;
    let mut s = Scratch::default();
    let (mut buf, mut pbuf) = (Vec::new(), Vec::new());
    let mut out = Vec::with_capacity(y.len());
    for plane in y {
        let mut num = vec![0f32; t * t];
        let mut den = vec![0f32; t * t];
        let c = out.len();
        let pilot_plane = pilot.and_then(|pl| pl.get(c));
        for members in groups {
            let m = members.len();
            gather(plane, t, n, members, &mut buf);
            group_transform(tf, &mut buf, m, &mut s, true);
            let weight = match pilot_plane {
                None => {
                    let mut kept = 0usize;
                    for v in buf.iter_mut() {
                        if v.abs() < lambda {
                            *v = 0.0;
                        } else {
                            kept += 1;
                        }
                    }
                    1.0 / kept.max(1) as f32
                }
                Some(pl) => {
                    gather(pl, t, n, members, &mut pbuf);
                    group_transform(tf, &mut pbuf, m, &mut s, true);
                    let mut energy = 0f32;
                    for (v, &q) in buf.iter_mut().zip(&pbuf) {
                        let w = q * q / (q * q + sigma * sigma);
                        *v *= w;
                        energy += w * w;
                    }
                    1.0 / energy.max(1e-3)
                }
            };
            group_transform(tf, &mut buf, m, &mut s, false);
            for (&pos, blk) in members.iter().zip(buf.chunks_exact(n * n)) {
                for (j, row) in blk.chunks_exact(n).enumerate() {
                    let at = pos as usize + j * t;
                    let (Some(nu), Some(de)) = (num.get_mut(at..at + n), den.get_mut(at..at + n)) else { continue };
                    let win = tf.window.get(j * n..(j + 1) * n).unwrap_or_default();
                    for (((nu, de), &v), &w) in nu.iter_mut().zip(de.iter_mut()).zip(row).zip(win) {
                        *nu += weight * w * v;
                        *de += weight * w;
                    }
                }
            }
        }
        // cells no block reached keep their value
        out.push(num.iter().zip(&den).zip(plane).map(|((&nu, &de), &own)| if de > 0.0 { nu / de } else { own }).collect());
    }
    out
}

/// What a run over tiles of `t` cells uses once the parameters are made safe: shared by every device that runs the filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub params: Params,
    /// The first step's hard threshold, in noise deviations (before `params.sigma` scales it).
    pub lambda: f32,
}

/// The parameters as they are used for tiles of `t` cells and a photo of noise `strength` (0 light to 1 heavy), or
/// `None` when the tiles are smaller than a block (they are then left as they are).
pub fn plan(p: &Params, t: usize, strength: f32) -> Option<Plan> {
    let n = p.block.clamp(2, 16);
    if t < n {
        return None;
    }
    let params = Params { block: n, search: p.search.min(t), group: p.group.clamp(1, 64), ..*p };
    let k = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 0.0 };
    Some(Plan { params, lambda: p.lambda + (p.lambda_heavy - p.lambda) * k })
}

/// The block-to-block DCT matrix (`n × n`, row `k` is frequency `k`) and the window (`n × n`) the filtered blocks are put
/// back with: the constants the graphics-card version is given so that both use the very same numbers.
pub fn constants(n: usize) -> (Vec<f32>, Vec<f32>) {
    let tf = Transform::new(n.clamp(2, 16));
    (tf.c, tf.window)
}

/// Denoise four unit-noise planes of `t × t` cells (see the module description); `strength` is the photo's noise
/// from 0 (light) to 1 (heavy), see [`crate::nlm::Tuning::strength`].
pub fn filter(planes: &[Vec<f32>], t: usize, p: &Params, strength: f32) -> Vec<Vec<f32>> {
    let y = hadamard(planes);
    let Some(Plan { params: p, lambda }) = plan(p, t, strength) else { return hadamard(&y) };
    if y.len() != 4 || y.iter().any(|pl| pl.len() != t * t) {
        return hadamard(&y);
    }
    let tf = Transform::new(p.block);
    let var = p.sigma.max(1e-3).powi(2);
    let Some(brightness) = y.first() else { return hadamard(&y) };
    let first = match_blocks(brightness, t, &p, p.tau1 * var);
    let basic = collaborate(&y, None, &first, t, &p, lambda, &tf);
    if !p.wiener {
        return hadamard(&basic);
    }
    let Some(pilot) = basic.first() else { return hadamard(&basic) };
    let second = match_blocks(pilot, t, &p, p.tau2 * var);
    hadamard(&collaborate(&y, Some(&basic), &second, t, &p, lambda, &tf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dct_and_haar_transforms_invert() {
        for n in [2, 4, 8] {
            let tf = Transform::new(n);
            let block: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.7).sin() * 3.0).collect();
            let (mut b, mut tmp) = (block.clone(), vec![0f32; n * n]);
            tf.forward(&mut b, &mut tmp);
            let energy = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
            assert!((energy(&b) - energy(&block)).abs() < 1e-3, "the DCT keeps energy");
            tf.inverse(&mut b, &mut tmp);
            assert!(b.iter().zip(&block).all(|(a, c)| (a - c).abs() < 1e-4));
        }
        for m in [1usize, 2, 4, 16] {
            let v: Vec<f32> = (0..m).map(|i| i as f32 * 0.3 - 1.0).collect();
            let (mut w, mut tmp) = (v.clone(), vec![0f32; m]);
            haar_forward(&mut w, &mut tmp);
            assert!((w.iter().map(|x| x * x).sum::<f32>() - v.iter().map(|x| x * x).sum::<f32>()).abs() < 1e-4);
            haar_inverse(&mut w, &mut tmp);
            assert!(w.iter().zip(&v).all(|(a, c)| (a - c).abs() < 1e-5));
        }
    }

    /// Four planes of unit noise (an LCG and Box-Muller, so the tests need no dependency) round `level`.
    fn noisy(t: usize, level: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut uniform = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        };
        (0..4)
            .map(|_| {
                (0..t * t)
                    .map(|_| {
                        let (u, v) = (uniform(), uniform());
                        level + (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
                    })
                    .collect()
            })
            .collect()
    }

    fn spread(planes: &[Vec<f32>], level: f32) -> f32 {
        let all: Vec<f32> = planes.iter().flatten().copied().collect();
        (all.iter().map(|v| (v - level) * (v - level)).sum::<f32>() / all.len().max(1) as f32).sqrt()
    }

    #[test]
    fn a_flat_noisy_picture_comes_out_much_quieter() {
        let t = 48;
        let planes = noisy(t, 5.0, 7);
        let out = filter(&planes, t, &Params::default(), 0.5);
        // the Hadamard transform keeps the level, so what is left of the spread is the noise that got through
        assert!(spread(&out, 5.0) < 0.35 * spread(&planes, 5.0), "{} vs {}", spread(&out, 5.0), spread(&planes, 5.0));
        assert!(out.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn a_step_edge_stays_sharp() {
        let t = 48;
        let mut planes = noisy(t, 0.0, 9);
        for pl in &mut planes {
            for (i, v) in pl.iter_mut().enumerate() {
                if i % t >= t / 2 {
                    *v += 12.0;
                }
            }
        }
        let out = filter(&planes, t, &Params::default(), 0.5);
        let row = t / 2;
        let at = |x: usize| out.iter().map(|pl| pl.get(row * t + x).copied().unwrap_or(0.0)).sum::<f32>() / 4.0;
        assert!(at(t / 2 - 3) < 2.0 && at(t / 2 + 2) > 10.0, "{} {}", at(t / 2 - 3), at(t / 2 + 2));
    }

    #[test]
    fn odd_inputs_never_panic() {
        // the sizes and values a tile or a caller could hand over
        for t in [0usize, 1, 3, 4, 5, 9] {
            let planes = noisy(t.max(1), 1.0, 3);
            let out = filter(&planes, t, &Params::default(), 0.5);
            assert!(out.iter().flatten().all(|v| v.is_finite()));
        }
        let planes = vec![vec![f32::NAN; 64]; 4];
        let _ = filter(&planes, 8, &Params::default(), 0.5);
        let wild = Params { block: 0, step: 0, search: 1000, group: 0, ..Params::default() };
        assert!(filter(&noisy(16, 2.0, 1), 16, &wild, 0.5).iter().flatten().all(|v| v.is_finite()));
        assert!(filter(&[], 8, &Params::default(), 0.5).iter().all(|p| p.is_empty()));
    }
}
