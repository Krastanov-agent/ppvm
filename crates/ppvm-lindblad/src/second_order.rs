// SPDX-FileCopyrightText: 2026 The PPVM Authors
// SPDX-License-Identifier: Apache-2.0

//! Candidates for [`crate::Admission::SecondOrder`]: every string outside
//! the basis within two applications of `L*`, weighted by its end-of-step
//! coefficient to second order in `dt`,
//! `w_Q = dt·(L*x)_Q + ½dt²·(L*²x)_Q`.
//!
//! Two accumulation passes, both chunked like
//! [`LindbladSpec::leakage_with_prune`]:
//!
//! 1. the basis, in descending `|x_j|`, gives the first-generation leakage
//!    `ℓ_Q = (L*x)_Q` off the basis and the in-basis part `(L*x)|_B`;
//! 2. the parents — basis strings weighted by `(L*x)|_B` and the surviving
//!    first-generation strings weighted by `ℓ_Q`, jointly in descending
//!    weight — add `½dt²·(L*²x)_Q` onto a map seeded with `dt·ℓ_Q`.
//!
//! With `cap = None` every candidate is kept and the weights are exact.
//! With a cap, each map is cut to its `cap` largest entries after every
//! chunk; a string cut early restarts from zero if a later chunk feeds it,
//! and a first-generation string cut in pass 1 can only re-enter through
//! its second-order term (it is then counted as second generation).

use crate::Error;
use crate::basis::{CHUNK_SIZE, build_basis_index};
use crate::spec::LindbladSpec;
use crate::truncate::{cap_map_to_room, order_by_desc_mag};
use crate::word::Word;
use fxhash::{FxBuildHasher, FxHashMap, FxHashSet};
use num::Complex;
use rayon::prelude::*;
use std::time::Instant;

/// Per-string action outputs of one chunk: in-basis `(row, value)` pairs
/// and off-basis, unprotected `(word, value)` pairs.
type Scattered = Vec<(Vec<(u32, f64)>, Vec<(Word, f64)>)>;

/// Result of [`LindbladSpec::second_order_candidates`].
pub(crate) struct SecondOrderCandidates {
    /// `(word, w_Q, is_second_generation)` for every surviving candidate.
    pub cands: Vec<(Word, f64, bool)>,
    /// First-generation candidates after pass 1 (all of them when uncapped).
    pub n_first_gen: usize,
    /// Largest candidate-map size reached in either pass.
    pub peak: usize,
    /// Wall time of the two passes, in microseconds.
    pub pass1_us: u64,
    pub pass2_us: u64,
}

impl LindbladSpec {
    /// Apply `L*` to each weighted string of `chunk` and split the outputs
    /// by basis membership. Protected strings outside the basis are
    /// dropped, as in the first-order leakage. `keep_inside = false` skips
    /// collecting the in-basis part.
    fn scatter_chunk(
        &self,
        chunk: &[(Word, f64)],
        index: &FxHashMap<Word, u32>,
        protected: &FxHashSet<Word>,
        keep_inside: bool,
    ) -> Scattered {
        let n_qubits = self.n_qubits();
        chunk
            .par_iter()
            .map_init(
                || {
                    (
                        Vec::<u32>::with_capacity(n_qubits),
                        Vec::<u32>::with_capacity(128),
                        FxHashMap::<Word, Complex<f64>>::with_capacity_and_hasher(
                            128,
                            FxBuildHasher::default(),
                        ),
                    )
                },
                |(s1, s2, lm), (p, c)| {
                    let terms = self.compute_action_terms(p, s1, s2, lm);
                    let mut inside = Vec::new();
                    let mut outside = Vec::with_capacity(terms.len());
                    for (w, v) in terms.iter() {
                        if let Some(&row) = index.get(w) {
                            if keep_inside {
                                inside.push((row, c * *v));
                            }
                        } else if !protected.contains(w) {
                            outside.push((*w, c * *v));
                        }
                    }
                    (inside, outside)
                },
            )
            .collect()
    }

    /// Second-order admission candidates for the state `(basis, coeffs)`
    /// over a step `dt`; see the module docs. `cap` bounds the live
    /// candidate maps (`None` = exact). Returns nothing when `room == 0`.
    pub(crate) fn second_order_candidates(
        &self,
        basis: &[Word],
        coeffs: &[f64],
        protected: &[Word],
        room: usize,
        dt: f64,
        cap: Option<usize>,
    ) -> Result<SecondOrderCandidates, Error> {
        if basis.len() != coeffs.len() {
            return Err(Error::LengthMismatch {
                what: "basis and coeffs",
                a: basis.len(),
                b: coeffs.len(),
            });
        }
        let mut out = SecondOrderCandidates {
            cands: Vec::new(),
            n_first_gen: 0,
            peak: 0,
            pass1_us: 0,
            pass2_us: 0,
        };
        if room == 0 {
            return Ok(out);
        }
        let cap = cap.unwrap_or(usize::MAX);
        let index = build_basis_index(basis);
        let protected: FxHashSet<Word> = protected.iter().copied().collect();

        // Pass 1: first generation off the basis, and (L*x) on it.
        let t0 = Instant::now();
        let mut y_in = vec![0.0; basis.len()];
        let mut first: FxHashMap<Word, f64> = FxHashMap::default();
        for chunk in order_by_desc_mag(coeffs).chunks(CHUNK_SIZE) {
            let items: Vec<(Word, f64)> = chunk.iter().map(|&i| (basis[i], coeffs[i])).collect();
            for (inside, outside) in self.scatter_chunk(&items, &index, &protected, true) {
                for (row, v) in inside {
                    y_in[row as usize] += v;
                }
                for (w, v) in outside {
                    *first.entry(w).or_insert(0.0) += v;
                }
            }
            out.peak = out.peak.max(first.len());
            cap_map_to_room(&mut first, cap);
        }
        first.retain(|_, v| *v != 0.0);
        out.n_first_gen = first.len();
        out.pass1_us = t0.elapsed().as_micros() as u64;

        // Pass 2: ½dt²·(L*²x) onto a map seeded with dt·ℓ.
        let t0 = Instant::now();
        let half_dt2 = 0.5 * dt * dt;
        let mut parents: Vec<(Word, f64)> = basis
            .iter()
            .zip(&y_in)
            .filter(|(_, y)| **y != 0.0)
            .map(|(w, y)| (*w, *y))
            .collect();
        parents.extend(first.iter().map(|(w, l)| (*w, *l)));
        let weights: Vec<f64> = parents.iter().map(|p| p.1).collect();
        let mut w: FxHashMap<Word, f64> = first.iter().map(|(q, l)| (*q, dt * l)).collect();
        for chunk in order_by_desc_mag(&weights).chunks(CHUNK_SIZE) {
            let items: Vec<(Word, f64)> = chunk.iter().map(|&i| parents[i]).collect();
            for (_, outside) in self.scatter_chunk(&items, &index, &protected, false) {
                for (q, v) in outside {
                    *w.entry(q).or_insert(0.0) += half_dt2 * v;
                }
            }
            out.peak = out.peak.max(w.len());
            cap_map_to_room(&mut w, cap);
        }
        out.cands = w
            .into_iter()
            .filter(|(_, v)| *v != 0.0)
            .map(|(q, v)| {
                let second = !first.contains_key(&q);
                (q, v, second)
            })
            .collect();
        out.pass2_us = t0.elapsed().as_micros() as u64;
        Ok(out)
    }
}
