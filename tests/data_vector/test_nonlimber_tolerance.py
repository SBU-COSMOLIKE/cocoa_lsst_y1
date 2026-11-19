"""Unit test: the non-Limber early-exit tolerance, the FKEM hand-off
and the noise of the low-rank (SVD) engine.

The non-Limber real-space C_l switch a lens bin (gg) or a lens-source
pair (gs) to Limber once they agree with Limber within a tolerance; the
switch multipole moves with the parameters and puts steps into
finite-difference derivatives. init_nonlimber_tolerance sets the
tolerances (core defaults 0.002 for gg and 0.01 for gs; 0 = no early
exit). This project's likelihood keys nonlimber_tolerance_gg and
nonlimber_tolerance_gs are 0: production runs without the early
exit, and tests 2-4 and 8 switch it on with the core defaults. The
shared factory (cosmolike_core/cocoa_testing_suites.py,
nonlimber_tolerance_suite) documents the checks. On example2:

    This project's 3x2pt data vector is in real space: w(theta)
    (and gamma_t where adopt_limber_gs = 0) runs the non-Limber
    path with the early exit, so tests 3 and 4 measure what the
    tolerance does here.

The stencil noise is the whitened fourth difference of five evaluations
(cosmolike_notebook_utils.fisher.get_stencil_noise, step h = 0.025) of
the w0, wa and LSST_B1_5 columns.

At the stored fiducial, the hand-off check compares FKEM caps 150 and
200 with the numerical bounds supplied below. Its one-sided fourth-
difference bound is a regression diagnostic, not a proof that a hand-off
contribution is absent. Smooth fourth-order curvature also contributes.

The SVD comparison sets the same galaxy-shear FKEM cap in both runs:
only the galaxy-clustering engine changes. The measured fourth differences
must satisfy this project's unchanged relative margin and absolute floor.
The controlled comparison and validation are recorded in
`tests/nonlimber_matched_caps_20261119.json`.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lsst_y1/tests/data_vector/test_nonlimber_tolerance.py -s
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = os.environ.get("COCOA_TEST_OMP_THREADS", "4")

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The bounds below come from this project's measurement (the values
# are in the comment of the module docstring above).
TestNonLimberTolerance = u.nonlimber_tolerance_suite(
    u, name="TestNonLimberTolerance", example="example2",
    bias_parameter="LSST_B1_5",
    expected_defaults=(0.0, 0.0),
    change_bound=0.1,
    noise_bound=0.002,
    handoff_margin=0.1,
    svd_rule='not_larger',
    svd_margin=0.15)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
