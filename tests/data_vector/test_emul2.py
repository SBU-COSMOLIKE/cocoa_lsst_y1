"""Advisory checks E1-E4 and hybrid-emulator study E5: the EMUL2 emulated pipelines.

The EXAMPLE_EMUL2 examples replace the Boltzmann code with trained
machine-learning emulators (the emulrdrag, emulbaosn, and emulmps
theory blocks) while the cosmolike likelihood is not emulated. An
emulator approximates the full Boltzmann calculation, so these
accuracy comparisons report their differences without an accuracy cut.
Repeatability still fails on changed numerical results. The report includes:

  - the emulator chi2 against the frozen emulator reference (did the
    installed emulator stack change since the freeze);
  - the emulator chi2 against the non-emulated reference at the same
    cosmology (how closely the emulated pipeline tracks the frozen
    pipeline without the emulator), followed by the printed verdict:
    "agrees at this configuration" when |emulator - non-emulated|
    chi2 < 0.2, "does not agree at this configuration" otherwise;
  - a race check identical in structure to tests 2/4/6/8, required
    to pass the ordinary repeatability tolerance.

  E1. emul2 example1 (cosmic shear): accuracy and agreement verdict.
  E2. emul2 example1: race check, required repeatability.
  E3. emul2 example2 (3x2pt): accuracy and agreement verdict.
  E4. emul2 example2: race check, required repeatability.

E5 uses EXAMPLE_EVALUATE1.yaml for 50 prior points, its bounded Fisher
calculation and 25 focused comparison points in a combined error plot. Its accuracy differences are advisory; numerical
pipeline failures are not. Outputs go in a new timestamped folder under
chains/hybrid_emulator_tests/. This study does not refresh frozen data.

Two caveats a reader should know before trusting the deltas. First,
the emulated configurations sample fewer parameters than the
non-emulated ones (mnu is fixed inside the emulator training), so
part of the difference is a modeling choice, not emulator error. Second, the
trained-network files are read from external_modules/data/emultrf
(pinned by the EMULTRF keys in set_installation_options.sh), not from
the frozen state: retraining or updating the emulators changes these
numbers, and detecting that is part of this file's job. The network
device is frozen to "cpu" so the numbers do not depend on GPU
availability.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lsst_y1/tests/data_vector/test_emul2.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
# E5 may use one thread; the race methods still check their required count.
os.environ.setdefault(
    "OMP_NUM_THREADS", os.environ.get("COCOA_TEST_OMP_THREADS", "4"))

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
# insert(0, ...) puts the folder first in the search order, ahead of
# every other place a same-named module could hide.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The verdict threshold on |emulator - non-emulated| chi2. It equals
# CHI2_TOLERANCE on purpose: the reference tests allow the
# non-emulated pipeline a 0.2 chi2 band, and an emulator whose own
# difference already rivals that band would eat the whole error
# margin of a real analysis.
RECOMMENDATION_LIMIT = u.CHI2_TOLERANCE


class TestEmul2Advisory(unittest.TestCase):
    """Advisory checks E1-E4, sharing one frozen-state verification.

    setUpClass runs once: it moves to ROOTDIR, verifies every frozen
    file against the SHA-256 manifest, and loads the frozen reference
    chi2 values. The manifest check still applies here: the frozen
    emulator configurations are pinned even though the trained-network
    files are not.
    """

    # the classmethod decorator hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        """Move to ROOTDIR, verify the frozen state, load the references."""
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def _accuracy_check(self, name, example, label):
        """Evaluate one emulated configuration and print the verdict.

        Arguments:
          name    = the advisory label (E1 or E3) for the report.
          example = a key of cocoa_test_utils.EXAMPLES with the
                    "emulator" flag.
          label   = one line naming the emulated configuration.

        Returns:
          nothing; the report is printed and nothing is asserted.
        """
        chi2 = u.single_model_chi2(example, tatt=False)
        frozen_ref = self.reference[f"{example}_nla"]
        exact_ref = self.reference[u.EXAMPLES[example]["exact_reference"]]
        u.report_emul2_advisory(f"{name}: {label}", chi2, frozen_ref,
                                exact_ref, RECOMMENDATION_LIMIT)

    def _race_check(self, name, example, label):
        """Require repeatability after changing and restoring emulator inputs.

        Arguments:
          name    = the advisory label (E2 or E4) for the report.
          example = a key of cocoa_test_utils.EXAMPLES with the
                    "emulator" flag.
          label   = one line naming the emulated configuration.

        Returns:
          nothing; a difference above RACE_TOLERANCE fails the test.
        """
        u.assert_omp_threads()
        # the function returns a (fresh, tenth) pair; the assignment
        # unpacks it into the two names
        fresh, tenth = u.ten_in_a_row_chi2(example, tatt=False)
        u.report_emul2_race(f"{name}: {label}", fresh, tenth)
        # Emulator accuracy against CAMB is advisory; identical inputs
        # must still give a repeatable numerical likelihood.
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg="The emulated likelihood changed after returning "
                "to its fiducial point.",
        )

    def test_e1_emul2_example1_accuracy(self):
        """emul2 example1: agreement with the non-emulated chi2, advisory."""
        self._accuracy_check("E1", "emul2_example1",
                             "example1 (cosmic shear, emulated)")

    def test_e2_emul2_example1_race(self):
        """emul2 example1: 10-in-a-row race check, required repeatability."""
        self._race_check("E2", "emul2_example1",
                         "example1 (cosmic shear, emulated) race check")

    def test_e3_emul2_example2_accuracy(self):
        """emul2 example2: agreement with the non-emulated chi2, advisory."""
        self._accuracy_check("E3", "emul2_example2",
                             "example2 (3x2pt, emulated)")

    def test_e4_emul2_example2_race(self):
        """emul2 example2: 10-in-a-row race check, required repeatability."""
        self._race_check("E4", "emul2_example2",
                         "example2 (3x2pt, emulated) race check")


class TestHybridEmulator(unittest.TestCase):
    """E5: EV1 cosmic shear, 50 prior points, Fisher and 25 focused points.

    This uses current example priors, not frozen E1--E4 reference vectors.
    It runs when the candidate code and networks are installed. Broad-prior
    accuracy exceedances and individual nonfinite candidate predictions are
    advisory. Missing CAMB references, no valid comparisons, or incomplete
    Fisher integration or plotting fail the test. No separate opt-in is required.
    """

    def test_e5_hybrid_emulator_fisher(self):
        """Use EV1 for 50 prior points, Fisher and 25 focused comparisons."""
        from datetime import datetime, timezone
        from pathlib import Path
        import subprocess

        project = Path(__file__).resolve().parents[2]
        root = project.parents[1]
        code = Path(os.environ.get(
            "EMULATOR_CODE_DIR", root / "external_modules/code/emulators/emultrf",
        ))
        data = Path(os.environ.get(
            "EMULATOR_DATA_DIR", root / "external_modules/data/emultrf",
        ))
        # The CLI uses the installed linear network and the selected hybrid
        # nonlinear network. Missing installations are a skip, while an
        # installed but broken numerical pipeline is a failed test.
        linear = root / "external_modules/data/emultrf/emulmps/npce_emul"
        nonlinear = data / "emulmps/w0wa_mead2020"
        required = [
            code / "emulmps/emulmps.py",
            code / "emulmps/emulmps_emul/emulmps_w0wa.py",
            linear / "emulator_npce.keras",
            linear / "metadata.joblib",
            nonlinear / "emulator_mead2020.keras",
            nonlinear / "metadata.joblib",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            self.skipTest("Hybrid emulator inputs are not installed: " + ", ".join(missing))

        # Each invocation gets its own output folder. Earlier scientific
        # results are retained, including failed draws and their provenance.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = project / "chains/hybrid_emulator_tests" / (stamp + "_" + str(os.getpid()))
        environment = os.environ.copy()
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        print("E5 hybrid emulator comparison and Fisher figure:", output, flush=True)
        frozen = project / "tests/frozen/emulator_precision"
        prior = frozen / "prior_50.txt"
        focused = frozen / "focused_25.txt"
        # A missing row changes the denominator even when CAMB itself succeeds.
        for path, count in ((prior, 50), (focused, 25)):
            rows = [s for s in path.read_text().splitlines() if s.strip() and not s.startswith("#")]
            self.assertEqual(len(rows), count, str(path))
        result = subprocess.run(
            [sys.executable, str(project / "tests/compare_emulator_camb.py"),
             "--points", str(prior), "--local-points", str(focused),
             "--advisory", "--output", str(output)],
            cwd=root,
            env=environment,
            check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            "Hybrid emulator numerical/Fisher pipeline failed; inspect " + str(output),
        )

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest

if __name__ == "__main__":
    unittest.main(verbosity=2)
