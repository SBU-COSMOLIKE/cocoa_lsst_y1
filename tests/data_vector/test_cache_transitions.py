"""Check ownership of loaded measurements, baryon PCs and FAST-PT tables.

The optical data vector carries a mask and covariance for one angular
binning. Baryon principal components (PCs) add a correction in that same
layout. The perturbation tables store either the native or the Python
calculation. A notebook can switch any of these settings in one process.
The tests run each sequence in a separate interpreter because the native
library exits the process when the requested state is missing or stale.
"""

from ctypes import CDLL, POINTER, Structure, c_double, c_int, c_void_p, cast
import os
from pathlib import Path
import subprocess
import sys

import pytest


class FastPTTable(Structure):
    """Expose the existing native FPT struct for read-only table checks."""

    _fields_ = [
        ("count", c_int),
        ("internal_count", c_int),
        ("k_range", c_double * 2),
        ("cutoff", c_double),
        ("sigma4", c_double),
        ("table", POINTER(POINTER(c_double))),
        ("internal_table", POINTER(POINTER(c_double))),
    ]


def _table_rows(table, count):
    """Copy the read-only native table before a producer replaces its allocation.

    Arguments:
        table = the FAST-PT struct exported by the native library.
        count = number of spectrum and wavenumber rows.
    Returns:
        A float array with shape (count, number of wavenumbers).
    """
    import numpy as np

    rows = []
    for index in range(count):
        row = np.ctypeslib.as_array(table.table[index], shape=(table.count,))
        rows.append(row.copy())
    return np.array(rows)


def _switch_fastpt(ci, sector, changed_payload):
    """Check that native FAST-PT replaces a Python table at the same cosmology.

    Arguments:
        ci = the populated LSST Y1 compiled interface.
        sector = which table to switch, either intrinsic alignment or galaxy bias.
        changed_payload = whether to perturb one Python spectrum value.
    """
    import numpy as np

    library = CDLL(ci.__file__)
    if sector == "ia":
        symbol = "FPTIA"
        rebuild = library.get_FPT_IA
        read = library.FPT_IA_tables
        install = ci.set_IA_PS
        rows, k_row = 12, 10
    else:
        symbol = "FPTbias"
        rebuild = library.get_FPT_bias
        read = library.FPT_bias_tables
        install = ci.set_bias_PS
        rows, k_row = 8, 6
    rebuild.restype = None
    rebuild.argtypes = []
    read.restype = None
    read.argtypes = []
    rebuild()
    table = FastPTTable.in_dll(library, symbol)
    original = _table_rows(table, rows)
    # The setters take k in h/Mpc and P in (Mpc/h)^3. The C table
    # stores k times c/(100 km/s) and the spectra divided by its cube.
    scale = 2997.92458
    arguments = {
        "kmin": table.k_range[0] / scale,
        "kmax": table.k_range[1] / scale,
        "cutoff": table.cutoff / scale,
        "N": table.count,
    }
    if sector == "bias":
        arguments["sigma4"] = table.sigma4 * scale**3
    payload = original * scale**3
    payload[k_row] = original[k_row] / scale
    if changed_payload:
        # A changed spectrum makes the producer switch visible in the table.
        column = int(np.argmax(np.abs(payload[0])))
        assert payload[0, column] != 0.0
        payload[0, column] *= 1.125
    ci.init_IA(ia_model=1, ia_redshift_evolution=3, ia_code=1)
    address = cast(table.table, c_void_p).value
    install(PS=payload.ravel(), **arguments)
    read()
    if changed_payload:
        assert table.table[0][column] != original[0, column]
    else:
        # Identical input takes the setter's no-allocation path. The Python
        # setter still claims the table even though the address is unchanged.
        assert cast(table.table, c_void_p).value == address
        np.testing.assert_array_equal(_table_rows(table, rows), original)
    rebuild()
    np.testing.assert_array_equal(_table_rows(table, rows), original)
    rebuild()
    np.testing.assert_array_equal(_table_rows(table, rows), original)
    if not changed_payload:
        read()


def _run_sequence(check):
    """Evaluate the real optical likelihood, then exercise one native transition.

    Arguments:
        check = the geometry, data, PCA or FAST-PT transition to run.
    """
    threads = os.environ.get("COCOA_TEST_OMP_THREADS", "4")
    # Cap OpenMP-linked BLAS during import. The survey uses its normal
    # OpenMP team once the library has finished loading.
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    import cosmolike_lsst_y1_interface as ci
    from threadpoolctl import threadpool_info

    os.environ["OMP_NUM_THREADS"] = threads
    ci.set_omp_threads(n=int(threads))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import cocoa_test_utils as u

    u.verify_frozen()
    info = u.load_frozen_info("example2", tatt=False)
    with u.make_model(info) as model:
        point = u.build_point(model, "example2", tatt=False)
        result = model.logposterior(point, cached=False)
        assert np.isfinite(result.logpost)
        assert CDLL(ci.__file__).omp_get_max_threads() == int(threads)
        blas_threads = {
            pool["num_threads"]
            for pool in threadpool_info()
            if pool["user_api"] == "blas"
        }
        assert blas_threads == {1}, blas_threads
        name = u.EXAMPLES["example2"]["likelihood"]
        like = model.likelihood[name]
        baseline = np.asarray(ci.compute_data_vector_masked())
        mask = np.asarray(ci.get_mask(), dtype=bool)
        assert mask.any() and baseline.shape == mask.shape
        if check.startswith(("ia", "bias")):
            sector, _, payload = check.partition("_")
            _switch_fastpt(ci, sector=sector, changed_payload=payload == "changed")
            return
        if check.startswith("geometry_"):
            # Finer quadrature acts on the same measured rows and must
            # leave the loaded mask and covariance usable.
            accuracy = {
                "accuracy_boost": like.accuracyboost,
                "integration_accuracy": int(like.integration_accuracy),
                "integration_accuracy_ss": like.integration_accuracy_ss,
                "integration_accuracy_gs": like.integration_accuracy_gs,
                "integration_accuracy_gg": like.integration_accuracy_gg,
            }
            finer_accuracy = accuracy.copy()
            finer_accuracy["integration_accuracy_ss"] *= 2.0
            ci.init_accuracy_boost(**finer_accuracy)
            prediction = np.asarray(ci.compute_data_vector_masked())
            assert prediction.shape == baseline.shape and np.isfinite(prediction).all()
            assert np.isfinite(ci.compute_chi2(datavector=prediction))
            ci.init_accuracy_boost(**accuracy)
            reference = {
                "ntheta_bins": int(like.ntheta),
                "theta_min_arcmin": like.theta_min_arcmin,
                "theta_max_arcmin": like.theta_max_arcmin,
            }
            shifted = reference.copy()
            shifted["theta_min_arcmin"] *= 1.05
            shifted["theta_max_arcmin"] *= 1.05
            ci.init_binning(**shifted)
            ci.init_binning(**reference)
            # Restoring the original edges keeps the measured rows aligned.
            np.testing.assert_array_equal(ci.compute_data_vector_masked(), baseline)
            assert np.isfinite(ci.compute_chi2(datavector=baseline))
            ci.init_binning(**shifted)
            if check == "geometry_vector":
                ci.compute_data_vector_masked()
            else:
                ci.compute_chi2(datavector=baseline)
            return
        if check == "pca":
            assert not like.use_baryon_pca
            # A known addition makes stale PCs distinguishable from a
            # newly built model's unmodified prediction.
            eigenvectors = np.ones((len(baseline), 1), order="F")
            ci.set_baryon_pcs(eigenvectors=eigenvectors)
            shifted = np.asarray(ci.compute_data_vector_masked_with_baryon_pcs(Q=[1.0]))
            np.testing.assert_array_equal(shifted[mask], baseline[mask] + 1.0)

    if check == "data":
        ci.initial_setup()
        assert np.asarray(ci.get_mask()).size == 0
        assert np.asarray(ci.get_dv_masked()).size == 0
        assert np.asarray(ci.get_cov_masked()).size == 0
        assert np.asarray(ci.get_inv_cov_masked()).size == 0
    else:
        with u.make_model(info) as model:
            result = model.logposterior(point, cached=False)
            assert np.isfinite(result.logpost)
            np.testing.assert_array_equal(ci.compute_data_vector_masked(), baseline)
            ci.compute_data_vector_masked_with_baryon_pcs(Q=[1.0])


@pytest.mark.parametrize(
    ("check", "expected"),
    [
        ("data", ""),
        ("geometry_vector", "the measurement binning changed"),
        ("geometry_chi2", "the measurement binning changed"),
        ("pca", "compute_add_baryons_pcs: baryon PCs not set"),
        ("ia", "call set_IA_PS again"),
        ("bias", "set_bias_PS again"),
        ("ia_changed", ""),
        ("bias_changed", ""),
    ],
)
def test_native_cache_transition(check, expected):
    """Check one restored or refused state in an independent interpreter.

    Arguments:
        check = sequence passed to the worker.
        expected = diagnostic after a native refusal, or empty for normal exit.
    """
    worker = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), check],
        capture_output=True, text=True, timeout=120,
    )
    output = worker.stdout + worker.stderr
    if expected:
        assert worker.returncode == 1 and expected in output, output
    else:
        assert worker.returncode == 0, output


if __name__ == "__main__":
    _run_sequence(sys.argv[1])
