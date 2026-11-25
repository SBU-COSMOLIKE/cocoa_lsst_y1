"""Unit test: the FoM-noise estimate of cosmolike_notebook_utils.fisher.

Three functions of the shared Fisher module estimate how the numerical
noise of the data vector moves a figure of merit (the method of the
Fisher note of the lssty1xdesi project,
projects/lssty1xdesi/notes/fisher_machinery, section "Derivative
accuracy"):

  get_stencil_noise      measures the noise sigma of one evaluation
                         from the five stencil points (the fourth
                         difference, whitened by the data covariance),
                         the average spurious addition
                         a = (130/144) (sigma/eps)^2 it makes to the
                         diagonal of F, and the five-point derivative;
  get_fom_noise_weights  R_j^2 (the squared multiple correlation of
                         p_j with the pair), sigma_j^2 = (F^-1)_jj and
                         the lever R_j^2 sigma_j^2/eps_j^2;
  predict_fom_noise      the first-order change
                         dlnFoM = 1/2 sum_j R_j^2 a_j sigma_j^2 and the
                         exact ratio FoM(F + diag(a))/FoM(F).

The checks are pure numpy: no cosmolike, no CAMB. They use
  - the 3x3 toy Fisher matrix of the note, whose numbers are worked
    out by hand there (R_o^2 = 1/3, the quadratic form 1/6, the three
    noise cases of its table);
  - a random positive-definite F, for the agreement of the first order
    with the exact ratio as a -> 0 (their difference is second order);
  - a synthetic data vector: a smooth function of three parameters
    plus independent Gaussian noise of known whitened size sigma0 per
    evaluation, with a known correlated covariance. Whitened size means
    sqrt(n^T C^-1 n) for one evaluation's noise n: the noise is drawn
    as n = (sigma0/sqrt(N)) L z, with C = L L^T (Cholesky) and z a
    vector of N independent unit normals, so <n^T C^-1 n> = sigma0^2.

To run (from the Cocoa/ folder, cocoa environment active):

    python -m pytest ./projects/lsst_y1/tests/data_vector/test_fisher_noise.py
"""

import numpy as np
import pytest

# cosmolike_core is on the import path because tests/conftest.py puts it
# there before pytest imports this module
from cosmolike_notebook_utils.fisher import (get_FoM, get_ddv,
                                             get_fom_noise_weights,
                                             get_stencil_noise,
                                             predict_fom_noise)

# The toy Fisher matrix of the note, parameters (w0, wa, o) in this
# order: positive definite, a strong w0-wa anticorrelation and a
# sizeable correlation of o with the pair. Its inverse is
# (1/18) [[11, -10, 3], [-10, 14, -6], [3, -6, 9]].
TOY_F = np.array([
    [5.0, 4.0, 1.0],
    [4.0, 5.0, 2.0],
    [1.0, 2.0, 3.0],
])

# The synthetic data vector: N_DATA entries; three parameters with the
# three step branches of get_ddv: a positive fiducial (relative step),
# a zero fiducial (absolute step) and a negative fiducial (relative
# step whose signed size h p0 is negative)
N_DATA = 240
FIDUCIAL = np.array([0.3, 0.0, -0.9])
# the relative step of the checks: get_ddv's default
STEP = 0.02


def smooth_model_coefficients():
    """Return the fixed coefficients of the smooth synthetic model.

    The model, entry by entry (a = 0 .. N_DATA - 1),

        m_a(p) = A_a exp(c_a p_0) + B_a cos(k_a p_1 + phase_a)
                 + G_a p_2^5 + E_a p_0 p_2,

    is smooth in every parameter, with a nonzero fourth derivative
    along each one, and couples p_0 with p_2 so that F has an
    off-diagonal entry.

    Returns:
      dict of 1D arrays (N_DATA,): "A", "c", "B", "k", "phase", "G", "E".
    """
    # seed 20261117 only fixes the coefficients; any seed works
    rng = np.random.default_rng(seed=20261117)
    coefficients = {
        "A": rng.uniform(low=0.5, high=1.5, size=N_DATA),
        "c": rng.uniform(low=-1.5, high=1.5, size=N_DATA),
        "B": rng.uniform(low=0.2, high=0.8, size=N_DATA),
        "k": rng.uniform(low=0.5, high=2.0, size=N_DATA),
        "phase": rng.uniform(low=0.0, high=2.0*np.pi, size=N_DATA),
        "G": rng.uniform(low=-0.3, high=0.3, size=N_DATA),
        "E": rng.uniform(low=-0.5, high=0.5, size=N_DATA),
    }
    return coefficients


def smooth_model(param, coefficients):
    """Evaluate the smooth synthetic model at one parameter vector.

    Arguments:
      param        = 1D array (3,), the parameters (p_0, p_1, p_2).
      coefficients = the dict of smooth_model_coefficients.

    Returns:
      1D array (N_DATA,).
    """
    exponential = coefficients["A"]*np.exp(coefficients["c"]*param[0])
    oscillation = coefficients["B"]*np.cos(coefficients["k"]*param[1]
                                           + coefficients["phase"])
    power = coefficients["G"]*param[2]**5
    coupling = coefficients["E"]*param[0]*param[2]
    return exponential + oscillation + power + coupling


def synthetic_covariance():
    """Return a correlated data covariance, its inverse and Cholesky factor.

    C = S R S with S = diag(error bars) and R the correlation
    rho^|a - b| of neighbouring entries (rho = 0.6), so that whitening
    differs from dividing by the error bars.

    Returns:
      (covariance, inverse_covariance, cholesky): three (N_DATA, N_DATA)
      arrays; cholesky is the lower-triangular L with C = L L^T.
    """
    # error bars between 2% and 8% of an order-one data vector; seed
    # 7 only fixes them
    rng = np.random.default_rng(seed=7)
    error_bars = rng.uniform(low=0.02, high=0.08, size=N_DATA)
    entry = np.arange(N_DATA)
    distance = np.abs(entry[:, None] - entry[None, :])
    correlation = 0.6**distance
    covariance = error_bars[:, None]*correlation*error_bars[None, :]
    inverse_covariance = np.linalg.inv(covariance)
    cholesky = np.linalg.cholesky(covariance)
    return covariance, inverse_covariance, cholesky


class NoisyModel:
    """A synthetic data-vector function: smooth model plus Gaussian noise.

    Every call draws a fresh noise vector n = (sigma0/sqrt(N_DATA)) L z
    (z standard normal, C = L L^T), so the noise of different
    evaluations is independent and its whitened size is sigma0:
    <n^T C^-1 n> = sigma0^2. With deterministic = True the draw is
    seeded by the bytes of the parameter vector instead, so the same
    parameters give the same noisy vector: a stand-in for numerical
    noise, irregular in the parameters but reproducible. The calls are
    counted and the AccuracyBoost values recorded, so the tests can
    check what the functions under test did.
    """

    def __init__(self, sigma0, cholesky, coefficients, seed=0,
                 deterministic=False):
        """Store the noise size, the covariance factor and the generator.

        Arguments:
          sigma0       = whitened noise of one evaluation (>= 0).
          cholesky     = the (N_DATA, N_DATA) factor L of C = L L^T.
          coefficients = the dict of smooth_model_coefficients.
          seed         = seed of the noise generator (ignored when
                         deterministic is True).
          deterministic = True seeds each draw by the parameters.
        """
        self.sigma0 = sigma0
        self.cholesky = cholesky
        self.coefficients = coefficients
        self.rng = np.random.default_rng(seed=seed)
        self.deterministic = deterministic
        self.calls = 0
        self.boosts = []

    def dv(self, param, AccuracyBoost=1.0):
        """Return the noisy data vector at param (the dv argument form).

        Arguments:
          param         = 1D array (3,) of parameters.
          AccuracyBoost = recorded only; the model has no accuracy knob.

        Returns:
          1D array (N_DATA,).
        """
        self.calls = self.calls + 1
        self.boosts.append(AccuracyBoost)
        if self.deterministic:
            # the 3 float64 parameters as 6 unsigned 32-bit integers:
            # numpy's seed sequence takes such a list as the seed
            words = np.frombuffer(np.asarray(param, dtype=np.float64).tobytes(),
                                  dtype=np.uint32)
            generator = np.random.default_rng(seed=[int(w) for w in words])
        else:
            generator = self.rng
        unit_normals = generator.standard_normal(size=N_DATA)
        noise = (self.sigma0/np.sqrt(N_DATA))*(self.cholesky @ unit_normals)
        return smooth_model(param, self.coefficients) + noise


def test_toy_example_of_the_note():
    """Reproduce the hand-worked toy of the note (3x3 F, pair (w0, wa)).

    With F^-1 = (1/18) [[11, -10, 3], [-10, 14, -6], [3, -6, 9]]:
      - sigma_j^2 = (11/18, 14/18, 1/2);
      - for the other parameter o, (F^-1)_{o,w} = (1/6, -1/3) and the
        quadratic form (F^-1)_{o,w} C_w^-1 (F^-1)_{w,o} = 1/6, so
        R_o^2 = (1/6)/(1/2) = 1/3; R^2 = 1 for w0 and wa themselves;
      - noise a_o = 0.6 on the o column alone: first order
        dlnFoM = 1/2 (1/3)(1/2)(0.6) = 0.05, exact ratio
        (6.50/6)^1/2 = 1.041 (the Schur complement with f_33 = 3.6 has
        determinant 6.5);
      - the other two cases of the note's table: a = (0.5, 0.3, 0)
        gives the exact 1.251 against the first order 1 + 0.269, and
        a = (0.05, 0.05, 0) gives 1.034 against 1 + 0.035.
    """
    steps = np.array([0.1, 0.2, 0.3])
    weights = get_fom_noise_weights(F=TOY_F, i=0, j=1, eps=steps)
    np.testing.assert_allclose(weights["sigma2"], [11/18, 14/18, 0.5],
                               rtol=1e-13)
    np.testing.assert_allclose(weights["R2"], [1.0, 1.0, 1.0/3.0],
                               rtol=1e-13)
    quadratic_form = weights["R2"][2]*weights["sigma2"][2]
    assert quadratic_form == pytest.approx(1.0/6.0, rel=1e-13)
    expected_lever = weights["R2"]*weights["sigma2"]/steps**2
    np.testing.assert_allclose(weights["lever"], expected_lever, rtol=1e-15)

    # the o column alone
    result = predict_fom_noise(F=TOY_F, i=0, j=1, a=[0.0, 0.0, 0.6])
    assert result["dlnFoM"] == pytest.approx(0.05, rel=1e-13)
    np.testing.assert_allclose(result["terms"], [0.0, 0.0, 0.1],
                               rtol=1e-13, atol=1e-15)
    assert result["ratio"] == pytest.approx(np.sqrt(6.5/6.0), rel=1e-13)
    assert round(result["ratio"], 3) == 1.041

    # noise on the pair's own columns: the exact determinant of
    # F + diag(a_w0, a_wa, 0) by cofactor expansion,
    # det F' = 18 + 11 a_w0 + 14 a_wa + 3 a_w0 a_wa, against the two
    # rounded reference values worked out by hand in the note
    cases = (
        (0.5, 0.3, 1.251, 0.269),
        (0.05, 0.05, 1.034, 0.035),
    )
    for a_w0, a_wa, exact_rounded, first_order_rounded in cases:
        result = predict_fom_noise(F=TOY_F, i=0, j=1, a=[a_w0, a_wa, 0.0])
        determinant_ratio = (18.0 + 11.0*a_w0 + 14.0*a_wa
                             + 3.0*a_w0*a_wa)/18.0
        assert result["ratio"] == pytest.approx(np.sqrt(determinant_ratio),
                                                rel=1e-13)
        assert round(result["ratio"], 3) == exact_rounded
        assert round(result["dlnFoM"], 3) == first_order_rounded
    print(f"\n  toy: R_o^2 = {weights['R2'][2]:.6f}, "
          f"first order 0.05 against exact ratio "
          f"{np.sqrt(6.5/6.0):.4f}")


def test_first_order_agrees_with_exact_ratio_at_small_a():
    """The first order and the exact ratio differ at second order in a.

    A random positive-definite 6x6 F, the pair (0, 1), and additions
    a_j = s u_j / sigma_j^2 with fixed positive u_j, so every term
    a_j sigma_j^2 is of order s. Then ln(ratio) - dlnFoM = O(s^2): the
    difference falls a hundredfold per tenfold step in s.
    """
    # seed 11 only fixes the random matrix and directions
    rng = np.random.default_rng(seed=11)
    factor = rng.standard_normal(size=(6, 6))
    fisher = factor @ factor.T + 0.5*np.eye(6)
    directions = rng.uniform(low=0.5, high=1.5, size=6)
    sigma2 = np.diag(np.linalg.inv(fisher))
    differences = []
    for scale in (1e-2, 1e-3, 1e-4):
        additions = scale*directions/sigma2
        result = predict_fom_noise(F=fisher, i=0, j=1, a=additions)
        exact = np.log(result["ratio"])
        differences.append(abs(exact - result["dlnFoM"]))
        # the first order itself is of order s
        assert 0.1*scale < result["dlnFoM"] < 10.0*scale
    for k in range(len(differences) - 1):
        reduction = differences[k + 1]/differences[k]
        assert 0.0095 < reduction < 0.0105, (
            f"|ln ratio - dlnFoM| fell by {reduction} for a tenfold "
            "smaller a; second order predicts 0.01")
    print(f"\n  |ln ratio - dlnFoM| at s = 1e-2, 1e-3, 1e-4: "
          f"{differences[0]:.3e}, {differences[1]:.3e}, {differences[2]:.3e}")


def test_noise_estimate_recovers_sigma0():
    """get_stencil_noise recovers the planted noise size, on average.

    The squared estimate sigma^2 = |L^-1 Delta4 m|^2/70 has the mean
    sigma0^2 for independent noise of whitened size sigma0, as long as
    the smooth part of the fourth difference, eps^4 m'''', is
    negligible; the test first checks that the noise-free estimate is
    below 2% of sigma0 = 1e-2 (a share below 4e-4 of sigma^2). Over 400
    seeds the mean of sigma^2/sigma0^2 must then be 1 within five
    standard errors; the standard error is about
    sqrt(2/N_DATA)/sqrt(400) = 0.5%.
    """
    coefficients = smooth_model_coefficients()
    _, inverse_covariance, cholesky = synthetic_covariance()
    sigma0 = 1e-2
    n_seeds = 400
    smooth = NoisyModel(sigma0=0.0, cholesky=cholesky,
                        coefficients=coefficients)
    for index in range(len(FIDUCIAL)):
        smooth_result = get_stencil_noise(smooth.dv, index=index, h=STEP,
                                          CV=FIDUCIAL, AccuracyBoost=1.0,
                                          invcov=inverse_covariance)
        assert smooth_result["sigma"] < 0.02*sigma0
        ratios = np.zeros(n_seeds)
        for seed in range(n_seeds):
            model = NoisyModel(sigma0=sigma0, cholesky=cholesky,
                               coefficients=coefficients, seed=seed)
            result = get_stencil_noise(model.dv, index=index, h=STEP,
                                       CV=FIDUCIAL, AccuracyBoost=1.0,
                                       invcov=inverse_covariance)
            ratios[seed] = (result["sigma"]/sigma0)**2
        mean = np.mean(ratios)
        standard_error = np.std(ratios, ddof=1)/np.sqrt(n_seeds)
        assert abs(mean - 1.0) < 5.0*standard_error, (
            f"parameter {index}: <sigma^2>/sigma0^2 = {mean} +- "
            f"{standard_error}; it must be 1")
        print(f"\n  parameter {index}: <sigma^2>/sigma0^2 = {mean:.4f} "
              f"+- {standard_error:.4f}")


def test_smooth_vector_gives_eps4_noise():
    """A noise-free smooth vector gives sigma = eps^4 |m''''|/sqrt(70).

    Along p_0 the model is A exp(c p_0) plus terms linear in p_0, so its
    fourth derivative is m'''' = A c^4 exp(c p_0) and
        Delta4 m = eps^4 m'''' + (eps^6/6) m'''''' + ...
    The estimate must equal eps^4 sqrt(m''''^T C^-1 m'''')/sqrt(70) up
    to the relative correction of order eps^2 c^2 (about 1e-5 here),
    and halving the step must divide it by 16.
    """
    coefficients = smooth_model_coefficients()
    _, inverse_covariance, cholesky = synthetic_covariance()
    model = NoisyModel(sigma0=0.0, cholesky=cholesky,
                       coefficients=coefficients)
    estimates = []
    for step in (STEP, 0.5*STEP):
        result = get_stencil_noise(model.dv, index=0, h=step, CV=FIDUCIAL,
                                   AccuracyBoost=1.0,
                                   invcov=inverse_covariance)
        eps = step*FIDUCIAL[0]
        fourth_derivative = (coefficients["A"]*coefficients["c"]**4
                             *np.exp(coefficients["c"]*FIDUCIAL[0]))
        whitened_length = np.sqrt(fourth_derivative @ inverse_covariance
                                  @ fourth_derivative)
        expected = eps**4*whitened_length/np.sqrt(70.0)
        assert result["sigma"] == pytest.approx(expected, rel=1e-3)
        estimates.append(result["sigma"])
    assert estimates[0]/estimates[1] == pytest.approx(16.0, rel=1e-3)
    print(f"\n  smooth vector: sigma = {estimates[0]:.3e} at h = {STEP}, "
          f"ratio {estimates[0]/estimates[1]:.4f} for half the step")


def test_derivative_equals_get_ddv():
    """The returned derivative is get_ddv's, bit for bit, on all branches.

    With a deterministic noisy model (same parameters, same vector), the
    four off-center evaluations of get_stencil_noise and get_ddv see
    the same vectors, and the same arithmetic must give the same
    derivative exactly: for a positive fiducial (relative step), a zero
    fiducial (absolute step) and a negative one. The call makes five
    evaluations, forwards AccuracyBoost to each, returns eps = h |p0|
    (or h) and leaves CV unchanged.
    """
    coefficients = smooth_model_coefficients()
    _, inverse_covariance, cholesky = synthetic_covariance()
    expected_eps = (STEP*0.3, STEP, STEP*0.9)
    for index in range(len(FIDUCIAL)):
        model = NoisyModel(sigma0=1e-3, cholesky=cholesky,
                           coefficients=coefficients, deterministic=True)
        fiducial = FIDUCIAL.copy()
        result = get_stencil_noise(model.dv, index=index, h=STEP,
                                   CV=fiducial, AccuracyBoost=2.5,
                                   invcov=inverse_covariance)
        assert model.calls == 5
        assert model.boosts == [2.5]*5
        np.testing.assert_array_equal(fiducial, FIDUCIAL)
        reference = get_ddv(model.dv, index=index, h=STEP, CV=FIDUCIAL,
                            AccuracyBoost=2.5)
        np.testing.assert_array_equal(result["ddv"], reference)
        assert result["eps"] == pytest.approx(expected_eps[index], rel=1e-15)
        expected_a = (130.0/144.0)*(result["sigma"]/result["eps"])**2
        assert result["a"] == pytest.approx(expected_a, rel=1e-15)


def test_mean_spurious_prior():
    """Noisy columns raise the diagonal of F by a = (130/144) sigma0^2/eps^2.

    Over 1000 seeds, the Fisher matrix of the three noisy five-point
    columns, F = D^T C^-1 D, is compared with the one of the noise-free
    columns (the same stencil on the smooth model, so the eps^4
    truncation cancels). On average the diagonal entry j exceeds the
    noise-free one by a_j = <e_j^T C^-1 e_j> = (130/144) sigma0^2/eps_j^2,
    the 130 = 1 + 64 + 64 + 1 of the squared stencil weights, and the
    off-diagonal entries are unchanged. sigma0 is chosen so that every
    a_j is at least F_jj: the zero-mean cross term 2 d_j^T C^-1 e_j then
    scatters by at most 2/sqrt(N_DATA) of a_j per seed. The mean of
    the a returned by get_stencil_noise (from the estimated sigma) must
    match as well. Five standard errors stay below the 10% gap between
    130/144 and 1, so the check tells the two apart.
    """
    coefficients = smooth_model_coefficients()
    _, inverse_covariance, cholesky = synthetic_covariance()
    smooth = NoisyModel(sigma0=0.0, cholesky=cholesky,
                        coefficients=coefficients)
    n_params = len(FIDUCIAL)
    columns = []
    for index in range(n_params):
        columns.append(get_ddv(smooth.dv, index=index, h=STEP, CV=FIDUCIAL,
                               AccuracyBoost=1.0))
    derivatives_smooth = np.column_stack(columns)
    fisher_smooth = derivatives_smooth.T @ inverse_covariance @ derivatives_smooth
    eps = np.array([STEP*0.3, STEP, STEP*0.9])
    # the smallest sigma0 with a_j >= F_jj for every column
    sigma0 = np.max(np.sqrt(np.diag(fisher_smooth)*144.0/130.0)*eps)
    expected_a = (130.0/144.0)*sigma0**2/eps**2

    n_seeds = 1000
    excess = np.zeros((n_seeds, n_params, n_params))
    estimated_a = np.zeros((n_seeds, n_params))
    for seed in range(n_seeds):
        model = NoisyModel(sigma0=sigma0, cholesky=cholesky,
                           coefficients=coefficients, seed=seed)
        columns = []
        for index in range(n_params):
            result = get_stencil_noise(model.dv, index=index, h=STEP,
                                       CV=FIDUCIAL, AccuracyBoost=1.0,
                                       invcov=inverse_covariance)
            columns.append(result["ddv"])
            estimated_a[seed, index] = result["a"]
        derivatives = np.column_stack(columns)
        fisher = derivatives.T @ inverse_covariance @ derivatives
        excess[seed] = fisher - fisher_smooth

    for index in range(n_params):
        diagonal = excess[:, index, index]/expected_a[index]
        mean = np.mean(diagonal)
        standard_error = np.std(diagonal, ddof=1)/np.sqrt(n_seeds)
        assert 5.0*standard_error < 1.0 - 130.0/144.0
        assert abs(mean - 1.0) < 5.0*standard_error, (
            f"column {index}: <F_jj - F_jj,true>/a_j = {mean} +- "
            f"{standard_error}; it must be 1")
        estimate = estimated_a[:, index]/expected_a[index]
        estimate_error = np.std(estimate, ddof=1)/np.sqrt(n_seeds)
        assert abs(np.mean(estimate) - 1.0) < 5.0*estimate_error
        print(f"\n  column {index}: <F_jj - F_jj,true>/a_j = {mean:.4f} "
              f"+- {standard_error:.4f}; <a_estimated>/a_j = "
              f"{np.mean(estimate):.4f} +- {estimate_error:.4f}")
    # off the diagonal: zero on average, in units of the geometric mean
    # of the two diagonal additions
    for row in range(n_params):
        for col in range(row + 1, n_params):
            scale = np.sqrt(expected_a[row]*expected_a[col])
            off_diagonal = excess[:, row, col]/scale
            standard_error = np.std(off_diagonal, ddof=1)/np.sqrt(n_seeds)
            assert abs(np.mean(off_diagonal)) < 5.0*standard_error


def test_input_validation():
    """Each malformed input stops with one ValueError naming the problem."""
    coefficients = smooth_model_coefficients()
    _, inverse_covariance, cholesky = synthetic_covariance()
    model = NoisyModel(sigma0=0.0, cholesky=cholesky,
                       coefficients=coefficients)
    bad_stencil_calls = (
        ({"CV": None, "invcov": inverse_covariance}, "CV is None"),
        ({"CV": np.ones((3, 1)), "invcov": inverse_covariance}, "1D array"),
        ({"CV": [0.3, np.nan, 1.0], "invcov": inverse_covariance},
         "nonfinite"),
        ({"CV": FIDUCIAL, "invcov": inverse_covariance, "index": 1.0},
         "not an integer"),
        ({"CV": FIDUCIAL, "invcov": inverse_covariance, "index": 3},
         "outside CV"),
        ({"CV": FIDUCIAL, "invcov": inverse_covariance, "h": "0.02"},
         "not a number"),
        ({"CV": FIDUCIAL, "invcov": inverse_covariance, "h": 0.0},
         "finite and > 0"),
        ({"CV": FIDUCIAL, "invcov": None}, "invcov is None"),
        ({"CV": FIDUCIAL, "invcov": np.ones((N_DATA, 3))}, "square"),
        ({"CV": FIDUCIAL, "invcov": np.eye(N_DATA + 1)}, "dv returned shape"),
        ({"CV": FIDUCIAL, "invcov": -np.eye(N_DATA)},
         "not positive semi-definite"),
    )
    for arguments, message in bad_stencil_calls:
        with pytest.raises(ValueError, match=message):
            get_stencil_noise(model.dv, **arguments)

    bad_weight_calls = (
        ({"F": np.ones((3, 2)), "i": 0, "j": 1}, "square"),
        ({"F": np.full((3, 3), np.inf), "i": 0, "j": 1}, "nonfinite"),
        ({"F": TOY_F, "i": 0.0, "j": 1}, "not an"),
        ({"F": TOY_F, "i": 0, "j": 3}, "outside F"),
        ({"F": TOY_F, "i": 1, "j": 1}, "two different"),
        ({"F": -TOY_F, "i": 0, "j": 1}, "not positive definite"),
        ({"F": TOY_F, "i": 0, "j": 1, "eps": [0.1, 0.2]}, "one step"),
        ({"F": TOY_F, "i": 0, "j": 1, "eps": [0.1, 0.0, 0.3]},
         "finite and > 0"),
    )
    for arguments, message in bad_weight_calls:
        with pytest.raises(ValueError, match=message):
            get_fom_noise_weights(**arguments)

    bad_additions = (
        ([0.1, 0.2], "one addition per column"),
        ([0.1, np.nan, 0.3], "nonfinite"),
        ([0.1, -0.2, 0.3], "negative"),
    )
    for additions, message in bad_additions:
        with pytest.raises(ValueError, match=message):
            predict_fom_noise(F=TOY_F, i=0, j=1, a=additions)

    # zero additions leave the FoM unchanged, exactly
    result = predict_fom_noise(F=TOY_F, i=0, j=1, a=np.zeros(3))
    assert result["dlnFoM"] == 0.0
    assert result["ratio"] == 1.0
    assert get_FoM(0, 1, TOY_F) == pytest.approx(np.sqrt(6.0), rel=1e-13)


@pytest.mark.parametrize("real_space", [False, True])
def test_exact_shear_calibration_matches_five_point(real_space):
    """Auto/cross pairs and masked rows agree with independent stencils."""
    from cosmolike_notebook_utils.fisher import shear_calibration_derivatives

    calibration = np.array([0.03, -0.02])
    nrows = 12 if real_space else 6
    baseline = np.arange(1.0, nrows + 1.0)
    baseline[1] = 0.0  # One masked angular entry must stay zero.

    def prediction(values):
        # Two angular entries for each of (0,0), (0,1), (1,1).
        factors = np.repeat([(1 + values[0])**2,
                             (1 + values[0])*(1 + values[1]),
                             (1 + values[1])**2], 2)
        if real_space:
            factors = np.tile(factors, 2)  # Separate xi+ and xi- blocks.
        return np.r_[baseline*factors, np.zeros(3)]

    exact = shear_calibration_derivatives(
        vector=prediction(calibration), calibration=calibration,
        nangular=2, real_space=real_space)
    for index in range(2):
        for step in (0.01, 0.005):
            direction = np.eye(2)[index]*step
            numerical = (-prediction(calibration + 2*direction)
                         + 8*prediction(calibration + direction)
                         - 8*prediction(calibration - direction)
                         + prediction(calibration - 2*direction))/(12*step)
            np.testing.assert_allclose(exact[:, index], numerical,
                                       rtol=1.e-12, atol=1.e-12)


def test_exact_shear_calibration_refuses_other_active_probes():
    """A calibration formula for shear alone must not zero a ggl derivative."""
    from cosmolike_notebook_utils.fisher import shear_calibration_derivatives

    with pytest.raises(ValueError, match="shear-only"):
        shear_calibration_derivatives(
            vector=np.ones(4), calibration=[0.0], nangular=1)
    with pytest.raises(ValueError, match="nonzero 1.m"):
        shear_calibration_derivatives(
            vector=np.zeros(2), calibration=[-1.0], nangular=1)


def test_bounded_fisher_gaussian_proposal_matches_truncated_normal(tmp_path):
    """All bounds apply to a joint Gaussian, including nuisance bounds."""
    from scipy.stats import norm
    from cocoa_testing import sample_matter_emulator_fisher

    np.savez(tmp_path / "fisher.npz", names=["As_1e9", "nuisance"],
             center=[2.0, 0.0], lower=[1.9, -1.5], upper=[2.1, 1.5],
             fisher_fine=np.diag([100.0, 4.0]),
             fisher_coarse=np.diag([100.0, 4.0]))
    result = sample_matter_emulator_fisher(tmp_path, sample_power=14)
    # For a standard Gaussian truncated symmetrically at +/-a,
    # Var = 1 - 2*a*phi(a)/(2*Phi(a)-1), from integrating x^2 phi(x).
    limits = np.array([1.0, 3.0])
    expected = np.array([0.01, 0.25]) * (
        1 - 2*limits*norm.pdf(limits)/(2*norm.cdf(limits)-1))
    np.testing.assert_allclose(np.diag(result["covariance"]), expected,
                               rtol=0.004)
    with np.load(tmp_path / "bounded_samples.npz") as saved:
        assert str(saved["proposal"]) == "joint Gaussian"
        values = saved["values"][saved["weights"] > 0]
        assert np.all((values >= saved["lower"]) & (values <= saved["upper"]))


def test_bounded_fisher_keeps_uniform_null_direction(tmp_path):
    """A proper flat prior remains available when no full Gaussian exists."""
    from cocoa_testing import sample_matter_emulator_fisher

    np.savez(tmp_path / "fisher.npz", names=["As_1e9", "nuisance"],
             center=[2.0, 0.0], lower=[1.0, -2.0], upper=[3.0, 2.0],
             fisher_fine=np.diag([100.0, 0.0]),
             fisher_coarse=np.diag([100.0, 0.0]))
    result = sample_matter_emulator_fisher(tmp_path, sample_power=14)
    np.testing.assert_allclose(result["covariance"][1, 1], 4.0/3.0,
                               rtol=0.005)
    with np.load(tmp_path / "bounded_samples.npz") as saved:
        assert str(saved["proposal"]) == "uniform and conditional Gaussian"


def test_bounded_fisher_gaussian_keeps_coupled_amplitude_cut(tmp_path):
    """The Gaussian proposal must retain a coupled amplitude/bias prior."""
    from cocoa_testing import sample_matter_emulator_fisher

    np.savez(tmp_path / "fisher.npz", names=["As_1e9", "bias"],
             center=[2.0, 1.0], lower=[1.0, 0.5], upper=[3.0, 1.5],
             fisher_fine=np.diag([100.0, 100.0]),
             fisher_coarse=np.diag([100.0, 100.0]),
             amplitude_bias_cut_parameters=[["As_1e9", "bias"]],
             amplitude_bias_cut_scales=[[2.0, 1.0, 1.0]])
    result = sample_matter_emulator_fisher(tmp_path, sample_power=13)
    values = result["values"][result["weights"] > 0]
    assert np.all((values[:, 0]/2.0)*values[:, 1]**2 <= 1.0)
    assert len(values) > 1000
