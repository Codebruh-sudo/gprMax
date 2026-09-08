"""Independent sphere and plane-interface references, exp(+j omega t).

Spherical waves are outgoing h_n^(2). Impedance-sphere coefficients follow
the generalized Mie boundary construction of Sihvola et al. (2018),
https://doi.org/10.1103/PhysRevB.98.235417, converted to this time convention.
The bulk-conductor reference uses exponentially scaled Bessel ratios to
avoid overflow inside good conductors (https://dlmf.nist.gov/10.51).
"""

import numpy as np
from scipy.constants import c, epsilon_0, mu_0
from scipy.special import jve, spherical_jn, spherical_yn


ETA0 = np.sqrt(mu_0 / epsilon_0)


def _radial_functions(x):
    if not np.isfinite(x) or x <= 0:
        raise ValueError("sphere size parameter must be positive and finite")
    n = np.arange(1, max(1, int(np.ceil(x + 4*np.cbrt(x) + 2))) + 1)
    j, y = spherical_jn(n, x), spherical_yn(n, x)
    dj = spherical_jn(n, x, derivative=True)
    dy = spherical_yn(n, x, derivative=True)
    return n, x*j, j+x*dj, x*(j-1j*y), j-1j*y+x*(dj-1j*dy)


def impedance_mie_coefficients(x, impedance):
    """Return a_n,b_n for E_t=Z_s(n cross H), with n outward from metal."""
    impedance = complex(impedance)
    if not np.isfinite(impedance) or impedance.real < 0:
        raise ValueError("surface impedance must be finite and passive")
    _, psi, dp, xi, dx = _radial_functions(x)
    zeta = impedance / ETA0
    return (dp-1j*zeta*psi)/(dx-1j*zeta*xi), (psi+1j*zeta*dp)/(xi+1j*zeta*dx)


def conductor_mie_coefficients(x, relative_permittivity):
    """Exact homogeneous nonmagnetic sphere, including displacement current.

    D_n(z)=psi'_n(z)/psi_n(z)=J_(n-1/2)(z)/J_(n+1/2)(z)-n/z.
    Scaling cancels in the ratio, even when unscaled j_n overflows.
    """
    er = complex(relative_permittivity)
    if not np.isfinite(er) or er.real <= 0 or er.imag > 0:
        raise ValueError("relative permittivity must have positive real and nonpositive imaginary parts")
    n, psi, dp, xi, dx = _radial_functions(x)
    m = np.sqrt(er)
    argument = m*x
    derivative = jve(n-0.5, argument)/jve(n+0.5, argument)-n/argument
    return ((m*dp-derivative*psi)/(m*dx-derivative*xi),
            (dp-m*derivative*psi)/(dx-m*derivative*xi))


def mie_amplitudes(coefficients, angles):
    """Dimensionless perpendicular/parallel amplitudes; angles in radians."""
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError("scattering angles must be a finite vector")
    cosine = np.cos(angles)
    s1, s2 = np.zeros(angles.shape, complex), np.zeros(angles.shape, complex)
    previous, current = np.zeros_like(cosine), np.ones_like(cosine)
    for n, (a, b) in enumerate(zip(*coefficients), 1):
        if n == 1:
            pi_n = current
        else:
            pi_n = ((2*n-1)*cosine*current-n*previous)/(n-1)
            previous, current = current, pi_n
        tau = n*cosine*pi_n-(n+1)*previous
        factor = (2*n+1)/(n*(n+1))
        s1 += factor*(a*pi_n+b*tau)
        s2 += factor*(a*tau+b*pi_n)
    return s1, s2


def sphere_reference(frequencies, radius, conductivity, angles, *, impedance=None):
    """Return RCS and Etheta/Eincident in the equatorial, Ez-incidence cut.

    Supplying impedance selects the impedance-sphere solution. Otherwise the
    reference resolves the full bulk-conductor constitutive law analytically.
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if radius <= 0 or conductivity <= 0 or np.any(frequencies <= 0):
        raise ValueError("radius, conductivity, and frequencies must be positive")
    impedances = None if impedance is None else np.broadcast_to(impedance, frequencies.shape)
    amplitudes = []
    for index, frequency in enumerate(frequencies):
        k = 2*np.pi*frequency/c
        coefficients = (conductor_mie_coefficients(k*radius, 1-1j*conductivity/(2*np.pi*frequency*epsilon_0))
                        if impedances is None else impedance_mie_coefficients(k*radius, impedances[index]))
        perpendicular, _ = mie_amplitudes(coefficients, angles)
        amplitudes.append(1j*perpendicular/k)
    amplitudes = np.asarray(amplitudes)
    return 4*np.pi*np.abs(amplitudes)**2, amplitudes


def normal_reflection(surface_impedance, host_relative_permittivity=1):
    """Continuum electric reflection coefficient at the actual wall plane."""
    eta = ETA0/np.sqrt(np.asarray(host_relative_permittivity, dtype=complex))
    return (surface_impedance-eta)/(surface_impedance+eta)


def bulk_discrete_permittivity(frequencies, dt, *, er_inf=1., delta_er=0., tau=80e-12):
    """Closed-form symbol of the single-Debye bulk recurrence used by FDTD.

    This computes the symbol from physical parameters, independently of the
    compiled boundary records. delta_er=0 gives the nondispersive case.
    """
    theta = 2*np.pi*np.asarray(frequencies)*dt
    omega = 2*np.sin(theta/2)/dt
    if not delta_er:
        return np.full(theta.shape, er_inf, complex)
    q, w = -1/tau, delta_er/tau
    f, half = np.exp(q*dt), np.exp(q*dt/2)
    b = (w/q)*(-np.expm1(q*dt))/dt
    zt2 = (w/q)*(-np.expm1(q*dt/2))
    instant = -epsilon_0*zt2/dt
    history = epsilon_0*half*b/(np.exp(1j*theta)-f)
    return er_inf + (2j*np.sin(theta/2)*(instant-history))/(1j*omega*epsilon_0)


def discrete_normal_reflection(frequencies, dt, dl, surface_impedance, *, er_inf=1., delta_er=0., tau=80e-12):
    """Exact flat half-dual-cell reflection and axial propagation symbol.

    surface_impedance is Z_alg, i.e. midpoint E / half-step surface current.
    The retained half-cell Ampere mass cancels the sine part of the staggered
    H samples, leaving Gamma=(Z_alg*cos(k*dl/2)-eta*cos(theta/2))/(...).
    """
    theta = 2*np.pi*np.asarray(frequencies)*dt
    omega = 2*np.sin(theta/2)/dt
    er = bulk_discrete_permittivity(frequencies, dt, er_inf=er_inf, delta_er=delta_er, tau=tau)
    eta = ETA0/np.sqrt(er)
    k = 2*np.arcsin(0.5*dl*omega/c*np.sqrt(er))/dl
    wall = surface_impedance*np.cos(k*dl/2)
    bulk = eta*np.cos(theta/2)
    return (wall-bulk)/(wall+bulk), k
