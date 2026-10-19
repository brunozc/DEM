"""Binary collision of two identical particles: restitution and time-step convergence tables of the manual.

Run from the repository root with ``python manual/scripts/collision.py``.
"""

from typing import Tuple

from DEMcookies import DEMSimulation, Geometry, GlobalSettings, MaterialProperties, Particle, rayleigh_time_step

RADIUS = 0.01


def collide(restitution: float, relative_speed: float, alpha: float) -> Tuple[float, float, int]:
    """Launch two particles head-on and measure the collision.

    Args:
        restitution: input coefficient of restitution.
        relative_speed: approach speed of the two particles [m/s].
        alpha: time step as a fraction of the Rayleigh time step.

    Returns:
        The measured restitution, the contact duration [s] and the number of time steps in contact.
    """
    material = MaterialProperties(density=2500.0,
                                  youngs_modulus=5.0e6,
                                  poisson_ratio=0.25,
                                  restitution=restitution,
                                  friction=0.5)
    dt = alpha * rayleigh_time_step(material, RADIUS, safety=1.0)
    half = 0.5 * relative_speed
    # the particles start 0.1 mm apart, so the first contact step is resolved like any other
    particles = [
        Particle(0.5 - RADIUS - 5.0e-5, 0.5, RADIUS, vx=half),
        Particle(0.5 + RADIUS + 5.0e-5, 0.5, RADIUS, vx=-half)
    ]
    sim = DEMSimulation(Geometry.rectangle(1.0, 1.0),
                        material,
                        particles=particles,
                        settings=GlobalSettings(gravity=0.0),
                        dt=dt)
    steps = 0
    while True:
        sim.step()
        if 2.0 * RADIUS - (sim.pos[1, 0] - sim.pos[0, 0]) > 0.0:
            steps += 1
        elif steps:
            break
    return abs(sim.vel[1, 0] - sim.vel[0, 0]) / relative_speed, steps * dt, steps


def main() -> None:
    """Print the rows of the restitution and convergence tables."""
    print("restitution table: e_in & v_rel & e_out & error % & t_c ms & steps")
    for restitution in (0.9, 0.7, 0.5, 0.4, 0.2):
        for speed in (0.5, 1.0):
            e_out, duration, steps = collide(restitution, speed, 0.02)
            error = 100.0 * (e_out - restitution) / restitution
            print(f"{restitution:.2f} & {speed:.2f} & {e_out:.4f} & {error:5.2f} & {duration * 1e3:.3f} & {steps}")

    print("convergence table: alpha & dt & steps & e_out & error % relative to the finest")
    results = [(alpha, *collide(0.4, 0.5, alpha)) for alpha in (0.2, 0.1, 0.05, 0.02, 0.01)]
    finest = results[-1][1]
    for alpha, e_out, duration, steps in results:
        dt = duration / steps
        print(f"{alpha:.3f} & {dt:.3e} & {steps} & {e_out:.4f} & {100.0 * (e_out - finest) / finest:.2f}")


if __name__ == "__main__":
    main()
