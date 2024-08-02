import sympy as sym
import copy


def do_everything():
    q, u, v = sym.symbols('q u v')

    m1 = sym.Matrix(
        [
            [1, q, u, v],
            [1, q, -u, -v],
            [1, -q, u, -v],
            [1, -q, -u, v]
        ]
    )

    m2 = sym.Matrix(
        [
            [1, -q, -u, -v],
            [1, -q, u, v],
            [1, q, -u, v],
            [1, q, u, -v]
        ]
    )

    I_sun = sym.symbols('I_{sun}')
    Q_sun = sym.symbols('Q_{sun}')
    U_sun = sym.symbols('U_{sun}')
    V_sun = sym.symbols('V_{sun}')

    I_seeing_1, Q_seeing_1, U_seeing_1, V_seeing_1 = sym.symbols('I_{seeing\,1}, Q_{seeing\,1}, U_{seeing\,1}, V_{seeing\,1}')
    I_seeing_2, Q_seeing_2, U_seeing_2, V_seeing_2 = sym.symbols('I_{seeing\,2}, Q_{seeing\,2}, U_{seeing\,2}, V_{seeing\,2}')
    I_seeing_3, Q_seeing_3, U_seeing_3, V_seeing_3 = sym.symbols('I_{seeing\,3}, Q_{seeing\,3}, U_{seeing\,3}, V_{seeing\,3}')
    I_seeing_4, Q_seeing_4, U_seeing_4, V_seeing_4 = sym.symbols('I_{seeing\,4}, Q_{seeing\,4}, U_{seeing\,4}, V_{seeing\,4}')

    without_time_variation = sym.Matrix([
        [
            I_sun,
            Q_sun,
            U_sun,
            V_sun,
        ]
    ]
    ).T

    time_variation_1 = sym.Matrix(
        [
            [I_seeing_1],
            [Q_seeing_1],
            [U_seeing_1],
            [V_seeing_1]
        ]
    )

    time_variation_2 = sym.Matrix(
        [
            [I_seeing_2],
            [Q_seeing_2],
            [U_seeing_2],
            [V_seeing_2]
        ]
    )

    time_variation_3 = sym.Matrix(
        [
            [I_seeing_3],
            [Q_seeing_3],
            [U_seeing_3],
            [V_seeing_3]
        ]
    )

    time_variation_4 = sym.Matrix(
        [
            [I_seeing_4],
            [Q_seeing_4],
            [U_seeing_4],
            [V_seeing_4]
        ]
    )

    tt1_1 = m1 * time_variation_1

    tt1_1 = tt1_1[0]

    tt2_1 = m1 * time_variation_2

    tt2_1 = tt2_1[1]

    tt3_1 = m1 * time_variation_3

    tt3_1 = tt3_1[2]

    tt4_1 = m1 * time_variation_4

    tt4_1 = tt4_1[3]

    time_vector_1 = sym.Matrix(
        [
            [tt1_1],
            [tt2_1],
            [tt3_1],
            [tt4_1]
        ]
    )

    tt1_2 = m2 * time_variation_1

    tt1_2 = tt1_2[0]

    tt2_2 = m2 * time_variation_2

    tt2_2 = tt2_2[1]

    tt3_2 = m2 * time_variation_3

    tt3_2 = tt3_2[2]

    tt4_2 = m2 * time_variation_4

    tt4_2 = tt4_2[3]

    time_vector_2 = sym.Matrix(
        [
            [tt1_2],
            [tt2_2],
            [tt3_2],
            [tt4_2]
        ]
    )

    S_obs_top = m1 * without_time_variation + time_vector_1

    S_obs_bot = m2 * without_time_variation + time_vector_2

    res_1 = sym.simplify(m1.inv() * S_obs_top)

    res_2 = sym.simplify(m2.inv() * S_obs_bot)
