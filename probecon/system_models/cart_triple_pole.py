import sympy as sp
import symbtools as st
from symbtools import modeltools as mt

from probecon.system_models.core import StateSpaceEnv



def modeling():
    t = sp.Symbol('t') # time
    params = sp.symbols('m0, m1, m2, m3, J1, J2, J3, a1, a2, a3, l1, l2, l3, g, d0, d1, d2, d3') # system parameters
    m0, m1, m2, m3, J1, J2, J3, a1, a2, a3, l1, l2, l3, g, d0, d1, d2, d3 = params

    # force
    F = sp.Symbol('F')

    # generalized coordinates
    q0_t = sp.Function('q0')(t)
    dq0_t = q0_t.diff(t)
    ddq0_t = q0_t.diff(t, 2)
    q1_t = sp.Function('q1')(t)
    dq1_t = q1_t.diff(t)
    ddq1_t = q1_t.diff(t, 2)
    q2_t = sp.Function('q2')(t)
    dq2_t = q2_t.diff(t)
    ddq2_t = q2_t.diff(t, 2)
    q3_t = sp.Function('q3')(t)
    dq3_t = q3_t.diff(t)
    ddq3_t = q3_t.diff(t, 2)

    # position vectors
    p0 = sp.Matrix([q0_t, 0])
    p1 = sp.Matrix([q0_t - a1*sp.sin(q1_t), a1*sp.cos(q1_t)])
    p2 = sp.Matrix([q0_t - l1*sp.sin(q1_t) - a2*sp.sin(q2_t), l1*sp.cos(q1_t) + a2*sp.cos(q2_t)])
    p3 = sp.Matrix([q0_t - l1*sp.sin(q1_t) - l2*sp.sin(q2_t) - a3*sp.sin(q3_t),
                    l1*sp.cos(q1_t) + l2*sp.cos(q2_t) + a3*sp.cos(q3_t)])

    # velocity vectors
    dp0 = p0.diff(t)
    dp1 = p1.diff(t)
    dp2 = p2.diff(t)
    dp3 = p3.diff(t)

    # kinetic energy T
    T0 = m0/2*(dp0.T*dp0)[0]
    T1 = (m1*(dp1.T*dp1)[0] + J1*dq1_t**2)/2
    T2 = (m2*(dp2.T*dp2)[0] + J2*dq2_t**2)/2
    T3 = (m3*(dp3.T*dp3)[0] + J3*dq3_t**2)/2
    T = T0 + T1 + T2 + T3

    # potential energy V
    V = m1*g*p1[1] + m2*g*p2[1] + m3*g*p3[1]

    # lagrangian L
    L = T - V

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i = Q_i

    Q0 = F - d0*dq0_t
    Q1 =   - d1*dq1_t + d2*(dq2_t - dq1_t)
    Q2 =   - d2*(dq2_t - dq1_t) + d3*(dq3_t - dq2_t)
    Q3 =   - d3*(dq3_t - dq2_t)

    Eq0 = L.diff(dq0_t, t) - L.diff(q0_t) - Q0 # = 0
    Eq1 = L.diff(dq1_t, t) - L.diff(q1_t) - Q1 # = 0
    Eq2 = L.diff(dq2_t, t) - L.diff(q2_t) - Q2 # = 0
    Eq3 = L.diff(dq3_t, t) - L.diff(q3_t) - Q3  # = 0
    # equations of motion
    Eq = sp.Matrix([Eq0, Eq1, Eq2, Eq3])


    # from symbtools.modeltools
    np = 1
    nq = 2
    pp = sp.Matrix(sp.symbols("p1:{0}".format(np+1) ) )
    qq = sp.Matrix(sp.symbols("q1:{0}".format(nq+1) ) )
    ttheta = st.row_stack(pp, qq)
    Q1, Q2 = sp.symbols('Q1, Q2')

    p1_d, q1_d, q2_d = st.time_deriv(ttheta, ttheta)
    p1_dd, q1_dd, q2_dd = st.time_deriv(ttheta, ttheta, order=2)

    p1, q1, q2 = ttheta

    # reordering according to chain
    kk = sp.Matrix([q1, q2, p1])
    kd1, kd2, kd3 = q1_d, q2_d, p1_d
    params = sp.symbols('l1, l2, l3, l4, s1, s2, s3, s4, J1, J2, J3, J4, m1, m2, m3, m4, g')
    l1, l2, l3, l4, s1, s2, s3, s4, J1, J2, J3, J4, m1, m2, m3, m4, g = params

    # geometry
    mey = -sp.Matrix([0,1])

    # coordinates for centers of inertia and joints
    S1 = mt.Rz(kk[0])*mey*s1
    G1 = mt.Rz(kk[0])*mey*l1

    S2 = G1 + mt.Rz(sum(kk[:2]))*mey*s2
    G2 = G1 + mt.Rz(sum(kk[:2]))*mey*l2

    S3 = G2 + mt.Rz(sum(kk[:3]))*mey*s3
    # noinspection PyUnusedLocal
    G3 = G2 + mt.Rz(sum(kk[:3]))*mey*l3

    # velocities of joints and center of inertia
    Sd1 = st.time_deriv(S1, ttheta)
    Sd2 = st.time_deriv(S2, ttheta)
    Sd3 = st.time_deriv(S3, ttheta)

    # energy
    T_rot = ( J1*kd1**2 + J2*(kd1 + kd2)**2 + J3*(kd1 + kd2 + kd3)**2)/2
    T_trans = ( m1*Sd1.T*Sd1 + m2*Sd2.T*Sd2 + m3*Sd3.T*Sd3)/2

    T = T_rot + T_trans[0]
    V = m1*g*S1[1] + m2*g*S2[1] + m3*g*S3[1]

    external_forces = [0, Q1, Q2]
    mod = mt.generate_symbolic_model(T, V, ttheta, external_forces, simplify=False)

    return mod

if __name__ == '__main__':
    modeling()