# Interacting momentum balance and stress

**The declared local balance passes; a source-free balance and a universally vanishing product-Hartree interaction stress fail. [R,N]** The corrected energy current defines a finite-range momentum density and a corresponding flux, but its integrated charge is not conserved by either converter. Exact controlled witnesses give nonzero interaction stress on number-conserving species-product Slater states.

Scope: inbox hreduce #102 and clause (c) of the pinned vertex ruling. This result treats the on-site axial-pair converter and the already defined three-dimensional filtered, kernel-averaged converter. It fixes the energy, momentum and bond allocations explicitly. The numerical work consists only of new contractions of admitted exact current and filter operands: all 26 prescribed raw objects and all 22 decisions are complete. It supplies no production-history stress, interacting-sea cancellation, continuum force bound or unique metric stress. The manuscript is unchanged. Tags: [R] re-derived; [N] numerical with controls; [S] single source. Shared-primary algebraic readouts are identified below.

SOURCES: [inbox #102](<internal>/hreduce.md); [pinned reviewer A ruling](<internal>/ReviewA_vgate_v1.md); [pinned reviewer B ruling](<internal>/ReviewB_vgate_v1.md); [actual model source](<internal>/03_model.tex); [frozen derivation and finite-check contract](<internal>/frozen_spec.tex); [complete author record](<internal>/VERTEX_stress_author_v1.tex); [independent peer record](<internal>/redteam_VERTEX_stress_v1.tex). The admitted predecessors and all box-only output identities are listed below.

| Claim | Verdict | Scope |
|---|---|---|
| Exact local momentum balance with interaction and a declared finite-range flux | **PASS** | The force term is explicit and its sum is \(i[H,Q_i]\). |
| Source-free local continuity for the same corrected energy-current momentum | **FAIL** | Nonzero integrated-force witnesses survive on compatible periodic parents. |
| Interaction stress vanishes on every number-conserving species-product state | **FAIL** | Both on-site and filtered Slater-product witnesses have nonzero quadratic interaction stress. |
| Converter potential and odd-converter selection rule on that product class | **PASS, scoped** | This does not remove neutral quadratic terms in the full current, stress or force. |
| Clock/flow coupling and reference bookkeeping | **PASS for the declared coupling** | The exact equation includes the flat interaction force and the external-field commutator. No gravitational consistency theorem is inferred. |

## Operator, kernel and allocation

Use \(\hbar=c=1\), sites \(x=an\), canonical CAR, a heavy field of mass \(m>0\), and a massless light field. The free one-body operator for species \(A\) is

\[
h_{0,A}=m_A\beta+\sum_i\alpha_i\frac{T_i-T_i^\dagger}{2ia},
\qquad \alpha_i=\tau_x\otimes\sigma_i,\quad \beta=\tau_z\otimes I_2.
\]

Here \((T_i u)_x=u_{x+ae_i}\), and \(d\Gamma\) denotes the many-body lift, summed over species. The Pauli matrices \(\tau_i,\sigma_i\) act on the upper/lower and spin factors, in component order \((0,1,2,3)=(\text{upper up},\text{upper down},\text{lower up},\text{lower down})\). Put \(H=H_0+UV\), \(V=\sum_xW_x\), \(U=\lambda_X/a^3\in\mathbb R\). The unchanged pair normalization is

\[
\Phi_A^\mu=c_A^TM^\mu c_A,
\quad M=(\tau_z\otimes\sigma_y,\ \tau_y\otimes\sigma_z,
\ i\tau_y\otimes I_2,\ -\tau_y\otimes\sigma_x),
\quad\eta=\operatorname{diag}(1,-1,-1,-1).
\]

There is no factor \(1/2\) in \(\Phi\). On-site,

\[
W_x=\sum_\mu\eta_{\mu\mu}\Phi_{L,x}^{\mu\dagger}\Phi_{H,x}^{\mu}
+\mathrm{h.c.}
\tag{1}
\]

For the filtered control retain the actual accepted operator, including its averaging kernel:

\[
F_b=\prod_j\frac{1+(-1)^{b_j}(T_j+T_j^\dagger)/2}{2},
\quad \Phi_{A,b,x}^\mu=(F_bc_A)_x^TM^\mu(F_bc_A)_x,
\quad b\in\{0,1\}^3,
\]
\[
O_{b,xy}=\sum_\mu\eta_{\mu\mu}\Phi_{L,b,x}^{\mu\dagger}\Phi_{H,b,y}^{\mu},
\quad
W_x=\frac12\sum_{b,y}g_{(x-y)/a}
(O_{b,xy}+O_{b,yx}+O_{b,xy}^\dagger+O_{b,yx}^\dagger).
\tag{2}
\]

All four legs in a term have the same filter label. The coefficients are the literal lattice Gram coefficients

\[
B_a(z)=a^{-3}\mathbf1_{[-a/2,a/2]^3}(z),\qquad
K_\ell(z)=\frac{9009}{1024\pi\ell^3}(1-|z|^2/\ell^2)_+^5,
\quad k_{\ell,a}=K_\ell*B_a,
\]
\[
g_r=a^3\int k_{\ell,a}(y)k_{\ell,a}(y-ar)\,d^3y,
\qquad \kappa(P)=\sum_rg_re^{iaP\cdot r}.
\tag{3}
\]

They are nonnegative, even, finite range and satisfy \(\sum_rg_r=1\). Define \(h=\sum_rg_rr_z^2\geq0\), so the second derivative of \(\kappa\) with respect to dimensionless \(\theta=aP_z\) at zero is \(-h\). No kernel quadrature or replacement of this lattice Gram by a single continuum Fourier factor is used. [R]

Let \(\Pi_x\) project onto site \(x\), including its components. Allocate the free energy and the corrected momentum by

\[
e_{0,x}=d\Gamma(\{\Pi_x,h_0\}/2),\qquad e_x=e_{0,x}+UW_x,
\]
\[
j^E_{xy}=i[e_x,e_y],\qquad
P_{x,i}=P^0_{x,i}+\frac12\sum_y(y_i-x_i)(j^E_{xy}-j^{E,0}_{xy}),
\]
\[
P^0_{x,i}=d\Gamma(\{\Pi_x,q_i\}/2),\qquad
q_i=\frac{T_i^2-(T_i^\dagger)^2}{4ia},\qquad Q_i=\sum_xP_{x,i}.
\tag{4}
\]

All finite-range displacements use their consistent local lift on a periodic parent. The interaction contribution to each current bond moment is split equally between its endpoints. Thus \(Q=Q_0+UQ_1\) on-site, and \(Q=Q_0+UQ_1+U^2Q_2\) for the filtered converter. The latter quadratic term is retained. [R; inherited current controls N]

## Exact local balance and face stress

The free momentum-transfer operator is

\[
J^{0,P}_{xy,i}=d\Gamma\!\left(
\frac{i}{2}\{\Pi_xh_0\Pi_y-\Pi_yh_0\Pi_x,q_i\}\right).
\]

Since \([h_0,q_i]=0\), it obeys \(i[H_0,P^0_{x,i}]+\sum_yJ^{0,P}_{xy,i}=0\) on the infinite or periodic translational lattice. Its first moment is \(d\Gamma(\{v_j,q_i\}/2)\), with \(v_j=\alpha_j\cos(ap_j)\); its positive-band packet value is the paper's \(q_iq_j/\varepsilon\). Open finite controls retain their boundary terms. [R]

A definite interaction flux and force allocation is

\[
J^{I,P}_{xy,i}=i\bigl([e_x,P_{y,i}]-[e_y,P_{x,i}]
-[e_{0,x},P^0_{y,i}]+[e_{0,y},P^0_{x,i}]\bigr),
\]
\[
s^I_{x,i}=i[e_x,Q_i]-i[e_{0,x},Q_{0,i}].
\tag{5}
\]

The flux is antisymmetric in \(x,y\). Write \(J^P=J^{0,P}+J^{I,P}\). Summing its definition gives the exact identity

\[
\boxed{\dot P_{x,i}+\sum_y(J^{0,P}_{xy,i}+J^{I,P}_{xy,i})=s^I_{x,i},
\qquad \sum_xs^I_{x,i}=i[H,Q_i].}
\tag{6}
\]

This splitting is declared and is not unique. A source-free periodic flux for the same momentum would require \([H,Q_i]=0\). Changing only its bond routing cannot change that necessary integrated condition.

For a literal local stress, route every \(x\to y\) transfer along all shortest cubic paths with equal weight. Let \(\chi_{xy}(z,j)\) be the signed mean traversal of the edge \((z,z+ae_j)\), reversing sign on reversed paths. Then

\[
\sum_j[\chi_{xy}(z,j)-\chi_{xy}(z-ae_j,j)]=\delta_{zx}-\delta_{zy},
\qquad
\mathcal T_{ij}(z+ae_j/2)=\frac1{2a^2}\sum_{xy}\chi_{xy}(z,j)J^P_{xy,i},
\]
\[
\partial_T(P_{z,i}/a^3)+\sum_j\frac{\mathcal T_{ij}(z+ae_j/2)-
\mathcal T_{ij}(z-ae_j/2)}a=\frac{s^I_{z,i}}{a^3}.
\tag{7}
\]

Assign half a face value to each adjacent cell when a site stress is wanted. The integrated stress is \(S_{ij}=\frac12\sum_{xy}(y_j-x_j)J^P_{xy,i}\). No symmetry in \(i,j\) is assumed. In the filtered case, \(R_*=\max(2a,2\ell+2\sqrt3a)\) bounds the energy support; momentum support is at most \(3R_*\), interacting anchors are separated by at most \(4R_*\), and the union support is within \(7R_*\). Even cubic parents with \(L\geq6\) and \(La>16R_*\) suffice for the local lifts used here. The checks evaluate finite Laurent coefficients without constructing a parent Fock space. [R]

## On-site product-state witness

Reuse the accepted two-site, two-particle current operands at \(a=1\). Modes \(0,\ldots,7\) belong to site zero and \(8,\ldots,15\) to site one; four heavy modes precede four light modes at each site. In the ascending two-particle bitmask basis, define

\[
A=|0,2\rangle,\quad B=|8,10\rangle,\quad C=|0,8\rangle,
\quad Q_1=\frac i2[T,W_1-W_0],
\]
\[
F_2=i[W_0+W_1,Q_1],\qquad
J_2=\frac i2[W_0-W_1,Q_1],\qquad s_{2,x}=i[W_x,Q_1].
\tag{8}
\]

Here \(T\) is the saved open \(z\)-bond hopping, and the existing \(Q_1\) is reused. Since \(P^1_0=P^1_1=Q_1/2\), the full local identities are

\[
F_2/2+J_2=s_{2,0},\qquad F_2/2-J_2=s_{2,1}.
\tag{9}
\]

The no-half pair normalization and hopping signs give

\[
W_0A=-8|4,6\rangle,\quad W_1B=-8|12,14\rangle,\quad
TC=-i(A+B)/2,
\]
\[
Q_1C=-2|4,6\rangle+2|12,14\rangle,
\quad F_2C=16i(A-B),\quad J_2C=8i(A+B).
\]

Take the normalized Slater product

\[
|\psi\rangle=\frac{A+iC}{\sqrt2}
=c^\dagger_{H,0,0}\frac{c^\dagger_{H,0,2}+i c^\dagger_{H,1,0}}{\sqrt2}
|0_H\rangle\otimes|0_L\rangle.
\]

It has \((N_H,N_L)=(2,0)\). The complete retained matrices and normalized contractions give

\[
\boxed{\langle F_2\rangle=-16,\quad\langle J_2\rangle=-8,
\quad\langle s_{2,0}\rangle=-16,\quad\langle s_{2,1}\rangle=0.}
\tag{10}
\]

All four Hermiticity residuals and both entire matrices in (9) vanish exactly. Both selected columns agree with their independent known answers. Restoring the physical factors, the interaction force is \(-16U^2\) and the integrated bond stress is \(-8aU^2\). Only the shared adjacent bond connects the occupied supports in the relevant matrix element, so the witness survives embedding in a larger periodic lattice; it is not an open-wall force. These are instantaneous product-state statements, not production-history or sea expectations. [R,N]

## Filtered stress and fixed-mass force

Let \(V_s\) be the number of sites used to normalize plane waves. Use the ordered spin-pair basis \((01,02,03,12,13,23)\), with

\[
K_{ab;cd}=4\sum_\mu\eta_{\mu\mu}(M^\mu_{ab})^*M^\mu_{cd}.
\]

In dimensionless total and relative momenta \(\boldsymbol\theta=aP\), \(q=a(p_1-p_2)/2\), the pair rows are

\[
L_b(\boldsymbol\theta,q)=f_b(\boldsymbol\theta/2+q)f_b(\boldsymbol\theta/2-q),
\qquad f_b(p)=\prod_j\frac{1+(-1)^{b_j}\cos p_j}{2}.
\]

Primes below differentiate \(\theta_z\). Regard \(L\) as the finite-rank map from relative momentum to filter labels. Relative-momentum composition uses the normalized Brillouin-zone inner product \(\int_{[-\pi,\pi]^3}d^3q/(2\pi)^3\), equivalently its exact normalized finite Fourier sum on the compatible parents. Write

\[
R=\kappa L^\dagger L,\quad
A=\kappa L^\dagger L'+\frac{\kappa'}2L^\dagger L,\quad
B=\kappa L^\dagger L''+\kappa'L^\dagger L'+\frac{\kappa''}2L^\dagger L.
\]

The endpoint allocation, including its second-moment contact, gives

\[
D^W=ia(R\partial_\theta+A)K\tau_x^{HL},\qquad
E_2^W=-a^2(R\partial_\theta^2+2A\partial_\theta+B)K\tau_x^{HL}.
\tag{11}
\]

Here \(\tau_x^{HL}\) exchanges the heavy-pair and light-pair sectors, \(D^W=\sum_xx_zW_x\), and \(E_2^W=\sum_xx_z^2W_x\). Let \(D^0=\sum_xx_ze_{0,x}\), \(E_2^0=\sum_xx_z^2e_{0,x}\). The interaction momentum dipoles and quadratic stress coefficient are

\[
D^P_1=\frac i2([H_0,E_2^W]+[V,E_2^0]),\qquad
D^P_2=\frac i2[V,E_2^W],
\]
\[
S^I_2=i[H_0,D^P_2]+i[V,D^P_1]-i[D^0,Q_2]-i[D^W,Q_1].
\tag{12}
\]

Choose heavy components \(0,1\), both at zero momentum, and light vacuum. At \(\theta=0\), \(A=0\), \(B=A'=L^\dagger L''-hR/2\). The kinetic contribution to this stress mean is odd under \(\beta_1\beta_2\) and vanishes. Odd converter powers vanish by species selection. With \((K^2)_{01,01}=32\) admitted from the old exact matrix, the remaining mass contribution is the following **selected upper-01 spin matrix element**, before relative-momentum evaluation:

\[
\langle01|S^I_{2,m}|01\rangle_{\rm spin}
=32ma^2\left[RA'+A'^\dagger R+2RA'^\dagger+2A'R
+\frac14\{R^2,\partial_{q_z}^2\}\right]_{\theta=0}.
\tag{13}
\]

It is not a full-spin operator identity. The three nonzero contributions to (12), after this spin projection and common factor, are respectively
\(-RA'+A'^\dagger R+\{R^2,\partial_{q_z}^2\}/4\),
\(2(RA'^\dagger+A'R)\), and \(2RA'\).
The diagonal mass commutator in the first term of (12) vanishes. The relative-coordinate derivative contact is retained.

The zero-relative-momentum evaluation has \(l_0=L(0,0)=e_{000}\), \(l_{qq}=-l_0\). Put \(L_0=L|_{\theta=0}\), \(S_0=(LL^\dagger)|_0\), and \(S_2=\partial_\theta^2(LL^\dagger)|_0\). The admitted Gram entries are \((S_0)_{00}=42875/2097152\) and \((S_2)_{00}=-6125/524288\). Consequently the entire interaction-stress mean in this normalized product state is

\[
\boxed{\frac{V_s}{ma^2U^2}\langle S^I_{zz}\rangle
=32\{2(S_2)_{00}-(1+3h)(S_0)_{00}\}
=-\frac{91875+128625h}{65536}<0.}
\tag{14}
\]

This statement uses \(m>0\), \(h\geq0\) and nonzero \(U\); its displayed normalized coefficient is independent of the sign of \(U\). [R,N]

A separate operator test keeps the mass fixed and uses heavy spin-pair bra \(01\) and ket \(23\), each with both momenta \((0,0,\theta/2a)\). These are not positive-band scattering states. The admitted coefficient is \((K^2)_{01,23}=-32\). With \(\mathcal C=RA^\dagger+AR\), the relevant mass part is

\[
\langle01|i([H_0,Q_2]+[V,Q_1])|23\rangle
=-32ima(4\mathcal C-2RA-2A^\dagger R).
\]

The kinetic part is grade-odd and odd converter powers cannot connect this heavy-pair sector. The exact coefficient therefore gives

\[
\boxed{\frac{V_s}{imaU^2}\left.\partial_\theta
\langle HH,01;\theta|i[H,Q_z]|HH,23;\theta\rangle\right|_0
=\frac{55125+85750h}{65536}>0.}
\tag{15}
\]

The matrix element is zero at \(\theta=0\); its nonzero derivative implies nonzero values at sufficiently small compatible momenta on sufficiently large periodic parents. This uses the analytic finite-range dependence, without a momentum sweep. It disproves exact conservation for the same corrected current at fixed \(m>0\) for every admissible actual kernel. It is distinct from the force expectation in the stress state of (14). [R,N]

The new exact contractions from the saved Laurent operands are

\[
r^Tu=-42875/8388608,\qquad r^Tt=-6125/1048576,\qquad
l_0^TS_0l_{qq}=-42875/2097152,
\]

where \(r=L_0^Tl_0\), \(t=L_2^Tl_0\), \(u=L_0^Tl_2\), \(L_2=\partial_\theta^2L|_0\), and \(l_2=\partial_\theta^2L(\theta,0)|_0=-l_0/4\), in the retained real Laurent coefficient representation. The direct and Gram readouts both return the stress coefficients \((-91875,-128625)/65536\), and the force coefficients \((55125/65536,42875/32768)\), in \((1,h)\) order. Separate sign checks pass. These routes share the admitted primaries; their agreement does not represent independent dynamics or a new kernel computation. [N]

## Hartree scope and clock/flow coupling

For \(\rho=\rho_H\otimes\rho_L\) with \([\rho_A,N_A]=0\), each converter changes the species numbers by two. Its expectation and partial-trace Hartree potentials vanish. Operators of odd converter degree likewise have zero expectation in this class. Neutral even-degree products need not vanish. This leaves the on-site interaction-current Hartree result intact while preserving the already established failure of blanket full-flow Hartree vanishing for the filtered converter. Equations (10) and (14) disprove the extension to all interaction stresses, even for Slater products. Exact evolution can leave the product manifold; a projected Hartree evolution is a different dynamics. [R,N]

The full integrated force makes the surviving terms explicit:

\[
i[H,Q_i]=Ui([H_0,Q_{1,i}]+[V,Q_{0,i}])
+U^2i([H_0,Q_{2,i}]+[V,Q_{1,i}])+U^3i[V,Q_{2,i}].
\tag{16}
\]

Although \(Q_2=0\) on-site, its quadratic force \(i[V,Q_1]\) survives. Discrete translation symmetry supplies no identity setting (16) to zero.

For the paper's classical, back-averaged fields, the declared interacting extension is

\[
G[N,v]=\sum_xN_xe_x+\sum_{x,j}v_{x,j}P_{x,j},\qquad
\frac{\partial G}{\partial N_x}=e_x,\quad
\frac{\partial G}{\partial v_{x,j}}=P_{x,j}.
\]

The flow couples to the corrected operator in (4). If the bare operators have no explicit time dependence, \(\dot P_{x,i}=i[G,P_{x,i}]\). With
\(A_{\rm ext}=\sum_y(N_y-1)e_y+\sum_{y,j}v_{y,j}P_{y,j}\), the same flat-space flux satisfies

\[
\dot P_{x,i}+\sum_yJ^P_{xy,i}=s^I_{x,i}+i[A_{\rm ext},P_{x,i}].
\tag{17}
\]

Equivalently, separate the field gradients without assuming a conserved boost or momentum:

\[
i[G,P_{x,i}]=N_xi[H,P_{x,i}]+\sum_jv_{x,j}i[Q_j,P_{x,i}]
+i\sum_y\left[(N_y-N_x)e_y+\sum_j(v_{y,j}-v_{x,j})P_{y,j},P_{x,i}\right].
\]

A c-number subtraction \(p^{\rm ref}_{x,i}(T)\) adds \(-\partial_Tp^{\rm ref}_{x,i}\) to the referenced balance. A state- or field-dependent reference cannot be held fixed silently while differentiating. The declared lattice couplings and routing do not define a unique variation with respect to an arbitrary spatial metric, a symmetric relativistic stress or an interacting gravitational constraint. Any continuum source-free approximation needs a bound on the retained force. None is supplied by these finite witnesses. [R]

## Numerical completion, reuse and identities

The new plan contains O14/F12 raw objects and O12/F10 decisions. The 32 admission records, 62 on-site records and 52 filtered-stress records contain every operand reference and every actual, expected, residual and decision object. All residuals are exact zero; all 22 decisions are true. The result is complete, the child is terminal and reaped with return code zero, and both logs are empty. Input/source identities agree through admission and terminal postchecks. The recorded execution uses one CPU, one library thread, nice 10 and a 2 GiB address-space limit. Resource measurements remain in the box record.

The old accepted 74/80/18 controls, current matrices, selected spin factors and Laurent jets were reused. No current, spinor, amplitude, kernel Gram or full-lattice Fock product was regenerated. The older broad amplitude plan remains historically incomplete; its separately published named amplitude cut is unaffected. The analytic reductions in (6), (12), (13) and (15) are [R]; the selected exact contractions and their controls are [N]. The result is complete at this scope and requires no additional allocation.

The two used source files are [producer](<internal>/vertex_stress_v1.py) and [wrapper](<internal>/run_vertex_stress_v1.py). Their SHA256 values are respectively `327651aaf32a0de7d92ca510e0a30d749b339931622c63485cf85b232ae5802c` and `41bb2488a08cf1c2d46b7f62044af5c9a31e9bc643bc75d676cfd9ea58dc7d18`. The frozen specification SHA256 is `157ef5878346c8b7c6d01f380c441eee3b610bda205aba1d53bf6e155717b8c2`.

The pinned model source SHA256 is `c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb`; the reviewer A and reviewer B ruling hashes are `49a7e54761d97173731d7d6c147bc8e6d56ee53d94110e71a6d82f938f055e50` and `4f3f586c3d5fb3894a086e68caf6322250c448d6f9fdae85421f22993012a13b`. The source conventions come from the pinned free model, with the converters explicitly supplied as extensions. [S]

The scientific predecessors are [on-site gate](<internal>/VERTEX_gate_v1.md), [filtered gate](<internal>/VERTEX_gate_filtered_v1.md), and [named amplitude cut](<internal>/VERTEX_amplitudes_v1.md). The seven admitted box primaries are:

| Full box path | SHA256 |
|---|---|
| `<internal>/on_site_completion.json` | `77f415126fc7cd271970169b134bf713f237b973b1bfdb598e6bf44e3a201f4d` |
| `<internal>/bond_current.json` | `d93c8604e18ee2d7ee22d8900ba25b65bcd9e1435bb1c7380322c0d12d3ffd8f` |
| `<internal>/vertex_gate_missing_result.json` | `52d430f7981514c291f9b1d9640d53ed3d9b0b1d0570e53a2f434c700510389d` |
| `<internal>/execution.json` | `be3a8a26d0db48dbbe67d56f75380aab137c6e7fa408c9a4b36c7fad6c9e88f9` |
| `<internal>/current.jsonl` | `daae7293de7df50c64a732ad34cd9bab396b0f04cf4379aa029f73d1f5beb64a` |
| `<internal>/vertex_gate_filtered_result.json` | `32300d99c630cb5ff9d9b8615e932a23c7dbf4b0b8bd75bfe5b24d30d2c17027` |
| `<internal>/execution.json` | `b5d726a8c3ba9a333c9bf391ae33abf92b75ece5174dd66a755eb68373377869` |

The admitted old on-site \(W^2\) entries are only \((3,3)=32\) and \((3,12)=-32\), with the saved basis. The bond operands are \(W,W_0,W_1,Q_1\) and the saved basis. Filtered journal sequences 278, 282, 285, 292, 293, 294, 297, 298 and 299 supply `relative_indices`, `L0`, `S0`, `S2`, `evaluation`, `l0`, `r`, `t` and `u`. These pointers and their schemas are retained in the new admission journal.

All new records remain on box `compute-host`:

| Full box path | SHA256 |
|---|---|
| `<internal>/admission.jsonl` | `4682c63ae1ad67d8122b0846c6aad0d050b64e929a632ee779c9135459dbef82` |
| `<internal>/onsite.jsonl` | `e108de4138112d1e17797dfff5725e82907037e0a9df04a3fbbff27c2be61916` |
| `<internal>/filtered_stress.jsonl` | `0f26d5af8f3b228fc20df40bfbf7dc3aa02da1ce211898fb802f386962e2ba14` |
| `<internal>/vertex_stress_result.json` | `0b462a21fd99f652219da57343ab615a99669b35539b1835530e0b9899d98d6d` |
| `<internal>/execution.json` | `8df49f9417dff368ac1f1a799b2cb0b65a462d3f1f73b59bfe871d5a39964259` |
| `<internal>/stdout.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `<internal>/stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
