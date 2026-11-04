# Pointer 5: the transition-mode count and the point-crossing undercount (07_page_curve.tex line 45)

Printed claim: the frozen-symbol transfer at each block's back-averaged receiver fields reproduces the count of eigenvalues between 0.1 and 0.9 in all 120 blocks (two independent calculations), and the point-crossing limit, with the centrifugal term removed and each dispersion branch linearized, undercounts them at every block because it drops the finite-time mixing of momenta and bands in the ramp-averaged projections.

## What each file shows

1. The measured counts on the stored exterior fields, 120 blocks: box <internal>/packets_manifest.json (md5 c8745fc9). Values per block: N_mid, the ramp statistic r, Tr F.

2. Calculation 1, the homogeneous frozen-symbol transfer with the centrifugal term k/r_n, full lattice dispersion and spinors: code c22a4_homog_r1_v1.py (faa9925e), output a4_homog_r1.json (36f9f433), spectra a4_homog_r1_spectra.npz (60204acf). The comparison with the measured counts is a4_compare_r1_vs_measured_v1.txt (c7a1e261): the count difference is 0 at all 120 blocks, and every point-crossing value (24.1 to 24.7) lies below the measured count (26 to 30).

3. Calculation 2, independent code: c22a4_homog_r2_v6.py (172da924), merged output a4_r2v6_merged.json (d3653ea5), comparison a4_r2compare_v2_out.txt (9427cd61). It gives the same count at all 120 blocks, agrees with calculation 1 on r and Tr F to 1e-10, and matches its sorted eigenvalues to 1.4e-12.

4. Chain-length check: a4_homog_r1_L6144.json (fd89a417). Lengthening the chain from 4096 to 6144 sites changes no count and moves r and Tr F by at most 8e-11.

5. Which part of the undercount comes from what: a4_kscan.json (df2c9ca3), summary a4_finalsum_out.txt (e35c2d23). With the centrifugal term set to zero, the frozen-symbol count already sits about 3.4 modes above the point-crossing value at every block. That is the linearization and band-mixing part. The centrifugal term adds one mode pair, as a step, only where N_R kappa Delta u exceeds about 1, which is the receiver 16 r_s, Delta u = 10 r_s, second-channel configuration. The held-chain diagnostic a4_heldiag.json (04cfae48) shows the larger rise/fall noncommutativity on the transition modes there (0.46 against 0.31-0.33 elsewhere).

6. Agreed record of the comparison, with both reviewers' acceptance: C22_A4_AMENDMENT_RECORD_v1.md (5281af33).

Files 2-5 sit under deliveries/route2/calcC/a4diag/out/ (on the box under <internal>/calcC and calcC/out/), and the record sits in deliveries/route2/calcC/.

## Mechanism in two sentences

The block operator is the positive part of the difference between the closing- and opening-ramp averages of the time-evolved exterior-cut projector, and its intermediate eigenvalues count modes that the finite ramps neither keep nor release whole. Point crossing assigns each crossing a single scalar weight at a single time, so it cannot represent the finite-time mixing of lattice momenta and spinor band components inside the ramps, and it counts fewer intermediate modes.
