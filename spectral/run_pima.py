dfp2 = run_pima(d=2, nmin=200, nmax=50000, npts=6, nreps=50,
                orders=[1, 2, 4, 6], m_max=300)
dfp4a = run_pima(d=4, nmin=1000, nmax=40000, npts=5, nreps=50,
                 orders=[1, 2, 4], m_max=60, scale=1.5)
dfp4b = run_pima(d=4, nmin=1000, nmax=40000, npts=5, nreps=50,
                 orders=[1, 2, 4], m_max=60, scale=3.0)
