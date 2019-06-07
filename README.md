# LU_Solver

If you want to invert a big matrix but you ain't got the time, you should probably use an LU factorization
and then backward-forward solve it.

Only worth it when the matrix is more than 600 x 600 or so.

Scipy's linalg.lu_solve is about twice as fast as this code, I think due to the fact I don't save any of the values
I calculate during my steps, but both scipy's and numpy's inverters are much slower.
