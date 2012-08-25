set terminal png
set output "ex1a.png"
set samples 1001  # high quality

u_A(x) = (3<=x&&x<=5) ? 0.5*x-1.5 \
    : (5<x&&x<=7) ? -0.5*x+3.5  \
    : 0

plot [0:10] u_A(x)
