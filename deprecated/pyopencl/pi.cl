__kernel void pi_cl(__global float *pi_vec, float dx)
{
  int gid = get_global_id(0);
  float x,partial_pi;
  x = (gid + 0.5) * dx;
  partial_pi = 4.0/(1.0 + x*x)*dx;
  pi_vec[gid] = partial_pi;
}
