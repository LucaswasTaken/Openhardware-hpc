__kernel void hello()
{
  int gid = get_global_id(0);
  printf("Hello from process: %d,  ",gid);
}