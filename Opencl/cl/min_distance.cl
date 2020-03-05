float compute_distance(global float *x, global float *y, int i, unsigned int n){

	float min_distance=0;

	float xi = x[i];
	float yi = y[i];
	
	int cont=0;
	for(int j=0;j<n;j++){
		if(j!=i){

			float xj = x[j];
			float yj = y[j];
			float dist = sqrt( (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) );

			if(cont==0){
				min_distance = dist;
			}
			if(dist<min_distance){
				min_distance = dist;
			}
			cont = cont +1 ;
		}
	}	
	return min_distance;
}



__kernel void min_distance(
	global float* x,
	global float* y,
	global float* distance,
	const unsigned int count)
{                                            


	int i = get_global_id(0);
	int n = (int)count;
   
   if(i < n){
   		distance[i] = compute_distance(x,y,i,count);
   }
}