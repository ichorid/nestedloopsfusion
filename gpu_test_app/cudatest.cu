#include "cuda_error_check.cu"
#include <stdio.h>
#include <stdlib.h>
#define BLOCKSIZE 5
typedef  unsigned short uint16;

__device__ unsigned int rolMaskA(unsigned int value, unsigned int amount)
{
	    return (value << amount) | (value >> ((32 - amount) & 31));
}

__device__ unsigned int lanemask_eq(){
	unsigned int mask;
	asm("mov.u32 %0, %%lanemask_eq;" : "=r"(mask));
	return mask;
}

__device__ unsigned int lfsr(uint16 start_state, int cycles = 1){
	uint16 lfsr = start_state;
	uint16 bit;                    /* Must be 16bit to allow bit<<15 later in the code */
	for (int i=0;i<cycles;i++){
		/* taps: 16 14 13 11; feedback polynomial: x^16
		* + x^14 + x^13 + x^11 + 1 */
		bit  = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5) ) & 1;
		lfsr =  (lfsr >> 1) | (bit << 15);
	}
	return lfsr;
}
__global__ void kernel_gpu (int* out, unsigned int seed, uint16 l2, int limit, unsigned int zt)
{
	const unsigned tid = (blockIdx.x << BLOCKSIZE) + threadIdx.x;
	const unsigned int lmask = lanemask_eq();
	int  total_cnt=0;
	const unsigned int tmask = 0xffffffff>>(32 - zt);
	//for (int k=0; k < 10; k++){
		uint16 rnd = lfsr(tid^seed + 1233,  100);
		uint16  o_cnt=0;
		do {
			int i_cnt=0;
			o_cnt++;
			//uint16 irnd = lfsr(tid+1, 100) ^ seed ^ lfsr(o_cnt+1, 20);
			uint16 irnd = lfsr(tid^seed +321 , 103) ^ lfsr(o_cnt^seed +231 , 117);
			// Random bitmask from o_cnt
			//zt = 0xffffffff ^ (1<<(o_cnt%32));
			zt = rolMaskA(tmask, rnd%32);
			do{
				i_cnt++;
				total_cnt+=(irnd&1);
				irnd=lfsr(irnd);
			//} while(!((rnd&zt&3) && (irnd & l2)==l2) && i_cnt < 10000 );
			//} while(!((lmask&zt) && (irnd & l2)==l2) && i_cnt < 10000 );
			} while(!((lmask&zt)) && i_cnt < 10000 );
			//} while(!((((threadIdx.x+1)&rnd)==threadIdx.x) && (irnd & l2)==l2) && i_cnt < 10000 );
			rnd = lfsr(rnd);
		//}while((rnd & l1)!=l1 && o_cnt<limit);
		}while(o_cnt<limit);
	//}
	out[tid] = total_cnt;
}
/*
__global__ void kernel_gpu (int* out, unsigned int seed, uint16 l2, int limit, unsigned int zt)
{
	const unsigned tid = (blockIdx.x << BLOCKSIZE) + threadIdx.x;
	const unsigned int lmask = lanemask_eq();
	int  total_cnt=0;
	//for (int k=0; k < 10; k++){
		uint16 rnd = lfsr(tid^seed + 1,  100);
		uint16  o_cnt=0;

		int i_cnt;
		uint irnd;
		bool bcond = 0;
		do {
			if (!bcond){ 
				i_cnt=0;
				o_cnt++;
				irnd = lfsr(tid+1, 100) ^ seed ^ lfsr(o_cnt+1, 20);
			}
			//do{
				i_cnt++;
				total_cnt+=(irnd&1);
				irnd=lfsr(irnd);
			//} while(!((lmask&zt)   && (irnd & l2)==l2) && i_cnt < 10000 );
			//} while(!((((threadIdx.x+1)&rnd)==threadIdx.x) && (irnd & l2)==l2) && i_cnt < 10000 );
			//bcond = (!((((threadIdx.x+1)&rnd)==(threadIdx.x+1)) && (irnd & l2)==l2) && i_cnt < 10000 );
			zt = 0xffffffff ^ (1<<(o_cnt%32));
			bcond =  !((irnd & l2)==l2 && (lmask&zt)) && i_cnt < 10000 ;
			//bcond = (!((lmask&zt)   && (irnd & l2)==l2) && i_cnt < 10000 );
			if (!bcond)
				rnd = lfsr(rnd);
		//}while((rnd & l1)!=l1 && o_cnt<limit);
		}while(bcond || o_cnt<limit);
	//}
	out[tid] = total_cnt;
}
*/

int main (int argc, char* argv[]){
	
	uint16 l2 = atoi(argv[1]);
	uint16 seed = atoi(argv[2]);
	int limit = atoi(argv[3]);
	unsigned int zt = strtoul(argv[4], NULL, 0);

	int* D_out;
	
	int num_blocks = 1 << 0;
	int num_threads = num_blocks * (1<<BLOCKSIZE);
	cudaEvent_t start, stop;
	CudaSafeCall(cudaMalloc((void**) &D_out, num_threads * sizeof(int)));
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	kernel_gpu <<< num_blocks, (1<<BLOCKSIZE) >>> (D_out, l2, seed, limit, zt);
	CudaCheckError();
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	double tm=elapsedTime/1000;
		
	int* H_out = (int*) malloc (num_threads * sizeof(int));
	CudaSafeCall(cudaMemcpy((void*) H_out, D_out, num_threads * sizeof(int), cudaMemcpyDeviceToHost));
	long int sum = 0;
	for (int i=0; i<num_threads; i++)
	{
		sum+=H_out[i];
		printf("\n %i", H_out[i]);
	}
	printf("\n Time Sum Avg Avgt/elem %f %li %li %f", tm, sum, sum/num_threads, sum/num_threads/tm);
	printf("\n");



}
