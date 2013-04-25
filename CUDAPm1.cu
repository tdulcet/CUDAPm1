char program[] = "CUDAPm1 v0.00";
/* CUDALucas.c
   Shoichiro Yamada Oct. 2010 

   This is an adaptation of Richard Crandall lucdwt.c, John Sweeney MacLucasUNIX.c,
   and Guillermo Ballester Valor MacLucasFFTW.c code.
   Improvement From Prime95.
   
   It also contains mfaktc code by Oliver Weihe and Eric Christenson
   adapted for CUDALucas use. Such code is under the GPL, and is noted as such.
*/

/* Include Files */
#include <stdlib.h>
#include <stdio.h>
#include <gmp.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <signal.h>
#ifndef _MSC_VER
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <direct.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "cuda_safecalls.h"
#include "parse.h"
//#include "kernel.h"
//#include "erato.h"

#ifdef _MSC_VER
#define strncasecmp strnicmp // _strnicmp
#endif

/* In order to have the gettimeofday() function, you need these includes on Linux:
#include <sys/time.h>
#include <unistd.h>
On Windows, you need 
#include <winsock2.h> and a definition for
int gettimeofday (struct timeval *tv, struct timezone *) {}
Both platforms are taken care of in parse.h and parse.c. */

/************************ definitions ************************************/
/* http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
   base code from Takuya OOURA.  */
/* global variables needed */
double *ttmp, *ttp;
double *g_ttp, *g_ttmp, *g_ttp1;
double *g_x, *g_y, *g_save, *g_ct;
double *e_data;
double *rp_data;
//double *g_bg[13];
//double *g_rp[480];

char *size;
int threads; 
char *g_numbits;
float *g_err;
int *g_data; 
cufftHandle plan; 

int quitting, checkpoint_iter, fftlen, tfdepth=0, llsaved=0, s_f, t_f, r_f, d_f, k_f;
int polite, polite_f;//, bad_selftest=0;
int b1 = 600000;
int g_b2 = 12000000;
int g_d = 2310;
int g_e = 6;
int g_nrp = 20;

char folder[132];
char input_filename[132], RESULTSFILE[132];
char INIFILE[132] = "CUDAPm1.ini";
char AID[132]; // Assignment key
char s_residue[32];

/************************ kernels ************************************/
# define RINT_x86(x) (floor(x+0.5))
# define RINT(x)  __rintd(x)

#ifdef _MSC_VER
long long int __double2ll (double);
#endif

__global__ void square (int n,
                         double *a,
                         double *ct)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki;
  double new_ajr, new_aji, new_akr, new_aki;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
    {
      int nminusj = n - j;

      wkr = 0.5 - ct[nc - j2];
      wki = ct[j2];
      ajr = a[j];
      aji = a[1 + j];
      akr = a[nminusj];
      aki = a[1 + nminusj];
      xr = ajr - akr;
      xi = aji + aki;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      ajr -= yr;
      aji -= yi;
      akr += yr;
      aki -= yi;

      new_aji = 2.0 * ajr * aji;
      new_ajr = (ajr - aji) * (ajr + aji);
      new_aki = 2.0 * akr * aki;
      new_akr = (akr - aki) * (akr + aki);

      xr = new_ajr - new_akr;
      xi = new_aji + new_aki;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;

      a[j] = new_ajr - yr;
      a[1 + j] = yi - new_aji;
      a[nminusj] = new_akr + yr;
      a[1 + nminusj] =  yi - new_aki;
    }
  else
    {
      xi = a[0] - a[1];
      a[0] += a[1];
      a[1] = xi;
      a[0] *= a[0];
      a[1] *= a[1];
      a[1] = 0.5 * (a[1] - a[0]);
      a[0] += a[1];
      xr = a[0 + m];
      xi = a[1 + m];
      a[1 + m] = -2.0 * xr * xi;
      a[0 + m] = (xr + xi) * (xr - xi);
    }
}

__global__ void mult2 (double *g_out,
                         double *a,
                         double *b,
                         double *ct,
                         int n)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki;
  double new_ajr, new_aji, new_akr, new_aki;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
    {
      int nminusj = n - j;

      wkr = 0.5 - ct[nc - j2];
      wki = ct[j2];
      ajr = a[j];
      aji = a[1 + j];
      akr = a[nminusj];
      aki = a[1 + nminusj];
      xr = ajr - akr;
      xi = aji + aki;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      ajr -= yr;
      aji -= yi;
      akr += yr;
      aki -= yi;
      xr = b[j];
      xi = b[1 + j];
      yr = b[nminusj];
      yi = b[1 + nminusj];
      
      new_aji = ajr * xi + xr * aji;
      new_ajr = ajr * xr - aji * xi;

      new_aki = akr * yi + yr * aki;
      new_akr = akr * yr - aki * yi;

      xr = new_ajr - new_akr;
      xi = new_aji + new_aki;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;

      g_out[j] = new_ajr - yr;
      g_out[1 + j] = yi - new_aji;
      g_out[nminusj] = new_akr + yr;
      g_out[1 + nminusj] =  yi - new_aki;
    }
  else
    {
      xr = a[0];
      xi = a[1];
      yr = b[0];
      yi = b[1];
      g_out[0] = xr * yr + xi * yi;
      g_out[1] = -xr * yi - xi * yr;
      xr = a[0 + m];
      xi = a[1 + m];
      yr = b[0 + m];
      yi = b[1 + m];
      g_out[1 + m] = -xr * yi - xi * yr;
      g_out[0 + m] = xr * yr - xi * yi;
    }
}

__global__ void mult3 (double *g_out, 
                         double *a, 
                         double *b, 
                         double *ct, 
                         int n)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki, bjr, bji, bkr, bki;
  double new_ajr, new_aji, new_akr, new_aki;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
    {
      int nminusj = n - j;

      wkr = 0.5 - ct[nc - j2];
      wki = ct[j2];

      ajr = a[j];
      aji = a[1 + j];
      akr = a[nminusj];
      aki = a[1 + nminusj];
      xr = ajr - akr;
      xi = aji + aki;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      ajr -= yr;
      aji -= yi;
      akr += yr;
      aki -= yi;

      bjr = b[j];
      bji = b[1 + j];
      bkr = b[nminusj];
      bki = b[1 + nminusj];
      xr = bjr - bkr;
      xi = bji + bki;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      bjr -= yr;
      bji -= yi;
      bkr += yr;
      bki -= yi;
      
      new_aji = ajr * bji + bjr * aji;
      new_ajr = ajr * bjr - aji * bji;
      new_aki = akr * bki + bkr * aki;
      new_akr = akr * bkr - aki * bki;

      xr = new_ajr - new_akr;
      xi = new_aji + new_aki;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;
      g_out[j] = new_ajr - yr;
      g_out[1 + j] = yi - new_aji;
      g_out[nminusj] = new_akr + yr;
      g_out[1 + nminusj] =  yi - new_aki;
    }
  else
    {
      xr = a[0];
      xi = a[1];
      yr = b[0];
      yi = b[1];
      g_out[0] = xr * yr + xi * yi;
      g_out[1] = -xr * yi - xi * yr;
      xr = a[0 + m];
      xi = a[1 + m];
      yr = b[0 + m];
      yi = b[1 + m];
      g_out[1 + m] = -xr * yi - xi * yr;
      g_out[0 + m] = xr * yr - xi * yi;
    }
}

__global__ void sub_mul (double *g_out, 
                         double *a, 
                         double *b1, 
                         double *b2, 
                         double *ct, 
                         int n)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki, bjr, bji, bkr, bki;
  double new_ajr, new_aji, new_akr, new_aki;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
    {
      int nminusj = n - j;

      wkr = 0.5 - ct[nc - j2];
      wki = ct[j2];

      ajr = a[j];
      aji = a[1 + j];
      akr = a[nminusj];
      aki = a[1 + nminusj];
      xr = ajr - akr;
      xi = aji + aki;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      ajr -= yr;
      aji -= yi;
      akr += yr;
      aki -= yi;

      bjr = b1[j] - b2[j];
      bji = b1[1 + j] - b2[1 + j];
      bkr = b1[nminusj] - b2[nminusj];
      bki = b1[1 + nminusj] - b2[1 + nminusj];
      
      new_aji = ajr * bji + bjr * aji;
      new_ajr = ajr * bjr - aji * bji;
      new_aki = akr * bki + bkr * aki;
      new_akr = akr * bkr - aki * bki;

      xr = new_ajr - new_akr;
      xi = new_aji + new_aki;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;
      g_out[j] = new_ajr - yr;
      g_out[1 + j] = yi - new_aji;
      g_out[nminusj] = new_akr + yr;
      g_out[1 + nminusj] =  yi - new_aki;
    }
  else
    {
      xr = a[0];
      xi = a[1];
      yr = b1[0] - b2[0];
      yi = b1[1] - b2[1];
      g_out[0] = xr * yr + xi * yi;
      g_out[1] = -xr * yi - xi * yr;
      xr = a[0 + m];
      xi = a[1 + m];
      yr = b1[0 + m] - b2[0 + m];
      yi = b1[1 + m] - b2[1 + m];
      g_out[1 + m] = -xr * yi - xi * yr;
      g_out[0 + m] = xr * yr - xi * yi;
    }
}

__global__ void pre_mul (int n, 
                           double *a, 
                           double *ct)
{
  const int j2 = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, ajr, aji, akr, aki;
  const int nc = n >> 2;
  const int j = j2 << 1;

  if (j2)
  {
    int nminusj = n - j;

    wkr = 0.5 - ct[nc - j2];
    wki = ct[j2];
    ajr = a[j];
    aji = a[1 + j];
    akr = a[nminusj];
    aki = a[1 + nminusj];
    xr = ajr - akr;
    xi = aji + aki;
    yr = wkr * xr - wki * xi;
    yi = wkr * xi + wki * xr;
    ajr -= yr;
    aji -= yi;
    akr += yr;
    aki -= yi;
    a[j] = ajr;
    a[1 + j] = aji;
    a[nminusj] = akr;
    a[1 + nminusj] =  aki;
  }
}

__device__ static double __rintd (double z)
{
  double y;
  
  asm ("cvt.rni.f64.f64 %0, %1;": "=d" (y):"d" (z));
  return (y);
}


__device__ static long long int __double2ll (double z)
{
  long long int y;
  
  asm ("cvt.rni.s64.f64 %0, %1;": "=l" (y):"d" (z));
  return (y);
}

__global__ void norm1 (double *g_in,
                         int *g_data,
                         double *g_ttp,
                         double *g_ttmp,
                         char *g_numbits,
		                     volatile float *g_err,
		                     float maxerr,
		                     int g_err_flag)
{
  long long int bigint;
  int val, numbits, mask, shifted_carry;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int index1 = blockIdx.x << 2;
  double tval, trint;
  float ferr;
  __shared__ int carry[1024 + 1];
 
  tval = g_in[index] * g_ttmp[index];
  trint = RINT (tval);
  ferr = tval - trint;
  ferr = fabs (ferr);
  bigint = trint;
  if (ferr > maxerr) atomicMax((int*)g_err, __float_as_int(ferr));

  numbits = g_numbits[index]; 
  mask = -1 << numbits;
  carry[threadIdx.x + 1] = (int) (bigint >> numbits);
  val = ((int) bigint) & ~mask;
  __syncthreads ();

  if (threadIdx.x) val += carry[threadIdx.x];
  shifted_carry = val - (mask >> 1);
  val = val - (shifted_carry & mask);
  carry[threadIdx.x] = shifted_carry >> numbits;
  __syncthreads ();

  if (threadIdx.x == (blockDim.x - 1))
  { 
    if (blockIdx.x == gridDim.x - 1) g_data[2] = carry[threadIdx.x + 1] + carry[threadIdx.x];
    else   g_data[index1 + 6] =  carry[threadIdx.x + 1] + carry[threadIdx.x]; 
  }

  if (threadIdx.x) val += carry[threadIdx.x - 1]; 
  if (threadIdx.x > 1)
  {
    if(g_err_flag) g_in[index] = (double) val;
    else g_in[index] = (double) val * g_ttp[index];
  }
  else g_data[index1 + threadIdx.x] = val;
}

__global__ void norm1a (double *g_in,
                         int *g_data,
                         double *g_ttp,
                         double *g_ttmp,
                         char *g_numbits,
		                     volatile float *g_err,
		                     float maxerr,
		                     int g_err_flag)
{
  long long int bigint[2];
  int val[2], numbits[2], mask[2], shifted_carry, i;
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  const int index1 = blockIdx.x << 2;
  double tval, trint;
  float ferr[2];
  __shared__ int carry[1024 + 1];
 
  for(i = 0; i < 2; i++)
  {
    tval = g_in[index + i] * g_ttmp[index + i];
    trint = RINT (tval);
    ferr[i] = tval - trint;
    ferr[i] = fabs (ferr[i]);
    bigint[i] = trint;
    if (ferr[i] > maxerr) atomicMax((int*) g_err, __float_as_int(ferr[i]));
    numbits[i] = g_numbits[index + i]; 
    mask[i] = -1 << numbits[i];

  }

  val[0] = ((int) bigint[0]) & ~mask[0];
  //carry[threadIdx.x + 1] = (int) (bigint >> numbits1);
  bigint[0] >>= numbits[0];
  bigint[1] += bigint[0];
  val[1] = ((int) bigint[1]) & ~mask[1];
  carry[threadIdx.x + 1] = (int) (bigint[1] >> numbits[1]);
  __syncthreads ();

  if (threadIdx.x) val[0] += carry[threadIdx.x];
  shifted_carry = val[0] - (mask[0] >> 1);
  val[0] = val[0] - (shifted_carry & mask[0]);
  val[1] += shifted_carry >> numbits[0];
  shifted_carry = val[1] - (mask[1] >> 1);
  val[1] = val[1] - (shifted_carry & mask[1]);
  carry[threadIdx.x] = shifted_carry >> numbits[1];
  __syncthreads ();

  if (threadIdx.x == (blockDim.x - 1))
  { 
    if (blockIdx.x == gridDim.x - 1) g_data[2] = carry[threadIdx.x + 1] + carry[threadIdx.x];
    else   g_data[index1 + 6] =  carry[threadIdx.x + 1] + carry[threadIdx.x]; 
  }

  if (threadIdx.x)
  {
    val[0] += carry[threadIdx.x - 1];
    if(g_err_flag)
      for(i = 0; i < 2; i++)
        g_in[index + i] = (double) val[i];
    else
      for(i = 0; i < 2; i++)
        g_in[index + i] = (double) val[i] * g_ttp[index + i];
  }
  else
  {
    g_data[index1 + threadIdx.x] = val[threadIdx.x];
    g_data[index1 + threadIdx.x + 1] = val[threadIdx.x + 1];
  }
}



__global__ void
normalize2_kernel (double *g_x, int g_N, int threads, int *g_data, double *g_ttp1, int g_err_flag)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadID1 = threadID << 2;
  const int threadID2 = threadID << 1;
  const int j = threads * threadID;
  double temp0, temp1;
  int mask, shifted_carry, numbits;

  if (j < g_N)
    {
      temp0 = g_data[threadID1] + g_data[threadID1 + 2];
      numbits = g_data[threadID1 + 3];
      mask = -1 << numbits;
      shifted_carry = temp0 - (mask >> 1) ;
      temp0 = temp0 - (shifted_carry & mask);
      temp1 = g_data[threadID1 + 1] + (shifted_carry >> numbits);
      if(!g_err_flag)
      {
        g_x[j] = temp0 * g_ttp1[threadID2];
        g_x[j + 1] = temp1 * g_ttp1[threadID2 + 1];
      }
      else
      {
        g_x[j] = temp0;
        g_x[j + 1] = temp1;
      }
    }
}

__global__ void
normalize3_kernel (double *g_in,  int *g_data, double *g_ttp, char *g_numbits,
		  volatile float *g_err, float maxerr)
{
  int val, numbits, mask, shifted_carry;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int index1 = blockIdx.x << 2;
  __shared__ int carry[1024 + 1];
 
  val = (int) __double2ll (g_in[index]);
  val *= 3;
  numbits = g_numbits[index]; 
  mask = -1 << numbits;
  shifted_carry = val - (mask >> 1);
  val = val - (shifted_carry & mask);
  carry[threadIdx.x] = shifted_carry >> numbits;
  __syncthreads ();

  if (threadIdx.x == (blockDim.x - 1))
  { if (blockIdx.x == gridDim.x - 1) g_data[2] = carry[threadIdx.x];
    else   g_data[index1 + 6] =  carry[threadIdx.x];
  }

  if (threadIdx.x) val += carry[threadIdx.x - 1]; 
  if (threadIdx.x > 1) g_in[index] = (double) val * g_ttp[index];
  else g_data[index1 + threadIdx.x] = val;
}

__global__ void
normalize4_kernel (double *g_x, int g_N, int threads, int *g_data, double *g_ttp1)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadID1 = threadID << 2;
  const int threadID2 = threadID << 1;
  const int j = threads * threadID;
  double temp0, temp1;
  int mask, shifted_carry, numbits;

  if (j < g_N)
    {
      temp0 = g_data[threadID1] + g_data[threadID1 + 2];
      numbits = g_data[threadID1 + 3];
      mask = -1 << numbits;
      shifted_carry = temp0 - (mask >> 1) ;
      temp0 = temp0 - (shifted_carry & mask);
      temp1 = g_data[threadID1 + 1] + (shifted_carry >> numbits);
      g_x[j] = temp0 * g_ttp1[threadID2];
      g_x[j + 1] = temp1 * g_ttp1[threadID2 + 1];
    }
}

__global__ void
copy_kernel (double *save, double *y)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  save[threadID] = y[threadID];
}

/****************************************************************************
 *                                Erato                                     *
 ***************************************************************************/
//Many thanks to Ben Buhrow. 

typedef unsigned char u8;
typedef unsigned int uint32;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef long long unsigned int uint64;

const int threadsPerBlock = 256;
const uint32 block_size = 8192;
const int startprime = 8;


__constant__ uint32 _step5[5] = 	{2418280706,604570176,151142544,37785636,1083188233};
__constant__ uint32 _step7[7] = 	{1107363844,69210240,2151809288,134488080,
									276840961,17302560,537952322};
__constant__ uint32 _step11[11] = 	{33816584,1073774848,135266336,132096,541065345,
									528384,2164261380,2113536,67110928,8454146,268443712};
__constant__ uint32 _step13[13] = 	{1075838992,16809984,262656,536875016,8388672,
									67239937,1050624,2147500064,33554688,268959748,4202496,
									65664,134218754};
__constant__ uint32 _step17[17] = 	{268435488,1073741952,512,2049,8196,32784,131136,
									524544,2098176,8392704,33570816,134283264,537133056,
									2148532224,4194304,16777218,67108872};
__constant__ uint32 _step19[19] = 	{2147483712,4096,262176,16779264,1073872896,8388608,
									536870928,1024,65544,4194816,268468224,2097152,134217732,
									256,16386,1048704,67117056,524288,33554433};

__global__ static void SegSieve(uint32 *count, uint32 *primes, int maxp, int nump, uint32 maxID, uint8 *results)
{ 
	/* 
	expect as input a set of primes to sieve with, how many of those primes there are (maxp)
	how many primes each thread will be responsible for (nump), and the maximum index
	that we need to worry about for the requested sieve interval.  Also, an array into
	which we can put this block's count of primes.
	
	This routine implements a segmented sieve using a wheel mod 6.  Each thread block on the gpu
	sieves a different segment of the number line.  Each thread within each block simultaneously 
	sieves a small set of primes, marking composites within shared memory.  There is no memory
	contention between threads because the marking process is write only.  Because each thread
	block starts at a different part of the number line, a small amount of computation must
	be done for each prime prior to sieving to figure out where to start.  After sieving
	is done, each thread counts primes in part of the shared memory space; the final count
	is returned in the provided array for each block.  The host cpu will do the final sum
	over blocks.  Note, it would not be much more difficult to compute and return the primes
	in the block instead of just the count, but it would be slower due to the extra
	memory transfer required.
	*/
	
	uint32 i,j,k;
	uint32 bid = blockIdx.y * gridDim.x + blockIdx.x;	
	uint32 range = block_size / threadsPerBlock;
	__shared__ uint8 locsieve[block_size];
	__shared__ uint32 bitsieve[block_size / 32];
	

	// everyone init the array.
	if ((bid+1)*block_size > maxID)
	{
		for (j=threadIdx.x * range, k=0; k<range; k++)
		{
			// we're counting hits in the kernel as well, so clear the bytes representing primes > N
			if ((bid * block_size + j + k) < maxID)
				locsieve[j+k] = 1;
			else
				locsieve[j+k] = 0;
		}
	}
	else
	{
		for (j=threadIdx.x * range/4, k=0; k<range/4; k++)
		{
			((uint32 *) locsieve)[j+k] = 0x01010101;
		}
	}
	
	// the smallest primes are dealt with a bit differently.  They are sieved in a separate
	// shared memory space in a packed bit array.  constant memory holds pre-computed
	// information about where each prime lands within a given 32 bit region.  each thread
	// in the block will use this info to simultaneously sieve a small portion of the 
	// packed bit array (that way we make use of the broadcast capabilities of constant memory).
	// When counting or computing primes, we then have to check both the packed bit array as
	// well as the regular byte array, but overall it is a win to greatly speed up the 
	// sieving of the smallest primes.
	
	// compute starting offset for prime 5:
	i = (bid * 256 + threadIdx.x) % 5;
	// then sieve prime 5 in the bit array
	bitsieve[threadIdx.x] = _step5[i];
	
	// compute starting offset for prime 7:
	i = (bid * 256 + threadIdx.x) % 7;
	// then sieve prime 7 in the bit array
	bitsieve[threadIdx.x] |= _step7[i];
	
	// compute starting offset for prime 11:
	i = (bid * 256 + threadIdx.x) % 11;
	// then sieve prime 11 in the bit array
	bitsieve[threadIdx.x] |= _step11[i];
	
	// compute starting offset for prime 13:
	i = (bid * 256 + threadIdx.x) % 13;
	// then sieve prime 13 in the bit array
	bitsieve[threadIdx.x] |= _step13[i];
	
	// compute starting offset for prime 17:
	i = (bid * 256 + threadIdx.x) % 17;
	// then sieve prime 17 in the bit array
	bitsieve[threadIdx.x] |= _step17[i];
	
	// compute starting offset for prime 19:
	i = (bid * 256 + threadIdx.x) % 19;
	// then sieve prime 19 in the bit array
	bitsieve[threadIdx.x] |= _step19[i];
	
	
	// regroup before sieving
	__syncthreads();
		
	// now sieve the array
	for (j=0; j<nump; j++)
	{
		int pid = (j * threadsPerBlock) + threadIdx.x + startprime;

		if (pid < maxp)
		{
			uint32 p = primes[pid];
			uint32 pstart = p/3;
			uint32 p2 = 2*p;
			uint32 block_start = bid * block_size;
			uint32 start_offset;
			uint32 s[2];

			// the wheel sieve with all multiples of 2 and 3 removed from the array is equivalent to
			// alternately stepping through the number line by (p+2)*mult, (p-2)*mult, 
			// where mult = (p+1)/6
			s[0] = p+(2*((p+1)/6));
			s[1] = p-(2*((p+1)/6));
			
			// compute the starting location of this prime in this block
			if ((bid == 0) || (pstart >= block_start))
			{
				// start one increment past the starting value of p/3, since
				// we want to count the prime itself as a prime.
				start_offset = pstart + s[0] - block_start;
				k = 1;				
			}
			else
			{
				// measure how far the start of this block is from where the prime first landed,
				// as well as how many complete (+2/-2) steps it would need to take
				// to cover that distance
				uint32 dist = (block_start - pstart);
				uint32 steps = dist / p2;

				if ((dist % p2) == 0)
				{
					// if the number of steps is exact, then we hit the start
					// of this block exactly, and we start below with the +2 step.
					start_offset = 0;
					k = 0;
				}
				else
				{			
					uint32 inc = pstart + steps * p2 + s[0];
					if (inc >= block_start)
					{
						// if the prime reaches into this block on the first stride,
						// then start below with the -2 step
						start_offset = inc - block_start;
						k = 1;
					}
					else
					{ 
						// we need both +2 and -2 strides to get into the block,
						// so start below with the +2 stride.
						start_offset = inc + s[1] - block_start;
						k = 0;
					}
				}				
			}
			
			// unroll the loop for the smallest primes.
			if (p < 1024)
			{
				uint32 stop = block_size - (2 * p * 4);
				
				if (k == 0)
				{				
					for(i=start_offset ;i < stop; i+=8*p)
					{
						locsieve[i] = 0;
						locsieve[i+s[0]] = 0;
						locsieve[i+p2] = 0;
						locsieve[i+p2+s[0]] = 0;
						locsieve[i+4*p] = 0;
						locsieve[i+4*p+s[0]] = 0;
						locsieve[i+6*p] = 0;
						locsieve[i+6*p+s[0]] = 0;
					}
				}
				else
				{
					for(i=start_offset ;i < stop; i+=8*p)
					{
						locsieve[i] = 0;
						locsieve[i+s[1]] = 0;
						locsieve[i+p2] = 0;
						locsieve[i+p2+s[1]] = 0;
						locsieve[i+4*p] = 0;
						locsieve[i+4*p+s[1]] = 0;
						locsieve[i+6*p] = 0;
						locsieve[i+6*p+s[1]] = 0;
					}
				}
			}
			else
				i=start_offset;
			
			// alternate stepping between the large and small strides this prime takes.
			for( ;i < block_size; k = !k)
			{
				locsieve[i] = 0;
				i += s[k];
			}
		}
	}
	
	// regroup before counting
	__syncthreads();

	// each thread sum a range of the array.
	// we can't sum the whole array in one big reduction because the result
	// won't fit in a uint8.  so do this partial sum so that the final reduction
	// is over 32x less data.
	// for a value to be prime, the regular byte array must be 1, and corresponding bit in
	// the packet bit array must be zero.
	/*if(bid == 1)
	{
	  for(j = 0; j < block_size; j++)
	  {
	    results[j + 32 * threadIdx.x] = locsieve[j + 32 * threadIdx.x];
	  }
	}*/
	//j=threadIdx.x * range;
	//locsieve[j] = (locsieve[j] & ((bitsieve[j >> 5] & (1 << (j & 31))) == 0));
	//results[bid * block_size + j] = locsieve[j];
	for (j=threadIdx.x * range, k=0; k<range; k++)
		{
		  locsieve[j + k] = (locsieve[j+k] & ((bitsieve[(j+k) >> 5] & (1 << ((j+k) & 31))) == 0));
	    //results[bid * block_size + j + k] = locsieve[j + k];
	    //locsieve[threadIdx.x * range] += locsieve[j + k];
	    //locsieve[threadIdx.x * range] += locsieve[j + k];
	  }
	bitsieve[threadIdx.x] = locsieve[threadIdx.x * range];
	for (j=threadIdx.x * range + 1, k = 0; k<range-1; k++)
	    bitsieve[threadIdx.x] += locsieve[j + k];
	// regroup before reducing
	__syncthreads();
	
	// finally return the total count in global memory
	if (threadIdx.x == 0)
	{		
		k = 0;
		// we could do a thread parallel logrithmic reduction here, but with
		// only 256 elements to reduce, it probably won't matter.
		//for (j=0; j<block_size; j+=range)
		for (j=0; j < block_size/range; j++)
			k += bitsieve[j];
		count[bid] = k;
	}
	if(threadIdx.x == 0 && (bid * block_size + j + k) < maxID)
	  for (k=0; k<block_size; k++)
	  {
	    j = (k * 3 + 1 + (k & 1)) >> 1;
	    results[((bid * block_size * 3) >> 1) + j] = locsieve[k];
	  }
}

uint32 tiny_soe(uint32 limit, uint32 *primes)
{
	//simple sieve of erathosthenes for small limits - not efficient
	//for large limits.
	uint8 *flags;
	uint16 prime;
	uint32 i,j;
	int it;

	//allocate flags
	flags = (uint8 *)malloc(limit/2 * sizeof(uint8));
	if (flags == NULL)
		printf("error allocating flags\n");
	memset(flags,1,limit/2);

	//find the sieving primes, don't bother with offsets, we'll need to find those
	//separately for each line in the main sieve.
	primes[0] = 2;
	it=1;
	
	//sieve using primes less than the sqrt of the desired limit
	//flags are created only for odd numbers (mod2)
	for (i=1;i<(uint32)(sqrt(limit)/2+1);i++)
	{
		if (flags[i] > 0)
		{
			prime = (uint32)(2*i + 1);
			for (j=i+prime;j<limit/2;j+=prime)
				flags[j]=0;

			primes[it]=prime;
			it++;
		}
	}

	//now find the rest of the prime flags and compute the sieving primes
	for (;i<limit/2;i++)
	{
		if (flags[i] == 1)
		{
			primes[it] = (uint32)(2*i + 1);
			it++;
		}
	}

	free(flags);
	return it;
}

/*bool InitCUDA(void)
{
  int count = 0;
  int i = 0;
  
  cudaGetDeviceCount(&count);
  if(count == 0) {
    fprintf(stderr, "There is no device.\n");
    return false;
  }

  for(i = 0; i < count; i++) {
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
      if(prop.major >= 1) {
        printf("Device %d supports CUDA %d.%d\n",i, prop.major, prop.minor);
        printf("It has warp size %d, %d regs per block, %d threads per block\n",prop.warpSize, prop.regsPerBlock, prop.maxThreadsPerBlock);
        printf("max Threads %d x %d x %d\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("max Grid %d x %d x %d\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("total constant memory %d\n",prop.totalConstMem);
        break;
      }
    }
  }
  if(i == count) {
    fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
    return false;
  }
  cudaSetDevice(i);
  return true;
}*/

int gtpr(int n, uint8* bprimes)
{
	uint64 N = 100000;
	uint32 Nsmall;
	int numblocks;
	int primes_per_thread;
	uint64 array_size;
	uint32* primes;
	uint32* device_primes;
	uint32* block_counts;
	uint32 np;
	//unsigned int thandle; 
	uint32* block_counts_on_host;
	uint8* results;
	//uint8* bprimes;
	
	// handle input arguments
	//if (argc > 1)
	//	N = strtoull(argv[1],NULL,10);
	N = n;
	Nsmall = (uint32)sqrt(N);
	 
	if (N > 10000000000)
	{
		printf("input range too large, limit is 10e9");
		exit(0);
	}
	
	// look for a card and properties
	//if(!InitCUDA())
	//{
	//	return 0;
	//} 
	
	// find seed primes
	primes = (uint32*)malloc(Nsmall*sizeof(uint32));
	np = tiny_soe(Nsmall, primes);
	//printf("%d small primes (< %d) found\n", np, Nsmall);
	
	// put the primes on the device
	cudaMalloc((void**) &device_primes, sizeof(uint32) * np);
	cudaMemcpy(device_primes, primes, sizeof(uint32)*np, cudaMemcpyHostToDevice);

	// compute how many whole blocks we have to sieve and how many primes each
	// thread will be responsible for.
	array_size = (N / 3 / block_size + 1) * block_size;
	numblocks = array_size / block_size;
	primes_per_thread = ((np - startprime) + threadsPerBlock - 1) / threadsPerBlock;
	//printf("using grid of %dx%d blocks with %d threads per block and %d primes per thread\n",
		//(int)sqrt(numblocks)+1, (int)sqrt(numblocks)+1, threadsPerBlock, primes_per_thread);
	//printf("sieved blocks have %d extra flags\n", 
		//((int)sqrt(numblocks)+1)*((int)sqrt(numblocks)+1)*block_size*3 - N);
	dim3 grid((uint32)sqrt(numblocks)+1,(uint32)sqrt(numblocks)+1);
	
	// init result array of block counts
	//printf("number of blocks: %d\n", numblocks);
	cudaMalloc((void**) &results, sizeof(uint8) * (N >> 1));
	cudaMemset(results, 0, sizeof(uint8) * (N >> 1));
	//bprimes = (uint8*)malloc(array_size*sizeof(uint8));
	cudaMalloc((void**) &block_counts, sizeof(uint32) * numblocks);
	cudaMemset(block_counts, 0, sizeof(uint32) * numblocks);
	
	//cutCreateTimer(&thandle);
	//cutStartTimer(thandle);

	SegSieve<<<grid, threadsPerBlock, 0>>>(block_counts, device_primes, np, primes_per_thread, (N+1)/3, results);

	cudaThreadSynchronize();  
	//cutStopTimer(thandle);
	//printf("%f milliseconds for big sieve\n",cutGetTimerValue(thandle)); 

	block_counts_on_host = (uint32 *)malloc(numblocks * sizeof(uint32));
	cudaMemcpy(block_counts_on_host, block_counts, sizeof(uint32) * numblocks, cudaMemcpyDeviceToHost);
	cudaMemcpy (bprimes, results, sizeof (uint8) * (n >> 1), cudaMemcpyDeviceToHost);

	cudaFree(device_primes);
	cudaFree(block_counts);
	cudaFree(results);
	
	uint32 nbig = startprime-1;		// start here because we aren't sieving 3, and the sieve for
									// 5,7,11,13,17 and 19 is special and crosses off those primes.
	for (int i=0; i<numblocks; i++)
	{
		//printf("found %u primes in block %d\n", block_counts_on_host[i], i);
		nbig += block_counts_on_host[i];
	}
	//for (int i = 0; i < (N+1)/3; i++)
	//{
	//	printf("%d", bprimes[i]);
	//	if(i % 64 == 63) printf("\n");
	//}
		
	//printf("\n%u big primes (< %llu) found\n",nbig,N);

	free(block_counts_on_host);
	free(primes);
	//free(bprimes);
	
	return 0;
}

/**************************************************************
 *
 *      FFT and other related Functions
 *
 **************************************************************/
/* rint is not ANSI compatible, so we need a definition for 
 * WIN32 and other platforms with rint.
 * Also we use that to write the trick to rint()
 */

/****************************************************************************
 *           Lucas Test - specific routines                                 *
 ***************************************************************************/


float
lucas_square (double *x, int q, int n, int iter, int last, float* maxerr, int error_flag, int tib, int stage)
{
  float terr = 0.0;
  
 {
    cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
    square <<< n / 512, 128 >>> (n, g_x, g_ct);
    cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
    norm1 <<<n / (threads), threads >>> 
                    (g_x, g_data, g_ttp, g_ttmp, g_numbits, g_err, *maxerr, tib);
    normalize2_kernel <<< ((n + threads - 1) / threads + 127) / 128, 128 >>> 
                     (g_x, n, threads, g_data, g_ttp1, tib);
    if(tib)
    {
      normalize3_kernel <<<n / threads, threads >>> 
                    (g_x, g_data, g_ttp, g_numbits, g_err, *maxerr);
      normalize4_kernel <<< ((n + threads - 1) / threads + 127) / 128, 128 >>> 
                     (g_x, n, threads, g_data, g_ttp1);
    }
  }
  if (error_flag)
  {
    cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
    if(terr > *maxerr) *maxerr = terr;
  }
  else if (polite_f && (iter % polite) == 0) cutilSafeThreadSync();
  return (terr);
}

void init_x(double *x, unsigned *x_packed, int q, int n, int *offset)
{
  int j;

  if(*offset < 0)
  { 
    *offset = 0;
    for(j = 0; j < n; j++) x[j] = 0.0;
    x[0] = 1.0;
    if(x_packed)
    {
      for(j = 0; j < (q + 31) /32; j++) x_packed[j] = 0;
      x_packed[0] = 1;
    }
    cudaMemcpy (g_x, x, sizeof (double) * n , cudaMemcpyHostToDevice);
  }
}

void E_init_d(double *g, double value, int length)
{
  double x[1] = {value};
  
  cutilSafeCall (cudaMemset (g, 0.0, sizeof (double) * length));
  cudaMemcpy (g, x, sizeof (double) , cudaMemcpyHostToDevice);
}

void E_pre_mul(double *g_out, double *g_in, int length)
{
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_in, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
  pre_mul <<<length / 512, 128>>> (length, g_out, g_ct);
}

void E_mul(double *g_out, double *g_in1, double *g_in2, int length)
{
  float maxerr = 0.3f;
  
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_in1, (cufftDoubleComplex *) g_in1, CUFFT_INVERSE));
  mult3 <<<length / 512, 128>>> (g_out, g_in1, g_in2, g_ct, length);
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
  norm1 <<<length / threads, threads >>> 
                    (g_out, g_data, g_ttp, g_ttmp, g_numbits, g_err, maxerr, 0);
  normalize2_kernel <<< ((length + threads - 1) / threads + 127) / 128, 128 >>> 
                     (g_out, length, threads, g_data, g_ttp1, 0);
}

void E_sub_mul(double *g_out, double *g_in1, double *g_in2, double *g_in3, int length)
{
  float maxerr = 0.3f;
  
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_in1, (cufftDoubleComplex *) g_in1, CUFFT_INVERSE));
  sub_mul <<<length / 512, 128>>> (g_out, g_in1, g_in2, g_in3, g_ct, length);
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
  norm1 <<<length / threads, threads >>> 
                    (g_out, g_data, g_ttp, g_ttmp, g_numbits, g_err, maxerr, 0);
  normalize2_kernel <<< ((length + threads - 1) / threads + 127) / 128, 128 >>> 
                     (g_out, length, threads, g_data, g_ttp1, 0);
}

void E_half_mul(double *g_out, double *g_in1, double *g_in2, int length)
{
  float maxerr = 0.3f;
  
  mult2 <<<length / 512, 128>>> (g_out, g_in1, g_in2, g_ct, length);
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
  norm1 <<<length / threads, threads >>> 
                    (g_out, g_data, g_ttp, g_ttmp, g_numbits, g_err, maxerr, 0);
  normalize2_kernel <<< ((length + threads - 1) / threads + 127) / 128, 128 >>> 
                     (g_out, length, threads, g_data, g_ttp1, 0);
}

void E_to_the_p(double *g_out, double *g_in, mpz_t p, int n)
{
  
  int last, j;
  float maxerr = 0.3f;
  
  last = mpz_sizeinbase (p, 2);

  E_pre_mul(g_save, g_in, n);
  if(g_out != g_in) copy_kernel<<<n / threads, threads>>>(g_out, g_in);

  for(j = 2; j <= last; j++)
  {
    cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
    square <<< n / 512, 128 >>> (n, g_out, g_ct);
    cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
    norm1 <<<n / threads, threads >>> 
                    (g_out, g_data, g_ttp, g_ttmp, g_numbits, g_err, maxerr, 0);
    normalize2_kernel <<< ((n + threads - 1) / threads + 127) / 128, 128 >>> 
                     (g_out, n, threads, g_data, g_ttp1, 0);
    if(mpz_tstbit (p, last - j)) 
    {
      cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
      mult2 <<< n / 512, 128 >>> (g_out, g_out, g_save, g_ct, n);
      cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_out, (cufftDoubleComplex *) g_out, CUFFT_INVERSE));
      norm1 <<<n / threads, threads >>> 
                    (g_out, g_data, g_ttp, g_ttmp, g_numbits, g_err, maxerr, 0);
      normalize2_kernel <<< ((n + threads - 1) / threads + 127) / 128, 128 >>> 
                     (g_out, n, threads, g_data, g_ttp1, 0);
    }                 
  }
}

/* -------- initializing routines -------- */
void
makect (int nc, double *c)
{
  int j;
  double d = (double) (nc << 1);

  for (j = 1; j <= nc; j++) c[j] = 0.5 * cospi (j / d);
}

void get_weights(int q, int n)
{
  int a, b, c, bj, j;

  ttmp = (double *) malloc (sizeof (double) * (n));
  ttp = (double *) malloc (sizeof (double) * (n));
  size = (char *) malloc (sizeof (char) * n);

  b = q % n;
  c = n - b;
  ttmp[0] = 1.0;
  ttp[0] = 1.0;
  bj = 0;
  for (j = 1; j < n; j++)
  {
    bj += b;
    bj %= n;
    a = bj - n;
    ttmp[j] = exp2 (a / (double) n);
    ttp[j] = exp2 (-a / (double) n);
    size[j] = (bj >= c);
  }
  size[0] = 1;
  size[n-1] = 0;
}

void alloc_gpu_mem(int n)
{
  cufftSafeCall (cufftPlan1d (&plan, n / 2, CUFFT_Z2Z, 1));
  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * n ));
  cutilSafeCall (cudaMalloc ((void **) &g_y, sizeof (double) * n ));
  cutilSafeCall (cudaMalloc ((void **) &g_ct, sizeof (double) * n / 4));
  cutilSafeCall (cudaMalloc ((void **) &g_err, sizeof (float)));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ttmp, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_numbits, sizeof (char) * n));
  //cutilSafeCall (cudaMalloc ((void **) &g_rp, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp1, sizeof (double) * 2 * n / threads));
  cutilSafeCall (cudaMalloc ((void **) &g_data, sizeof (int) * 4 * n / threads));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  cutilSafeCall (cudaMalloc ((void **) &g_save, sizeof (double) * n));
}

void write_gpu_data(int q, int n)
{
  double *s_ttp, *s_ttmp, *s_ttp1, *s_ct;
  char *s_numbits;
  int *s_data;
  int i, j, qn = q / n;

  s_ct = (double *) malloc (sizeof (double) * (n / 4));
  s_ttp = (double *) malloc (sizeof (double) * (n));
  s_ttmp = (double *) malloc (sizeof (double) * (n));
  s_numbits = (char *) malloc (sizeof (char) * (n));
  s_ttp1 = (double *) malloc (sizeof (double) * 2 * (n / threads));
  s_data = (int *) malloc (sizeof (int) * (4 * n / threads));

  for (j = 0; j < n; j++)
  {
    s_ttp[j] = ttp[j];//1 / ttmp[j];
    s_ttmp[j] = ttmp[j] * 2.0 / n; 
    if(j % 2) s_ttmp[j] = -s_ttmp[j]; 
    s_numbits[j] = qn + size[j];
  }

  for (i = 0, j = 0; i < n; i++)
  {
    if ((i % threads) == 0)
    {
      s_ttp1[2 * j] = s_ttp[i];
      s_ttp1[2 * j + 1] = s_ttp[i + 1];
      s_data[4 * j + 3] = s_numbits[i];
      j++;
    }
  }

  makect (n / 4, s_ct);
  
  cudaMemcpy (g_ttmp, s_ttmp, sizeof (double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy (g_numbits, s_numbits, sizeof (char) * n, cudaMemcpyHostToDevice);
  cudaMemcpy (g_ttp, s_ttp, sizeof (double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy (g_ttp1, s_ttp1, sizeof (double) * 2 * n / threads, cudaMemcpyHostToDevice);
  cudaMemcpy (g_data, s_data, sizeof (int) * 4 * n / threads, cudaMemcpyHostToDevice);
  cudaMemcpy (g_ct, s_ct, sizeof (double) * (n / 4), cudaMemcpyHostToDevice);

  free ((char *) s_ct);
  free ((char *) s_ttp);
  free ((char *) s_ttmp);
  free ((char *) s_ttp1);
  free ((char *) s_data);
  free ((char *) s_numbits);
}

void free_host (double *x)
{
  free ((char *) size);
  free ((char *) x);
  free ((char *) ttmp);
  free ((char *) ttp);
}

void free_gpu(void)
{
  cufftSafeCall (cufftDestroy (plan));
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_y));
  cutilSafeCall (cudaFree ((char *) g_ct));
  cutilSafeCall (cudaFree ((char *) g_err));
  cutilSafeCall (cudaFree ((char *) g_ttp));
  cutilSafeCall (cudaFree ((char *) g_ttp1));
  cutilSafeCall (cudaFree ((char *) e_data));
  cutilSafeCall (cudaFree ((char *) rp_data));
  cutilSafeCall (cudaFree ((char *) g_ttmp));
  cutilSafeCall (cudaFree ((char *) g_data));
  cutilSafeCall (cudaFree ((char *) g_numbits));
  cutilSafeCall (cudaFree ((char *) g_save));
}

void close_lucas (double *x)
{
  free_host(x);
  free_gpu();
}

void reset_err(float* maxerr, float value)
{
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  *maxerr *= value;
}


/**************************************************************************
 *                                                                        *
 *       End LL/GPU Functions, Begin Utility/CPU Functions                *
 *                                                                        *
 **************************************************************************/

int
choose_fft_length (int q, int* index)
{  
/* In order to increase length if an exponent has a round off issue, we use an
extra paramter that we can adjust on the fly. In check(), index starts as -1,
the default. In that case, choose from the table. If index >= 0, we must assume
it's an override index and return the corresponding length. If index > table-count,
then we assume it's a manual fftlen and return the proper index. */
  #define COUNT 119
  int multipliers[COUNT] = {  6,     8,    12,    16,    18,    24,    32,    
                             40,    48,    64,    72,    80,    96,   120,   
                            128,   144,   160,   192,   224,   240,   256,   
                            288,   320,   336,   384,   448,   480,   512,   
                            576,   640,   672,   768,   800,   864,   896,   
                            960,  1024,  1120,  1152,  1200,  1280,  1344,
                           1440,  1568,  1600,  1680,  1728,  1792,  1920, 
                           2048,  2240,  2304,  2400,  2560,  2688,  2880,  
                           3072,  3200,  3360,  3456,  3584,  3840,  4000,  
                           4096,  4480,  4608,  4800,  5120,  5376,  5600,  
                           5760,  6144,  6400,  6720,  6912,  7168,  7680,  
                           8000,  8192,  8960,  9216,  9600, 10240, 10752, 
                          11200, 11520, 12288, 12800, 13440, 13824, 14366, 
                          15360, 16000, 16128, 16384, 17920, 18432, 19200, 
                          20480, 21504, 22400, 23040, 24576, 25600, 26880, 
                          29672, 30720, 32000, 32768, 34992, 36864, 38400,
                          40960, 46080, 49152, 51200, 55296, 61440, 65536  };
  // Largely copied from Prime95's jump tables, up to 32M
  // Support up to 64M, the maximum length with threads == 1024
   if( 0 < *index && *index < COUNT ) // override
    return 1024*multipliers[*index];  
  else if( *index >= COUNT || q == 0) 
  { /* override with manual fftlen passed as arg; set pointer to largest index <= fftlen */
    int len, i;
    for(i = COUNT - 1; i >= 0; i--)
    {
      len = 1024*multipliers[i];
      if( len <= *index )
      {
        *index = i;
        return len; /* not really necessary, but now we could decide to override ftlen with this value here */
      }
    }
  }
  else  
  { // *index < 0, not override, choose length and set pointer to proper index
    int len, i, estimate = q / 18; // reduce to 18, perhaps even 17??
    for(i = 0; i < COUNT; i++)
    {
      len = 1024*multipliers[i];
      if( len >= estimate ) 
      {
        *index = i;
        return len;
      }
    }
  }
  return 0;
}

int fft_from_str(const char* str)
/* This is really just strtoul with some extra magic to deal with K or M */
{
  char* endptr;
  const char* ptr = str;
  int len, mult = 0;
  while( *ptr ) {
    if( *ptr == 'k' || *ptr == 'K' ) {
      mult = 1024;
      break;
    }
    if( *ptr == 'm' || *ptr == 'M' ) {
      mult = 1024*1024;
      break;
    }
    ptr++;
  }
  if( !mult ) { // No K or M, treat as before    (PS The Python else clause on loops I mention in parse.c would be useful here :) )
    mult = 1;
  }
  len = (int) strtoul(str, &endptr, 10)*mult;
  if( endptr != ptr ) { // The K or M must directly follow the num (or the num must extend to the end of the str)
    fprintf (stderr, "can't parse fft length \"%s\"\n\n", str);
    exit (2);
  }
  return len;
}

//From apsen
void
print_time_from_seconds (int sec)
{
  if (sec > 3600)
    {
      printf ("%d", sec / 3600);
      sec %= 3600;
      printf (":%02d", sec / 60);
    }
  else
    printf ("%d", sec / 60);
  sec %= 60;
  printf (":%02d", sec);
}

void
init_device (int device_number)
{
  int device_count = 0;
  cudaGetDeviceCount (&device_count);
  if (device_number >= device_count)
    {
      printf ("device_number >=  device_count ... exiting\n");
      printf ("(This is probably a driver problem)\n\n");
      exit (2);
    }
  if (d_f)
    {
      cudaDeviceProp dev;
      cudaGetDeviceProperties (&dev, device_number);
      printf ("------- DEVICE %d -------\n", device_number);
      printf ("name                %s\n", dev.name);
      printf ("totalGlobalMem      %d\n", (int) dev.totalGlobalMem);
      printf ("sharedMemPerBlock   %d\n", (int) dev.sharedMemPerBlock);
      printf ("regsPerBlock        %d\n", (int) dev.regsPerBlock);
      printf ("warpSize            %d\n", (int) dev.warpSize);
      printf ("memPitch            %d\n", (int) dev.memPitch);
      printf ("maxThreadsPerBlock  %d\n", (int) dev.maxThreadsPerBlock);
      printf ("maxThreadsDim[3]    %d,%d,%d\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
      printf ("maxGridSize[3]      %d,%d,%d\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
      printf ("totalConstMem       %d\n", (int) dev.totalConstMem);
      printf ("Compatibility       %d.%d\n", dev.major, dev.minor);
      printf ("clockRate (MHz)     %d\n", dev.clockRate/1000);
      printf ("textureAlignment    %d\n", (int) dev.textureAlignment);
      printf ("deviceOverlap       %d\n", dev.deviceOverlap);
      printf ("multiProcessorCount %d\n\n", dev.multiProcessorCount);
      // From Iain
      if (dev.major == 1 && dev.minor < 3)
      {
        printf("A GPU with compute capability >= 1.3 is required for double precision arithmetic\n\n");
        exit (2);
      }
    }
  cudaSetDeviceFlags (cudaDeviceBlockingSync);
  cudaSetDevice (device_number);
}


void
rm_checkpoint (int q)
{
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c" "%d", q);
  sprintf (chkpnt_tfn, "t" "%d", q);
  (void) unlink (chkpnt_cfn);
  (void) unlink (chkpnt_tfn);
}


int standardize_digits (double *x, int q, int n, int offset, int num_digits)  
{
  int j, digit, stop, qn = q / n;
  double temp, carry = 0.0;
  double lo = (double) (1 << qn);
  double hi = lo + lo;

  digit = floor(offset * (n / (double) q));
  j = (n + digit - 1) % n;
  while(RINT_x86(x[j]) == 0.0 && j != digit) j = (n + j - 1) % n;
  if(j == digit && RINT_x86(x[digit]) == 0.0) return(1);
  else if (x[j] < 0.0) carry = -1.0;
  {  
    stop = (digit + num_digits) % n;
    j = digit;
    do
    {
      x[j] = RINT_x86(x[j] * ttmp[j]) + carry;
      carry = 0.0;
      if (size[j]) temp = hi;
      else temp = lo;
      if(x[j] < -0.5)
      {
        x[j] += temp;
        carry = -1.0;
      }
      j = (j + 1) % n;
    }
    while(j != stop);
  }
  return(0);
}

void balance_digits(double* x, int q, int n)
{
  double half_low = (double) (1 << (q / n - 1));
  double low = 2.0 * half_low;
  double high = 2.0 * low;
  double upper, adj, carry = 0.0;
  int j;
 
  for(j = 0; j < n; j++)
  { 
    if(size[j])
    {
      upper = low - 0.5;
      adj = high;
    }
    else
    {
      upper = half_low - 0.5;
      adj = low;
    }
    x[j] += carry;
    carry = 0.0;
    if(x[j] > upper)
    {
      x[j] -= adj;
      carry = 1.0;
    }
    x[j] /= ttmp[j];
  }
  x[0] += carry; // Good enough for our purposes.
}


unsigned *
read_checkpoint_packed (int q)
{
  struct stat FileAttrib;
  FILE *fPtr;
  unsigned *x_packed;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  int end = (q + 31) / 32;
  
  x_packed = (unsigned *) malloc (sizeof (unsigned) * (end + 10));
  
  sprintf (chkpnt_cfn, "c%d", q);
  sprintf (chkpnt_tfn, "t%d", q);
  fPtr = fopen (chkpnt_cfn, "rb");
  if (!fPtr)
  {
    if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the checkpoint file. Trying the backup file.\n");
  }
  else if (fread (x_packed, 1, sizeof (unsigned) * (end + 10) , fPtr) != (sizeof (unsigned) * (end + 10)))
  {
    fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
    fclose (fPtr);
  }
  else if(x_packed[end] != q)
  {
    fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
    fclose(fPtr);
  }
  else  
  {
    fclose(fPtr);
    return x_packed;
  }  
  fPtr = fopen(chkpnt_tfn, "rb");
  if (!fPtr)
  {
    if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the backup file. Restarting test.\n");
  }
  else if (fread (x_packed, 1, sizeof (unsigned) * (end + 10) , fPtr) != (sizeof (unsigned) * (end + 10)))
  {
    fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
    fclose (fPtr);
  }
  else if(x_packed[end] != q)
  {
    fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");;
    fclose(fPtr);
  }
  else
  {
    fclose(fPtr);
    return x_packed;
  }  
  x_packed[end + 1] = 0;
  x_packed[end + 2] = 1;
  x_packed[end + 3] = (unsigned) -1;
  x_packed[end + 4] = 0;
  return x_packed;
}

void pack_bits(double *x, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int i, j = 0, k = 0;
  int qn = q / n;

  for(i = 0; i < n; i++)
  {
    temp1 = (int) x[i];
    temp2 += (temp1 << k);
    k += qn + size[i];
    if(k >= 32)
    {
      packed_x[j] = (unsigned) temp2;
      temp2 >>= 32;
      k -= 32;
      j++;
    }
  }
  packed_x[j] = (unsigned) temp2;  
}

void set_checkpoint_data(unsigned *x_packed, int q, int n, int j, int offset, int time)
{
  int end = (q + 31) / 32;

  x_packed[end + 5] = q;
  x_packed[end + 6] = n;
  x_packed[end + 7] = j;
  x_packed[end + 8] = offset;
  x_packed[end + 9] = time;
}

void
commit_checkpoint_packed (double *x, unsigned *x_packed, int q, int n)
{
  int i;
  int end = (q + 31) / 32;

  for(i = 0; i < 5; i++) x_packed[end + i] = x_packed[end + i + 5];
  pack_bits(x, x_packed, q, n);
  
}

void
write_checkpoint_packed (unsigned *x_packed, int q)
{
  //FILE *fPtr;
  //char chkpnt_cfn[32];
  //char chkpnt_tfn[32];
  
  /*sprintf (chkpnt_cfn, "c%d", q);
  sprintf (chkpnt_tfn, "t%d", q);
  (void) unlink (chkpnt_tfn);
  (void) rename (chkpnt_cfn, chkpnt_tfn);
  fPtr = fopen (chkpnt_cfn, "wb");
  if (!fPtr) 
  {
    fprintf(stderr, "Couldn't write checkpoint.\n");
    return;
  }
  fwrite (x_packed, 1, sizeof (unsigned) * (((q + 31) / 32) + 10), fPtr);
  fclose (fPtr);*/
 /* if (s_f > 0)			// save all checkpoint files
    {
      char chkpnt_sfn[64];
#ifndef _MSC_VER
      sprintf (chkpnt_sfn, "%s/s" "%d.%d.%s", folder, q, j, s_residue);
#else
      sprintf (chkpnt_sfn, "%s\\s" "%d.%d.%s.txt", folder, q, j, s_residue);
#endif
      fPtr = fopen (chkpnt_sfn, "wb");
      if (!fPtr) return;
      fwrite (&q, 1, sizeof (q), fPtr);
      fwrite (&n, 1, sizeof (n), fPtr);
      fwrite (&j, 1, sizeof (j), fPtr);
      fwrite (x, 1, sizeof (double) * n, fPtr);
      fwrite (&total_time, 1, sizeof(total_time), fPtr);
      fwrite (&offset, 1, sizeof(offset), fPtr);
      fclose (fPtr);
    }*/
}




int
printbits (double *x, int q, int n, int offset, FILE* fp, char *expectedResidue, int o_f)
{ 
  int j, k = 0;
  int digit, bit;
  unsigned long long temp, residue = 0;

    digit = floor(offset *  (n / (double) q));
    bit = offset - ceil(digit * (q / (double) n));
    j = digit;
    while(k < 64)
    {
      temp = (int) x[j];
      residue = residue + (temp << k);
      k += q / n + size[j % n];
      if(j == digit) 
      {  
         k -= bit;
         residue >>= bit;
      }
      j = (j + 1) % n;
    }
    sprintf (s_residue, "%016llx", residue);

    printf ("M%d, 0x%s,", q, s_residue);
    if(o_f) printf(" offset = %d,", offset); 
    printf (" n = %dK, %s", n/1024, program);
    if (fp)
    {
      fprintf (fp, "M%d, 0x%s,", q, s_residue);
      if(o_f) fprintf(fp, " offset = %d,", offset); 
      fprintf (fp, " n = %dK, %s", n/1024, program);
    }
  return 0;
}

void unpack_bits(double *x, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int temp3;
  int i, j = 0, k = 0;
  int qn = q / n;
  int mask1 = -1 << (qn + 1);
  int mask2;
  int mask;
  
  mask1 = ~mask1;
  mask2 = mask1 >> 1;
  for(i = 0; i < n; i++)
  {
    if(k < qn + size[i])
    {
      temp1 = packed_x[j];
      temp2 += (temp1 << k);
      k += 32;
      j++;
    }
    if(size[i]) mask = mask1;
    else mask = mask2;
    temp3 = ((int) temp2) & mask;
    x[i] = (double) temp3;
    temp2 >>= (qn + size[i]);
    k -= (qn + size[i]);
  }
}

/*void unpack_bits1(double *x, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int temp3;
  int i, j = 0, k = 0;
  int qn = q / n;
  int mask1 = -1 << (qn + 1);
  int mask2;
  int mask;
  int total = 0;
  
  mask1 = ~mask1;
  mask2 = mask1 >> 1;
  for(i = 0; i < n; i++)
  {
    if(k < qn + size[i])
    {
      temp1 = packed_x[j];
      temp2 += (temp1 << k);
      k += 32;
      j++;
    }
    if(size[i]) mask = mask1;
    else mask = mask2;
    temp3 = ((int) temp2) & mask;
    if(temp3 != (int) x[i]) total++;
    temp2 >>= (qn + size[i]);
    k -= (qn + size[i]);
  }
  printf("Bad digits: %d\n", total);
}*/

double* init_lucas_packed(unsigned * x_packed, int q , int *n, int *j, int *offset, int *total_time)
{
  double *x;
  int new_n, old_n;
  int end = (q + 31) / 32;
  int new_test = 0;
  
  *n = x_packed[end + 1];
  if(*n == 0) new_test = 1;
  *j = x_packed[end + 2];
  *offset = x_packed[end + 3];
  if(total_time) *total_time = x_packed[end + 4];
  
  old_n = fftlen; 
  if(fftlen == 0) fftlen = *n;
  new_n = choose_fft_length(q, &fftlen); 
  if(old_n > COUNT) *n = old_n;
  else if (new_test || old_n) *n = new_n;

  if ((*n / threads) > 65535)
  {
    fprintf (stderr, "over specifications Grid = %d\n", (int) *n / threads);
    fprintf (stderr, "try increasing threads (%d) or decreasing FFT length (%dK)\n\n",  threads, *n / 1024);
    return NULL;
  }
  if (q < *n)
  {
    fprintf (stderr, "The prime %d is less than the fft length %d. This will cause problems.\n\n", q, *n / 1024);
    return NULL;
  }
  x = (double *) malloc (sizeof (double) * *n);
    get_weights(q, *n);
    alloc_gpu_mem(*n);
    write_gpu_data(q, *n);
    if(!new_test)
    {
      unpack_bits(x, x_packed, q, *n);
      balance_digits(x, q, *n);
    }
    init_x(x, x_packed, q, *n, offset);
    return x;
}

void
cufftbench (int cufftbench_s, int cufftbench_e, int cufftbench_d)
{
  cudaEvent_t start, stop;
  double *x;
  float outerTime;
  int i, j;
  printf ("CUFFT bench start = %d end = %d distance = %d\n", cufftbench_s,
	  cufftbench_e, cufftbench_d);

  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * cufftbench_e));
  x = ((double *) malloc (sizeof (double) * cufftbench_e + 1));
  for (i = 0; i <= cufftbench_e; i++)
    x[i] = 0;
  cutilSafeCall (cudaMemcpy
		 (g_x, x, sizeof (double) * cufftbench_e,
		  cudaMemcpyHostToDevice));
  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreate (&stop));
  for (j = cufftbench_s*1024; j <= cufftbench_s*1024; j += 1/*cufftbench_d*/)
    {
      cufftSafeCall (cufftPlan1d (&plan, j / 2, CUFFT_Z2Z, 1));
      cufftSafeCall (cufftExecZ2Z
		     (plan, (cufftDoubleComplex *) g_x,
		      (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
      cutilSafeCall (cudaEventRecord (start, 0));
      for (i = 0; i < 1000; i++)
	cufftSafeCall (cufftExecZ2Z
		       (plan, (cufftDoubleComplex *) g_x,
			(cufftDoubleComplex *) g_x, CUFFT_INVERSE));
      cutilSafeCall (cudaEventRecord (stop, 0));
      cutilSafeCall (cudaEventSynchronize (stop));
      cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
      printf ("CUFFT_Z2Z size= %d time= %f msec\n", j, outerTime / 1000);
      cufftSafeCall (cufftDestroy (plan));
    }
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));
  free ((char *) x);
}

void
SetQuitting (int sig)
{
  quitting = 1;
 sig==SIGINT ? printf( "\tSIGINT") : (sig==SIGTERM ? printf( "\tSIGTERM") : printf( "\tUnknown signal")) ;
 printf( " caught, writing checkpoint.");
}

#ifndef _MSC_VER
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
int
_kbhit (void)
{
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr (STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr (STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl (STDIN_FILENO, F_GETFL, 0);
  fcntl (STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar ();

  tcsetattr (STDIN_FILENO, TCSANOW, &oldt);
  fcntl (STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF)
    {
      ungetc (ch, stdin);
      return 1;
    }

  return 0;
}
#else
#include <conio.h>
#endif

int interact(void); // defined below everything else

int get_bit(int location, unsigned *control)
{
  int digit = location / 32;
  int bit = location % 32;

  bit = 1 << bit;
  bit = control[digit] & bit;
  if(bit) bit /= bit;
  return(bit);
}

int round_off_test(double *x, int q, int n, int *j, int *offset, unsigned *control, int last)
{
  int k;
  float totalerr = 0.0;
  float terr, avgerr, maxerr = 0.0;
  float max_err = 0.0, max_err1 = 0.0;
  int l_offset = *offset;
  int bit;

      printf("Running careful round off test for 1000 iterations. If average error >= 0.25, the test will restart with a longer FFT.\n");
      for (k = 0; k < 1000 && k < last; k++) 
	    {
        bit = get_bit(last - k - 1, control);
        terr = lucas_square (x, q, n, k, last, &maxerr, 1, bit, 1); 
        if(terr > maxerr) maxerr = terr;
        if(terr > max_err) max_err = terr;
        if(terr > max_err1) max_err1 = terr;
        totalerr += terr; 
        reset_err(&maxerr, 0.85);
        if(terr >= 0.43)
        {
	        printf ("Iteration = %d < 1000 && err = %5.5f >= 0.35, increasing n from %dK\n", k, terr, n/1024);
	        fftlen++; 
	        return 1;
        }
	      if(k && (k % 100 == 0)) 
        {
	        printf( "Iteration  %d, average error = %5.5f, max error = %5.5f\n", k, totalerr / k, max_err);
	        max_err = 0.0;
	      }
	    } 
      avgerr = totalerr/1000.0; 
      if( avgerr >= 0.32) 
      {
        printf("Iteration 1000, average error = %5.5f >= 0.25 (max error = %5.5f), increasing FFT length and restarting\n", avgerr, max_err);
        fftlen++; 
        return 1;
      } 
      else if( avgerr < 0 ) 
      {
        fprintf(stderr, "Something's gone terribly wrong! Avgerr = %5.5f < 0 !\n", avgerr);
        exit (2);
      } 
      else 
      {
        printf("Iteration 1000, average error = %5.5f < 0.25 (max error = %5.5f), continuing test.\n", avgerr, max_err1);
        reset_err(&maxerr, 0.0);
      }
      *offset = l_offset;
      *j += 1000;
      return 0;
}    

int isprime(unsigned int n)
{
  unsigned int i;
  
  if(n<=1) return 0;
  if(n>2 && n%2==0)return 0;

  i=3;
  while(i*i <= n && i < 0x10000)
  {
    if(n%i==0)return 0;
    i+=2;
  }
  return 1;
}

unsigned *get_control(int *j, int lim1, int lim2, int q)
{
  mpz_t result;
  int p = 2;
  int limit;
  int prime_power = 1;
  unsigned *control = NULL;
  
	mpz_init(result);
  if(lim2 == 0)
  {
    mpz_set_ui (result, 2 * q);
    limit = lim1 / p;
    while (prime_power <= limit) prime_power *= p;
    mpz_mul_ui(result, result, prime_power);
    p = 3;
    while (p <= lim1)
    {
      while(p <= lim1 && !isprime(p)) p += 2;
      limit = lim1 / p;
      prime_power = p;
      while (prime_power <= limit) prime_power *= p;
      mpz_mul_ui(result, result, prime_power);
      p += 2;
    }
  }
  else
  {
    p = lim1;
    if(!(lim1 & 1)) p++;
    mpz_set_ui (result, 1);
    while (p <= lim2)
    {
      while(p <= lim2 && !isprime(p)) p += 2;
      mpz_mul_ui(result, result, p);
      printf("prime_power: %d, %d\n", prime_power, p);
      p += 2;
    }
    
  }
  *j = mpz_sizeinbase (result, 2);
  control = (unsigned *) malloc (sizeof (unsigned) * ((*j + 31) / 32));
  mpz_export (control, NULL, -1, 4, 0, 0, result);
  mpz_clear (result);       
  return control;
}

int get_gcd(double *x, unsigned *x_packed, int q, int n, int stage)
{
  	  mpz_t result, prime, prime1;
	    int end = (q + 31) / 32;
          int rv = 0;
      
	    mpz_init2( result, q);
	    mpz_init2( prime, q);
	    mpz_init2( prime1, q);
	    mpz_import (result, end, -1, sizeof(x_packed[0]), 0, 0, x_packed);
	    if(stage == 1) mpz_sub_ui (result, result, 1);
	    mpz_setbit (prime, q);
	    mpz_sub_ui (prime, prime, 1);
	    if (mpz_cmp_ui (result, 0))
	    {
	      mpz_gcd (prime1, prime, result);
	      if (mpz_cmp_ui (prime1, 1))
	      {
	        rv = 1;
		printf( "M%d has a factor: ", q);
	        mpz_out_str (stdout, 10, prime1);
                printf (" (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1,g_b2,g_e,n/1024, program);
                FILE* fp = fopen_and_lock(RESULTSFILE, "a");
	        fprintf (fp, "M%d has a factor: ", q);
	        mpz_out_str (fp, 10, prime1);
		if  (AID[0] && strncasecmp(AID, "N/A", 3))
                   fprintf (fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK, aid=%s %s)\n", b1,g_b2,g_e,n/1024, AID, program);
		else
                   fprintf (fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1,g_b2,g_e,n/1024, program);
                unlock_and_fclose(fp);
	      }
	     } 
	   if (rv == 0) {
                printf( "M%d Stage %d found no factor", q, stage);
                printf (" (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1,g_b2,g_e,n/1024, program);
                if (stage==2) {
		  FILE* fp = fopen_and_lock(RESULTSFILE, "a");
                  fprintf (fp, "M%d found no factor", q);
                  if  (AID[0] && strncasecmp(AID, "N/A", 3))
                    fprintf (fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK, aid=%s %s)\n", b1,g_b2,g_e,n/1024, AID, program);
		  else
                    fprintf (fp, " (P-1, B1=%d, B2=%d, e=%d, n=%dK %s)\n", b1,g_b2,g_e,n/1024, program);
                  unlock_and_fclose(fp);
	      }

	    }
	    mpz_clear (result);       
	    mpz_clear (prime);      
	    mpz_clear (prime1);      
      return rv;
}

/**************************************************************
 *
 *      Main Function
 *
 **************************************************************/
void next_base(int e, int n)
{ 
  int j;
  
  E_half_mul(&e_data[(e - 1) * n], &e_data[(e - 1) * n], &e_data[e * n], n);
  for(j = 1; j < e - 1; j++)
  {
    E_mul(&e_data[(e - j - 1) * n], &e_data[(e - j) * n], &e_data[(e - j - 1) * n], n);
  }
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) &e_data[1 * n], (cufftDoubleComplex *) &e_data[1 * n], CUFFT_INVERSE));
  E_half_mul(&e_data[0], &e_data[1 * n], &e_data[0], n);
  E_pre_mul(&e_data[0], &e_data[0], n);
}

int stage2_init_param1(int k, int base, int e, int n)
{
  int i, j;
  mpz_t exponents[e];
  
  for(j = 0; j <= e; j++) mpz_init(exponents[j]);
  for(j = e; j >= 0; j--) mpz_ui_pow_ui (exponents[j], (k - j * 2) * base, e);
  for(j = 0; j < e; j++)
    for(i = e; i > j; i--) mpz_sub(exponents[i], exponents[i-1], exponents[i]);
  for(j = 0; j <= e; j++)
	{
	    //mpz_out_str (stdout, 10, exponents[j]);
	    //printf("\nDoing %d iterations.\n",(int) mpz_sizeinbase (exponents[j], 2) + (int) mpz_popcount(exponents[j]));
      E_to_the_p(&e_data[j * n], g_y, exponents[j], n);
 	    cutilSafeThreadSync();
  }
  for(j = 0; j < e; j++) mpz_clear(exponents[j]);
  E_pre_mul(&e_data[0], &e_data[0], n);
  E_pre_mul(&e_data[e * n], &e_data[e * n], n);
  for(j = 1; j < e; j++) 
    cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) &e_data[j * n], (cufftDoubleComplex *) &e_data[j * n], CUFFT_INVERSE));
  return 1;
}

int stage2_init_param2(int num, int last_rp, int base, int e, int n)
{ 
  int i = 0, rp = last_rp, k, top = 5;
  mpz_t exponent;
  
  mpz_init(exponent);
  while((base % top) == 0)
  {
    top += 2;
    if(top == 9) top += 2;
  }
  while( i < num)
  {
    for (k = 3; k < top; k += 2)
      if(( rp % k) == 0) break;
    if( k == top) 
    {
      mpz_ui_pow_ui (exponent, rp, e);
	    //mpz_out_str (stdout, 10, exponent);
      //printf("  -  %d ",rp);
	    //printf("\nDoing %d iterations.\n",(int) mpz_sizeinbase (exponent, 2) + (int) mpz_popcount(exponent));
      E_to_the_p(&rp_data[i * n], g_y, exponent, n);
      E_pre_mul(&rp_data[i * n], &rp_data[i * n], n);
      cutilSafeThreadSync();
      i++; 
    }
    rp += 2;
  }
  mpz_clear(exponent);
  return rp;
}

int stage2(double *x, unsigned *x_packed, int q, int n)
{
  int j, i, t;
  // int e = 2, d = 2310, b2 = 12035000, nrp = 20;
  int e = g_e, d = g_d, b2 = g_b2, nrp = g_nrp;
  int rpt, rp; 
  int ks, ke, m = 0, k;
  int last = 0;
  uint8 *bprimes = NULL;
  int prime, prime_pair;
  uint8 *rp_gaps = NULL;
  int sprimes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 43, 47, 53, 0};
  uint8 two_to_i[] = {1, 2, 4, 8, 16, 32, 64, 128};
  int count0 = 0, count1 = 0, count2 = 0;
  mpz_t control;
  timeval time0, time1;
  
  
	if(d % 2310 == 0)
	{
	  i = 4;
	  rpt = 480 * d / 2310;
	}
	else if(d % 210 == 0)
	{
	  i = 3;
	  rpt =  48 * d / 210;
	}
	else if(d % 30 == 0)
	{
	  i = 2;
	  rpt = 8 * d / 30;
	}
	else 
	{
	  i = 1;
	  rpt = 2 * d / 6;
	}

  cutilSafeCall (cudaMalloc ((void **) &e_data, sizeof (double) * n * (e + 1)));
  cutilSafeCall (cudaMalloc ((void **) &rp_data, sizeof (double) * n * nrp));
	bprimes = (uint8*) malloc((b2) * sizeof(uint8));
  if(!bprimes) 
  {
    printf("failed to allocate bprimes\n");
    exit (1);
  }
  gtpr(b2, bprimes);

  ks = ((((b2 / sprimes[i] + 1) >> 1) + d - 1) / d - 1) * d;
  ke = ((((b2 + 1) >> 1) + d - 1) / d) * d;

  for( j = (b1 + 1) >> 1; j < ks; j++)
  {
    if(bprimes[j] == 1)
    {
      m = i;
      last = j;
      while(sprimes[m])
      {
        prime = sprimes[m] * j + (sprimes[m] >> 1);
        m++;
        if(prime < ks) continue;
        if(prime > ke) break;
        prime_pair = prime + d - 1 - ((prime % d) << 1);
        bprimes[last] = 0;
        bprimes[prime] = 1;
        if(bprimes[prime_pair]) break;
        last = prime;
      }
    }
  }

	rp_gaps = (uint8*) malloc(rpt * sizeof(uint8));
  if(!rp_gaps) 
  {
    printf("failed to allocate rp_gaps\n");
    exit (1);
  }
  j = 0;
  k = 0;

  for(rp = 1; rp < d; rp += 2)
  {
    k++;
    for (m = 0; m < i; m++)
      if((rp % sprimes[m]) == 0) break;
    if(m == i) 
    {
      rp_gaps[j] = k;
      j++; 
      k = 0;
    }
  }
	
	k = ks + (d >> 1);
  m = k - 1;
  j = 0;
  rp = 0;
  while(m < ke)
  {
    bprimes[rp] = 0;
    for(i = 0; i < 8; i++)
    {
      m += rp_gaps[j];
      k -= rp_gaps[j];
      if (bprimes[m] || bprimes[k])
      {
        bprimes[rp] |= two_to_i[i];
        count1++;
      }
      else count0++;
      if (bprimes[m] && bprimes[k]) count2++;
      //if(m < ks + d) printf("m: %d, k: %d, p[m]: %d, p[k]: %d, rp: %d, p[rp]: %d \n",m,k,bprimes[m],bprimes[k],rp,bprimes[rp]);
      j++;
      if(j == rpt)
      {
        j = 0;
        m += (d >> 1);
        k = m + 1;
      }
    }
    rp++;
  }
  printf("Zeros: %d, Ones: %d, Pairs %d\n", count0, count1, count2);
	/*for (i = 0; i < 180; i++)
	{
		for(j = 0; j < 8; j++)
		printf("%d, ", (bprimes[i] & two_to_i[j]) / two_to_i[j]);
		if(i % 4 == 3) printf("\n");
	}
	printf("\n");*/
	
  mpz_init(control);
  mpz_import(control, (ke - ks) / d * rpt / sizeof(bprimes[0]) , -1, sizeof(bprimes[0]), 0, 0, bprimes);
  t = 0;
  m = 0;
  last = 1;
  ks = ((ks / d) << 1) + 1;
  ke = (ke / d) << 1;
  copy_kernel<<<n / threads, threads>>>(g_y, g_x);
  int num_tran = 1;
  int time;
  float ttime=0;
  gettimeofday (&time1, NULL);
  do
  {
    num_tran = stage2_init_param1(ks, d, e, n);
    last = stage2_init_param2(nrp, last, d, e, n); 
    t = m;
      gettimeofday (&time0, NULL);
      time = 1000000 * (time0.tv_sec - time1.tv_sec) + time0.tv_usec - time1.tv_usec;
      ttime += time / 1000000.0;
      printf("itime: %f, transforms: %d, average: %f\n", time / (float) 1000000, num_tran, time / (float) (num_tran * 1000));
	    num_tran = 0;
    for(k = ks; k < ke; k += 2)
    {
      for(j = 0; j < nrp; j++)
      {
        if(mpz_tstbit (control, t))
        {
          E_sub_mul(g_x, g_x, &e_data[0], &rp_data[j * n], n);
          num_tran += 2;
        }
        t++;
	    }
	    next_base(e, n);
	    cutilSafeThreadSync();
	    num_tran += 2 * e;
	    t += rpt - nrp;
    }
      gettimeofday (&time1, NULL);
      int time = 1000000 * (time1.tv_sec - time0.tv_sec) + time1.tv_usec - time0.tv_usec;
      ttime += time / 1000000.0;
      printf("ptime: %f, transforms: %d, average: %f\n", time / (float) 1000000, num_tran, time / (float) (num_tran * 1000));
      m += nrp;
      printf("ETA: ");
      print_time_from_seconds((int)(ttime * rpt / m - ttime));
      printf("\n");
  }
  while(m < rpt);
  printf("Stage 2 complete, estimated total time = ");
  print_time_from_seconds((int)ttime);
  printf("\n");
	free(bprimes);
  free(rp_gaps);
  mpz_clear (control);
  return 0;
}

int
check_pm1 (int q, char *expectedResidue)
{
  int n, j, last, error_flag;
  double  *x = NULL;
  unsigned *x_packed = NULL;
  float maxerr, terr;
  int restarting = 0;
  timeval time0, time1;
  int j_save = 0, offset_save = 0;
  int total_time = 0, start_time;
  int offset;
  int reset_fft = 0;
  int j_resume = 0;
  int interact_result = 0;
  int bit;
  unsigned *control = NULL;
  int stage = 1;
  
  signal (SIGTERM, SetQuitting);
  signal (SIGINT, SetQuitting);
  control = get_control(&last, b1, 0, q);
  do
  {				/* while (restarting) */
    maxerr = 0.0;

    if(stage == 1)
    {
      if(!x_packed) x_packed = read_checkpoint_packed(q);
      x = init_lucas_packed (x_packed, q, &n, &j, &offset, &total_time); 
    }
    if(!x) exit (2);
   /* double temp;
    temp = exp2((q%n) / (double) n);
    printf("%0.55f\n",ttmp[4]);
    printf("%0.55f\n",ttmp[5]);
    printf("%0.55f\n",ttmp[4] * temp);
    printf("%0.55f\n",ttmp[6]);
    temp = exp2((q%n - n) / (double) n);
    printf("%0.55f\n",ttmp[5] * temp);
    if(ttmp) exit (1);*/
    gettimeofday (&time0, NULL);
    start_time = time0.tv_sec;
 
    restarting = 0;
    if(j == 1)
    {
      if(stage == 1) printf ("Starting stage 1 P-1, M%d, B1 = %d, B2 = %d, e = %d, fft length = %dK\n", q, b1, g_b2, g_e, n/1024);
      printf ("Doing %d iterations\n", last);
      //if(stage == 1) restarting = round_off_test(x, q, n, &j, &offset, control, last);
    }
    else
    {
      printf ("Continuing work from a partial result of M%d fft length = %dK iteration = %d\n", q, n/1024, j);
      j_resume = j % checkpoint_iter - 1;
    }
    fflush (stdout);

   for (; !restarting && j <= last; j++) // Main LL loop
    {
	    if ((j % 100) == 0) error_flag = 1;
	    else error_flag = 0;
      bit = get_bit(last - j, control);
      //printf("bit: %d, j: %d\n",bit,j);
      terr = lucas_square (x, q, n, j, last, &maxerr, error_flag, bit, stage);//error_flag);
	    if ((j % checkpoint_iter) == 0 || quitting)
      {
	      cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
        standardize_digits(x, q, n, 0, n);
	      gettimeofday (&time1, NULL);
	      //if(!expectedResidue)
        //{
 	          total_time += (time1.tv_sec - start_time);
	          start_time = time1.tv_sec;
        //}
        set_checkpoint_data(x_packed, q, n, j + 1, offset, total_time);
	      if(quitting)
	      {
          cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
	        if(terr < 0.35)
	        {
	          commit_checkpoint_packed (x, x_packed, q, n); 
	          write_checkpoint_packed (x_packed, q);
	        } 
	        printf(" Estimated time spent so far: ");
	        print_time_from_seconds(total_time);
	        printf("\n\n");
		      j = last + 1;
	      }
	      else
        {
          printf ("Iteration %d ", j);
          printbits (x, q, n, offset, 0, NULL, 0);
	        long long diff = time1.tv_sec - time0.tv_sec;
	        long long diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
	        long long diff2 = (last - j) * diff1 / ((checkpoint_iter - j_resume) *  1e6);
	        gettimeofday (&time0, NULL);
	        printf (" err = %5.5f (", maxerr);
	        print_time_from_seconds ((int) diff);
	        printf (" real, %4.4f ms/iter, ETA ", diff1 / 1000.0 / (checkpoint_iter - j_resume));
	        print_time_from_seconds ((int) diff2);
	        printf (")\n");
	        fflush (stdout);
	        if(j_resume) j_resume = 0;
	        reset_err(&maxerr, 0.85); // Instead of tracking maxerr over whole run, reset it at each checkpoint.
	      }
	    }
      if (error_flag)
	    { 
	      if (terr >= 0.43)
		    {
		      if (t_f)
	        {
		        gettimeofday(&time1, NULL);
		        printf ("Iteration = %d >= 1000 && err = %5.5g >= 0.35, fft length = %dK, writing checkpoint file (because -t is enabled) and exiting.\n\n", j, (double) terr, (int) n/1024);
		        cutilSafeCall (cudaMemcpy (x, g_save, sizeof (double) * n, cudaMemcpyDeviceToHost));
		        total_time += (time1.tv_sec - start_time);
	          set_checkpoint_data(x_packed, q, n, j_save + 1, offset_save, total_time);
	          commit_checkpoint_packed (x, x_packed, q, n); 
	          write_checkpoint_packed (x_packed, q); 
		      }
		      printf ("Iteration = %d, err = %5.5g >= 0.43, quitting.\n", j, terr);
		               //fft length = %dK, restarting from last checkpoint with longer fft.\n\n", j, (double) terr, (int) n/1024);
          quitting = 1;
          //fftlen++;
          //restarting = 1;
          //reset_fft = 1;
          //reset_err(&maxerr, 0.0);
	      }
	      else		// error_flag && terr < 0.35
	      {
	        int end = (q + 31) / 32;
          if(x_packed[end + 3] != x_packed[end + 8])
          {
            commit_checkpoint_packed (x, x_packed, q, n); 
            write_checkpoint_packed (x_packed, q); 
            if(reset_fft)
            {  
              printf("Sticking point passed. Switching back to shorter fft.\n");
              fftlen--;
              restarting = 1;
              reset_fft = 0;
              reset_err(&maxerr, 0.0);
            }
          }
		      if (t_f)
		      {
		        copy_kernel <<< n / 128, 128 >>> (g_save, g_x);
		        j_save = j;
		        offset_save = offset;
		      }
		    }
	    }  
      if ( k_f && !quitting && (!(j & 15)) && _kbhit()) interact_result = interact(); // abstracted to clean up check()
      if(interact_result == 2) reset_fft = 0;
      if(interact_result == 1)
      {  
        if((j % checkpoint_iter) != 0)
        {
           cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
           if(terr < 0.35)
           {
		         cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
             standardize_digits(x, q, n, 0, n);
	           set_checkpoint_data(x_packed, q, n, j + 1, offset, total_time);
	           commit_checkpoint_packed (x, x_packed, q, n); 
             reset_err(&maxerr, 0.0);
           }
        }
        restarting = 1;
      } 
	    interact_result = 0;
	    fflush (stdout);	    
	  }
	  
    if (!restarting && !quitting)
	  { // done with stage 1
	    gettimeofday (&time1, NULL);
	    //FILE* fp = fopen_and_lock(RESULTSFILE, "a");
	    //if(!fp) 
	    //{
	    //  fprintf (stderr, "Cannot write results to %s\n\n", RESULTSFILE);
	    //  exit (1);
	    //}
	    cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
      if (standardize_digits(x, q, n, 0, n))
      {
        // printf ("M( %d )P, n = %dK, %s", q, n / 1024, program);
        // if (fp) fprintf (fp, "M( %d )P, n = %dK, %s", q, n / 1024, program);
      }
	    else printbits (x, q, n, offset, NULL , 0, 1); 
      total_time += (time1.tv_sec - start_time);
      if (stage == 1) printf ("\nStage 1 complete, estimated total time = ");
      else printf ("\nStage 2 complete, estimated total time = ");
      print_time_from_seconds(total_time);
	    
	//    if( AID[0] && strncasecmp(AID, "N/A", 3) )  
      //{ // If (AID is not empty), AND (AID is NOT "N/A") (case insensitive)
        //fprintf(fp, ", AID: %s\n", AID);
	 //   } 
      //else fprintf(fp, "\n");
	//    unlock_and_fclose(fp);
	    fflush (stdout);
      free ((char *) control);
      printf("\nStarting stage 1 gcd.\n");
      if(!get_gcd(x, x_packed, q, n, 1))
      {
          printf("Starting stage 2.\n");
          stage2(x, x_packed, q, n);
	      /*for(j = 0; j < 2; j++)
	      {
	        cutilSafeCall (cudaMemcpy (x, g_bg[j], sizeof (double) * n, cudaMemcpyDeviceToHost));
          if (!standardize_digits(x, q, n, 0, n))
          { 
            set_checkpoint_data(x_packed, q, n, j + 1, offset, total_time);
            commit_checkpoint_packed (x, x_packed, q, n); 
            printbits (x, q, n, 0, NULL, 0, 0); 
	          printf("\n");
	        }
	      }
	        cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
          if (!standardize_digits(x, q, n, 0, n))
          { 
            set_checkpoint_data(x_packed, q, n, j + 1, offset, total_time);
            commit_checkpoint_packed (x, x_packed, q, n); 
            printf("Base: ");
            printbits (x, q, n, 0, NULL, 0, 0); 
	          printf("\n");
	        }*/
	        cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
          if (!standardize_digits(x, q, n, 0, n))
          { 
            set_checkpoint_data(x_packed, q, n, j + 1, offset, total_time);
            commit_checkpoint_packed (x, x_packed, q, n); 
            printf("Accumulated Product: ");
            printbits (x, q, n, 0, NULL, 0, 0); 
	          printf("\n");
	   }
	   printf("Starting stage 2 gcd.\n");
	   get_gcd(x, x_packed, q, n, 2);
      }
      stage++;
	    rm_checkpoint (q);
	    printf("\n");
	  }
    close_lucas (x);
  }
  while (restarting);
  free ((char *) x_packed);
  //if(control) free ((char *) control);
  return (0);
}

int
check (int q, char *expectedResidue)
{
  int n, j, last = q - 2, error_flag;
  double  *x = NULL;
  unsigned *x_packed = NULL;
  float maxerr, terr;
  int restarting = 0;
  timeval time0, time1;
  int j_save = 0, offset_save = 0;
  int total_time = 0, start_time;
   // use start_total because every time we write a checkpoint, total_time is increased, 
   // but we don't quit everytime we write a checkpoint
  int offset;
  int reset_fft = 0;
  int j_resume = 0;
  int interact_result = 0;
  
  signal (SIGTERM, SetQuitting);
  signal (SIGINT, SetQuitting);
  do
  {				/* while (restarting) */
    maxerr = 0.0;

    if(!x_packed) x_packed = read_checkpoint_packed(q);
    x = init_lucas_packed (x_packed, q, &n, &j, &offset, &total_time); 
    if(!x) exit (2);

    gettimeofday (&time0, NULL);
    start_time = time0.tv_sec;
    //last = q - 2;		/* the last iteration done in the primary loop */
 
    restarting = 0;
    if(j == 1)
    {
      if(!restarting) printf ("Starting M%d fft length = %dK\n", q, n/1024);
      //restarting = round_off_test(x, q, n, &j, &offset);
    }
    else if (!restarting) 
    {
      printf ("Continuing work from a partial result of M%d fft length = %dK iteration = %d\n", q, n/1024, j);
      j_resume = j % checkpoint_iter - 1;
    }
    fflush (stdout);

    //if( j == 1 ) restarting = round_off_test(x, q, n, &j, &offset);
 last = 2;
    for (; !restarting && j <= last; j++) // Main LL loop
    {
	    if ((j % 100) == 0) error_flag = 1;
	    else error_flag = 0;
      terr = lucas_square (x, q, n, j, last, &maxerr, 1, 0,0);//error_flag);
	    if ((j % checkpoint_iter) == 0 || quitting)
      {
	      cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
        standardize_digits(x, q, n, 0, n);
	      gettimeofday (&time1, NULL);
	      //if(!expectedResidue)
        //{
 	          total_time += (time1.tv_sec - start_time);
	          start_time = time1.tv_sec;
        //}
        set_checkpoint_data(x_packed, q, n, j + 1, offset, total_time);
	      if(quitting)
	      {
          cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
	        if(terr < 0.35)
	        {
	          commit_checkpoint_packed (x, x_packed, q, n); 
	          write_checkpoint_packed (x_packed, q);
	        } 
	        printf(" Estimated time spent so far: ");
	        print_time_from_seconds(total_time);
	        printf("\n\n");
		      j = last + 1;
	      }
	      else
        {
          printf ("Iteration %d ", j);
          printbits (x, q, n, offset, 0, expectedResidue, 0);
	        long long diff = time1.tv_sec - time0.tv_sec;
	        long long diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
	        long long diff2 = (last - j) * diff1 / ((checkpoint_iter - j_resume) *  1e6);
	        gettimeofday (&time0, NULL);
	        printf (" err = %5.5f (", maxerr);
	        print_time_from_seconds ((int) diff);
	        printf (" real, %4.4f ms/iter, ETA ", diff1 / 1000.0 / (checkpoint_iter - j_resume));
	        print_time_from_seconds ((int) diff2);
	        printf (")\n");
	        fflush (stdout);
	        if(j_resume) j_resume = 0;
	        reset_err(&maxerr, 0.85); // Instead of tracking maxerr over whole run, reset it at each checkpoint.
	      }
	    }
      if (error_flag)
	    { 
	      if (terr >= 0.35)
		    {
		      if (t_f)
	        {
		        gettimeofday(&time1, NULL);
		        printf ("Iteration = %d >= 1000 && err = %5.5g >= 0.35, fft length = %dK, writing checkpoint file (because -t is enabled) and exiting.\n\n", j, (double) terr, (int) n/1024);
		        cutilSafeCall (cudaMemcpy (x, g_save, sizeof (double) * n, cudaMemcpyDeviceToHost));
		        total_time += (time1.tv_sec - start_time);
	          set_checkpoint_data(x_packed, q, n, j_save + 1, offset_save, total_time);
	          commit_checkpoint_packed (x, x_packed, q, n); 
	          write_checkpoint_packed (x_packed, q); 
		      }
		      printf ("Iteration = %d >= 1000 && err = %5.5g >= 0.35, fft length = %dK, restarting from last checkpoint with longer fft.\n\n", j, (double) terr, (int) n/1024);
          fftlen++;
          restarting = 1;
          reset_fft = 1;
          reset_err(&maxerr, 0.0);
	      }
	      else		// error_flag && terr < 0.35
	      {
	        int end = (q + 31) / 32;
          if(x_packed[end + 3] != x_packed[end + 8])
          {
            commit_checkpoint_packed (x, x_packed, q, n); 
            write_checkpoint_packed (x_packed, q); 
            if(reset_fft)
            {  
              printf("Sticking point passed. Switching back to shorter fft.\n");
              fftlen--;
              restarting = 1;
              reset_fft = 0;
              reset_err(&maxerr, 0.0);
            }
          }
		      if (t_f)
		      {
		        copy_kernel <<< n / 128, 128 >>> (g_save, g_x);
		        j_save = j;
		        offset_save = offset;
		      }
		    }
	    }  
      if ( k_f && !quitting && (!(j & 15)) && _kbhit()) interact_result = interact(); // abstracted to clean up check()
      if(interact_result == 2) reset_fft = 0;
      if(interact_result == 1)
      {  
        if((j % checkpoint_iter) != 0)
        {
           cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
           if(terr < 0.35)
           {
		         cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
             standardize_digits(x, q, n, 0, n);
	           set_checkpoint_data(x_packed, q, n, j + 1, offset, total_time);
	           commit_checkpoint_packed (x, x_packed, q, n); 
             reset_err(&maxerr, 0.0);
           }
        }
        restarting = 1;
      } 
	    interact_result = 0;
	    fflush (stdout);	    
	  } /* end main LL for-loop */
	
    if (!restarting && !quitting)
	  { // done with test
	    gettimeofday (&time1, NULL);
	    FILE* fp = fopen_and_lock(RESULTSFILE, "a");
	    if(!fp) 
	    {
	      fprintf (stderr, "Cannot write results to %s\n\n", RESULTSFILE);
	      exit (1);
	    }
	    cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
      if (standardize_digits(x, q, n, 0, n))
      {
        printf ("M( %d )P, n = %dK, %s", q, n / 1024, program);
        if (fp) fprintf (fp, "M( %d )P, n = %dK, %s", q, n / 1024, program);
      }
	    else printbits (x, q, n, offset, fp, 0, 1); 
      total_time += (time1.tv_sec - start_time);
      printf (", estimated total time = ");
      print_time_from_seconds(total_time);
	  
	    if( AID[0] && strncasecmp(AID, "N/A", 3) )  
      { // If (AID is not empty), AND (AID is NOT "N/A") (case insensitive)
        fprintf(fp, ", AID: %s\n", AID);
	    } 
      else fprintf(fp, "\n");
	    unlock_and_fclose(fp);
	    fflush (stdout);
	    rm_checkpoint (q);
	    printf("\n\n");
	          set_checkpoint_data(x_packed, q, n, j + 1, offset, total_time);
	          commit_checkpoint_packed (x, x_packed, q, n); 
	          write_checkpoint_packed (x_packed, q); 
	  }
    close_lucas (x);
  }
  while (restarting);
  free ((char *) x_packed);
  return (0);
}

int
check_residue (int q, char *expectedResidue)
{
  int n, j, last, offset;
  unsigned *x_packed = NULL;
  double  *x = NULL;
  float maxerr = 0.0;
  int restarting = 0;
  timeval time0, time1;
 
  do
  {				
    if(!x_packed) x_packed = read_checkpoint_packed(q);
    x = init_lucas_packed (x_packed, q, &n, &j, &offset, NULL);
    if(!x) exit (2);
    gettimeofday (&time0, NULL);
    if(!restarting) printf ("Starting self test M%d fft length = %dK\n", q, n/1024);
    //restarting = round_off_test(x, q, n, &j, &offset);
    if(restarting) close_lucas (x);
  }
  while (restarting);

  fflush (stdout);
  last = 10000;

  for (; j <= last; j++)
  {
    lucas_square (x, q, n, j, last, &maxerr, 0, 0, 0);
    if(j % 100 == 0) cutilSafeCall (cudaMemcpy (&maxerr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
  }
  cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
  standardize_digits(x, q, n, 0, n);
  gettimeofday (&time1, NULL);

  printf ("Iteration %d ", j - 1);
  printbits (x, q, n, offset, 0, expectedResidue, 0);
  long long diff = time1.tv_sec - time0.tv_sec;
  long long diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
  printf (" err = %5.5f (", maxerr);
  print_time_from_seconds ((int) diff);
  printf (" real, %4.4f ms/iter", diff1 / (1000.0 * last));
  printf (")\n");

  fftlen = 0;
  close_lucas (x);
  free ((char *) x_packed);
  if (strcmp (s_residue, expectedResidue))
  {
    printf("Expected residue [%s] does not match actual residue [%s]\n\n", expectedResidue, s_residue);
    fflush (stdout);	
    return 1;    
  }
  else
  {
    printf("This residue is correct.\n\n");
    fflush (stdout);	
    return (0);
  }
}

void parse_args(int argc, char *argv[], int* q, int* device_numer, 
		int* cufftbench_s, int* cufftbench_e, int* cufftbench_d);
		/* The rest of the opts are global */
int main (int argc, char *argv[])
{ 
  printf("\n");
  quitting = 0;
#define THREADS_DFLT 256
#define CHECKPOINT_ITER_DFLT 10000
#define SAVE_FOLDER_DFLT "savefiles"
#define S_F_DFLT 0
#define T_F_DFLT 0
#define K_F_DFLT 0
#define D_F_DFLT 0
#define POLITE_DFLT 1
#define WORKFILE_DFLT "worktodo.txt"
#define RESULTSFILE_DFLT "results.txt"
  
  /* "Production" opts to be read in from command line or ini file */
  int q = -1;
  int device_number = -1, f_f = 0;
  checkpoint_iter = -1;
  threads = -1;
  fftlen = -1;
  s_f = t_f = d_f = k_f = -1;
  polite_f = polite = -1;
  AID[0] = input_filename[0] = RESULTSFILE[0] = 0; /* First character is null terminator */
  char fft_str[132] = "\0";
  
  /* Non-"production" opts */
  r_f = 0;
  int cufftbench_s, cufftbench_e, cufftbench_d;  
  cufftbench_s = cufftbench_e = cufftbench_d = 0;

  parse_args(argc, argv, &q, &device_number, &cufftbench_s, &cufftbench_e, &cufftbench_d);
  /* The rest of the args are globals */
  
  if (file_exists(INIFILE))
  {  
   if( checkpoint_iter < 1 && 		!IniGetInt(INIFILE, "CheckpointIterations", &checkpoint_iter, CHECKPOINT_ITER_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option CheckpointIterations; using default: %d\n", CHECKPOINT_ITER_DFLT)*/;
   if( threads < 1 && 			!IniGetInt(INIFILE, "Threads", &threads, THREADS_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option Threads; using default: %d\n", THREADS_DFLT);
   if( s_f < 0 && 			!IniGetInt(INIFILE, "SaveAllCheckpoints", &s_f, S_F_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveAllCheckpoints; using default: off\n")*/;
   if( 		     	     s_f > 0 && !IniGetStr(INIFILE, "SaveFolder", folder, SAVE_FOLDER_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveFolder; using default: \"%s\"\n", SAVE_FOLDER_DFLT)*/;
   if( t_f < 0 && 			!IniGetInt(INIFILE, "CheckRoundoffAllIterations", &t_f, 0) )
    fprintf(stderr, "Warning: Couldn't parse ini file option CheckRoundoffAllIterations; using default: off\n");
   if( polite < 0 && 			!IniGetInt(INIFILE, "Polite", &polite, POLITE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option Polite; using default: %d\n", POLITE_DFLT);
   if( k_f < 0 && 			!IniGetInt(INIFILE, "Interactive", &k_f, 0) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option Interactive; using default: off\n")*/;
   if( device_number < 0 &&		!IniGetInt(INIFILE, "DeviceNumber", &device_number, 0) )
    fprintf(stderr, "Warning: Couldn't parse ini file option DeviceNumber; using default: 0\n");
   if( d_f < 0 &&			!IniGetInt(INIFILE, "PrintDeviceInfo", &d_f, D_F_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option PrintDeviceInfo; using default: off\n")*/;
   if( !input_filename[0] &&		!IniGetStr(INIFILE, "WorkFile", input_filename, WORKFILE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option WorkFile; using default \"%s\"\n", WORKFILE_DFLT);
    /* I've readded the warnings about worktodo and results due to the multiple-instances-in-one-dir feature. */
   if( !RESULTSFILE[0] && 		!IniGetStr(INIFILE, "ResultsFile", RESULTSFILE, RESULTSFILE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option ResultsFile; using default \"%s\"\n", RESULTSFILE_DFLT);
   if( fftlen < 0 && 			!IniGetStr(INIFILE, "FFTLength", fft_str, "\0") )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option FFTLength; using autoselect.\n")*/;
  }
  else // no ini file
    {
      fprintf(stderr, "Warning: Couldn't find .ini file. Using defaults for non-specified options.\n");
      if( checkpoint_iter < 1 ) checkpoint_iter = CHECKPOINT_ITER_DFLT;
      if( threads < 1 ) threads = THREADS_DFLT;
      if( fftlen < 0 ) fftlen = 0;
      if( s_f < 0 ) s_f = S_F_DFLT;
      if( t_f < 0 ) t_f = T_F_DFLT;
      if( k_f < 0 ) k_f = K_F_DFLT;
      if( device_number < 0 ) device_number = 0;
      if( d_f < 0 ) d_f = D_F_DFLT;
      if( polite < 0 ) polite = POLITE_DFLT;
      if( !input_filename[0] ) sprintf(input_filename, WORKFILE_DFLT);
      if( !RESULTSFILE[0] ) sprintf(RESULTSFILE, RESULTSFILE_DFLT);
  }
  
  if( fftlen < 0 ) { // possible if -f not on command line
      fftlen = fft_from_str(fft_str);
  }
  if (polite == 0) {
    polite_f = 0;
    polite = 1;
  } else {
    polite_f = 1;
  }
  if (threads != 32 && threads != 64 && threads != 128
	      && threads != 256 && threads != 512 && threads != 1024)
  {
    fprintf(stderr, "Error: thread count is invalid.\n");
    fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
    exit(2);
  }
  f_f = fftlen; // if the user has given an override... then note this length must be kept between tests
    
  
  init_device (device_number);

  if (r_f)
    {
      int bad_selftest = 0;
      fftlen = 0;
      bad_selftest += check_residue (86243, "23992ccd735a03d9");
      bad_selftest += check_residue (132049, "4c52a92b54635f9e");
      bad_selftest += check_residue (216091, "30247786758b8792");
      bad_selftest += check_residue (756839, "5d2cbe7cb24a109a");
      bad_selftest += check_residue (859433, "3c4ad525c2d0aed0");
      bad_selftest += check_residue (1257787, "3f45bf9bea7213ea");
      bad_selftest += check_residue (1398269, "a4a6d2f0e34629db");
      bad_selftest += check_residue (2976221, "2a7111b7f70fea2f");
      bad_selftest += check_residue (3021377, "6387a70a85d46baf");
      bad_selftest += check_residue (6972593, "88f1d2640adb89e1");
      bad_selftest += check_residue (13466917, "9fdc1f4092b15d69");
      bad_selftest += check_residue (20996011, "5fc58920a821da11");
      bad_selftest += check_residue (24036583, "cbdef38a0bdc4f00");
      bad_selftest += check_residue (25964951, "62eb3ff0a5f6237c");
      bad_selftest += check_residue (30402457, "0b8600ef47e69d27");
      bad_selftest += check_residue (32582657, "02751b7fcec76bb1");
      bad_selftest += check_residue (37156667, "67ad7646a1fad514");
      bad_selftest += check_residue (42643801, "8f90d78d5007bba7");
      bad_selftest += check_residue (43112609, "e86891ebf6cd70c4");
      if (bad_selftest)
      {
        fprintf(stderr, "Error: There ");
        bad_selftest > 1 ? fprintf(stderr, "were %d bad selftests!\n",bad_selftest) 
        		 : fprintf(stderr, "was a bad selftest!\n");
      }
    }
  else if (cufftbench_d)
    cufftbench (cufftbench_s, cufftbench_e, cufftbench_d);
  else
    {
      if (s_f)
	{
#ifndef _MSC_VER
	  mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
	  if (mkdir (folder, mode) != 0)
	    fprintf (stderr,
		     "mkdir: cannot create directory `%s': File exists\n",
		     folder);
#else
	  if (_mkdir (folder) != 0)
	    fprintf (stderr,
		     "mkdir: cannot create directory `%s': File exists\n",
		     folder);
#endif
	}
      if (q <= 0)
        {
          int error;
	
	  #ifdef EBUG
	  printf("Processed INI file and console arguments correctly; about to call get_next_assignment().\n");
	  #endif
	  do 
            { // while(!quitting)
	
	                 
	      fftlen = f_f; // fftlen and AID change between tests, so be sure to reset them
	      AID[0] = 0;
	                   
  	      error = get_next_assignment(input_filename, &q, &fftlen, &tfdepth, &llsaved, &AID);
               /* Guaranteed to write to fftlen ONLY if specified on workfile line, so that if unspecified, the pre-set default is kept. */
	      if( error > 0) exit (2); // get_next_assignment prints warning message	  
	      #ifdef EBUG
	      printf("Gotten assignment, about to call check().\n");
	      #endif
              check_pm1 (q, 0);
	  
	      if(!quitting) // Only clear assignment if not killed by user, i.e. test finished 
	        {
	          error = clear_assignment(input_filename, q);
	          if(error) exit (2); // prints its own warnings
	        }
	    
	    } 
          while(!quitting);  
      }
    else // Exponent passed in as argument
      {
	if (!valid_assignment(q, fftlen)) {printf("\n");} //! v_a prints warning
	else check_pm1 (q, 0);
      }
  } // end if(-r) else if(-cufft) else(workfile)
} // end main()

void parse_args(int argc, char *argv[], int* q, int* device_number, 
		int* cufftbench_s, int* cufftbench_e, int* cufftbench_d)
{
int ptest;
while (argc > 1)
    {
      if (strcmp (argv[1], "-t") == 0)
	{
	  t_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-h") == 0)
        {
      	  fprintf (stderr,
	       "$ CUDAPm1 -h|-v\n\n");
      	  fprintf (stderr,
	       "$ CUDAPm1 [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-c checkpoint_iteration] [-f fft_length] [-s folder] [-t] [-polite iteration] [-k] exponent|input_filename\n\n");
      	  fprintf (stderr,
	       "$ CUDAPm1 [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-polite iteration] -r\n\n");
      	  fprintf (stderr,
	       "$ CUDAPm1 [-d device_number] [-info] -cufftbench start end distance\n\n");
	  fprintf (stderr,
	       "                       -h          print this help message\n");
	  fprintf (stderr,
	       "                       -v          print version number\n");
	  fprintf (stderr,
	       "                       -info       print device information\n");
	  fprintf (stderr,
	       "                       -i          set .ini file name (default = \"CUDAPm1.ini\")\n");
      	  fprintf (stderr,
	       "                       -threads    set threads number (default = 256)\n");
      	  fprintf (stderr,
	       "                       -f          set fft length (if round off error then exit)\n");
      	  fprintf (stderr,
	       "                       -s          save all checkpoint files\n");
      	  fprintf (stderr,
	       "                       -t          check round off error all iterations\n");
      	  fprintf (stderr,
	       "                       -polite     GPU is polite every n iterations (default -polite 1) (-polite 0 = GPU aggressive)\n");
      	  fprintf (stderr,
	       "                       -cufftbench exec CUFFT benchmark (Ex. $ ./CUDAPm1 -d 1 -cufftbench 1179648 6291456 32768 )\n");
      	  fprintf (stderr, 
      	       "                       -r          exec residue test.\n");
      	  fprintf (stderr,
	       "                       -k          enable keys (p change -polite, t disable -t, s change -s)\n\n");
      	  fprintf (stderr,
	       "                       -b2         set b2\n\n");
      	  fprintf (stderr,
	       "                       -d2         Brent-Suyama coefficient (multiple of 30, 210, or 2310) \n\n");
      	  fprintf (stderr,
	       "                       -e2         Brent-Suyama exponent (2-12) \n\n");
      	  fprintf (stderr,
	       "                       -nrp2       Relative primes per pass (divisor of 8, 48, or 480)\n\n");
      	  exit (2);          
      	}
      else if (strcmp (argv[1], "-v") == 0)
        {  
          printf("%s\n\n", program);
          exit (2);
        }
      else if (strcmp (argv[1], "-polite") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -polite option\n\n");
	      exit (2);
	    }
	  polite = atoi (argv[2]);
	  if (polite == 0)
	    {
	      polite_f = 0;
	      polite = 1;
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-r") == 0)
	{
	  r_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-k") == 0)
	{
	  k_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-d") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -d option\n\n");
	      exit (2);
	    }
	  *device_number = atoi (argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-i") == 0)
	{
	  if(argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -i option\n\n");
	      exit (2);
	    }
	  sprintf (INIFILE, "%s", argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-info") == 0)
        {
          d_f = 1;
          argv++;
          argc--;
        }
      else if (strcmp (argv[1], "-cufftbench") == 0)
	{
	  if (argc < 5 || argv[2][0] == '-' || argv[3][0] == '-' || argv[4][0] == '-')
	    {
	      fprintf (stderr, "can't parse -cufftbench option\n\n");
	      exit (2);
	    }
	  *cufftbench_s = atoi (argv[2]);
	  *cufftbench_e = atoi (argv[3]);
	  *cufftbench_d = atoi (argv[4]);
	  argv += 4;
	  argc -= 4;
	}
      else if (strcmp (argv[1], "-threads") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -threads option\n\n");
	      exit (2);
	    }
	  threads = atoi (argv[2]);
	  if (threads != 32 && threads != 64 && threads != 128
	      && threads != 256 && threads != 512 && threads != 1024)
	    {
	      fprintf(stderr, "Error: thread count is invalid.\n");
	      fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-c") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	  checkpoint_iter = atoi (argv[2]);
	  if (checkpoint_iter == 0)
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-f") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -f option\n\n");
	      exit (2);
	    }
	  fftlen = fft_from_str(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-b1") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -b1 option\n\n");
	      exit (2);
	    }
	  b1 = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-e2") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -e2 option\n\n");
	      exit (2);
	    }
	  g_e = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-d2") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -d2 option\n\n");
	      exit (2);
	    }
	  g_d = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-b2") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -b2 option\n\n");
	      exit (2);
	    }
	  g_b2 = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-nrp2") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -nrp option\n\n");
	      exit (2);
	    }
	  g_nrp = atoi(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-s") == 0)
	{
	  s_f = 1;
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -s option\n\n");
	      exit (2);
	    }
	  sprintf (folder, "%s", argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else
	{
	  if (*q != -1 || strcmp (input_filename, "") != 0 )
	    {
	      fprintf (stderr, "can't parse options\n\n");
	      exit (2);
	    }
	  int derp = atoi (argv[1]);
	  if (derp == 0) {
	    sprintf (input_filename, "%s", argv[1]);
	  } else { *q = derp; }
	  argv++;
	  argc--;
	}
    }
    if ((g_d%30 != 0) && (g_d%210 != 0) && (g_d%2310 != 0)) {
	printf("-d2 must be a multiple of 30, 210, or 2310.\n");
	exit(3);
    }
    if ((g_e%2 != 0) || (g_e < 2) || (g_e > 12)) {
	printf("-e2 must be 2, 4, 6, 8, 10, or 12.\n");
	exit(3);
    }
    // move this to later assignment validation?
    // also add check for e2 not too large...
    if (g_d%7 != 0) ptest=7;
    else if (g_d%11 !=0) ptest=11;
    else if (g_d%13 !=0) ptest=13;
    else if (g_d%17 !=0) ptest=17;
    else if (g_d%19 !=0) ptest=19;
    else if (g_d%23 !=0) ptest=23;
    else if (g_d%27 !=0) ptest=27;
    else if (g_d%29 !=0) ptest=29;
    else ptest=0;
    // printf("p=%d\n",ptest);
    if (ptest > 0) {
	if (b1 * ptest * 53 < g_b2) {
		printf("b1 should be at least %d\n", g_b2/(ptest * 53));
		exit(3);
    	}
    	if (g_b2 < ptest * (2*g_e+1)) {
		printf("b2 should be at least %d\n", ptest * (2*g_e+1));
		exit(3);
    	}
    	if (g_b2 < ptest * b1) {
		printf("b2 should be at least %d\n", ptest * b1);
		exit(3);
    	}
    }
}

int interact(void)
{
  int c = getchar ();
  if (c == 'p')
    if (polite_f)
	  {
	    polite_f = 0;
	    printf ("   -polite 0\n");
	  }
    else
     {
      polite_f = 1;
      printf ("   -polite %d\n", polite);
     }
  else if (c == 't')
    {
      t_f = 0;
      printf ("   disabling -t\n");
    }
  else if (c == 's')
    if (s_f == 1)
      {
        s_f = 2;
        printf ("   disabling -s\n");
      }
    else if (s_f == 2)
      {
        s_f = 1;
        printf ("   enabling -s\n");
      }
   if (c == 'F')
      {
         printf(" -- Increasing fft length.\n");
         fftlen++;
          return 1;
      }
   if (c == 'f')
      {
         printf(" -- Decreasing fft length.\n");
         fftlen--;
         return 1;
      }
   if (c == 'k')
      {
         printf(" -- fft length reset cancelled.\n");
         return 2;
      }
   fflush (stdin);
   return 0;
}
