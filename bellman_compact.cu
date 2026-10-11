#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
// F=1 value; F=3 value,g0,g1; F=9 value,g0,g1,gk,H00,H01,H11,M0,M1.
template<int F> __global__ void defend(int n,int bins,const int32_t* next,
 const double* reward,const double* timeout,const uint8_t* alive,
 const double* p0,const double* p1,const uint8_t* terminal,const double* immediate,
 double k,double gamma,bool last,const double* value,double* mid){
 int s=blockIdx.x*blockDim.x+threadIdx.x;if(s>=n)return;
 int phys=s/bins,ib=s%bins;double z[F]={0};
 if(terminal[phys]){z[0]=immediate[phys];}
 else for(int a=0;a<9;a++){
  long e=(long)phys*9+a;double d=p1[(long)s*9+a]-p0[e];
  double p=(1-k)*p0[e]+k*p1[(long)s*9+a];
  const double* u=value+((long)next[e]*bins+ib)*F;
  double g=last?0.:gamma*alive[e];double q=(last?timeout[e]:reward[e])+g*u[0];
  z[0]+=p*q;
  #pragma unroll
  for(int j=1;j<F;j++)z[j]+=p*g*u[j];
  if(F==9){z[3]+=d*q;z[7]+=d*g*u[1];z[8]+=d*g*u[2];}
 }
 #pragma unroll
 for(int j=0;j<F;j++)mid[(long)s*F+j]=z[j];
}
template<int F> __global__ void attack(int n,int bins,int size,const int32_t* next,
 const double* delta,const double* laws,const double* belief,const double* ell,
 const double* pi,const uint8_t* terminal,double limit,const double* mid,double* value){
 int s=blockIdx.x*blockDim.x+threadIdx.x;if(s>=n)return;
 int phys=s/bins,ib=s%bins;double z[F]={0};
 if(!terminal[phys]){
  int key_y=(phys/(size*size*size*size))/size;
  const double* p=pi+((long)key_y*bins+ib)*4;
  double f0=2*belief[ib]-1,f1=2.*key_y/(size-1)-1;
  for(int m=0;m<16;m++){
   long e=(long)phys*16+m;
   double x=fmin(limit,fmax(-limit,ell[ib]+delta[e]));
   int low=0,upper=bins-1;while(upper-low>1){int center=(low+upper)/2;if(ell[center]<=x)low=center;else upper=center;}
   double b=(x-ell[low])/(ell[low+1]-ell[low]);
   const double* lo=mid+((long)next[e]*bins+low)*F;const double* hi=lo+F;double u[F];
   #pragma unroll
   for(int j=0;j<F;j++)u[j]=(1-b)*lo[j]+b*hi[j];
   double w=0.;for(int a=0;a<4;a++)w+=p[a]*laws[a*16+m];z[0]+=w*u[0];
   if(F>1){
    double l0=laws[m],l1=laws[16+m];double d0=f0*p[0]*(l0-w),d1=f1*p[1]*(l1-w);
    z[1]+=w*u[1]+d0*u[0];z[2]+=w*u[2]+d1*u[0];
    if(F==9){
     z[3]+=w*u[3];z[7]+=w*u[7]+d0*u[3];z[8]+=w*u[8]+d1*u[3];
     double h00=f0*f0*(p[0]*(l0-w)-p[0]*p[0]*(l0+l0-2*w));
     double h01=-f0*f1*p[0]*p[1]*(l0+l1-2*w);
     double h11=f1*f1*(p[1]*(l1-w)-p[1]*p[1]*(l1+l1-2*w));
     z[4]+=w*u[4]+d0*u[1]+d0*u[1]+h00*u[0];
     z[5]+=w*u[5]+d0*u[2]+d1*u[1]+h01*u[0];
     z[6]+=w*u[6]+d1*u[2]+d1*u[2]+h11*u[0];
    }
   }
  }
 }
 #pragma unroll
 for(int j=0;j<F;j++)value[(long)s*F+j]=z[j];
}
extern "C" int select_device(int device){return (int)cudaSetDevice(device);}
extern "C" int step(int n,int bins,int size,int fields,
 const int32_t* dn,const double* reward,const double* timeout,const uint8_t* alive,
 const double* p0,const double* p1,const uint8_t* terminal,const double* immediate,
 const int32_t* an,const double* delta,const double* laws,const double* belief,
 const double* ell,const double* pi,double k,double gamma,double limit,int last,
 double* value,double* mid,uintptr_t stream){
 cudaStream_t st=(cudaStream_t)stream;int blocks=(n+127)/128;
 #define CALL(F) defend<F><<<blocks,128,0,st>>>(n,bins,dn,reward,timeout,alive,p0,p1,terminal,immediate,k,gamma,last,value,mid); attack<F><<<blocks,128,0,st>>>(n,bins,size,an,delta,laws,belief,ell,pi,terminal,limit,mid,value)
 if(fields==1){CALL(1);}else if(fields==3){CALL(3);}else if(fields==9){CALL(9);}else return -1;
 cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return (int)e;
 return (int)cudaStreamSynchronize(st);
}
