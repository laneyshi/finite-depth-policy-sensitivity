#include <cmath>
#include <cstdint>
#include <algorithm>
// Jet layout: value, theta gradient (3), kappa gradient, theta Hessian (9), mixed (3).
// All transition factors are differentiated analytically in double precision.
extern "C" void defender_step(int n, int fields, const int32_t* next,
 const double* reward, const double* alive, const double* prob, const double* dp,
 const uint8_t* immediate, const double* immediate_reward,
 const double* v, double gamma, double* out) {
 #pragma omp parallel for schedule(static) num_threads(8)
 for(int s=0;s<n;s++) {
  double* z=out+(long)s*fields;
  std::fill(z,z+fields,0.0);
  if(immediate[s]) {z[0]=immediate_reward[s]; continue;}
  for(int a=0;a<9;a++) {
   long e=(long)s*9+a; const double* u=v+(long)next[e]*fields;
   double g=gamma*alive[e], p=prob[e], k=dp[e], q=reward[e]+g*u[0];
   z[0]+=p*q;
   if(fields==1) continue;
   for(int i=1;i<fields;i++) z[i]+=p*g*u[i];
   if(fields==17) {
    z[4]+=k*q;
    for(int i=0;i<3;i++) z[14+i]+=k*g*u[1+i];
   }
  }
 }
}

extern "C" void attacker_step(int n,int fields,const int32_t* lower,
 const double* blend,const double* laws,const double* pi,const double* feature,
 const uint8_t* terminal,const double* value,double* out) {
 #pragma omp parallel for schedule(static) num_threads(8)
 for(int s=0;s<n;s++) {
  double* z=out+(long)s*fields; std::fill(z,z+fields,0.0);
  if(terminal[s]) continue;
  const double* p=pi+(long)s*4; const double* f=feature+(long)s*3;
  for(int m=0;m<16;m++) {
   long e=(long)s*16+m; double b=blend[e];
   const double* lo=value+(long)lower[e]*fields;
   const double* hi=lo+fields;
   double u[17];
   for(int j=0;j<fields;j++) u[j]=(1-b)*lo[j]+b*hi[j];
   double w=0; for(int a=0;a<4;a++) w+=p[a]*laws[a*16+m];
   z[0]+=w*u[0]; if(fields==1) continue;
   double dw[3];
   for(int i=0;i<3;i++) {
    dw[i]=f[i]*p[i]*(laws[i*16+m]-w);
    z[1+i]+=w*u[1+i]+dw[i]*u[0];
   }
   if(fields==4) continue;
   z[4]+=w*u[4];
   for(int i=0;i<3;i++) {
    z[14+i]+=w*u[14+i]+dw[i]*u[4];
    for(int j=0;j<3;j++) {
     double d2=f[i]*f[j]*((i==j?p[i]*(laws[i*16+m]-w):0.0)
                           -p[i]*p[j]*(laws[i*16+m]+laws[j*16+m]-2*w));
     z[5+3*i+j]+=w*u[5+3*i+j]+dw[i]*u[1+j]+dw[j]*u[1+i]+d2*u[0];
    }
   }
  }
 }
}
