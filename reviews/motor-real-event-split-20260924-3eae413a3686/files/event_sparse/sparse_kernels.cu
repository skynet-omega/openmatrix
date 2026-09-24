// Generic pure timestamp query and base CSR currents. No anatomical IDs.
__device__ double convolution(double t, double tq, double ts) {
    double z=t*(1./ts-1./tq), den=1.-ts/tq;
    if(fabs(z)<1e-5)
        return t/ts*exp(-t/ts)*(1.+z/2.+z*z/6.+z*z*z/24.);
    return (z>=0. ? exp(-t/tq)*(-expm1(-fmax(z,0.)))
                  : exp(-t/ts)*expm1(fmin(z,0.)))/(den==0. ? 1. : den);
}

extern "C" __global__ void project_ports(
    int p, int query, const double* query_times, const int* rows,
    const double* q, const double* s, const double* tau, double ts,
    const long long* event_ptr, const double* times, const double* jumps,
    const bool* sets, const double* posts,
    double* release, double* q_trace, double* s_trace) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p) return;
    double t=query_times[query], qi=q[i], si=s[i], prev=0.;
    for(long long k=event_ptr[i];k<event_ptr[i+1];++k) {
        double mark=times[k];
        if(mark>t) break; // event_time == query_time is visible
        double h=mark-prev;
        si=si*exp(-h/ts)+qi*convolution(h,tau[i],ts);
        qi=qi*exp(-h/tau[i]);
        qi=sets[k] ? posts[k] : qi+jumps[k];
        prev=mark;
    }
    double h=t-prev;
    double qnow=qi*exp(-h/tau[i]);
    double snow=si*exp(-h/ts)+qi*convolution(h,tau[i],ts);
    release[rows[i]]=snow;
    q_trace[(long long)query*p+i]=qnow;
    s_trace[(long long)query*p+i]=snow;
}

extern "C" __global__ void csr_current(
    int nr, int n, int query, const int* receiver_rows, bool compact,
    const long long* ptr, const int* sources, const double* weights,
    const double* release, const double* caps, const bool* visual,
    double scale, bool connected, double* output) {
    int ordinal=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    int lane=threadIdx.x%32;
    if(ordinal>=nr) return;
    int row=compact ? receiver_rows[ordinal] : ordinal;
    double a=0., b=0.;
    for(long long e=ptr[row]+lane;e<ptr[row+1];e+=32) {
        int source=sources[e];
        if(visual[row]) {
            double value=(weights[e]*scale)*release[source];
            if(value>=0.) a+=value; else b-=value;
        } else if(connected || !visual[source]) {
            a+=weights[e]*(release[source]*caps[source]);
        }
    }
    for(int d=16;d>0;d/=2) {
        a+=__shfl_down_sync(0xffffffff,a,d);
        b+=__shfl_down_sync(0xffffffff,b,d);
    }
    if(lane==0) {
        long long offset=((long long)query*n+row)*2;
        output[offset]=a;
        output[offset+1]=b;
    }
}
