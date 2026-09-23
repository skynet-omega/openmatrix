// Exact difference propagation between two SET event streams in a linear
// exponential release -> first-order synaptic filter. Not a recurrent-brain bound.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
struct Event { double t,post; int stream; };
int main(){try{
 double tq,ts,q0,end;int na,nb;
 if(!(std::cin>>tq>>ts>>q0>>end>>na>>nb)||!(tq>0&&ts>0&&end>0)||na<0||nb<0)
  throw std::runtime_error("invalid input");
 std::vector<Event> e; for(int s=0;s<2;s++)for(int i=0;i<(s?nb:na);i++){
  double t,p;if(!(std::cin>>t>>p)||!std::isfinite(t)||!std::isfinite(p)||t<0||t>end)
   throw std::runtime_error("invalid event"); e.push_back({t,p,s});}
 std::stable_sort(e.begin(),e.end(),[](const Event&a,const Event&b){return a.t<b.t;});
 double qa=q0,qb=q0,ds=0,t=0,maxq=0,maxs=0,l1q=0,l1s=0;
 auto propagate=[&](double h){
  if(h<0)throw std::runtime_error("decreasing time");
  const double dq=qa-qb,initial=ds;
  auto value=[&](double u){
   if(tq==ts)return (initial+dq*u/ts)*std::exp(-u/ts);
   const double B=dq*tq/(tq-ts),A=initial-B;
   return A*std::exp(-u/ts)+B*std::exp(-u/tq);
  };
  auto integral=[&](double u){
   if(tq==ts)return initial*ts*(-std::expm1(-u/ts))+
      dq*(ts*(-std::expm1(-u/ts))-u*std::exp(-u/ts));
   const double B=dq*tq/(tq-ts),A=initial-B;
   return A*ts*(-std::expm1(-u/ts))+B*tq*(-std::expm1(-u/tq));
  };
  maxq=std::max(maxq,std::abs(dq));
  l1q+=std::abs(dq)*tq*(-std::expm1(-h/tq));
  maxs=std::max({maxs,std::abs(initial),std::abs(value(h))});
  double extremum=-1,zero=-1;
  if(tq==ts){
   if(dq!=0){extremum=ts*(1-initial/dq);zero=-initial*ts/dq;}
  }else{
   const double B=dq*tq/(tq-ts),A=initial-B;
   if(A!=0 && -B*ts/(A*tq)>0)extremum=std::log(-B*ts/(A*tq))/(1/tq-1/ts);
   if(A!=0 && -B/A>0)zero=std::log(-B/A)/(1/tq-1/ts);
  }
  if(extremum>0&&extremum<h)maxs=std::max(maxs,std::abs(value(extremum)));
  if(zero>0&&zero<h)l1s+=std::abs(integral(zero))+std::abs(integral(h)-integral(zero));
  else l1s+=std::abs(integral(h));
  ds=value(h);qa*=std::exp(-h/tq);qb*=std::exp(-h/tq);
 };
 size_t i=0;
 while(i<e.size()){
  double next=e[i].t;propagate(next-t);t=next;
  do{(e[i].stream?qb:qa)=e[i].post;i++;}while(i<e.size()&&e[i].t==t);
  maxq=std::max(maxq,std::abs(qa-qb));
 }
 propagate(end-t);
 for(double x:{maxq,maxs,l1q,l1s,qa-qb,ds})if(!std::isfinite(x))throw std::runtime_error("nonfinite result");
 std::cout<<std::setprecision(17)<<maxq<<' '<<maxs<<' '<<l1q<<' '<<l1s<<' '<<qa-qb<<' '<<ds<<'\n';
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
