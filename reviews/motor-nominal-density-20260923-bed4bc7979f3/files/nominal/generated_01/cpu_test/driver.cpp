
#include CONTROL_FILE
#include <cstdio>
#include <cstring>
static_assert(sizeof(long)==8,"Linux64 ABI required");
int main(){
 const char*names[]={"no_cut","easy_tail","rejection","new_rejection","domain",
 "nonfinite","flag","short_reject","fractional","bad_events","bad_budget",
 "threshold_equal","late_domain","ordinary_no_events"};
 puts("[");
 for(int i=0;i<14;i++){
  double clock[2]={},status[3]={},x[1]={0},fine[1]={0};
  int mode=i==0?1:i==2?2:i==3?3:i==4?4:i==5?5:i==6?6:
           i==7?7:i==11?8:i==12?9:0;
  Mock g{clock,status,x,fine,mode,{}};
  void*r=engine_create(&g,(void*)1,clock,status,x,fine,1);if(!r)return 3;
  long n=i==0?500:2000,c[3]={};double e=0;
  double simple[]={80e-9}, reject[]={800e-9};
  double fractional[]={0,80e-9,80e-9,1.000001e-6,3.99995e-6};
  double bad[]={2e-6,1e-6};
  const double*ev=i==0?nullptr:i==2?reject:i==8?fractional:i==9?bad:simple;
  long count=i==0?0:i==8?5:i==9?2:1;double budget=i==10?0:10.;
  int code=i==13?engine_advance(r,4000,&n,100,4000,budget,c,&e):
                 engine_advance_events(r,4000,&n,100,4000,budget,ev,count,c,&e);
  bool cuts=true;
  for(auto row:g.log)for(long k=0;k<count;k++){
   if(i==13)break;
   if(row[0]<ev[k]&&row[0]+row[1]>ev[k]+1e-20)cuts=false;
  }
  double partial=x[0];if(code)x[0]=0.; // Emulates graph_core's outer rollback.
  printf("%s{\"case\":\"%s\",\"return\":%d,\"next\":%ld,\"accepted\":%ld,\"rejected\":%ld,"
         "\"state_hex\":\"%a\",\"partial_hex\":\"%a\",\"cuts_ok\":%s,\"error\":\"%s\",\"trace\":[",
     i?",":"",names[i],code,n,c[0],c[1],x[0],partial,cuts?"true":"false",engine_error());
  for(size_t k=0;k<g.log.size();k++){
   auto v=g.log[k];printf("%s[\"%a\",\"%a\",\"%a\",\"%a\"]",k?",":"",v[0],v[1],v[2],v[3]);
  }
  puts("]}");engine_destroy(r);
 }
 puts("]");return 0;
}
