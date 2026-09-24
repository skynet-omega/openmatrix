from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
import numpy as np
import verify as v


def fixture(arm='plus'):
    n=440;g=v.GEOMETRY
    q=np.broadcast_to(g['prepared_qpos_root'],(n,7)).copy()
    antenna=np.broadcast_to(g['prepared_antennae_mm'],(n,2,3)).copy()
    antenna[40:,:,0]+=np.arange(1,401)[:,None]*.001
    actual=np.zeros((n,4));actual[40:,2]=np.arange(1,401)*(.00001 if arm=='plus' else .000001)
    usedq=np.vstack((np.zeros((1,4)),actual[:-1]));baseline=np.zeros_like(actual)
    spec=v.specs_from_geometry()[arm];after=v.concentration(antenna[40:],spec)
    before=np.vstack((v.concentration(antenna[39],spec),after[:-1]))
    t={'fase':np.array(['preparacion']*40+['ensayo']*400),'paso':np.r_[np.arange(1,41),np.arange(1,401)],
       'CNS_time_ns':44_000_000_000+np.arange(1,441,dtype=np.int64)*v.DT,
       'qpos':q,'antenas_mm':antenna,'DN_q_actual':actual,'DN_q_usada':usedq,'DN_baseline':baseline,
       'sensores_usados':np.vstack((np.zeros((40,3)),np.c_[before,np.zeros(400)])),
       'sensores_pendientes':np.vstack((np.zeros((40,3)),np.c_[after,np.zeros(400)])),
       'concentracion_campo':np.vstack((np.zeros((40,2)),after)),
       'command_yaw_rate_rad_s':np.tanh(250*usedq[:,2])*np.deg2rad(5.),
       'command_forward_mm_s':np.full(n,.2),'yaw_delta_deg':np.zeros(n),'PN_general_transmission':np.zeros((n,2))}
    t['PN_time_ns']=t['CNS_time_ns'].copy();t['body_time_ns']=t['CNS_time_ns'].copy()
    return t


def intervals(t,arm='plus'):
    spec=v.specs_from_geometry()[arm]
    return {'field':dict(arm=arm,sampling_order=['L','R'],source_mm=spec['source_mm'],sigma_mm=spec['sigma_mm'],geometry_sha256=spec['geometry_sha256'],installed_ns=int(t['CNS_time_ns'][39])),
            'prepared_pose_guard':dict(geometry_sha256=spec['geometry_sha256'],qpos_root_max_abs_native=0.,antennae_max_abs_mm=0.),
            'intervals':[dict(sample_used_ns=int(t['CNS_time_ns'][i])-v.DT,sample_pending_ns=int(t['CNS_time_ns'][i]),
                             used=t['sensores_usados'][i].tolist(),pending=t['sensores_pendientes'][i].tolist(),
                             expected_80c_used_NOT_consumer_witness=(80*t['sensores_usados'][i,:2]).tolist(),
                             PN_available_NOT_consumer_witness=t['PN_general_transmission'][i].tolist(),
                             command_yaw_rate_rad_s=float(t['command_yaw_rate_rad_s'][i])) for i in range(40,440)],
            'complete':True,'stage4_admission':False}


def desc(a,indices):
    a=np.asarray(a,dtype=np.float64)
    return dict(dtype='float64',shape=list(a.shape),bytes=a.nbytes,sha256=hashlib.sha256(a.tobytes()).hexdigest(),indices=indices,values=a[indices].tolist())


def drive_fixture(t):
    selected=v.PLAN['observation']['bounded_drive_epochs'];labels=[str(f)+':'+str(k) for f,k in zip(t['fase'],t['paso'])]
    d={'selected_epochs':selected,'source':dict(adapter_sha256=v.LOCK[str(v.ROOT/'campanas/etapa3_motor_nuevo_20260922/organism_adapter.py')],observer_sha256=v.LOCK[str(v.ROOT/'motor_nuevo/adapter_drive_observer_20260923/observer.py')]),
       'capture_limits':v.read_json(v.ROOT/'motor_nuevo/adapter_drive_observer_20260923/PLAN.json')['capture_limits'],
       'epochs':[dict(label=s,calls=16,captured=2 if s in selected else 0) for s in labels],
       'coefficient_build_calls_observed':18,'actual_pn_reader_calls_during_build':18,'pn_capture_binding_observed':True,
       'open_epoch':None,'failure':None,'failed':False,'records':[]}
    for i,s in enumerate(labels):
        if s not in selected:continue
        a=np.array([80*t['sensores_usados'][i,0]]*4+[80*t['sensores_usados'][i,1]]*4,dtype=np.float64)
        for ordinal in (0,1):
            r=dict(epoch=s,ordinal=ordinal,start_time_ns=int(t['CNS_time_ns'][i])-v.DT,duration_ns=62500 if ordinal==0 else 125000,
                   captured_device_pointers=[128,256],host_pn_reads=1,status='OBSERVED_EXACT_BOUNDARIES',argument_unchanged=True,buffers_unchanged_during_advance=True,
                   argument_dtype_before_validation='float64')
            for name in ('drive_argument','drive_before','drive_after'):r[name]=desc(a,list(range(8)))
            for name in ('pn_refresh','pn_before','pn_after'):r[name]=desc([.1,.2],[])
            d['records'].append(r)
    index=dict(capture_epochs=selected,selected_drive_rows=list(range(8)),ORN_DM1_L_all_rows=list(range(4)),ORN_DM1_R_all_rows=list(range(4,8)))
    return d,index


def route_fixture(folder):
    source=v.ROOT/'motor_nuevo/native_hybrid_20260922/legacy_sources/gpu_coefficient_layout.py'
    name='29__legacy_sources__gpu_coefficient_layout.py'
    (folder/'executed_sources').mkdir(exist_ok=True)
    shutil.copy2(source,folder/'executed_sources'/name)
    frozen=v.read_json(folder/'FROZEN.json') if (folder/'FROZEN.json').exists() else {}
    frozen[name]=v.sha(source);(folder/'FROZEN.json').write_text(json.dumps(frozen))
    (folder/'INTERVENTION.json').write_text(json.dumps(dict(operation='disable left PN629 general replacement',
        only_changes=['general_outputs.enabled','record_sha256'],dynamic_466_enabled=True,left_pn_id=10208,
        anatomy_changed=False,motor_decoder_changed=False,after_hash='a'*64)))
    for name in ('prepared_state','final_state'):
        state=folder/name;state.mkdir(exist_ok=True)
        manifest=dict(general_outputs={'enabled':False},electrical_outputs={'enabled':True},
                      orn_peripheral_terminal={'local_PN_feedback_pairs':31,'recurrent_connected':True},record_sha256='a'*64)
        (state/'session.json').write_text(json.dumps({'hybrid':{'pn_online_manifest':manifest}}))
        update_route_inventory(state)


def update_route_inventory(state):
    path=state/'session.json'
    (state/'MANIFEST.json').write_text(json.dumps({'files':{'session.json':{'bytes':path.stat().st_size,'sha256':v.sha(path)}}}))


def flow_fixture(t):
    length=max(v.ROWS)+1;operator={k:np.full(length,value) for k,value in [('gain',.001),('theta',1.),('tau',.02)]}
    records=[];raw=np.tile(np.arange(6,dtype=float),(440,1));target=np.maximum(0,np.tanh(.001*(raw-1)))
    metadata={'kernel_sha256':v.kernel_hashes(),'rows':{str(identity):dict(row=row,type='DM1_lPN' if j<2 else 'DNa02' if j<4 else 'DNb05',side='R' if j%2==0 else 'L',operator='orn_pn_synaptic' if j<2 else 'ordinary_rate') for j,(identity,row) in enumerate(zip(v.IDS,v.ROWS))}}
    for i in range(440):
        records.append(dict(phase=str(t['fase'][i]),ms=int(t['paso'][i]),time_ns=int(t['CNS_time_ns'][i]),graph_build_coefficient_calls=18,target_max_error=0.,rate_max_error=0.,
                            rows={str(k):dict(raw_signed=float(raw[i,j]),target=float(target[i,j]),rate=50.,drive=0.,theta=1.) for j,k in enumerate(v.IDS)}))
    arrays=dict(time_ns=t['CNS_time_ns'],phase=t['fase'],ids=np.array(v.IDS),raw_signed=raw,target=target)
    return records,arrays,metadata,operator


class Tests(unittest.TestCase):
    def test_route_is_semantic_and_frozen(self):
        with tempfile.TemporaryDirectory(dir=v.HERE) as tmp:
            folder=Path(tmp);route_fixture(folder)
            self.assertEqual(v.check_pn_route(folder)['saved_states']['prepared_state']['local_PN_feedback_pairs'],31)
            for case in ('enabled','dynamic','count','recurrent','hash','layout','wrong_semantic_path'):
                route_fixture(folder)
                path=folder/'final_state/session.json';saved=v.read_json(path);m=saved['hybrid']['pn_online_manifest']
                if case=='enabled':m['general_outputs']['enabled']=True
                elif case=='dynamic':m['electrical_outputs']['enabled']=False
                elif case=='count':m['orn_peripheral_terminal']['local_PN_feedback_pairs']=30
                elif case=='recurrent':m['orn_peripheral_terminal']['recurrent_connected']=False
                elif case=='hash':m['record_sha256']='b'*64
                elif case=='layout':(folder/'executed_sources/29__legacy_sources__gpu_coefficient_layout.py').write_text('changed')
                else:saved={'history':[saved],'hybrid':{'pn_online_manifest':dict(m,general_outputs={'enabled':True})}}
                path.write_text(json.dumps(saved));update_route_inventory(path.parent)
                with self.subTest(case=case),self.assertRaises(ValueError):v.check_pn_route(folder)
            route_fixture(folder);(folder/'final_state/session.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'integrity'):v.check_pn_route(folder)

    def test_reader_counts_cannot_hide_general_replacement(self):
        t=fixture()
        for count,binding in ((0,False),(36,True),(17,True),(18,False),(True,True)):
            d,index=drive_fixture(t);d['actual_pn_reader_calls_during_build']=count;d['pn_capture_binding_observed']=binding
            with self.subTest(count=count,binding=binding),self.assertRaisesRegex(ValueError,'reader count/binding'):
                v.check_drive(d,index,t)

    def test_inherited_quaternion_domain(self):
        t=fixture();t['qpos'][100,3:7]*=1.1
        with self.assertRaisesRegex(ValueError,'Quaternion norm-squared domain'):
            v.check_trace(t,'plus')

    def test_observed_dtype_and_unwarranted_rounding(self):
        t=fixture();d,index=drive_fixture(t)
        d['records'][4]['argument_dtype_before_validation']='float32'
        with self.assertRaisesRegex(ValueError,'Actual drive argument dtype'):
            v.check_drive(d,index,t)
        d,index=drive_fixture(t)
        for name in ('drive_argument','drive_before','drive_after'):
            row=d['records'][4][name]
            rounded=np.asarray(row['values'],dtype=np.float32).astype(np.float64)
            d['records'][4][name]=desc(rounded,list(range(8)))
        with self.assertRaisesRegex(ValueError,'Actual bilateral ORN drive'):
            v.check_drive(d,index,t)

    def test_valid_trace_and_interval_reconstruction(self):
        for arm in ('plus','minus'):
            t=fixture(arm);m=v.check_trace(t,arm);v.check_intervals(intervals(t,arm),t,arm,m)
            self.assertEqual(m['steps'],400)
            self.assertAlmostEqual(m['signed_command_deg'],np.rad2deg(t['command_yaw_rate_rad_s'][40:].sum()*.001))

    def test_trace_corruptions(self):
        for key,index,delta in [('PN_time_ns',42,1),('paso',40,1),('sensores_usados',(41,0),.01),('DN_q_usada',(40,2),.1),('concentracion_campo',(41,0),.1),('yaw_delta_deg',50,.01),('command_yaw_rate_rad_s',60,.01),('qpos',(20,0),np.nan)]:
            with self.subTest(key=key):
                t=fixture();t[key][index]+=delta
                with self.assertRaises(ValueError):v.check_trace(t,'plus')
        t=fixture();t['antenas_mm'][41]=t['antenas_mm'][41,::-1]
        with self.assertRaises(ValueError):v.check_trace(t,'plus')

    def test_claimed_complete_and_source_corruption(self):
        t=fixture();m=v.check_trace(t,'plus')
        for field in ('complete','source'):
            d=intervals(t)
            if field=='complete':d['complete']=False
            else:d['field']['source_mm'][0]+=.01
            with self.assertRaises(ValueError):v.check_intervals(d,t,'plus',m)

    def test_drive_hash_index_clock_value_corruptions(self):
        t=fixture();d,index=drive_fixture(t);self.assertEqual(v.check_drive(d,index,t)['captured_calls'],16)
        for key in ('hash','value','clock','ordinal','index','pn_binding'):
            with self.subTest(key=key):
                bad=copy.deepcopy(d);ix=copy.deepcopy(index)
                if key=='hash':bad['records'][0]['drive_after']['sha256']='0'*64
                elif key=='value':
                    for name in ('drive_argument','drive_before','drive_after'):bad['records'][0][name]['values'][0]=1.
                elif key=='clock':bad['records'][0]['start_time_ns']+=1
                elif key=='ordinal':bad['records'][1]['ordinal']=0
                elif key=='index':ix['ORN_DM1_R_all_rows'][0]=0
                else:bad['pn_capture_binding_observed']=False
                with self.assertRaises(ValueError):v.check_drive(bad,ix,t)

    def test_flow_recompute_and_raw_corruption(self):
        t=fixture();r,a,m,o=flow_fixture(t);self.assertEqual(v.check_flow(r,a,m,t,o)['samples'],440)
        for kind in ('raw','target','clock','id','kernel'):
            with self.subTest(kind=kind):
                rr,aa,mm=copy.deepcopy((r,a,m))
                if kind=='raw':rr[0]['rows']['10176']['raw_signed']=900.
                elif kind=='target':aa['target'][0,0]=.1
                elif kind=='clock':rr[0]['time_ns']+=1
                elif kind=='id':aa['ids'][0]+=1
                else:mm['kernel_sha256']['pn_tap']='0'*64
                with self.assertRaises(ValueError):v.check_flow(rr,aa,mm,t,o)

    def test_window_is_300_to_400_inclusive(self):
        a=dict(complete=True,concentration=np.zeros((400,2)),command=np.zeros(400),yaw=np.zeros(400));b=copy.deepcopy(a)
        a['concentration'][298]=100.;a['concentration'][299,0]=1.;a['concentration'][399,1]=1.;a['command'][399]=1.
        m=v.pair_metrics(a,b);self.assertAlmostEqual(m['late_concentration_mean_abs'],2/202)
        self.assertAlmostEqual(m['command_L1_deg'],.001*180/np.pi)

    def test_decisions_missing_B_C_A_and_parity(self):
        self.assertEqual(v.decide({})['classification'],'INCOMPLETO')
        a=dict(complete=True,concentration=np.zeros((400,2)),command=np.zeros(400),yaw=np.zeros(400));b=copy.deepcopy(a)
        runs={('causal_cuda','plus'):{'metrics':a},('causal_cuda','minus'):{'metrics':b}}
        self.assertEqual(v.decide(runs)['rival'],'C')
        b['concentration']+=.03;self.assertEqual(v.decide(runs)['rival'],'B')
        b['command']+=.01;self.assertEqual(v.decide(runs)['classification'],'PROMETEDOR_NO_CONFIRMADO')
        runs['reference_cuda','plus']={'metrics':copy.deepcopy(a)};runs['reference_cuda','minus']={'metrics':copy.deepcopy(b)}
        self.assertEqual(v.decide(runs)['rival'],'A')
        runs['reference_cuda','minus']['metrics']['yaw'][20]=.003
        self.assertIsNone(v.decide(runs)['rival'])
        self.assertEqual(v.decide(runs,invalid=True)['rival'],'C')

    def test_frozen_plan_and_absent_campaign(self):
        self.assertEqual(v.source_check(v.CAMPAIGN)['sources'],len(v.LOCK))
        with tempfile.TemporaryDirectory(dir=v.HERE) as tmp:
            p=Path(tmp);(p/'PLAN.json').write_text('{}');(p/'SOURCE_LOCK.json').write_text('{}')
            with self.assertRaises(ValueError):v.source_check(p)

    def test_budget_recomputed_and_corruptions(self):
        t=fixture();rows=[dict(phase=str(f),step=int(k),clock_ns=int(clock),elapsed_s=float(i+1),step_wall_s=.5,
                             rss_peak_sample_gib=1.,gpu_device_used_sample_gib=2.,gpu_pool_used_sample_gib=1.,yaw_delta_deg=0.)
                        for i,(f,k,clock) in enumerate(zip(t['fase'],t['paso'],t['CNS_time_ns']))]
        result={'wall_total_s':441.}
        self.assertEqual(v.check_budget_progress(rows,result,t)['wall_s'],441.)
        for kind in ('ram','gpu','wall','clock','nonfinite'):
            with self.subTest(kind=kind):
                rr=copy.deepcopy(rows);r=dict(result)
                if kind=='ram':rr[-1]['rss_peak_sample_gib']=18.01
                elif kind=='gpu':rr[-1]['gpu_device_used_sample_gib']=12.01
                elif kind=='wall':r['wall_total_s']=2051.
                elif kind=='clock':rr[-1]['clock_ns']+=1
                else:rr[-1]['elapsed_s']=float('nan')
                with self.assertRaises(ValueError):v.check_budget_progress(rr,r,t)

    def test_prepared_receipt_raw_hash_and_flag_corruption(self):
        comparator=v.ROOT/'motor_nuevo/gaussian_prepared_semantics_20260923/compare_prepared.py'
        spec=importlib.util.spec_from_file_location('prepared_comparator_fixture',comparator)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=v.HERE) as tmp:
            base=Path(tmp);folders=[]
            for arm in ('plus','minus'):
                folder=base/arm/'prepared_state';folder.mkdir(parents=True);folders.append(folder)
                contract={'engine':'causal_cuda','source_identity':{'source_lock_sha256':'1'*64},'plan_sha256':'2'*64,'checkpoint_manifest_sha256':'3'*64,'preparation_ms':40}
                (folder.parent/'RUN_CONTRACT.json').write_text(json.dumps(contract))
                for kind in ('session','prosthesis','published','effective_operator'):
                    (folder/(kind+'.json')).write_text(json.dumps({'data':{'__array__':'array_0'}}))
                    np.savez_compressed(folder/(kind+'.npz'),array_0=np.array([1.,2.]))
                (folder/'boundary.json').write_text('{}')
                entries={p.name:{'bytes':p.stat().st_size,'sha256':v.sha(p)} for p in folder.iterdir()}
                (folder/'MANIFEST.json').write_text(json.dumps({'files':entries}))
            receipt=module.compare_prepared(*folders,'causal_cuda');receipt['comparator_sha256']=v.sha(comparator)
            out=base/'receipt.json';out.write_text(json.dumps(receipt))
            self.assertTrue(v.check_prepared_receipt(out,*folders,'causal_cuda')['recomputed_section_equality'])
            for kind in ('flag','payload','metadata'):
                bad=copy.deepcopy(receipt)
                if kind=='flag':bad['scientific_state_exact']=False
                elif kind=='payload':bad['sections']['session']['arrays'][0]['payload_sha256_right']='0'*64
                else:bad['sections']['session']['right_metadata_sha256']='0'*64
                out.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):v.check_prepared_receipt(out,*folders,'causal_cuda')
            out.write_text(json.dumps(receipt));(folders[0]/'session.npz').write_bytes(b'corruption')
            with self.assertRaises(ValueError):v.check_prepared_receipt(out,*folders,'causal_cuda')

    def test_complete_arm_disk_reader_with_cpu_fixture(self):
        # These files are explicitly synthetic; no organism state or trajectory
        # is claimed. Exercise the same disk path used for future real arms.
        with tempfile.TemporaryDirectory(dir=v.HERE) as tmp:
            p=Path(tmp);(p/'executed_sources').mkdir();(p/'flow').mkdir();(p/'prepared_state').mkdir()
            def save(name,data):(p/name).write_text(json.dumps(data,allow_nan=False))
            t=fixture();np.savez_compressed(p/'traces.npz',**t)
            sources={}
            for i,(path,digest) in enumerate(v.LOCK.items()):
                name='29__legacy_sources__gpu_coefficient_layout.py' if Path(path).name=='gpu_coefficient_layout.py' else str(i)+'_'+Path(path).name
                shutil.copy2(path,p/'executed_sources'/name);sources[name]=digest
            save('FROZEN.json',sources)
            route_fixture(p)
            save('RUN_CONTRACT.json',dict(field='plus',engine='causal_cuda',plan_sha256=v.sha(v.HERE/'reference/PLAN.json'),
                                         source_identity=dict(source_lock_sha256=v.sha(v.HERE/'reference/SOURCE_LOCK.json'),sources_count=len(v.LOCK)),
                                         preparation_ms=40,trial_ms=400,drive_capture_epochs=v.PLAN['observation']['bounded_drive_epochs']))
            save('GAUSSIAN_SPEC.json',v.specs_from_geometry());save('GAUSSIAN_INTERVALS.json',intervals(t))
            m=v.check_trace(t,'plus');save('GAUSSIAN_INITIAL_GUARD.json',dict(**m['initial'],pair_max_abs=m['initial_pair_abs'],limit=v.PLAN['preflight']['initial_plus_minus_concentration_max_abs']))
            drive,index=drive_fixture(t);save('DRIVE_OBSERVATION.json',drive);save('DRIVE_INDEX_MAP.json',index)
            r,a,metadata,operator=flow_fixture(t)
            np.savez_compressed(p/'flow/FLOW.npz',**a);save('flow/METADATA.json',metadata)
            (p/'flow/FLOW.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in r))
            values={**operator,**{'cuda_'+k:val for k,val in operator.items()}}
            manifests={k:dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest()) for k,a in values.items()}
            bindings={k:[k] for k in values};identity=dict(schema='effective_operator_v2',bindings=bindings,manifest=manifests)
            op={**identity,'identity_sha256':hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'values':{k:{'__array__':'array_'+str(i)} for i,k in enumerate(values)}}
            save('prepared_state/effective_operator.json',op)
            np.savez_compressed(p/'prepared_state/effective_operator.npz',**{'array_'+str(i):a for i,a in enumerate(values.values())})
            progress=[dict(phase=str(f),step=int(k),clock_ns=int(clock),elapsed_s=float(i+1),step_wall_s=.5,
                           rss_peak_sample_gib=1.,gpu_device_used_sample_gib=2.,gpu_pool_used_sample_gib=1.,yaw_delta_deg=0.)
                      for i,(f,k,clock) in enumerate(zip(t['fase'],t['paso'],t['CNS_time_ns']))]
            (p/'PROGRESS.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in progress))
            save('RESULT.json',dict(field='plus',engine='causal_cuda',runtime={'profile':'causal_cuda'},completed_trial_ms=400,
                                   completed_preparation_ms=40,stage4_pass=None,wall_total_s=441.,error=None,cleanup_errors=[],status='COMPLETE'))
            self.assertTrue(v.inspect_arm(p)['metrics']['complete'])
            t['sensores_usados'][200,0]=float('nan');np.savez_compressed(p/'traces.npz',**t)
            with self.assertRaises(ValueError):v.inspect_arm(p)


if __name__=='__main__':unittest.main(verbosity=2)
