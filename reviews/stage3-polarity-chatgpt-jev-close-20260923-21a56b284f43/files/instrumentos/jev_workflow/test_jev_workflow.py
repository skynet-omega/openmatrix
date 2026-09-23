import copy
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

p = Path(__file__).with_name('jev_workflow.py')
spec = importlib.util.spec_from_file_location('jev', p)
jev = importlib.util.module_from_spec(spec)
sys.modules['jev'] = jev
spec.loader.exec_module(jev)

class RouterTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [{'id':'repair', 'description':'Fix a UnicodeDecodeError in postprocessing.'}]
        self.payload = jev.build_payload(self.tasks)
        self.response = {'model':jev.MODEL, 'answers':{'repair':{
            'type':'choice', 'choice':'codex_implementation', 'confidence':.9,
            'probabilities':{k:float(k=='codex_implementation') for k in jev.ROUTES}}},
            'usage':{'input_tokens':200,'output_tokens':30}}

    def test_valid_and_always_advisory(self):
        r=jev.validate_response(self.response,self.payload)
        self.assertFalse(jev.disposition(r)['repair']['automatic_execution'])
        self.assertFalse(jev.disposition(r)['repair']['numerical_or_scientific_acceptance'])

    def test_secret_rejected_before_network(self):
        self.tasks[0]['description']='key: apikey_EXAMPLE_NOT_A_REAL_SECRET'
        with self.assertRaises(ValueError):jev.build_payload(self.tasks)

    def test_scope_is_bounded(self):
        with self.assertRaises(ValueError):jev.build_payload(self.tasks * 9)
        with self.assertRaises(ValueError):jev.build_payload(self.tasks * 2)
        self.tasks[0]['description']='x'*1501
        with self.assertRaises(ValueError):jev.build_payload(self.tasks)

    def test_bad_numeric_or_shape_rejected(self):
        for key,value in [('confidence',float('nan')),('choice','launch_shell'),('type','noul')]:
            r=copy.deepcopy(self.response);r['answers']['repair'][key]=value
            with self.assertRaises(ValueError):jev.validate_response(r,self.payload)
        r=copy.deepcopy(self.response);r['answers']['repair']['probabilities']['uncertain']=.5
        with self.assertRaises(ValueError):jev.validate_response(r,self.payload)

    def test_model_and_usage_not_trusted(self):
        r=copy.deepcopy(self.response);r['model']='unknown'
        with self.assertRaises(ValueError):jev.validate_response(r,self.payload)
        r=copy.deepcopy(self.response);r['usage']['input_tokens']=-1
        with self.assertRaises(ValueError):jev.validate_response(r,self.payload)

    def test_no_redirect_of_credentials(self):
        with self.assertRaises(RuntimeError):jev.NoRedirect().redirect_request(None,None,302,'',{},'https://example.org')

    def test_network_failure_no_retry(self):
        with patch.object(jev.urllib.request,'build_opener') as builder:
            builder.return_value.open.side_effect=jev.urllib.error.URLError('example')
            with self.assertRaisesRegex(RuntimeError,'zero automatic retries'):
                jev.call_api(self.payload,'FAKE_KEY_FOR_UNIT_TEST')
            self.assertEqual(builder.return_value.open.call_count,1)
            request=builder.return_value.open.call_args.args[0]
            self.assertEqual(request.get_header('User-agent'),jev.USER_AGENT)
            self.assertEqual(request.get_header('Accept'),'application/json')

    def test_http_error_records_safe_category_without_remote_body(self):
        raw=b'{"detail":{"error_type":"authentication_error","message":"do not print me"}}'
        error=jev.urllib.error.HTTPError(jev.ENDPOINT,403,'Forbidden',{},io.BytesIO(raw))
        with patch.object(jev.urllib.request,'build_opener') as builder:
            builder.return_value.open.side_effect=error
            with self.assertRaises(jev.TypeSafeHTTPError) as raised:
                jev.call_api(self.payload,'FAKE_KEY_FOR_UNIT_TEST')
            self.assertEqual((raised.exception.status,raised.exception.category),(403,'authentication_error'))
            self.assertNotIn('do not print me',str(raised.exception))
        self.assertEqual(jev.http_error_category(b'error code: 1010','FAKE_KEY_FOR_UNIT_TEST'),'cloudflare_1010')
        self.assertEqual(jev.http_error_category(b'FAKE_KEY_FOR_UNIT_TEST','FAKE_KEY_FOR_UNIT_TEST'),'credential_echo_discarded')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'failed'
            with self.assertRaises(jev.TypeSafeHTTPError):
                jev.run_advisory(self.tasks,out,live=True,key_provider=lambda:'FAKE_KEY_FOR_UNIT_TEST',
                    transport=Mock(side_effect=jev.TypeSafeHTTPError(403,'cloudflare_1010')))
            receipt=json.loads((out/'receipt.json').read_text())
            self.assertEqual((receipt['http_status'],receipt['http_category']),(403,'cloudflare_1010'))
            self.assertFalse(receipt['automatic_retry_allowed'])

    def test_repeated_request_uses_cache_without_key_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            transport=Mock(return_value=self.response)
            key=Mock(return_value='FAKE_KEY_FOR_UNIT_TEST')
            first=jev.run_advisory(self.tasks,root/'first',live=True,key_provider=key,transport=transport)
            second=jev.run_advisory(self.tasks,root/'second',live=True,
                                    key_provider=Mock(side_effect=AssertionError('Key requested on cache hit')),
                                    transport=Mock(side_effect=AssertionError('Duplicate charge')))
            self.assertEqual(transport.call_count,1)
            self.assertEqual(first['api_requests'],1)
            self.assertEqual(second['api_requests'],0)
            self.assertEqual(second['usage'],{'input_tokens':0,'output_tokens':0})
            self.assertEqual(second['source_response_usage'],self.response['usage'])
            self.assertEqual(second['estimated_USD'],0)

    def test_import_paid_result_then_reuse_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);previous=root/'previous';previous.mkdir()
            (previous/'request.json').write_bytes(jev.encode(self.payload))
            (previous/'response.json').write_bytes(jev.encode(self.response))
            receipt=jev.run_advisory(self.tasks,root/'imported',reuse_result=previous)
            self.assertEqual(receipt['mode'],'cache')
            self.assertEqual(receipt['api_requests'],0)
            changed=copy.deepcopy(self.tasks);changed[0]['description']='Another task entirely'
            with self.assertRaisesRegex(ValueError,'another request'):
                jev.run_advisory(changed,root/'mismatch',reuse_result=previous)

    def test_changed_task_does_not_reuse_previous_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            jev.save_cache(root/'cache',self.payload,self.response)
            changed=copy.deepcopy(self.tasks);changed[0]['description']='Profile GPU kernels'
            receipt=jev.run_advisory(changed,root/'new')
            self.assertEqual(receipt['state'],'PREPARED_NO_NETWORK')
            self.assertFalse((root/'new'/'response.json').exists())

    def test_damaged_cache_rejected_before_key_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache'
            jev.save_cache(cache,self.payload,self.response)
            path=cache/(jev.digest(self.payload)+'.json')
            record=json.loads(path.read_text());record['response']['usage']['input_tokens']+=1
            path.write_text(json.dumps(record))
            key=Mock();transport=Mock()
            with self.assertRaisesRegex(ValueError,'damaged'):
                jev.run_advisory(self.tasks,root/'bad',live=True,key_provider=key,transport=transport)
            key.assert_not_called();transport.assert_not_called()

    def test_uncertain_failure_blocks_resubmit_before_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);transport=Mock(side_effect=TimeoutError('Unknown server outcome'))
            with self.assertRaises(TimeoutError):
                jev.run_advisory(self.tasks,root/'first',live=True,
                                 key_provider=lambda:'FAKE_KEY_FOR_UNIT_TEST',transport=transport)
            key=Mock()
            with self.assertRaisesRegex(ValueError,'interrupted'):
                jev.run_advisory(self.tasks,root/'second',live=True,key_provider=key,transport=transport)
            self.assertEqual(transport.call_count,1);key.assert_not_called()
            lock=json.loads(next((root/'cache').glob('*.pending')).read_text())
            self.assertEqual(lock['receipt'],str(root/'first'/'receipt.json'))

    def test_inflight_request_cannot_be_entered_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            with jev.single_request(tmp,self.payload):
                with self.assertRaisesRegex(ValueError,'in flight'):
                    with jev.single_request(tmp,self.payload):
                        self.fail('Duplicate request entered')
            self.assertFalse(list(Path(tmp).glob('*.pending')))

    def test_existing_cache_response_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            jev.save_cache(tmp,self.payload,self.response)
            another=copy.deepcopy(self.response);another['usage']['output_tokens']+=1
            jev.save_cache(tmp,self.payload,another)
            self.assertEqual(jev.read_cache(tmp,self.payload),self.response)
            self.assertEqual(len(list(Path(tmp).iterdir())),1)

if __name__=='__main__':unittest.main()
