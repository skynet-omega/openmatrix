"""Deliberate corruption checks for portable evidence readback; no network."""
import hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from verify_remote import validate_archive

class EvidenceReadback(unittest.TestCase):
    def test_inventory_and_corruptions(self):
        data=b'actual measured values\n'
        expected=[{'path':'data/trace.csv','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]
        def package(payload=data,extra=None,manifest=None,duplicate=False):
            out=io.BytesIO()
            with zipfile.ZipFile(out,'w') as z:
                z.writestr('data/trace.csv',payload)
                z.writestr('MANIFEST.json',json.dumps({'files':expected if manifest is None else manifest}))
                if extra is not None:z.writestr(extra,b'unlisted')
                if duplicate:z.writestr('data/trace.csv',payload)
            return out.getvalue()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            p=Path(td)/'evidence.zip';p.write_bytes(package())
            self.assertEqual(validate_archive(p,expected),1)
            cases=[package(payload=b'changed measured values'),package(extra='extra.txt'),
                   package(extra='../escape.txt'),package(manifest=[dict(expected[0],sha256='0'*64)]),
                   package(duplicate=True)]
            for i,corrupt in enumerate(cases):
                with self.subTest(corruption=i):
                    p.write_bytes(corrupt)
                    with self.assertRaises(ValueError):validate_archive(p,expected)
if __name__=='__main__':unittest.main()
