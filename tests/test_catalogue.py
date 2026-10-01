"""Negative controls mutate only temporary definition files, never biological data."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('validator', ROOT/'scripts/validate_catalogue.py')
validator=importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)

class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        shutil.copytree(ROOT/'knowledge',self.root/'knowledge')
    def tearDown(self):
        self.tmp.cleanup()
    def mutate(self,name,fn):
        path=self.root/'knowledge'/name
        doc=json.loads(path.read_text(encoding='utf-8')); fn(doc)
        path.write_text(json.dumps(doc),encoding='utf-8')
    def test_current_snapshot(self):
        self.assertFalse(validator.validate(self.root)['training_ready'])
    def test_unreviewed_gold_rejected(self):
        self.mutate('evidence_seed.json',lambda d:d['records'][0].update(training_eligible=True))
        with self.assertRaisesRegex(ValueError,'gold'): validator.validate(self.root)
    def test_frozen_role_rejected(self):
        self.mutate('datasets.json',lambda d:d['datasets'][-1].update(split='train'))
        with self.assertRaisesRegex(ValueError,'Frozen'): validator.validate(self.root)
    def test_broken_citation_rejected(self):
        self.mutate('modules.json',lambda d:d['modules'][0]['source_ids'].append('nonexistent'))
        with self.assertRaisesRegex(ValueError,'Unknown source'): validator.validate(self.root)
    def test_corrupt_go_snapshot_rejected(self):
        path=self.root/'knowledge/snapshots/go_2026-10-01/quickgo_response.json'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'checksum'): validator.validate(self.root)

if __name__=='__main__': unittest.main()
