"""Split-safety regressions; synthetic inputs only, no private identifiers."""
import unittest

from scripts.private_split_audit import build_graph
from scripts.rebuild_split_protocol import assign_components
from v3.build_sft_v3 import intensity_of


class PrivateRebuildTests(unittest.TestCase):
    def test_full_frame_box_uses_source_delta_without_fabricating_ring(self):
        sample=dict(fin=101, tin=100, fring=None,tring=None,fring_c=None,tring_c=None)
        self.assertEqual(intensity_of(sample,"remove",1),("brighter","source_delta_no_ring"))
        self.assertEqual(intensity_of(dict(sample,fin=100),"remove",1),("none","unavailable_ring_unknown"))

    def test_every_derived_forgery_and_known_patient_connected(self):
        rows={'real-a':{'source':'real-a','patient_id':'dataset:patient-1'},
              'real-b':{'source':'real-b','patient_id':'dataset:patient-1'},
              'fake-a':{'source':'real-a'},'fake-b':{'source':'real-b'}}
        graph,_=build_graph(rows)
        self.assertEqual(len({graph.f(path) for path in rows}),1)

    def test_distinct_identity_namespaces_remain_separate(self):
        rows={'a':{'source':'a','patient_id':'dataset-a:123'},'b':{'source':'b','patient_id':'dataset-b:123'}}
        graph,_=build_graph(rows)
        self.assertNotEqual(graph.f('a'),graph.f('b'))

    def test_same_decoded_pixels_connect_different_encoded_bytes(self):
        rows={'a':{'source':'a','content_md5':'bytes1','pixel_sha256':'pixels'},
              'b':{'source':'b','content_md5':'bytes2','pixel_sha256':'pixels'}}
        graph,_=build_graph(rows)
        self.assertEqual(graph.f('a'),graph.f('b'))

    def test_new_content_link_merges_prior_groups_without_dropping_rows(self):
        records={'a':{'component':'content:shared'},'b':{'component':'content:shared'},'c':{'component':'content:other'}}
        old=[{'orig':'a','group':'old-a'},{'orig':'b','group':'old-b'},{'orig':'c','group':'old-c'}]
        groups,merges=assign_components(records,old)
        self.assertEqual(set(groups),set(records))
        self.assertEqual(groups['a'],groups['b'])
        self.assertNotEqual(groups['a'],groups['c'])
        self.assertEqual(len(merges),1)

    def test_new_group_assignment_is_portable_across_workspace_paths(self):
        first={'/mount-a/image':{'component':'path-dependent-component-a','sha256':'same-image-bytes'}}
        second={'/mount-b/image':{'component':'path-dependent-component-b','sha256':'same-image-bytes'}}
        groups_a,_=assign_components(first,[])
        groups_b,_=assign_components(second,[])
        self.assertEqual(next(iter(groups_a.values())),next(iter(groups_b.values())))


if __name__=='__main__':
    unittest.main()
