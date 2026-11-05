"""逐图collate血缘检查：小CPU张量、无媒体/模型/采样更新。"""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

_ENV=os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
from futsalmot.batches import collate_bbox_samples

_THREADS=None
_BLOCK=None

def setUpModule():
    global _THREADS,_BLOCK
    _THREADS=torch.get_num_threads();torch.set_num_threads(2)
    _BLOCK=mock.patch("torch.cuda._lazy_init",side_effect=AssertionError("CPU fixture禁止CUDA初始化"));_BLOCK.start()

def tearDownModule():
    if _BLOCK is not None:_BLOCK.stop()
    if _THREADS is not None:torch.set_num_threads(_THREADS)
    if _ENV is None:os.environ.pop("CUDA_VISIBLE_DEVICES",None)
    else:os.environ["CUDA_VISIBLE_DEVICES"]=_ENV


def prepared(root,domain="R",stream=None,exposure=0,empty=False,split="train"):
    stream=stream or domain
    uid=f"fixture:{domain}:one_UID"
    s={"img":torch.zeros((3,960,960),dtype=torch.uint8),"uid":uid,"domain":domain,
       "uid_field":"sample_uid" if domain == "R" else "derived_sample_uid",
       "cls":torch.empty((0,1)) if empty else torch.zeros((1,1)),
       "bboxes":torch.empty((0,4)) if empty else torch.tensor([[.5,.5,.25,.25]]),
       "ignore_bboxes":torch.empty((0,4)) if empty else torch.tensor([[0.,0.,8.,8.]]),
       "targets":[] if empty else [{"track_id":1}],"ignore_targets":[] if empty else [{"track_id":2}],
       "absent_targets":[],"geometry":{"input_shape":(960,960),"source_uid":uid}}
    req={"uid":uid,"domain":stream,"epoch":0,"exposure_index":exposure,"augmentation_seed":100+exposure}
    ref={"domain":domain,"split":split}
    if domain == "R":ref["index_path"]=str(root/"dataset.json")
    else:ref.update(seed_dir=str(root/"seed_00000001"),camera_id="C5",frame_index=1)
    s.update(request=req,source_reference=ref,image_source_binding={"path":str(root/"fixture.png"),"stat":[1,2,3,4]},
             sampling_stream=stream,data_domain=domain)
    return s


class BatchLineageTests(unittest.TestCase):
    def test_R_S_R_extra_duplicate_UID_exposure_order_and_empty_targets(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            items=[prepared(root,exposure=0,empty=True),prepared(root,"S",exposure=7),
                   prepared(root,stream="R_extra",exposure=12),prepared(root,exposure=2)]
            result=collate_bbox_samples(items)
            self.assertEqual(result["uids"],[s["uid"] for s in items])
            self.assertEqual(result["domains"],["R","S","R","R"])
            self.assertEqual([row["sampling_stream"] for row in result["sample_lineage"]],["R","S","R_extra","R"])
            self.assertEqual([row["request"]["exposure_index"] for row in result["sample_lineage"]],[0,7,12,2])
            self.assertEqual(result["batch_idx"].tolist(),[1,2,3])
            self.assertEqual(result["ignore_batch_idx"].tolist(),[1,2,3])
            self.assertEqual(result["sample_lineage"][2]["data_domain"],"R")
            self.assertEqual(result["sample_lineage"][2]["request"]["domain"],"R_extra")
            self.assertEqual(result["sample_lineage"][2]["source_reference"]["split"],"train")

    def test_none_only_for_wholly_absent_old_letterbox_fields(self):
        with tempfile.TemporaryDirectory() as t:
            full=prepared(Path(t));plain=dict(full)
            for key in ("request","source_reference","image_source_binding","sampling_stream","data_domain"):
                plain.pop(key)
            result=collate_bbox_samples([plain,full])
            self.assertIsNone(result["sample_lineage"][0])
            self.assertIsNotNone(result["sample_lineage"][1])
            for missing in ("request","source_reference","image_source_binding","sampling_stream","data_domain"):
                partial=dict(full);partial.pop(missing)
                with self.assertRaisesRegex(ValueError,"五字段"):collate_bbox_samples([partial])

    def test_each_field_and_repeated_sample_is_deeply_isolated(self):
        with tempfile.TemporaryDirectory() as t:
            a=prepared(Path(t));b=prepared(Path(t),exposure=1)
            b["source_reference"]=a["source_reference"]
            b["image_source_binding"]=a["image_source_binding"]
            result=collate_bbox_samples([a,b])
            first,second=result["sample_lineage"]
            self.assertIsNot(first["source_reference"],second["source_reference"])
            self.assertIsNot(first["image_source_binding"],second["image_source_binding"])
            a["request"]["augmentation_seed"]=-999
            a["image_source_binding"]["stat"][2]=999
            a["source_reference"]["split"]="test"
            self.assertEqual(first["request"]["augmentation_seed"],100)
            self.assertEqual(first["image_source_binding"]["stat"],[1,2,3,4])
            first["image_source_binding"]["stat"][2]=8
            first["source_reference"]["split"]="val"
            self.assertEqual(second["image_source_binding"]["stat"],[1,2,3,4])
            self.assertEqual(second["source_reference"]["split"],"train")

    def test_compatible_explicit_R_extra_domain_R_and_validation_split_no_default(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            extra=prepared(root,stream="R_extra")
            extra["request"].update(domain="R",stream="R_extra",data_domain="R")
            rval=prepared(root,split="val") # 请求未提供split/view，不能擅自默认train。
            sdev=prepared(root,"S",split="dev");sdev["request"]["view"]="val"
            got=collate_bbox_samples([extra,rval,sdev])
            self.assertEqual([row["source_reference"]["split"] for row in got["sample_lineage"]],["train","val","dev"])

    def test_request_reference_and_image_binding_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            original=prepared(Path(t))
            changes=[lambda s:s["request"].update(uid="wrong"),
                     lambda s:s["request"].update(domain="S"),
                     lambda s:s["request"].update(stream="S"),
                     lambda s:s["request"].update(exposure_index=True),
                     lambda s:s["request"].update(augmentation_seed=-1),
                     lambda s:s["request"].update(epoch=1.5),
                     lambda s:s["request"].pop("epoch"),
                     lambda s:s["request"].update(split="val"),
                     lambda s:s["request"].update(view="val"),
                     lambda s:s.update(data_domain="S"),
                     lambda s:s["source_reference"].update(domain="S"),
                     lambda s:s["source_reference"].update(split="test"),
                     lambda s:s["image_source_binding"].update(path="relative.png"),
                     lambda s:s["image_source_binding"].update(stat=[1,2,0,4]),
                     lambda s:s["image_source_binding"].update(stat=[1,2,3,False])]
            for change in changes:
                wrong=deepcopy(original);change(wrong)
                with self.assertRaises(ValueError):collate_bbox_samples([wrong])
            s=prepared(Path(t),"S")
            s["source_reference"]["split"]="val"
            with self.assertRaises(ValueError):collate_bbox_samples([s])
            s=prepared(Path(t),"S");s["source_reference"]["frame_index"]=True
            with self.assertRaises(ValueError):collate_bbox_samples([s])

if __name__=="__main__":unittest.main()
