"""给定框/极小自造RGB的候选CPU后处理检查，无真实推理或数据集成绩。"""
from copy import deepcopy
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from futsalmot.adapters import bbox_geometry
from futsalmot.batches import letterbox_bbox_sample
from futsalmot.postprocess import postprocess_decoded_person, write_candidate_evaluation, DEFINITION


def geometry(uid="fixture", h=960, w=960):
    ratio=min(960/h,960/w)
    resized=(round(h*ratio),round(w*ratio))
    dw,dh=(960-resized[1])/2,(960-resized[0])/2
    return {"source_uid":uid,"original_shape":(h,w),"input_shape":(960,960),
            "resized_shape":resized,"scale":(ratio,ratio),
            "padding_ltrb":(round(dw-.1),round(dh-.1),round(dw+.1),round(dh+.1)),
            "channel_order":"RGB","letterbox_candidate_only":True}


class PostprocessTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,self.threads)
        for name in ("_lazy_init","init","is_available"):
            guard=patch.object(torch.cuda,name,side_effect=AssertionError("禁止CUDA初始化/探测"))
            guard.start();self.addCleanup(guard.stop)

    def test_person_first_native_nms_and_full_raw_filter_reasons(self):
        rows=[[0,0,10,10,.99,3],[0,0,10,10,.9,0],
              [.5,.5,10.5,10.5,.8,0],[50,50,60,60,.001,0],[30,30,40,40,.7,0]]
        before=deepcopy(rows)
        result=postprocess_decoded_person("fixture",rows,geometry=geometry())
        self.assertEqual([p["input_prediction_index"] for p in result["predictions"]],[1,4])
        self.assertEqual(len(result["raw_predictions"]),5)
        self.assertEqual(result["raw_predictions"][0]["filter_reasons"],["non_person"])
        self.assertEqual(result["raw_predictions"][2]["filter_reasons"],["nms_suppressed"])
        self.assertEqual(result["raw_predictions"][3]["filter_reasons"],["score_below_or_equal_conf"])
        self.assertEqual(rows,before)
        self.assertTrue(result["candidate_only"])
        self.assertFalse(result["official_benchmark"])
        tensor=torch.tensor(rows,dtype=torch.float32)
        before_tensor=tensor.clone()
        converted=postprocess_decoded_person("fixture",tensor,geometry=geometry())
        self.assertEqual(converted["final_prediction_count"],2)
        self.assertTrue(torch.equal(tensor,before_tensor))

    def test_maxdet_applies_after_person_nms_and_before_restore(self):
        rows=[[0,0,5,5,1.,79]]
        rows += [[(i%30)*20,(i//30)*20,(i%30)*20+5,(i//30)*20+5,.9-i*.001,0]
                 for i in range(301)]
        result=postprocess_decoded_person("fixture",rows,geometry=geometry())
        self.assertEqual(result["final_prediction_count"],300)
        self.assertEqual(result["raw_predictions"][-1]["filter_reasons"],["max_det"])
        self.assertEqual(result["raw_predictions"][0]["filter_reasons"],["non_person"])

    def test_empty_and_native_same_score_tie_limitation_is_recorded(self):
        result=postprocess_decoded_person("fixture",[],geometry=geometry())
        self.assertEqual(result["predictions"],[])
        self.assertEqual(result["raw_predictions"],[])
        rows=[[0,0,10,10,.5,0],[0,0,10,10,.5,0],[20,20,30,30,.5,0]]
        first=postprocess_decoded_person("fixture",rows,geometry=geometry())
        second=postprocess_decoded_person("fixture",rows,geometry=geometry())
        self.assertEqual(first,second)  # 仅本锁定CPU运行观察，不扩称跨backend保证。
        self.assertEqual(first["final_prediction_count"],2)
        self.assertTrue(first["same_score_candidates_present"])
        self.assertIn("no_cross_backend_tie_guarantee",first["definition"]["tie_policy"])
        kept={p["input_prediction_index"] for p in first["predictions"]}
        self.assertEqual(len(kept & {0,1}),1)
        self.assertIn(2,kept)
        self.assertTrue(all(row["score"]==.5 for row in first["raw_predictions"]))

    def test_actual_letterbox_odd_padding_inverse_clip_and_degenerate_policy(self):
        source=[1,1,6,2]
        item={**bbox_geometry(source,width=7,height=3,format="xyxy_half_open"),"class_id":0}
        sample={"width":7,"height":3,"domain":"R","sample_uid":"odd",
                "class_mapping":{"person":0},"model_head_classes":80,"targets":[item],"ignore_targets":[]}
        prepared=letterbox_bbox_sample(sample,np.zeros((3,7,3),dtype=np.uint8))
        g=prepared["geometry"]
        self.assertEqual(g["padding_ltrb"],(0,274,0,275))
        mapped=prepared["targets"][0]["input_xyxy"]
        result=postprocess_decoded_person("odd",[[*mapped,.9,0]],geometry=g)
        for actual,expected in zip(result["predictions"][0]["bbox"],source):
            self.assertAlmostEqual(actual,expected,places=6)
        ratio=g["scale"][0]
        rows=[[-ratio,274-ratio,2*ratio,274+2*ratio,.9,0],[0,0,10,10,.8,0]]
        result=postprocess_decoded_person("odd",rows,geometry=g)
        for actual,expected in zip(result["predictions"][0]["bbox"],[0,0,2,2]):
            self.assertAlmostEqual(actual,expected,places=6)
        self.assertEqual(result["raw_predictions"][1]["filter_reasons"],["degenerate_after_restore_clip"])
        self.assertEqual(result["final_prediction_count"],1)

    def test_UID_geometry_nonfinite_and_unsupported_input_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"UID"):
            postprocess_decoded_person("wrong",[],geometry=geometry())
        bad=geometry();bad["scale"]=(.9,.9)
        with self.assertRaisesRegex(ValueError,"geometry"):
            postprocess_decoded_person("fixture",[],geometry=bad)
        for rows in ([[0,0,1,1,float("nan"),0]],[[0,0,float("inf"),1,.9,0]],
                     [[0,0,0,1,.9,0]],[[0,0,1,1,.9,.5]],[[0,0,1,1,1.1,0]]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                postprocess_decoded_person("fixture",rows,geometry=geometry())
        for tensor in (torch.zeros((1,6,2)),torch.zeros((1,6),dtype=torch.int64),
                       torch.empty((1,6),device="meta")):
            with self.subTest(tensor=str(tensor.device)),self.assertRaises(ValueError):
                postprocess_decoded_person("fixture",tensor,geometry=geometry())

    def test_candidate_given_boxes_AP_complete_document_exclusive_write_and_cleanup(self):
        result=postprocess_decoded_person("fixture",[[0,0,10,10,.9,0],[0,0,10,10,.7,3]],geometry=geometry())
        processed={"fixture":result}
        gt={"fixture":{"positive_gt":[[0,0,10,10]],"ignore":[]}}
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/"candidate"
            receipt=write_candidate_evaluation(target,processed,gt,
                                                bindings={"origin":"self_made_boxes"},scope="fixture")
            document=json.loads(Path(receipt["path"]).read_text(encoding="utf-8"))
            self.assertEqual(document["metrics"]["AP50_95"],1)
            self.assertTrue(document["candidate_only"])
            self.assertFalse(document["official_benchmark"])
            self.assertFalse(document["model_inference_performed"])
            self.assertEqual(document["definition"],DEFINITION)
            self.assertEqual(len(document["processed_images"]["fixture"]["raw_predictions"]),2)
            self.assertEqual(len(document["processed_images"]["fixture"]["predictions"]),1)
            self.assertEqual(set(p.name for p in target.iterdir()),{"candidate_evaluation.json"})
            before=Path(receipt["path"]).read_bytes()
            with self.assertRaises(FileExistsError):
                write_candidate_evaluation(target,processed,gt,bindings={"origin":"fixture"},scope="fixture")
            self.assertEqual(Path(receipt["path"]).read_bytes(),before)
            with self.assertRaisesRegex(ValueError,"ignore"):
                write_candidate_evaluation(Path(d)/"missing",processed,
                                           {"fixture":{"positive_gt":[]}},bindings={"origin":"fixture"},scope="fixture")
            self.assertFalse((Path(d)/"missing").exists())
            with self.assertRaises(ValueError):
                write_candidate_evaluation(Path(d)/"test",processed,gt,bindings={"split":"test"},scope="test")
        self.assertFalse(target.exists())

    def test_reparse_target_parent_and_unsupported_versions_are_refused(self):
        from futsalmot import postprocess
        processed={"fixture":postprocess_decoded_person("fixture",[],geometry=geometry())}
        gt={"fixture":{"positive_gt":[],"ignore":[]}}
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d)
            original=Path.lstat
            def reparse(path,*args,**kwargs):
                if path==parent:
                    return SimpleNamespace(st_mode=stat.S_IFDIR,st_file_attributes=0x400)
                return original(path,*args,**kwargs)
            with patch.object(Path,"lstat",reparse):
                with self.assertRaisesRegex(ValueError,"reparse"):
                    write_candidate_evaluation(parent/"candidate",processed,gt,bindings={"origin":"fixture"},scope="fixture")
            self.assertFalse((parent/"candidate").exists())
        postprocess._backend.cache_clear()
        try:
            with patch("futsalmot.postprocess.version",return_value="unexpected"):
                with self.assertRaisesRegex(RuntimeError,"锁定"):
                    postprocess_decoded_person("fixture",[],geometry=geometry())
        finally:
            postprocess._backend.cache_clear()


if __name__=="__main__":unittest.main()
