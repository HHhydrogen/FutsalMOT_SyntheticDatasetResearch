"""只读BBox小夹具；不需要真实媒体、Torch或GPU，临时文件自动删除。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from futsalmot.adapters import (
    _POLICIES, bbox_geometry, read_real_bbox_sample, read_synthetic_bbox_sample,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def real_fixture(root, source="soccernet_tracking", split="train"):
    target, ignore = _POLICIES[source]
    sequence = dict(source_id=source, clip_uid="source:clip", match_uid="source:match",
                    split=split, complete=True, selected=True, official_frame_count=3,
                    detection_frames=[1,3], width=100, height=50, sequence_path="sequences/clip",
                    files={"gt":"gt/gt.txt", "gameinfo":"gameinfo.ini"},
                    frame_uid_pattern="source:clip:{frame_number:06d}",
                    detection_sample_uid_pattern="detect:{frame_uid}",
                    image_file_pattern="img1/{frame_number:06d}.jpg", track_namespace="clip:original_ID",
                    target_policy=target, ignore_policy=ignore)
    d = dict(schema_version="canonical_real_dataset_v1", status="READY_SOURCE_LOCAL",
             source_panel_id=source, source={"source_id":source, "target_semantics":target, "ignore_policy":ignore},
             matches=[{"match_uid":"source:match","split":split}], sequences=[sequence],
             limitations=["fixture_source_limit"], joint_use_allowed=False, cross_source_disjoint_certified=False)
    write_json(root/"dataset.json", d)
    gt = root/sequence["sequence_path"]/"gt/gt.txt"
    gt.parent.mkdir(parents=True, exist_ok=True)
    if source == "soccernet_tracking":
        # 按track排列：第二帧先出现，不能提前停扫；ball/court referee/unknown保留ignore。
        gt.write_text("3,1,20,10,5,5,1,-1,-1,-1\n1,1,-5,5,20,20,1,-1,-1,-1\n"
                      "1,2,20,10,10,10,1,-1,-1,-1\n1,3,30,10,10,10,1,-1,-1,-1\n"
                      "1,4,40,10,10,10,1,-1,-1,-1\n1,5,50,10,10,10,1,-1,-1,-1\n"
                      "1,6,60,10,10,10,0,-1,-1,-1\n", encoding="utf-8")
        (gt.parent.parent/"gameinfo.ini").write_text("[Sequence]\ntrackletID_1= player team left;4\n"
            "trackletID_2= goalkeepers team right;A\ntrackletID_3= referee;main\n"
            "trackletID_4= ball;1\ntrackletID_6= player team left;8\n", encoding="utf-8")
    else:
        gt.write_text("1,0,0,10,10,20,1,1,1\n1,1,95,0,10,10,1,1,1\n3,0,1,10,10,20,1,1,1\n",encoding="utf-8")
    return root/"dataset.json", d, gt


def synthetic_fixture(root):
    entities = [dict(entity_id=f"{team}{i}", track_id=n+1, team=team,
                     role="goalkeeper" if i==0 else "player", jersey_number=n+10,
                     instance_mask_id=n+1, number_mask_id=n+32)
                for n,(team,i) in enumerate((team,i) for team in "LR" for i in range(5))]
    m = dict(schema="futsalmot_seed_manifest", schema_version="1.0.0", dataset_id="fixture_synth",
             dataset_version="v3_source", seed_id="seed_00007546", split="train", frame_count=1,
             camera_ids=["C5"], entities=entities, trajectory_uid="original_trajectory_uid")
    write_json(root/"seed_manifest.json", m)
    camera = dict(schema="futsalmot_camera_metadata",schema_version="1.0.0",camera_id="C5",width=100,height=50)
    write_json(root/"C5/camera.json",camera)
    objects = [{**e,"class":"player","bbox_xyxy":[10,20,20,40],"bbox_source":"instance_mask",
                "visible_pixel_count":100,"visibility":"visible"} for e in entities]
    objects[-1].update(bbox_xyxy=None,bbox_source="not_visible",visible_pixel_count=0,visibility="out_of_frame")
    r = dict(schema="futsalmot_frame_annotation",schema_version="1.0.0",dataset_id=m["dataset_id"],
             dataset_version=m["dataset_version"],seed_id=m["seed_id"],camera_id="C5",frame_index=1,
             source_step=788,pose_schema_id="futsal_coco17_v1",camera_state={"width":100,"height":50},objects=objects)
    (root/"C5/annotations.jsonl").write_text(json.dumps(r)+"\n",encoding="utf-8")
    return m,r


class BBoxAdapterTests(unittest.TestCase):
    def test_geometry_native_coordinates_and_half_open(self):
        original = [-5,5,20,20]
        g = bbox_geometry(original,width=100,height=50,format="xywh")
        self.assertEqual(original,[-5,5,20,20])
        self.assertEqual(g["source_xyxy"],[-5,5,15,25])
        self.assertEqual(g["clipped_xyxy"],[0,5,15,25])
        self.assertEqual(g["normalized_cxcywh"],[.075,.3,.15,.4])
        self.assertEqual(bbox_geometry([0,0,1,1],width=100,height=50,format="xyxy_half_open")["normalized_cxcywh"],[.005,.01,.01,.02])
        self.assertFalse(bbox_geometry([110,0,10,10],width=100,height=50,format="xywh")["bbox_supervision_valid"])
        for box in [[0,0,0,1],[0,0,float("nan"),1],[False,0,1,1]]:
            with self.assertRaises(ValueError):bbox_geometry(box,width=100,height=50,format="xywh")

    def test_soccernet_roles_ignore_unknown_and_native_labels(self):
        with tempfile.TemporaryDirectory() as t:
            p,d,gt=real_fixture(Path(t))
            before=gt.read_bytes()
            result=read_real_bbox_sample(p,"detect:source:clip:000001")
            self.assertEqual([o["track_id"] for o in result["targets"]],[1,2])
            self.assertEqual([o["track_id"] for o in result["ignore_targets"]],[3,4,5,6])
            self.assertEqual([o["ignore_reason"] for o in result["ignore_targets"]],["non_target_role","non_target_role","unknown_role","invalid_gt_flag"])
            self.assertEqual(result["yolo_labels"],[[0,.075,.3,.15,.4],[0,.25,.3,.1,.2]])
            self.assertEqual(result["model_head_classes"],80)
            self.assertTrue(result["supervision"]["requires_ignore_aware_loss"])
            self.assertFalse(result["supervision"]["ignore_loss_implemented"])
            self.assertFalse(result["supervision"]["native_yolo_training_ready"])
            self.assertEqual(before,gt.read_bytes())
            self.assertFalse(Path(result["image_path"]).exists()) # 图片不存在仍可适配，证明未读媒体。

    def test_sportsmot_id_zero_val_and_missing_ignore_geometry(self):
        with tempfile.TemporaryDirectory() as t:
            p,_,_=real_fixture(Path(t),"sportsmot_football",split="val")
            result=read_real_bbox_sample(p,"detect:source:clip:000001",split="val")
            self.assertEqual(result["targets"][0]["track_id"],0)
            self.assertEqual(result["ignore_targets"],[])
            self.assertEqual(result["yolo_labels"][1],[0,.975,.1,.05,.2])
            self.assertIn("unannotated_non_target_people_have_no_ignore_geometry",result["source_limitations"])
            self.assertFalse(result["supervision"]["native_yolo_training_ready"])

    def test_real_split_policy_and_path_guards_before_GT(self):
        with tempfile.TemporaryDirectory() as t:
            p,d,gt=real_fixture(Path(t),split="test")
            gt.unlink()
            with self.assertRaisesRegex(ValueError,"train/val"):read_real_bbox_sample(p,"anything",split="test")
            with self.assertRaisesRegex(ValueError,"UID"):read_real_bbox_sample(p,"detect:source:clip:000001")
            d["matches"][0]["split"]=d["sequences"][0]["split"]="train"
            d["sequences"][0]["sequence_path"]="../outside"
            write_json(p,d)
            with self.assertRaisesRegex(ValueError,"越出"):read_real_bbox_sample(p,"detect:source:clip:000001")
            d["source"]["ignore_policy"]="arbitrary_changed_policy"
            write_json(p,d)
            with self.assertRaisesRegex(ValueError,"协议变化"):read_real_bbox_sample(p,"detect:source:clip:000001")

    def test_synthetic_original_version_identity_and_absent_instance(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);m,row=synthetic_fixture(root)
            before=(root/"C5/annotations.jsonl").read_bytes()
            result=read_synthetic_bbox_sample(root,"C5",1)
            self.assertEqual(len(result["targets"]),9)
            self.assertEqual(result["absent_targets"][0]["entity_id"],"R4")
            self.assertEqual(result["yolo_labels"][0],[0,.15,.6,.1,.4])
            self.assertEqual(result["dataset_version"],"v3_source")
            self.assertIn(":v3_source:seed_00007546:C5:000001",result["derived_sample_uid"])
            self.assertEqual(result["source_step"],788)
            self.assertFalse(result["release_validation_performed"])
            self.assertEqual(before,(root/"C5/annotations.jsonl").read_bytes())
            self.assertFalse(Path(result["image_path"]).exists())

    def test_synthetic_rejects_cross_split_and_inconsistent_fields(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);_,row=synthetic_fixture(root)
            with self.assertRaisesRegex(ValueError,"split"):read_synthetic_bbox_sample(root,"C5",1,split="dev")
            with self.assertRaisesRegex(ValueError,"机位或帧"):read_synthetic_bbox_sample(root,"P01",1)
            with self.assertRaisesRegex(ValueError,"机位或帧"):read_synthetic_bbox_sample(root,"C5",2)
            path=root/"C5/annotations.jsonl"
            mutations = [lambda r:r.update(dataset_version="v4_fabricated"),
                         lambda r:r["objects"][0].update(track_id=99),
                         lambda r:r["objects"][-1].update(visible_pixel_count=1),
                         lambda r:r["objects"][0].update(bbox_xyxy=[-1,20,20,40])]
            for change in mutations:
                bad=copy.deepcopy(row);change(bad)
                path.write_text(json.dumps(bad)+"\n",encoding="utf-8")
                with self.assertRaises(ValueError):read_synthetic_bbox_sample(root,"C5",1)

if __name__ == "__main__":unittest.main()
