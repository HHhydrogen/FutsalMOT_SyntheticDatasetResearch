"""硬链接apply小夹具；mock原planner仅缩小发布规模，真实apply仍重跑原planner。

所有文件均TemporaryDirectory的自造1帧PNG/JSON，不访问G或生产。不是生成质量验收。
"""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from futsalmot.layout import VERSION,SCHEMAS,_seed_file_plan,_stat_files,_canonical_sha
from futsalmot.layout_apply import begin_synthetic_layout,MIN_FREE_BYTES


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False),encoding="utf8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(base,corrupt=False):
    root=base/"synthetic_research_v1"/VERSION;target=base/"synthetic"
    rows=[];files=[]
    for number,split in ((30022,"train"),(30005,"dev")):
        sid=f"seed_{number:08x}";folder=root/split/sid;version="v3_original_capture" if split=="train" else VERSION
        seed={"schema":"futsalmot_seed_manifest","seed_id":sid,"root_seed":f"{number:08x}","split":split,
              "dataset_id":"futsalmot_synth_multitask","dataset_version":version,"frame_count":1,
              "camera_ids":["C5"],"trajectory_uid":"original_trajectory_UID"}
        seed_sha=put(folder/"seed_manifest.json",seed)
        (folder/"trajectory.jsonl").write_text(json.dumps({"step":0,"time_seconds":0,"players":[]})+"\n")
        put(folder/"person_keypoints_coco.json",{"images":[],"annotations":[],"fixture":True})
        put(folder/"C5/camera.json",{"camera_id":"C5","width":4,"height":3})
        (folder/"C5/annotations.jsonl").write_text(json.dumps({"seed_id":sid,"camera_id":"C5","frame_index":1,
             "pose_schema_id":"futsal_coco17_v1","dataset_version":version,"objects":[]})+"\n")
        for kind in ("img1","mask","number_mask"):
            path=folder/"C5"/kind/"000001.png";path.parent.mkdir(parents=True)
            Image.new("RGB" if kind=="img1" else "L",(4,3),(20,30,40) if kind=="img1" else 0).save(path)
        if corrupt and split=="train":(folder/"C5/img1/000001.png").write_bytes(b"corrupt fixture PNG")
        records,_=_stat_files(folder)
        row={"root_seed":number,"seed_id":sid,"split":split,"directory":f"{split}/{sid}","camera_ids":["C5"],
             "frames_per_camera":1,"counts":{"files":len(records),"bytes":sum(x[1] for x in records),"rgb_frames":1},
             "file_stat_fingerprint":_canonical_sha(records),"seed_manifest_sha256":seed_sha,
             "trajectory_uid":seed["trajectory_uid"],"original_seed_dataset_version":version,
             "coco_export_sha256":"b"*64} # 明确是导出收据SHA，不能拿它比COCO内容。
        rows.append(row);files.extend(_seed_file_plan(root,target,row))
    schemas={name:put(root/"schemas"/name,{"fixture":True}) for name in SCHEMAS}
    msha=put(root/"dataset_manifest.json",{"resolution":[4,3],"schema_sha256":schemas,"fixture":True})
    vsha=put(root/"validation_receipt.json",{"status":"PASS","fixture":True})
    g5=base/"G5_complete.json";gsha=put(g5,{"status":"published","release_path":str(root),"fixture":True})
    binding={"release_root":str(root),"dataset_manifest":str(root/"dataset_manifest.json"),"manifest_sha256":msha,
             "validation_receipt":str(root/"validation_receipt.json"),"validation_sha256":vsha,
             "g5_receipt":str(g5),"g5_sha256":gsha,"fingerprint":"a"*64,"dataset_version":VERSION}
    logical=sum(x["bytes"] for x in files)
    plan={"status":"PLANNED_DRY_RUN","media_enumerated":True,"source_root":str(root),"target_root":str(target),
          "source_binding":binding,"seed_count":2,"files":files,
          "candidate_dataset":{"schema_version":"canonical_synthetic_dataset_v1","status":"PLANNED_NOT_MIGRATED",
                               "source_release":binding,"seeds":rows,"totals":{"files":len(files),"bytes":logical}},
          "relocation":{"linked_file_count":len(files),"logical_bytes":logical,"media_copy_bytes":0}}
    evidence={"assessment":"NO_ACTIVE_OWNED_WRITERS","source_root":str(root),"observed_at_utc":datetime.now(timezone.utc).isoformat(),
              "active_writers":[],"evidence":[{"path":str(g5),"sha256":gsha}],"caller_process_observation":"fixture; no production process queried"}
    return plan,evidence


def gates(plan):
    # 没有allow_fixture之类的生产API绕行开关；仅单测patch旧planner以构造tiny源。
    return patch("futsalmot.layout_apply.plan_synthetic_layout",return_value=deepcopy(plan))


def disk():return patch("futsalmot.layout_apply.shutil.disk_usage",return_value=SimpleNamespace(free=MIN_FREE_BYTES+1024**3))


class LayoutApplyTests(unittest.TestCase):
    def test_begin_pilot_finish_preserve_original_bytes_UID_split_v3_and_source_refs(self):
        with tempfile.TemporaryDirectory() as t:
            plan,proof=fixture(Path(t));root=Path(plan["source_root"]);target=Path(plan["target_root"])
            before={x["relative"]:(Path(x["source"]).read_bytes(),Path(x["source"]).stat().st_ino) for x in plan["files"]}
            with gates(plan) as planner,disk(),patch("shutil.copyfile",side_effect=AssertionError("禁止copy")),patch("os.rename",side_effect=AssertionError("禁止媒体rename")):
                session=begin_synthetic_layout(plan,writer_evidence=proof,expected_plan_sha256=_canonical_sha(plan))
                planner.assert_called_once()
                self.assertTrue(planner.call_args.kwargs["enumerate_files"])
                self.assertEqual(json.loads((target/"dataset.json").read_text())["status"],"BUILDING")
                with patch("PIL.Image.open",wraps=Image.open) as images:pilot=session.pilot()
                self.assertEqual(images.call_count,1)
                self.assertEqual(pilot["verified_files"],8);self.assertEqual(pilot["rgb_images_decoded"],1)
                self.assertEqual(pilot["original_seed_dataset_version"],"v3_original_capture")
                self.assertNotEqual(pilot["root_file_source_sha256"]["person_keypoints_coco.json"],"b"*64)
                self.assertEqual(json.loads((target/"dataset.json").read_text())["status"],"BUILDING")
                self.assertFalse((target/"dev").exists())
                result=session.finish()
            self.assertEqual(result["status"],"READY");self.assertEqual(result["linked_files"],16)
            receipt=json.loads((target/"RELOCATION.json").read_text())
            self.assertEqual(receipt["status"],"PASS");self.assertEqual(receipt["media_copy_bytes"],0)
            self.assertEqual((receipt["created_links"],receipt["verified_links"]),(16,16))
            self.assertEqual(len(receipt["files"]),16)
            index=json.loads((target/"dataset.json").read_text())
            self.assertEqual(index["status"],"READY")
            self.assertEqual(index["source_release"],plan["source_binding"])
            self.assertTrue(all(str(root/"schemas") in v["path"] for v in index["source_schemas"].values()))
            for item in plan["files"]:
                source,dest=Path(item["source"]),Path(item["target"])
                self.assertTrue(os.path.samefile(source,dest))
                self.assertEqual(source.read_bytes(),before[item["relative"]][0])
                self.assertEqual(source.stat().st_ino,before[item["relative"]][1])
            self.assertTrue((root/"dataset_manifest.json").exists())

    def test_requires_complete_matching_plan_explicit_evidence_and_empty_target(self):
        with tempfile.TemporaryDirectory() as t:
            plan,proof=fixture(Path(t));target=Path(plan["target_root"])
            bad=deepcopy(plan);bad["status"]="READY_FOR_ENUMERATION"
            with self.assertRaises(ValueError):begin_synthetic_layout(bad,writer_evidence=proof)
            with gates(plan),disk():
                with self.assertRaises(ValueError):begin_synthetic_layout(plan,writer_evidence=True)
                with self.assertRaises(ValueError):begin_synthetic_layout(plan,writer_evidence={**proof,"active_writers":[123]})
                with self.assertRaises(ValueError):begin_synthetic_layout(plan,writer_evidence=proof,expected_plan_sha256="0"*64)
            self.assertFalse(target.exists())
            changed=deepcopy(plan);changed["files"][0]["bytes"]+=1
            with gates(changed),disk():
                with self.assertRaises(ValueError):begin_synthetic_layout(plan,writer_evidence=proof)
            target.mkdir();(target/"foreign_partial.txt").write_text("keep")
            with self.assertRaises(ValueError):begin_synthetic_layout(plan,writer_evidence=proof)
            self.assertEqual((target/"foreign_partial.txt").read_text(),"keep")

    def test_no_finish_without_real_pilot_and_no_automatic_retry(self):
        with tempfile.TemporaryDirectory() as t:
            plan,proof=fixture(Path(t));target=Path(plan["target_root"])
            with gates(plan),disk():
                session=begin_synthetic_layout(plan,writer_evidence=proof)
                with self.assertRaises(ValueError):session.finish()
                with self.assertRaises(ValueError):session.pilot()
            self.assertEqual(json.loads((target/"RELOCATION.json").read_text())["status"],"FAIL_STOP")
            self.assertNotEqual(json.loads((target/"dataset.json").read_text())["status"],"READY")
            with self.assertRaises(ValueError):begin_synthetic_layout(plan,writer_evidence=proof)

    def test_failed_PNG_pilot_keeps_partial_and_originals_no_READY(self):
        with tempfile.TemporaryDirectory() as t:
            plan,proof=fixture(Path(t),corrupt=True);target=Path(plan["target_root"])
            with gates(plan),disk():
                session=begin_synthetic_layout(plan,writer_evidence=proof)
                with self.assertRaises(Exception):session.pilot()
            receipt=json.loads((target/"RELOCATION.json").read_text())
            self.assertEqual(receipt["status"],"FAIL_STOP");self.assertEqual(receipt["created_links"],8)
            self.assertFalse(receipt["automatic_resume_allowed"])
            self.assertEqual(json.loads((target/"dataset.json").read_text())["status"],"FAIL_STOP")
            self.assertTrue(all(Path(x["source"]).exists() for x in plan["files"]))
            self.assertTrue((target/plan["candidate_dataset"]["seeds"][0]["directory"]).exists())

    def test_link_race_or_unknown_destination_stops_without_fixing_or_deleting(self):
        with tempfile.TemporaryDirectory() as t:
            plan,proof=fixture(Path(t));target=Path(plan["target_root"]);original=os.link
            def race(source,dest,**kwargs):
                original(source,dest,**kwargs)
                if Path(dest).parent.name=="img1":Path(source).write_bytes(Path(source).read_bytes()+b"changed_fixture")
            with gates(plan),disk():
                session=begin_synthetic_layout(plan,writer_evidence=proof)
                with patch("futsalmot.layout_apply.os.link",race):
                    with self.assertRaises(ValueError):session.pilot()
            receipt=json.loads((target/"RELOCATION.json").read_text())
            self.assertEqual(receipt["status"],"FAIL_STOP")
            self.assertGreater(receipt["created_links"],receipt["verified_links"])
            self.assertTrue(all(Path(x["source"]).exists() for x in plan["files"]))
        with tempfile.TemporaryDirectory() as t:
            plan,proof=fixture(Path(t));target=Path(plan["target_root"])
            with gates(plan),disk():
                session=begin_synthetic_layout(plan,writer_evidence=proof)
                image=Path(plan["files"][0]["target"]);image.parent.mkdir(parents=True,exist_ok=True);image.write_bytes(b"unknown")
                with self.assertRaises(ValueError):session.pilot()
            self.assertEqual(image.read_bytes(),b"unknown")

    def test_metadata_or_writer_evidence_changes_refuse_and_budgets_fail_before_target(self):
        with tempfile.TemporaryDirectory() as t:
            plan,proof=fixture(Path(t));target=Path(plan["target_root"])
            with gates(plan),patch("futsalmot.layout_apply.shutil.disk_usage",return_value=SimpleNamespace(free=MIN_FREE_BYTES-1)):
                with self.assertRaisesRegex(ValueError,"100GiB"):begin_synthetic_layout(plan,writer_evidence=proof)
            self.assertFalse(target.exists())
            with gates(plan),disk(),patch("futsalmot.layout_apply.MAX_METADATA_BYTES",16):
                with self.assertRaisesRegex(ValueError,"512MiB"):begin_synthetic_layout(plan,writer_evidence=proof)
            self.assertFalse(target.exists())
            with gates(plan),disk():
                session=begin_synthetic_layout(plan,writer_evidence=proof)
                Path(proof["evidence"][0]["path"]).write_bytes(b"changed_proof")
                with self.assertRaises(ValueError):session.pilot()
            self.assertEqual(json.loads((target/"RELOCATION.json").read_text())["status"],"FAIL_STOP")

    def test_unknown_extra_target_never_becomes_READY_and_reparse_source_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            plan,proof=fixture(Path(t));target=Path(plan["target_root"])
            with gates(plan),disk():
                session=begin_synthetic_layout(plan,writer_evidence=proof);session.pilot()
                (target/"unknown.txt").write_text("preserve")
                with self.assertRaises(ValueError):session.finish()
            self.assertEqual((target/"unknown.txt").read_text(),"preserve")
            self.assertNotEqual(json.loads((target/"dataset.json").read_text())["status"],"READY")
        with tempfile.TemporaryDirectory() as t:
            plan,proof=fixture(Path(t));source=Path(plan["source_root"]);original=Path.lstat
            def reparse(path,*args,**kwargs):
                info=original(path,*args,**kwargs)
                if path==source:return SimpleNamespace(st_mode=info.st_mode,st_file_attributes=0x400)
                return info
            with patch.object(Path,"lstat",reparse):
                with self.assertRaisesRegex(ValueError,"重解析"):begin_synthetic_layout(plan,writer_evidence=proof)
            self.assertFalse(Path(plan["target_root"]).exists())

if __name__=="__main__":unittest.main()
