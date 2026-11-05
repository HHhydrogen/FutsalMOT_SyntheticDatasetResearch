"""纯stdlib事务prefetch检查：不启动worker、不读写数据或训练。"""
from copy import deepcopy
import json
import unittest
from futsalmot.data import CycleSampler, PrefetchSampler


def stream(domain="R",uids=("a","b","c","d","e")):
    return CycleSampler(uids,task="detect",domain=domain,seed=17)


class PrefetchTransactionTests(unittest.TestCase):
    def test_reserve_never_advances_checkpoint_FIFO_commit_matches_old_stream(self):
        baseline=stream();q=PrefetchSampler(stream())
        initial=q.state_dict()
        first=q.reserve(3);second=q.reserve(4)
        self.assertEqual(q.state_dict(),initial)
        self.assertEqual(q.committed_draws,0)
        self.assertEqual(q.pending_batches,2)
        self.assertEqual(first["samples"],baseline.take(3))
        self.assertEqual(second["samples"],baseline.take(4))
        committed_first=stream();committed_first.take(3)
        self.assertEqual(q.commit(first["token"]),first)
        self.assertEqual(q.state_dict()["sampler"],committed_first.state_dict())
        self.assertEqual(q.committed_draws,3)
        q.commit(second["token"])
        self.assertEqual(q.state_dict()["sampler"],baseline.state_dict())
        self.assertEqual(q.pending_batches,0)
        self.assertEqual(q.committed_draws,7)

    def test_bounded_queue_and_strict_count_wrong_order_repeat_unknown_tokens(self):
        q=PrefetchSampler(stream())
        for bad in [0,-1,True,1.5,"2",None]:
            with self.assertRaises(ValueError):q.reserve(bad)
        self.assertEqual(q.pending_batches,0)
        a,b=q.reserve(2),q.reserve(2)
        before=q.state_dict()
        with self.assertRaisesRegex(RuntimeError,"已满"):q.reserve(1)
        for token in [b["token"],"unknown",None,{}]:
            with self.assertRaises(ValueError):q.commit(token)
        self.assertEqual(q.state_dict(),before)
        self.assertEqual(q.pending_batches,2)
        q.commit(a["token"])
        with self.assertRaises(ValueError):q.commit(a["token"])
        q.commit(b["token"])
        for bad in [0,False,1.5]:
            with self.assertRaises(ValueError):PrefetchSampler(stream(),max_pending_batches=bad)

    def test_json_restore_discards_prefetch_and_replays_exact_UID_epoch_seed(self):
        q=PrefetchSampler(stream(),max_pending_batches=3)
        a=q.reserve(7);q.commit(a["token"])
        old=q.reserve(6);later=q.reserve(3)
        state=json.loads(json.dumps(q.state_dict()))
        self.assertEqual(set(state),{"schema_version","sampler"})
        self.assertEqual(state["sampler"]["draws"],7)
        restored=PrefetchSampler(stream(),max_pending_batches=3)
        restored.load_state_dict(state)
        replay=restored.reserve(6)
        self.assertEqual(replay["samples"],old["samples"])
        self.assertNotEqual(replay["token"],old["token"])
        with self.assertRaises(ValueError):restored.commit(old["token"])
        restored.commit(replay["token"])
        self.assertEqual(restored.reserve(3)["samples"],later["samples"])
        # 同实例恢复也必须隔离旧token世代。
        q.load_state_dict(state)
        with self.assertRaises(ValueError):q.commit(old["token"])
        self.assertEqual(q.reserve(6)["samples"],old["samples"])

    def test_cancel_replays_pending_and_old_tokens_remain_invalid(self):
        q=PrefetchSampler(stream())
        a=q.reserve(2);q.commit(a["token"])
        b,c=q.reserve(6),q.reserve(3)
        committed=q.state_dict()
        self.assertEqual(q.cancel_pending(),2)
        self.assertEqual(q.state_dict(),committed)
        self.assertEqual(q.pending_batches,0)
        replay=q.reserve(6)
        self.assertEqual(replay["samples"],b["samples"])
        self.assertNotEqual(replay["token"],b["token"])
        for token in [b["token"],c["token"]]:
            with self.assertRaises(ValueError):q.commit(token)
        q.commit(replay["token"])
        self.assertEqual(q.reserve(3)["samples"],c["samples"])

    def test_worker_and_checkpoint_mutation_cannot_pollute_internal_state(self):
        q=PrefetchSampler(stream());baseline=stream()
        request=q.reserve(3);expected=baseline.take(3);token=request["token"]
        request["samples"][0]["uid"]="worker_corruption"
        request["samples"].clear();request["count"]=999
        result=q.commit(token)
        self.assertEqual(result["samples"],expected)
        self.assertEqual(result["count"],3)
        result["samples"][0]["augmentation_seed"]=-1
        detached=q.state_dict();detached["sampler"]["order"].reverse()
        self.assertEqual(q.reserve(4)["samples"],baseline.take(4))

    def test_bad_state_wrong_pool_no_partial_mutation_or_pending_loss(self):
        q=PrefetchSampler(stream())
        request=q.reserve(2);before=q.state_dict()
        cases=[]
        bad=deepcopy(before);bad["sampler"]["pool_sha256"]="wrong_pool";cases.append(bad)
        bad=deepcopy(before);bad["sampler"]["rng_state"]=[999,[],None];cases.append(bad)
        bad=deepcopy(before);bad["sampler"]["draws"]=1;cases.append(bad)
        bad=deepcopy(before);bad["sampler"].pop("order");cases.append(bad)
        bad=deepcopy(before);bad["sampler"]["position"]=True;cases.append(bad)
        cases.extend([{}, {**before,"pending":"must_not_restore"}])
        for bad in cases:
            with self.assertRaises(ValueError):q.load_state_dict(bad)
            self.assertEqual(q.state_dict(),before)
            self.assertEqual(q.pending_batches,1)
        q.commit(request["token"])
        self.assertEqual(q.committed_draws,2)
        # 原API也保持恢复失败原子性。
        original=stream();original.take(4);saved=original.state_dict()
        bad=deepcopy(saved);bad["rng_state"]=[3,[],"changed_gauss"]
        with self.assertRaises(ValueError):original.load_state_dict(bad)
        self.assertEqual(original.state_dict(),saved)

    def test_independent_domains_and_R_extra_is_real_sampling_namespace(self):
        r,s,extra=[PrefetchSampler(stream(d)) for d in ("R","S","R_extra")]
        s.reserve(5);extra_req=extra.reserve(4)
        self.assertEqual(r.reserve(4)["samples"],stream("R").take(4))
        self.assertEqual(extra_req["samples"],stream("R_extra").take(4))
        self.assertEqual(extra_req["stream"],"R_extra")
        self.assertEqual(extra_req["data_domain"],"R")
        self.assertNotEqual(extra_req["samples"][0]["augmentation_seed"],stream("R").take(1)[0]["augmentation_seed"])
        self.assertEqual(r.committed_draws,s.committed_draws)
        self.assertEqual(r.committed_draws,0)

if __name__=="__main__":unittest.main()
