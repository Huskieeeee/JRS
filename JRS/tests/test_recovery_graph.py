import copy,sys,unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from floor_tase.model import ScenarioConfig,make_scenario
from floor_tase.environment import RecoveryEnv,audit_trace
from floor_tase.recovery_graph import GraphOptions,GraphCompiler,BatchedCompletion,RecoveryGraphPlanner,templates
from floor_tase.rg_variants import ALL_METHODS,RG_METHODS,make_rg
from prepare_v16 import prepare
from run_repair_round import jobs_for,scenario_for,planner_for

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.scenario=make_scenario(16090001,ScenarioConfig(repair_failure_prob=.15,assumed_repair_failure_prob=.15))
        self.env=RecoveryEnv(self.scenario);self.obs=self.env.observe();self.opt=GraphOptions(samples=16)
        self.plans=[GraphCompiler(self.obs,self.opt).compile(t) for t in templates(self.obs,self.opt,4)]
    def test_topology_and_typed_nodes(self):
        kinds=set()
        for plan in self.plans:
            for i,n in enumerate(plan['nodes']):
                kinds.add(n['kind']);self.assertTrue(all(d<i for d in n['deps']))
                self.assertTrue(set(n['quality_deps'])<=set(n['deps']))
        self.assertTrue({'repair','inspect','delivery','successor','move'}<=kinds)
    def test_common_random_candidate_permutation(self):
        ev=BatchedCompletion();a=ev.evaluate(self.plans,32,77);b=ev.evaluate(self.plans[::-1],32,77)
        self.assertEqual(a,b[::-1])
    def test_zero_failure_cannot_make_same_graph_slower(self):
        clean=copy.deepcopy(self.plans)
        for plan in clean:
            for n in plan['nodes']:n['repair_prob']=0.;n['risk']=[0.]*len(n['risk'])
        a=BatchedCompletion().evaluate(clean,64,21);b=BatchedCompletion().evaluate(self.plans,64,21)
        for x,y in zip(a,b):self.assertLessEqual(x['finish_p90'],y['finish_p90']+1e-5)
    def test_certain_failure_reduces_qualified_work(self):
        opt=replace(self.opt,stochastic=False)
        clean=[GraphCompiler(self.obs,opt).compile(t) for t in templates(self.obs,opt,4)]
        bad=copy.deepcopy(clean)
        for p in bad:
            for n in p['nodes']:
                if n['kind']=='repair':n['repair_prob']=1.
        ev=BatchedCompletion();a=ev.evaluate(clean,32,1);b=ev.evaluate(bad,32,1)
        self.assertTrue(any(x['qualified_fraction']>y['qualified_fraction'] for x,y in zip(a,b)))
    def test_delivery_delay_is_in_graph(self):
        slow=copy.deepcopy(self.plans)
        for p in slow:
            for n in p['nodes']:
                if n['kind']=='delivery':n['duration']+=10
        a=BatchedCompletion().evaluate(self.plans,16,1);b=BatchedCompletion().evaluate(slow,16,1)
        self.assertTrue(any(y['finish_p90']>x['finish_p90'] for x,y in zip(a,b)))
    def test_no_batch_and_factorial(self):
        for groups,scope in templates(self.obs,replace(self.opt,batch=False),1):self.assertTrue(all(len(g)==1 for g in groups))
        self.assertEqual(len(ALL_METHODS),22);self.assertEqual(len(set(ALL_METHODS)),22)
        for j in (0,1):
            for u in (0,1):
                for e in (0,1):
                    o=make_rg(f'jrs_rg_f{j}{u}{e}',1,{}).options
                    self.assertEqual((o.joint,o.stochastic,o.continuation),(bool(j),bool(u),'adaptive' if e else 'none'))
    def test_all_variants_execute_and_audit(self):
        for method in RG_METHODS:
            with self.subTest(method=method):
                cfg=replace(self.scenario.config,evidence_policy='whole_reset') if method=='jrs_rg_no_reuse' else self.scenario.config
                env=RecoveryEnv(replace(self.scenario,config=cfg));planner=make_rg(method,8,{'rg_samples':8,'rg_candidates':4})
                for step in range(1000):
                    if env.done:break
                    acts=planner.select(env.observe(),env.allowed_actions());env.step_joint(acts) if acts else env.advance()
                self.assertTrue(env.done);self.assertTrue(audit_trace(env.get_trace())['pass'])
                self.assertEqual(planner.diagnostics()['baseline_policy_calls'],0)
    def test_repeatable_actions(self):
        a=RecoveryGraphPlanner(9,self.opt);b=RecoveryGraphPlanner(9,self.opt)
        self.assertEqual(a.select(self.obs,self.env.allowed_actions()),b.select(self.obs,self.env.allowed_actions()))
    def test_shards_preserve_job_identity(self):
        c=prepare('probe','cpu');whole=list(jobs_for(c));parts=[]
        for i in range(3):parts+=list(jobs_for(prepare('probe','cpu',i,3)))
        self.assertEqual({j['job_id']:j for j in whole},{j['job_id']:j for j in parts})
        self.assertEqual(len(parts),len(whole))
    def test_reset_policy_preserves_physical_world(self):
        c=prepare('smoke','cpu');jobs=list(jobs_for(c));ref=jobs[0];a,pa=scenario_for(ref,c)
        b,pb=scenario_for(dict(ref,method='jrs_rg_no_reuse'),c)
        self.assertEqual(pa['base_world_sha256'],pb['base_world_sha256']);self.assertEqual(a.initial_quality,b.initial_quality)
        self.assertEqual(b.config.evidence_policy,'whole_reset')
    def test_all_planner_adapters_construct(self):
        c=prepare('smoke','cpu');jobs=list(jobs_for(c))[:22]
        self.assertEqual(len(jobs),22)
        for j in jobs:
            with self.subTest(method=j['method']):self.assertTrue(callable(planner_for(j).select))

if __name__=='__main__':unittest.main()
