from dataclasses import replace
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test
from conwip.pair_v2 import Conditions, PairV2Config, PairV2Env, CentralizedPairV2, BASE_MEANS


def env(**kwargs):
    e=PairV2Env(PairV2Config(schedule=(),warmup_minutes=0,horizon_minutes=120,**kwargs));e.reset(seed=42);return e


def test_serial_pipeline_wip_changes_throughput():
    c=Conditions(means=((1.,)*5,(.01,)*5,(.01,)*3),distribution=('deterministic',)*3)
    rates=[]
    for cap in (1,5):
        e=PairV2Env(PairV2Config(conditions=c,schedule=(),initial_caps=(cap,8,8),warmup_minutes=60,horizon_minutes=120))
        e.reset(seed=42);e._advance(180);e.assert_invariants();rates.append(e.metrics()['kits_completed']/2)
    assert rates==pytest.approx([12,60],abs=.5)


def test_assembly_three_serial_stages_two_parallel_servers():
    c=Conditions(means=((.01,)*5,(.01,)*5,(1.,)*3),distribution=('deterministic',)*3)
    rates=[]
    for cap in (1,6):
        e=PairV2Env(PairV2Config(conditions=c,schedule=(),initial_caps=(8,8,cap),warmup_minutes=60,horizon_minutes=120))
        e.reset(seed=42);e._advance(180);rates.append(e.metrics()['kits_completed']/2)
        assert all(x[1]>=3 for x in e.finished)  # every kit visits all 3 stages
        e.assert_invariants()
    assert rates==pytest.approx([20,120],abs=.5)


def test_lowering_caps_preserves_jobs_and_cap_zero_drains_only_existing_kits():
    e=env();e._advance(80);before=(e.uid.copy(),e.kit_uid,e.total_wip(),len(e.finished))
    e.set_caps([0,0,0]);assert before==(e.uid,e.kit_uid,e.total_wip(),len(e.finished))
    e._advance(400);e.assert_invariants()
    assert e.uid==before[0] and e.kit_uid==before[1] and e.wip[2]==0


def test_running_operation_duration_unchanged_new_stage_uses_new_conditions():
    c=Conditions(means=((1.,)*5,(1.,)*5,(1.,)*3),distribution=('deterministic',)*3)
    e=env(conditions=c,initial_caps=(1,1,1));job=e.active[0][0][0];original=job['service'][0]
    means=((10.,)*5,(2.,)*5,(3.,)*3)
    e.set_conditions(means=means,gamma=(2.,2.,2.))
    assert job['service'][0]==original
    e._advance(1.)
    assert job['service'][1][1:3]==(10.,10.)


def test_scheduled_conditions_before_operations_start_at_exact_boundary():
    c=Conditions(means=((1.,)*5,(1.,)*5,(1.,)*3),distribution=('deterministic',)*3)
    cfg=PairV2Config(conditions=c,warmup_minutes=0,schedule=((1.,{'means':((10.,)*5,(1.,)*5,(1.,)*3)}),))
    e=PairV2Env(cfg);e.reset(seed=4);e._advance(1)
    assert e.active[0][1][0]['service'][1][1]==10


def test_target_window_uses_historical_targets_and_fractional_schedule_time():
    e=PairV2Env(PairV2Config(schedule=((60.,{'target':6}),),warmup_minutes=120,horizon_minutes=120,window_minutes=60))
    e.reset(seed=1);e._advance(200)
    assert e.rolling_metrics()[1]==pytest.approx(4.)
    e.schedule_conditions(80.5,target=9);e._advance(200.5)
    assert e.current.target==9 and e.target_history[-1][0]==200.5
    e.set_conditions(target=4,keep_future_schedule=False)
    assert e.current.target==4


@pytest.mark.parametrize('patch',[{'target':3.5},{'target':float('nan')},{'gamma':[0,1,1]}, {'means':[[1],[1],[1]]},{'distribution':['unknown']*3}])
def test_invalid_live_conditions_atomic(patch):
    e=env();before=(e.current,list(e.schedule),list(e.condition_changes))
    with pytest.raises(ValueError):e.set_conditions(keep_future_schedule=False,**patch)
    assert (e.current,e.schedule,e.condition_changes)==before


@pytest.mark.parametrize('caps',[[1.5,2,3],[True,2,3],[-1,2,3],[9,2,3]])
def test_invalid_caps(caps):
    e=env()
    with pytest.raises(ValueError):e.set_caps(caps)


def test_replay_and_crn_independent_of_parallel_server_and_caps():
    a=env(initial_caps=(2,2,2));b=env(initial_caps=(8,8,8))
    for i in range(3):
        for j in range(len(BASE_MEANS[i])):
            assert a._duration(i,j,15)==b._duration(i,j,15)
    rows=[]
    for _ in range(2):
        e=env()
        while e.agents:e.step({a:1 for a in e.agents})
        rows.append((e.finished,e.metrics()))
    assert rows[0]==rows[1]


def test_distributions_and_observation_conditions():
    e=env();old=e._observations()['cell_0']
    e.set_conditions(gamma=(2.,3.,4.),distribution=('uniform','deterministic','gamma'))
    assert not np.array_equal(old,e._observations()['cell_0'])
    assert 3.5<=e._duration(0,0,0)<=10.5
    assert e._duration(1,0,0)==8
    for a,o in e._observations().items():assert e.observation_space(a).contains(o)


def test_common_demand_gate_and_statistics():
    e=env(conditions=Conditions(target=6),demand_gated=True)
    while e.agents:e.step({a:1 for a in e.agents})
    assert e.demand_integral==pytest.approx(12) and max(e.uid)<=12
    m=e.metrics()
    assert 0<=m['kits_completed']<=12
    assert all(b+k+i==pytest.approx(1) for b,k,i in zip(m['busy_fraction'],m['blocked_fraction'],m['idle_fraction']))
    assert m['mean_wip_component_equiv']<=m['peak_wip_component_equiv']<=38


def test_parallel_and_centralized_interfaces():
    cfg=PairV2Config(schedule=(),warmup_minutes=10,horizon_minutes=50)
    parallel_api_test(PairV2Env(cfg),num_cycles=10)
    check_env(CentralizedPairV2(cfg),skip_render_check=True)


def test_scan_evaluator_matches_environment_step_reward():
    from run_pair_v2 import evaluate,profiles
    caps=(4,6,5);seed=101
    r=evaluate(('feed_variance',caps,seed,60.,20.,False))
    e=PairV2Env(PairV2Config(conditions=profiles()['feed_variance'],schedule=(),initial_caps=caps,warmup_minutes=20,horizon_minutes=60))
    e.reset(seed=seed);costs=[]
    while e.agents:
        _,rew,_,_,_=e.step({a:1 for a in e.agents});costs.append(-rew['cell_0'])
    assert r['mean_cost']==pytest.approx(np.mean(costs))
    assert r['mean_wip']==pytest.approx(e.metrics()['mean_wip_component_equiv'])
    assert r['throughput']==e.metrics()['kits_completed']
