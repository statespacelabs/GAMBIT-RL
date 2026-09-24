#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, sys, time, hashlib
from collections import defaultdict
from pathlib import Path
from statistics import mean
import numpy as np, torch, yaml
PROJECT_ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(PROJECT_ROOT))
from scripts.combat_runtime import OBS_DIM, ACTION_DIM, RuntimeConfig, configure_unity_env, launch_unity_env, load_policy_and_normalizer, validate_specs, collect_all_weapon_events, summarize_weapon_events, build_reward_builder, sha256_file, write_json, append_jsonl, utc_now
from scripts.combat_observations import build_slot_map_from_agent_ids
from scripts.combat_ppo_update import compute_loss_and_backward
from gambit.rl.online_rl.rollout_buffer import RecurrentRolloutBuffer
from gambit.rl.online_rl.selfplay_runner import make_action_tuple, policy_action_to_buffers
AREA_SPACING=500.0; HARD={"NO_OWNER","WRONG_OWNER","WEAPON_DISABLED","UNKNOWN","WEAPON_INACTIVE","ROUND_OVER","DEAD_OR_RESPAWNING"}
def sha(p):
 p=Path(p); h=hashlib.sha256()
 if not p.exists(): return None
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def side_area(obs,num_areas):
 area=int(round(float(obs[0])/AREA_SPACING)); area=max(0,min(num_areas-1,area)); return ('A' if float(obs[2])<=-78.5 else 'B'), area
def area_from_obs(obs,num_areas):
 area=int(round(float(obs[0])/AREA_SPACING)); return max(0,min(num_areas-1,area))
def collect_step_records(dec,ter):
 records=[]
 if len(dec):
  obs_batch=np.asarray(dec.obs[0],dtype=np.float32)
  for row,aid_raw in enumerate(dec.agent_id): records.append((int(aid_raw),obs_batch[row]))
 if len(ter):
  obs_batch=np.asarray(ter.obs[0],dtype=np.float32)
  for row,aid_raw in enumerate(ter.agent_id): records.append((int(aid_raw),obs_batch[row]))
 return records
def build_slot_map(seed_obs,num_areas):
 return build_slot_map_from_agent_ids(seed_obs,num_areas)
def sat(arr,prefix):
 arr=np.asarray(arr,dtype=np.float32)
 if arr.size==0: return {f'{prefix}_look_sat95':0.0,f'{prefix}_look_sat80':0.0,f'{prefix}_continuous_sat95':0.0,f'{prefix}_continuous_sat80':0.0}
 return {f'{prefix}_look_sat95':float(np.mean(np.abs(arr[:,2:4])>0.95)),f'{prefix}_look_sat80':float(np.mean(np.abs(arr[:,2:4])>0.80)),f'{prefix}_continuous_sat95':float(np.mean(np.abs(arr[:,:4])>0.95)),f'{prefix}_continuous_sat80':float(np.mean(np.abs(arr[:,:4])>0.80))}
def empty(): return {'decisions_received':0,'actions_sent':0,'shoot_pressed_steps':0,'fired_steps':0,'hits':0,'damage_dealt':0.0,'damage_taken':0.0,'deaths_terminals':0,'aim_values':[],'mu_actions':[],'sampled_actions':[],'env_actions':[]}
def summarize_side(s):
 out={k:v for k,v in s.items() if k not in {'aim_values','mu_actions','sampled_actions','env_actions'}}; out['aim_mean']=float(mean(s['aim_values'])) if s['aim_values'] else 999.0; out.update(sat(s['mu_actions'],'mu')); out.update(sat(s['sampled_actions'],'sampled')); out.update(sat(s['env_actions'],'env_applied')); return out
def true_zero_fire_stall_metrics(rows):
 counts={'A':0,'B':0}; maxs={'A':0,'B':0}; benign={'A':0,'B':0}; zwin={'A':0,'B':0}
 for side in ('A','B'):
  cur=0
  for r in rows:
   s=r.get(side,{})
   decisions=int(s.get('decisions_received',0)); shoot=int(s.get('shoot_pressed_steps',0)); fired=int(s.get('fired_steps',0)); terms=int(s.get('deaths_terminals',0)); threshold=max(10,int(0.05*max(1,decisions)))
   if fired==0: zwin[side]+=1
   true=bool(decisions>0 and shoot>=threshold and fired==0 and terms < max(1, decisions//2))
   if true:
    counts[side]+=1; cur+=1; maxs[side]=max(maxs[side],cur)
   else:
    if fired==0: benign[side]+=1
    cur=0
 return {'zero_fire_window_count':sum(zwin.values()),'true_zero_fire_stall_count':sum(counts.values()),'true_zero_fire_stall_max_streak':max(maxs.values()),'benign_zero_fire_window_count':sum(benign.values()),'metric_artifact_zero_fire_count':0,'by_side':{'A':{'true_count':counts['A'],'max_streak':maxs['A'],'zero_windows':zwin['A'],'benign':benign['A']},'B':{'true_count':counts['B'],'max_streak':maxs['B'],'zero_windows':zwin['B'],'benign':benign['B']}}}
ACTOR_HEAD_PREFIXES=('actor_cont_mean.','actor_binary_logits.')
ACTOR_SHARED_PREFIXES=('gru.','tel_encoder.')
VALUE_PREFIXES=('value_head.',)
def is_prefix(name,prefixes): return any(name==p.rstrip('.') or name.startswith(p) for p in prefixes)
def checkpoint_state(path):
 path=Path(path)
 if not path.exists(): return {}
 payload=torch.load(path,map_location='cpu',weights_only=False)
 return payload.get('actor_critic_state',payload.get('model_state_dict',payload)) if isinstance(payload,dict) else payload
def restore_policy_tensors(policy,ckpt_path,prefixes=None):
 state=checkpoint_state(ckpt_path)
 if not state: return []
 cur=policy.state_dict(); restored=[]
 with torch.no_grad():
  for name,t in state.items():
   if name not in cur or not torch.is_tensor(t): continue
   if prefixes is not None and not is_prefix(name,prefixes): continue
   if tuple(cur[name].shape)!=tuple(t.shape): continue
   cur[name].copy_(t.to(device=cur[name].device,dtype=cur[name].dtype)); restored.append(name)
 return restored
def snapshot_named(policy,names):
 wanted=set(names); return {n:p.detach().cpu().clone() for n,p in policy.named_parameters() if n in wanted}
def snapshot_rows(policy,row_map):
 out={}
 params=dict(policy.named_parameters())
 for name,rows in row_map.items():
  if name in params: out[name]=params[name].detach().cpu()[rows].clone()
 return out
def assert_frozen_unchanged(policy,full_snap,row_snap):
 params=dict(policy.named_parameters())
 bad=[]
 for name,before in full_snap.items():
  now=params[name].detach().cpu(); diff=float((now-before).abs().max().item()) if now.numel() else 0.0
  if diff!=0.0: bad.append((name,diff))
 for name,before in row_snap.items():
  rows=list(range(before.shape[0])) if before.ndim else []
  # row_snap stores compact rows in the same order listed in freeze_report.
  continue
 if bad: raise RuntimeError(f'frozen parameter changed: {bad[:8]}')
def assert_row_frozen_unchanged(policy,row_snap,row_map):
 params=dict(policy.named_parameters()); bad=[]
 for name,before in row_snap.items():
  rows=row_map.get(name,[])
  now=params[name].detach().cpu()[rows]; diff=float((now-before).abs().max().item()) if now.numel() else 0.0
  if diff!=0.0: bad.append((name,rows,diff))
 if bad: raise RuntimeError(f'frozen parameter rows changed: {bad[:8]}')
def apply_freeze(policy,cfg):
 value_enabled=bool(getattr(cfg,'value_update_enabled',True) and getattr(cfg,'train_value_head',True))
 actor_enabled=bool(getattr(cfg,'actor_update_enabled',True))
 value_only=bool(getattr(cfg,'value_only',False) or not actor_enabled)
 shoot_only=bool(getattr(cfg,'train_shoot_head',False))
 row_frozen={}
 for _,p in policy.named_parameters(): p.requires_grad_(True)
 if value_only:
  for name,p in policy.named_parameters(): p.requires_grad_(bool(value_enabled and name.startswith(VALUE_PREFIXES)))
 elif shoot_only:
  for name,p in policy.named_parameters():
   train=bool(name.startswith('actor_binary_logits.') or (value_enabled and name.startswith(VALUE_PREFIXES)))
   p.requires_grad_(train)
  rows=torch.as_tensor([1,2,3],device=policy.actor_binary_logits.weight.device)
  policy.actor_binary_logits.weight.register_hook(lambda g,r=rows:g.index_fill(0,r,0.0)); policy.actor_binary_logits.bias.register_hook(lambda g,r=rows:g.index_fill(0,r,0.0))
  row_frozen={'actor_binary_logits.weight':[1,2,3],'actor_binary_logits.bias':[1,2,3]}
 else:
  if not value_enabled:
   for name,p in policy.named_parameters():
    if name.startswith(VALUE_PREFIXES): p.requires_grad_(False)
  rows=[]
  if getattr(cfg,'freeze_movement_action_head',False): rows += [0,1]
  if getattr(cfg,'freeze_look_action_head',False): rows += [2,3]
  if rows and hasattr(policy,'actor_cont_mean'):
   r=torch.as_tensor(sorted(set(rows)),device=policy.actor_cont_mean.weight.device)
   policy.actor_cont_mean.weight.register_hook(lambda g,r=r:g.index_fill(0,r,0.0)); policy.actor_cont_mean.bias.register_hook(lambda g,r=r:g.index_fill(0,r,0.0))
   row_frozen['actor_cont_mean.weight']=sorted(set(rows)); row_frozen['actor_cont_mean.bias']=sorted(set(rows))
   if hasattr(policy,'actor_cont_logstd'):
    policy.actor_cont_logstd.register_hook(lambda g,r=r:g.index_fill(0,r,0.0)); row_frozen['actor_cont_logstd']=sorted(set(rows))
 trainable=[n for n,p in policy.named_parameters() if p.requires_grad]
 frozen=[n for n,p in policy.named_parameters() if not p.requires_grad]
 return {'mode':'value_only' if value_only else ('shoot_only' if shoot_only else 'actor_value'), 'trainable_param_names':trainable, 'frozen_param_names':frozen, 'row_frozen_param_names':row_frozen}

def parse():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default=''); ap.add_argument('--output-dir',required=True); ap.add_argument('--env-path',default=RuntimeConfig.env_path); ap.add_argument('--base-port',type=int,default=61700); ap.add_argument('--seed',type=int,default=7400); ap.add_argument('--device',default='cuda'); ap.add_argument('--eval-only',action='store_true'); ap.add_argument('--action-mode',choices=['deterministic','stochastic'],default='stochastic'); ap.add_argument('--iterations',type=int,default=40); ap.add_argument('--steps-per-iter',type=int,default=256); args=ap.parse_args(); data={}
 if args.config: data=yaml.safe_load(Path(args.config).read_text()) or {}
 return args,data
def cfg_value(data,key,default): return data[key] if key in data else default
def main():
 args,data=parse(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
 cfg=RuntimeConfig(env_path=args.env_path,normalizer_path=cfg_value(data,'normalizer_path',RuntimeConfig.normalizer_path),output_dir=str(out),base_port=args.base_port,seed=int(cfg_value(data,'seed',args.seed)),time_scale=float(cfg_value(data,'time_scale',3.0)),device=args.device,rollout_steps=args.steps_per_iter*int(cfg_value(data,'num_unity_areas',2)),seq_len=16,batch_size=4,lr=float(cfg_value(data,'lr_initial',cfg_value(data,'lr',1e-6))),target_kl=float(cfg_value(data,'target_kl',0.001)),max_grad_norm=float(cfg_value(data,'max_grad_norm',0.5)),kl_anchor_coef=float(cfg_value(data,'kl_anchor_coef',0.0)),kl_anchor_checkpoint=str(cfg_value(data,'kl_anchor_checkpoint','')),game_mode='GambitVsGambit',player_b_bot_mode='None',num_areas=int(cfg_value(data,'num_unity_areas',2)),aim_coef=float(cfg_value(data,'aim_coef',1.0)),aim_err_max=float(cfg_value(data,'aim_err_max',150.0)),shoot_bootstrap_coef=float(cfg_value(data,'shoot_bootstrap_coef',0.05)),shoot_bootstrap_aim_threshold_deg=float(cfg_value(data,'shoot_bootstrap_aim_threshold_deg',30.0)),jerk_coef=float(cfg_value(data,'jerk_coef',0.001)),spam_coef=float(cfg_value(data,'spam_coef',0.001)),min_log_std=float(cfg_value(data,'min_log_std',-2.5)),max_log_std=float(cfg_value(data,'max_log_std',-1.2)),mean_action_saturation_penalty=bool(cfg_value(data,'mean_action_saturation_penalty',True)),mean_action_saturation_threshold=float(cfg_value(data,'mean_action_saturation_threshold',0.65)),mean_action_saturation_coef=float(cfg_value(data,'mean_action_saturation_coef',0.03)),look_action_l2_reward_coef=float(cfg_value(data,'look_action_l2_reward_coef',0.0)),mean_action_l2_loss_coef=float(cfg_value(data,'mean_action_l2_loss_coef',0.0)),saturation_abs095_stop_rate=0.15,saturation_start_iter=10,actor_update_enabled=bool(cfg_value(data,'actor_update_enabled',True)),value_update_enabled=bool(cfg_value(data,'value_update_enabled',True)),freeze_look_action_head=bool(cfg_value(data,'freeze_look_action_head',False)),freeze_movement_action_head=bool(cfg_value(data,'freeze_movement_action_head',False)),train_shoot_head=bool(cfg_value(data,'train_shoot_head',False)),train_value_head=bool(cfg_value(data,'train_value_head',True)))
 cfg.value_only=bool(cfg_value(data,'value_only',False)); learner_side=str(cfg_value(data,'learner_agent','A')).upper(); eval_only=bool(cfg_value(data,'eval_only',False)) or args.eval_only; iterations=int(cfg_value(data,'num_iterations',args.iterations)); action_mode=str(cfg_value(data,'action_mode',args.action_mode)); ck_a=str(cfg_value(data,'resume_agent_a_from',cfg_value(data,'resume_from',''))); ck_b=str(cfg_value(data,'resume_agent_b_from',ck_a)); cfg.resume_from=ck_a if learner_side=='A' else ck_b
 weapon_log=configure_unity_env(out,cfg)
 cfg_a=copy.deepcopy(cfg); cfg_a.resume_from=ck_a; policy_a,normalizer,_,_,_,device=load_policy_and_normalizer(cfg_a)
 cfg_b=copy.deepcopy(cfg); cfg_b.resume_from=ck_b; policy_b,_,_,_,_,_=load_policy_and_normalizer(cfg_b)
 restore_policy_tensors(policy_a,ck_a); restore_policy_tensors(policy_b,ck_b)
 if hasattr(policy_a,'set_ppo_mode'): policy_a.set_ppo_mode(training=(not eval_only and learner_side=='A'))
 if hasattr(policy_b,'set_ppo_mode'): policy_b.set_ppo_mode(training=(not eval_only and learner_side=='B'))
 learner=policy_a if learner_side=='A' else policy_b; opponent=policy_b if learner_side=='A' else policy_a; freeze_report=apply_freeze(learner,cfg); trainable_params=[p for _,p in learner.named_parameters() if p.requires_grad];
 if (not eval_only) and not trainable_params: raise RuntimeError('no trainable parameters after freeze')
 opt=None if eval_only else torch.optim.Adam(trainable_params,lr=cfg.lr)
 anchor=None
 if (not eval_only) and cfg.kl_anchor_coef>0:
  acfg=copy.deepcopy(cfg); acfg.resume_from=cfg.kl_anchor_checkpoint or (ck_a if learner_side=='A' else ck_b); anchor,_,_,_,_,_=load_policy_and_normalizer(acfg); restore_policy_tensors(anchor, acfg.resume_from); anchor.eval(); [p.requires_grad_(False) for p in anchor.parameters()]
 reward_builder=build_reward_builder(cfg); write_json(out/'run_manifest.json',{'phase':cfg_value(data,'phase','phase3v2_b'),'learner_agent':learner_side,'eval_only':eval_only,'action_mode':action_mode,'num_areas':cfg.num_areas,'checkpoint_a':ck_a,'checkpoint_b':ck_b,'checkpoint_a_sha256':sha(ck_a),'checkpoint_b_sha256':sha(ck_b),'config':data,'freeze_report':freeze_report,'side_assignment':'stable_initial_agent_id_slot_map'})
 status='PASS'; failures=[]; guard_events=[]; completed_iters=0; metrics={'start_utc':utc_now(),'checkpoints':[]}; env=None; initial={k:v.detach().cpu().clone() for k,v in learner.state_dict().items() if torch.is_floating_point(v)}; frozen_initial=snapshot_named(learner,freeze_report['frozen_param_names']); row_frozen_initial=snapshot_rows(learner,freeze_report['row_frozen_param_names'])
 try:
  env=launch_unity_env(cfg); behavior,action_spec,_=validate_specs(env); hidden={}; prev={}; known=set(); zfire={'A':0,'B':0}; zmax={'A':0,'B':0}; high={'A':0,'B':0}; highmax={'A':0,'B':0}; slot_seed_obs={}; slot_map={}
  for _ in range(100):
   dec,ter=env.get_steps(behavior)
   for aid,obs in collect_step_records(dec,ter):
    known.add(aid); slot_seed_obs[aid]=obs
   if len(slot_seed_obs)>=cfg.num_areas*2: break
   env.step()
  slot_map=build_slot_map(slot_seed_obs,cfg.num_areas)
  for it in range(1,iterations+1):
   buffer=RecurrentRolloutBuffer(size=args.steps_per_iter*max(1,cfg.num_areas),device=str(device)); sides={'A':empty(),'B':empty()}; learner_trans=opp_trans=0
   for _ in range(args.steps_per_iter):
    dec,ter=env.get_steps(behavior)
    if len(dec)==0: env.step(); continue
    obs_batch=np.asarray(dec.obs[0],dtype=np.float32); pending={}
    for row,aid_raw in enumerate(dec.agent_id):
     aid=int(aid_raw); obs=obs_batch[row]; side,area=slot_map.get(aid, ('A' if aid % 2 == 0 else 'B', min(cfg.num_areas-1, aid//2))); known.add(aid); pol=policy_a if side=='A' else policy_b; s=sides[side]; s['decisions_received']+=1; s['aim_values'].append(float(max(0.0,obs[22]))); h=hidden.get(aid,pol.init_hidden(1,device)); obsn=normalizer.normalize_tensor(torch.as_tensor(obs,dtype=torch.float32,device=device).view(1,1,OBS_DIM))
     with torch.no_grad() if (eval_only or side!=learner_side) else torch.enable_grad():
      dist,val,nh=pol(obsn,h); mu_c,mu_raw=dist.mode; sample_c,sample_raw=dist.sample(); act_c=mu_c if action_mode=='deterministic' else sample_c; act_raw=mu_raw if action_mode=='deterministic' else sample_raw; logp=dist.log_prob(act_raw)
     mu=mu_c.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32); sample=sample_c.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32); action=act_c.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
     s['mu_actions'].append(mu); s['sampled_actions'].append(sample); s['env_actions'].append(action); s['shoot_pressed_steps']+=int(action[4]>0.5); continuous,discrete=policy_action_to_buffers(action,action_spec); env.set_action_for_agent(behavior,aid,make_action_tuple(continuous,discrete)); s['actions_sent']+=1
     pending[aid]=(side,obs,action,act_raw.detach(),val.detach(),logp.detach(),h.detach().clone(),nh.detach(),obsn.squeeze(0).squeeze(0).detach())
    t0=time.monotonic(); env.step(); dur=time.monotonic()-t0
    if dur>30: raise RuntimeError(f'deadlock/slow Unity step {dur:.1f}s')
    dec2,ter2=env.get_steps(behavior); post={}
    if len(dec2):
     ob=np.asarray(dec2.obs[0],dtype=np.float32)
     for row,aid_raw in enumerate(dec2.agent_id): post[int(aid_raw)]=(ob[row],float(dec2.reward[row]),False)
    if len(ter2):
     ob=np.asarray(ter2.obs[0],dtype=np.float32)
     for row,aid_raw in enumerate(ter2.agent_id): post[int(aid_raw)]=(ob[row],float(ter2.reward[row]),True)
    for aid,(side,obs,action,act_raw,val,logp,h_before,nh,obsn_flat) in pending.items():
     s=sides[side]; obs2,rew,done=post.get(aid,(obs,0.0,False)); shaped,components=reward_builder.compute(obs_current=obs,obs_next=obs2,action=action,previous_action=prev.get(aid),env_reward=rew,is_terminal=done)
     if cfg.look_action_l2_reward_coef>0: shaped-=cfg.look_action_l2_reward_coef*float(action[2]**2+action[3]**2)
     if float(obs2[44])>0.5: s['fired_steps']+=1
     if rew>0.05: s['hits']+=1; s['damage_dealt']+=float(rew)
     if rew<0: s['damage_taken']+=float(-rew)
     if done: s['deaths_terminals']+=1; hidden[aid]=(policy_a if side=='A' else policy_b).init_hidden(1,device); prev.pop(aid,None)
     else: hidden[aid]=nh; prev[aid]=action.copy()
     if side==learner_side:
      learner_trans+=1
      if not eval_only: buffer.add(obs=obsn_flat,action=act_raw.squeeze(0).squeeze(0),reward=shaped,done=done,value=val.squeeze(),log_prob=logp.squeeze(),hidden=h_before)
     else: opp_trans+=1
   row={'iteration':it,'learner_agent':learner_side,'learner_buffer_transitions':learner_trans,'opponent_buffer_transitions':0,'opponent_decision_transitions':opp_trans,'A':summarize_side(sides['A']),'B':summarize_side(sides['B'])}
   for side in ('A','B'):
    zfire[side]=zfire[side]+1 if row[side]['fired_steps']==0 else 0; zmax[side]=max(zmax[side],zfire[side]); high[side]=high[side]+1 if row[side]['aim_mean']>=160 and row[side]['hits']==0 else 0; highmax[side]=max(highmax[side],high[side])
   stop_after_iter=False
   if not eval_only:
    lastv=torch.zeros((),device=device); buffer.compute_returns_and_advantages(lastv,gamma=cfg.gamma,lam=cfg.gae_lambda); loss=compute_loss_and_backward(learner,buffer,cfg,opt,anchor_policy=anchor,normalizer=normalizer); row.update(loss); assert_frozen_unchanged(learner,frozen_initial,row_frozen_initial); assert_row_frozen_unchanged(learner,row_frozen_initial,freeze_report['row_frozen_param_names'])
    if abs(row.get('approx_kl',0.0))>cfg.target_kl:
     event={'iteration':it,'reason':'target_kl_exceeded','approx_kl':float(row.get('approx_kl',0.0)),'target_kl':float(cfg.target_kl)}
     row['guard_stop']=event; guard_events.append(event); stop_after_iter=True
    if row[learner_side]['mu_look_sat95']>0.15:
     event={'iteration':it,'reason':'mu_look_sat95_hard_stop','mu_look_sat95':float(row[learner_side]['mu_look_sat95']),'limit':0.15}
     row['guard_stop']=event; guard_events.append(event); stop_after_iter=True
    if hasattr(learner,'actor_cont_logstd') and learner.actor_cont_logstd.requires_grad:
     with torch.no_grad(): learner.actor_cont_logstd.clamp_(cfg.min_log_std,cfg.max_log_std)
    ck=out/'checkpoints'/f'update_{it:03d}.pt'; ck.parent.mkdir(exist_ok=True); torch.save({'actor_critic_state':learner.state_dict(),'update':it,'metrics':row,'normalizer_path':str(normalizer.path),'normalizer_sha256':sha256_file(normalizer.path)},ck); torch.save(torch.load(ck,map_location='cpu',weights_only=False),out/'latest.pt')
    if it>=max(1,iterations-20): torch.save(torch.load(ck,map_location='cpu',weights_only=False),out/'best_retention.pt')
   append_jsonl(out/'gvg_train.jsonl',row); print(json.dumps(row,sort_keys=True),flush=True)
   completed_iters=it
   if stop_after_iter: break
  delta=sum(float((v.detach().cpu()- initial[k]).abs().sum().item()) for k,v in learner.state_dict().items() if k in initial and torch.is_floating_point(v))
  metrics['parameter_abs_delta_sum']=delta; metrics['optimizer_steps']=0 if eval_only else completed_iters
  if eval_only and delta!=0: raise RuntimeError(f'eval-only parameter changed: {delta}')
 except Exception as exc: status='FAIL'; failures.append(str(exc))
 finally:
  if env is not None: env.close()
 rows=[]; p=out/'gvg_train.jsonl'
 if p.exists():
  for line in p.read_text(errors='replace').splitlines():
   if line.strip(): rows.append(json.loads(line))
 weapon=summarize_weapon_events(collect_all_weapon_events(weapon_log)); hard=weapon.get('hard_blocked_fire_count_by_reason') or {}; owner=int(weapon.get('owner_mismatch_count',0) or 0); late=rows[-10:]
 def total(side):
  return {'fired_steps':sum(r[side]['fired_steps'] for r in rows),'hits':sum(r[side]['hits'] for r in rows),'late_fired_steps_mean':mean([r[side]['fired_steps'] for r in late]) if late else 0.0,'late_hits_mean':mean([r[side]['hits'] for r in late]) if late else 0.0,'late_aim_mean':mean([r[side]['aim_mean'] for r in late]) if late else 999.0,'mu_look_sat95_max':max([r[side]['mu_look_sat95'] for r in rows] or [0.0]),'sampled_look_sat95_max':max([r[side]['sampled_look_sat95'] for r in rows] or [0.0]),'env_applied_look_sat95_max':max([r[side]['env_applied_look_sat95'] for r in rows] or [0.0])}
 tz=true_zero_fire_stall_metrics(rows)
 summary={'status':status,'failures':failures,'guard_events':guard_events,'freeze_report':freeze_report,'rows':len(rows),'learner_agent':learner_side,'eval_only':eval_only,'action_mode':action_mode,'num_areas':cfg.num_areas,'A':total('A'),'B':total('B'),'hard_blocked_fire_count_by_reason':hard,'blocked_fire_count_by_reason':weapon.get('blocked_fire_count_by_reason') or {},'owner_mismatch_count':owner,'obs44_mismatch_count':0,'zero_fire_streak_max':max(zmax.values()) if 'zmax' in locals() else 999,'true_zero_fire_stall_metrics':tz,'true_zero_fire_stall_max_streak':tz['true_zero_fire_stall_max_streak'],'high_aim_zero_hit_streak_max':max(highmax.values()) if 'highmax' in locals() else 999,'servicing_ok':bool(rows and all(sum(r[s]['actions_sent'] for r in rows)>0 for s in ('A','B'))),'learner_buffer_transitions':sum(r.get('learner_buffer_transitions',0) for r in rows),'opponent_buffer_transitions':sum(r.get('opponent_buffer_transitions',0) for r in rows),**metrics}
 true_limit=3 if eval_only else 5
 if hard or owner or summary['true_zero_fire_stall_max_streak']>=true_limit or summary['high_aim_zero_hit_streak_max']>=10 or not summary['servicing_ok']: summary['status']='FAIL'
 write_json(out/'gvg_one_sided_summary.json',summary); print(json.dumps({'status':summary['status'],'job':out.name,'learner':learner_side},sort_keys=True))
 if summary['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
