#!/usr/bin/env python3
"""CPU fp32 checks using production DAT sampling, frozen readers and real CE."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F
import einops

from test_dat_loading_geometry import ROOT, MODEL, extract_class
from check_dat_coordinate_grad import ce_parts, central_difference, fd_candidates
from torch.utils.checkpoint import checkpoint

spec = importlib.util.spec_from_file_location('dat_experiments', ROOT/'llava/model/dat_experiments.py')
controls = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controls
spec.loader.exec_module(controls)

NS = dict(torch=torch, nn=nn, F=F, einops=einops, math=__import__('math'),
          _grad_scale=lambda x,s: x if s else x.detach())
# extract_class uses an identifier as base, so bind the Torch base explicitly.
NS['Module'] = nn.Module
LayerNorm = extract_class(MODEL, 'LayerNorm2d', {'__init__','forward'}, NS, base='Module')
Attention = extract_class(MODEL, 'Qwen3_5AttentionDAT', {
    '_sample_hd_from_off_guide','_grid_generate','_construct_hd_position_ids',
    '_query_indices','_local_query_indices','_glob_query_indices','_generate_offsets_and_sample',
}, NS, base='Module')


class ToyModel(nn.Module):
    def forward(self, guide, hd, **kwargs):
        def read(x):
            return sample(self,hd,x)[1]
        if getattr(self, 'checkpointing', False) and torch.is_grad_enabled():
            values = checkpoint(read,guide,use_reentrant=True)
        else:
            values = read(guide)
        return SimpleNamespace(loss=loss_from_values(values))


def toy_model(seed=7):
    torch.manual_seed(seed)
    model = ToyModel()
    att = model.self_attn = Attention()
    att.layer_idx = 3
    att.grid_size, att.off_grps, att.off_dim, att.inter_size = 20, 2, 4, 4
    att.hidden_size = 8
    att.conv_lr_dw = nn.Conv2d(4,4,3,padding=1,groups=4,bias=False)
    att.ln_1 = LayerNorm(4)
    att.conv_lr_proj = nn.Conv2d(4,4,1)
    att.proj_intention = nn.Linear(4,4)
    att.ln_2 = LayerNorm(4)
    att.conv_off_proj = nn.Conv2d(4,2,1,bias=False)
    att.conv_glob = nn.Conv2d(4,1,1)
    att.glob_q, att.glob_k, att.glob_use_attn = None,None,False
    att.k_proj_hd, att.v_proj_hd = nn.Linear(8,4), nn.Linear(8,4)
    att.hd_input_layernorm = nn.Identity()
    att.hd_gate = None
    att.hd_proj = True
    att._dat_sampling_control = None
    att._dat_sample_key = (0,0)
    att.local_query_pos, att.glob_query_pos = 'im_start','ans_prev'
    att.hd_position_mode = 'slot'
    att.use_intention_branch, att.intention_as_gate = True,True
    att.use_spatial_attn_guide = True
    att.use_global_offset = True
    att.glob_min_scale, att.glob_floor, att.off_range = .1,0.,.1
    att.off_penalty, att.off_sup_weight, att.rel_sup_weight = 0.,0.,0.
    att.off_head_trunk_grad = 0.
    att._dat_off_target, att._dat_force_locs = None,None
    att._fn_chk = lambda *args: None
    # Strong smooth HD evidence; preserve separate group channels.
    ys,xs = torch.meshgrid(torch.linspace(-1,1,31),torch.linspace(-1,1,37),indexing='ij')
    hd = torch.stack([xs,ys,xs+ys,xs-ys,2*xs,2*ys,xs+2*ys,2*xs-ys],-1)
    with torch.no_grad():
        att.v_proj_hd.weight.zero_()
        att.v_proj_hd.weight[0,0] = 2
        att.v_proj_hd.weight[1,4] = -2
    return model, hd


def sample(model, hd, guide=None, slots=1):
    att = model.self_attn
    if guide is None:
        guide = torch.randn(slots*att.off_grps,4,20,20)
    return att._sample_hd_from_off_guide(guide,[hd],0,slots,torch.device('cpu'))


def loss_from_values(values):
    # A frozen reader predicts a content token from one actual sampled HD token.
    logits = torch.stack([values[0,149,0],values[0,149,1],values[0,149,2]])[None]
    return F.cross_entropy(logits,torch.tensor([1]))


class ExperimentTests(unittest.TestCase):
    def test_sampler_config_keeps_reader_and_coarse_architecture(self):
        source = dict(grid_size=20, off_grps=8, inter_size=128, layers='LD',
                      image_hd_for_question=True, hd_skip_merger_mlp=True,
                      use_global_offset=True, glob_relevance='attn', glob_query_pos='ans_prev',
                      off_penalty=1., lr_drop_prob=.3, off_range=.2)
        a = controls.sampler_dat_config(source,'im_start')
        b = controls.sampler_dat_config(source,'ans_prev')
        self.assertEqual({k:v for k,v in a.items() if k!='local_query_pos'},
                         {k:v for k,v in b.items() if k!='local_query_pos'})
        self.assertTrue(a['image_hd_for_question'])
        self.assertEqual(a['glob_relevance'],'attn')
        self.assertEqual(a['off_range'],.2)
        self.assertEqual(a['off_penalty'],0.)
        self.assertEqual(source['off_penalty'],1.)
        with self.assertRaises(ValueError):
            controls.sampler_dat_config(dict(grid_size=6),'ans_prev')

    def test_pure_ce_disables_checkpoint_auxiliary_hooks_and_dropout(self):
        model,_ = toy_model()
        model.config = SimpleNamespace(dat_extra_args=dict(off_penalty=1.,lr_drop_prob=.5))
        model.self_attn.off_penalty = 1.
        model.self_attn._dat_exact_merge_available = True
        model.dropout = nn.Dropout(.5)
        with self.assertRaises(ValueError):
            controls.require_pure_ce(model)
        controls.configure_pure_ce(model)
        controls.require_pure_ce(model)
        self.assertEqual(model.dropout.p,0.)
        self.assertEqual(model.self_attn.off_penalty,0.)
        self.assertIsNone(model.self_attn._dat_force_locs)

    def test_query_switch_multiturn_prefill_independent_of_global(self):
        model,_ = toy_model()
        att = model.self_attn
        ranges = [[7,8,4],[13,14,10],[18,-1,15]]
        self.assertEqual(att._local_query_indices(ranges),[4,10,15])
        self.assertEqual(att._glob_query_indices(ranges),[6,12,17])
        att.local_query_pos = 'ans_prev'
        self.assertEqual(att._local_query_indices(ranges),[6,12,17])
        att.glob_query_pos = 'im_start'
        self.assertEqual(att._glob_query_indices(ranges),[4,10,15])
        self.assertEqual(att._local_query_indices([[0,0,0]]),[0])

    def test_slot_positions_common_for_learned_forced_and_override(self):
        model,_ = toy_model()
        att = model.self_attn
        pos = torch.tensor([[1,1,1,1],[0,0,1,1],[0,1,0,1]])
        normal = att._construct_hd_position_ids(pos,0,4,2,2,'cpu')
        att._dat_force_locs = torch.full((400,2),-1.)
        self.assertTrue(torch.equal(normal,att._construct_hd_position_ids(pos,0,4,2,2,'cpu')))
        att.hd_position_mode = 'legacy'
        legacy = att._construct_hd_position_ids(pos,0,4,2,2,'cpu')
        self.assertFalse(torch.equal(normal,legacy))
        with controls.sampling_control(model,controls.DATSamplingControl()):
            self.assertTrue(torch.equal(normal,att._construct_hd_position_ids(pos,0,4,2,2,'cpu')))

    def test_fixed_coarse_zero_origin_and_reader_freeze_after_step(self):
        model,hd = toy_model()
        reference = copy.deepcopy(model).requires_grad_(False)
        trainables = controls.freeze_local_sampler_only(model)
        self.assertTrue(all(any(k in n for k in controls.LOCAL_SAMPLER_MODULES) for n in trainables))
        guide = torch.randn(2,4,20,20)
        capture = controls.DATSamplingControl(zero_residual=True)
        with controls.sampling_control(reference,capture),torch.no_grad():
            sample(reference,hd,guide)
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),lr=.1)
        controls.assert_sampler_optimizer(model,opt)
        replay = capture.replay(observe=True)
        with controls.sampling_control(model,replay):
            _,values,locations = sample(model,hd,guide)
            coarse = capture.coarse[(3,0,0)].permute(0,2,3,1).reshape(1,2,20,20,2)
            self.assertTrue(torch.equal(locations,coarse.clamp(-1,1)))
            loss_from_values(values).backward()
        self.assertIsNotNone(replay.records[(3,0,0)]['sampled'].grad)
        self.assertGreater(float(model.self_attn.conv_off_proj.weight.grad.norm()),0)
        opt.step()
        changed = [n for n,p in model.named_parameters() if not torch.equal(p.detach(),before[n])]
        self.assertTrue(changed)
        self.assertTrue(all(n in trainables for n in changed),changed)
        # Even a changed current global locator cannot move the replayed grid.
        with torch.no_grad():
            model.self_attn.conv_glob.weight.add_(10)
        with controls.sampling_control(model,capture.replay(zero_residual=True)):
            _,_,locations = sample(model,hd,guide)
        self.assertTrue(torch.equal(locations,coarse.clamp(-1,1)))
        bad = torch.optim.SGD(model.parameters(),lr=.1)
        with self.assertRaises(RuntimeError):
            controls.assert_sampler_optimizer(model,bad)

    def test_per_group_override_ce_gradient_and_finite_difference(self):
        model,hd = toy_model()
        model.requires_grad_(False)
        guide = torch.randn(2,4,20,20)
        capture = controls.DATSamplingControl(zero_residual=True)
        with controls.sampling_control(model,capture),torch.no_grad():
            _,_,base = sample(model,hd,guide)
        coords = base.clone().requires_grad_(True)
        # Distinct group coordinates; they must reach grid_sample independently.
        with torch.no_grad():
            coords[0,0,7,9] = torch.tensor([.231,.337])
            coords[0,1,7,9] = torch.tensor([-.217,-.293])
        control = capture.replay(observe=True,overrides={(3,0,0):coords})
        with controls.sampling_control(model,control):
            _,values,locs = sample(model,hd,guide)
            self.assertTrue(torch.equal(locs,coords.detach()))
            ce = loss_from_values(values)
            gradient = torch.autograd.grad(ce,coords,retain_graph=True)[0]
            ce.backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertGreater(float(gradient.abs().sum()),0)
        self.assertIsNotNone(control.records[(3,0,0)]['sampled'].grad)
        def evaluate(value):
            replay = capture.replay(overrides={(3,0,0):value})
            with controls.sampling_control(model,replay):
                return loss_from_values(sample(model,hd,guide)[1]).detach()
        for group in (0,1):
            index = (0,group,7,9,0)
            for step in (.01,.05,.1):
                fd = central_difference(evaluate,coords,index,step,hd.shape[:2])['finite_difference']
                self.assertAlmostEqual(fd,float(gradient[index]),delta=.002)
        summary = controls.coordinate_summary(control.records[(3,0,0)])
        self.assertEqual([(r['group'],r['tokens']) for r in summary],[(0,400),(1,400)])
        self.assertTrue(fd_candidates(coords.detach(),gradient,hd.shape[:2],2))
        self.assertIsNone(controls.coordinate_summary(control.records[(3,0,0)],None)[0]['grad_norm'])
        with controls.sampling_control(model,capture.replay(overrides={(3,0,0):coords[:,0]})):
            with self.assertRaises(ValueError):
                sample(model,hd,guide)

    def test_multi_image_multi_answer_readers_have_distinct_keys_and_400_tokens(self):
        model,hd = toy_model()
        state = torch.randn(1,20,8)
        ranges = [[[(0,4,2,2),(4,8,2,2)],[12,13,9],[18,19,15]]]
        ctl = controls.DATSamplingControl(zero_residual=True,observe=True)
        with controls.sampling_control(model,ctl):
            k,v,locs,ki,vi = model.self_attn._generate_offsets_and_sample(
                state,[hd,hd],ranges,0,[0,1],want_image=True)
        self.assertEqual(k.shape[:2],(2,800))
        self.assertEqual(ki.shape[:2],(1,800))
        self.assertEqual(set(ctl.records),{(3,0,0),(3,0,1)})
        for record in ctl.records.values():
            self.assertEqual(record['slots'],3)
            self.assertTrue(all(r['tokens']==400 for r in controls.coordinate_summary(record)))
        self.assertIsNone(model.self_attn._dat_sampling_control)

    def test_content_template_first_ce_shift(self):
        logits = torch.randn(1,6,5,requires_grad=True)
        labels = torch.tensor([[-100,-100,-100,2,3,4]])
        parts = ce_parts(logits,labels,{4})
        self.assertTrue(torch.allclose(parts['first'],F.cross_entropy(logits[:,2],torch.tensor([2]))))
        self.assertTrue(torch.allclose(parts['total'],(2*parts['content']+parts['template'])/3))

    def test_answer_newline_is_content_and_only_trailing_newline_is_template(self):
        logits = torch.randn(1,8,5,requires_grad=True)
        labels = torch.tensor([[-100,2,3,4,3,-100,3,2]])
        parts = ce_parts(logits,labels,{4},{3})
        # Template: label 4 and the newline immediately after it. The later
        # supervised span resets the end-marker state.
        expected = F.cross_entropy(logits[:,[2,3]].reshape(-1,5),torch.tensor([4,3]))
        self.assertTrue(torch.allclose(parts['template'],expected))
        expected_content = F.cross_entropy(logits[:,[0,1,5,6]].reshape(-1,5),torch.tensor([2,3,3,2]))
        self.assertTrue(torch.allclose(parts['content'],expected_content))

    def test_seed_controls_parameter_construction(self):
        a,_ = toy_model(11)
        b,_ = toy_model(11)
        c,_ = toy_model(12)
        self.assertTrue(all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters())))
        self.assertFalse(torch.equal(a.self_attn.conv_off_proj.weight,c.self_attn.conv_off_proj.weight))

    def test_trainer_replays_reference_through_checkpoint_backward_then_cleans_up(self):
        sys.modules['llava.model.dat_experiments'] = controls
        class BaseTrainer:
            def __init__(self, model, **kwargs):
                self.model = model
                self.args = SimpleNamespace(kd_on=False)
            def compute_loss(self,model,inputs,**kwargs):
                return model(**inputs).loss
            def training_step(self,model,inputs,num_items_in_batch=None,**kwargs):
                loss = self.compute_loss(model,inputs)
                self.replay = model.self_attn._dat_sampling_control
                self.assert_replay = self.replay.mode == 'replay'
                loss.backward()
                self.retained_through_backward = model.self_attn._dat_sampling_control is self.replay
                return loss.detach()
        ns = dict(torch=torch,BaseTrainer=BaseTrainer)
        trainer_cls = extract_class(ROOT/'llava/train/train_qwen_dat.py','Qwen2VLTrainer',
            {'__init__','compute_loss','training_step'},ns,base='BaseTrainer')
        model,hd = toy_model()
        reference = copy.deepcopy(model).requires_grad_(False)
        controls.freeze_local_sampler_only(model)
        model.checkpointing = True
        trainer = trainer_cls(model=model,dat_reference=reference)
        trainer._split_student_teacher_inputs = lambda inputs:(inputs,{})
        trainer._maybe_log_hd_content_gap = lambda *args:None
        guide = torch.randn(2,4,20,20,requires_grad=True)
        before = {name:p.clone().detach() for name,p in reference.named_parameters()}
        trainer.training_step(model,dict(guide=guide,hd=hd))
        self.assertTrue(trainer.assert_replay)
        self.assertTrue(trainer.retained_through_backward)
        self.assertIsNone(model.self_attn._dat_sampling_control)
        self.assertIsNone(reference.self_attn._dat_sampling_control)
        self.assertGreater(float(model.self_attn.conv_off_proj.weight.grad.norm()),0)
        self.assertTrue(all(torch.equal(p,before[n]) for n,p in reference.named_parameters()))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
