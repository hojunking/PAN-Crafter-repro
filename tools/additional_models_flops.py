"""Forward-dispatch arithmetic accounting; separate from latency profiling.

1 multiply-add = 2 FLOPs. Conv/linear/bmm included, including functional calls
and repeated query MLPs. Nonlinear primitives count as one scalar operation;
interpolation and normalization use explicitly stated algorithmic estimates.
Memory/index/comparison operations are inventoried separately. Unknown float
operators fail the audit rather than silently contributing zero.
"""
from collections import Counter
import math
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class ArithmeticCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.calls=Counter();self.flops=Counter();self.zero=Counter();self.unknown=Counter()

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}));name=str(func);op=name.split('.')[1]
        self.calls[name]+=1
        out=result[0] if isinstance(result,(tuple,list)) else result
        n=out.numel() if isinstance(out,torch.Tensor) else 0
        floating=isinstance(out,torch.Tensor) and out.is_floating_point()
        value=0;kind=None
        if op in ('convolution','_convolution','conv2d','conv_transpose2d'):
            transposed=(op=='conv_transpose2d' or (op not in ('conv2d','conv_transpose2d') and bool(args[6])))
            mac=args[0].numel()*args[1].shape[1]*math.prod(args[1].shape[2:]) if transposed else n*math.prod(args[1].shape[1:])
            self.flops['conv_2MAC']+=2*mac
            kind='bias';value=n if len(args)>2 and args[2] is not None else 0
        elif op in ('mm','bmm','matmul','addmm','linear','mv','dot'):
            a,b=(args[1],args[2]) if op=='addmm' else (args[0],args[1])
            mac=n*a.shape[-1]
            self.flops['matrix_2MAC']+=2*mac
            kind='bias';value=n if op=='addmm' or (op=='linear' and len(args)>2 and args[2] is not None) else 0
        elif op in ('add','add_','sub','sub_','rsub','mul','mul_','div','div_','neg','abs','sqrt','rsqrt','exp','reciprocal','sinc','pow') and floating:
            kind='elementwise';value=n
            if op in ('add','add_','sub','sub_') and (kwargs or {}).get('alpha',1)!=1:value+=n
        elif op in ('sum','mean') and floating:
            kind='reduction';value=args[0].numel() if op=='mean' else args[0].numel()-n
        elif op in ('var','std'):
            kind='variance_estimate';value=4*args[0].numel()+ (n if op=='std' else 0)
        elif op=='linalg_vector_norm':
            if args[1]!=2:raise ValueError('Only L2 norm accounting is defined')
            kind='l2_norm_estimate';value=2*args[0].numel()
        elif op in ('native_group_norm','native_layer_norm'):
            # mean N; centered variance 3N; normalize 2N; affine 2N; eps/rsqrt 2/group.
            groups=result[1].numel();affine=args[1] is not None if op=='native_group_norm' else args[2] is not None
            kind='normalization_estimate';value=(8 if affine else 6)*n+2*groups
        elif op in ('group_norm','layer_norm'):
            groups=args[0].shape[0]*args[1] if op=='group_norm' else n//math.prod(args[1])
            affine=len(args)>2 and args[2] is not None
            kind='normalization_estimate';value=(8 if affine else 6)*n+2*groups
        elif op in ('dropout','dropout_'):
            if args[2]:raise ValueError('Training-mode dropout in inference counter')
            self.zero[name]+=1
        elif op in ('_softmax','softmax'):
            kind='softmax_estimate';value=4*n-n//args[0].shape[args[1]]
        elif op in ('leaky_relu','leaky_relu_','_prelu_kernel','prelu'):
            kind='activation_estimate';value=n  # one potential scale multiplication per element
        elif op in ('silu','silu_','sigmoid'):
            kind='activation_estimate';value=(5 if op.startswith('silu') else 4)*n
        elif op=='gelu':
            approximate=(kwargs or {}).get('approximate',args[1] if len(args)>1 else 'none')
            if approximate not in ('none','tanh'):raise ValueError('Unknown GELU approximation')
            # erf(x/sqrt(2)) form: 3 multiplies, 1 add, 1 erf primitive.
            # tanh cubic form: 6 multiplies, 2 adds, 1 tanh primitive.
            kind='activation_estimate';value=(5 if approximate=='none' else 9)*n
        elif op.startswith('upsample_bicubic2d'):
            kind='bicubic_kernel_estimate';value=32*n  # legacy common 16-tap 2op convention
        elif op.startswith('upsample_bilinear2d'):
            kind='bilinear_kernel_estimate';value=8*n
        elif op in ('grid_sampler_2d','grid_sampler'):
            mode=args[2];kind='grid_interpolation_estimate';value={0:8,1:0,2:32}[mode]*n
        elif op in ('_adaptive_avg_pool2d','adaptive_avg_pool2d'):
            kind='average_pool_estimate';value=args[0].numel()
        elif (not floating or op in {
            'view','_unsafe_view','reshape','permute','transpose','transpose_','t','slice','select','squeeze','squeeze_',
            'unsqueeze','unsqueeze_','expand','expand_as','as_strided','clone','copy_','_to_copy','to','detach','alias',
            'empty','empty_like','empty_strided','new_empty','new_zeros','new_ones','zeros','zeros_like','ones','ones_like','full','im2col',
            'lift_fresh','lift_fresh_copy','cat','stack','repeat','flip','arange','meshgrid','gather','index','index_select',
            'pad','reflection_pad2d','constant_pad_nd','relu','relu_','threshold','clamp','clamp_','clamp_min','clamp_min_',
            'where','floor','ceil','round','adaptive_max_pool2d','_adaptive_max_pool2d','max','min','maximum','minimum',
            'ge','gt','le','lt','eq','ne','isfinite','all','any','_local_scalar_dense','unbind','split','split_with_sizes','chunk'
        }):
            self.zero[name]+=1
        else:self.unknown[name]+=1
        if kind is not None:self.flops[kind]+=int(value)
        return result

    def report(self):
        return dict(flops=sum(self.flops.values()),gflops=sum(self.flops.values())/1e9,
            conv_matrix_gflops=(self.flops['conv_2MAC']+self.flops['matrix_2MAC'])/1e9,
            components=dict(self.flops),operator_calls=dict(self.calls),non_arithmetic_calls=dict(self.zero),
            unsupported_operator_calls=dict(self.unknown),convention='1 MAC = 2 FLOPs; G=1e9; algorithmic forward estimate',
            coverage='Conv/Linear/bmm + bias + scalar arithmetic/reductions + DWT/IWT + activation/norm/softmax + interpolation estimates',
            exclusions=['memory movement and indexing','comparisons, max-pool comparisons, rounding',
                        'grid_sample coordinate/address arithmetic; interpolation coefficient generation',
                        'CPU Python scalar arithmetic; transcendental hardware instruction expansion'],
            interpolation_convention='bicubic 32 op/output; bilinear 8 op/output; nearest sample has 0 interpolation arithmetic',
            activation_convention='Leaky/PReLU 1 potential multiply/output; SiLU 5; sigmoid 4; GELU erf 5/tanh 9; ReLU comparisons excluded',
            complete_hardware_instruction_count=False)

def count_forward(fn):
    counter=ArithmeticCounter()
    with torch.inference_mode(),counter:fn()
    report=counter.report()
    if report['unsupported_operator_calls']:raise ValueError('Unaccounted float operators: '+str(report['unsupported_operator_calls']))
    return report

def self_check():
    # Independent small arithmetic cases catch the MAC factor and broadcast/reduction accounting.
    x=torch.ones(1,2,5,5);conv=torch.nn.Conv2d(2,3,3,bias=True)
    r=count_forward(lambda:conv(x));assert r['flops']==1*3*3*3*(2*2*3*3+1)
    a=torch.ones(2,3,4);b=torch.ones(2,4,5)
    r=count_forward(lambda:torch.bmm(a,b));assert r['flops']==2*2*3*4*5
    r=count_forward(lambda:(a*2+1).sum(-1));assert r['flops']==2*a.numel()+a.numel()-6

if __name__=='__main__':self_check();print('Arithmetic counter self-check PASS')
