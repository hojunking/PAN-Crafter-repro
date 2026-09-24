"""CPU-only coefficient and routing evidence (not a CUDA admission receipt)."""
import unittest
import torch
from fh12.losses import student_losses as original_losses,routed_student_backward
from maina_hqnr.training import student_losses


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Linear(1,1,bias=False)
        self.aligner=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():self.backbone.weight.fill_(.37);self.aligner.weight.fill_(.12)

    def image(self,x):
        return x*self.backbone.weight.reshape(1,1,1,1)+self.aligner.weight.reshape(1,1,1,1)*x.square()


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        gen=torch.Generator().manual_seed(810)
        self.x=torch.randn(2,8,16,16,generator=gen)
        self.gt=torch.randn(2,8,16,16,generator=gen)
        self.teacher=self.gt*.83
        self.weights=torch.tensor([.4,.7])

    def terms(self,model,alpha=1,beta=.1,edge=.002):
        return student_losses({'y':model.image(self.x)},{'y':self.teacher},self.gt,
                              .012118559330701828,self.weights,alpha,beta,edge)

    def test_base_bitwise_original_loss_and_gradients(self):
        a,b=ToyModel(),ToyModel()
        new=self.terms(a)
        old=original_losses({'y':b.image(self.x)},{'y':self.teacher},self.gt,
                            .012118559330701828,self.weights)
        for key in new:self.assertTrue(torch.equal(new[key],old[key]),key)
        routed_student_backward(a,new);routed_student_backward(b,old)
        for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p.grad,q.grad))

    def test_beta_and_edge_exactonce_no_direct_aligner_gradient(self):
        grads=[]
        for beta,edge in ((.1,.002),(.05,.002),(.2,.002),(.1,.0006),(.1,.006)):
            model=ToyModel();loss=self.terms(model,beta=beta,edge=edge)
            y=model.image(self.x);et=(self.teacher-self.gt).abs().mean(1,keepdim=True)
            es=(y-self.gt).abs().mean(1,keepdim=True)
            d=et/(et+.012118559330701828);adv=((es-et).clamp_min(0)/(es+1e-6)).detach()
            expected=(beta*(1-d)*adv*(y-self.teacher).abs().mean(1,keepdim=True)).flatten(1).mean(1)
            self.assertTrue(torch.equal(loss['soft_i'],expected))
            self.assertTrue(torch.equal(loss['edge_weighted'],(edge*self.weights*loss['edge_i']).mean()))
            routed_student_backward(model,loss)
            grads.append((model.backbone.weight.grad.clone(),model.aligner.weight.grad.clone()))
        for _,agrad in grads[1:]:self.assertTrue(torch.equal(agrad,grads[0][1]))
        self.assertFalse(torch.equal(grads[0][0],grads[2][0]))
        self.assertFalse(torch.equal(grads[0][0],grads[4][0]))

    def test_alpha_changes_hard_for_both_U_and_A(self):
        grads=[]
        for alpha in (.5,1.,1.5):
            model=ToyModel();terms=self.terms(model,alpha=alpha)
            expected=((1+alpha*terms['difficulty'])*(model.image(self.x)-self.gt).abs().mean(1,keepdim=True)).flatten(1).mean(1)
            self.assertTrue(torch.equal(terms['hard_i'],expected))
            routed_student_backward(model,terms)
            grads.append([p.grad.clone() for p in model.parameters()])
        for index in (0,1):self.assertFalse(torch.equal(grads[0][index],grads[2][index]))

    def test_teacher_gt_q_and_cues_detached(self):
        model=ToyModel();self.gt.requires_grad_();self.teacher.requires_grad_();self.weights.requires_grad_()
        terms=self.terms(model,alpha=.5,beta=.2,edge=.006)
        for key in ('difficulty','advantage','q_weights'):self.assertFalse(terms[key].requires_grad)
        terms['L_U'].backward()
        self.assertIsNone(self.gt.grad);self.assertIsNone(self.teacher.grad);self.assertIsNone(self.weights.grad)

    def test_reject_nonfinite_coefficient(self):
        for value in (float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError):self.terms(ToyModel(),alpha=value)


if __name__=='__main__':unittest.main()
