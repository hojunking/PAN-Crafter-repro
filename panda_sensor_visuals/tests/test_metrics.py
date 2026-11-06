"""Compare the portable metric against the supplied, unchanged analysis source."""
import ast
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, uniform_filter, sobel
from panda_sensor_visuals.metrics import measure


class ReferenceTests(unittest.TestCase):
    def test_literal_user_reference_on_rgb_and_pan(self):
        source=Path(__file__).resolve().parents[1]/'reference_51_sharpness.py.txt'
        module=ast.parse(source.read_text())
        functions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ('gm','lstd','objs','load','lum')]
        sensor_loop=next(n for n in module.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='s')
        scene_loop=next(n for n in sensor_loop.body if isinstance(n,ast.For))
        # Execute the original scene body through its method loop, without its
        # CLI, foreign paths, plotting or JSON writes.
        body=[]
        for node in scene_loop.body:
            body.append(node)
            if isinstance(node,ast.For) and isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Attribute) and node.iter.func.attr=='items':break
        code=compile(ast.Module(body=functions+body,type_ignores=[]),str(source),'exec')
        rng=np.random.default_rng(42)
        pan=rng.integers(0,256,(512,512,3),dtype=np.uint8)
        rgb=np.roll(pan,2,axis=0).copy();rgb[::7,::3]=0
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'PANDA/gf2/full/pan';p.mkdir(parents=True)
            Image.fromarray(pan).save(p/'testimg0.png');Image.fromarray(rgb).save(root/'testimg0.png')
            namespace=dict(np=np,os=os,Image=Image,gaussian_filter=gaussian_filter,
                maximum_filter=maximum_filter,minimum_filter=minimum_filter,uniform_filter=uniform_filter,sobel=sobel,
                Q=str(root),s='gf2',i=0,M={'candidate':str(root)},res={'candidate':[]},nobj=0)
            exec(code,namespace)
            ref=namespace['res']['candidate'][0];actual=measure(rgb,pan)
            self.assertEqual(actual,dict(edge_exact=float(ref[0]),hf=float(ref[1]),flat_grain=float(ref[2]),edge_pm2=float(ref[6])))


if __name__=='__main__':unittest.main()
