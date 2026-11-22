"""Direct Python and command-line inference; no labels required."""
import argparse
import json
import torch
from nano_runtime import load_checkpoint, EVAL_BATCH
from models import LPTFNet
from data.dataset import load_point_cloud_sample

class GesturePredictor:

    def __init__(self, checkpoint, method='base', device=None, mode='soft'):
        if method not in ('base', 'efcir', 'multi') or mode not in ('soft', 'hard'):
            raise ValueError('Invalid method or routing mode')
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.method, self.mode = (method, mode)
        if method == 'efcir':
            from checkpoints_io import load_efcir
            self.modules = load_efcir(checkpoint, self.device)
            self.contexts = list(self.modules['bank'].experts)
        else:
            ck = load_checkpoint(checkpoint)
            cfg = ck.get('model_config', ck.get('config'))
            if cfg is None:
                raise ValueError('Checkpoint has no model configuration')
            self.model = LPTFNet(**cfg)
            self.model.load_state_dict(ck['model'])
            self.model.to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def predict(self, points):
        x = torch.as_tensor(points, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4 or tuple(x.shape[1:]) != (28, 120, 3) or (not len(x)):
            raise ValueError('Expected (28,120,3) or (B,28,120,3) XYZ input')
        if not bool(torch.isfinite(x).all()):
            raise ValueError('Input contains non-finite coordinates')
        probabilities, routing = ([], [])
        for part in x.split(EVAL_BATCH):
            part = part.to(self.device)
            if self.method == 'efcir':
                from models.context_router import context_router_input
                c = self.modules
                tokens = c['base'].frame_encoder(part)
                semantic = c['base'].temporal(tokens)[0]
                features = c['transform'](context_router_input(part, semantic, c['encoder']))
                weights = c['router'].weights(features, self.contexts)
                if self.mode == 'soft':
                    logits = c['bank'].mixture_logits(tokens, c['router'], self.contexts, router_features=features)
                else:
                    chosen = weights.argmax(-1)
                    logits = torch.stack([c['bank'].oracle_logits(tokens, name) for name in self.contexts], dim=1)
                    logits = logits[torch.arange(len(part), device=self.device), chosen]
                routing.append(weights.cpu())
            else:
                logits = self.model(part)
            probabilities.append(torch.softmax(logits, -1).cpu())
        probs = torch.cat(probabilities)
        result = {'labels': probs.argmax(-1), 'confidence': probs.max(-1).values, 'probabilities': probs}
        if routing:
            result['routing_probabilities'] = torch.cat(routing)
            result['context_names'] = self.contexts
        return result

    def predict_file(self, path):
        return self.predict(load_point_cloud_sample(path))

    def predict_scene(self, path, num_people):
        if self.method != 'multi' or num_people not in (2, 3, 4):
            raise ValueError('Use multi checkpoint and known person count 2/3/4')
        from data.multi_person import load_multi_person_scene
        points, _, people, mask = load_multi_person_scene(path, context='multi_{}p'.format(num_people))
        active = mask.bool()
        if not bool(active.any()):
            return {'labels': [], 'slots': [], 'num_people': people}
        result = self.predict(points[active])
        result['slots'] = torch.nonzero(active).flatten()
        result['num_people'] = people
        return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--method', choices=['base', 'efcir', 'multi'], default='base')
    p.add_argument('--mode', choices=['soft', 'hard'], default='soft')
    p.add_argument('--device', default=None)
    p.add_argument('--num-people', type=int, default=None)
    a = p.parse_args()
    predictor = GesturePredictor(a.checkpoint, a.method, a.device, a.mode)
    result = predictor.predict_scene(a.input, a.num_people) if a.num_people else predictor.predict_file(a.input)
    print(json.dumps({k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in result.items()}, indent=2))
if __name__ == '__main__':
    main()
