from nano_runtime import load_checkpoint
'Portable state dictionaries; load with weights_only=True.'
import torch
from models import LPTFNet
from models.prototype_context import PrototypeContextEncoder
from models.context_interaction import ContextInteractionTransform
from models.efcir import TemporalContextExpert, EFCIRExpertBank, AnalyticContextRouter

def save_efcir(path, c):

    def cpu(m):
        return {k: v.detach().cpu() for k, v in m.state_dict().items()}
    encoder_config = c['encoder'].get_config()
    encoder_config.pop('architecture', None)
    transform_config = c['transform'].get_config()
    transform_config.pop('architecture', None)
    d = dict(format_version=1, method='Exemplar-Free Context Isolation and Routing (EF-CIR)', base_config=c['base'].get_config(), base=cpu(c['base']), encoder_config=encoder_config, encoder=cpu(c['encoder']), transform_config=transform_config, transform=cpu(c['transform']), router=c['router'].state_dict(), experts={name: dict(labels=m.global_labels.cpu().tolist(), state=cpu(m)) for name, m in c['bank'].experts.items()})
    torch.save(d, path)

def load_efcir(path, device='cpu'):
    d = load_checkpoint(path, map_location='cpu', weights_only=True)
    base = LPTFNet(**d['base_config'])
    base.load_state_dict(d['base'])
    base.to(device).eval().requires_grad_(False)
    encoder = PrototypeContextEncoder(**d['encoder_config'])
    encoder.load_state_dict(d['encoder'])
    encoder.to(device).eval().requires_grad_(False)
    transform = ContextInteractionTransform(**d['transform_config'])
    transform.load_state_dict(d['transform'])
    bank = EFCIRExpertBank(16).to(device)
    for name, v in d['experts'].items():
        expert = TemporalContextExpert(base.temporal, 96, v['labels'], classifier_type='linear').to(device)
        expert.load_state_dict(v['state'])
        expert.eval().requires_grad_(False)
        bank.add_expert(name, expert)
    router = AnalyticContextRouter.from_state_dict(d['router'])
    return dict(base=base, encoder=encoder, transform=transform, bank=bank, router=router)
