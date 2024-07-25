import os
import torch

from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor

dependencies = ['torch', 'torchvision', 'PIL', 'mobilesamv2']

def _get_my_repo_dir(github="RogerQi/MobileSAMV2"):
    # Not the recommended way to do it, but torch doesn't have an API for this
    # Use from https://pytorch.org/docs/stable/_modules/torch/hub.html#load
    hub_dir = torch.hub.get_dir()
    os.makedirs(hub_dir, exist_ok=True)
    # Parse github repo information
    repo_owner, repo_name, ref = torch.hub._parse_repo_info(github)
    # Github allows branch name with slash '/',
    # this causes confusion with path on both Linux and Windows.
    # Backslash is not allowed in Github branch name so no need to
    # to worry about it.
    normalized_br = ref.replace('/', '_')
    # Github renames folder repo-v1.x.x to repo-1.x.x
    # We don't know the repo name before downloading the zip file
    # and inspect name from it.
    # To check if cached repo exists, we need to normalize folder names.
    owner_name_branch = '_'.join([repo_owner, repo_name, normalized_br])
    repo_dir = os.path.join(hub_dir, owner_name_branch)
    return repo_dir

def _get_object_aware_model():
    _url = 'https://huggingface.co/RogerQi/MobileSAMV2/resolve/main/ObjectAwareModel.pt'

    # Download file
    repo_dir = _get_my_repo_dir()
    assert os.path.exists(repo_dir) and os.path.isdir(repo_dir), f"Repo dir {repo_dir} does not exist"
    object_aware_model_path = os.path.join(repo_dir, 'ObjectAwareModel.pt')
    torch.hub.download_url_to_file(_url, object_aware_model_path)

    # Set up model
    ObjAwareModel = ObjectAwareModel(object_aware_model_path)

    return ObjAwareModel

def _get_mobilesamv2(encoder_type):
    _encoder_url_dict = {
        'efficientvit_l2': 'https://huggingface.co/RogerQi/MobileSAMV2/resolve/main/l2.pt',
        'tiny_vit': 'https://huggingface.co/RogerQi/MobileSAMV2/resolve/main/mobile_sam.pt',
        'sam_vit_h': 'https://huggingface.co/RogerQi/MobileSAMV2/resolve/main/sam_vit_h.pt'
    }
    _encoder_url = _encoder_url_dict[encoder_type]
    _decoder_url = 'https://huggingface.co/RogerQi/MobileSAMV2/resolve/main/Prompt_guided_Mask_Decoder.pt'
    
    # Download file
    repo_dir = _get_my_repo_dir()
    assert os.path.exists(repo_dir) and os.path.isdir(repo_dir), f"Repo dir {repo_dir} does not exist"
    encoder_path = os.path.join(repo_dir, f'{encoder_type}.pt')
    decoder_path = os.path.join(repo_dir, 'Prompt_guided_Mask_Decoder.pt')
    torch.hub.download_url_to_file(_encoder_url, encoder_path)
    torch.hub.download_url_to_file(_decoder_url, decoder_path)

    # Set up model
    PromptGuidedDecoder = sam_model_registry['PromptGuidedDecoder'](decoder_path)
    mobilesamv2 = sam_model_registry['vit_h']()
    mobilesamv2.prompt_encoder = PromptGuidedDecoder['PromtEncoder']
    mobilesamv2.mask_decoder = PromptGuidedDecoder['MaskDecoder']
    image_encoder = sam_model_registry[encoder_type](encoder_path)
    mobilesamv2.image_encoder = image_encoder
    mobilesamv2.eval()

    return mobilesamv2

def _get_everything(encoder_type):
    obj_aware_model = _get_object_aware_model()
    mobilesamv2 = _get_mobilesamv2(encoder_type)
    predictor = SamPredictor(mobilesamv2)
    return mobilesamv2, obj_aware_model, predictor

def mobilesamv2_efficientvit_l2(pretrained=True, **kwargs):
    assert pretrained, "Inference only. Training not supported."
    return _get_everything('efficientvit_l2')

def mobilesamv2_tiny_vit(pretrained=True, **kwargs):
    assert pretrained, "Inference only. Training not supported."
    return _get_everything('tiny_vit')

def mobilesamv2_sam_vit_h(pretrained=True, **kwargs):
    assert pretrained, "Inference only. Training not supported."
    return _get_everything('sam_vit_h')
