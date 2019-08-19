from delira import get_backends
import os

if "TORCH" in get_backends():
    from .generative_adversarial_network import \
        GenerativeAdversarialNetworkBasePyTorch
'''
if "tf" in os.environ["DELIRA_BACKEND"]:
    from.generative_adversarial_network_tf import \
        GenerativeAdversarialNetworkBaseTf

'''

if "TF" in get_backends():
    from .conditional_generative_adversarial_network_tf import \
        ConditionalGenerativeAdversarialNetworkBaseTf
    from .bicycle_generative_adversarial_network_tf import BicycleGenerativeAdversarialNetworkBaseTf
    from .DS_bicycle import DSGenerativeAdversarialNetworkBaseTf


