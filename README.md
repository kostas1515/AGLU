<h1> Adaptive Parametric Activation </h1>

This is the official implementation of Adaptive Parametric Activation (APA) for ECCV2024 accepted paper. 

 <img src="./assets/unified_activations_combined.jpg"
     alt="APA unifies most activation functions under the same formula."
     style="float: left; margin-right: 10px;"
/>

 <h3>Abstract</h3>

The activation function plays a crucial role in model optimisation, yet the optimal choice remains unclear. For example, the Sigmoid activation is the de-facto activation in balanced classification tasks, however, in imbalanced classification, it proves inappropriate due to bias towards frequent classes.  In this work, we delve deeper in this phenomenon by performing a comprehensive statistical analysis in the classification and intermediate layers of both balanced and imbalanced networks and we empirically show that aligning the activation function with the data distribution, enhances the performance in both balanced and imbalanced tasks. To this end, we propose the Adaptive Parametric Activation (APA) function, a novel and versatile activation function that unifies most common activation functions under a single formula. APA can be applied in both intermediate layers and attention layers, significantly outperforming the state-of-the-art on several imbalanced benchmarks such as ImageNet-LT, iNaturalist2018, Places-LT, CIFAR100-LT and LVIS and balanced benchmarks such as ImageNet1K, COCO and V3DET.
<h3>Definition</h3>

APA is defined as: $APA(z,λ,κ) = (λ exp(−κz) + 1) ^{\frac{1}{−λ}}$. APA unifies most activation functions under the same formula.

APA can be used insed the intermediate layers using Adaptive Generalised Linear Unit (AGLU): $AGLU(z,λ,κ) = z APA(z,λ,κ)$. The derivatives of AGLU with respect to κ (top), λ (middle) and z (bottom) are shown below:
<img src="./assets/derivative_visualisations.jpg"
     alt="APA unifies most activation functions under the same formula."
     style="float: left; margin-right: 10px;"
/>


<h3> Simple Code implementation </h3>

```
class Unified(nn.Module):
    def __init__(self,device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        lambda_param = torch.nn.init.uniform_(torch.empty(1, **factory_kwargs))
        kappa_param = torch.nn.init.uniform_(torch.empty(1, **factory_kwargs))
        self.softplus = nn.Softplus(beta=-1.0)
        self.lambda_param = nn.Parameter(lambda_param)
        self.kappa_param = nn.Parameter(kappa_param)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        l = torch.clamp(self.lambda_param,min=0.0001)
        p = torch.exp((1/l) * self.softplus((self.kappa_param*input) - torch.log(l)))
        
        return p # for AGLU simply return p*input
```



     
<h1> Acknowledgements </h1>
This code uses <a href='https://pytorch.org/'>Pytorch</a> and the <a href='https://github.com/open-mmlab/mmdetection'>mmdet</a> framework. Thank you for your wonderfull work!
     
