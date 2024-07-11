<h1> Adaptive Parametric Activation </h1>

This is the official implementation of Adaptive Parametric Activation (APA) for ECCV2024 accepted paper. 

APA is defined as: $APA(z,λ,κ) = (λ exp(−κz) + 1) ^{\frac{1}{−λ}}$.
APA unifies most activation functions under the same formula. It's behaviour is shown below:

<img src="./assets/unified_activations_combined.jpg"
     alt="APA unifies most activation functions under the same formula."
     style="float: left; margin-right: 10px;"
/>

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
This repo uses
     
