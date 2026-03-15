use candle_core::{Result, Tensor};
use candle_nn::{Linear,Module,VarBuilder,linear};

pub struct MLP{
    layer1: Linear, //784 ->10
    layer2: Linear, //128 ->10
}

impl MLP{
    pub fn new(vb:VarBuilder) -> Result<Self> {
        let layer1 = linear(784,128,vb.pp("layer1"))?;
        let layer2 = linear(128,10,vb.pp("layer2"))?;
        Ok(Self{layer1,layer2})
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor>{
        let x = self.layer1.forward(x)?;
        let x =x.relu()?;
        let x =self.layer2.forward(&x)?;
        Ok(x)
    }
}

