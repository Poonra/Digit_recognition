mod data;
mod model;
mod train;
use candle_core::Device;
use candle_nn::{VarBuilder,VarMap};


fn main()->candle_core::Result<()> {

    let device = Device::Cpu;
    let mnist = data::load_mnist(&device)?;

    println!("Data loaded");
   

    //varmap keep all trainable parameter

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap,candle_core::DType::F32, &device);
    let model = model::MLP::new(vb)?;
    println!("Model ready!");

    train::train(&mnist, &varmap, &model)?;
    println!("training complete");

    Ok(())
}
